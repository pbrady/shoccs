"""NSGA-II multi-objective Pareto CLI for Brady-Livescu 2D stability.

Wraps :func:`stencil_gen.pareto.run_nsga2` behind a ``sweeps pareto``
subcommand.  Produces a non-dominated set across 2+ stability metrics
simultaneously (e.g. ``layer1.boundary_gv_err`` vs.
``layer_bl42.max_spectral_abscissa``) — distinct from the existing
``gv-stability-pareto`` subcommand, which is a 1D parametric scan with
dominance filtering and is retained as-is.

See ``plans/45-pareto-optimization.md`` items 45.3a-c for the argparse
surface, 45.4a-c for JSON persistence (``--persist``), and 45.5a-b for
L8 C++ validation of front members (``--validate-with-cpp``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from stencil_gen.optimizer import DEFAULT_BOUNDS
from stencil_gen.pareto import ParetoResult, run_nsga2

from ._pareto_io import save_pareto_front

_KERNEL_CHOICES = ("tension", "gaussian", "multiquadric", "classical")
_KERNEL_DIM = {"tension": 1, "gaussian": 1, "multiquadric": 1, "classical": 2}
_CPP_SUPPORTED_KERNELS = ("classical", "tension", "gaussian", "multiquadric")


def _mangle_objectives(fields: Sequence[str]) -> str:
    """Encode a list of dotted-path objective fields into a filesystem-safe token.

    Replaces ``.`` with ``_`` inside each field and joins the fields with
    ``__`` (double-underscore) so the original field boundary is still
    recoverable by splitting on ``__``.  Used to build per-run persistence
    filenames in :mod:`sweeps._pareto_io` (plan 45.4a).

    Examples
    --------
    >>> _mangle_objectives(["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"])
    'layer1_boundary_gv_err__layer_bl42_max_spectral_abscissa'
    """
    return "__".join(f.replace(".", "_") for f in fields)


def _parse_bounds(raw: list[float] | None) -> list[tuple[float, float]] | None:
    if raw is None:
        return None
    if len(raw) == 0 or len(raw) % 2 != 0:
        raise ValueError(
            f"--bounds expects pairs of values (lo hi [lo hi ...]); got {len(raw)} value(s)"
        )
    return [(float(raw[2 * i]), float(raw[2 * i + 1])) for i in range(len(raw) // 2)]


def _resolve_bounds(
    scheme: str,
    kernel: str,
    raw: list[float] | None,
) -> list[tuple[float, float]]:
    parsed = _parse_bounds(raw)
    if parsed is not None:
        return parsed
    key = (scheme, kernel)
    if key not in DEFAULT_BOUNDS:
        raise ValueError(
            f"no default bounds for scheme={scheme!r}, kernel={kernel!r}; "
            "pass --bounds explicitly"
        )
    return list(DEFAULT_BOUNDS[key])


def _validate_kernel_bounds_dim(
    kernel: str,
    bounds: list[tuple[float, float]],
) -> None:
    """Reject bounds whose pair count disagrees with ``kernel``'s dimensionality.

    Mirrors :func:`sweeps.optimize._validate_kernel_bounds_dim`: a mismatch
    (e.g. ``--kernel classical --bounds 0.5 20``) would otherwise feed a 1D
    sample into a 2D ``params_from_vector``, get swallowed by
    :func:`make_multi_objective`'s sentinel path, and silently return an
    all-sentinel front instead of flagging the user error.
    """
    expected = _KERNEL_DIM[kernel]
    if len(bounds) != expected:
        raise ValueError(
            f"kernel={kernel!r} expects {expected} bound pair(s); "
            f"got {len(bounds)}"
        )


def _print_summary(result: ParetoResult) -> None:
    print(f"\n{'=' * 72}")
    print(f"  [pareto] scheme={result.scheme}  kernel={result.kernel}  method={result.method}")
    print(f"  [pareto] objectives={list(result.objective_fields)}")
    print(f"  [pareto] bounds={list(result.bounds)}")
    print(f"{'=' * 72}")
    print(f"  front_size     = {len(result.front)}")
    print(f"  pop_size       = {result.pop_size}")
    print(f"  n_gen          = {result.n_gen}")
    print(f"  n_evals        = {result.n_evals}")
    print(f"  compute_time   = {result.compute_time:.3f} s")
    print(f"  ref_point      = {list(result.ref_point)}")
    hv_final = result.hv_trace[-1] if result.hv_trace else float("nan")
    print(f"  hv_trace[-1]   = {hv_final:.6e}")
    extras = result.extras or {}
    for k, v in extras.items():
        # Per-gen n_nds trace is too noisy for the headline summary.
        if k == "hv_n_nds":
            continue
        if isinstance(v, np.ndarray):
            print(f"  extras.{k:<20s} = {np.array2string(v, precision=6)}")
        else:
            print(f"  extras.{k:<20s} = {v}")

    if not result.front:
        print("\n  (empty front — every evaluation returned the sentinel vector)")
        return

    n_obj = len(result.objective_fields)
    for i, field in enumerate(result.objective_fields):
        k_show = min(5, len(result.front))
        print(f"\n  Top-{k_show} members by {field} (ascending):")
        ordered = sorted(result.front, key=lambda p: float(p.objectives[i]))[:k_show]
        obj_header = "  ".join(f"{f:>28s}" for f in result.objective_fields)
        print(f"    idx  {obj_header}  params")
        print("    " + "-" * (5 + 30 * n_obj + 8))
        for idx, p in enumerate(ordered):
            objs = "  ".join(f"{float(v):28.6e}" for v in p.objectives)
            print(f"    {idx:3d}  {objs}  {p.params}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.pareto",
        description=(
            "NSGA-II multi-objective Pareto front over Brady-Livescu 2D "
            "stability metrics.  Minimises 2+ objectives simultaneously and "
            "returns the non-dominated set.  Distinct from "
            "'gv-stability-pareto', a 1D parametric scan retained as a "
            "research / documentation aid."
        ),
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument("--kernel", choices=list(_KERNEL_CHOICES), required=True)
    parser.add_argument(
        "--objectives",
        nargs="+",
        required=True,
        metavar="FIELD",
        help=(
            "Two or more dotted-path report fields, e.g. "
            '"layer1.boundary_gv_err layer_bl42.max_spectral_abscissa".'
        ),
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=None,
        metavar="VAL",
        help="Flat list of bound pairs (lo hi [lo hi ...]). Falls back to DEFAULT_BOUNDS if absent.",
    )
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--n-gen", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--ref-point",
        type=float,
        nargs="+",
        default=None,
        metavar="V",
        help=(
            "Reference point for the hypervolume indicator (one value per "
            "--objectives). Default: 1.1 * max of 20 uniform-random feasible "
            "samples, auto-picked by run_nsga2."
        ),
    )
    parser.add_argument(
        "--gate-layer",
        type=int,
        default=None,
        help=(
            "Highest layer whose failure forces the sentinel vector. "
            "Default: max_layer-1 (auto-inferred from --objectives; floored at 0)."
        ),
    )
    parser.add_argument(
        "--max-layer",
        type=int,
        default=None,
        help=(
            "Highest layer executed per evaluation. "
            "Default: max(_infer_max_layer(f) for f in --objectives)."
        ),
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help=(
            "Persist the ParetoResult as JSON under "
            "sweeps/pareto_fronts/<scheme>_<kernel>_<mangled>.json (plan 45.4)."
        ),
    )
    parser.add_argument(
        "--validate-with-cpp",
        action="store_true",
        help=(
            "Re-run up to 10 front members at max_layer=8 via the C++ bridge "
            "(plan 45.5)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward to pymoo's minimize(verbose=True).",
    )

    args = parser.parse_args(argv)

    if len(args.objectives) < 2:
        parser.error(
            f"--objectives requires at least 2 fields; got {len(args.objectives)}"
        )

    try:
        bounds = _resolve_bounds(args.scheme, args.kernel, args.bounds)
        _validate_kernel_bounds_dim(args.kernel, bounds)
    except ValueError as exc:
        parser.error(str(exc))

    ref_point: tuple[float, ...] | None
    if args.ref_point is None:
        ref_point = None
    else:
        if len(args.ref_point) != len(args.objectives):
            parser.error(
                f"--ref-point length {len(args.ref_point)} does not match "
                f"--objectives length {len(args.objectives)}"
            )
        ref_point = tuple(float(v) for v in args.ref_point)

    try:
        result = run_nsga2(
            scheme=args.scheme,
            kernel=args.kernel,
            report_fields=args.objectives,
            bounds=bounds,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            seed=args.seed,
            ref_point=ref_point,
            gate_layer=args.gate_layer,
            max_layer=args.max_layer,
            verbose=args.verbose,
        )
    except ValueError as exc:
        parser.error(str(exc))

    _print_summary(result)

    if args.persist:
        written = save_pareto_front(result)
        print(f"\n[pareto] persisted front to {written}")

    if args.validate_with_cpp:
        # L8 bridge validation lands in plan 45.5a.  Until then, acknowledge
        # the request but do not pretend validation ran.
        print(
            "\n[pareto] --validate-with-cpp: wiring lands in plan 45.5; no "
            "L8 runs executed this invocation."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
