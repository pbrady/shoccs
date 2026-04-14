"""Optimization CLI for Brady-Livescu 2D stability objectives.

Wraps the drivers in :mod:`stencil_gen.optimizer` behind a ``sweeps optimize``
subcommand.  Picks one of Nelder-Mead, COBYQA, SHGO, differential_evolution,
or the staged cheap-inner + expensive-validator pipeline, runs it against a
kernel-specific parameter vector, and prints a summary.

See ``plans/43-stability-optimization-framework.md`` items 43.7 and 43.8 for
the argparse surface and persistence schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Any

import numpy as np

from stencil_gen.optimizer import (
    DEFAULT_BOUNDS,
    OptimizeResult,
    make_objective,
    multi_start_optimize,
    params_from_vector,
    run_scipy_de,
    run_scipy_shgo,
    run_staged_optimize,
)

from ._common import load_known_values, save_known_values

_METHOD_CHOICES = ("Nelder-Mead", "COBYQA", "SHGO", "DE", "staged")
_KERNEL_CHOICES = ("tension", "gaussian", "multiquadric", "classical")
_KERNEL_DIM = {"tension": 1, "gaussian": 1, "multiquadric": 1, "classical": 2}


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

    Without this gate, a mismatch (e.g. ``--kernel classical --bounds 0.5 20``)
    would feed a 1D Sobol start into a 2D ``params_from_vector``, the exception
    would be swallowed by ``make_objective``'s try/except, and every evaluation
    would silently return ``+inf`` — the CLI would exit 0 with an infeasible
    result instead of flagging the user error.
    """
    expected = _KERNEL_DIM[kernel]
    if len(bounds) != expected:
        raise ValueError(
            f"kernel={kernel!r} expects {expected} bound pair(s); "
            f"got {len(bounds)}"
        )


def _result_to_persist_dict(
    result: OptimizeResult,
    *,
    scheme: str,
    kernel: str,
    objective: str,
    bounds: list[tuple[float, float]],
) -> dict[str, Any]:
    """Serialise ``result`` to a JSON-friendly dict, dropping ``history``."""
    return {
        "scheme": scheme,
        "kernel": kernel,
        "objective": objective,
        "bounds": [list(b) for b in bounds],
        "best_x": [float(v) for v in np.asarray(result.best_x, dtype=float).ravel()],
        "best_params": result.best_params,
        "best_objective": float(result.best_objective),
        "method": result.method,
        "n_evals": int(result.n_evals),
        "compute_time": float(result.compute_time),
        "converged": bool(result.converged),
        "best_report": result.best_report,
    }


def _print_summary(
    result: OptimizeResult,
    *,
    scheme: str,
    kernel: str,
    objective: str,
    bounds: list[tuple[float, float]],
) -> None:
    print(f"\n{'=' * 64}")
    print(f"  [optimize] scheme={scheme}  kernel={kernel}  method={result.method}")
    print(f"  [optimize] objective={objective}")
    print(f"  [optimize] bounds={bounds}")
    print(f"{'=' * 64}")
    best_x = np.asarray(result.best_x, dtype=float).ravel()
    print(f"  best_x         = {np.array2string(best_x, precision=6)}")
    print(f"  best_params    = {result.best_params}")
    print(f"  best_objective = {result.best_objective:.6e}")
    print(f"  converged      = {result.converged}")
    print(f"  n_evals        = {result.n_evals}")
    print(f"  compute_time   = {result.compute_time:.3f} s")
    extras = result.extras or {}
    for k, v in extras.items():
        if k in ("validator_ranking", "local_minima"):
            try:
                n = len(v)
            except TypeError:
                n = "?"
            print(f"  extras.{k:<20s} (len={n})")
        elif isinstance(v, np.ndarray):
            print(f"  extras.{k:<20s} = {np.array2string(v, precision=6)}")
        else:
            print(f"  extras.{k:<20s} = {v}")


def _run_method(
    args: argparse.Namespace,
    bounds: list[tuple[float, float]],
) -> OptimizeResult:
    method = args.method
    if method == "staged":
        return run_staged_optimize(
            scheme=args.scheme,
            kernel=args.kernel,
            report_field=args.objective,
            bounds=bounds,
            inner_gate=args.gate_layer,
            inner_max_layer=args.max_layer if args.max_layer is not None else 3,
            validator_max_layer=args.validator_max_layer,
            top_k=args.top_k,
            method=args.inner_method,
            n_restarts=args.n_restarts,
            seed=args.seed,
            max_evals=args.max_evals,
        )

    f = make_objective(
        scheme=args.scheme,
        kernel=args.kernel,
        report_field=args.objective,
        gate_layer=args.gate_layer,
        max_layer=args.max_layer,
    )

    if method in ("Nelder-Mead", "COBYQA"):
        result = multi_start_optimize(
            f,
            bounds=bounds,
            n_restarts=args.n_restarts,
            method=method,
            seed=args.seed,
            max_evals=args.max_evals,
        )
        # multi_start_optimize is kernel-agnostic, so fill in best_params here.
        if np.isfinite(result.best_objective):
            return replace(
                result,
                best_params=params_from_vector(args.kernel, result.best_x),
            )
        return result

    if method == "SHGO":
        result = run_scipy_shgo(f, bounds=bounds, n=args.shgo_n, iters=args.shgo_iters)
        if np.isfinite(result.best_objective):
            return replace(
                result,
                best_params=params_from_vector(args.kernel, result.best_x),
            )
        return result

    if method == "DE":
        result = run_scipy_de(
            f,
            bounds=bounds,
            popsize=args.de_popsize,
            maxiter=args.de_maxiter,
            seed=args.seed,
        )
        if np.isfinite(result.best_objective):
            return replace(
                result,
                best_params=params_from_vector(args.kernel, result.best_x),
            )
        return result

    raise ValueError(f"unknown method: {method}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.optimize",
        description=(
            "Optimize boundary-closure parameters against a Brady-Livescu 2D "
            "stability objective (layered cascade)."
        ),
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument("--kernel", choices=list(_KERNEL_CHOICES), required=True)
    parser.add_argument(
        "--objective",
        required=True,
        help='Dotted-path report field (e.g. "layer3.max_stab_eig", "layer6.transient_growth_bound").',
    )
    parser.add_argument("--gate-layer", type=int, default=3)
    parser.add_argument(
        "--max-layer",
        type=int,
        default=None,
        help="Highest layer run by the objective (default: inferred from --objective).",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=None,
        metavar="VAL",
        help="Flat list of bound pairs (lo hi [lo hi ...]).  If absent, falls back to DEFAULT_BOUNDS.",
    )
    parser.add_argument(
        "--method",
        choices=list(_METHOD_CHOICES),
        default="staged",
    )
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument(
        "--max-evals",
        type=int,
        default=200,
        help="Max objective evaluations per local run (Nelder-Mead / COBYQA / staged inner).",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Staged-specific knobs
    parser.add_argument(
        "--validator-max-layer",
        type=int,
        default=6,
        help="Validator stage max_layer (staged method only; default: 6).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of inner-stage survivors re-evaluated by the validator (staged method only).",
    )
    parser.add_argument(
        "--inner-method",
        choices=["Nelder-Mead", "COBYQA"],
        default="Nelder-Mead",
        help="Local method used inside the staged inner multi-start (default: Nelder-Mead).",
    )

    # SHGO-specific knobs
    parser.add_argument("--shgo-n", type=int, default=100)
    parser.add_argument("--shgo-iters", type=int, default=3)

    # DE-specific knobs
    parser.add_argument("--de-popsize", type=int, default=15)
    parser.add_argument("--de-maxiter", type=int, default=100)

    # Post-run knobs
    parser.add_argument(
        "--validate-with-cpp",
        action="store_true",
        help="Re-run the winner at max_layer=8 via the C++ bridge (plan 43.10a).",
    )
    parser.add_argument(
        "--update-known-values",
        action="store_true",
        help='Persist the result to known_values.json["brady2d_optima"][scheme][kernel][objective].',
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to write the full result as JSON.",
    )

    args = parser.parse_args(argv)

    try:
        bounds = _resolve_bounds(args.scheme, args.kernel, args.bounds)
        _validate_kernel_bounds_dim(args.kernel, bounds)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        result = _run_method(args, bounds=bounds)
    except ValueError as exc:
        parser.error(str(exc))

    _print_summary(
        result,
        scheme=args.scheme,
        kernel=args.kernel,
        objective=args.objective,
        bounds=bounds,
    )

    persisted = _result_to_persist_dict(
        result,
        scheme=args.scheme,
        kernel=args.kernel,
        objective=args.objective,
        bounds=bounds,
    )

    if args.validate_with_cpp:
        # Stubbed: full implementation lands in 43.10a.
        print("\n[optimize] --validate-with-cpp is wired in plan item 43.10a; skipping.")

    if args.update_known_values:
        kv = load_known_values()
        optima_root = kv.setdefault("brady2d_optima", {})
        scheme_bucket = optima_root.setdefault(args.scheme, {})
        kernel_bucket = scheme_bucket.setdefault(args.kernel, {})
        kernel_bucket[args.objective] = persisted
        save_known_values(kv)
        print(
            f'\n[optimize] Updated known_values.json: '
            f"brady2d_optima.{args.scheme}.{args.kernel}.{args.objective}"
        )

    if args.json_output is not None:
        with open(args.json_output, "w") as fp:
            json.dump(persisted, fp, indent=2)
            fp.write("\n")
        print(f"[optimize] Wrote JSON result to {args.json_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
