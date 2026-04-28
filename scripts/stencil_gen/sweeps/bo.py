"""Multi-fidelity Bayesian optimization CLI for the stability cascade.

Wraps :func:`stencil_gen.bo.run_mfbo` behind a ``sweeps bo`` subcommand.
Drives a cost-aware qMFKG loop over a configurable subset of the cascade's
layers (cheap surrogates + an HF target) and prints a summary table.

See ``plans/47-mfbo.md`` items 47.4a (this CLI), 47.4c (per-run JSON
persistence), 47.5a (``--validate-with-cpp`` wiring), and 47.5b
(``--baseline staged`` head-to-head against ``run_staged_optimize``) for
each capability.  Items 47.4c and 47.5a/b plug into the stub branches at
the bottom of :func:`main` — they share the parse + dispatch surface
defined here.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.bo import (
    DEFAULT_COST_TABLE,
    BOResult,
    run_mfbo,
)
from stencil_gen.optimizer import DEFAULT_BOUNDS, _infer_max_layer


_KERNEL_CHOICES = ("tension", "gaussian", "multiquadric", "classical")
_KERNEL_DIM = {"tension": 1, "gaussian": 1, "multiquadric": 1, "classical": 2}
_BASELINE_CHOICES = ("none", "staged")
_COST_MODEL_CHOICES = ("constant", "empirical")

# Default canonical report field for each external fidelity index.  Keys
# match :data:`stencil_gen.bo.DEFAULT_COST_TABLE`; in particular external
# index ``5`` is the synthetic slot for L3r (``layer_bl42``), which sits
# between L3 and L6 in cost.  ``brady2d_stability_score`` populates
# ``layer_bl42`` whenever ``max_layer >= 3``, so passing ``m=5`` to
# :func:`make_multi_fidelity_objective` runs more layers than strictly
# required for ``layer_bl42.max_spectral_abscissa``; we accept that small
# inefficiency in exchange for a unique fidelity index that the ICM
# kernel can identify separately from L3.
_DEFAULT_FIDELITY_FIELDS: dict[int, str] = {
    1: "layer1.boundary_gv_err",
    3: "layer3.max_stab_eig",
    5: "layer_bl42.max_spectral_abscissa",
    6: "layer6.transient_growth_bound",
    7: "layer7.max_spectral_abscissa",
}


def _mangle_objective(field: str) -> str:
    """Encode a single dotted-path objective into a filesystem-safe token.

    Mirrors :func:`sweeps.pareto._mangle_objectives` but for the BO
    subcommand's single-objective case — replaces ``.`` with ``_`` so the
    persisted filename ``<scheme>_<kernel>_<mangled>_<seed>.json`` (47.4c)
    stays readable and unambiguous.
    """
    return field.replace(".", "_")


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
    expected = _KERNEL_DIM[kernel]
    if len(bounds) != expected:
        raise ValueError(
            f"kernel={kernel!r} expects {expected} bound pair(s); "
            f"got {len(bounds)}"
        )


def _parse_fidelity_fields(raw: list[str] | None) -> dict[int, str]:
    """Parse ``--fidelity-fields LAYER=FIELD [...]`` into ``{int: str}``.

    Each item must look like ``5=layer_bl42.max_spectral_abscissa``; the
    layer index parses as ``int``, the field as the literal remainder.
    Empty / ``None`` input returns an empty dict (no overrides).
    """
    if not raw:
        return {}
    out: dict[int, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(
                f"--fidelity-fields expects LAYER=FIELD pairs; got {item!r}"
            )
        layer_str, _, field = item.partition("=")
        try:
            layer = int(layer_str)
        except ValueError as exc:
            raise ValueError(
                f"--fidelity-fields layer index must be an int; got {layer_str!r}"
            ) from exc
        if not field:
            raise ValueError(
                f"--fidelity-fields field is empty for LAYER={layer}"
            )
        out[layer] = field
    return out


def _build_report_fields_by_layer(
    objective: str,
    cheap_fidelities: list[int],
    overrides: dict[int, str],
) -> dict[int, str]:
    """Assemble the per-layer report-fields mapping for :func:`run_mfbo`.

    Combines:

    1. The HF entry: ``{hf_layer: objective}`` where ``hf_layer`` is
       inferred from *objective* via :func:`_infer_max_layer`.
    2. One entry per *cheap_fidelity*: pulls the canonical field from
       :data:`_DEFAULT_FIDELITY_FIELDS`, else from *overrides*.
    3. *overrides* always win (lets the caller swap, e.g.,
       ``layer3.max_stab_eig`` for ``layer3.something_else``).

    The cheap entries must all use a layer index strictly less than the
    HF layer — otherwise the user is mis-specifying the "cheap" surrogate
    set.  Raises :class:`ValueError` on any of: unknown HF layer prefix,
    cheap layer ≥ HF layer, missing field for a cheap layer (no default
    and no override), HF override conflicting with --objective.
    """
    hf_layer = _infer_max_layer(objective)
    if hf_layer is None:
        raise ValueError(
            f"cannot infer HF layer from --objective={objective!r}; pass an "
            "objective with a recognised layer prefix (layer1.* … layer8.*) "
            "or a known alias (kreiss.*, layer_bl42.*, non_normality.*)"
        )

    fields: dict[int, str] = {}
    for layer in cheap_fidelities:
        if layer >= hf_layer:
            raise ValueError(
                f"--cheap-fidelities entry {layer} must be strictly less than "
                f"the HF layer {hf_layer} (inferred from --objective)"
            )
        if layer in overrides:
            fields[layer] = overrides[layer]
        elif layer in _DEFAULT_FIDELITY_FIELDS:
            fields[layer] = _DEFAULT_FIDELITY_FIELDS[layer]
        else:
            raise ValueError(
                f"no default report field for --cheap-fidelities entry {layer}; "
                "supply one via --fidelity-fields LAYER=FIELD"
            )

    # HF: --objective always wins over any --fidelity-fields override on the
    # HF slot, since the user explicitly named the HF target.
    if hf_layer in overrides and overrides[hf_layer] != objective:
        raise ValueError(
            f"--fidelity-fields override at HF layer {hf_layer} "
            f"({overrides[hf_layer]!r}) conflicts with --objective={objective!r}"
        )
    fields[hf_layer] = objective
    return fields


def _print_summary(result: BOResult, *, baseline: dict | None = None) -> None:
    print(f"\n{'=' * 72}")
    print(f"  [bo] scheme={result.scheme}  kernel={result.kernel}  method={result.method}")
    print(f"  [bo] objective={result.report_fields_by_layer[result.hf_level]} (HF=L{result.hf_level})")
    print(f"  [bo] bounds={list(result.bounds)}")
    print(f"  [bo] fidelity_levels={list(result.fidelity_levels)}")
    print(f"{'=' * 72}")
    best_x = np.asarray(result.best_x, dtype=float).ravel()
    print(f"  best_x         = {np.array2string(best_x, precision=6)}")
    print(f"  best_params    = {result.best_params}")
    print(f"  best_objective = {result.best_objective:.6e}")
    print(f"  converged      = {result.converged}")
    print(f"  stop_reason    = {result.stop_reason}")
    print(f"  total_eval_count = {sum(result.n_evals_per_fidelity.values())}")
    print(f"  total_compute_time = {result.total_compute_time:.3f} s")

    print(f"\n  per-fidelity breakdown:")
    print(f"    {'layer':>6s}  {'n_evals':>8s}  {'wall (s)':>10s}  {'cost (s)':>10s}  field")
    print(f"    {'-' * 6}  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 30}")
    for layer in result.fidelity_levels:
        n = result.n_evals_per_fidelity.get(layer, 0)
        wt = result.wall_time_per_fidelity.get(layer, 0.0)
        cost = result.cost_model.get(layer, float("nan"))
        field = result.report_fields_by_layer.get(layer, "")
        marker = "  *" if layer == result.hf_level else "   "
        print(f"   {marker}{layer:>3d}  {n:>8d}  {wt:>10.3f}  {cost:>10.4f}  {field}")

    extras = result.extras or {}
    if extras:
        print(f"\n  extras:")
        for k, v in extras.items():
            if k == "baseline":  # printed separately below
                continue
            if isinstance(v, np.ndarray):
                print(f"    {k:<24s} = {np.array2string(v, precision=6)}")
            else:
                print(f"    {k:<24s} = {v}")

    if baseline is not None:
        print(f"\n  baseline (staged):")
        print(f"    best_x         = {baseline.get('best_x')}")
        print(f"    best_objective = {baseline.get('best_objective')}")
        print(f"    n_evals        = {baseline.get('n_evals')}")
        print(f"    compute_time   = {baseline.get('compute_time')} s")
        print(f"\n  comparison (mfbo vs staged):")
        b_obj = baseline.get("best_objective", float("nan"))
        b_t = baseline.get("compute_time", float("nan"))
        print(
            f"    objective: mfbo={result.best_objective:.6e}  "
            f"staged={b_obj}  delta={result.best_objective - b_obj:+.6e}"
            if isinstance(b_obj, float) else
            f"    objective: mfbo={result.best_objective:.6e}  staged={b_obj}"
        )
        print(
            f"    wall (s) : mfbo={result.total_compute_time:.3f}  "
            f"staged={b_t}"
        )


def _resolve_cost_table(
    cost_model: str,
    fidelity_layers: list[int],
) -> dict[int, float] | None:
    """Resolve the per-layer cost table to forward to :func:`run_mfbo`.

    Returns ``None`` when *cost_model* is ``"constant"`` and every layer
    in *fidelity_layers* has an entry in :data:`DEFAULT_COST_TABLE` —
    letting :func:`run_mfbo` apply its own default behaviour.
    Returns a fresh dict slice when explicit costs are needed (currently
    just the same default values, but kept as an explicit dict so future
    overrides plug in here).  Raises :class:`NotImplementedError` for the
    ``"empirical"`` choice (deferred to a future item).
    """
    if cost_model == "empirical":
        raise NotImplementedError(
            "--cost-model empirical is reserved for a future item; pass "
            "--cost-model constant (or omit the flag) to use the plan-46 "
            "calibrated table."
        )
    missing = [m for m in fidelity_layers if m not in DEFAULT_COST_TABLE]
    if missing:
        raise ValueError(
            f"DEFAULT_COST_TABLE has no entries for layers {missing}; "
            "pass --fidelity-fields with a custom cost table is not yet "
            "supported on the CLI — file a follow-up if needed."
        )
    return None  # let run_mfbo slice DEFAULT_COST_TABLE itself


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.bo",
        description=(
            "Multi-fidelity Bayesian optimisation (BoTorch qMFKG) over the "
            "Brady-Livescu stability cascade.  Picks (x, m) jointly to "
            "maximise expected information gain at the HF target per second "
            "of wall time.  See plans/47-mfbo.md."
        ),
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument("--kernel", choices=list(_KERNEL_CHOICES), required=True)
    parser.add_argument(
        "--objective",
        required=True,
        help=(
            'HF target as a dotted-path report field, e.g. '
            '"layer7.max_spectral_abscissa".  The HF layer is inferred from '
            "the prefix."
        ),
    )
    parser.add_argument(
        "--cheap-fidelities",
        type=int,
        nargs="+",
        required=True,
        metavar="N",
        help=(
            "External cascade layer indices to use as cheap surrogates, e.g. "
            "'1 3' or '1 3 5 6'.  Each must be < the HF layer inferred from "
            "--objective.  Default field per layer comes from a built-in "
            "table; override with --fidelity-fields."
        ),
    )
    parser.add_argument(
        "--fidelity-fields",
        nargs="+",
        default=None,
        metavar="LAYER=FIELD",
        help=(
            "Per-layer field overrides, e.g. '3=layer3.something_else'.  "
            "Useful when the canonical default is not what you want."
        ),
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=None,
        metavar="VAL",
        help=(
            "Flat list of bound pairs (lo hi [lo hi ...]).  Falls back to "
            "DEFAULT_BOUNDS for the (scheme, kernel) pair if absent."
        ),
    )
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument(
        "--budget-evals",
        type=int,
        default=None,
        help="Total number of cascade evaluations (init + acquisition + final HF).",
    )
    budget.add_argument(
        "--budget-seconds",
        type=float,
        default=None,
        help="Wall-time budget in seconds (mutually exclusive with --budget-evals).",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Initial design size (default: 5*d + 3 per Loeppky et al. 2009).",
    )
    parser.add_argument(
        "--num-fantasies",
        type=int,
        default=64,
        help="Number of fantasies for qMFKG (default: 64).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--cost-model",
        choices=list(_COST_MODEL_CHOICES),
        default="constant",
        help=(
            "'constant' uses the plan-46 calibrated DEFAULT_COST_TABLE.  "
            "'empirical' (per-eval learned cost) is reserved for a future item."
        ),
    )
    parser.add_argument(
        "--baseline",
        choices=list(_BASELINE_CHOICES),
        default="none",
        help=(
            "Run a comparator alongside MF-BO with the same seed.  'staged' "
            "invokes run_staged_optimize against the same HF objective.  "
            "Wired in plan 47.5b."
        ),
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help=(
            "Persist the BOResult as JSON under "
            "sweeps/bo_runs/<scheme>_<kernel>_<mangled>_<seed>.json (plan 47.4c)."
        ),
    )
    parser.add_argument(
        "--validate-with-cpp",
        action="store_true",
        help=(
            "Re-run best_x at max_layer=8 via the C++ bridge after MF-BO "
            "completes (plan 47.5a)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward to run_mfbo(verbose=True): one line per evaluation.",
    )

    args = parser.parse_args(argv)

    try:
        bounds = _resolve_bounds(args.scheme, args.kernel, args.bounds)
        _validate_kernel_bounds_dim(args.kernel, bounds)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        overrides = _parse_fidelity_fields(args.fidelity_fields)
        report_fields_by_layer = _build_report_fields_by_layer(
            args.objective,
            list(args.cheap_fidelities),
            overrides,
        )
    except ValueError as exc:
        parser.error(str(exc))

    fidelity_layers = sorted(report_fields_by_layer)
    try:
        cost_table = _resolve_cost_table(args.cost_model, fidelity_layers)
    except (NotImplementedError, ValueError) as exc:
        parser.error(str(exc))

    try:
        result = run_mfbo(
            scheme=args.scheme,
            kernel=args.kernel,
            report_fields_by_layer=report_fields_by_layer,
            bounds=bounds,
            budget_evals=args.budget_evals,
            budget_seconds=args.budget_seconds,
            cost_table=cost_table,
            seed=args.seed,
            n_init=args.n_init,
            num_fantasies=args.num_fantasies,
            verbose=args.verbose,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # --- baseline (47.5b) ---------------------------------------------------
    baseline_record: dict | None = None
    if args.baseline == "staged":
        # The full implementation lives in 47.5b: run run_staged_optimize at
        # the same (scheme, kernel, objective, bounds, seed), serialise via
        # _result_to_persist_dict, and store under result.extras["baseline"].
        # For 47.4a we leave the wiring stub here so the flag parses today
        # without changing the dispatch contract when 47.5b lands.
        print(
            "\n[bo] --baseline staged: deferred to plan 47.5b; flag accepted "
            "but no baseline run executed yet."
        )

    # --- C++ validation (47.5a) --------------------------------------------
    if args.validate_with_cpp:
        # The full implementation lives in 47.5a: re-run result.best_x at
        # max_layer=8 via brady2d_stability_score and store under
        # result.extras["cpp_validation"].  Validation must run BEFORE
        # --persist so the persisted JSON includes the cpp_validation
        # payload (lesson from plan 45.5a.1).
        print(
            "\n[bo] --validate-with-cpp: deferred to plan 47.5a; flag accepted "
            "but no L8 re-evaluation executed yet."
        )

    _print_summary(result, baseline=baseline_record)

    # --- persistence (47.4c) -----------------------------------------------
    if args.persist:
        try:
            from ._bo_io import save_bo_run
        except ImportError:
            print(
                "\n[bo] --persist: deferred to plan 47.4c; sweeps/_bo_io.py "
                "not yet present.  No file written."
            )
        else:
            written = save_bo_run(result)
            print(f"\n[bo] persisted run to {written}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
