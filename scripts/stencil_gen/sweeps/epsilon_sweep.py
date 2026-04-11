"""Epsilon (Gaussian/Multiquadric) parameter sweep for RBF-augmented stencils.

Extracted from TestCorrectedSweepE2, TestCorrectedSweepE4, TestEpsilonSweepE2,
TestEpsilonSweepE4 in test_phs.py.

Sweeps the shape parameter epsilon over a log-spaced range and reports
stability of the resulting differentiation matrix at each grid size.

Usage:
    uv run python -m sweeps.epsilon_sweep --scheme E2 --kernel gaussian
    uv run python -m sweeps.epsilon_sweep --scheme E4 --kernel multiquadric
    uv run python -m sweeps.epsilon_sweep --scheme E2 --n-eps 10  # quick smoke test
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.phs import (
    build_diff_matrix_rbf,
    stability_eigenvalue,
    stability_eigenvalue_from_matrix,
)

from ._common import (
    SCHEME_PARAMS,
    STABILITY_TOL,
    load_known_values,
    print_sweep_table,
    report_stable_ranges,
    save_known_values,
)
from .gv_objectives import boundary_gv_error_max

# Floating-point eigenvalue solvers return tiny positive real parts (~1e-14)
# for genuinely stable operators.  Use this threshold to distinguish true
# instability from numerical noise.

def sweep_stability(
    kernel: str,
    n_values: list[int],
    epsilons: np.ndarray,
    *,
    p: int,
    q: int,
    nextra: int,
    nu: int,
    include_gv: bool = False,
) -> tuple[
    dict[int, list[tuple[float, float]]],
    dict[float, float] | None,
]:
    """Run epsilon sweep using stability_eigenvalue.

    Returns ``(stab_results, gv_by_eps)`` where ``stab_results`` maps
    ``n -> list of (eps, stab_eig)`` and ``gv_by_eps`` maps
    ``eps -> boundary_gv_error_max`` (independent of ``n``) when
    ``include_gv`` is set, else ``None``.
    """
    results: dict[int, list[tuple[float, float]]] = {}
    gv_by_eps: dict[float, float] | None = {} if include_gv else None
    if include_gv:
        for eps in epsilons:
            gv_by_eps[float(eps)] = boundary_gv_error_max(
                p=p, q=q, nextra=nextra, nu=nu,
                sigma=float(eps), kernel=kernel,
            )
    for n in n_values:
        rows = []
        for eps in epsilons:
            se = stability_eigenvalue(
                n, p=p, q=q, epsilon=eps,
                kernel=kernel, nu=nu, nextra=nextra,
            )
            rows.append((float(eps), se))
        results[n] = rows
    return results, gv_by_eps


def fine_sweep(
    n: int,
    kernel: str,
    epsilons_coarse: np.ndarray,
    *,
    p: int,
    q: int,
    nextra: int,
    nu: int,
    n_fine: int = 200,
) -> tuple[float, float, float, float]:
    """Coarse-then-fine sweep at a single grid size.

    Returns (best_coarse_eps, best_coarse_se, best_fine_eps, best_fine_se).
    """
    coarse = []
    for eps in epsilons_coarse:
        se = stability_eigenvalue(
            n, p=p, q=q, epsilon=eps,
            kernel=kernel, nu=nu, nextra=nextra,
        )
        coarse.append((float(eps), se))

    best_coarse = min(coarse, key=lambda r: r[1])
    eps_best = best_coarse[0]

    # Fine sweep: ±1 decade around best
    lo = max(0.001, eps_best / 10)
    hi = min(100, eps_best * 10)
    epsilons_fine = np.linspace(lo, hi, n_fine)
    fine = []
    for eps in epsilons_fine:
        se = stability_eigenvalue(
            n, p=p, q=q, epsilon=eps,
            kernel=kernel, nu=nu, nextra=nextra,
        )
        fine.append((float(eps), se))

    best_fine = min(fine, key=lambda r: r[1])
    return best_coarse[0], best_coarse[1], best_fine[0], best_fine[1]


def run_epsilon_sweep(
    scheme: str,
    kernel: str,
    n_values: list[int],
    n_eps: int,
    *,
    include_gv: bool = False,
) -> dict:
    """Run a full epsilon sweep for a scheme/kernel combination.

    Returns a summary dict with best epsilon and stable grid sizes.
    """
    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]
    label = params["label"]

    epsilons = np.logspace(np.log10(0.01), np.log10(10), n_eps)

    # Main sweep
    results, gv_by_eps = sweep_stability(
        kernel, n_values, epsilons,
        p=p, q=q, nextra=nextra, nu=nu,
        include_gv=include_gv,
    )
    print_sweep_table(
        f"{label} {kernel.capitalize()} — Stability Sweep (p={p}, q={q}, nextra={nextra})",
        results,
        param_label="epsilon",
    )
    print()
    report_stable_ranges(results, param_label="epsilon")

    # Fine sweep at n=40 (or largest available)
    n_fine_grid = 40 if 40 in n_values else max(n_values)
    coarse_eps, coarse_se, fine_eps, fine_se = fine_sweep(
        n_fine_grid, kernel, epsilons,
        p=p, q=q, nextra=nextra, nu=nu,
    )
    stable = fine_se < STABILITY_TOL
    print(f"\n  Fine sweep (n={n_fine_grid}):")
    print(f"  Coarse best: eps={coarse_eps:.6f}, stab_eig={coarse_se:.6e}")
    print(f"  Fine best:   eps={fine_eps:.6f}, stab_eig={fine_se:.6e}")
    print(f"  Stable: {stable}")

    # Check fine-sweep best across all grid sizes
    eps_star = fine_eps
    stable_at = []
    print(f"\n  Checking eps*={eps_star:.6f} across grid sizes:")
    for nn in sorted(set(n_values + [20, 40, 80, 160])):
        se = stability_eigenvalue(
            nn, p=p, q=q, epsilon=eps_star,
            kernel=kernel, nu=nu, nextra=nextra,
        )
        status = "STABLE" if se < STABILITY_TOL else "unstable"
        print(f"    n={nn:4d}: stab_eig={se:.6e} [{status}]")
        if se < STABILITY_TOL:
            stable_at.append(nn)

    return {
        "epsilon": round(eps_star, 6),
        "stable_at": stable_at,
        "fine_stab_eig": fine_se,
        "gv_by_eps": gv_by_eps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.epsilon_sweep",
        description="Epsilon (Gaussian/MQ) parameter sweep for RBF-augmented stencils",
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument(
        "--kernel", choices=["gaussian", "multiquadric"], default="gaussian",
    )
    parser.add_argument(
        "--n-values", default="20,40,80",
        help="Comma-separated grid sizes (default: 20,40,80)",
    )
    parser.add_argument(
        "--n-eps", type=int, default=60,
        help="Number of epsilon sample points in coarse sweep (default: 60)",
    )
    parser.add_argument(
        "--update-known-values", action="store_true",
        help="Update known_values.json with discovered optimal epsilon",
    )
    parser.add_argument(
        "--include-gv", action="store_true",
        help="Also compute boundary group-velocity error at each epsilon "
             "(advisory secondary objective; does not alter the stability optimum)",
    )

    args = parser.parse_args(argv)
    n_values = [int(x) for x in args.n_values.split(",")]

    summary = run_epsilon_sweep(
        args.scheme, args.kernel, n_values, args.n_eps,
        include_gv=args.include_gv,
    )

    if args.update_known_values:
        kv = load_known_values()
        scheme_key = SCHEME_PARAMS[args.scheme]["label"]
        if scheme_key not in kv:
            kv[scheme_key] = {}
        kv[scheme_key][args.kernel] = {
            "epsilon": summary["epsilon"],
            "stable_at": summary["stable_at"],
        }
        save_known_values(kv)
        print(f"\n  Updated known_values.json: {scheme_key}.{args.kernel}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
