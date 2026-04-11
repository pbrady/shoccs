"""Tension spline sigma parameter sweep for RBF-augmented stencils.

Extracted from TestCorrectedTensionE2, TestCorrectedTensionE4,
TestTensionSweepE2, TestTensionSweepE4 in test_phs.py.

Sweeps the tension parameter sigma over a range including sigma=0
(PHS k=2 limit) and reports stability of the resulting differentiation
matrix at each grid size.

Usage:
    uv run python -m sweeps.tension_sweep --scheme E2
    uv run python -m sweeps.tension_sweep --scheme E4
    uv run python -m sweeps.tension_sweep --scheme E2 --n-sigma 10  # quick smoke test
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.phs import stability_eigenvalue

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
    n_values: list[int],
    sigmas: np.ndarray,
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
    """Run sigma sweep using stability_eigenvalue with tension kernel.

    Returns ``(stab_results, gv_by_sigma)`` where ``stab_results`` maps
    ``n -> list of (sigma, stab_eig)`` and ``gv_by_sigma`` maps
    ``sigma -> boundary_gv_error_max`` (independent of ``n``) when
    ``include_gv`` is set, else ``None``.
    """
    results: dict[int, list[tuple[float, float]]] = {}
    gv_by_sigma: dict[float, float] | None = {} if include_gv else None
    if include_gv:
        for sigma in sigmas:
            gv_by_sigma[float(sigma)] = boundary_gv_error_max(
                p=p, q=q, nextra=nextra, nu=nu,
                sigma=float(sigma), kernel="tension",
            )
    for n in n_values:
        rows = []
        for sigma in sigmas:
            se = stability_eigenvalue(
                n, p=p, q=q, epsilon=sigma,
                kernel="tension", nu=nu, nextra=nextra,
            )
            rows.append((float(sigma), se))
        results[n] = rows
    return results, gv_by_sigma


def fine_sweep(
    n: int,
    sigmas_coarse: np.ndarray,
    *,
    p: int,
    q: int,
    nextra: int,
    nu: int,
    n_fine: int = 200,
) -> tuple[float, float, float, float]:
    """Coarse-then-fine sweep at a single grid size.

    Returns (best_coarse_sigma, best_coarse_se, best_fine_sigma, best_fine_se).
    """
    coarse = []
    for sigma in sigmas_coarse:
        se = stability_eigenvalue(
            n, p=p, q=q, epsilon=sigma,
            kernel="tension", nu=nu, nextra=nextra,
        )
        coarse.append((float(sigma), se))

    best_coarse = min(coarse, key=lambda r: r[1])
    sigma_best = best_coarse[0]

    # Fine sweep: ±factor around best (or [0, 2] if best is near 0)
    if sigma_best < 0.1:
        lo, hi = 0.0, 2.0
    else:
        lo = max(0.0, sigma_best / 5)
        hi = sigma_best * 5
    sigmas_fine = np.linspace(lo, hi, n_fine)
    fine = []
    for sigma in sigmas_fine:
        se = stability_eigenvalue(
            n, p=p, q=q, epsilon=sigma,
            kernel="tension", nu=nu, nextra=nextra,
        )
        fine.append((float(sigma), se))

    best_fine = min(fine, key=lambda r: r[1])
    return best_coarse[0], best_coarse[1], best_fine[0], best_fine[1]


def run_tension_sweep(
    scheme: str,
    n_values: list[int],
    n_sigma: int,
    sigma_max: float = 20.0,
    *,
    include_gv: bool = False,
) -> dict:
    """Run a full tension sigma sweep for a scheme.

    Returns a summary dict with best sigma and stable grid sizes.
    """
    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]
    label = params["label"]

    # Include sigma=0 (PHS k=2 limit) plus logarithmic spacing for sigma > 0
    sigmas = np.concatenate(
        [[0.0], np.logspace(np.log10(0.01), np.log10(sigma_max), n_sigma)]
    )

    # Main sweep
    results, gv_by_sigma = sweep_stability(
        n_values, sigmas,
        p=p, q=q, nextra=nextra, nu=nu,
        include_gv=include_gv,
    )
    print_sweep_table(
        f"{label} Tension Spline — Stability Sweep (p={p}, q={q}, nextra={nextra})",
        results,
        param_label="sigma",
    )
    print()
    report_stable_ranges(results, param_label="sigma")

    # Fine sweep at n=40 (or largest available)
    n_fine_grid = 40 if 40 in n_values else max(n_values)
    coarse_sigma, coarse_se, fine_sigma, fine_se = fine_sweep(
        n_fine_grid, sigmas,
        p=p, q=q, nextra=nextra, nu=nu,
    )
    stable = fine_se < STABILITY_TOL
    print(f"\n  Fine sweep (n={n_fine_grid}):")
    print(f"  Coarse best: sigma={coarse_sigma:.6f}, stab_eig={coarse_se:.6e}")
    print(f"  Fine best:   sigma={fine_sigma:.6f}, stab_eig={fine_se:.6e}")
    print(f"  Stable: {stable}")

    # Check fine-sweep best across all grid sizes
    sigma_star = fine_sigma
    stable_at = []
    print(f"\n  Checking sigma*={sigma_star:.6f} across grid sizes:")
    for nn in sorted(set(n_values + [20, 40, 80, 160])):
        se = stability_eigenvalue(
            nn, p=p, q=q, epsilon=sigma_star,
            kernel="tension", nu=nu, nextra=nextra,
        )
        status = "STABLE" if se < STABILITY_TOL else "unstable"
        print(f"    n={nn:4d}: stab_eig={se:.6e} [{status}]")
        if se < STABILITY_TOL:
            stable_at.append(nn)

    return {
        "sigma": round(sigma_star, 6),
        "stable_at": stable_at,
        "fine_stab_eig": fine_se,
        "gv_by_sigma": gv_by_sigma,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.tension_sweep",
        description="Tension spline sigma parameter sweep for RBF-augmented stencils",
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument(
        "--n-values", default="20,40,80",
        help="Comma-separated grid sizes (default: 20,40,80)",
    )
    parser.add_argument(
        "--n-sigma", type=int, default=61,
        help="Number of sigma sample points in coarse sweep (default: 61)",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=20.0,
        help="Maximum sigma value in coarse sweep (default: 20.0)",
    )
    parser.add_argument(
        "--update-known-values", action="store_true",
        help="Update known_values.json with discovered optimal sigma",
    )
    parser.add_argument(
        "--include-gv", action="store_true",
        help="Also compute boundary group-velocity error at each sigma "
             "(advisory secondary objective; does not alter the stability optimum)",
    )

    args = parser.parse_args(argv)
    n_values = [int(x) for x in args.n_values.split(",")]

    summary = run_tension_sweep(
        args.scheme, n_values, args.n_sigma, args.sigma_max,
        include_gv=args.include_gv,
    )

    if args.update_known_values:
        kv = load_known_values()
        scheme_key = SCHEME_PARAMS[args.scheme]["label"]
        if scheme_key not in kv:
            kv[scheme_key] = {}
        kv[scheme_key]["tension"] = {
            "sigma": summary["sigma"],
            "stable_at": summary["stable_at"],
        }
        save_known_values(kv)
        print(f"\n  Updated known_values.json: {scheme_key}.tension")

    return 0


if __name__ == "__main__":
    sys.exit(main())
