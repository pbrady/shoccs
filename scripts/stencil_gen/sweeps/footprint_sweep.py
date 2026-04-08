"""Stencil footprint (nextra) sweep for E4 tension kernel.

Extracted from TestCorrectedFootprint, TestFootprintE4Quick,
TestFootprintSweep, TestFootprintPenalty in test_phs.py.

Sweeps nextra (number of extra boundary rows) across sigma values
to determine how stencil footprint affects stability.  Optionally
includes a conservation penalty (gamma) dimension.

Three phases:
  1. nextra x sigma sweep — stability landscape per nextra
  2. nextra x sigma x gamma penalty sweep — stability-conservation trade-off
  3. Grid independence — verify best parameters across grid sizes

Usage:
    uv run python -m sweeps.footprint_sweep
    uv run python -m sweeps.footprint_sweep --n-sigma 10  # quick smoke test
    uv run python -m sweeps.footprint_sweep --update-known-values
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.phs import (
    build_diff_matrix_rbf_penalty,
    stability_eigenvalue,
    stability_eigenvalue_from_matrix,
)

from ._common import STABILITY_TOL, load_known_values, save_known_values

# E4 scheme parameters (footprint sweep is E4-only)
P = 2
Q = 3
NU = 1


def run_nextra_sigma_sweep(
    n: int,
    nextra_values: list[int],
    n_sigma: int,
    sigma_max: float,
) -> dict[int, dict]:
    """Phase 1: nextra x sigma sweep.

    For each nextra, sweeps sigma in [0, sigma_max] and reports stability.
    Returns {nextra: {best_sigma, best_se, n_stable, total, rows}}.
    """
    sigmas = np.concatenate([
        [0.0],
        np.logspace(np.log10(0.01), np.log10(sigma_max), n_sigma),
    ])

    results = {}
    for nx in nextra_values:
        r = Q + 1 + nx
        if n < 2 * r:
            print(f"  nextra={nx}: grid too small (n={n} < 2*r={2*r}), skipping")
            continue

        rows = []
        for sigma in sigmas:
            se = stability_eigenvalue(
                n, p=P, q=Q, epsilon=sigma,
                kernel="tension", nu=NU, nextra=nx,
            )
            rows.append((float(sigma), se))

        best_sigma, best_se = min(rows, key=lambda r: r[1])
        n_stable = sum(1 for _, se in rows if se < STABILITY_TOL)

        results[nx] = {
            "best_sigma": best_sigma,
            "best_se": best_se,
            "n_stable": n_stable,
            "total": len(rows),
            "rows": rows,
        }

    # Print table
    print(f"\n{'='*80}")
    print(f"  E4 Tension — nextra x sigma Sweep (n={n})")
    print(f"{'='*80}")

    # Header
    header = f"  {'sigma':>10s}"
    for nx in nextra_values:
        if nx in results:
            header += f"  {'nx=' + str(nx):>16s}"
    print(header)
    divider = f"  {'-'*10}"
    for nx in nextra_values:
        if nx in results:
            divider += f"  {'-'*16}"
    print(divider)

    # Print every 5th row to keep output readable
    n_rows = len(sigmas)
    for idx in range(0, n_rows, max(1, n_rows // 20)):
        line = f"  {sigmas[idx]:10.4f}"
        for nx in nextra_values:
            if nx in results:
                _, se = results[nx]["rows"][idx]
                marker = " *" if se < STABILITY_TOL else ""
                line += f"  {se:14.6e}{marker}"
        print(line)

    # Summary
    print(f"\n  {'='*70}")
    print(f"  Summary: Stability per nextra")
    print(f"  {'='*70}")
    print(f"  {'nextra':>6s}  {'t':>3s}  {'r':>3s}  "
          f"{'extra DOF':>9s}  {'best sigma':>10s}  {'stab_eig':>16s}  "
          f"{'stable':>8s}  {'status':>10s}")
    print(f"  {'-'*6}  {'-'*3}  {'-'*3}  "
          f"{'-'*9}  {'-'*10}  {'-'*16}  "
          f"{'-'*8}  {'-'*10}")

    for nx in nextra_values:
        if nx not in results:
            continue
        res = results[nx]
        t = P + Q + 1 + nx
        r = Q + 1 + nx
        extra_dof = r * (P + nx)
        status = "STABLE" if res["best_se"] < STABILITY_TOL else "unstable"
        print(f"  {nx:6d}  {t:3d}  {r:3d}  "
              f"{extra_dof:9d}  {res['best_sigma']:10.4f}  {res['best_se']:16.6e}  "
              f"{res['n_stable']:>3d}/{res['total']:<3d}  {status:>10s}")

    return results


def run_nextra_penalty_sweep(
    n: int,
    nextra_values: list[int],
    n_sigma: int,
    n_gamma: int,
    sigma_max: float,
) -> dict[int, dict]:
    """Phase 2: nextra x sigma x gamma penalty sweep.

    For each nextra, sweeps (sigma, gamma) and reports the best point
    and gamma=0 baseline.
    Returns {nextra: {gamma0_se, gamma0_sigma, best_se, best_sigma, best_gamma, best_deficit}}.
    """
    sigmas = np.concatenate([
        [0.0],
        np.logspace(np.log10(0.01), np.log10(sigma_max), n_sigma),
    ])
    gammas = np.concatenate([
        [0.0],
        np.logspace(-1, 3, n_gamma),  # 0.1 to 1000
    ])

    results = {}
    for nx in nextra_values:
        r = Q + 1 + nx
        if n < 2 * r:
            continue

        best_gamma0_se = float("inf")
        best_gamma0_sigma = None
        best_se = float("inf")
        best_sigma = None
        best_gamma = None
        best_deficit = None

        for sigma in sigmas:
            for gamma in gammas:
                D = build_diff_matrix_rbf_penalty(
                    n, P, Q, sigma, "tension", NU, nx,
                    gamma=gamma,
                )
                se = stability_eigenvalue_from_matrix(D)
                deficit = float(np.max(np.abs(np.sum(D, axis=0))))

                if se < best_se:
                    best_se = se
                    best_sigma = sigma
                    best_gamma = gamma
                    best_deficit = deficit

                if gamma == 0.0 and se < best_gamma0_se:
                    best_gamma0_se = se
                    best_gamma0_sigma = sigma

        results[nx] = {
            "gamma0_se": best_gamma0_se,
            "gamma0_sigma": best_gamma0_sigma,
            "best_se": best_se,
            "best_sigma": best_sigma,
            "best_gamma": best_gamma,
            "best_deficit": best_deficit,
            "t": P + Q + 1 + nx,
            "r": Q + 1 + nx,
            "extra_dof": (Q + 1 + nx) * (P + nx),
        }

    # Print results
    total_per_nx = len(sigmas) * len(gammas)
    print(f"\n{'='*85}")
    print(f"  E4 Tension + Penalty — nextra x sigma x gamma Sweep (n={n})")
    print(f"{'='*85}")
    print(f"  Grid: {len(nextra_values)} nextra x {len(sigmas)} sigma x "
          f"{len(gammas)} gamma = {len(nextra_values) * total_per_nx} points")

    print(f"\n  {'nextra':>6s}  {'t':>3s}  {'r':>3s}  {'DOF':>5s}  "
          f"{'g=0 stab_eig':>14s}  {'best sigma':>10s}  {'best gamma':>10s}  "
          f"{'(s,g) stab_eig':>16s}  {'status':>10s}")
    print(f"  {'-'*6}  {'-'*3}  {'-'*3}  {'-'*5}  "
          f"{'-'*14}  {'-'*10}  {'-'*10}  "
          f"{'-'*16}  {'-'*10}")

    for nx in nextra_values:
        if nx not in results:
            continue
        res = results[nx]
        status = "STABLE" if res["best_se"] < STABILITY_TOL else "unstable"
        print(f"  {nx:6d}  {res['t']:3d}  {res['r']:3d}  {res['extra_dof']:5d}  "
              f"{res['gamma0_se']:14.6e}  {res['best_sigma']:10.4f}  "
              f"{res['best_gamma']:10.4f}  "
              f"{res['best_se']:16.6e}  {status:>10s}")

    for nx in nextra_values:
        if nx not in results:
            continue
        res = results[nx]
        print(f"\n  nextra={nx}: gamma=0 sigma*={res['gamma0_sigma']:.4f} "
              f"-> {res['gamma0_se']:.6e}")
        print(f"          : (sigma,gamma)*=({res['best_sigma']:.4f}, {res['best_gamma']:.4f}) "
              f"-> {res['best_se']:.6e}  deficit={res['best_deficit']:.6e}")

    return results


def run_grid_independence(
    nextra_values: list[int],
    grid_sizes: list[int],
) -> dict[int, list[int]]:
    """Phase 3: Grid independence check at sigma=0 (PHS k=2).

    Returns {nextra: [stable grid sizes]}.
    """
    print(f"\n{'='*60}")
    print(f"  E4 PHS k=2 (sigma=0) — Grid Independence")
    print(f"{'='*60}")
    print(f"  {'nextra':>6s}  {'n':>6s}  {'stab_eig':>14s}  {'status':>10s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*14}  {'-'*10}")

    results = {}
    for nx in nextra_values:
        stable_at = []
        for nn in grid_sizes:
            r = Q + 1 + nx
            if nn < 2 * r:
                continue
            se = stability_eigenvalue(
                nn, p=P, q=Q, epsilon=0.0,
                kernel="tension", nu=NU, nextra=nx,
            )
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"  {nx:6d}  {nn:6d}  {se:14.6e}  {status:>10s}")
            if se < STABILITY_TOL:
                stable_at.append(nn)
        results[nx] = stable_at

    return results


def run_footprint_sweep(
    n_sigma: int,
    n_gamma: int,
    sigma_max: float = 50.0,
    nextra_values: list[int] | None = None,
) -> dict:
    """Run all three phases and return summary for known_values.json.

    Returns dict with per-nextra stability info.
    """
    if nextra_values is None:
        nextra_values = [0, 1, 2, 3]
    n = 40  # primary grid size, matching test classes
    grid_sizes = [20, 40, 80, 160]

    # Phase 1: nextra x sigma
    sigma_results = run_nextra_sigma_sweep(n, nextra_values, n_sigma, sigma_max)

    # Phase 2: nextra x sigma x gamma penalty
    penalty_results = run_nextra_penalty_sweep(
        n, nextra_values, n_sigma, n_gamma, sigma_max,
    )

    # Phase 3: grid independence at sigma=0 (PHS k=2)
    grid_results = run_grid_independence(nextra_values, grid_sizes)

    # Build summary
    summary = {}
    for nx in nextra_values:
        key = f"E4_nextra{nx}_phs"
        entry = {"nextra": nx}
        if nx in grid_results:
            entry["stable_at"] = grid_results[nx]
        summary[key] = entry

        # Also record best tension sigma if nextra is in sigma_results
        if nx in sigma_results:
            res = sigma_results[nx]
            if res["best_se"] < STABILITY_TOL and res["best_sigma"] > 0:
                t_key = f"E4_nextra{nx}_tension_{res['best_sigma']:.0f}"
                summary[t_key] = {
                    "nextra": nx,
                    "sigma": round(res["best_sigma"], 4),
                    "stable_at": [n],
                }

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.footprint_sweep",
        description="Stencil footprint (nextra) sweep for E4 tension kernel",
    )
    parser.add_argument(
        "--n-sigma", type=int, default=20,
        help="Number of sigma sample points (default: 20)",
    )
    parser.add_argument(
        "--n-gamma", type=int, default=20,
        help="Number of gamma sample points for penalty phase (default: 20)",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=50.0,
        help="Maximum sigma value (default: 50.0)",
    )
    parser.add_argument(
        "--nextra-values", default="0,1,2,3",
        help="Comma-separated nextra values (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--update-known-values", action="store_true",
        help="Update known_values.json with discovered footprint stability",
    )

    args = parser.parse_args(argv)
    nextra_values = [int(x) for x in args.nextra_values.split(",")]

    summary = run_footprint_sweep(
        args.n_sigma, args.n_gamma, args.sigma_max, nextra_values,
    )

    if args.update_known_values:
        kv = load_known_values()
        if "footprint" not in kv:
            kv["footprint"] = {}
        kv["footprint"] = summary
        save_known_values(kv)
        print(f"\n  Updated known_values.json: footprint")

    return 0


if __name__ == "__main__":
    sys.exit(main())
