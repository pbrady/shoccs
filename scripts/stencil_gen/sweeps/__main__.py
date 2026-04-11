"""CLI entry point for sweep scripts.

Usage:
    uv run python -m sweeps epsilon --scheme E2
    uv run python -m sweeps tension --scheme E4
    uv run python -m sweeps all --quick
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps",
        description="Parameter-space exploration sweeps for PHS/RBF stencils",
    )
    subparsers = parser.add_subparsers(dest="command", help="sweep type")

    # Placeholder subcommands — implementations added as sweep scripts land
    sub_eps = subparsers.add_parser("epsilon", help="Epsilon (Gaussian/MQ) sweep")
    sub_eps.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_eps.add_argument("--kernel", choices=["gaussian", "multiquadric"], default="gaussian")
    sub_eps.add_argument("--n-values", default="20,40,80", help="Comma-separated grid sizes")
    sub_eps.add_argument("--n-eps", type=int, default=60, help="Number of epsilon sample points")
    sub_eps.add_argument("--update-known-values", action="store_true", help="Update known_values.json with discovered optimal epsilon")
    sub_eps.add_argument("--include-gv", action="store_true", help="Also compute boundary group-velocity error at each epsilon (advisory)")

    sub_tension = subparsers.add_parser("tension", help="Tension spline sigma sweep")
    sub_tension.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_tension.add_argument("--n-values", default="20,40,80", help="Comma-separated grid sizes")
    sub_tension.add_argument("--n-sigma", type=int, default=61, help="Number of sigma sample points")
    sub_tension.add_argument("--sigma-max", type=float, default=20.0)
    sub_tension.add_argument("--update-known-values", action="store_true", help="Update known_values.json with discovered optimal sigma")
    sub_tension.add_argument("--include-gv", action="store_true", help="Also compute boundary group-velocity error at each sigma (advisory)")

    sub_tension_pen = subparsers.add_parser("tension-penalty", help="Tension + conservation penalty sweep")
    sub_tension_pen.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_tension_pen.add_argument("--n-sigma", type=int, default=25)
    sub_tension_pen.add_argument("--n-gamma", type=int, default=25)
    sub_tension_pen.add_argument("--sigma-max", type=float, default=20.0)
    sub_tension_pen.add_argument("--update-known-values", action="store_true", help="Update known_values.json")

    sub_footprint = subparsers.add_parser("footprint", help="Stencil footprint (nextra) sweep")
    sub_footprint.add_argument("--n-sigma", type=int, default=20)
    sub_footprint.add_argument("--n-gamma", type=int, default=20)
    sub_footprint.add_argument("--sigma-max", type=float, default=50.0)
    sub_footprint.add_argument("--nextra-values", default="0,1,2,3", help="Comma-separated nextra values")
    sub_footprint.add_argument("--update-known-values", action="store_true", help="Update known_values.json")
    sub_footprint.add_argument("--include-gv", action="store_true", help="Also compute boundary group-velocity error per (nextra, sigma) (advisory)")

    sub_comparison = subparsers.add_parser("comparison", help="Multi-method comparison table")
    sub_comparison.add_argument("--scheme", choices=["E2", "E4"], default=None)
    sub_comparison.add_argument("--n-values", default="20,40,80", help="Comma-separated grid sizes")
    sub_comparison.add_argument("--update-known-values", action="store_true", help="Update known_values.json")

    sub_alpha = subparsers.add_parser("alpha", help="Boundary alpha extraction at optimal epsilon")
    sub_alpha.add_argument("--scheme", choices=["E2", "E4"], required=True)

    sub_mixed = subparsers.add_parser("mixed-epsilon", help="Mixed (per-row) epsilon sweep")
    sub_mixed.add_argument("--scheme", choices=["E2", "E4"], default="E4")
    sub_mixed.add_argument("--kernel", choices=["gaussian", "multiquadric"], default="gaussian")
    sub_mixed.add_argument("--n-eps", type=int, default=20)
    sub_mixed.add_argument("--update-known-values", action="store_true", help="Update known_values.json")

    sub_pareto = subparsers.add_parser(
        "gv-stability-pareto",
        help="GV-vs-stability Pareto sweep (research / documentation aid)",
    )
    sub_pareto.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_pareto.add_argument(
        "--param", choices=["tension", "gaussian", "multiquadric"], required=True,
        help="Kernel to sweep (parameter is sigma for tension, epsilon otherwise)",
    )
    sub_pareto.add_argument("--n-points", type=int, default=61, help="Number of sample points on the parameter grid")
    sub_pareto.add_argument("--n", type=int, default=40, help="Grid size for the stability eigenvalue")
    sub_pareto.add_argument("--param-max", type=float, default=20.0, help="Upper end of the parameter grid")

    sub_all = subparsers.add_parser("all", help="Run all sweeps")
    sub_all.add_argument("--quick", action="store_true", help="Reduced resolution for quick verification")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to sweep modules (imported lazily to avoid loading numpy/sympy at parse time)
    if args.command == "epsilon":
        from .epsilon_sweep import main as eps_main

        return eps_main([
            "--scheme", args.scheme,
            "--kernel", args.kernel,
            "--n-values", args.n_values,
            "--n-eps", str(args.n_eps),
            *(["--update-known-values"] if args.update_known_values else []),
            *(["--include-gv"] if args.include_gv else []),
        ])

    if args.command == "tension":
        from .tension_sweep import main as tension_main

        return tension_main([
            "--scheme", args.scheme,
            "--n-values", args.n_values,
            "--n-sigma", str(args.n_sigma),
            "--sigma-max", str(args.sigma_max),
            *(["--update-known-values"] if args.update_known_values else []),
            *(["--include-gv"] if args.include_gv else []),
        ])

    if args.command == "mixed-epsilon":
        from .mixed_epsilon_sweep import main as mixed_main

        return mixed_main([
            "--scheme", args.scheme,
            "--kernel", args.kernel,
            "--n-eps", str(args.n_eps),
            *(["--update-known-values"] if args.update_known_values else []),
        ])

    if args.command == "tension-penalty":
        from .tension_penalty_sweep import main as tp_main

        return tp_main([
            "--scheme", args.scheme,
            "--n-sigma", str(args.n_sigma),
            "--n-gamma", str(args.n_gamma),
            "--sigma-max", str(args.sigma_max),
            *(["--update-known-values"] if args.update_known_values else []),
        ])

    if args.command == "footprint":
        from .footprint_sweep import main as fp_main

        return fp_main([
            "--n-sigma", str(args.n_sigma),
            "--n-gamma", str(args.n_gamma),
            "--sigma-max", str(args.sigma_max),
            "--nextra-values", args.nextra_values,
            *(["--update-known-values"] if args.update_known_values else []),
            *(["--include-gv"] if args.include_gv else []),
        ])

    if args.command == "comparison":
        from .comparison import main as comp_main

        return comp_main([
            *(["--scheme", args.scheme] if args.scheme else []),
            "--n-values", args.n_values,
            *(["--update-known-values"] if args.update_known_values else []),
        ])

    if args.command == "alpha":
        from .alpha_extraction import main as alpha_main

        return alpha_main(["--scheme", args.scheme])

    if args.command == "gv-stability-pareto":
        from .gv_stability_pareto import main as pareto_main

        return pareto_main([
            "--scheme", args.scheme,
            "--param", args.param,
            "--n-points", str(args.n_points),
            "--n", str(args.n),
            "--param-max", str(args.param_max),
        ])

    if args.command == "all":
        return _run_all(quick=args.quick)

    print(f"sweeps: command '{args.command}' not recognized")
    return 1


def _run_all(*, quick: bool) -> int:
    """Run all sweeps sequentially. --quick reduces resolution for fast verification."""
    from .alpha_extraction import main as alpha_main
    from .comparison import main as comp_main
    from .epsilon_sweep import main as eps_main
    from .footprint_sweep import main as fp_main
    from .gv_stability_pareto import main as pareto_main
    from .mixed_epsilon_sweep import main as mixed_main
    from .tension_penalty_sweep import main as tp_main
    from .tension_sweep import main as tension_main

    quick_n_eps = "10" if quick else "60"
    quick_n_sigma = "10" if quick else "61"
    quick_n_gamma = "5" if quick else "25"
    quick_n_values = "20,40" if quick else "20,40,80"
    quick_mixed_n_eps = "5" if quick else "20"
    quick_fp_n_sigma = "10" if quick else "20"
    quick_fp_n_gamma = "10" if quick else "20"
    quick_tp_n_sigma = "5" if quick else "25"
    quick_pareto_n_points = "10" if quick else "61"

    sweeps: list[tuple[str, Callable[[list[str]], int], list[str]]] = [
        ("Epsilon sweep E2 (gaussian)", eps_main,
         ["--scheme", "E2", "--kernel", "gaussian", "--n-eps", quick_n_eps, "--n-values", quick_n_values]),
        ("Epsilon sweep E4 (gaussian)", eps_main,
         ["--scheme", "E4", "--kernel", "gaussian", "--n-eps", quick_n_eps, "--n-values", quick_n_values]),
        ("Epsilon sweep E2 (multiquadric)", eps_main,
         ["--scheme", "E2", "--kernel", "multiquadric", "--n-eps", quick_n_eps, "--n-values", quick_n_values]),
        ("Epsilon sweep E4 (multiquadric)", eps_main,
         ["--scheme", "E4", "--kernel", "multiquadric", "--n-eps", quick_n_eps, "--n-values", quick_n_values]),
        ("Mixed epsilon sweep E4", mixed_main,
         ["--scheme", "E4", "--n-eps", quick_mixed_n_eps]),
        ("Tension sweep E2", tension_main,
         ["--scheme", "E2", "--n-sigma", quick_n_sigma, "--n-values", quick_n_values]),
        ("Tension sweep E4", tension_main,
         ["--scheme", "E4", "--n-sigma", quick_n_sigma, "--n-values", quick_n_values]),
        ("Tension-penalty sweep E2", tp_main,
         ["--scheme", "E2", "--n-sigma", quick_tp_n_sigma, "--n-gamma", quick_n_gamma]),
        ("Tension-penalty sweep E4", tp_main,
         ["--scheme", "E4", "--n-sigma", quick_tp_n_sigma, "--n-gamma", quick_n_gamma]),
        ("Footprint sweep", fp_main,
         ["--n-sigma", quick_fp_n_sigma, "--n-gamma", quick_fp_n_gamma]),
        ("Comparison (all schemes)", comp_main,
         ["--n-values", quick_n_values]),
        ("GV-stability Pareto E2 (tension)", pareto_main,
         ["--scheme", "E2", "--param", "tension", "--n-points", quick_pareto_n_points]),
        ("Alpha extraction E2", alpha_main, ["--scheme", "E2"]),
    ]

    failures: list[str] = []
    for label, fn, argv in sweeps:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}\n")
        try:
            rc = fn(argv)
            if rc != 0:
                print(f"\n*** {label} returned exit code {rc}")
                failures.append(label)
        except Exception as exc:
            print(f"\n*** {label} failed: {exc}")
            failures.append(label)

    print(f"\n{'=' * 60}")
    if failures:
        print(f"  {len(failures)} sweep(s) failed:")
        for f in failures:
            print(f"    - {f}")
        print(f"{'=' * 60}")
        return 1
    print(f"  All {len(sweeps)} sweeps completed successfully")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
