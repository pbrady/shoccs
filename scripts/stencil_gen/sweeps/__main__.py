"""CLI entry point for sweep scripts.

Usage:
    uv run python -m sweeps epsilon --scheme E2
    uv run python -m sweeps tension --scheme E4
    uv run python -m sweeps all --quick
"""

from __future__ import annotations

import argparse
import sys


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

    sub_tension = subparsers.add_parser("tension", help="Tension spline sigma sweep")
    sub_tension.add_argument("--scheme", choices=["E2", "E4"], required=True)
    sub_tension.add_argument("--n-values", default="20,40,80", help="Comma-separated grid sizes")
    sub_tension.add_argument("--n-sigma", type=int, default=61, help="Number of sigma sample points")
    sub_tension.add_argument("--sigma-max", type=float, default=20.0)
    sub_tension.add_argument("--update-known-values", action="store_true", help="Update known_values.json with discovered optimal sigma")

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

    sub_comparison = subparsers.add_parser("comparison", help="Multi-method comparison table")
    sub_comparison.add_argument("--scheme", choices=["E2", "E4"], default=None)

    sub_alpha = subparsers.add_parser("alpha", help="Boundary alpha extraction at optimal epsilon")
    sub_alpha.add_argument("--scheme", choices=["E2", "E4"], required=True)

    sub_mixed = subparsers.add_parser("mixed-epsilon", help="Mixed (per-row) epsilon sweep")
    sub_mixed.add_argument("--scheme", choices=["E2", "E4"], default="E4")
    sub_mixed.add_argument("--kernel", choices=["gaussian", "multiquadric"], default="gaussian")
    sub_mixed.add_argument("--n-eps", type=int, default=20)
    sub_mixed.add_argument("--update-known-values", action="store_true", help="Update known_values.json")

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
        ])

    if args.command == "tension":
        from .tension_sweep import main as tension_main

        return tension_main([
            "--scheme", args.scheme,
            "--n-values", args.n_values,
            "--n-sigma", str(args.n_sigma),
            "--sigma-max", str(args.sigma_max),
            *(["--update-known-values"] if args.update_known_values else []),
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
        ])

    print(f"sweeps: command '{args.command}' recognized (implementation pending)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
