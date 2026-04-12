"""CLI entry point for Brady-Livescu 2D stability scoring.

Usage:
    uv run python -m stencil_gen.brady2d_cli --scheme E4 --kernel tension --sigma 3.0 --max-layer 3
    uv run python -m stencil_gen.brady2d_cli --scheme E4 --kernel classical --alpha 0.3487 --max-layer 6
    uv run python -m stencil_gen.brady2d_cli --scheme E4 --kernel tension --sigma 3.0 --max-layer 7 --json-output result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np


def _json_serializer(obj):
    """Custom JSON serializer for numpy types and dataclass fields."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _build_params(args: argparse.Namespace) -> dict:
    """Build the params dict from CLI arguments."""
    params: dict = {}
    if args.sigma is not None:
        params["sigma"] = args.sigma
    if args.epsilon is not None:
        params["epsilon"] = args.epsilon
    if args.alpha is not None:
        params["alpha"] = [float(a) for a in args.alpha.split(",")]
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="brady2d_cli",
        description="Brady-Livescu 2D analytical stability scoring pipeline",
    )
    parser.add_argument(
        "--scheme",
        choices=["E2", "E4"],
        required=True,
        help="Scheme name",
    )
    parser.add_argument(
        "--kernel",
        choices=["classical", "tension", "gaussian", "multiquadric", "phs"],
        required=True,
        help="Kernel type",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Tension/PHS sigma parameter",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Gaussian/multiquadric epsilon parameter",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        default=None,
        help="Classical alpha values (comma-separated, e.g. '0.3487' or '0.1,0.2')",
    )
    parser.add_argument(
        "--max-layer",
        type=int,
        default=7,
        help="Highest layer to run (1-7, default: 7)",
    )
    parser.add_argument(
        "--short-circuit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop at first failing layer (default: --short-circuit)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to write JSON output",
    )

    args = parser.parse_args(argv)
    params = _build_params(args)

    # Validate that at least one parameter is provided for non-classical kernels
    if args.kernel == "classical" and args.alpha is None:
        print("Error: --alpha is required for classical kernel", file=sys.stderr)
        return 1
    if args.kernel in ("tension", "phs") and args.sigma is None:
        print(
            f"Error: --sigma is required for {args.kernel} kernel",
            file=sys.stderr,
        )
        return 1
    if args.kernel in ("gaussian", "multiquadric") and args.epsilon is None:
        print(
            f"Error: --epsilon is required for {args.kernel} kernel",
            file=sys.stderr,
        )
        return 1

    # Lazy import to avoid loading numpy/scipy at parse time
    from stencil_gen.brady2d_stability import brady2d_stability_score

    report = brady2d_stability_score(
        scheme=args.scheme,
        kernel=args.kernel,
        params=params,
        max_layer=args.max_layer,
        short_circuit=args.short_circuit,
    )

    print(report)

    if args.json_output:
        result = asdict(report)
        with open(args.json_output, "w") as f:
            json.dump(result, f, indent=2, default=_json_serializer)
        print(f"\nJSON output written to: {args.json_output}")

    return 0 if report.overall_verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
