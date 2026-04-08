"""Shared helpers for sweep scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KNOWN_VALUES_PATH = Path(__file__).parent / "known_values.json"


@dataclass
class SweepResult:
    """Result of a single parameter sweep point."""

    parameter: float
    eigenvalue: float
    stable: bool
    n: int | None = None
    label: str = ""
    extra: dict = field(default_factory=dict)


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    col_widths: list[int] | None = None,
) -> None:
    """Print a formatted table to stdout."""
    if col_widths is None:
        col_widths = [
            max(len(h), max((len(str(r)) for r in col), default=0)) + 2
            for h, col in zip(headers, zip(*rows))
        ]
        # Ensure header widths are respected
        col_widths = [max(w, len(h) + 2) for w, h in zip(col_widths, headers)]

    print(f"\n{'=' * sum(col_widths)}")
    print(f"  {title}")
    print(f"{'=' * sum(col_widths)}")
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    print()


def load_known_values() -> dict:
    """Load known optimal values from JSON."""
    if not KNOWN_VALUES_PATH.exists():
        return {}
    with open(KNOWN_VALUES_PATH) as f:
        return json.load(f)


def save_known_values(data: dict) -> None:
    """Save known optimal values to JSON."""
    with open(KNOWN_VALUES_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def dense_sweep_min(f, params):
    """Evaluate f at each param value, return (best_param, best_val, all_results).

    Finds the parameter that minimizes f(param).
    """
    results = [(p, f(p)) for p in params]
    best_param, best_val = min(results, key=lambda x: x[1])
    return best_param, best_val, results


def bisect_threshold(f, a, b, threshold, *, tol=1e-4, maxiter=60):
    """Bisect to find x where f(x) crosses threshold from above.

    Assumes f(a) > threshold and f(b) < threshold.
    Returns x such that f(x) is near threshold.
    """
    for _ in range(maxiter):
        mid = (a + b) / 2
        if (b - a) < tol:
            break
        if f(mid) > threshold:
            a = mid
        else:
            b = mid
    return (a + b) / 2
