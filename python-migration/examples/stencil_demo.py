"""
Demonstration of SHOCCS finite difference stencils.

This script shows how to use the stencil module for computing derivatives.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shoccs.stencils import (
    centered_diff_1st_order2,
    centered_diff_2nd_order2,
    centered_diff_1st_order4,
    centered_diff_2nd_order4,
    apply_stencil_1d,
    make_e2_poly_stencil,
    e2_poly_interior,
)


def demo_basic_stencils():
    """Demonstrate basic centered difference stencils."""
    print("=" * 70)
    print("BASIC CENTERED DIFFERENCE STENCILS")
    print("=" * 70)

    # Create a grid
    h = 0.1
    x = np.arange(0, 2.0, h)

    # Define a test function: f(x) = x^3
    f = x ** 3
    df_exact = 3 * x ** 2
    d2f_exact = 6 * x

    print(f"\nGrid spacing: h = {h}")
    print(f"Grid points: {len(x)}")
    print(f"Test function: f(x) = x³")

    # Test 2nd-order 1st derivative
    print("\n1. Second-order first derivative (3-point stencil)")
    print("-" * 50)
    stencil = centered_diff_1st_order2(h)
    print(f"Stencil coefficients: {stencil}")

    # Apply at a point in the middle
    i = len(x) // 2
    df_approx = apply_stencil_1d(f, stencil, i)
    df_true = df_exact[i]
    error = abs(df_approx - df_true)

    print(f"\nAt x = {x[i]:.2f}:")
    print(f"  Exact derivative: {df_true:.6f}")
    print(f"  Approx derivative: {df_approx:.6f}")
    print(f"  Error: {error:.2e}")

    # Test 2nd-order 2nd derivative
    print("\n2. Second-order second derivative (3-point stencil)")
    print("-" * 50)
    stencil = centered_diff_2nd_order2(h)
    print(f"Stencil coefficients: {stencil}")

    d2f_approx = apply_stencil_1d(f, stencil, i)
    d2f_true = d2f_exact[i]
    error = abs(d2f_approx - d2f_true)

    print(f"\nAt x = {x[i]:.2f}:")
    print(f"  Exact 2nd derivative: {d2f_true:.6f}")
    print(f"  Approx 2nd derivative: {d2f_approx:.6f}")
    print(f"  Error: {error:.2e}")

    # Test 4th-order 1st derivative
    print("\n3. Fourth-order first derivative (5-point stencil)")
    print("-" * 50)
    stencil = centered_diff_1st_order4(h)
    print(f"Stencil coefficients: {stencil}")

    df_approx = apply_stencil_1d(f, stencil, i)
    error = abs(df_approx - df_true)

    print(f"\nAt x = {x[i]:.2f}:")
    print(f"  Exact derivative: {df_true:.6f}")
    print(f"  Approx derivative: {df_approx:.6f}")
    print(f"  Error: {error:.2e}")
    print("  (Note: Much smaller error than 2nd-order!)")


def demo_convergence():
    """Demonstrate convergence rates of different stencils."""
    print("\n" + "=" * 70)
    print("CONVERGENCE RATE DEMONSTRATION")
    print("=" * 70)

    # Test function
    def f(x):
        return np.sin(2 * np.pi * x)

    def df(x):
        return 2 * np.pi * np.cos(2 * np.pi * x)

    grid_sizes = [0.1, 0.05, 0.025, 0.0125]

    print("\nComparing 2nd-order vs 4th-order stencils")
    print("Test function: f(x) = sin(2πx)")
    print("\n{:>10s}  {:>15s}  {:>15s}".format("h", "Error (2nd)", "Error (4th)"))
    print("-" * 50)

    for h in grid_sizes:
        x = np.arange(0, 1.0, h)
        y = f(x)
        dy_exact = df(x)

        # 2nd order
        stencil_2 = centered_diff_1st_order2(h)
        errors_2 = []
        for i in range(1, len(x) - 1):
            dy_approx = apply_stencil_1d(y, stencil_2, i)
            errors_2.append(abs(dy_approx - dy_exact[i]))
        max_error_2 = max(errors_2)

        # 4th order
        stencil_4 = centered_diff_1st_order4(h)
        errors_4 = []
        for i in range(2, len(x) - 2):
            dy_approx = apply_stencil_1d(y, stencil_4, i)
            errors_4.append(abs(dy_approx - dy_exact[i]))
        max_error_4 = max(errors_4)

        print(f"{h:>10.4f}  {max_error_2:>15.2e}  {max_error_4:>15.2e}")

    print("\nNote: 4th-order converges much faster as h → 0")


def demo_e2_poly():
    """Demonstrate E2-Poly stencils."""
    print("\n" + "=" * 70)
    print("E2-POLY STENCILS")
    print("=" * 70)

    print("\n1. E2-Poly Interior Stencil")
    print("-" * 50)
    h = 0.1
    stencil = e2_poly_interior(h)
    print(f"Grid spacing: h = {h}")
    print(f"Interior stencil: {stencil}")
    print("(Same as centered_diff_1st_order2)")

    print("\n2. E2-Poly Stencil Configuration")
    print("-" * 50)
    fa = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    da = np.array([0.5, 1.5, 2.5])
    ia = np.array([0.1, 0.2, 0.3, 0.4])

    config = make_e2_poly_stencil(fa, da, ia)
    print(f"Floating BC params (fa): {config.fa}")
    print(f"Dirichlet BC params (da): {config.da}")
    print(f"Interpolation params (ia): {config.ia}")


def demo_polynomial_exactness():
    """Demonstrate that stencils reproduce polynomials exactly."""
    print("\n" + "=" * 70)
    print("POLYNOMIAL EXACTNESS")
    print("=" * 70)

    h = 0.1
    x = np.arange(-1, 1 + h, h)

    print("\nTesting that stencils reproduce polynomials exactly:")
    print("(up to machine precision)")

    # Test linear: f(x) = 3x + 2, f'(x) = 3
    print("\n1. Linear function: f(x) = 3x + 2")
    print("-" * 50)
    f = 3 * x + 2
    df_exact = 3.0

    stencil = centered_diff_1st_order2(h)
    i = len(x) // 2
    df_approx = apply_stencil_1d(f, stencil, i)
    error = abs(df_approx - df_exact)

    print(f"Exact derivative: {df_exact}")
    print(f"Approximation: {df_approx:.15f}")
    print(f"Error: {error:.2e}")
    print("✓ Reproduced exactly!" if error < 1e-12 else "✗ Error too large")

    # Test quadratic: f(x) = 2x², f''(x) = 4
    print("\n2. Quadratic function: f(x) = 2x²")
    print("-" * 50)
    f = 2 * x ** 2
    d2f_exact = 4.0

    stencil = centered_diff_2nd_order2(h)
    d2f_approx = apply_stencil_1d(f, stencil, i)
    error = abs(d2f_approx - d2f_exact)

    print(f"Exact 2nd derivative: {d2f_exact}")
    print(f"Approximation: {d2f_approx:.15f}")
    print(f"Error: {error:.2e}")
    print("✓ Reproduced exactly!" if error < 1e-12 else "✗ Error too large")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "SHOCCS STENCIL DEMONSTRATION" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")

    demo_basic_stencils()
    demo_convergence()
    demo_polynomial_exactness()
    demo_e2_poly()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nAll stencils are JIT-compiled with Numba for high performance.")
    print("See tests/test_stencils.py for comprehensive validation.\n")


if __name__ == "__main__":
    main()
