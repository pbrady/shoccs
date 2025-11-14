"""
Deep numerical analysis of stencils.

This script performs detailed numerical analysis to verify:
- Coefficient precision
- Symmetry properties
- Sum properties for derivative operators
- Scaling behavior
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from shoccs.stencils import (
    centered_diff_1st_order2,
    centered_diff_2nd_order2,
    centered_diff_1st_order4,
    centered_diff_2nd_order4,
    e2_poly_interior,
    e2_poly_nbs_dirichlet,
    e2_poly_nbs_floating,
)
from cpp_reference import StencilReferenceData


def analyze_interior_stencils():
    """Analyze interior stencil properties."""
    print("=" * 80)
    print("INTERIOR STENCIL ANALYSIS")
    print("=" * 80)

    h = 1.0

    # 2nd order, 1st derivative
    s1 = centered_diff_1st_order2(h)
    print("\n2nd-order 1st derivative (h=1.0):")
    print(f"  Coefficients: {s1}")
    print(f"  Sum: {np.sum(s1):.2e} (should be ~0 for 1st derivative)")
    print(f"  Anti-symmetric: {abs(s1[0] + s1[2]):.2e} (should be ~0)")
    print(f"  Center coeff:   {abs(s1[1]):.2e} (should be ~0)")

    # 2nd order, 2nd derivative
    s2 = centered_diff_2nd_order2(h)
    print("\n2nd-order 2nd derivative (h=1.0):")
    print(f"  Coefficients: {s2}")
    print(f"  Sum: {np.sum(s2):.2e} (should be ~0 for 2nd derivative)")
    print(f"  Symmetric: {abs(s2[0] - s2[2]):.2e} (should be ~0)")

    # 4th order, 1st derivative
    s3 = centered_diff_1st_order4(h)
    print("\n4th-order 1st derivative (h=1.0):")
    print(f"  Coefficients: {s3}")
    print(f"  Sum: {np.sum(s3):.2e} (should be ~0)")
    print(f"  Anti-symmetric pairs:")
    print(f"    c[-2] + c[2]: {abs(s3[0] + s3[4]):.2e}")
    print(f"    c[-1] + c[1]: {abs(s3[1] + s3[3]):.2e}")

    # 4th order, 2nd derivative
    s4 = centered_diff_2nd_order4(h)
    print("\n4th-order 2nd derivative (h=1.0):")
    print(f"  Coefficients: {s4}")
    print(f"  Sum: {np.sum(s4):.2e} (should be ~0)")
    print(f"  Symmetric pairs:")
    print(f"    c[-2] - c[2]: {abs(s4[0] - s4[4]):.2e}")
    print(f"    c[-1] - c[1]: {abs(s4[1] - s4[3]):.2e}")


def analyze_scaling():
    """Analyze how stencils scale with h."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    h_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    print("\n1st derivative stencils should scale as 1/h:")
    for h in h_values:
        s = centered_diff_1st_order2(h)
        expected_scale = 0.5 / h
        actual = s[2]
        relative_error = abs(actual - expected_scale) / expected_scale
        print(f"  h={h:8.3f}: c[1]={actual:12.6e}, expected={expected_scale:12.6e}, "
              f"rel_err={relative_error:.2e}")

    print("\n2nd derivative stencils should scale as 1/h²:")
    for h in h_values:
        s = centered_diff_2nd_order2(h)
        expected_scale = 1.0 / (h * h)
        actual = s[0]
        relative_error = abs(actual - expected_scale) / expected_scale
        print(f"  h={h:8.3f}: c[-1]={actual:12.6e}, expected={expected_scale:12.6e}, "
              f"rel_err={relative_error:.2e}")


def analyze_cpp_comparison():
    """Compare with C++ reference data."""
    print("\n" + "=" * 80)
    print("C++ REFERENCE COMPARISON")
    print("=" * 80)

    ref_data = StencilReferenceData()

    # Interior stencil
    h = 1.0
    py_interior = e2_poly_interior(h)
    cpp_interior = ref_data.get_interior("polyE2_1", h)

    print(f"\npolyE2_1 interior (h={h}):")
    print(f"  Python: {py_interior}")
    print(f"  C++:    {cpp_interior}")
    print(f"  Diff:   {py_interior - cpp_interior}")
    print(f"  Max abs diff: {np.max(np.abs(py_interior - cpp_interior)):.2e}")
    print(f"  Max rel diff: {np.max(np.abs((py_interior - cpp_interior) / cpp_interior)):.2e}")

    # Dirichlet boundary
    psi = 0.001
    da = ref_data.get_alpha("polyE2_1", "dirichlet_alpha")
    py_dirichlet = e2_poly_nbs_dirichlet(h, psi, da, right=False)
    cpp_dirichlet, r, t = ref_data.get_boundary("polyE2_1", "dirichlet", h, psi, False)

    print(f"\npolyE2_1 Dirichlet NBS (h={h}, psi={psi}):")
    print(f"  Python shape: {py_dirichlet.shape}")
    print(f"  C++ shape:    {cpp_dirichlet.shape}")
    print(f"  Python: {py_dirichlet}")
    print(f"  C++:    {cpp_dirichlet}")
    print(f"  Diff:   {py_dirichlet - cpp_dirichlet}")
    print(f"  Max abs diff: {np.max(np.abs(py_dirichlet - cpp_dirichlet)):.2e}")
    nonzero_mask = cpp_dirichlet != 0
    if np.any(nonzero_mask):
        rel_diff = np.abs((py_dirichlet - cpp_dirichlet) / cpp_dirichlet)[nonzero_mask]
        print(f"  Max rel diff: {np.max(rel_diff):.2e}")


def analyze_boundary_stencils():
    """Analyze boundary stencil properties across parameter ranges."""
    print("\n" + "=" * 80)
    print("BOUNDARY STENCIL ANALYSIS")
    print("=" * 80)

    h = 1.0
    da = np.array([0.12, 0.13, 0.14])
    fa = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    print("\nDirichlet NBS at different psi values:")
    psi_values = [0.001, 0.01, 0.1, 0.5, 1.0]
    for psi in psi_values:
        s = e2_poly_nbs_dirichlet(h, psi, da, right=False)
        print(f"  psi={psi:6.3f}: max_coeff={np.max(np.abs(s)):10.6f}, "
              f"sum={np.sum(s):10.6e}, finite={np.all(np.isfinite(s))}")

    print("\nFloating NBS at different psi values:")
    for psi in psi_values:
        s = e2_poly_nbs_floating(h, psi, fa, right=False)
        print(f"  psi={psi:6.3f}: max_coeff={np.max(np.abs(s)):10.6f}, "
              f"sum={np.sum(s):10.6e}, finite={np.all(np.isfinite(s))}")

    print("\nLeft vs Right boundary consistency:")
    psi = 0.5
    left = e2_poly_nbs_dirichlet(h, psi, da, right=False)
    right = e2_poly_nbs_dirichlet(h, psi, da, right=True)
    print(f"  Left sum:  {np.sum(left):10.6e}")
    print(f"  Right sum: {np.sum(right):10.6e}")
    print(f"  |Left| == |Right|: {np.allclose(np.abs(left), np.abs(right[::-1]))}")


def check_numerical_precision():
    """Check for numerical precision issues."""
    print("\n" + "=" * 80)
    print("NUMERICAL PRECISION CHECK")
    print("=" * 80)

    h_extreme = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e2, 1e4, 1e6]

    print("\nExtreme h values (checking for overflow/underflow):")
    for h in h_extreme:
        try:
            s = centered_diff_1st_order2(h)
            finite = np.all(np.isfinite(s))
            max_val = np.max(np.abs(s))
            print(f"  h={h:10.2e}: finite={finite}, max_coeff={max_val:10.2e}")
        except Exception as e:
            print(f"  h={h:10.2e}: ERROR - {e}")

    print("\nExtreme psi values (checking for singularities):")
    psi_extreme = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.9, 0.99, 0.999, 1.0]
    da = np.array([0.12, 0.13, 0.14])
    h = 1.0

    for psi in psi_extreme:
        try:
            s = e2_poly_nbs_dirichlet(h, psi, da, right=False)
            finite = np.all(np.isfinite(s))
            max_val = np.max(np.abs(s))
            min_nonzero = np.min(np.abs(s[s != 0])) if np.any(s != 0) else 0.0
            condition = max_val / min_nonzero if min_nonzero > 0 else 0.0
            print(f"  psi={psi:10.2e}: finite={finite}, max={max_val:10.2e}, "
                  f"condition={condition:10.2e}")
        except Exception as e:
            print(f"  psi={psi:10.2e}: ERROR - {e}")


if __name__ == "__main__":
    analyze_interior_stencils()
    analyze_scaling()
    analyze_cpp_comparison()
    analyze_boundary_stencils()
    check_numerical_precision()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
