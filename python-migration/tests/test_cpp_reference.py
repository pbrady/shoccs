"""
Test Python stencils against C++ reference data.

This test suite validates that the Python implementations produce
identical results to the C++ reference implementation.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src and tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from shoccs.stencils import (
    e2_poly_interior,
    e2_poly_nbs_dirichlet,
    e2_poly_nbs_floating,
    make_e2_poly_stencil,
)
from cpp_reference import StencilReferenceData


class TestPolyE2_1ReferenceComparison:
    """Compare polyE2_1 Python implementation against C++ reference."""

    @classmethod
    def setup_class(cls):
        """Load reference data once for all tests."""
        cls.ref_data = StencilReferenceData()

    def test_interior_h_1_0(self):
        """Test interior stencil with h=1.0 against C++ reference."""
        h = 1.0

        # Get Python result
        python_coeffs = e2_poly_interior(h)

        # Get C++ reference
        cpp_coeffs = self.ref_data.get_interior("polyE2_1", h)

        # Compare with tight tolerance (machine precision)
        np.testing.assert_allclose(
            python_coeffs, cpp_coeffs,
            rtol=1e-14, atol=1e-15,
            err_msg="Interior stencil coefficients do not match C++ reference"
        )

    def test_dirichlet_nbs_h_1_0_psi_0_001_left(self):
        """Test Dirichlet NBS with h=1.0, psi=0.001, left boundary."""
        h = 1.0
        psi = 0.001
        ray_outside = False  # left boundary

        # Get alpha parameters from reference
        da = self.ref_data.get_alpha("polyE2_1", "dirichlet_alpha")

        # Get Python result
        python_coeffs = e2_poly_nbs_dirichlet(h, psi, da, right=False)

        # Get C++ reference
        cpp_coeffs, r, t = self.ref_data.get_boundary(
            "polyE2_1", "dirichlet", h, psi, ray_outside
        )

        # Compare with tight tolerance
        np.testing.assert_allclose(
            python_coeffs, cpp_coeffs,
            rtol=1e-14, atol=1e-15,
            err_msg=f"Dirichlet NBS coefficients do not match C++ reference\n"
                    f"Python: {python_coeffs}\n"
                    f"C++:    {cpp_coeffs}\n"
                    f"Diff:   {python_coeffs - cpp_coeffs}"
        )

        # Verify shape
        assert len(python_coeffs) == r * t, f"Expected {r*t} coefficients, got {len(python_coeffs)}"

    def test_coefficient_sum_properties(self):
        """Test that stencil coefficients have correct sum properties."""
        h = 1.0

        # For 1st derivative interior stencil, sum should be ~0
        interior = e2_poly_interior(h)
        assert abs(np.sum(interior)) < 1e-14, "1st derivative stencil should sum to ~0"

    def test_symmetry_properties(self):
        """Test that interior stencil has expected symmetry."""
        h = 1.0
        stencil = e2_poly_interior(h)

        # For centered difference: c[-1] = -c[1], c[0] = 0
        assert abs(stencil[0] + stencil[2]) < 1e-14, "Stencil should be anti-symmetric"
        assert abs(stencil[1]) < 1e-14, "Center coefficient should be zero"


class TestNumericalStability:
    """Test numerical stability across parameter ranges."""

    def test_small_h_stability(self):
        """Test stencil computation with very small h (fine grids)."""
        h_values = [1e-2, 1e-4, 1e-6]

        for h in h_values:
            stencil = e2_poly_interior(h)

            # Stencil should not have NaN or Inf
            assert np.all(np.isfinite(stencil)), f"Non-finite values for h={h}"

            # Coefficients should scale properly with h
            # For 1st derivative: coefficients ~ 1/h
            assert np.max(np.abs(stencil)) > 0.1 / h, f"Coefficients too small for h={h}"

    def test_large_h_stability(self):
        """Test stencil computation with large h (coarse grids)."""
        h_values = [1.0, 10.0, 100.0]

        for h in h_values:
            stencil = e2_poly_interior(h)

            # Stencil should not have NaN or Inf
            assert np.all(np.isfinite(stencil)), f"Non-finite values for h={h}"

            # Verify proper scaling
            expected_magnitude = 0.5 / h
            assert abs(stencil[2] - expected_magnitude) < 1e-14 * expected_magnitude

    def test_psi_near_zero(self):
        """Test boundary stencil with psi very close to 0 (cut-cell near solid)."""
        h = 1.0
        psi_values = [1e-10, 1e-6, 1e-3, 0.001]
        da = np.array([0.12, 0.13, 0.14])

        for psi in psi_values:
            stencil = e2_poly_nbs_dirichlet(h, psi, da, right=False)

            # Should not have NaN or Inf
            assert np.all(np.isfinite(stencil)), f"Non-finite values for psi={psi}"

            # Should have reasonable magnitudes
            assert np.max(np.abs(stencil)) < 1e6, f"Coefficients too large for psi={psi}"

    def test_psi_near_one(self):
        """Test boundary stencil with psi close to 1 (cut-cell near fluid)."""
        h = 1.0
        psi_values = [0.9, 0.99, 0.999, 1.0]
        da = np.array([0.12, 0.13, 0.14])

        for psi in psi_values:
            stencil = e2_poly_nbs_dirichlet(h, psi, da, right=False)

            # Should not have NaN or Inf
            assert np.all(np.isfinite(stencil)), f"Non-finite values for psi={psi}"

            # Should have reasonable magnitudes
            assert np.max(np.abs(stencil)) < 1e6, f"Coefficients too large for psi={psi}"


class TestPolynomialReproduction:
    """Test that stencils reproduce polynomials to machine precision."""

    def test_interior_constant_function(self):
        """Interior stencil should give zero derivative for constants."""
        h = 0.1
        stencil = e2_poly_interior(h)

        # f(x) = c for any constant c
        f = np.array([5.0, 5.0, 5.0])
        result = np.dot(stencil, f)

        assert abs(result) < 1e-14, f"Derivative of constant should be zero, got {result}"

    def test_interior_linear_function(self):
        """Interior stencil should be exact for linear functions."""
        h = 0.1
        stencil = e2_poly_interior(h)

        # f(x) = 3x + 2 at x = {-h, 0, h}
        x = np.array([-h, 0, h])
        f = 3.0 * x + 2.0

        # Exact derivative is 3.0
        result = np.dot(stencil, f)

        np.testing.assert_allclose(result, 3.0, rtol=1e-12, atol=1e-14)

    def test_interior_quadratic_function(self):
        """Interior stencil should be exact for quadratic functions."""
        h = 0.1
        stencil = e2_poly_interior(h)

        # f(x) = 2x^2 + 3x + 1 at x = 0
        # f'(x) = 4x + 3 = 3 at x = 0
        x = np.array([-h, 0, h])
        f = 2.0 * x**2 + 3.0 * x + 1.0

        result = np.dot(stencil, f)
        expected = 3.0  # derivative at x=0

        np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-13)


class TestOperatorReadiness:
    """Test properties critical for operator construction (Phase 3)."""

    def test_coefficient_array_sizes(self):
        """Verify coefficient arrays have correct sizes for operator assembly."""
        h = 1.0
        psi = 0.5

        # Interior: should be 3 elements for 3-point stencil
        interior = e2_poly_interior(h)
        assert len(interior) == 3, f"Interior stencil should have 3 coefficients"

        # Dirichlet NBS: should be (R-1)*T = 2*4 = 8 elements
        da = np.array([0.12, 0.13, 0.14])
        dirichlet = e2_poly_nbs_dirichlet(h, psi, da, right=False)
        assert len(dirichlet) == 8, f"Dirichlet NBS should have 8 coefficients"

        # Floating NBS: should be R*T = 3*4 = 12 elements
        fa = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        floating = e2_poly_nbs_floating(h, psi, fa, right=False)
        assert len(floating) == 12, f"Floating NBS should have 12 coefficients"

    def test_left_right_boundary_consistency(self):
        """Test that left and right boundaries are consistent."""
        h = 1.0
        psi = 0.5
        da = np.array([0.12, 0.13, 0.14])

        # Get both boundaries
        left = e2_poly_nbs_dirichlet(h, psi, da, right=False)
        right = e2_poly_nbs_dirichlet(h, psi, da, right=True)

        # Should both have same length
        assert len(left) == len(right), "Left and right boundaries should have same size"

        # Should both be finite
        assert np.all(np.isfinite(left)), "Left boundary has non-finite values"
        assert np.all(np.isfinite(right)), "Right boundary has non-finite values"

    def test_no_catastrophic_cancellation(self):
        """Test for catastrophic cancellation in coefficient computation."""
        h = 1.0
        psi_values = [1e-10, 0.001, 0.5, 0.999, 1.0]
        da = np.array([0.12, 0.13, 0.14])

        for psi in psi_values:
            stencil = e2_poly_nbs_dirichlet(h, psi, da, right=False)

            # Check relative precision: no coefficient should be dominated by roundoff
            # (using machine epsilon for double precision)
            max_coeff = np.max(np.abs(stencil))
            min_nonzero = np.min(np.abs(stencil[stencil != 0]))

            # Ratio should not exceed ~1e14 (close to double precision limit)
            if min_nonzero > 0:
                ratio = max_coeff / min_nonzero
                assert ratio < 1e14, f"Potential catastrophic cancellation at psi={psi}: ratio={ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
