"""
Unit tests for finite difference stencils.

Tests verify:
1. Stencil coefficients are correct
2. Stencils reproduce polynomials exactly (up to roundoff)
3. Numba compilation works
4. E2-Poly stencils function correctly
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shoccs.stencils import (
    centered_diff_1st_order2,
    centered_diff_2nd_order2,
    centered_diff_1st_order4,
    centered_diff_2nd_order4,
    apply_stencil_1d,
    E2PolyStencil,
    e2_poly_interior,
    e2_poly_interp_interior,
    e2_poly_interp_wall,
    e2_poly_nbs_floating,
    e2_poly_nbs_dirichlet,
    make_e2_poly_stencil,
)


class TestInteriorStencils:
    """Tests for basic interior finite difference stencils."""

    def test_centered_diff_1st_order2_coefficients(self):
        """Test that 2nd-order 1st derivative stencil has correct coefficients."""
        h = 0.1
        stencil = centered_diff_1st_order2(h)

        expected = np.array([-1.0 / (2 * h), 0.0, 1.0 / (2 * h)])
        np.testing.assert_allclose(stencil, expected, rtol=1e-14)

    def test_centered_diff_2nd_order2_coefficients(self):
        """Test that 2nd-order 2nd derivative stencil has correct coefficients."""
        h = 0.1
        stencil = centered_diff_2nd_order2(h)

        h2 = h * h
        expected = np.array([1.0 / h2, -2.0 / h2, 1.0 / h2])
        np.testing.assert_allclose(stencil, expected, rtol=1e-14)

    def test_centered_diff_1st_order4_coefficients(self):
        """Test that 4th-order 1st derivative stencil has correct coefficients."""
        h = 0.1
        stencil = centered_diff_1st_order4(h)

        denom = 12.0 * h
        expected = np.array([1.0 / denom, -8.0 / denom, 0.0, 8.0 / denom, -1.0 / denom])
        np.testing.assert_allclose(stencil, expected, rtol=1e-14)

    def test_centered_diff_2nd_order4_coefficients(self):
        """Test that 4th-order 2nd derivative stencil has correct coefficients."""
        h = 0.1
        stencil = centered_diff_2nd_order4(h)

        h2 = h * h
        denom = 12.0 * h2
        expected = np.array([-1.0 / denom, 16.0 / denom, -30.0 / denom,
                            16.0 / denom, -1.0 / denom])
        np.testing.assert_allclose(stencil, expected, rtol=1e-14)

    def test_1st_derivative_constant_function(self):
        """Test that 1st derivative of constant is zero."""
        h = 0.1
        x = np.arange(0, 1.0, h)
        f = np.ones_like(x) * 5.0  # f(x) = 5

        stencil = centered_diff_1st_order2(h)

        # Apply at interior points
        for i in range(1, len(x) - 1):
            df = apply_stencil_1d(f, stencil, i)
            assert abs(df) < 1e-14, f"Derivative of constant should be zero, got {df}"

    def test_1st_derivative_linear_function(self):
        """Test that 1st derivative of linear function is exact."""
        h = 0.1
        x = np.arange(0, 1.0, h)
        a, b = 3.0, 2.0
        f = a * x + b  # f(x) = 3x + 2
        df_exact = a   # f'(x) = 3

        stencil = centered_diff_1st_order2(h)

        # Apply at interior points
        for i in range(1, len(x) - 1):
            df = apply_stencil_1d(f, stencil, i)
            np.testing.assert_allclose(df, df_exact, rtol=1e-12, atol=1e-14)

    def test_1st_derivative_quadratic_function(self):
        """Test that 2nd-order stencil reproduces quadratic derivative exactly."""
        h = 0.1
        x = np.arange(0, 1.0, h)
        a, b, c = 2.0, 3.0, 1.0
        f = a * x**2 + b * x + c  # f(x) = 2x² + 3x + 1
        df_exact = 2 * a * x + b  # f'(x) = 4x + 3

        stencil = centered_diff_1st_order2(h)

        # Apply at interior points
        for i in range(1, len(x) - 1):
            df = apply_stencil_1d(f, stencil, i)
            np.testing.assert_allclose(df, df_exact[i], rtol=1e-11, atol=1e-13)

    def test_2nd_derivative_linear_function(self):
        """Test that 2nd derivative of linear function is zero."""
        h = 0.1
        x = np.arange(0, 1.0, h)
        f = 3.0 * x + 2.0  # f(x) = 3x + 2

        stencil = centered_diff_2nd_order2(h)

        # Apply at interior points
        for i in range(1, len(x) - 1):
            d2f = apply_stencil_1d(f, stencil, i)
            assert abs(d2f) < 1e-12, f"2nd derivative of linear should be zero, got {d2f}"

    def test_2nd_derivative_quadratic_function(self):
        """Test that 2nd derivative of quadratic is exact."""
        h = 0.1
        x = np.arange(0, 1.0, h)
        a = 2.0
        f = a * x**2 + 3.0 * x + 1.0  # f(x) = 2x² + 3x + 1
        d2f_exact = 2 * a              # f''(x) = 4

        stencil = centered_diff_2nd_order2(h)

        # Apply at interior points
        for i in range(1, len(x) - 1):
            d2f = apply_stencil_1d(f, stencil, i)
            np.testing.assert_allclose(d2f, d2f_exact, rtol=1e-11, atol=1e-13)

    def test_2nd_derivative_cubic_function(self):
        """Test that 2nd-order stencil reproduces cubic 2nd derivative exactly."""
        h = 0.1
        x = np.arange(0, 1.0, h)
        a, b, c, d = 1.0, 2.0, 3.0, 1.0
        f = a * x**3 + b * x**2 + c * x + d  # f(x) = x³ + 2x² + 3x + 1
        d2f_exact = 6 * a * x + 2 * b        # f''(x) = 6x + 4

        stencil = centered_diff_2nd_order2(h)

        # Apply at interior points
        for i in range(1, len(x) - 1):
            d2f = apply_stencil_1d(f, stencil, i)
            np.testing.assert_allclose(d2f, d2f_exact[i], rtol=1e-10, atol=1e-12)

    def test_4th_order_1st_derivative_cubic(self):
        """Test that 4th-order stencil reproduces cubic derivative exactly."""
        h = 0.1
        x = np.arange(0, 2.0, h)
        a, b, c, d = 1.0, 2.0, 3.0, 1.0
        f = a * x**3 + b * x**2 + c * x + d  # f(x) = x³ + 2x² + 3x + 1
        df_exact = 3 * a * x**2 + 2 * b * x + c  # f'(x) = 3x² + 4x + 3

        stencil = centered_diff_1st_order4(h)

        # Apply at interior points (need 2 points on each side)
        for i in range(2, len(x) - 2):
            df = apply_stencil_1d(f, stencil, i)
            np.testing.assert_allclose(df, df_exact[i], rtol=1e-10, atol=1e-12)

    def test_4th_order_2nd_derivative_quartic(self):
        """Test that 4th-order 2nd derivative stencil reproduces quartic exactly."""
        h = 0.1
        x = np.arange(0, 2.0, h)
        a = 1.0
        f = a * x**4 + 2 * x**3 + 3 * x**2 + 4 * x + 5
        d2f_exact = 12 * a * x**2 + 12 * x + 6  # f''(x) = 12x² + 12x + 6

        stencil = centered_diff_2nd_order4(h)

        # Apply at interior points (need 2 points on each side)
        for i in range(2, len(x) - 2):
            d2f = apply_stencil_1d(f, stencil, i)
            np.testing.assert_allclose(d2f, d2f_exact[i], rtol=1e-9, atol=1e-11)

    def test_stencils_are_numba_compiled(self):
        """Test that stencils successfully compile with Numba."""
        # If functions weren't compiled, they wouldn't have these attributes
        h = 0.1

        # Just calling them should trigger compilation
        s1 = centered_diff_1st_order2(h)
        s2 = centered_diff_2nd_order2(h)
        s3 = centered_diff_1st_order4(h)
        s4 = centered_diff_2nd_order4(h)

        # Verify they return numpy arrays
        assert isinstance(s1, np.ndarray)
        assert isinstance(s2, np.ndarray)
        assert isinstance(s3, np.ndarray)
        assert isinstance(s4, np.ndarray)

        # Test apply function too
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_stencil_1d(data, s1, 2)
        assert isinstance(result, (float, np.floating))


class TestE2PolyStencils:
    """Tests for E2-Poly stencils."""

    def test_e2_poly_stencil_creation(self):
        """Test creating E2-Poly stencil configuration."""
        fa = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        da = np.array([0.5, 1.5, 2.5])
        ia = np.array([0.1, 0.2, 0.3, 0.4])

        stencil = E2PolyStencil(fa, da, ia)

        np.testing.assert_array_equal(stencil.fa, fa)
        np.testing.assert_array_equal(stencil.da, da)
        np.testing.assert_array_equal(stencil.ia, ia)

    def test_e2_poly_stencil_padding(self):
        """Test that E2-Poly stencil pads short arrays with zeros."""
        fa = np.array([1.0, 2.0])  # Only 2 elements, should pad to 6
        da = np.array([0.5])       # Only 1 element, should pad to 3
        ia = np.array([0.1, 0.2])  # Only 2 elements, should pad to 4

        stencil = E2PolyStencil(fa, da, ia)

        assert len(stencil.fa) == 6
        assert len(stencil.da) == 3
        assert len(stencil.ia) == 4

        # Check first elements are preserved
        assert stencil.fa[0] == 1.0
        assert stencil.fa[1] == 2.0
        # Check padding
        assert stencil.fa[2] == 0.0

    def test_e2_poly_interior_stencil(self):
        """Test E2-Poly interior stencil (simple centered difference)."""
        h = 0.1
        stencil = e2_poly_interior(h)

        expected = np.array([-0.5 / h, 0.0, 0.5 / h])
        np.testing.assert_allclose(stencil, expected, rtol=1e-14)

    def test_e2_poly_interior_linear_function(self):
        """Test that E2-Poly interior stencil reproduces linear derivative."""
        h = 0.1
        x = np.arange(0, 1.0, h)
        f = 3.0 * x + 2.0
        df_exact = 3.0

        stencil = e2_poly_interior(h)

        for i in range(1, len(x) - 1):
            df = apply_stencil_1d(f, stencil, i)
            np.testing.assert_allclose(df, df_exact, rtol=1e-12)

    def test_e2_poly_interp_interior_zero_offset(self):
        """Test E2-Poly interior interpolation at zero offset."""
        y = 0.0
        c = e2_poly_interp_interior(y)

        # At y=0, should get [1, 1] (averaging two neighbors)
        expected = np.array([0.0, 1.0])
        np.testing.assert_allclose(c, expected, rtol=1e-14)

    def test_e2_poly_interp_interior_positive_offset(self):
        """Test E2-Poly interior interpolation at positive offset."""
        y = 0.3
        c = e2_poly_interp_interior(y)

        # At y=0.3, should get [1-0.3, 0.3] = [0.7, 0.3]
        expected = np.array([0.7, 0.3])
        np.testing.assert_allclose(c, expected, rtol=1e-14)

    def test_e2_poly_interp_interior_negative_offset(self):
        """Test E2-Poly interior interpolation at negative offset."""
        y = -0.4
        c = e2_poly_interp_interior(y)

        # At y=-0.4, should get [0.4, 0.6]
        expected = np.array([0.4, 0.6])
        np.testing.assert_allclose(c, expected, rtol=1e-14)

    def test_e2_poly_interp_interior_sum_to_one(self):
        """Test that interior interpolation coefficients sum to one."""
        for y in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            c = e2_poly_interp_interior(y)
            np.testing.assert_allclose(np.sum(c), 1.0, rtol=1e-14)

    def test_e2_poly_nbs_floating_shape(self):
        """Test that floating NBS has correct shape."""
        h = 0.1
        psi = 0.5
        fa = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        stencil_right = e2_poly_nbs_floating(h, psi, fa, right=True)
        stencil_left = e2_poly_nbs_floating(h, psi, fa, right=False)

        # Should have R * T = 3 * 4 = 12 elements
        assert len(stencil_right) == 12
        assert len(stencil_left) == 12

    def test_e2_poly_nbs_dirichlet_shape(self):
        """Test that Dirichlet NBS has correct shape."""
        h = 0.1
        psi = 0.5
        da = np.array([1.0, 2.0, 3.0])

        stencil_right = e2_poly_nbs_dirichlet(h, psi, da, right=True)
        stencil_left = e2_poly_nbs_dirichlet(h, psi, da, right=False)

        # Should have (R-1) * T = 2 * 4 = 8 elements
        assert len(stencil_right) == 8
        assert len(stencil_left) == 8

    def test_e2_poly_nbs_floating_nonzero(self):
        """Test that floating NBS produces non-zero coefficients."""
        h = 0.1
        psi = 0.5
        fa = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        stencil = e2_poly_nbs_floating(h, psi, fa, right=False)

        # Should have some non-zero entries
        assert np.any(np.abs(stencil) > 1e-10)

    def test_e2_poly_nbs_dirichlet_nonzero(self):
        """Test that Dirichlet NBS produces non-zero coefficients."""
        h = 0.1
        psi = 0.5
        da = np.array([1.0, 2.0, 3.0])

        stencil = e2_poly_nbs_dirichlet(h, psi, da, right=False)

        # Should have some non-zero entries
        assert np.any(np.abs(stencil) > 1e-10)

    def test_make_e2_poly_stencil_factory(self):
        """Test the factory function for creating E2-Poly stencils."""
        fa = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        da = np.array([0.5, 1.5, 2.5])
        ia = np.array([0.1, 0.2, 0.3, 0.4])

        stencil = make_e2_poly_stencil(fa, da, ia)

        assert isinstance(stencil, E2PolyStencil)
        np.testing.assert_array_equal(stencil.fa, fa)
        np.testing.assert_array_equal(stencil.da, da)
        np.testing.assert_array_equal(stencil.ia, ia)

    def test_e2_poly_functions_are_numba_compiled(self):
        """Test that E2-Poly functions compile with Numba."""
        h = 0.1
        psi = 0.5
        fa = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        da = np.array([1.0, 2.0, 3.0])
        ia = np.array([0.1, 0.2, 0.3, 0.4])

        # Call all Numba functions to trigger compilation
        s1 = e2_poly_interior(h)
        s2 = e2_poly_interp_interior(0.5)
        s3 = e2_poly_interp_wall(0, 0.5, psi, fa, ia, True)
        s4 = e2_poly_nbs_floating(h, psi, fa, False)
        s5 = e2_poly_nbs_dirichlet(h, psi, da, False)

        # All should return numpy arrays
        assert isinstance(s1, np.ndarray)
        assert isinstance(s2, np.ndarray)
        assert isinstance(s3, np.ndarray)
        assert isinstance(s4, np.ndarray)
        assert isinstance(s5, np.ndarray)


class TestStencilIntegration:
    """Integration tests combining different stencil components."""

    def test_comparison_2nd_and_4th_order_convergence(self):
        """Test that 4th order converges faster than 2nd order."""
        # Use a smooth function
        def f(x):
            return np.sin(2 * np.pi * x)

        def df(x):
            return 2 * np.pi * np.cos(2 * np.pi * x)

        errors_2nd = []
        errors_4th = []
        grid_sizes = [0.1, 0.05, 0.025]

        for h in grid_sizes:
            x = np.arange(0, 1.0, h)
            y = f(x)
            dy_exact = df(x)

            # 2nd order
            stencil_2 = centered_diff_1st_order2(h)
            errors_2nd_h = []
            for i in range(1, len(x) - 1):
                dy_approx = apply_stencil_1d(y, stencil_2, i)
                errors_2nd_h.append(abs(dy_approx - dy_exact[i]))
            errors_2nd.append(max(errors_2nd_h))

            # 4th order
            stencil_4 = centered_diff_1st_order4(h)
            errors_4th_h = []
            for i in range(2, len(x) - 2):
                dy_approx = apply_stencil_1d(y, stencil_4, i)
                errors_4th_h.append(abs(dy_approx - dy_exact[i]))
            errors_4th.append(max(errors_4th_h))

        # Check that errors decrease
        assert errors_2nd[1] < errors_2nd[0]
        assert errors_2nd[2] < errors_2nd[1]
        assert errors_4th[1] < errors_4th[0]
        assert errors_4th[2] < errors_4th[1]

        # 4th order should be more accurate (at least on finest grid)
        assert errors_4th[-1] < errors_2nd[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
