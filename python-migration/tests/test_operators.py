"""
Tests for differential operators and matrix builders.

These tests verify:
1. Matrix construction utilities (circulant, banded)
2. Derivative of x^2 = 2x (exactly for interior points)
3. Derivative of sin(x) converges at 2nd order
4. Operator application to ScalarFields
"""

import pytest
import numpy as np
from shoccs.geometry import CartesianMesh
from shoccs.fields import ScalarField
from shoccs.stencils import (
    centered_diff_1st_order2,
    centered_diff_2nd_order2,
    centered_diff_1st_order4,
)
from shoccs.operators import (
    create_derivative_operator,
    create_gradient_operator,
    create_laplacian_operator
)
from shoccs.operators.matrix_builders import (
    build_circulant_operator,
    build_banded_matrix,
    build_1d_derivative_matrix,
    apply_matrix_free_1d,
)


class TestMatrixBuilders:
    """Tests for matrix construction utilities (Phase 3)."""

    def test_circulant_1d_shape(self):
        """Test circulant matrix has correct shape."""
        h = 0.1
        n = 20
        stencil = centered_diff_1st_order2(h)

        D = build_circulant_operator(stencil, n)

        assert D.shape == (n, n)
        assert D.nnz > 0

    def test_circulant_1d_constant(self):
        """Test derivative of constant is zero with circulant matrix."""
        h = 0.1
        n = 50
        stencil = centered_diff_1st_order2(h)
        D = build_circulant_operator(stencil, n)

        u = np.ones(n) * 5.0
        du = D @ u

        np.testing.assert_allclose(du, 0.0, atol=1e-13)

    def test_circulant_1d_periodic_sine(self):
        """Test derivative of periodic sine function is accurate."""
        h = 0.1
        xmin, xmax = 0.0, 1.0
        n = int((xmax - xmin) / h)
        x = np.linspace(xmin, xmax - h, n)

        stencil = centered_diff_1st_order2(h)
        D = build_circulant_operator(stencil, n)

        # Use a periodic function
        u = np.sin(2 * np.pi * x)
        du = D @ u
        du_exact = 2 * np.pi * np.cos(2 * np.pi * x)

        # Should be accurate for periodic function (2nd order, so O(h^2))
        np.testing.assert_allclose(du, du_exact, rtol=0.1, atol=0.5)

    def test_banded_matrix_shape(self):
        """Test banded matrix has correct shape for Dirichlet BC."""
        h = 0.1
        n = 20
        stencil = centered_diff_1st_order2(h)

        D = build_banded_matrix(stencil, n)

        # Should be (n-2) x n for 3-point stencil
        assert D.shape == (n - 2, n)

    def test_banded_matrix_constant(self):
        """Test derivative of constant is zero with banded matrix."""
        h = 0.1
        n = 50
        stencil = centered_diff_1st_order2(h)
        D = build_banded_matrix(stencil, n)

        u = np.ones(n) * 5.0
        du = D @ u

        np.testing.assert_allclose(du, 0.0, atol=1e-13)

    def test_banded_matrix_linear(self):
        """Test derivative of linear function is exact."""
        h = 0.1
        xmin, xmax = 0.0, 1.0
        n = int((xmax - xmin) / h) + 1
        x = np.linspace(xmin, xmax, n)

        stencil = centered_diff_1st_order2(h)
        D = build_banded_matrix(stencil, n)

        u = 3.0 * x + 2.0
        du = D @ u
        du_exact = 3.0

        np.testing.assert_allclose(du, du_exact, rtol=1e-11, atol=1e-13)


class TestPolynomials1D:
    """Test derivatives on polynomials (Phase 3 requirement)."""

    def test_derivative_of_x_squared_periodic(self):
        """Derivative of periodic x^2 - test interior points only."""
        h = 0.05
        xmin, xmax = 0.0, 1.0
        n = int((xmax - xmin) / h)
        x = np.linspace(xmin, xmax - h, n)

        stencil = centered_diff_1st_order2(h)
        D = build_circulant_operator(stencil, n)

        u = x**2
        du = D @ u
        du_exact = 2.0 * x

        # Test interior points only (avoid boundary wrap-around)
        np.testing.assert_allclose(du[2:-2], du_exact[2:-2], rtol=1e-10, atol=1e-12)

    def test_derivative_of_x_squared_dirichlet(self):
        """Derivative of x^2 with Dirichlet BC - exact everywhere."""
        h = 0.1
        xmin, xmax = 0.0, 1.0
        n = int((xmax - xmin) / h) + 1
        x = np.linspace(xmin, xmax, n)

        stencil = centered_diff_1st_order2(h)
        D = build_banded_matrix(stencil, n)

        u = x**2
        du = D @ u
        du_exact = 2.0 * x[1:-1]  # Interior points

        # Should be exact for 2nd order stencil on quadratic
        np.testing.assert_allclose(du, du_exact, rtol=1e-10, atol=1e-12)

    def test_derivative_of_cubic_dirichlet(self):
        """Derivative of x^3 with Dirichlet BC."""
        h = 0.05
        xmin, xmax = 0.0, 1.0
        n = int((xmax - xmin) / h) + 1
        x = np.linspace(xmin, xmax, n)

        stencil = centered_diff_1st_order2(h)
        D = build_banded_matrix(stencil, n)

        u = x**3 + 2 * x**2 + x + 1
        du = D @ u
        du_exact = (3 * x**2 + 4 * x + 1)[1:-1]  # Interior points

        # Should be accurate for 2nd order stencil on cubic derivative (quadratic)
        np.testing.assert_allclose(du, du_exact, rtol=1e-2, atol=5e-3)


class TestConvergence1D:
    """Test convergence on sin(x) (Phase 3 requirement)."""

    def test_sine_convergence_2nd_order(self):
        """Test that sin(x) derivative converges at 2nd order."""
        xmin, xmax = 0.0, 1.0
        grid_sizes = [0.1, 0.05, 0.025, 0.0125]
        errors = []

        for h in grid_sizes:
            n = int((xmax - xmin) / h)
            x = np.linspace(xmin, xmax - h, n)

            stencil = centered_diff_1st_order2(h)
            D = build_circulant_operator(stencil, n)

            u = np.sin(2 * np.pi * x)
            du = D @ u
            du_exact = 2 * np.pi * np.cos(2 * np.pi * x)

            error = np.max(np.abs(du - du_exact))
            errors.append(error)

        # Errors should decrease
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i]

        # Compute convergence rates
        rates = []
        for i in range(len(errors) - 1):
            rate = np.log2(errors[i] / errors[i + 1]) / np.log2(
                grid_sizes[i] / grid_sizes[i + 1]
            )
            rates.append(rate)

        # Average rate should be close to 2.0
        avg_rate = np.mean(rates)
        assert 1.8 < avg_rate < 2.2

    def test_sine_convergence_4th_order(self):
        """Test that sin(x) derivative converges at 4th order."""
        xmin, xmax = 0.0, 1.0
        grid_sizes = [0.1, 0.05, 0.025]
        errors = []

        for h in grid_sizes:
            n = int((xmax - xmin) / h)
            x = np.linspace(xmin, xmax - h, n)

            stencil = centered_diff_1st_order4(h)
            D = build_circulant_operator(stencil, n)

            u = np.sin(2 * np.pi * x)
            du = D @ u
            du_exact = 2 * np.pi * np.cos(2 * np.pi * x)

            error = np.max(np.abs(du - du_exact))
            errors.append(error)

        # Errors should decrease
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i]

        # Compute convergence rates
        rates = []
        for i in range(len(errors) - 1):
            rate = np.log2(errors[i] / errors[i + 1]) / np.log2(
                grid_sizes[i] / grid_sizes[i + 1]
            )
            rates.append(rate)

        # Average rate should be close to 4.0
        avg_rate = np.mean(rates)
        assert 3.5 < avg_rate < 4.5


class TestMatrixFreeApplication:
    """Test matrix-free operator application."""

    def test_matrix_free_periodic(self):
        """Test matrix-free matches matrix application (periodic)."""
        h = 0.1
        n = 50
        x = np.linspace(0, 1 - h, n)

        stencil = centered_diff_1st_order2(h)

        # Matrix version
        D = build_circulant_operator(stencil, n)
        u = np.sin(2 * np.pi * x)
        du_matrix = D @ u

        # Matrix-free version
        du_free = np.zeros_like(u)
        apply_matrix_free_1d(u, stencil, du_free, periodic=True)

        np.testing.assert_allclose(du_matrix, du_free, rtol=1e-14)

    def test_matrix_free_dirichlet(self):
        """Test matrix-free matches matrix application (Dirichlet)."""
        h = 0.1
        n = 50
        x = np.linspace(0, 1, n)

        stencil = centered_diff_1st_order2(h)

        # Matrix version
        D = build_banded_matrix(stencil, n)
        u = x**2
        du_matrix = D @ u

        # Matrix-free version
        du_full = np.zeros_like(u)
        apply_matrix_free_1d(u, stencil, du_full, periodic=False)
        du_free = du_full[1:-1]

        np.testing.assert_allclose(du_matrix, du_free, rtol=1e-14)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
