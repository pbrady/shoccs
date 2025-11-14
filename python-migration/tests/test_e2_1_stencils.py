"""
Tests for E2_1 stencils.

Validates the Python/Numba translation against C++ reference implementation.
"""

import numpy as np
import pytest
from shoccs.stencils import e2_1


class TestE2_1Interior:
    """Test E2_1 interior stencils."""

    def test_interior_coefficients(self):
        """Test interior stencil coefficients."""
        h = 0.1
        c = e2_1.interior(h)

        # Should be 3-point centered difference
        expected = np.array([-1.0, 0.0, 1.0]) / (2.0 * h)

        np.testing.assert_allclose(c, expected, rtol=1e-14, atol=1e-15)

    def test_interior_different_spacing(self):
        """Test interior stencil with different grid spacing."""
        h = 0.05
        c = e2_1.interior(h)

        expected = np.array([-1.0, 0.0, 1.0]) / (2.0 * h)

        np.testing.assert_allclose(c, expected, rtol=1e-14, atol=1e-15)


class TestE2_1Interpolation:
    """Test E2_1 interpolation stencils."""

    @pytest.mark.parametrize("y", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_interp_interior_positive(self, y):
        """Test interior interpolation for positive offsets."""
        c = e2_1.interp_interior(y)

        # Should have 2 coefficients
        assert c.shape == (2,)

        # Coefficients should sum to 1 (partition of unity)
        np.testing.assert_allclose(np.sum(c), 1.0, rtol=1e-14)

        # Expected values for linear interpolation
        # Note: C++ uses y > 0, not y >= 0, so y=0 goes to else branch
        if y > 0:
            expected = np.array([1.0 - y, y])
        else:
            expected = np.array([-y, 1.0 + y])

        np.testing.assert_allclose(c, expected, rtol=1e-14, atol=1e-15)

    @pytest.mark.parametrize("y", [-1.0, -0.75, -0.5, -0.25])
    def test_interp_interior_negative(self, y):
        """Test interior interpolation for negative offsets."""
        c = e2_1.interp_interior(y)

        assert c.shape == (2,)
        np.testing.assert_allclose(np.sum(c), 1.0, rtol=1e-14)

        expected = np.array([-y, 1.0 + y])
        np.testing.assert_allclose(c, expected, rtol=1e-14, atol=1e-15)

    @pytest.mark.parametrize("i", [0, 1])
    @pytest.mark.parametrize("psi", [0.1, 0.5, 0.9])
    def test_interp_wall_right(self, i, psi):
        """Test wall interpolation for right boundary."""
        y = 0.5
        alpha = np.array([-1.47956, 0.26190, -0.14507, -0.22467])

        c = e2_1.interp_wall(i, y, psi, alpha, right=True)

        # Should have 3 coefficients
        assert c.shape == (3,)

        # All finite
        assert np.all(np.isfinite(c))

    @pytest.mark.parametrize("i", [0, 1])
    @pytest.mark.parametrize("psi", [0.1, 0.5, 0.9])
    def test_interp_wall_left(self, i, psi):
        """Test wall interpolation for left boundary."""
        y = 0.5
        alpha = np.array([-1.47956, 0.26190, -0.14507, -0.22467])

        c = e2_1.interp_wall(i, y, psi, alpha, right=False)

        # Should have 3 coefficients
        assert c.shape == (3,)

        # All finite
        assert np.all(np.isfinite(c))


class TestE2_1FloatingStencils:
    """Test E2_1 floating boundary stencils."""

    @pytest.fixture
    def alpha(self):
        """Standard alpha parameters for testing."""
        return np.array([-1.47956, 0.26190, -0.14507, -0.22467])

    @pytest.mark.parametrize("psi", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_floating_shape(self, alpha, psi):
        """Test that floating stencil has correct shape."""
        h = 0.1
        c = e2_1.nbs_floating(h, psi, alpha, right=False)

        # Should be R * T = 4 * 5 = 20 coefficients
        assert c.shape == (20,)

        # All should be finite
        assert np.all(np.isfinite(c))

    @pytest.mark.parametrize("psi", [0.1, 0.5, 0.9])
    def test_floating_left_vs_right(self, alpha, psi):
        """Test that left and right boundaries are related correctly."""
        h = 0.1
        c_left = e2_1.nbs_floating(h, psi, alpha, right=False)
        c_right = e2_1.nbs_floating(h, psi, alpha, right=True)

        # Right should be -reverse of left (from C++ implementation)
        expected_right = -c_left[::-1]

        np.testing.assert_allclose(c_right, expected_right, rtol=1e-14, atol=1e-15)

    def test_floating_grid_spacing_scaling(self, alpha):
        """Test that coefficients scale correctly with grid spacing."""
        psi = 0.5
        h1 = 0.1
        h2 = 0.2

        c1 = e2_1.nbs_floating(h1, psi, alpha, right=False)
        c2 = e2_1.nbs_floating(h2, psi, alpha, right=False)

        # Coefficients should scale as 1/h (derivative operator)
        expected = c1 * (h1 / h2)

        np.testing.assert_allclose(c2, expected, rtol=1e-14, atol=1e-15)


class TestE2_1DirichletStencils:
    """Test E2_1 Dirichlet boundary stencils."""

    @pytest.fixture
    def alpha(self):
        """Standard alpha parameters for testing."""
        return np.array([-1.47956, 0.26190, -0.14507, -0.22467])

    @pytest.mark.parametrize("psi", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_dirichlet_shape(self, alpha, psi):
        """Test that Dirichlet stencil has correct shape."""
        h = 0.1
        c = e2_1.nbs_dirichlet(h, psi, alpha, right=False)

        # Should be (R-1) * T = 3 * 5 = 15 coefficients
        assert c.shape == (15,)

        # All should be finite
        assert np.all(np.isfinite(c))

    @pytest.mark.parametrize("psi", [0.1, 0.5, 0.9])
    def test_dirichlet_left_vs_right(self, alpha, psi):
        """Test that left and right boundaries are related correctly."""
        h = 0.1
        c_left = e2_1.nbs_dirichlet(h, psi, alpha, right=False)
        c_right = e2_1.nbs_dirichlet(h, psi, alpha, right=True)

        # Right should be -reverse of left
        expected_right = -c_left[::-1]

        np.testing.assert_allclose(c_right, expected_right, rtol=1e-14, atol=1e-15)

    def test_dirichlet_grid_spacing_scaling(self, alpha):
        """Test that coefficients scale correctly with grid spacing."""
        psi = 0.5
        h1 = 0.1
        h2 = 0.2

        c1 = e2_1.nbs_dirichlet(h1, psi, alpha, right=False)
        c2 = e2_1.nbs_dirichlet(h2, psi, alpha, right=False)

        # Coefficients should scale as 1/h
        expected = c1 * (h1 / h2)

        np.testing.assert_allclose(c2, expected, rtol=1e-14, atol=1e-15)


class TestE2_1Neumann:
    """Test E2_1 Neumann boundary stencils."""

    def test_neumann_not_implemented(self):
        """Test that Neumann stencil returns empty array."""
        h = 0.1
        psi = 0.5
        alpha = np.array([-1.47956, 0.26190, -0.14507, -0.22467])

        c = e2_1.nbs_neumann(h, psi, alpha, right=False)

        # Should be empty (not implemented)
        assert c.shape == (0,)


class TestE2_1StencilClass:
    """Test E2_1Stencil configuration class."""

    def test_creation_with_alpha(self):
        """Test creating stencil with alpha parameters."""
        alpha = np.array([-1.47956, 0.26190, -0.14507, -0.22467])
        stencil = e2_1.E2_1Stencil(alpha)

        np.testing.assert_allclose(stencil.alpha, alpha, rtol=1e-14)

    def test_creation_without_alpha(self):
        """Test creating stencil without parameters."""
        stencil = e2_1.E2_1Stencil()

        # Should initialize to zeros
        expected = np.zeros(4)
        np.testing.assert_allclose(stencil.alpha, expected, rtol=1e-14)

    def test_alpha_padding(self):
        """Test that alpha is padded correctly."""
        alpha = np.array([1.0, 2.0])  # Only 2 elements
        stencil = e2_1.E2_1Stencil(alpha)

        # Should be padded to 4 elements
        expected = np.array([1.0, 2.0, 0.0, 0.0])
        np.testing.assert_allclose(stencil.alpha, expected, rtol=1e-14)

    def test_alpha_truncation(self):
        """Test that alpha is truncated correctly."""
        alpha = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 6 elements
        stencil = e2_1.E2_1Stencil(alpha)

        # Should be truncated to 4 elements
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(stencil.alpha, expected, rtol=1e-14)


class TestE2_1DispatchFunction:
    """Test the nbs dispatch function."""

    @pytest.fixture
    def params(self):
        """Standard parameters for testing."""
        alpha = np.array([-1.47956, 0.26190, -0.14507, -0.22467])
        return e2_1.E2_1Stencil(alpha)

    def test_dispatch_floating(self, params):
        """Test dispatch to floating boundary."""
        h = 0.1
        psi = 0.5

        result = e2_1.nbs(h, 'floating', psi, params, right=False)

        assert result.shape == (20,)  # R * T = 4 * 5
        assert np.all(np.isfinite(result))

    def test_dispatch_dirichlet(self, params):
        """Test dispatch to Dirichlet boundary."""
        h = 0.1
        psi = 0.5

        result = e2_1.nbs(h, 'dirichlet', psi, params, right=False)

        assert result.shape == (15,)  # (R-1) * T = 3 * 5
        assert np.all(np.isfinite(result))

    def test_dispatch_neumann(self, params):
        """Test dispatch to Neumann boundary."""
        h = 0.1
        psi = 0.5

        result = e2_1.nbs(h, 'neumann', psi, params, right=False)

        assert result.shape == (0,)  # Not implemented

    def test_dispatch_case_insensitive(self, params):
        """Test that boundary type is case-insensitive."""
        h = 0.1
        psi = 0.5

        result1 = e2_1.nbs(h, 'FLOATING', psi, params, right=False)
        result2 = e2_1.nbs(h, 'Floating', psi, params, right=False)
        result3 = e2_1.nbs(h, 'floating', psi, params, right=False)

        np.testing.assert_allclose(result1, result2, rtol=1e-14)
        np.testing.assert_allclose(result2, result3, rtol=1e-14)

    def test_dispatch_invalid_type(self, params):
        """Test that invalid boundary type raises error."""
        h = 0.1
        psi = 0.5

        with pytest.raises(ValueError, match="Unknown boundary condition type"):
            e2_1.nbs(h, 'invalid', psi, params, right=False)


class TestE2_1FactoryFunction:
    """Test the factory function."""

    def test_factory_with_params(self):
        """Test factory function with parameters."""
        alpha = np.array([-1.47956, 0.26190, -0.14507, -0.22467])

        stencil = e2_1.make_e2_1_stencil(alpha)

        assert isinstance(stencil, e2_1.E2_1Stencil)
        np.testing.assert_allclose(stencil.alpha, alpha, rtol=1e-14)

    def test_factory_without_params(self):
        """Test factory function without parameters."""
        stencil = e2_1.make_e2_1_stencil()

        assert isinstance(stencil, e2_1.E2_1Stencil)
        expected = np.zeros(4)
        np.testing.assert_allclose(stencil.alpha, expected, rtol=1e-14)


class TestE2_1Constants:
    """Test that stencil constants are correct."""

    def test_constants(self):
        """Test stencil constants match specification."""
        assert e2_1.P == 1  # Order of accuracy
        assert e2_1.R == 4  # Number of rows
        assert e2_1.T == 5  # Tail length
        assert e2_1.X == 0  # Extra parameter
