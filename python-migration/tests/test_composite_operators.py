"""
Tests for composite operators (Gradient and Laplacian).

These tests verify that the gradient and Laplacian operators,
built by composition from derivative operators, work correctly
on polynomial test functions.
"""

import pytest
import numpy as np
from shoccs.geometry import CartesianMesh
from shoccs.fields import ScalarField, VectorField
from shoccs.stencils import centered_diff_1st_order2, centered_diff_2nd_order2
from shoccs.operators import (
    create_derivative_operator,
    create_gradient_operator,
    create_laplacian_operator
)


class TestGradientOperator:
    """Tests for gradient operator built by composition."""

    def test_gradient_on_linear_function(self):
        """∇(2x + 3y + 4z) = (2, 3, 4) - interior points."""
        # Setup
        nx, ny, nz = 15, 15, 15
        mesh = CartesianMesh(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        grad = create_gradient_operator(mesh, centered_diff_1st_order2)

        # Create function u = 2x + 3y + 4z
        x, y, z = mesh.coordinates()
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = ScalarField(D=2.0 * X + 3.0 * Y + 4.0 * Z)

        # Compute gradient
        grad_u = grad(u)

        # Check interior points: ∇u = (2, 3, 4)
        assert np.allclose(grad_u.x.D[1:-1, 1:-1, 1:-1], 2.0, rtol=1e-12, atol=1e-12)
        assert np.allclose(grad_u.y.D[1:-1, 1:-1, 1:-1], 3.0, rtol=1e-12, atol=1e-12)
        assert np.allclose(grad_u.z.D[1:-1, 1:-1, 1:-1], 4.0, rtol=1e-12, atol=1e-12)

    def test_gradient_on_polynomial(self):
        """∇(x² + y² + z²) = (2x, 2y, 2z) - interior points."""
        # Setup
        nx, ny, nz = 20, 20, 20
        mesh = CartesianMesh(nx, ny, nz, 0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi)
        grad = create_gradient_operator(mesh, centered_diff_1st_order2)

        # Create function u = x² + y² + z²
        x, y, z = mesh.coordinates()
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = ScalarField(D=X**2 + Y**2 + Z**2)

        # Compute gradient
        grad_u = grad(u)

        # Check interior points: ∇u = (2x, 2y, 2z)
        expected_x = 2.0 * X
        expected_y = 2.0 * Y
        expected_z = 2.0 * Z

        assert np.allclose(grad_u.x.D[1:-1, 1:-1, 1:-1], expected_x[1:-1, 1:-1, 1:-1],
                          rtol=1e-10, atol=1e-10)
        assert np.allclose(grad_u.y.D[1:-1, 1:-1, 1:-1], expected_y[1:-1, 1:-1, 1:-1],
                          rtol=1e-10, atol=1e-10)
        assert np.allclose(grad_u.z.D[1:-1, 1:-1, 1:-1], expected_z[1:-1, 1:-1, 1:-1],
                          rtol=1e-10, atol=1e-10)

    def test_gradient_is_vectorfield(self):
        """Verify gradient returns a VectorField."""
        # Setup
        nx, ny, nz = 10, 10, 10
        mesh = CartesianMesh(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        grad = create_gradient_operator(mesh, centered_diff_1st_order2)

        # Create test function
        x, y, z = mesh.coordinates()
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = ScalarField(D=X + Y + Z)

        # Compute gradient
        grad_u = grad(u)

        # Verify it's a VectorField with correct components
        assert isinstance(grad_u, VectorField)
        assert isinstance(grad_u.x, ScalarField)
        assert isinstance(grad_u.y, ScalarField)
        assert isinstance(grad_u.z, ScalarField)
        assert grad_u.x.D.shape == (nx, ny, nz)
        assert grad_u.y.D.shape == (nx, ny, nz)
        assert grad_u.z.D.shape == (nx, ny, nz)


class TestLaplacianOperator:
    """Tests for Laplacian operator built by composition."""

    def test_laplacian_on_polynomial_2d(self):
        """∇²(x² + y²) = 4 in 2D - interior points."""
        # Setup
        nx, ny, nz = 20, 20, 10
        mesh = CartesianMesh(nx, ny, nz, 0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 0.0, 1.0)
        laplacian = create_laplacian_operator(mesh, centered_diff_2nd_order2)

        # Create function u = x² + y²
        x, y, z = mesh.coordinates()
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = ScalarField(D=X**2 + Y**2)

        # Compute Laplacian
        lap_u = laplacian(u)

        # Check interior points: ∇²u = 2 + 2 = 4
        expected = 4.0
        assert np.allclose(lap_u.D[1:-1, 1:-1, :], expected, rtol=1e-10, atol=1e-10)

    def test_laplacian_on_polynomial_3d(self):
        """∇²(x² + y² + z²) = 6 in 3D - interior points."""
        # Setup
        nx, ny, nz = 15, 15, 15
        mesh = CartesianMesh(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        laplacian = create_laplacian_operator(mesh, centered_diff_2nd_order2)

        # Create function u = x² + y² + z²
        x, y, z = mesh.coordinates()
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = ScalarField(D=X**2 + Y**2 + Z**2)

        # Compute Laplacian
        lap_u = laplacian(u)

        # Check interior points: ∇²u = 2 + 2 + 2 = 6
        expected = 6.0
        assert np.allclose(lap_u.D[1:-1, 1:-1, 1:-1], expected, rtol=1e-10, atol=1e-10)

    def test_laplacian_returns_scalarfield(self):
        """Verify Laplacian returns a ScalarField."""
        # Setup
        nx, ny, nz = 10, 10, 10
        mesh = CartesianMesh(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        laplacian = create_laplacian_operator(mesh, centered_diff_2nd_order2)

        # Create test function
        x, y, z = mesh.coordinates()
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = ScalarField(D=X**2 + Y**2 + Z**2)

        # Compute Laplacian
        lap_u = laplacian(u)

        # Verify it's a ScalarField
        assert isinstance(lap_u, ScalarField)
        assert lap_u.D.shape == (nx, ny, nz)


class TestCompositionPrinciple:
    """Tests verifying composition principle works correctly."""

    def test_gradient_components_are_derivatives(self):
        """Verify gradient.x == Dx(u), gradient.y == Dy(u), etc."""
        # Setup
        nx, ny, nz = 15, 15, 15
        mesh = CartesianMesh(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

        # Create operators
        Dx = create_derivative_operator(mesh, centered_diff_1st_order2, 0)
        Dy = create_derivative_operator(mesh, centered_diff_1st_order2, 1)
        Dz = create_derivative_operator(mesh, centered_diff_1st_order2, 2)
        grad = create_gradient_operator(mesh, centered_diff_1st_order2)

        # Create test function
        x, y, z = mesh.coordinates()
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = ScalarField(D=X**3 + Y**3 + Z**3)

        # Compute derivatives
        dudx = Dx(u)
        dudy = Dy(u)
        dudz = Dz(u)
        grad_u = grad(u)

        # Verify composition - gradient components match individual derivatives
        assert np.allclose(grad_u.x.D, dudx.D, rtol=1e-12, atol=1e-12)
        assert np.allclose(grad_u.y.D, dudy.D, rtol=1e-12, atol=1e-12)
        assert np.allclose(grad_u.z.D, dudz.D, rtol=1e-12, atol=1e-12)

    def test_laplacian_is_sum_of_second_derivatives(self):
        """Verify laplacian == Dxx(u) + Dyy(u) + Dzz(u)."""
        # Setup
        nx, ny, nz = 15, 15, 15
        mesh = CartesianMesh(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

        # Create operators
        Dxx = create_derivative_operator(mesh, centered_diff_2nd_order2, 0, derivative_order=2)
        Dyy = create_derivative_operator(mesh, centered_diff_2nd_order2, 1, derivative_order=2)
        Dzz = create_derivative_operator(mesh, centered_diff_2nd_order2, 2, derivative_order=2)
        laplacian = create_laplacian_operator(mesh, centered_diff_2nd_order2)

        # Create test function
        x, y, z = mesh.coordinates()
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = ScalarField(D=X**3 + Y**3 + Z**3)

        # Compute second derivatives
        d2udx2 = Dxx(u)
        d2udy2 = Dyy(u)
        d2udz2 = Dzz(u)
        lap_u = laplacian(u)

        # Verify composition - Laplacian is sum of second derivatives
        expected = d2udx2.D + d2udy2.D + d2udz2.D
        assert np.allclose(lap_u.D, expected, rtol=1e-12, atol=1e-12)

    def test_no_code_duplication(self):
        """Verify operators are thin wrappers with minimal code."""
        # This is a documentation test - gradient and laplacian
        # should be simple dataclasses with __call__ methods
        from shoccs.operators.gradient import GradientOperator
        from shoccs.operators.laplacian import LaplacianOperator

        # Check that they're dataclasses (composition pattern)
        assert hasattr(GradientOperator, '__dataclass_fields__')
        assert hasattr(LaplacianOperator, '__dataclass_fields__')

        # Check they have the expected component operators
        assert 'Dx' in GradientOperator.__dataclass_fields__
        assert 'Dy' in GradientOperator.__dataclass_fields__
        assert 'Dz' in GradientOperator.__dataclass_fields__

        assert 'Dxx' in LaplacianOperator.__dataclass_fields__
        assert 'Dyy' in LaplacianOperator.__dataclass_fields__
        assert 'Dzz' in LaplacianOperator.__dataclass_fields__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
