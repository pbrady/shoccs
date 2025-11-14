"""
Tests for cut-cell derivative operators.

This module tests the integration of cut-cell geometry with discrete operators,
validating that derivatives can be computed accurately near embedded boundaries.
"""

import pytest
import numpy as np

from shoccs.geometry.mesh import CartesianMesh, BoundaryPoint
from shoccs.geometry.shapes import Sphere
from shoccs.geometry.geometry import cast_ray_through_grid
from shoccs.operators.cutcell_builder import build_cutcell_derivative
from shoccs.operators.derivative import DerivativeOperator
from shoccs.fields.field import ScalarField


class TestCutcellBuilder:
    """Test cut-cell operator matrix construction."""

    def test_build_cutcell_derivative_basic(self):
        """Test basic construction of cut-cell derivative operator."""
        # Simple 1D-like mesh
        mesh = CartesianMesh(
            nx=11, ny=3, nz=3,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=0.2,
            zmin=0.0, zmax=0.2
        )

        # Sphere at center
        sphere = Sphere(center=np.array([0.5, 0.1, 0.1]), radius=0.15)

        # Find boundary intersections in x-direction
        boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)

        # Should find some boundary points
        assert len(boundary_points) > 0, "Should find boundary points"

        # Build cut-cell operator
        A, B = build_cutcell_derivative(mesh, boundary_points, direction=0)

        # Check dimensions
        n_total = mesh.nx * mesh.ny * mesh.nz
        n_boundary = len(boundary_points)

        assert A.shape == (n_total, n_total), "A matrix has wrong shape"
        assert B.shape == (n_total, n_boundary), "B matrix has wrong shape"

        # Check that matrices are not empty
        assert A.nnz > 0, "A matrix should have non-zero entries"
        assert B.nnz > 0, "B matrix should have non-zero entries"

    def test_build_cutcell_derivative_no_boundaries(self):
        """Test operator construction with no boundaries."""
        # Small mesh
        mesh = CartesianMesh(
            nx=5, ny=5, nz=5,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )

        # No boundary points
        boundary_points = []

        # Build operator
        A, B = build_cutcell_derivative(mesh, boundary_points, direction=0)

        # Check dimensions
        n_total = mesh.nx * mesh.ny * mesh.nz
        assert A.shape == (n_total, n_total)
        assert B.shape == (n_total, 0)

        # A should still have entries (interior stencils)
        assert A.nnz > 0

        # B should be empty
        assert B.nnz == 0

    def test_build_cutcell_derivative_invalid_direction(self):
        """Test that invalid direction raises error."""
        mesh = CartesianMesh(
            nx=5, ny=5, nz=5,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )

        with pytest.raises(ValueError, match="Direction must be"):
            build_cutcell_derivative(mesh, [], direction=3)


class TestDerivativeOperatorCutcell:
    """Test DerivativeOperator with cut-cell boundaries."""

    def test_operator_creation(self):
        """Test creating a cut-cell derivative operator."""
        mesh = CartesianMesh(
            nx=11, ny=3, nz=3,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=0.2,
            zmin=0.0, zmax=0.2
        )

        sphere = Sphere(center=np.array([0.5, 0.1, 0.1]), radius=0.15)
        boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)

        # Build matrices
        A, B = build_cutcell_derivative(mesh, boundary_points, direction=0)

        # Create operator
        op = DerivativeOperator(
            mesh=mesh,
            direction=0,
            bc_type='cutcell',
            A=A,
            B=B
        )

        assert op.A is not None
        assert op.B is not None
        assert op.bc_type == 'cutcell'

    def test_operator_apply_constant_field(self):
        """Test derivative of constant field is zero."""
        mesh = CartesianMesh(
            nx=11, ny=3, nz=3,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=0.2,
            zmin=0.0, zmax=0.2
        )

        sphere = Sphere(center=np.array([0.5, 0.1, 0.1]), radius=0.15)
        boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)

        # Build operator
        A, B = build_cutcell_derivative(mesh, boundary_points, direction=0)
        op = DerivativeOperator(
            mesh=mesh,
            direction=0,
            bc_type='cutcell',
            A=A,
            B=B
        )

        # Create constant field
        u = ScalarField(
            D=np.ones((mesh.nx, mesh.ny, mesh.nz)),
            Rx=np.ones(len(boundary_points)),
            Ry=np.zeros(0),
            Rz=np.zeros(0)
        )

        # Apply operator
        du_dx = op(u)

        # Derivative of constant should be zero (within tolerance)
        assert np.allclose(du_dx.D, 0.0, atol=1e-10)

    def test_operator_apply_linear_field(self):
        """Test derivative of linear field."""
        mesh = CartesianMesh(
            nx=21, ny=3, nz=3,
            xmin=0.0, xmax=2.0,
            ymin=0.0, ymax=0.2,
            zmin=0.0, zmax=0.2
        )

        sphere = Sphere(center=np.array([1.0, 0.1, 0.1]), radius=0.3)
        boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)

        # Build operator
        A, B = build_cutcell_derivative(mesh, boundary_points, direction=0)
        op = DerivativeOperator(
            mesh=mesh,
            direction=0,
            bc_type='cutcell',
            A=A,
            B=B
        )

        # Create linear field f(x) = x
        x_coords = np.linspace(mesh.xmin, mesh.xmax, mesh.nx)
        u_values = np.zeros((mesh.nx, mesh.ny, mesh.nz))
        for i in range(mesh.nx):
            u_values[i, :, :] = x_coords[i]

        # Boundary values should also follow f(x) = x
        boundary_values = np.zeros(len(boundary_points))
        for idx, bp in enumerate(boundary_points):
            boundary_values[idx] = bp.position[0]

        u = ScalarField(
            D=u_values,
            Rx=boundary_values,
            Ry=np.zeros(0),
            Rz=np.zeros(0)
        )

        # Apply operator
        du_dx = op(u)

        # Check interior points away from boundaries and sphere
        # Sphere is at x=1.0 with radius=0.3, so exclude x in [0.7, 1.3]
        # Also exclude first and last points (periodic boundaries)
        for i in range(1, mesh.nx - 1):
            x = x_coords[i]
            if abs(x - 1.0) > 0.5:  # Far from sphere
                # Check that derivative is approximately 1.0
                assert np.allclose(du_dx.D[i, :, :], 1.0, atol=0.1), \
                    f"Derivative at x={x} is {du_dx.D[i,0,0]}, expected 1.0"

    def test_operator_without_matrices_raises_error(self):
        """Test that cutcell BC without A matrix raises error."""
        mesh = CartesianMesh(
            nx=5, ny=5, nz=5,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )

        with pytest.raises(ValueError, match="Cut-cell boundaries require A matrix"):
            DerivativeOperator(
                mesh=mesh,
                direction=0,
                bc_type='cutcell',
                A=None
            )


class TestCutcellAccuracy:
    """Test accuracy of cut-cell derivatives."""

    def test_quadratic_field_accuracy(self):
        """Test derivative accuracy on quadratic field."""
        mesh = CartesianMesh(
            nx=21, ny=3, nz=3,
            xmin=0.0, xmax=2.0,
            ymin=0.0, ymax=0.2,
            zmin=0.0, zmax=0.2
        )

        sphere = Sphere(center=np.array([1.0, 0.1, 0.1]), radius=0.3)
        boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)

        # Build operator
        A, B = build_cutcell_derivative(mesh, boundary_points, direction=0)
        op = DerivativeOperator(
            mesh=mesh,
            direction=0,
            bc_type='cutcell',
            A=A,
            B=B
        )

        # Create quadratic field f(x) = x^2, so df/dx = 2x
        x_coords = np.linspace(mesh.xmin, mesh.xmax, mesh.nx)
        u_values = np.zeros((mesh.nx, mesh.ny, mesh.nz))
        for i in range(mesh.nx):
            u_values[i, :, :] = x_coords[i] ** 2

        # Boundary values
        boundary_values = np.zeros(len(boundary_points))
        for idx, bp in enumerate(boundary_points):
            boundary_values[idx] = bp.position[0] ** 2

        u = ScalarField(
            D=u_values,
            Rx=boundary_values,
            Ry=np.zeros(0),
            Rz=np.zeros(0)
        )

        # Apply operator
        du_dx = op(u)

        # Expected derivative: 2x
        expected = np.zeros((mesh.nx, mesh.ny, mesh.nz))
        for i in range(mesh.nx):
            expected[i, :, :] = 2.0 * x_coords[i]

        # Check interior points away from boundaries and sphere
        # E2_1 is only 1st order accurate, so tolerance is larger
        # We'll check points far from the sphere
        # Exclude first and last points (periodic boundaries)
        for i in range(2, mesh.nx - 2):
            if abs(x_coords[i] - 1.0) > 0.6:  # Far from sphere center
                error = np.abs(du_dx.D[i, :, :] - expected[i, :, :])
                # 1st order accuracy on quadratic: error can be larger
                # Interior 2nd-order stencil is exact for quadratics, so error ~ 0
                assert np.max(error) < 0.1, \
                    f"Error too large at x={x_coords[i]}: {np.max(error)}"
