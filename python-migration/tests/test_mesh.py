"""
Unit tests for mesh and geometry structures.
"""

import pytest
import numpy as np
from shoccs.geometry import CartesianMesh, BoundaryPoint


class TestCartesianMesh:
    """Tests for CartesianMesh class."""

    def test_mesh_creation(self):
        """Test basic mesh creation."""
        mesh = CartesianMesh(
            nx=10, ny=10, nz=10,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )
        assert mesh.nx == 10
        assert mesh.ny == 10
        assert mesh.nz == 10
        assert mesh.xmin == 0.0
        assert mesh.xmax == 1.0

    def test_mesh_shape(self):
        """Test mesh shape property."""
        mesh = CartesianMesh(
            nx=5, ny=10, nz=15,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=2.0,
            zmin=0.0, zmax=3.0
        )
        assert mesh.shape == (5, 10, 15)

    def test_grid_spacing_uniform(self):
        """Test grid spacing calculation for uniform mesh."""
        mesh = CartesianMesh(
            nx=11, ny=11, nz=11,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )
        assert np.isclose(mesh.dx, 0.1)
        assert np.isclose(mesh.dy, 0.1)
        assert np.isclose(mesh.dz, 0.1)

    def test_grid_spacing_non_uniform(self):
        """Test grid spacing for non-uniform domain."""
        mesh = CartesianMesh(
            nx=11, ny=21, nz=31,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=2.0,
            zmin=0.0, zmax=3.0
        )
        assert np.isclose(mesh.dx, 0.1)
        assert np.isclose(mesh.dy, 0.1)
        assert np.isclose(mesh.dz, 0.1)

    def test_coordinate_generation(self):
        """Test coordinate array generation."""
        mesh = CartesianMesh(
            nx=3, ny=3, nz=3,
            xmin=0.0, xmax=2.0,
            ymin=0.0, ymax=2.0,
            zmin=0.0, zmax=2.0
        )
        x, y, z = mesh.coordinates()

        # Check shapes
        assert x.shape == (3,)
        assert y.shape == (3,)
        assert z.shape == (3,)

        # Check values
        np.testing.assert_array_almost_equal(x, [0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(y, [0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(z, [0.0, 1.0, 2.0])

    def test_coordinate_values_known(self):
        """Test coordinate values against known reference."""
        mesh = CartesianMesh(
            nx=5, ny=5, nz=5,
            xmin=-1.0, xmax=1.0,
            ymin=-2.0, ymax=2.0,
            zmin=-3.0, zmax=3.0
        )
        x, y, z = mesh.coordinates()

        # Check first and last values
        assert np.isclose(x[0], -1.0)
        assert np.isclose(x[-1], 1.0)
        assert np.isclose(y[0], -2.0)
        assert np.isclose(y[-1], 2.0)
        assert np.isclose(z[0], -3.0)
        assert np.isclose(z[-1], 3.0)

        # Check middle value
        assert np.isclose(x[2], 0.0)
        assert np.isclose(y[2], 0.0)
        assert np.isclose(z[2], 0.0)

    def test_mesh_size(self):
        """Test total mesh size calculation."""
        mesh = CartesianMesh(
            nx=10, ny=20, nz=30,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )
        assert mesh.size() == 10 * 20 * 30

    def test_mesh_volume(self):
        """Test domain volume calculation."""
        mesh = CartesianMesh(
            nx=10, ny=10, nz=10,
            xmin=0.0, xmax=2.0,
            ymin=0.0, ymax=3.0,
            zmin=0.0, zmax=4.0
        )
        assert np.isclose(mesh.volume(), 24.0)  # 2 * 3 * 4

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            CartesianMesh(
                nx=0, ny=10, nz=10,
                xmin=0.0, xmax=1.0,
                ymin=0.0, ymax=1.0,
                zmin=0.0, zmax=1.0
            )

        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            CartesianMesh(
                nx=-5, ny=10, nz=10,
                xmin=0.0, xmax=1.0,
                ymin=0.0, ymax=1.0,
                zmin=0.0, zmax=1.0
            )

    def test_invalid_bounds(self):
        """Test that invalid bounds raise ValueError."""
        with pytest.raises(ValueError, match="xmin must be less than xmax"):
            CartesianMesh(
                nx=10, ny=10, nz=10,
                xmin=1.0, xmax=0.0,
                ymin=0.0, ymax=1.0,
                zmin=0.0, zmax=1.0
            )

        with pytest.raises(ValueError, match="ymin must be less than ymax"):
            CartesianMesh(
                nx=10, ny=10, nz=10,
                xmin=0.0, xmax=1.0,
                ymin=2.0, ymax=1.0,
                zmin=0.0, zmax=1.0
            )

        with pytest.raises(ValueError, match="zmin must be less than zmax"):
            CartesianMesh(
                nx=10, ny=10, nz=10,
                xmin=0.0, xmax=1.0,
                ymin=0.0, ymax=1.0,
                zmin=1.0, zmax=1.0
            )

    def test_coordinate_monotonicity(self):
        """Test that coordinates are monotonically increasing."""
        mesh = CartesianMesh(
            nx=20, ny=20, nz=20,
            xmin=-5.0, xmax=5.0,
            ymin=-10.0, ymax=10.0,
            zmin=-15.0, zmax=15.0
        )
        x, y, z = mesh.coordinates()

        # Check monotonicity
        assert np.all(np.diff(x) > 0)
        assert np.all(np.diff(y) > 0)
        assert np.all(np.diff(z) > 0)


class TestBoundaryPoint:
    """Tests for BoundaryPoint class."""

    def test_boundary_point_creation(self):
        """Test basic boundary point creation."""
        bp = BoundaryPoint(
            position=(0.5, 0.5, 0.5),
            psi=0.25,
            solid_coord=(5, 5, 5),
            shape_id=0,
            ray_outside=True
        )
        assert bp.position == (0.5, 0.5, 0.5)
        assert bp.psi == 0.25
        assert bp.solid_coord == (5, 5, 5)
        assert bp.shape_id == 0
        assert bp.ray_outside is True

    def test_boundary_point_attributes(self):
        """Test all boundary point attributes."""
        bp = BoundaryPoint(
            position=(1.0, 2.0, 3.0),
            psi=0.75,
            solid_coord=(10, 20, 30),
            shape_id=2,
            ray_outside=False
        )
        assert bp.position[0] == 1.0
        assert bp.position[1] == 2.0
        assert bp.position[2] == 3.0
        assert bp.psi == 0.75
        assert bp.solid_coord[0] == 10
        assert bp.solid_coord[1] == 20
        assert bp.solid_coord[2] == 30
        assert bp.shape_id == 2
        assert bp.ray_outside is False

    def test_boundary_point_psi_range(self):
        """Test that psi must be in [0, 1] range."""
        # Valid psi values
        BoundaryPoint(
            position=(0.0, 0.0, 0.0),
            psi=0.0,
            solid_coord=(0, 0, 0),
            shape_id=0,
            ray_outside=True
        )
        BoundaryPoint(
            position=(0.0, 0.0, 0.0),
            psi=1.0,
            solid_coord=(0, 0, 0),
            shape_id=0,
            ray_outside=True
        )
        BoundaryPoint(
            position=(0.0, 0.0, 0.0),
            psi=0.5,
            solid_coord=(0, 0, 0),
            shape_id=0,
            ray_outside=True
        )

        # Invalid psi values
        with pytest.raises(ValueError, match="psi must be in range"):
            BoundaryPoint(
                position=(0.0, 0.0, 0.0),
                psi=-0.1,
                solid_coord=(0, 0, 0),
                shape_id=0,
                ray_outside=True
            )

        with pytest.raises(ValueError, match="psi must be in range"):
            BoundaryPoint(
                position=(0.0, 0.0, 0.0),
                psi=1.5,
                solid_coord=(0, 0, 0),
                shape_id=0,
                ray_outside=True
            )

    def test_invalid_position(self):
        """Test that invalid position raises ValueError."""
        with pytest.raises(ValueError, match="Position must be a 3-tuple"):
            BoundaryPoint(
                position=(0.0, 0.0),  # Only 2 components
                psi=0.5,
                solid_coord=(0, 0, 0),
                shape_id=0,
                ray_outside=True
            )

    def test_invalid_solid_coord(self):
        """Test that invalid solid_coord raises ValueError."""
        with pytest.raises(ValueError, match="Solid coordinate must be a 3-tuple"):
            BoundaryPoint(
                position=(0.0, 0.0, 0.0),
                psi=0.5,
                solid_coord=(0, 0),  # Only 2 components
                shape_id=0,
                ray_outside=True
            )

    def test_boundary_point_negative_coords(self):
        """Test boundary point with negative coordinates."""
        bp = BoundaryPoint(
            position=(-1.0, -2.0, -3.0),
            psi=0.5,
            solid_coord=(-5, -10, -15),
            shape_id=1,
            ray_outside=False
        )
        assert bp.position == (-1.0, -2.0, -3.0)
        assert bp.solid_coord == (-5, -10, -15)


class TestMeshIntegration:
    """Integration tests combining mesh and boundary points."""

    def test_mesh_and_boundary_point(self):
        """Test using mesh and boundary point together."""
        mesh = CartesianMesh(
            nx=10, ny=10, nz=10,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )

        # Create a boundary point within the mesh domain
        bp = BoundaryPoint(
            position=(0.5, 0.5, 0.5),
            psi=0.3,
            solid_coord=(5, 5, 5),
            shape_id=0,
            ray_outside=True
        )

        # Verify boundary point is within mesh bounds
        x, y, z = bp.position
        assert mesh.xmin <= x <= mesh.xmax
        assert mesh.ymin <= y <= mesh.ymax
        assert mesh.zmin <= z <= mesh.zmax

    def test_multiple_boundary_points(self):
        """Test creating multiple boundary points on a mesh."""
        mesh = CartesianMesh(
            nx=20, ny=20, nz=20,
            xmin=-1.0, xmax=1.0,
            ymin=-1.0, ymax=1.0,
            zmin=-1.0, zmax=1.0
        )

        # Create multiple boundary points
        boundary_points = [
            BoundaryPoint(
                position=(0.0, 0.0, 0.5),
                psi=0.2,
                solid_coord=(10, 10, 15),
                shape_id=0,
                ray_outside=True
            ),
            BoundaryPoint(
                position=(0.5, 0.0, 0.0),
                psi=0.4,
                solid_coord=(15, 10, 10),
                shape_id=0,
                ray_outside=False
            ),
            BoundaryPoint(
                position=(0.0, 0.5, 0.0),
                psi=0.6,
                solid_coord=(10, 15, 10),
                shape_id=1,
                ray_outside=True
            ),
        ]

        assert len(boundary_points) == 3
        assert boundary_points[0].shape_id == 0
        assert boundary_points[1].shape_id == 0
        assert boundary_points[2].shape_id == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
