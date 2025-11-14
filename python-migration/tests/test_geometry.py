"""
Unit tests for geometry ray-tracing and shape intersection.
"""

import pytest
import numpy as np
from shoccs.geometry import (
    Ray, Hit, Sphere, Rectangle, CartesianMesh,
    cast_ray_through_grid, find_boundary_intersections
)


class TestRay:
    """Tests for Ray class."""

    def test_ray_creation(self):
        """Test basic ray creation."""
        ray = Ray(
            origin=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0])
        )
        np.testing.assert_array_almost_equal(ray.origin, [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(ray.direction, [1.0, 0.0, 0.0])

    def test_ray_direction_normalized(self):
        """Test that ray direction is automatically normalized."""
        ray = Ray(
            origin=np.array([0.0, 0.0, 0.0]),
            direction=np.array([3.0, 4.0, 0.0])  # magnitude = 5
        )
        np.testing.assert_array_almost_equal(ray.direction, [0.6, 0.8, 0.0])
        assert np.isclose(np.linalg.norm(ray.direction), 1.0)

    def test_ray_invalid_origin(self):
        """Test that invalid origin raises ValueError."""
        with pytest.raises(ValueError, match="origin must be a 3D vector"):
            Ray(
                origin=np.array([0.0, 0.0]),
                direction=np.array([1.0, 0.0, 0.0])
            )

    def test_ray_invalid_direction(self):
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="direction must be a 3D vector"):
            Ray(
                origin=np.array([0.0, 0.0, 0.0]),
                direction=np.array([1.0, 0.0])
            )

    def test_ray_zero_direction(self):
        """Test that zero direction raises ValueError."""
        with pytest.raises(ValueError, match="direction cannot be zero"):
            Ray(
                origin=np.array([0.0, 0.0, 0.0]),
                direction=np.array([0.0, 0.0, 0.0])
            )


class TestHit:
    """Tests for Hit class."""

    def test_hit_creation(self):
        """Test basic hit creation."""
        hit = Hit(
            t=1.0,
            position=np.array([1.0, 0.0, 0.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            ray_outside=True
        )
        assert hit.t == 1.0
        np.testing.assert_array_almost_equal(hit.position, [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(hit.normal, [1.0, 0.0, 0.0])
        assert hit.ray_outside is True

    def test_hit_normal_normalized(self):
        """Test that hit normal is automatically normalized."""
        hit = Hit(
            t=1.0,
            position=np.array([1.0, 0.0, 0.0]),
            normal=np.array([3.0, 4.0, 0.0]),  # magnitude = 5
            ray_outside=True
        )
        np.testing.assert_array_almost_equal(hit.normal, [0.6, 0.8, 0.0])
        assert np.isclose(np.linalg.norm(hit.normal), 1.0)

    def test_hit_negative_t(self):
        """Test that negative t raises ValueError."""
        with pytest.raises(ValueError, match="t must be non-negative"):
            Hit(
                t=-1.0,
                position=np.array([1.0, 0.0, 0.0]),
                normal=np.array([1.0, 0.0, 0.0]),
                ray_outside=True
            )


class TestSphere:
    """Tests for Sphere class."""

    def test_sphere_creation(self):
        """Test basic sphere creation."""
        sphere = Sphere(
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0
        )
        np.testing.assert_array_almost_equal(sphere.center, [0.0, 0.0, 0.0])
        assert sphere.radius == 1.0

    def test_sphere_invalid_radius(self):
        """Test that invalid radius raises ValueError."""
        with pytest.raises(ValueError, match="radius must be positive"):
            Sphere(
                center=np.array([0.0, 0.0, 0.0]),
                radius=0.0
            )

        with pytest.raises(ValueError, match="radius must be positive"):
            Sphere(
                center=np.array([0.0, 0.0, 0.0]),
                radius=-1.0
            )

    def test_ray_sphere_intersection(self):
        """Ray should hit sphere at correct point."""
        sphere = Sphere(
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0
        )
        ray = Ray(
            origin=np.array([-2.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0])
        )
        hit = sphere.intersect(ray)

        assert hit is not None
        assert np.isclose(hit.t, 1.0)  # Hits at x=-1
        np.testing.assert_array_almost_equal(hit.position, [-1.0, 0.0, 0.0])

    def test_sphere_normal(self):
        """Normal should point radially outward."""
        sphere = Sphere(
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0
        )
        ray = Ray(
            origin=np.array([-2.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0])
        )
        hit = sphere.intersect(ray)

        assert hit is not None
        # Normal at x=-1 should point in -x direction
        np.testing.assert_array_almost_equal(hit.normal, [-1.0, 0.0, 0.0])
        # Normal should be unit length
        assert np.isclose(np.linalg.norm(hit.normal), 1.0)

    def test_ray_sphere_miss(self):
        """Ray should miss sphere."""
        sphere = Sphere(
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0
        )
        ray = Ray(
            origin=np.array([0.0, 5.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0])
        )
        hit = sphere.intersect(ray)

        assert hit is None

    def test_ray_sphere_tangent(self):
        """Ray should hit sphere tangentially."""
        sphere = Sphere(
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0
        )
        ray = Ray(
            origin=np.array([-2.0, 1.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0])
        )
        hit = sphere.intersect(ray)

        assert hit is not None
        assert np.isclose(hit.t, 2.0)  # Tangent at (0, 1, 0)
        np.testing.assert_array_almost_equal(hit.position, [0.0, 1.0, 0.0])

    def test_ray_from_inside_sphere(self):
        """Ray from inside sphere should hit exit point."""
        sphere = Sphere(
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0
        )
        ray = Ray(
            origin=np.array([0.0, 0.0, 0.0]),  # Center of sphere
            direction=np.array([1.0, 0.0, 0.0])
        )
        hit = sphere.intersect(ray)

        assert hit is not None
        assert np.isclose(hit.t, 1.0)
        np.testing.assert_array_almost_equal(hit.position, [1.0, 0.0, 0.0])
        assert hit.ray_outside is False

    def test_sphere_multiple_intersections(self):
        """Ray through sphere should return closest intersection."""
        sphere = Sphere(
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0
        )
        ray = Ray(
            origin=np.array([-3.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0])
        )
        hit = sphere.intersect(ray)

        # Should return entry point, not exit
        assert hit is not None
        assert np.isclose(hit.t, 2.0)  # Entry at x=-1
        assert hit.ray_outside is True


class TestRectangle:
    """Tests for Rectangle class."""

    def test_rectangle_creation(self):
        """Test basic rectangle creation."""
        rect = Rectangle(
            axis=2,  # XY plane (perpendicular to z)
            offset=0.0,
            bounds=((0.0, 1.0), (0.0, 1.0))
        )
        assert rect.axis == 2
        assert rect.offset == 0.0
        assert rect.bounds == ((0.0, 1.0), (0.0, 1.0))

    def test_rectangle_invalid_axis(self):
        """Test that invalid axis raises ValueError."""
        with pytest.raises(ValueError, match="Axis must be"):
            Rectangle(
                axis=3,
                offset=0.0,
                bounds=((0.0, 1.0), (0.0, 1.0))
            )

    def test_rectangle_invalid_bounds(self):
        """Test that invalid bounds raise ValueError."""
        with pytest.raises(ValueError, match="min must be less than max"):
            Rectangle(
                axis=2,
                offset=0.0,
                bounds=((1.0, 0.0), (0.0, 1.0))  # First bound inverted
            )

    def test_ray_rectangle_intersection(self):
        """Ray should hit rectangle at correct point."""
        rect = Rectangle(
            axis=2,  # XY plane at z=0
            offset=0.0,
            bounds=((0.0, 1.0), (0.0, 1.0))
        )
        ray = Ray(
            origin=np.array([0.5, 0.5, -1.0]),
            direction=np.array([0.0, 0.0, 1.0])
        )
        hit = rect.intersect(ray)

        assert hit is not None
        assert np.isclose(hit.t, 1.0)
        np.testing.assert_array_almost_equal(hit.position, [0.5, 0.5, 0.0])

    def test_rectangle_normal(self):
        """Normal should be perpendicular to rectangle plane."""
        rect = Rectangle(
            axis=2,  # XY plane at z=0
            offset=0.0,
            bounds=((0.0, 1.0), (0.0, 1.0))
        )
        ray = Ray(
            origin=np.array([0.5, 0.5, -1.0]),
            direction=np.array([0.0, 0.0, 1.0])
        )
        hit = rect.intersect(ray)

        assert hit is not None
        # Normal should point in -z direction (toward ray origin)
        np.testing.assert_array_almost_equal(hit.normal, [0.0, 0.0, -1.0])
        assert np.isclose(np.linalg.norm(hit.normal), 1.0)

    def test_ray_rectangle_miss(self):
        """Ray should miss rectangle."""
        rect = Rectangle(
            axis=2,  # XY plane at z=0
            offset=0.0,
            bounds=((0.0, 1.0), (0.0, 1.0))
        )
        ray = Ray(
            origin=np.array([2.0, 2.0, -1.0]),  # Outside bounds
            direction=np.array([0.0, 0.0, 1.0])
        )
        hit = rect.intersect(ray)

        assert hit is None

    def test_ray_parallel_to_rectangle(self):
        """Ray parallel to rectangle should not intersect."""
        rect = Rectangle(
            axis=2,  # XY plane at z=0
            offset=0.0,
            bounds=((0.0, 1.0), (0.0, 1.0))
        )
        ray = Ray(
            origin=np.array([0.5, 0.5, -1.0]),
            direction=np.array([1.0, 0.0, 0.0])  # Parallel to plane
        )
        hit = rect.intersect(ray)

        assert hit is None


class TestRayCasting:
    """Tests for ray-casting through grid."""

    def test_psi_computation(self):
        """Psi should be in [0, 1] range."""
        mesh = CartesianMesh(
            nx=10, ny=10, nz=10,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )
        sphere = Sphere(
            center=np.array([0.5, 0.5, 0.5]),
            radius=0.2
        )

        boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)

        # All psi values should be in [0, 1]
        for bp in boundary_points:
            assert 0.0 <= bp.psi <= 1.0, f"psi={bp.psi} out of range"

    def test_grid_ray_casting(self):
        """Should find all intersections in grid."""
        mesh = CartesianMesh(
            nx=10, ny=10, nz=10,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )
        sphere = Sphere(
            center=np.array([0.5, 0.5, 0.5]),
            radius=0.2
        )

        # Cast rays in x direction
        boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)

        # Should find some intersections
        assert len(boundary_points) > 0

        # All boundary points should be on the sphere surface
        for bp in boundary_points:
            pos = np.array(bp.position)
            dist_to_center = np.linalg.norm(pos - sphere.center)
            assert np.isclose(dist_to_center, sphere.radius, atol=1e-6)

    def test_ray_casting_all_directions(self):
        """Should find intersections in all three directions."""
        mesh = CartesianMesh(
            nx=10, ny=10, nz=10,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )
        sphere = Sphere(
            center=np.array([0.5, 0.5, 0.5]),
            radius=0.2
        )

        # Cast rays in all directions
        for direction in [0, 1, 2]:
            boundary_points = cast_ray_through_grid(mesh, [sphere], direction)
            assert len(boundary_points) > 0, f"No intersections in direction {direction}"

    def test_ray_casting_multiple_shapes(self):
        """Should find intersections with multiple shapes."""
        mesh = CartesianMesh(
            nx=20, ny=20, nz=20,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )
        sphere1 = Sphere(center=np.array([0.3, 0.5, 0.5]), radius=0.1)
        sphere2 = Sphere(center=np.array([0.7, 0.5, 0.5]), radius=0.1)

        boundary_points = cast_ray_through_grid(mesh, [sphere1, sphere2], direction=0)

        # Should find intersections with both spheres
        shape_ids = {bp.shape_id for bp in boundary_points}
        assert 0 in shape_ids
        assert 1 in shape_ids

    def test_ray_casting_no_shapes(self):
        """Should return empty list when no shapes provided."""
        mesh = CartesianMesh(
            nx=10, ny=10, nz=10,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )

        boundary_points = cast_ray_through_grid(mesh, [], direction=0)
        assert len(boundary_points) == 0

    def test_ray_casting_invalid_direction(self):
        """Should raise ValueError for invalid direction."""
        mesh = CartesianMesh(
            nx=10, ny=10, nz=10,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )
        sphere = Sphere(center=np.array([0.5, 0.5, 0.5]), radius=0.2)

        with pytest.raises(ValueError, match="Direction must be"):
            cast_ray_through_grid(mesh, [sphere], direction=3)

    def test_find_boundary_intersections(self):
        """Should find intersections in all directions."""
        mesh = CartesianMesh(
            nx=10, ny=10, nz=10,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )
        sphere = Sphere(center=np.array([0.5, 0.5, 0.5]), radius=0.2)

        intersections = find_boundary_intersections(mesh, [sphere])

        # Should have entries for all three directions
        assert len(intersections) == 3
        assert 0 in intersections
        assert 1 in intersections
        assert 2 in intersections

        # Each direction should have some boundary points
        for direction, bps in intersections.items():
            assert len(bps) > 0, f"No intersections in direction {direction}"

    def test_boundary_point_solid_coord(self):
        """Boundary points should have valid solid coordinates."""
        mesh = CartesianMesh(
            nx=10, ny=10, nz=10,
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=1.0
        )
        sphere = Sphere(center=np.array([0.5, 0.5, 0.5]), radius=0.2)

        boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)

        for bp in boundary_points:
            # All coordinates should be valid indices
            i, j, k = bp.solid_coord
            assert 0 <= i < mesh.nx
            assert 0 <= j < mesh.ny
            assert 0 <= k < mesh.nz


class TestRaySpherePrecision:
    """Tests for numerical precision of ray-sphere intersection."""

    def test_sphere_intersection_analytical(self):
        """Ray-sphere intersection should match analytical solution."""
        # Unit sphere at origin
        sphere = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=1.0)

        # Ray from (-2, 0, 0) in +x direction
        ray = Ray(
            origin=np.array([-2.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0])
        )
        hit = sphere.intersect(ray)

        # Analytical solution: hits at (-1, 0, 0)
        assert hit is not None
        assert np.isclose(hit.t, 1.0, atol=1e-10)
        np.testing.assert_array_almost_equal(hit.position, [-1.0, 0.0, 0.0], decimal=10)

    def test_sphere_normal_unit_length(self):
        """Sphere normal should always be unit length."""
        sphere = Sphere(center=np.array([1.0, 2.0, 3.0]), radius=2.5)

        # Test multiple rays
        test_rays = [
            Ray(np.array([5.0, 2.0, 3.0]), np.array([-1.0, 0.0, 0.0])),
            Ray(np.array([1.0, 6.0, 3.0]), np.array([0.0, -1.0, 0.0])),
            Ray(np.array([1.0, 2.0, 8.0]), np.array([0.0, 0.0, -1.0])),
        ]

        for ray in test_rays:
            hit = sphere.intersect(ray)
            if hit is not None:
                normal_length = np.linalg.norm(hit.normal)
                assert np.isclose(normal_length, 1.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
