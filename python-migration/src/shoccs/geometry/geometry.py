"""
Geometric algorithms for cut-cell methods.

This module provides algorithms for computing boundary intersections in
cut-cell grids, including ray-casting through grid lines.
"""

from typing import List
import numpy as np
import numpy.typing as npt

from .mesh import CartesianMesh, BoundaryPoint
from .shapes import Ray, Shape, Hit


def cast_ray_through_grid(
    mesh: CartesianMesh,
    shapes: List[Shape],
    direction: int
) -> List[BoundaryPoint]:
    """
    Cast rays through grid to find boundary intersections.

    This function shoots rays along grid lines in the specified direction,
    testing for intersections with all provided shapes. For each intersection,
    it computes the normalized distance psi and creates a BoundaryPoint.

    The psi value represents the normalized distance from the grid point
    before the intersection to the boundary, in the range [0, 1].

    Args:
        mesh: CartesianMesh defining the computational grid
        shapes: List of Shape objects to test for intersections
        direction: Ray direction (0=x, 1=y, 2=z)

    Returns:
        List of BoundaryPoint objects representing all intersections found

    Examples:
        >>> mesh = CartesianMesh(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        >>> sphere = Sphere(center=np.array([0.5, 0.5, 0.5]), radius=0.2)
        >>> boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)
        >>> len(boundary_points) > 0
        True

    Notes:
        - Rays are cast from grid lines perpendicular to the specified direction
        - Only the first intersection along each ray is recorded
        - psi is computed as the fractional distance between grid points
    """
    if direction not in [0, 1, 2]:
        raise ValueError("Direction must be 0 (x), 1 (y), or 2 (z)")

    if not shapes:
        return []

    boundary_points = []

    # Get coordinate arrays
    x, y, z = mesh.coordinates()
    coords = [x, y, z]

    # Determine grid spacing in the ray direction
    spacings = [mesh.dx, mesh.dy, mesh.dz]
    ray_spacing = spacings[direction]

    # Get the perpendicular directions
    perp_dirs = [i for i in range(3) if i != direction]
    perp_coord1 = coords[perp_dirs[0]]
    perp_coord2 = coords[perp_dirs[1]]

    # Direction vector for rays
    ray_direction = np.zeros(3, dtype=np.float64)
    ray_direction[direction] = 1.0

    # Determine ray start position (before the grid in the ray direction)
    ray_start_coord = coords[direction][0] - ray_spacing

    # Iterate over all grid lines perpendicular to the ray direction
    for i, c1 in enumerate(perp_coord1):
        for j, c2 in enumerate(perp_coord2):
            # Construct ray origin
            origin = np.zeros(3, dtype=np.float64)
            origin[perp_dirs[0]] = c1
            origin[perp_dirs[1]] = c2
            origin[direction] = ray_start_coord

            # Create ray
            ray = Ray(origin=origin, direction=ray_direction.copy())

            # Find all intersections with all shapes
            hits: List[tuple[Hit, int]] = []  # (hit, shape_id)
            for shape_id, shape in enumerate(shapes):
                hit = shape.intersect(ray)
                if hit is not None:
                    hits.append((hit, shape_id))

            # Sort hits by distance (closest first)
            hits.sort(key=lambda h: h[0].t)

            # Process each hit
            for hit, shape_id in hits:
                # Determine which grid cell the intersection is in
                # This is the index of the grid point before the intersection
                intersection_coord = hit.position[direction]

                # Find the grid cell index
                grid_idx = None
                for k in range(len(coords[direction]) - 1):
                    if coords[direction][k] <= intersection_coord <= coords[direction][k + 1]:
                        grid_idx = k
                        break

                if grid_idx is None:
                    # Intersection is outside the grid
                    continue

                # Compute psi: normalized distance from grid_idx to intersection
                grid_point_before = coords[direction][grid_idx]
                grid_point_after = coords[direction][grid_idx + 1]
                cell_width = grid_point_after - grid_point_before

                psi = (intersection_coord - grid_point_before) / cell_width

                # Clamp psi to [0, 1] to handle numerical errors
                psi = np.clip(psi, 0.0, 1.0)

                # Construct solid_coord tuple based on direction
                solid_coord = [0, 0, 0]
                solid_coord[perp_dirs[0]] = i
                solid_coord[perp_dirs[1]] = j
                solid_coord[direction] = grid_idx

                # Create BoundaryPoint
                bp = BoundaryPoint(
                    position=tuple(hit.position),
                    psi=float(psi),
                    solid_coord=tuple(solid_coord),
                    shape_id=shape_id,
                    ray_outside=hit.ray_outside
                )
                boundary_points.append(bp)

    return boundary_points


def find_boundary_intersections(
    mesh: CartesianMesh,
    shapes: List[Shape]
) -> dict[int, List[BoundaryPoint]]:
    """
    Find all boundary intersections in all three coordinate directions.

    This is a convenience function that calls cast_ray_through_grid for
    each coordinate direction and organizes the results by direction.

    Args:
        mesh: CartesianMesh defining the computational grid
        shapes: List of Shape objects to test for intersections

    Returns:
        Dictionary mapping direction (0, 1, 2) to list of BoundaryPoints

    Examples:
        >>> mesh = CartesianMesh(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        >>> sphere = Sphere(center=np.array([0.5, 0.5, 0.5]), radius=0.2)
        >>> intersections = find_boundary_intersections(mesh, [sphere])
        >>> len(intersections)
        3
        >>> all(direction in intersections for direction in [0, 1, 2])
        True
    """
    return {
        direction: cast_ray_through_grid(mesh, shapes, direction)
        for direction in [0, 1, 2]
    }
