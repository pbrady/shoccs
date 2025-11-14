"""Geometry module for SHOCCS."""

from .mesh import CartesianMesh, BoundaryPoint
from .shapes import Ray, Hit, Shape, Sphere, Rectangle
from .geometry import cast_ray_through_grid, find_boundary_intersections

__all__ = [
    "CartesianMesh",
    "BoundaryPoint",
    "Ray",
    "Hit",
    "Shape",
    "Sphere",
    "Rectangle",
    "cast_ray_through_grid",
    "find_boundary_intersections",
]
