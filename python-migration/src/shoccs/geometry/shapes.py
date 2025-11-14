"""
Shape interfaces and implementations for embedded boundary geometry.

This module provides the basic infrastructure for ray-tracing algorithms used
in cut-cell methods. It defines shapes (like spheres and rectangles) and their
intersection algorithms with rays.
"""

from dataclasses import dataclass
from typing import Protocol, Optional
import numpy as np
import numpy.typing as npt


@dataclass
class Ray:
    """
    Simple ray for intersection tests.

    A ray is defined by an origin point and a direction vector. It represents
    a semi-infinite line starting at the origin and extending in the direction.

    Attributes:
        origin: Starting point of the ray (x, y, z)
        direction: Unit direction vector of the ray

    Examples:
        >>> ray = Ray(origin=np.array([0.0, 0.0, 0.0]),
        ...           direction=np.array([1.0, 0.0, 0.0]))
        >>> ray.origin
        array([0., 0., 0.])
        >>> ray.direction
        array([1., 0., 0.])
    """
    origin: npt.NDArray[np.float64]
    direction: npt.NDArray[np.float64]

    def __post_init__(self):
        """Validate and normalize the ray."""
        if self.origin.shape != (3,):
            raise ValueError("Ray origin must be a 3D vector")
        if self.direction.shape != (3,):
            raise ValueError("Ray direction must be a 3D vector")

        # Normalize the direction vector
        norm = np.linalg.norm(self.direction)
        if norm < 1e-10:
            raise ValueError("Ray direction cannot be zero")
        self.direction = self.direction / norm


@dataclass
class Hit:
    """
    Ray-shape intersection result.

    Represents a successful intersection between a ray and a shape, including
    the distance along the ray, the intersection point, and the surface normal.

    Attributes:
        t: Distance along ray to intersection (must be positive)
        position: 3D coordinates of intersection point
        normal: Unit surface normal at intersection
        ray_outside: True if ray originated from outside the shape

    Examples:
        >>> hit = Hit(t=1.0,
        ...           position=np.array([1.0, 0.0, 0.0]),
        ...           normal=np.array([1.0, 0.0, 0.0]),
        ...           ray_outside=True)
        >>> hit.t
        1.0
    """
    t: float
    position: npt.NDArray[np.float64]
    normal: npt.NDArray[np.float64]
    ray_outside: bool

    def __post_init__(self):
        """Validate the hit parameters."""
        if self.t < 0:
            raise ValueError("Hit distance t must be non-negative")
        if self.position.shape != (3,):
            raise ValueError("Hit position must be a 3D vector")
        if self.normal.shape != (3,):
            raise ValueError("Hit normal must be a 3D vector")

        # Ensure normal is unit length
        norm = np.linalg.norm(self.normal)
        if norm < 1e-10:
            raise ValueError("Normal cannot be zero")
        self.normal = self.normal / norm


class Shape(Protocol):
    """
    Protocol for shapes (duck typing).

    Any class that implements the intersect method can be used as a Shape.
    This allows for flexible addition of new shape types without inheritance.
    """

    def intersect(self, ray: Ray) -> Optional[Hit]:
        """
        Find ray-shape intersection.

        Args:
            ray: The ray to test for intersection

        Returns:
            Hit object if intersection exists, None otherwise.
            If multiple intersections exist, returns the closest one (smallest t > 0).
        """
        ...


class Sphere:
    """
    Sphere defined by center and radius.

    Implements the Shape protocol through ray-sphere intersection using
    the quadratic formula method.

    Attributes:
        center: Center point of the sphere (x, y, z)
        radius: Radius of the sphere (must be positive)

    Examples:
        >>> sphere = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=1.0)
        >>> ray = Ray(origin=np.array([-2.0, 0.0, 0.0]),
        ...           direction=np.array([1.0, 0.0, 0.0]))
        >>> hit = sphere.intersect(ray)
        >>> hit.t
        1.0
    """

    def __init__(self, center: npt.NDArray[np.float64], radius: float):
        """
        Initialize sphere.

        Args:
            center: Center point (x, y, z)
            radius: Sphere radius (must be positive)

        Raises:
            ValueError: If radius is not positive or center is not 3D
        """
        if not isinstance(center, np.ndarray):
            center = np.array(center, dtype=np.float64)
        if center.shape != (3,):
            raise ValueError("Sphere center must be a 3D vector")
        if radius <= 0:
            raise ValueError("Sphere radius must be positive")

        self.center = center.astype(np.float64)
        self.radius = float(radius)

    def intersect(self, ray: Ray) -> Optional[Hit]:
        """
        Ray-sphere intersection using quadratic formula.

        Solves: |origin + t*direction - center|^2 = radius^2
        This gives a quadratic equation: at^2 + bt + c = 0

        Args:
            ray: Ray to test for intersection

        Returns:
            Hit object for closest intersection with t > 0, or None if no intersection
        """
        # Vector from ray origin to sphere center
        oc = ray.origin - self.center

        # Quadratic equation coefficients
        # a = |direction|^2 = 1 (direction is normalized)
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius

        # Discriminant
        discriminant = b * b - 4 * a * c

        # No intersection if discriminant is negative
        if discriminant < 0:
            return None

        # Find the two potential intersection points
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # Choose the closest positive t
        t = None
        if t1 > 1e-10:  # Use small epsilon to avoid numerical issues
            t = t1
            ray_outside = True
        elif t2 > 1e-10:
            t = t2
            ray_outside = False
        else:
            # Both intersections are behind the ray origin
            return None

        # Compute intersection point and normal
        position = ray.origin + t * ray.direction
        normal = (position - self.center) / self.radius

        # Normal points outward from sphere center
        # If ray is inside (t2 case), we might want to flip it depending on convention
        if not ray_outside:
            normal = -normal

        return Hit(
            t=t,
            position=position,
            normal=normal,
            ray_outside=ray_outside
        )


class Rectangle:
    """
    Axis-aligned rectangle (2D in 3D space).

    Represents a rectangular region aligned with coordinate axes, lying in
    a plane perpendicular to one of the coordinate directions.

    Attributes:
        axis: Axis perpendicular to rectangle plane (0=x, 1=y, 2=z)
        offset: Position along the perpendicular axis
        bounds: ((min1, max1), (min2, max2)) bounds in the two parallel directions

    Examples:
        >>> # Rectangle in XY plane at z=0, from x=[0,1], y=[0,1]
        >>> rect = Rectangle(axis=2, offset=0.0, bounds=((0.0, 1.0), (0.0, 1.0)))
        >>> ray = Ray(origin=np.array([0.5, 0.5, -1.0]),
        ...           direction=np.array([0.0, 0.0, 1.0]))
        >>> hit = rect.intersect(ray)
        >>> hit.t
        1.0
    """

    def __init__(self, axis: int, offset: float,
                 bounds: tuple[tuple[float, float], tuple[float, float]]):
        """
        Initialize rectangle.

        Args:
            axis: Perpendicular axis (0=x, 1=y, 2=z)
            offset: Position along perpendicular axis
            bounds: ((min1, max1), (min2, max2)) in parallel directions

        Raises:
            ValueError: If axis is not 0, 1, or 2, or bounds are invalid
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Axis must be 0 (x), 1 (y), or 2 (z)")
        if bounds[0][0] >= bounds[0][1]:
            raise ValueError("First bound min must be less than max")
        if bounds[1][0] >= bounds[1][1]:
            raise ValueError("Second bound min must be less than max")

        self.axis = axis
        self.offset = offset
        self.bounds = bounds

        # Determine which axes are parallel to the rectangle
        self.parallel_axes = [i for i in range(3) if i != axis]

    def intersect(self, ray: Ray) -> Optional[Hit]:
        """
        Ray-rectangle intersection via plane intersection and bounds check.

        Steps:
        1. Find intersection with infinite plane
        2. Check if intersection is within rectangle bounds
        3. Compute normal (perpendicular to plane)

        Args:
            ray: Ray to test for intersection

        Returns:
            Hit object if ray intersects rectangle, None otherwise
        """
        # Check if ray is parallel to plane (direction perpendicular to normal)
        if abs(ray.direction[self.axis]) < 1e-10:
            return None

        # Compute t where ray intersects the plane
        t = (self.offset - ray.origin[self.axis]) / ray.direction[self.axis]

        # Intersection must be in front of ray origin
        if t <= 1e-10:
            return None

        # Compute intersection point
        position = ray.origin + t * ray.direction

        # Check if intersection is within rectangle bounds
        p1 = position[self.parallel_axes[0]]
        p2 = position[self.parallel_axes[1]]

        if not (self.bounds[0][0] <= p1 <= self.bounds[0][1]):
            return None
        if not (self.bounds[1][0] <= p2 <= self.bounds[1][1]):
            return None

        # Compute normal (points in positive axis direction or negative)
        normal = np.zeros(3, dtype=np.float64)

        # Determine if ray is coming from positive or negative side
        ray_from_positive = ray.origin[self.axis] > self.offset
        ray_outside = True  # Ray hitting rectangle is always from outside

        if ray_from_positive:
            normal[self.axis] = 1.0  # Normal points in positive axis direction
        else:
            normal[self.axis] = -1.0  # Normal points in negative axis direction

        return Hit(
            t=t,
            position=position,
            normal=normal,
            ray_outside=ray_outside
        )
