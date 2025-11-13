"""
Mesh and geometry structures for SHOCCS.

This module provides basic mesh structures for Cartesian grids and boundary
point representations used in the SHOCCS solver.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import numpy.typing as npt


@dataclass
class CartesianMesh:
    """
    Uniform Cartesian mesh for computational domain.

    This class represents a uniform 3D Cartesian grid with specified extents
    and number of grid points in each direction.

    Attributes:
        nx: Number of grid points in x-direction
        ny: Number of grid points in y-direction
        nz: Number of grid points in z-direction
        xmin: Minimum x-coordinate of domain
        xmax: Maximum x-coordinate of domain
        ymin: Minimum y-coordinate of domain
        ymax: Maximum y-coordinate of domain
        zmin: Minimum z-coordinate of domain
        zmax: Maximum z-coordinate of domain

    Examples:
        >>> mesh = CartesianMesh(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        >>> mesh.dx
        0.1111111111111111
        >>> mesh.shape
        (10, 10, 10)
    """
    nx: int
    ny: int
    nz: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def __post_init__(self):
        """Validate mesh parameters."""
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.xmin >= self.xmax:
            raise ValueError("xmin must be less than xmax")
        if self.ymin >= self.ymax:
            raise ValueError("ymin must be less than ymax")
        if self.zmin >= self.zmax:
            raise ValueError("zmin must be less than zmax")

    @property
    def dx(self) -> float:
        """Grid spacing in x-direction."""
        return (self.xmax - self.xmin) / (self.nx - 1)

    @property
    def dy(self) -> float:
        """Grid spacing in y-direction."""
        return (self.ymax - self.ymin) / (self.ny - 1)

    @property
    def dz(self) -> float:
        """Grid spacing in z-direction."""
        return (self.zmax - self.zmin) / (self.nz - 1)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the mesh grid (nx, ny, nz)."""
        return (self.nx, self.ny, self.nz)

    def coordinates(self) -> Tuple[npt.NDArray[np.float64],
                                    npt.NDArray[np.float64],
                                    npt.NDArray[np.float64]]:
        """
        Generate coordinate arrays for the mesh.

        Returns:
            Tuple of (x, y, z) coordinate arrays. Each array contains the
            coordinate values along that dimension.

        Examples:
            >>> mesh = CartesianMesh(3, 3, 3, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
            >>> x, y, z = mesh.coordinates()
            >>> x
            array([0., 1., 2.])
            >>> y
            array([0., 1., 2.])
            >>> z
            array([0., 1., 2.])
        """
        x = np.linspace(self.xmin, self.xmax, self.nx)
        y = np.linspace(self.ymin, self.ymax, self.ny)
        z = np.linspace(self.zmin, self.zmax, self.nz)
        return x, y, z

    def size(self) -> int:
        """Total number of grid points in the mesh."""
        return self.nx * self.ny * self.nz

    def volume(self) -> float:
        """Total volume of the computational domain."""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)


@dataclass
class BoundaryPoint:
    """
    Boundary point information for cut-cell methods.

    This class represents a point on the boundary of an embedded object
    within the computational domain, including information about its location,
    distance from the boundary, and associated geometric properties.

    Attributes:
        position: (x, y, z) position of the boundary point
        psi: 1D cut-cell distance (normalized distance to boundary)
        solid_coord: (i, j, k) integer coordinate in solid grid
        shape_id: Integer ID of the shape this point belongs to
        ray_outside: Boolean indicating if ray is outside the object

    Examples:
        >>> bp = BoundaryPoint(
        ...     position=(0.5, 0.5, 0.5),
        ...     psi=0.25,
        ...     solid_coord=(5, 5, 5),
        ...     shape_id=0,
        ...     ray_outside=True
        ... )
        >>> bp.psi
        0.25
    """
    position: Tuple[float, float, float]
    psi: float
    solid_coord: Tuple[int, int, int]
    shape_id: int
    ray_outside: bool

    def __post_init__(self):
        """Validate boundary point parameters."""
        if len(self.position) != 3:
            raise ValueError("Position must be a 3-tuple (x, y, z)")
        if len(self.solid_coord) != 3:
            raise ValueError("Solid coordinate must be a 3-tuple (i, j, k)")
        if not (0.0 <= self.psi <= 1.0):
            raise ValueError("psi must be in range [0, 1]")
