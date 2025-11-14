"""
Derivative operator for SHOCCS.

This module provides a basic derivative operator that can compute
derivatives of scalar fields in specified coordinate directions.
"""

from dataclasses import dataclass
import numpy as np
from ..fields.field import ScalarField
from ..geometry.mesh import CartesianMesh


@dataclass
class DerivativeOperator:
    """
    Derivative operator for computing directional derivatives.

    This operator computes derivatives of scalar fields using finite
    difference stencils. It applies the stencil in a specified coordinate
    direction with periodic boundary conditions.

    Attributes:
        mesh: CartesianMesh defining the computational grid
        stencil_func: Function that generates stencil coefficients
        direction: Coordinate direction (0=x, 1=y, 2=z)
        derivative_order: Order of derivative (1 or 2)
        bc_type: Boundary condition type (currently only 'periodic')

    Example:
        >>> from shoccs.geometry import CartesianMesh
        >>> from shoccs.stencils import centered_diff_1st_order2
        >>> mesh = CartesianMesh(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        >>> Dx = DerivativeOperator(mesh, centered_diff_1st_order2, 0)
        >>> u = ScalarField(D=np.random.rand(10, 10, 10))
        >>> dudx = Dx(u)
    """
    mesh: CartesianMesh
    stencil_func: callable
    direction: int
    derivative_order: int = 1
    bc_type: str = 'periodic'

    def __post_init__(self):
        """Validate operator parameters."""
        if self.direction not in [0, 1, 2]:
            raise ValueError("Direction must be 0 (x), 1 (y), or 2 (z)")
        if self.derivative_order not in [1, 2]:
            raise ValueError("Derivative order must be 1 or 2")
        if self.bc_type != 'periodic':
            raise NotImplementedError("Only periodic boundary conditions currently supported")

        # Get grid spacing in the appropriate direction
        if self.direction == 0:
            self.h = self.mesh.dx
        elif self.direction == 1:
            self.h = self.mesh.dy
        else:
            self.h = self.mesh.dz

        # Generate stencil coefficients
        self.stencil = self.stencil_func(self.h)

    def __call__(self, u: ScalarField) -> ScalarField:
        """
        Apply derivative operator to a scalar field.

        Args:
            u: Input scalar field

        Returns:
            ScalarField containing the derivative

        Example:
            >>> Dx = DerivativeOperator(mesh, centered_diff_1st_order2, 0)
            >>> u = ScalarField(D=np.random.rand(10, 10, 10))
            >>> dudx = Dx(u)
        """
        # Create output field
        result = u.zeros_like()

        # Apply stencil in the specified direction
        result.D = self._apply_derivative(u.D)

        return result

    def _apply_derivative(self, data: np.ndarray) -> np.ndarray:
        """
        Apply derivative stencil to data array.

        This method applies the finite difference stencil along the
        specified coordinate direction using periodic boundary conditions.

        Args:
            data: Input data array (3D)

        Returns:
            Derivative array with same shape as input
        """
        result = np.zeros_like(data)
        nx, ny, nz = data.shape
        stencil_width = len(self.stencil)
        half_width = stencil_width // 2

        if self.direction == 0:  # x-direction
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        for s in range(stencil_width):
                            # Periodic boundary conditions
                            idx = (i - half_width + s) % nx
                            result[i, j, k] += self.stencil[s] * data[idx, j, k]

        elif self.direction == 1:  # y-direction
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        for s in range(stencil_width):
                            # Periodic boundary conditions
                            jdx = (j - half_width + s) % ny
                            result[i, j, k] += self.stencil[s] * data[i, jdx, k]

        else:  # z-direction
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        for s in range(stencil_width):
                            # Periodic boundary conditions
                            kdx = (k - half_width + s) % nz
                            result[i, j, k] += self.stencil[s] * data[i, j, kdx]

        return result
