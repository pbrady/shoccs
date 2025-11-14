"""
Laplacian operator for SHOCCS.

This module provides a Laplacian operator composed from three second
derivative operators, following the principle of composition over duplication.
"""

from dataclasses import dataclass
from .derivative import DerivativeOperator
from ..fields.field import ScalarField


@dataclass
class LaplacianOperator:
    """
    Laplacian operator: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z².

    The Laplacian operator computes the sum of second partial derivatives
    in all three coordinate directions.

    This is a thin wrapper around three second-derivative operators,
    demonstrating composition over code duplication.

    Attributes:
        Dxx: Second derivative operator in x-direction
        Dyy: Second derivative operator in y-direction
        Dzz: Second derivative operator in z-direction

    Example:
        >>> from shoccs.operators import create_laplacian_operator
        >>> from shoccs.geometry import CartesianMesh
        >>> from shoccs.stencils import centered_diff_2nd_order2
        >>> mesh = CartesianMesh(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        >>> laplacian = create_laplacian_operator(mesh, centered_diff_2nd_order2)
        >>> u = ScalarField(D=...)
        >>> lap_u = laplacian(u)  # Returns ∇²u
    """
    Dxx: DerivativeOperator
    Dyy: DerivativeOperator
    Dzz: DerivativeOperator

    def __call__(self, u: ScalarField) -> ScalarField:
        """
        Compute Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z².

        Args:
            u: Input scalar field

        Returns:
            ScalarField containing the Laplacian

        Example:
            >>> lap_u = laplacian(u)
            >>> # lap_u contains ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
        """
        result = self.Dxx(u) + self.Dyy(u) + self.Dzz(u)
        return result
