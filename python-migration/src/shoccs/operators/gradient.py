"""
Gradient operator for SHOCCS.

This module provides a gradient operator composed from three derivative
operators, following the principle of composition over duplication.
"""

from dataclasses import dataclass
from .derivative import DerivativeOperator
from ..fields.field import ScalarField, VectorField


@dataclass
class GradientOperator:
    """
    Gradient operator composed from 3 derivative operators.

    The gradient operator computes the vector of partial derivatives:
    ∇u = (∂u/∂x, ∂u/∂y, ∂u/∂z)

    This is a thin wrapper around three DerivativeOperators, demonstrating
    composition over code duplication.

    Attributes:
        Dx: Derivative operator in x-direction
        Dy: Derivative operator in y-direction
        Dz: Derivative operator in z-direction

    Example:
        >>> from shoccs.operators import create_gradient_operator
        >>> from shoccs.geometry import CartesianMesh
        >>> from shoccs.stencils import centered_diff_1st_order2
        >>> mesh = CartesianMesh(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        >>> grad = create_gradient_operator(mesh, centered_diff_1st_order2)
        >>> u = ScalarField(D=...)
        >>> grad_u = grad(u)  # Returns VectorField with (∂u/∂x, ∂u/∂y, ∂u/∂z)
    """
    Dx: DerivativeOperator
    Dy: DerivativeOperator
    Dz: DerivativeOperator

    def __call__(self, u: ScalarField) -> VectorField:
        """
        Compute gradient: ∇u = (∂u/∂x, ∂u/∂y, ∂u/∂z).

        Args:
            u: Input scalar field

        Returns:
            VectorField containing the gradient components

        Example:
            >>> grad_u = gradient(u)
            >>> # grad_u.x contains ∂u/∂x
            >>> # grad_u.y contains ∂u/∂y
            >>> # grad_u.z contains ∂u/∂z
        """
        return VectorField(
            x=self.Dx(u),
            y=self.Dy(u),
            z=self.Dz(u)
        )
