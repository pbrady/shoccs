"""
Differential operators for SHOCCS.

This module provides differential operators (derivative, gradient, Laplacian)
that operate on scalar and vector fields.
"""

from .derivative import DerivativeOperator
from .gradient import GradientOperator
from .laplacian import LaplacianOperator

__all__ = [
    'DerivativeOperator',
    'GradientOperator',
    'LaplacianOperator',
    'create_derivative_operator',
    'create_gradient_operator',
    'create_laplacian_operator',
]


def create_derivative_operator(mesh, stencil_func, direction, derivative_order=1, bc_type='periodic'):
    """
    Factory function to create a derivative operator.

    Args:
        mesh: CartesianMesh object
        stencil_func: Stencil function (e.g., centered_diff_1st_order2)
        direction: Direction of derivative (0=x, 1=y, 2=z)
        derivative_order: Order of derivative (1 or 2)
        bc_type: Boundary condition type ('periodic', 'dirichlet', 'neumann')

    Returns:
        DerivativeOperator instance
    """
    return DerivativeOperator(
        mesh=mesh,
        stencil_func=stencil_func,
        direction=direction,
        derivative_order=derivative_order,
        bc_type=bc_type
    )


def create_gradient_operator(mesh, stencil_func, bc_type='periodic'):
    """
    Create gradient operator from 3 derivative operators.

    Args:
        mesh: CartesianMesh object
        stencil_func: Stencil function for first derivatives
        bc_type: Boundary condition type

    Returns:
        GradientOperator instance
    """
    return GradientOperator(
        Dx=create_derivative_operator(mesh, stencil_func, 0, derivative_order=1, bc_type=bc_type),
        Dy=create_derivative_operator(mesh, stencil_func, 1, derivative_order=1, bc_type=bc_type),
        Dz=create_derivative_operator(mesh, stencil_func, 2, derivative_order=1, bc_type=bc_type)
    )


def create_laplacian_operator(mesh, stencil_func, bc_type='periodic'):
    """
    Create Laplacian operator from 3 second derivative operators.

    Args:
        mesh: CartesianMesh object
        stencil_func: Stencil function for second derivatives
        bc_type: Boundary condition type

    Returns:
        LaplacianOperator instance
    """
    return LaplacianOperator(
        Dxx=create_derivative_operator(mesh, stencil_func, 0, derivative_order=2, bc_type=bc_type),
        Dyy=create_derivative_operator(mesh, stencil_func, 1, derivative_order=2, bc_type=bc_type),
        Dzz=create_derivative_operator(mesh, stencil_func, 2, derivative_order=2, bc_type=bc_type)
    )
