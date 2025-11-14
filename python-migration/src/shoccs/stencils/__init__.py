"""
Finite difference stencils for SHOCCS.

This module provides various finite difference stencils for computing
derivatives on structured grids, including:

- Interior stencils: Simple centered differences for uniform grids
- E2-Poly stencils: Polynomial-based stencils with boundary treatments

All performance-critical functions are JIT-compiled with Numba.
"""

from .interior import (
    centered_diff_1st_order2,
    centered_diff_2nd_order2,
    centered_diff_1st_order4,
    centered_diff_2nd_order4,
    apply_stencil_1d,
)

from .e2_poly import (
    E2PolyStencil,
    interior as e2_poly_interior,
    interp_interior as e2_poly_interp_interior,
    interp_wall as e2_poly_interp_wall,
    nbs_floating as e2_poly_nbs_floating,
    nbs_dirichlet as e2_poly_nbs_dirichlet,
    nbs_neumann as e2_poly_nbs_neumann,
    nbs as e2_poly_nbs,
    make_e2_poly_stencil,
    P as E2_POLY_P,
    R as E2_POLY_R,
    T as E2_POLY_T,
    X as E2_POLY_X,
)

__all__ = [
    # Interior stencils
    'centered_diff_1st_order2',
    'centered_diff_2nd_order2',
    'centered_diff_1st_order4',
    'centered_diff_2nd_order4',
    'apply_stencil_1d',
    # E2-Poly stencils
    'E2PolyStencil',
    'e2_poly_interior',
    'e2_poly_interp_interior',
    'e2_poly_interp_wall',
    'e2_poly_nbs_floating',
    'e2_poly_nbs_dirichlet',
    'e2_poly_nbs_neumann',
    'e2_poly_nbs',
    'make_e2_poly_stencil',
    # E2-Poly constants
    'E2_POLY_P',
    'E2_POLY_R',
    'E2_POLY_T',
    'E2_POLY_X',
]
