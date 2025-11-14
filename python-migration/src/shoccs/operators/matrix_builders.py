"""
Sparse matrix construction utilities for SHOCCS operators.

This module provides utilities for building CSR (Compressed Sparse Row) matrices
for finite difference operators. These matrices represent the action of derivative
operators on field data.

Key concepts:
    - Circulant operators: For periodic boundaries with uniform stencils
    - Boundary coupling: For non-periodic boundaries (B matrix)
    - CSR format: Efficient sparse matrix storage

Functions:
    build_circulant_operator: Build circulant matrix from stencil
    build_boundary_coupling: Build boundary-to-interior coupling matrix
    build_1d_derivative_matrix: Build full 1D derivative matrix
    build_banded_matrix: Build banded matrix from stencil (Dirichlet/Neumann)
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, diags
from typing import Optional, Tuple
from numba import njit


def build_circulant_operator(stencil_coeffs: np.ndarray, n: int) -> csr_matrix:
    """
    Build circulant operator matrix from stencil for periodic boundaries.

    A circulant matrix has the same stencil pattern at each row, with
    periodic wrapping at boundaries. This is used for periodic boundary
    conditions where the stencil is uniform across all points.

    Example:
        For a 3-point stencil [-1/(2h), 0, 1/(2h)] on n=5 points:

        [  0      1/(2h)   0      0     -1/(2h) ]
        [ -1/(2h)  0      1/(2h)  0      0     ]
        [  0     -1/(2h)   0     1/(2h)  0     ]
        [  0      0      -1/(2h)  0     1/(2h) ]
        [ 1/(2h)  0       0     -1/(2h)  0     ]

    Args:
        stencil_coeffs: Stencil coefficients (odd length)
                       For centered stencil: [c_{-k}, ..., c_0, ..., c_k]
        n: Number of grid points

    Returns:
        CSR sparse matrix of shape (n, n)

    Examples:
        >>> # First derivative with periodic BC
        >>> h = 0.1
        >>> stencil = np.array([-1/(2*h), 0, 1/(2*h)])
        >>> D = build_circulant_operator(stencil, 10)
        >>> # Apply to periodic function
        >>> u = np.sin(2*np.pi*x)
        >>> du_dx = D @ u
    """
    if len(stencil_coeffs) % 2 == 0:
        raise ValueError("Stencil must have odd length for centered differences")

    # Stencil width (number of points on each side of center)
    width = len(stencil_coeffs) // 2

    # Use lil_matrix for efficient construction
    A = lil_matrix((n, n), dtype=np.float64)

    # Fill each row with stencil pattern
    for i in range(n):
        for k in range(len(stencil_coeffs)):
            # Offset from center point
            offset = k - width
            # Column index with periodic wrapping
            j = (i + offset) % n
            A[i, j] = stencil_coeffs[k]

    return A.tocsr()


def build_banded_matrix(
    stencil_coeffs: np.ndarray,
    n: int,
    boundary_order: int = 1,
    boundary_value: float = 0.0
) -> csr_matrix:
    """
    Build banded matrix from stencil for Dirichlet boundaries.

    For non-periodic boundaries, we apply the stencil only at interior
    points. Boundary points can be:
    - Fixed (Dirichlet): value prescribed
    - One-sided stencils: use asymmetric stencils at boundaries

    Args:
        stencil_coeffs: Interior stencil coefficients
        n: Number of grid points (including boundaries)
        boundary_order: Order of boundary stencil (1=first order, 2=second order)
        boundary_value: Prescribed boundary value (for Dirichlet BC)

    Returns:
        CSR sparse matrix of shape (n-2, n) for interior points only,
        or (n, n) if including boundary rows

    Examples:
        >>> # First derivative with Dirichlet BC
        >>> h = 0.1
        >>> stencil = np.array([-1/(2*h), 0, 1/(2*h)])
        >>> D = build_banded_matrix(stencil, 10)
    """
    if len(stencil_coeffs) % 2 == 0:
        raise ValueError("Stencil must have odd length for centered differences")

    width = len(stencil_coeffs) // 2

    # For now, build matrix for interior points only (n-2 rows, n columns)
    # This allows boundary values to be handled separately
    n_interior = n - 2 * width

    if n_interior <= 0:
        raise ValueError(f"Grid too small for stencil width {width}")

    # Use lil_matrix for efficient construction
    A = lil_matrix((n_interior, n), dtype=np.float64)

    # Fill interior rows
    for i in range(n_interior):
        # Actual grid index (offset by width for boundaries)
        grid_i = i + width
        for k in range(len(stencil_coeffs)):
            offset = k - width
            j = grid_i + offset
            if 0 <= j < n:
                A[i, j] = stencil_coeffs[k]

    return A.tocsr()


def build_boundary_coupling(
    n: int,
    width: int,
    boundary_stencils: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Build sparse matrices coupling boundary points to interior.

    This constructs the B matrices that couple boundary regions to the
    interior domain. In the SHOCCS architecture:

        du/dx|_interior = O @ u_interior + B_left @ u_left + B_right @ u_right

    Args:
        n: Total number of grid points
        width: Stencil width (half-width, e.g., 1 for 3-point, 2 for 5-point)
        boundary_stencils: Optional tuple of (left_stencil, right_stencil)
                          If None, assumes values are prescribed (zero coupling)

    Returns:
        Tuple of (B_left, B_right) sparse matrices
        B_left: shape (n_interior, width) - couples left boundary to interior
        B_right: shape (n_interior, width) - couples right boundary to interior

    Examples:
        >>> # For 3-point stencil (width=1) on n=10 points
        >>> B_left, B_right = build_boundary_coupling(10, 1)
        >>> # B_left couples u[0] to interior, B_right couples u[-1] to interior
    """
    n_interior = n - 2 * width

    if boundary_stencils is None:
        # No coupling - boundary values are prescribed (Dirichlet)
        # But we still need to extract contribution from stencil
        # For centered stencil, the leftmost interior point uses u[0]
        # and rightmost interior point uses u[-1]

        B_left = lil_matrix((n_interior, width), dtype=np.float64)
        B_right = lil_matrix((n_interior, width), dtype=np.float64)

        # The first interior point may use boundary point(s) on the left
        # The last interior point may use boundary point(s) on the right
        # These coefficients come from the stencil applied at those points
        # For now, return zero matrices (will be filled by derivative builder)

        return B_left.tocsr(), B_right.tocsr()
    else:
        # Custom boundary stencils provided
        left_stencil, right_stencil = boundary_stencils

        B_left = lil_matrix((n_interior, width), dtype=np.float64)
        B_right = lil_matrix((n_interior, width), dtype=np.float64)

        # Apply left boundary stencil to leftmost interior points
        # Apply right boundary stencil to rightmost interior points
        # (Implementation depends on specific boundary treatment)

        return B_left.tocsr(), B_right.tocsr()


def build_1d_derivative_matrix(
    stencil_coeffs: np.ndarray,
    n: int,
    periodic: bool = False,
    boundary_type: str = 'dirichlet'
) -> csr_matrix:
    """
    Build complete 1D derivative operator matrix.

    Convenience function that builds the appropriate matrix structure
    based on boundary conditions.

    Args:
        stencil_coeffs: Stencil coefficients
        n: Number of grid points
        periodic: If True, use periodic boundaries (circulant matrix)
        boundary_type: Type of boundary condition if not periodic
                      'dirichlet' - fixed values at boundaries
                      'neumann' - fixed derivatives at boundaries

    Returns:
        CSR sparse matrix representing the derivative operator

    Examples:
        >>> # Periodic derivative operator
        >>> h = 0.1
        >>> stencil = np.array([-1/(2*h), 0, 1/(2*h)])
        >>> D_periodic = build_1d_derivative_matrix(stencil, 100, periodic=True)
        >>>
        >>> # Dirichlet derivative operator (interior points only)
        >>> D_dirichlet = build_1d_derivative_matrix(stencil, 100, periodic=False)
    """
    if periodic:
        return build_circulant_operator(stencil_coeffs, n)
    else:
        if boundary_type == 'dirichlet':
            return build_banded_matrix(stencil_coeffs, n)
        else:
            raise NotImplementedError(f"Boundary type '{boundary_type}' not yet implemented")


@njit
def apply_matrix_free_1d(
    u: np.ndarray,
    stencil: np.ndarray,
    result: np.ndarray,
    periodic: bool = False
) -> None:
    """
    Apply 1D derivative stencil matrix-free (without forming the matrix).

    This is a Numba-accelerated matrix-free implementation that can be
    faster than sparse matrix-vector multiplication for small problems.

    Args:
        u: Input array
        stencil: Stencil coefficients
        result: Output array (modified in-place)
        periodic: Use periodic boundary conditions

    Note:
        This function modifies 'result' in-place.
        For periodic=False, only interior points are computed.
    """
    n = len(u)
    width = len(stencil) // 2

    if periodic:
        # Apply stencil at all points with periodic wrapping
        for i in range(n):
            result[i] = 0.0
            for k in range(len(stencil)):
                offset = k - width
                j = (i + offset) % n
                result[i] += stencil[k] * u[j]
    else:
        # Apply stencil only at interior points
        for i in range(width, n - width):
            result[i] = 0.0
            for k in range(len(stencil)):
                offset = k - width
                j = i + offset
                result[i] += stencil[k] * u[j]
