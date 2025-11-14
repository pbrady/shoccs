"""
Cut-cell boundary operator construction for SHOCCS.

This module provides functions to build derivative operators that handle
embedded boundaries using E2_1 stencils with cut-cell geometry.
"""

from typing import List, Tuple
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from ..geometry.mesh import BoundaryPoint, CartesianMesh
from ..stencils.e2_1 import nbs_floating, nbs_dirichlet, interior
from ..stencils.interior import centered_diff_1st_order2


# Default alpha parameters for E2_1 stencils
# These values are optimized for floating boundary conditions and have been
# validated against C++ reference data in Phase 3 (see tests/test_e2_1.py).
# Alpha parameters control the accuracy and stability of near-boundary stencils.
DEFAULT_E2_1_ALPHA = np.array([-1.47956, 0.26190, -0.14507, -0.22467], dtype=np.float64)


def build_cutcell_derivative(
    mesh: CartesianMesh,
    boundary_points: List[BoundaryPoint],
    direction: int,
    order: int = 1,
    boundary_type: str = 'floating',
    alpha: np.ndarray = None
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Build derivative operator with cut-cell boundaries.

    This function constructs sparse matrices A and B for computing derivatives
    on a grid with embedded boundaries. The A matrix handles interior and
    near-boundary stencils, while the B matrix couples domain points to
    boundary values.

    Args:
        mesh: CartesianMesh defining the computational grid
        boundary_points: List of BoundaryPoint objects from ray-casting
        direction: Derivative direction (0=x, 1=y, 2=z)
        order: Derivative order (1 or 2, currently only 1 is supported)
        boundary_type: Boundary condition type ('floating' or 'dirichlet')
        alpha: Optional E2_1 stencil optimization parameters (default: DEFAULT_E2_1_ALPHA)

    Returns:
        Tuple of (A, B) sparse matrices:
            - A: Domain operator (n_total x n_total)
            - B: Boundary coupling (n_total x n_boundary)

    Notes:
        - Uses E2_1 near-boundary stencils for points near embedded boundaries
        - Assumes periodic boundary conditions at domain edges
        - Grid points far from boundaries use standard centered difference
        - Alpha parameters default to optimized values validated in Phase 3
    """
    if direction not in [0, 1, 2]:
        raise ValueError("Direction must be 0 (x), 1 (y), or 2 (z)")
    if order != 1:
        raise NotImplementedError("Only first-order derivatives currently supported")
    if boundary_type not in ['floating', 'dirichlet']:
        raise ValueError("boundary_type must be 'floating' or 'dirichlet'")

    # Grid spacing
    h = [mesh.dx, mesh.dy, mesh.dz][direction]
    n = [mesh.nx, mesh.ny, mesh.nz][direction]

    # Total number of grid points
    n_total = mesh.nx * mesh.ny * mesh.nz

    # Number of boundary points
    n_boundary = len(boundary_points)

    # Initialize sparse matrices
    A = lil_matrix((n_total, n_total))
    B = lil_matrix((n_total, n_boundary))

    # Use default alpha parameters if not provided
    if alpha is None:
        alpha = DEFAULT_E2_1_ALPHA

    # Build a mapping from solid_coord to boundary point index
    boundary_map = {}
    for bp_idx, bp in enumerate(boundary_points):
        boundary_map[bp.solid_coord] = bp_idx

    # Get standard interior stencil
    interior_stencil = centered_diff_1st_order2(h)

    # Iterate over all grid points
    for i in range(mesh.nx):
        for j in range(mesh.ny):
            for k in range(mesh.nz):
                # Flatten index (row-major / C order: last index varies fastest)
                # For array.shape = (nx, ny, nz), flatten gives ordering where k varies fastest
                flat_idx = i * (mesh.ny * mesh.nz) + j * mesh.nz + k

                # Current grid coordinate
                coord = (i, j, k)

                # Check if this is a near-boundary point
                if coord in boundary_map:
                    # Use E2_1 near-boundary stencil
                    bp_idx = boundary_map[coord]
                    bp = boundary_points[bp_idx]

                    # Determine if right-biased (for E2_1 stencil)
                    # The stencil is right-biased if the boundary is to the right
                    right_biased = bp.ray_outside

                    # Get E2_1 stencil coefficients
                    if boundary_type == 'floating':
                        coeffs = nbs_floating(h, bp.psi, alpha, right_biased)
                    else:
                        coeffs = nbs_dirichlet(h, bp.psi, alpha, right_biased)

                    # E2_1 returns R*T = 4*5 = 20 coefficients (or 15 for Dirichlet)
                    # When right_biased=False: use row 0 (closest to boundary on left)
                    # When right_biased=True: use row 3 (closest to boundary on right, after reversal)
                    # Structure of each row: [c_{i-2}, c_{i-1}, c_i, c_{i+1}, c_boundary]
                    if right_biased:
                        # For right boundary, use last row (row 3)
                        row_coeffs = coeffs[15:20]  # Row 3 of the stencil
                    else:
                        # For left boundary, use first row (row 0)
                        row_coeffs = coeffs[0:5]  # Row 0 of the stencil

                    # Apply stencil coefficients to A matrix
                    for offset in range(4):  # Interior points (i-2, i-1, i, i+1)
                        neighbor_idx = _get_neighbor_index(
                            i, j, k, direction, offset - 2, mesh
                        )
                        if neighbor_idx is not None:
                            A[flat_idx, neighbor_idx] = row_coeffs[offset]

                    # Boundary coupling coefficient
                    B[flat_idx, bp_idx] = row_coeffs[4]

                else:
                    # Use standard centered difference stencil
                    # Stencil is [-1, 0, 1] / (2h)
                    for offset, coeff in enumerate(interior_stencil):
                        neighbor_idx = _get_neighbor_index(
                            i, j, k, direction, offset - 1, mesh
                        )
                        if neighbor_idx is not None:
                            A[flat_idx, neighbor_idx] = coeff

    return A.tocsr(), B.tocsr()


def _get_neighbor_index(
    i: int, j: int, k: int,
    direction: int,
    offset: int,
    mesh: CartesianMesh
) -> int:
    """
    Get flattened index of a neighbor point with periodic boundary conditions.

    IMPORTANT: This function assumes periodic boundary conditions at domain edges.
    Neighbors wrap around via modulo arithmetic. For non-periodic boundaries,
    this function would need to return None for out-of-bounds indices.

    Args:
        i, j, k: Current grid point indices
        direction: Direction of offset (0=x, 1=y, 2=z)
        offset: Offset in the specified direction
        mesh: CartesianMesh

    Returns:
        Flattened index of neighbor (always valid due to periodic wrapping)
    """
    # Compute neighbor coordinates with periodic wrapping (modulo arithmetic)
    if direction == 0:  # x-direction
        ni = (i + offset) % mesh.nx
        nj = j
        nk = k
    elif direction == 1:  # y-direction
        ni = i
        nj = (j + offset) % mesh.ny
        nk = k
    else:  # z-direction
        ni = i
        nj = j
        nk = (k + offset) % mesh.nz

    # Flatten index (row-major / C order: last index varies fastest)
    flat_idx = ni * (mesh.ny * mesh.nz) + nj * mesh.nz + nk

    return flat_idx
