"""
E2-Poly finite difference stencils for SHOCCS.

Translation of polyE2_1.cpp - a simpler E2-type stencil with polynomial basis.
This stencil is 1st order accurate (P=1) with support of 3 rows (R=3) and
tail length 4 (T=4).

Constants:
    P: Order of accuracy (1)
    R: Number of rows in near-boundary stencil (3)
    T: Tail length for boundary stencils (4)
    X: Extra boundary parameter (0)

The stencil uses three parameter arrays:
    - fa: Floating boundary condition parameters (6 elements)
    - da: Dirichlet boundary condition parameters (3 elements)
    - ia: Interpolation parameters (4 elements)
"""

import numpy as np
from numba import njit
from typing import Tuple

# Stencil constants
P = 1  # Order of accuracy
R = 3  # Number of rows
T = 4  # Tail length
X = 0  # Extra parameter


class E2PolyStencil:
    """
    E2-Poly stencil configuration.

    Simple data container for stencil parameters. Not meant to be
    used in performance-critical code (use the @njit functions instead).

    Attributes:
        fa: Floating boundary parameters (6 elements)
        da: Dirichlet boundary parameters (3 elements)
        ia: Interpolation parameters (4 elements)
    """

    def __init__(
        self,
        fa: np.ndarray = None,
        da: np.ndarray = None,
        ia: np.ndarray = None
    ):
        """
        Initialize E2-Poly stencil parameters.

        Args:
            fa: Floating boundary parameters (will be padded/truncated to 6)
            da: Dirichlet boundary parameters (will be padded/truncated to 3)
            ia: Interpolation parameters (will be padded/truncated to 4)
        """
        # Initialize with zeros and copy provided values
        self.fa = np.zeros(6, dtype=np.float64)
        self.da = np.zeros(3, dtype=np.float64)
        self.ia = np.zeros(4, dtype=np.float64)

        if fa is not None:
            n = min(len(fa), 6)
            self.fa[:n] = fa[:n]

        if da is not None:
            n = min(len(da), 3)
            self.da[:n] = da[:n]

        if ia is not None:
            n = min(len(ia), 4)
            self.ia[:n] = ia[:n]


@njit
def interior(h: float) -> np.ndarray:
    """
    Interior stencil for E2-Poly (1st derivative).

    Simple 3-point centered difference: [-1, 0, 1] / (2h)

    Args:
        h: Grid spacing

    Returns:
        Stencil coefficients [c_{-1}, c_0, c_1]
    """
    c = np.zeros(3, dtype=np.float64)
    c[0] = -0.5
    c[1] = 0.0
    c[2] = 0.5

    # Scale by grid spacing
    c /= h
    return c


@njit
def interp_interior(y: float) -> np.ndarray:
    """
    Interior interpolation stencil.

    Linear interpolation between two grid points.

    Args:
        y: Interpolation offset (-1 <= y <= 1)

    Returns:
        Interpolation coefficients (2 elements)
    """
    c = np.zeros(2, dtype=np.float64)

    if y > 0:
        c[0] = 1.0 - y
        c[1] = y
    else:
        c[0] = -y
        c[1] = 1.0 + y

    return c


@njit
def interp_wall(
    i: int,
    y: float,
    psi: float,
    fa: np.ndarray,
    ia: np.ndarray,
    right: bool
) -> np.ndarray:
    """
    Wall interpolation stencil.

    Interpolation near boundaries using stencil parameters.

    Args:
        i: Stencil index (0 or 1)
        y: Interpolation offset
        psi: Boundary stretching parameter
        fa: Floating boundary parameters (6 elements)
        ia: Interpolation parameters (4 elements)
        right: True for right boundary, False for left

    Returns:
        Interpolation coefficients (4 elements)
    """
    c = np.zeros(4, dtype=np.float64)

    if right:
        t5 = fa[2]
        t6 = t5 * y
        t9 = ia[2]
        t13 = -y
        t7 = fa[3]
        t10 = ia[3]
        t16 = 1.0 + psi
        t17 = 1.0 / t16
        t8 = t7 * y
        t30 = fa[0]
        t31 = t30 * y
        t34 = ia[0]
        t32 = fa[1]
        t35 = ia[1]
        t33 = t32 * y

        if i == 0:
            c[0] = (t31 + t33 + t34 + t35) * 0.5
            c[1] = (-psi + t13 + (t31 + t34) * -2.0 +
                    (t13 + -2.0 * t35 + -psi * t35 + -2.0 * t32 * y +
                     -psi * t32 * y) * t17) * 0.5
            c[2] = (1.0 + psi + t31 + t34 + y) * 0.5
            c[3] = (1.0 + psi + t33 + t35 + y) * 0.5 * t17
        elif i == 1:
            c[0] = (t10 + t6 + t8 + t9) * 0.5
            c[1] = (t13 + (t6 + t9) * -2.0 +
                    (psi + -2.0 * t10 + -psi * t10 + t13 + -2.0 * t7 * y +
                     -psi * t7 * y) * t17) * 0.5
            c[2] = (1.0 + t6 + t9 + y) * 0.5
            c[3] = (1.0 + t10 + t8 + y) * 0.5 * t17
    else:
        t7 = -y
        t13 = fa[0]
        t14 = t13 * y
        t15 = ia[0]
        t5 = 1.0 + psi
        t6 = 1.0 / t5
        t8 = fa[1]
        t10 = ia[1]
        t9 = t8 * y
        t36 = fa[2]
        t37 = t36 * y
        t38 = ia[2]
        t31 = fa[3]
        t33 = ia[3]
        t32 = t31 * y

        if i == 0:
            c[0] = (1.0 + psi + t10 + t7 + t9) * 0.5 * t6
            c[1] = (1.0 + psi + t14 + t15 + t7) * 0.5
            c[2] = (-psi + (t14 + t15) * -2.0 + y +
                    (-2.0 * t10 + -psi * t10 + y + -2.0 * t8 * y +
                     -psi * t8 * y) * t6) * 0.5
            c[3] = (t10 + t14 + t15 + t9) * 0.5
        elif i == 1:
            c[0] = (1.0 + t32 + t33 + t7) * 0.5 * t6
            c[1] = (1.0 + t37 + t38 + t7) * 0.5
            c[2] = ((t37 + t38) * -2.0 + y +
                    (psi + -2.0 * t33 + -psi * t33 + y + -2.0 * t31 * y +
                     -psi * t31 * y) * t6) * 0.5
            c[3] = (t32 + t33 + t37 + t38) * 0.5

    return c


@njit
def nbs_floating(
    h: float,
    psi: float,
    fa: np.ndarray,
    right: bool
) -> np.ndarray:
    """
    Near-boundary stencil for floating boundary conditions.

    Args:
        h: Grid spacing
        psi: Boundary stretching parameter
        fa: Floating boundary parameters (6 elements)
        right: True for right boundary, False for left

    Returns:
        Stencil coefficients (R * T = 12 elements)
    """
    c = np.zeros(R * T, dtype=np.float64)

    t10 = fa[0]
    t5 = 1.0 + psi
    t6 = 1.0 / t5
    t7 = fa[1]
    t8 = -1.0 + t7
    t22 = fa[2]
    t14 = 2.0 + psi
    t19 = fa[3]
    t20 = -1.0 + t19
    t33 = fa[4]
    t30 = fa[5]
    t31 = -1.0 + t30

    c[0] = 0.5 * t6 * t8
    c[1] = (-1.0 + t10) * 0.5
    c[2] = -t10 - 0.5 * t14 * t6 * t8
    c[3] = (t10 + t7) * 0.5

    c[4] = 0.5 * t20 * t6
    c[5] = (-1.0 + t22) * 0.5
    c[6] = -t22 - 0.5 * t14 * t20 * t6
    c[7] = (t19 + t22) * 0.5

    c[8] = 0.5 * t31 * t6
    c[9] = (-1.0 + t33) * 0.5
    c[10] = -t33 - 0.5 * t14 * t31 * t6
    c[11] = (t30 + t33) * 0.5

    # Scale by grid spacing
    c /= h

    if right:
        # Reverse and negate for right boundary
        c = -c[::-1]

    return c


@njit
def nbs_dirichlet(
    h: float,
    psi: float,
    da: np.ndarray,
    right: bool
) -> np.ndarray:
    """
    Near-boundary stencil for Dirichlet boundary conditions.

    Args:
        h: Grid spacing
        psi: Boundary stretching parameter
        da: Dirichlet boundary parameters (3 elements)
        right: True for right boundary, False for left

    Returns:
        Stencil coefficients ((R-1) * T = 8 elements)
    """
    c = np.zeros((R - 1) * T, dtype=np.float64)

    t7 = da[0]
    t13 = da[1]
    t19 = da[2]
    t5 = 1.0 + psi
    t6 = 1.0 / t5
    t8 = -1.0 + t7
    t18 = 2.0 * psi
    t20 = 3.0 * t19
    t21 = 2.0 * psi * t19
    t22 = 1.0 + t18 + t20 + t21
    t23 = 1.0 / t22
    t14 = 2.0 * psi * t13
    t15 = 3.0 * t13 * t7
    t16 = 2.0 * psi * t13 * t7
    t29 = -t13
    t25 = 2.0 + psi
    t43 = -1.0 + t19

    c[0] = 0.5 * t6 * t8
    c[1] = (-1.0 + -2.0 * psi + t13 + t14 + t15 + t16 +
            -3.0 * t7 + -2.0 * psi * t7) * 0.5 * t23
    c[2] = ((-2.0 * psi * t13 + -3.0 * t19 + -2.0 * psi * t19 + t29 +
             3.0 * t7 + 2.0 * psi * t7 + -3.0 * t13 * t7 +
             -2.0 * psi * t13 * t7) * t23 +
            -0.5 * t25 * t6 * t8)
    c[3] = ((t13 + t14 + t15 + t16 + t20 + t21 + -2.0 * t7 +
             3.0 * t19 * t7 + 2.0 * psi * t19 * t7) * 0.5 * t23)

    c[4] = 0.5 * t43 * t6
    c[5] = (-1.0 + t13) * 0.5
    c[6] = t29 - 0.5 * t25 * t43 * t6
    c[7] = (t13 + t19) * 0.5

    # Scale by grid spacing
    c /= h

    if right:
        # Reverse and negate for right boundary
        c = -c[::-1]

    return c


@njit
def nbs_neumann(
    h: float,
    psi: float,
    right: bool
) -> np.ndarray:
    """
    Near-boundary stencil for Neumann boundary conditions.

    Currently not implemented (returns empty array).

    Args:
        h: Grid spacing
        psi: Boundary stretching parameter
        right: True for right boundary, False for left

    Returns:
        Empty stencil array
    """
    return np.zeros(0, dtype=np.float64)


def nbs(
    h: float,
    bc_type: str,
    psi: float,
    params: E2PolyStencil,
    right: bool
) -> np.ndarray:
    """
    Dispatch function for near-boundary stencils.

    Args:
        h: Grid spacing
        bc_type: Boundary condition type ('floating', 'dirichlet', 'neumann')
        psi: Boundary stretching parameter
        params: E2-Poly stencil parameters
        right: True for right boundary, False for left

    Returns:
        Near-boundary stencil coefficients
    """
    bc_type = bc_type.lower()

    if bc_type == 'floating':
        return nbs_floating(h, psi, params.fa, right)
    elif bc_type == 'dirichlet':
        return nbs_dirichlet(h, psi, params.da, right)
    elif bc_type == 'neumann':
        return nbs_neumann(h, psi, right)
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")


def make_e2_poly_stencil(
    fa: np.ndarray = None,
    da: np.ndarray = None,
    ia: np.ndarray = None
) -> E2PolyStencil:
    """
    Factory function to create E2-Poly stencil configuration.

    Args:
        fa: Floating boundary parameters
        da: Dirichlet boundary parameters
        ia: Interpolation parameters

    Returns:
        E2PolyStencil configuration object
    """
    return E2PolyStencil(fa, da, ia)
