"""
Interior finite difference stencils for SHOCCS.

This module provides simple centered difference stencils for interior points
on uniform grids. All stencils are compiled with Numba for performance.

Functions:
    centered_diff_1st_order2: 3-point centered difference for 1st derivative
    centered_diff_2nd_order2: 3-point centered difference for 2nd derivative
    centered_diff_1st_order4: 5-point centered difference for 1st derivative
    centered_diff_2nd_order4: 5-point centered difference for 2nd derivative
"""

import numpy as np
from numba import njit


@njit
def centered_diff_1st_order2(h: float) -> np.ndarray:
    """
    3-point centered difference for first derivative, 2nd order accurate.

    Stencil: [-1, 0, 1] / (2h)
    Exact for polynomials up to degree 2.

    Args:
        h: Grid spacing

    Returns:
        Array of stencil coefficients [c_{-1}, c_0, c_1]

    Example:
        >>> stencil = centered_diff_1st_order2(0.1)
        >>> # Apply to data: df/dx ≈ sum(stencil * [f_{i-1}, f_i, f_{i+1}])
    """
    return np.array([-1.0 / (2.0 * h), 0.0, 1.0 / (2.0 * h)])


@njit
def centered_diff_2nd_order2(h: float) -> np.ndarray:
    """
    3-point centered difference for second derivative, 2nd order accurate.

    Stencil: [1, -2, 1] / h^2
    Exact for polynomials up to degree 3.

    Args:
        h: Grid spacing

    Returns:
        Array of stencil coefficients [c_{-1}, c_0, c_1]

    Example:
        >>> stencil = centered_diff_2nd_order2(0.1)
        >>> # Apply to data: d²f/dx² ≈ sum(stencil * [f_{i-1}, f_i, f_{i+1}])
    """
    h2 = h * h
    return np.array([1.0 / h2, -2.0 / h2, 1.0 / h2])


@njit
def centered_diff_1st_order4(h: float) -> np.ndarray:
    """
    5-point centered difference for first derivative, 4th order accurate.

    Stencil: [1, -8, 0, 8, -1] / (12h)
    Exact for polynomials up to degree 4.

    Args:
        h: Grid spacing

    Returns:
        Array of stencil coefficients [c_{-2}, c_{-1}, c_0, c_1, c_2]

    Example:
        >>> stencil = centered_diff_1st_order4(0.1)
        >>> # Apply to data: df/dx ≈ sum(stencil * [f_{i-2}, ..., f_{i+2}])
    """
    denom = 12.0 * h
    return np.array([1.0 / denom, -8.0 / denom, 0.0, 8.0 / denom, -1.0 / denom])


@njit
def centered_diff_2nd_order4(h: float) -> np.ndarray:
    """
    5-point centered difference for second derivative, 4th order accurate.

    Stencil: [-1, 16, -30, 16, -1] / (12h^2)
    Exact for polynomials up to degree 5.

    Args:
        h: Grid spacing

    Returns:
        Array of stencil coefficients [c_{-2}, c_{-1}, c_0, c_1, c_2]

    Example:
        >>> stencil = centered_diff_2nd_order4(0.1)
        >>> # Apply to data: d²f/dx² ≈ sum(stencil * [f_{i-2}, ..., f_{i+2}])
    """
    h2 = h * h
    denom = 12.0 * h2
    return np.array([-1.0 / denom, 16.0 / denom, -30.0 / denom, 16.0 / denom, -1.0 / denom])


@njit
def apply_stencil_1d(data: np.ndarray, stencil: np.ndarray, index: int) -> float:
    """
    Apply a stencil at a given index in 1D data.

    Args:
        data: Input data array
        stencil: Stencil coefficients (must be odd length)
        index: Central index where stencil is applied

    Returns:
        Result of applying stencil at the given index

    Example:
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> stencil = centered_diff_1st_order2(1.0)
        >>> result = apply_stencil_1d(data, stencil, 2)
    """
    n = len(stencil)
    half_width = n // 2
    result = 0.0

    for k in range(n):
        result += stencil[k] * data[index - half_width + k]

    return result
