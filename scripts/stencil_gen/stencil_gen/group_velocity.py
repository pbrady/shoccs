"""Group velocity analysis for finite difference stencils.

Provides tools for computing modified wavenumber, phase velocity, and group
velocity from stencil coefficients.  For the model equation u_t + u_x = 0
semi-discretized as du/dt = -D*u, the modified wavenumber kappa*(xi) of D
gives the dispersion relation omega = Im(kappa*(xi)).  The group velocity
is C(xi) = d(omega)/d(xi) = d(Im(kappa*))/d(xi).

References:
  Trefethen, "Group velocity in finite difference schemes", 1982.
  Trefethen, "Stability and group velocity", 1983 (GKS connection).
"""

from dataclasses import dataclass

import numpy as np


def modified_wavenumber(
    weights,
    i_eval: int,
    node_indices,
    xi_array: np.ndarray,
) -> np.ndarray:
    """Compute modified wavenumber kappa*(xi) for given stencil weights.

    Parameters
    ----------
    weights : array-like
        Stencil coefficients w_j.
    i_eval : int
        Grid index where derivative is evaluated.
    node_indices : array-like of int
        Grid indices used by the stencil.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    np.ndarray (complex)
        kappa*(xi) = sum_j w_j exp(i (j - i_eval) xi)
    """
    w = np.asarray(weights, dtype=complex)
    offsets = np.asarray(node_indices) - i_eval
    phase = np.exp(1j * np.outer(xi_array, offsets))
    return phase @ w


def group_velocity(kappa_star: np.ndarray, xi_array: np.ndarray) -> np.ndarray:
    """Compute group velocity C(xi) = d(Im(kappa*))/d(xi) numerically.

    Uses numpy.gradient for the numerical differentiation.

    Parameters
    ----------
    kappa_star : np.ndarray (complex)
        Modified wavenumber array.
    xi_array : np.ndarray
        Wavenumber values xi.

    Returns
    -------
    np.ndarray (real)
        Group velocity C(xi).
    """
    return np.gradient(np.imag(kappa_star), xi_array)


def group_velocity_exact(
    weights,
    i_eval: int,
    node_indices,
    xi_array: np.ndarray,
) -> np.ndarray:
    """Compute group velocity analytically from stencil weights.

    C(xi) = Re(sum_j w_j (j - i_eval) exp(i (j - i_eval) xi))

    This avoids numerical differentiation entirely.

    Parameters
    ----------
    weights : array-like
        Stencil coefficients w_j.
    i_eval : int
        Grid index where derivative is evaluated.
    node_indices : array-like of int
        Grid indices used by the stencil.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    np.ndarray (real)
        Group velocity C(xi).
    """
    w = np.asarray(weights, dtype=complex)
    offsets = np.asarray(node_indices) - i_eval
    phase = np.exp(1j * np.outer(xi_array, offsets))
    # d(kappa*)/d(xi) = sum_j w_j * i*(j-i_eval) * exp(i*(j-i_eval)*xi)
    # C = d(Im(kappa*))/d(xi) = Re(sum_j w_j * (j-i_eval) * exp(...))
    return np.real(phase @ (w * offsets))


def phase_velocity(kappa_star: np.ndarray, xi_array: np.ndarray) -> np.ndarray:
    """Compute phase velocity c(xi) = Im(kappa*(xi)) / xi.

    At xi=0 the limit is taken from the next nonzero xi value.

    Parameters
    ----------
    kappa_star : np.ndarray (complex)
        Modified wavenumber array.
    xi_array : np.ndarray
        Wavenumber values xi.

    Returns
    -------
    np.ndarray (real)
        Phase velocity c(xi).
    """
    c = np.empty_like(xi_array, dtype=float)
    nonzero = xi_array != 0.0
    c[nonzero] = np.imag(kappa_star[nonzero]) / xi_array[nonzero]
    # Handle xi=0 via L'Hopital: lim Im(kappa*)/xi = group velocity at 0
    zero_mask = ~nonzero
    if np.any(zero_mask):
        # Use the first nonzero point as approximation
        first_nonzero = np.argmax(nonzero)
        c[zero_mask] = c[first_nonzero] if first_nonzero > 0 else 1.0
    return c


def group_velocity_error(
    C: np.ndarray,
    C_exact: float = 1.0,
) -> np.ndarray:
    """Compute relative group velocity error.

    Parameters
    ----------
    C : np.ndarray
        Computed group velocity.
    C_exact : float
        Exact group velocity (default 1.0 for u_t + u_x = 0).

    Returns
    -------
    np.ndarray
        (C - C_exact) / C_exact
    """
    return (C - C_exact) / C_exact


@dataclass
class GroupVelocityProfile:
    """Group velocity analysis results for a single stencil row."""

    xi: np.ndarray
    kappa_star: np.ndarray
    phase_velocity: np.ndarray
    group_velocity: np.ndarray
    gv_error: np.ndarray
    order: int
    cutoff_xi: float  # xi where C first goes to zero or negative


def _build_profile(
    weights,
    i_eval: int,
    node_indices,
    xi_array: np.ndarray,
    order: int,
) -> GroupVelocityProfile:
    """Build a GroupVelocityProfile from stencil weights."""
    w = list(weights)
    nodes = list(node_indices)

    kstar = modified_wavenumber(w, i_eval, nodes, xi_array)
    C = group_velocity_exact(w, i_eval, nodes, xi_array)
    c = phase_velocity(kstar, xi_array)
    gv_err = group_velocity_error(C)

    # Find cutoff: first xi where C <= 0 (skip xi=0)
    cutoff = float(xi_array[-1])
    for idx in range(1, len(xi_array)):
        if C[idx] <= 0.0:
            cutoff = float(xi_array[idx])
            break

    return GroupVelocityProfile(
        xi=xi_array,
        kappa_star=kstar,
        phase_velocity=c,
        group_velocity=C,
        gv_error=gv_err,
        order=order,
        cutoff_xi=cutoff,
    )


def interior_group_velocity(
    p: int,
    nu: int,
    xi_array: np.ndarray,
) -> GroupVelocityProfile:
    """Compute group velocity profile for an interior scheme.

    Parameters
    ----------
    p : int
        RHS half-bandwidth (explicit scheme, s=0).
    nu : int
        Derivative order (1 or 2).
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    GroupVelocityProfile
    """
    from stencil_gen.interior import derive_interior, full_gamma_array

    coeffs = derive_interior(0, p, nu)
    w = [float(c) for c in full_gamma_array(coeffs)]
    nodes = list(range(-p, p + 1))

    return _build_profile(w, 0, nodes, xi_array, order=2 * p)


def boundary_group_velocity(
    p: int,
    q: int,
    nextra: int,
    nu: int,
    sigma: float,
    kernel: str,
    xi_array: np.ndarray,
) -> dict[int, GroupVelocityProfile]:
    """Compute group velocity profiles for all boundary rows.

    Uses RBF/tension boundary weights from :func:`uniform_boundary_weights_rbf`.

    Parameters
    ----------
    p : int
        Interior half-bandwidth.
    q : int
        Polynomial degree for boundary RBF augmentation.
    nextra : int
        Extra boundary rows/columns.
    nu : int
        Derivative order (1 or 2).
    sigma : float
        RBF shape / tension parameter.
    kernel : str
        RBF kernel type (``"tension"``, ``"gaussian"``, ``"multiquadric"``).
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    dict[int, GroupVelocityProfile]
        Keyed by boundary row index (0 to r-1).
    """
    from stencil_gen.phs import uniform_boundary_weights_rbf
    from stencil_gen.temo import compute_dimensions

    dims = compute_dimensions(p, q, 0, nextra, nu)
    r, t = dims.r, dims.t
    nodes = list(range(t))

    profiles: dict[int, GroupVelocityProfile] = {}
    for i in range(r):
        w = uniform_boundary_weights_rbf(i, t, nu, q, sigma, kernel=kernel)
        w_float = [float(c) for c in w]
        profiles[i] = _build_profile(w_float, i, nodes, xi_array, order=q)

    return profiles
