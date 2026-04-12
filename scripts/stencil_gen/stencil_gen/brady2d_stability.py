"""Layered stability scoring for the Brady-Livescu 2D benchmark.

Implements a multi-layer analytical stability pipeline for the Brady & Livescu
2019 §4.3 two-dimensional varying-coefficient scalar advection test.  Each layer
is strictly cheaper than the next, allowing early rejection of unstable schemes.

Layer 1 (this module's first implementation): interior + boundary group velocity
error as a coarse dispersion-quality filter.

Dataclass will be expanded in 41.10a to include layer2 through layer7 fields,
kreiss, and non_normality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from stencil_gen.group_velocity import (
    GroupVelocityProfile,
    boundary_group_velocity,
    boundary_group_velocity_classical,
    interior_group_velocity,
    local_group_velocity_2d_varying,
    max_local_gv_error_2d,
)
from stencil_gen.phs import (
    build_diff_matrix_rbf,
    stability_eigenvalue,
    stability_eigenvalue_from_matrix,
)

logger = logging.getLogger("stencil_gen.brady2d_stability")

# Scheme parameters (duplicated from sweeps._common to avoid circular dep)
_SCHEME_PARAMS = {
    "E2": {"p": 1, "q": 1, "nextra": 1, "nu": 1},
    "E4": {"p": 2, "q": 3, "nextra": 0, "nu": 1},
}

# Layer-1 thresholds
L1_TOL = 0.05  # 5% dispersion error

# Layer-3 threshold: max Re(eigenvalue of -D_bc) must be non-positive
STABILITY_TOL = 1e-10

# Layer-4 threshold: 10% — looser than L1 because the varying-coefficient
# scaling amplifies the baseline dispersion error.
L4_TOL = 0.1

# Fraction of the resolved band over which to measure max |gv_error|.
# Using 10% of the cutoff restricts evaluation to very well-resolved
# wavenumbers where even boundary stencils (especially the outermost
# one-sided row) are expected to be accurate.  This makes L1 a coarse
# filter that only rejects schemes with fundamentally broken dispersion.
_RESOLVED_FRAC = 0.1


@dataclass
class StabilityReport:
    """Result of the layered stability analysis.

    Minimal skeleton — only layer1 fields for now.
    Full fields (layer2 through layer7, kreiss, non_normality) will be
    added in 41.10a.
    """

    layer1: dict | None = None
    failed_layer: int | None = None
    overall_verdict: str = "unknown"
    compute_time: float = 0.0


def _gv_error_scalar(profile: GroupVelocityProfile) -> float:
    """Max absolute GV error in the resolved portion of the spectrum.

    Evaluates over xi in (0, cutoff_xi * _RESOLVED_FRAC], giving a
    conservative measure that focuses on wavenumbers the scheme is expected
    to resolve well.
    """
    xi_max = profile.cutoff_xi * _RESOLVED_FRAC
    mask = (profile.xi > 0) & (profile.xi <= xi_max)
    if not np.any(mask):
        return 1.0  # no resolved wavenumbers → maximum error
    return float(np.max(np.abs(profile.gv_error[mask])))


def _derive_classical_boundary(p: int, nu: int, alpha_list: list[float]):
    """Derive classical boundary rows with conservation and substitute alphas.

    Parameters
    ----------
    p : int
        Interior half-bandwidth.
    nu : int
        Derivative order.
    alpha_list : list[float]
        Alpha values ordered by symbol index (alpha_0, alpha_1, ...).

    Returns
    -------
    (updated_rows, alpha_values_dict)
        updated_rows: list[BoundaryRow] with conservation applied.
        alpha_values_dict: {Symbol: float} mapping for substitution.
    """
    from stencil_gen.boundary import derive_boundary
    from stencil_gen.conservation import build_conservation_system, solve_conservation

    result = derive_boundary(p=p, nu=nu, s=0)
    equations, w_syms, last_free = build_conservation_system(
        result.r, result.t, p, result.rows, result.interior_coeffs,
    )
    _, updated_rows = solve_conservation(
        equations, w_syms, last_free, result.all_free_params, result.rows,
    )
    alpha_values = dict(zip(result.all_free_params, alpha_list))
    return updated_rows, alpha_values


def layer1_interior_boundary_gv(
    scheme: str,
    kernel: str,
    params: dict,
    n_xi: int = 200,
) -> dict:
    """L1: Interior + boundary group velocity error (1D, per direction).

    Checks dispersion quality by computing GV error profiles for the interior
    and boundary stencils, then reducing each to a single scalar (max absolute
    error over the well-resolved portion of the spectrum).

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.  For classical: {"alpha": [float, ...]}.
        For RBF kernels: {"sigma": float} or {"epsilon": float}.
    n_xi : int
        Number of wavenumber samples in [0.01, pi].

    Returns
    -------
    dict with keys:
        interior_gv_err_x : float
            Max |gv_error| for interior stencil over resolved wavenumbers.
        interior_gv_err_y : float
            Same as x (Cartesian grid → identical stencil in both directions).
        boundary_gv_err : float
            Max over all boundary rows of max |gv_error| over resolved band.
        cutoff_fraction : float
            min(cutoff_xi) over boundary rows / pi.
    """
    sp = _SCHEME_PARAMS[scheme]
    p, q, nextra, nu = sp["p"], sp["q"], sp["nextra"], sp["nu"]
    xi_array = np.linspace(0.01, np.pi, n_xi)

    # Interior GV profile (same stencil for x and y on Cartesian grid)
    interior_prof = interior_group_velocity(p, nu, xi_array)
    interior_err = _gv_error_scalar(interior_prof)

    # Boundary GV profiles
    if kernel == "classical":
        alpha_list = params["alpha"]
        boundary_rows, alpha_values = _derive_classical_boundary(p, nu, alpha_list)
        boundary_profiles = boundary_group_velocity_classical(
            boundary_rows, alpha_values, order=q, xi_array=xi_array,
        )
    else:
        # RBF kernels: sigma for tension, epsilon for gaussian/multiquadric
        sigma = params.get("sigma", params.get("epsilon", 0.0))
        boundary_profiles = boundary_group_velocity(
            p, q, nextra, nu, sigma, kernel, xi_array,
        )

    # Scalar reductions
    boundary_errs = [_gv_error_scalar(prof) for prof in boundary_profiles.values()]
    max_boundary_err = max(boundary_errs) if boundary_errs else 0.0

    cutoffs = [prof.cutoff_xi for prof in boundary_profiles.values()]
    min_cutoff = min(cutoffs) if cutoffs else 0.0
    cutoff_frac = min_cutoff / np.pi

    return {
        "interior_gv_err_x": interior_err,
        "interior_gv_err_y": interior_err,
        "boundary_gv_err": max_boundary_err,
        "cutoff_fraction": cutoff_frac,
    }


def _build_classical_diff_matrix(
    n: int,
    p: int,
    nu: int,
    alpha_list: list[float],
) -> np.ndarray:
    """Build an n×n differentiation matrix for the classical-alpha family.

    Uses the TEMO boundary derivation with conservation enforcement,
    substitutes alpha values, and assembles the full matrix with
    antisymmetric (nu=1) right boundary closure.

    Requires p >= 2 (E4+).  For E2 (p=1) the boundary closure has zero free
    alpha parameters — ``derive_boundary(p=1)`` computes a negative symbol
    count and crashes.  Use the RBF path (``kernel="tension"``, ``sigma=0.0``)
    for E2 instead.
    """
    from stencil_gen.interior import derive_interior, full_gamma_array

    boundary_rows, alpha_values = _derive_classical_boundary(p, nu, alpha_list)

    # Dimensions come from the boundary derivation itself
    r = len(boundary_rows)
    t = len(boundary_rows[0].coefficients)

    # Build numeric boundary block
    B = np.zeros((r, t))
    for i, brow in enumerate(boundary_rows):
        for j, coeff in enumerate(brow.coefficients):
            B[i, j] = float(coeff.subs(alpha_values))

    # Interior weights
    interior_coeffs = derive_interior(0, p, nu)
    interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

    # Assemble full matrix
    D = np.zeros((n, n))
    # Left boundary
    for i in range(r):
        for j in range(t):
            D[i, j] = B[i, j]
    # Interior
    for i in range(r, n - r):
        for k_idx, j in enumerate(range(i - p, i + p + 1)):
            D[i, j] = interior_w[k_idx]
    # Right boundary (antisymmetric for nu=1)
    sign = -1 if nu % 2 == 1 else 1
    for i in range(r):
        row = n - 1 - i
        for j in range(t):
            D[row, n - 1 - j] = sign * B[i, j]
    return D


def layer3_1d_eigenvalue(
    scheme: str,
    kernel: str,
    params: dict,
    n_values: tuple[int, ...] = (20, 40, 80),
) -> dict:
    """L3: 1D eigenvalue stability check at multiple grid sizes.

    For each grid size n, computes the maximum real part of eigenvalues of
    -D_bc (the semi-discrete advection operator with inflow BC removed).
    A non-positive value means the 1D constant-coefficient scheme is stable.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.  For classical: {"alpha": [float, ...]}.
        For RBF kernels: {"sigma": float} or {"epsilon": float}.
    n_values : tuple[int, ...]
        Grid sizes at which to evaluate stability.

    Returns
    -------
    dict with keys:
        eigenvalues : dict[int, float]
            {n: max_real_eigenvalue} for each grid size.
        max_stab_eig : float
            Maximum over all grid sizes.
    """
    sp = _SCHEME_PARAMS[scheme]
    p, q, nextra, nu = sp["p"], sp["q"], sp["nextra"], sp["nu"]

    eigenvalues = {}
    for n in n_values:
        if kernel == "classical":
            alpha_list = params["alpha"]
            D = _build_classical_diff_matrix(n, p, nu, alpha_list)
            se = stability_eigenvalue_from_matrix(D)
        else:
            epsilon = params.get("sigma", params.get("epsilon", 0.0))
            se = stability_eigenvalue(n, p, q, epsilon, kernel, nu, nextra)
        eigenvalues[n] = se

    return {
        "eigenvalues": eigenvalues,
        "max_stab_eig": max(eigenvalues.values()),
    }


def layer4_local_gv_2d(
    scheme: str,
    kernel: str,
    params: dict,
    N: int = 31,
) -> dict:
    """L4: per-point local group velocity error on the Brady-Livescu 2D field.

    Freezes coefficients at each grid point and evaluates the interior stencil's
    group velocity error scaled by the local wave speed.  This is the first-order
    WKB approximation to the varying-coefficient dispersion error.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type (unused for interior stencil — interior weights are
        scheme-determined — but kept for API consistency with other layers).
    params : dict
        Kernel-specific parameters (unused at this layer).
    N : int
        Grid resolution for the coefficient field.

    Returns
    -------
    dict with keys:
        max_local_gv_error : float
            Maximum absolute local GV error over all grid points and wavenumbers.
        worst_point : tuple[int, int]
            (i, j) indices of the grid point with the largest error.
        worst_xi : float
            Wavenumber at which the largest error occurs.
    """
    from stencil_gen.benchmarks.brady_livescu_2d import make_coefficient_field
    from stencil_gen.interior import derive_interior, full_gamma_array

    sp = _SCHEME_PARAMS[scheme]
    p, nu = sp["p"], sp["nu"]

    # Build interior stencil (same for x and y on Cartesian grid)
    coeffs = derive_interior(0, p, nu)
    w = np.array([float(c) for c in full_gamma_array(coeffs)])
    offsets = np.arange(-p, p + 1, dtype=float)
    stencil = (w, offsets)

    # Compute the interior GV profile to determine the resolved band cutoff
    profile = interior_group_velocity(p, nu, np.linspace(0.01, np.pi, 200))
    xi_max = profile.cutoff_xi * _RESOLVED_FRAC
    xi_array = np.linspace(0.01, xi_max, 200)

    _, _, c_x, c_y = make_coefficient_field(N)

    result = local_group_velocity_2d_varying(stencil, stencil, c_x, c_y, xi_array)

    # Compute max absolute error over all points and the resolved band
    err_x = np.abs(result["gv_error_x_field"])
    err_y = np.abs(result["gv_error_y_field"])
    combined = np.maximum(err_x, err_y)
    max_err = float(np.max(combined))

    # Find the worst point and wavenumber
    flat_idx = int(np.argmax(combined))
    i, j, k = np.unravel_index(flat_idx, combined.shape)

    return {
        "max_local_gv_error": max_err,
        "worst_point": (int(i), int(j)),
        "worst_xi": float(xi_array[k]),
    }
