"""Rigorous GKS Kreiss determinant stability test for semi-discrete boundary closures.

Implements the Kreiss stability condition from Trefethen 1983 (pp. 206-207,
"Group velocity interpretation of the stability theory of Gustafsson, Kreiss,
and Sundstrom", *J. Comput. Phys.* 49, pp. 199-217).

For the semi-discrete left-boundary problem u_t = -sum_k a_k u_{n+k}, a mode
u_n(t) = e^(st) kappa^n requires s + sum_k a_k kappa^k = 0. For each s in
the right half-plane, the admissible roots are those with |kappa| < 1. The
r x r Kreiss matrix M(s) is built from the r admissible roots and the r
boundary rows, and sigma_min(M(s)) is the Kreiss determinant condition
indicator. The boundary closure is GKS-stable iff sigma_min(M(s)) > 0 for
all s with Re(s) >= 0, with imaginary-axis perturbation used to classify
tangent modes per Trefethen p. 207.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("stencil_gen.gks_kreiss")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BoundaryRow = tuple[np.ndarray, np.ndarray]
"""(weights, column_offsets) for one boundary row of the closure."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DefectiveKappaError(RuntimeError):
    """Raised when admissible kappa roots are defective (repeated)."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KreissResult:
    """Result of the rigorous Kreiss determinant stability check."""

    is_stable: bool
    """True if no sigma_min violation was found in the sampled s-grid."""

    witness_s: Optional[complex] = None
    """The s value at which the minimum sigma_min was found (None if stable)."""

    witness_sigma_min: float = float("inf")
    """The minimum singular value at the witness point."""

    imaginary_axis_perturbation_verdict: str = "not_checked"
    """One of: 'not_checked', 'no_candidates', 'all_incoming',
    'outgoing_mode_detected', 'defective'."""

    defective_kappa_detected: bool = False
    """True if defective (repeated) admissible kappa roots were encountered."""

    s_grid_shape: tuple[int, ...] = ()
    """Shape of the s-grid used for the sweep."""

    compute_time: float = 0.0
    """Wall-clock time in seconds for the full check."""

    sigma_min_field: Optional[np.ndarray] = field(default=None, repr=False)
    """The sigma_min values over the entire s-grid (for diagnostics)."""

    s_grid: Optional[np.ndarray] = field(default=None, repr=False)
    """The complex s-grid used for the sweep."""

    n_admissible_roots: int = 0
    """Number of admissible kappa roots (|kappa| < 1) at the witness point."""


# ---------------------------------------------------------------------------
# Primitive functions (implemented in 41.3b-41.3f)
# ---------------------------------------------------------------------------


def kappa_roots(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    s: complex,
    *,
    repeat_tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Find all roots kappa of the characteristic polynomial Q(kappa) = 0.

    For the interior stencil u_t = -sum_k a_k u_{n+k}, the mode ansatz
    u_n(t) = e^(st) kappa^n yields s + sum_k a_k kappa^k = 0. Multiply
    through by kappa^L_left to clear negative powers.

    Returns
    -------
    all_roots : np.ndarray
        All roots of Q(kappa).
    admissible : np.ndarray
        Roots with |kappa| < 1 (strictly inside the unit disk).
    is_defective : bool
        True if any pair of admissible roots has separation < repeat_tol.
    """
    offsets = np.asarray(interior_offsets, dtype=int)
    weights = np.asarray(interior_weights, dtype=complex)
    L_left = -int(np.min(offsets))
    shifted = offsets + L_left  # now all >= 0

    # Polynomial degree
    degree = int(np.max(shifted))
    # If the s*kappa^L_left term contributes at a higher degree, extend
    degree = max(degree, L_left)

    # Build coefficient array: Q(kappa) = sum_k a_k * kappa^shifted[k] + s * kappa^L_left
    # numpy.roots expects coefficients from highest to lowest degree
    coeffs = np.zeros(degree + 1, dtype=complex)
    for w, sh in zip(weights, shifted):
        coeffs[sh] += w
    coeffs[L_left] += s

    # numpy.roots expects [c_n, c_{n-1}, ..., c_1, c_0] (highest degree first)
    poly_coeffs = coeffs[::-1]

    all_roots = np.roots(poly_coeffs)

    # Admissible: strictly inside the unit disk
    admissible = all_roots[np.abs(all_roots) < 1.0 - 1e-12]

    # Defective check: any pair of admissible roots with separation < repeat_tol
    is_defective = False
    if len(admissible) > 1:
        for i in range(len(admissible)):
            for j in range(i + 1, len(admissible)):
                if abs(admissible[i] - admissible[j]) < repeat_tol:
                    is_defective = True
                    break
            if is_defective:
                break

    return all_roots, admissible, is_defective


def _kappa_roots_from_poly(
    poly_coeffs: np.ndarray,
    *,
    repeat_tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Find roots from explicit polynomial coefficients (highest degree first).

    Helper for testing defective-root detection without constructing stencils.

    Returns same tuple as kappa_roots: (all_roots, admissible, is_defective).
    """
    all_roots = np.roots(poly_coeffs)
    admissible = all_roots[np.abs(all_roots) < 1.0 - 1e-12]

    is_defective = False
    if len(admissible) > 1:
        for i in range(len(admissible)):
            for j in range(i + 1, len(admissible)):
                if abs(admissible[i] - admissible[j]) < repeat_tol:
                    is_defective = True
                    break
            if is_defective:
                break

    return all_roots, admissible, is_defective


def kreiss_matrix(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s: complex,
) -> np.ndarray:
    """Build the r x r Kreiss matrix M(s) from admissible roots and boundary rows.

    M[i, ell] = s * kappa_ell^i + sum_j w_{ij} * kappa_ell^j

    where i indexes boundary rows and ell indexes admissible kappa roots.

    Raises ValueError if len(admissible) != len(boundary_rows).
    """
    raise NotImplementedError


def min_singular_value(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s: complex,
) -> float:
    """Compute sigma_min(M(s)), the minimum singular value of the Kreiss matrix.

    Returns np.inf on DefectiveKappaError or shape mismatch.
    """
    raise NotImplementedError


def make_s_grid(
    s_max: float = 10.0,
    n_radial: int = 40,
    n_imag: int = 120,
    imag_max: float = 20.0,
    eps_imag: float = 1e-6,
) -> np.ndarray:
    """Build an L-shaped contour grid in the right half of the complex s-plane.

    The grid combines a logarithmically-spaced radial sweep with a dense
    imaginary-axis strip at Re(s) = eps_imag, covering Im(s) in
    [-imag_max, imag_max].
    """
    raise NotImplementedError


def _sweep_grid(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s_grid: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Evaluate min_singular_value at every point of s_grid.

    Returns
    -------
    sigma_field : np.ndarray
        sigma_min values, same shape as s_grid.
    argmin_idx : int
        Flat index of the global minimum in sigma_field.
    """
    raise NotImplementedError


def _refine_witness(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s_start: complex,
) -> complex:
    """Refine a candidate witness s via Nelder-Mead minimization of log(sigma_min).

    Constrains Re(s) >= 0 by reflecting into the right half-plane.
    """
    raise NotImplementedError


def _classify_imag_axis(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    s_candidate: complex,
    delta: float = 1e-4,
) -> str:
    """Classify a near-imaginary-axis candidate s via kappa perturbation.

    For a candidate s_0 near the imaginary axis, recompute kappa roots at
    s_0 and s_0 + delta, match by nearest-neighbor, and classify
    unit-modulus kappas as incoming or outgoing per Trefethen p. 207.

    Note on naming convention: "outgoing_mode_detected" means a physical
    outgoing mode was found at the boundary — this is a GKS violation
    (instability). "all_incoming" means all near-unit-circle modes move
    outward under perturbation — this is the stable (non-violation) case.

    Returns
    -------
    str
        One of: 'no_candidates', 'all_incoming', 'outgoing_mode_detected',
        'defective'.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Orchestrator (implemented in 41.3g)
# ---------------------------------------------------------------------------


def kreiss_stability_check(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    *,
    s_grid_params: Optional[dict] = None,
    sigma_tol: float = 1e-8,
    refine: bool = True,
) -> KreissResult:
    """Run the full Kreiss determinant stability check.

    Sweeps the s-grid, optionally refines any witness, classifies
    imaginary-axis modes, and returns a KreissResult.

    Parameters
    ----------
    interior_weights : np.ndarray
        Weights of the interior finite-difference stencil.
    interior_offsets : np.ndarray
        Grid offsets of the interior stencil (e.g., [-2, -1, 0, 1, 2]).
    boundary_rows : list[BoundaryRow]
        Each element is (weights, column_offsets) for one boundary row.
    s_grid_params : dict, optional
        Keyword arguments passed to make_s_grid.
    sigma_tol : float
        Threshold below which sigma_min indicates instability.
    refine : bool
        Whether to refine the witness via Nelder-Mead.
    """
    raise NotImplementedError
