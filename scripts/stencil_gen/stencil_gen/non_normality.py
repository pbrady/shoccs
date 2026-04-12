"""Non-normality diagnostics for semi-discrete spatial operators.

Computes spectral and pseudospectral stability metrics that detect
transient growth not visible from eigenvalues alone.  Calibration bands
follow Trefethen & Embree, *Spectra and Pseudospectra* (2005), ch. 14.

Metrics provided:
- spectral abscissa: max Re(lambda) — asymptotic stability indicator
- numerical abscissa: max eigenvalue of (L + L^T)/2 — instantaneous growth rate
- Henrici departure from normality: ||LL^T - L^TL||_F / ||L||_F^2
- eigenvector condition number: cond(V) where L = V diag(lambda) V^{-1}
- pseudospectral abscissa: max Re(s) such that sigma_min(sI - L) <= epsilon
- Kreiss constant: max Re(s)/sigma_min(sI - L) for Re(s) > 0
- transient growth bound: e * Kreiss constant (Kreiss matrix theorem)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("stencil_gen.non_normality")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonNormalityReport:
    """Collected non-normality diagnostics for a spatial operator."""

    spectral_abscissa: float
    """max Re(lambda(L)) — asymptotic stability indicator."""

    numerical_abscissa: float
    """max eigenvalue of (L + L^T)/2 — instantaneous growth rate."""

    henrici_departure: float
    """||LL^T - L^TL||_F / ||L||_F^2 — departure from normality."""

    eigenvector_condition: float
    """cond(V) where L = V diag(lambda) V^{-1}.  NaN if N too large."""

    pseudospectral_abscissae: dict[float, float]
    """epsilon -> alpha_epsilon for each requested epsilon."""

    kreiss_constant: float
    """max Re(s) / sigma_min(sI - L) over Re(s) > 0."""

    transient_growth_bound: float
    """e * kreiss_constant — upper bound on max_{t>=0} ||exp(Lt)||."""

    n: int
    """Matrix dimension."""

    compute_time: float
    """Wall-clock seconds for the full computation."""

    notes: list[str] = field(default_factory=list)
    """Diagnostic notes (convergence warnings, fallbacks, etc.)."""


# ---------------------------------------------------------------------------
# Individual metric functions (stubs — implemented in 41.8b through 41.8e)
# ---------------------------------------------------------------------------


def spectral_abscissa_sparse(L, k: int = 20, shift_invert: bool = True):
    """Compute max Re(lambda) of sparse matrix L via Arnoldi iteration.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    k : int
        Number of eigenvalues to compute.
    shift_invert : bool
        Whether to use shift-invert mode on ArpackNoConvergence.

    Returns
    -------
    tuple[float, np.ndarray]
        (max_real_part, all_computed_eigenvalues)
    """
    raise NotImplementedError("spectral_abscissa_sparse: 41.8b")


def numerical_abscissa_sparse(L) -> float:
    """Compute max eigenvalue of (L + L^T)/2 — the numerical abscissa.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.

    Returns
    -------
    float
        The numerical abscissa (instantaneous growth rate).
    """
    raise NotImplementedError("numerical_abscissa_sparse: 41.8c")


def henrici_departure(L) -> float:
    """Compute Henrici departure from normality: ||LL^T - L^TL||_F / ||L||_F^2.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.

    Returns
    -------
    float
        Non-negative scalar; 0 for normal operators.
    """
    raise NotImplementedError("henrici_departure: 41.8c")


def eigenvector_condition(L, small_dense_threshold: int = 900) -> float:
    """Compute condition number of the eigenvector matrix.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    small_dense_threshold : int
        If N > threshold, return np.nan (too expensive for dense eig).

    Returns
    -------
    float
        cond(V) where L = V diag(lambda) V^{-1}, or np.nan if N too large.
    """
    raise NotImplementedError("eigenvector_condition: 41.8c")


def _sigma_field(L, s_grid: np.ndarray) -> np.ndarray:
    """Compute sigma_min(sI - L) over a grid of complex s values.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    s_grid : np.ndarray
        Complex-valued array of s points (arbitrary shape).

    Returns
    -------
    np.ndarray
        Array of same shape as s_grid with sigma_min values.
    """
    raise NotImplementedError("_sigma_field: 41.8d")


def pseudospectral_abscissa_estimate(
    L, epsilon_values, s_grid: np.ndarray
) -> dict[float, float]:
    """Estimate pseudospectral abscissa for each epsilon.

    For each epsilon, alpha_epsilon = max Re(s) such that sigma_min(sI - L) <= epsilon.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    epsilon_values : sequence of float
        Perturbation levels.
    s_grid : np.ndarray
        Complex-valued grid for the search.

    Returns
    -------
    dict[float, float]
        epsilon -> alpha_epsilon.  -inf if no grid point satisfies.
    """
    raise NotImplementedError("pseudospectral_abscissa_estimate: 41.8e")


def kreiss_constant_estimate(L, s_grid: np.ndarray) -> float:
    """Estimate Kreiss constant: max Re(s) / sigma_min(sI - L) for Re(s) > 0.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    s_grid : np.ndarray
        Complex-valued grid for the search.

    Returns
    -------
    float
        Estimated Kreiss constant.
    """
    raise NotImplementedError("kreiss_constant_estimate: 41.8e")


# ---------------------------------------------------------------------------
# Orchestrator (stub — implemented in 41.8f)
# ---------------------------------------------------------------------------


def compute_non_normality(
    L,
    *,
    small_dense_threshold: int = 900,
    epsilon_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
    s_grid_params: dict | None = None,
) -> NonNormalityReport:
    """Compute all non-normality diagnostics for a spatial operator.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    small_dense_threshold : int
        Threshold for dense-only computations (eigenvector condition).
    epsilon_values : tuple of float
        Perturbation levels for pseudospectral abscissa.
    s_grid_params : dict or None
        Parameters for the resolvent s-grid.  If None, uses defaults
        based on the spectral abscissa.

    Returns
    -------
    NonNormalityReport
        Fully populated report with all metrics.
    """
    raise NotImplementedError("compute_non_normality: 41.8f")
