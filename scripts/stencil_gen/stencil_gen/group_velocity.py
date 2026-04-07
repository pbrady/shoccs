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


def modified_wavenumber_nonuniform(
    weights,
    offsets,
    xi_array: np.ndarray,
) -> np.ndarray:
    """Compute modified wavenumber for non-uniformly spaced stencil nodes.

    Generalization of :func:`modified_wavenumber` for real-valued offsets
    (e.g. cut-cell stencils where the wall position is at a fractional
    distance from the evaluation point).

    Parameters
    ----------
    weights : array-like
        Stencil coefficients w_j.
    offsets : array-like of float
        Normalized distances (x_j - x_i)/h from the evaluation point.
        May be non-integer for cut-cell grids.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    np.ndarray (complex)
        kappa*(xi) = sum_j w_j exp(i * offset_j * xi)
    """
    w = np.asarray(weights, dtype=complex)
    d = np.asarray(offsets, dtype=float)
    phase = np.exp(1j * np.outer(xi_array, d))
    return phase @ w


def group_velocity_exact_nonuniform(
    weights,
    offsets,
    xi_array: np.ndarray,
) -> np.ndarray:
    """Compute group velocity analytically for non-uniform offsets.

    C(xi) = Re(sum_j w_j * offset_j * exp(i * offset_j * xi))

    Generalization of :func:`group_velocity_exact` for real-valued offsets.

    Parameters
    ----------
    weights : array-like
        Stencil coefficients w_j.
    offsets : array-like of float
        Normalized distances (x_j - x_i)/h from the evaluation point.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    np.ndarray (real)
        Group velocity C(xi).
    """
    w = np.asarray(weights, dtype=complex)
    d = np.asarray(offsets, dtype=float)
    phase = np.exp(1j * np.outer(xi_array, d))
    return np.real(phase @ (w * d))


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
    cutoff_xi: float  # first xi beyond which C stays permanently non-positive


@dataclass
class GKSModeInfo:
    """Diagnostic information for a boundary-localized eigenmode.

    Bridges per-stencil group velocity analysis with full-operator eigenvalue
    analysis by identifying nearly-neutral eigenmodes concentrated near a
    boundary and checking whether the interior stencil's group velocity at
    the mode's dominant wavenumber directs energy into the domain.
    """

    eigenvalue: complex  # eigenvalue of -D_bc
    boundary_wavenumber: float  # dominant xi from FFT of boundary eigenvector portion
    group_velocity: float  # interior C(xi) at boundary_wavenumber
    is_outgoing: bool  # True if mode radiates energy from boundary into domain


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

    # Find cutoff: first xi beyond which C stays non-positive.
    # Scan from high end to handle non-monotonic boundary stencils where
    # C(xi) may dip below zero briefly then recover.
    last_positive_idx = 0
    for idx in range(1, len(xi_array)):
        if C[idx] > 0.0:
            last_positive_idx = idx
    if last_positive_idx + 1 < len(xi_array):
        cutoff = float(xi_array[last_positive_idx + 1])
    else:
        cutoff = float(xi_array[-1])

    return GroupVelocityProfile(
        xi=xi_array,
        kappa_star=kstar,
        phase_velocity=c,
        group_velocity=C,
        gv_error=gv_err,
        order=order,
        cutoff_xi=cutoff,
    )


def _build_profile_nonuniform(
    weights,
    offsets,
    xi_array: np.ndarray,
    order: int,
) -> GroupVelocityProfile:
    """Build a GroupVelocityProfile from stencil weights with non-uniform offsets."""
    kstar = modified_wavenumber_nonuniform(weights, offsets, xi_array)
    C = group_velocity_exact_nonuniform(weights, offsets, xi_array)
    c = phase_velocity(kstar, xi_array)
    gv_err = group_velocity_error(C)

    last_positive_idx = 0
    for idx in range(1, len(xi_array)):
        if C[idx] > 0.0:
            last_positive_idx = idx
    if last_positive_idx + 1 < len(xi_array):
        cutoff = float(xi_array[last_positive_idx + 1])
    else:
        cutoff = float(xi_array[-1])

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


def boundary_group_velocity_classical(
    boundary_rows,
    alpha_values: dict,
    order: int,
    xi_array: np.ndarray,
) -> dict[int, GroupVelocityProfile]:
    """Compute group velocity profiles for classical (non-RBF) boundary rows.

    Takes the symbolic ``BoundaryRow`` list from :func:`derive_boundary` (or the
    conservation-updated rows from :func:`solve_conservation`) and substitutes
    concrete alpha values to obtain numerical stencil coefficients.

    Parameters
    ----------
    boundary_rows : list[BoundaryRow]
        Boundary rows with symbolic coefficients in alpha parameters.
    alpha_values : dict
        Mapping from alpha symbols to numeric values (e.g.,
        ``{alpha_0: -0.77, alpha_1: 0.16}``).
    order : int
        Polynomial accuracy order of the boundary scheme (q = 2*(p+s) - 1
        for the classical Brady & Livescu stencils).
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    dict[int, GroupVelocityProfile]
        Keyed by boundary row index.
    """
    t = len(boundary_rows[0].coefficients)
    nodes = list(range(t))

    profiles: dict[int, GroupVelocityProfile] = {}
    for row in boundary_rows:
        i = row.row_index
        w_float = [float(c.xreplace(alpha_values))
                   if hasattr(c, 'xreplace') else float(c)
                   for c in row.coefficients]
        profiles[i] = _build_profile(w_float, i, nodes, xi_array, order=order)

    return profiles


def cut_cell_group_velocity(
    cut_cell_result,
    psi_sym,
    psi_val: float,
    alpha_values: dict,
    xi_array: np.ndarray,
    order: int | None = None,
) -> dict[int, GroupVelocityProfile]:
    """Compute group velocity profiles for all rows of a cut-cell stencil.

    Evaluates the symbolic psi-dependent stencil coefficients at a specific
    ``psi_val`` and ``alpha_values``, then computes group velocity profiles
    using the non-uniform offsets from the cut-cell grid geometry.

    Parameters
    ----------
    cut_cell_result : CutCellResult
        Precomputed symbolic cut-cell stencil (from ``derive_cut_cell_mathematica``
        or ``derive_cut_cell_scheme``).
    psi_sym : Symbol
        The SymPy symbol for psi used in ``cut_cell_result``.
    psi_val : float
        Numeric psi value in [0, 1].
    alpha_values : dict
        Mapping from alpha symbols to numeric values.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].
    order : int, optional
        Polynomial accuracy order.  Defaults to the scheme's boundary
        accuracy ``q`` inferred from the dimensions.

    Returns
    -------
    dict[int, GroupVelocityProfile]
        Keyed by row index (0 to R-1) of the floating stencil.
    """
    F = cut_cell_result.floating
    dims = cut_cell_result.dims
    R, T = F.rows, F.cols

    if order is None:
        # dims.r = q + 1 + nextra, but nextra is not recoverable from dims
        # alone, so this overestimates q when nextra > 0.  Callers with
        # nextra > 0 should pass order explicitly.
        order = max(1, dims.r - 1)

    subs = {psi_sym: psi_val, **alpha_values}

    profiles: dict[int, GroupVelocityProfile] = {}
    for i in range(R):
        # Evaluate symbolic coefficients numerically
        w = [float(F[i, j].xreplace(subs)) for j in range(T)]

        # Non-uniform offsets: wall at -(psi_val + i), grid points at j - i
        offsets = [-(psi_val + i)] + [j - i for j in range(T - 1)]

        profiles[i] = _build_profile_nonuniform(w, offsets, xi_array, order=order)

    return profiles


def gks_group_velocity_check(
    D: np.ndarray,
    xi_array: np.ndarray,
    neutral_tol: float = 0.1,
    localization_tol: float = 0.3,
) -> list[GKSModeInfo]:
    """Identify boundary modes whose group velocity indicates GKS-type instability.

    For the advection equation u_t + u_x = 0 semi-discretized as du/dt = -Du,
    computes eigenvalues and eigenvectors of -D_bc (D with inflow row/column
    removed).  Identifies boundary-localized, nearly-neutral eigenmodes and
    checks whether the interior stencil's group velocity at each mode's dominant
    wavenumber directs energy from the boundary into the domain — the hallmark
    of GKS instability (Trefethen 1983).

    Parameters
    ----------
    D : np.ndarray
        Full N×N differentiation matrix (approximating d/dx).
    xi_array : np.ndarray
        Wavenumber array in [0, pi] for group velocity evaluation.
    neutral_tol : float
        Fraction of max|Re(lambda)| below which an eigenvalue is considered
        nearly-neutral.  Default 0.1.
    localization_tol : float
        Minimum fraction of eigenvector energy in the boundary region
        required to classify a mode as boundary-localized.  Default 0.3.

    Returns
    -------
    list[GKSModeInfo]
        One entry per boundary-localized, nearly-neutral eigenmode.
    """
    n = D.shape[0]
    D_bc = D[1:, 1:]  # remove inflow row/column (Dirichlet at x=0)
    m = D_bc.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(-D_bc)

    # Nearly-neutral threshold
    max_abs_real = np.max(np.abs(eigenvalues.real))
    threshold = max(neutral_tol * max_abs_real, 1e-10)

    # Interior group velocity profile from D's middle row
    mid = n // 2
    row = D[mid, :]
    cols = np.nonzero(np.abs(row) > 1e-15)[0]
    C_profile = group_velocity_exact(row[cols], mid, cols, xi_array)

    # Boundary region width: ~1/4 of domain, at least 4 points
    bw = max(4, m // 4)

    results: list[GKSModeInfo] = []
    for idx in range(len(eigenvalues)):
        lam = eigenvalues[idx]

        # Skip conjugate duplicates: keep the positive-imaginary member
        if lam.imag < -1e-10:
            continue

        if abs(lam.real) > threshold:
            continue

        v = eigenvectors[:, idx]
        energy = np.abs(v) ** 2
        total = np.sum(energy)
        if total < 1e-30:
            continue

        # Check boundary localization
        left_frac = np.sum(energy[:bw]) / total
        right_frac = np.sum(energy[-bw:]) / total

        if max(left_frac, right_frac) < localization_tol:
            continue  # interior mode, not boundary-localized

        # Use the side where more energy is concentrated
        if left_frac >= right_frac:
            portion = v[:bw]
            side = "left"
        else:
            portion = v[-bw:]
            side = "right"

        # Estimate dominant wavenumber via zero-padded FFT
        pad_len = max(256, 4 * len(portion))
        spec = np.abs(np.fft.fft(portion, n=pad_len))
        nyq = pad_len // 2
        # Skip DC (k=0); search bins 1..nyq for the peak
        peak = int(np.argmax(spec[1 : nyq + 1])) + 1
        dom_xi = float(peak * 2 * np.pi / pad_len)

        # Interpolate interior group velocity at the dominant wavenumber
        C_val = float(np.interp(dom_xi, xi_array, C_profile))

        # Outgoing = energy radiating from boundary into domain interior.
        # Left boundary: rightward (C > 0) enters the domain.
        # Right boundary: leftward (C < 0) enters the domain.
        if side == "left":
            is_out = C_val > 0
        else:
            is_out = C_val < 0

        results.append(
            GKSModeInfo(
                eigenvalue=lam,
                boundary_wavenumber=dom_xi,
                group_velocity=C_val,
                is_outgoing=is_out,
            )
        )

    return results
