"""TEMO (Truncation Error Matching Optimization) cut-cell stencil extension.

Implements the TEMO procedure from Brady & Livescu (2021) for deriving
psi-parameterized cut-cell boundary stencils from uniform boundary stencils.
"""

from dataclasses import dataclass
from typing import NamedTuple

from sympy import Matrix, Rational, Symbol, factorial


class Dimensions(NamedTuple):
    """Stencil dimensions for uniform and cut-cell cases."""

    r: int  # uniform boundary rows
    t: int  # uniform boundary columns
    R: int  # cut-cell rows
    T: int  # cut-cell columns (including wall)
    X: int  # Neumann extra rows


def compute_dimensions(p: int, q: int, s: int, nextra: int, nu: int) -> Dimensions:
    """Compute stencil dimensions from scheme parameters.

    Uses Eq. 11a/11b from Brady & Livescu (2021):
        t = p + q + 1 + nextra     (stencil width)
        r = q + 1 + nextra         (number of boundary rows)

    For cut-cell stencils:
        R = r_eff + 1, T = t + 1

    where r_eff = r for 1st derivatives, r_eff = r - 1 for 2nd derivatives
    (the last uniform boundary row overlaps with the first interior row).

    Note: verified for E2 schemes only. E4 schemes may require a different
    derivation (see D-R25 in meta.md).

    Parameters
    ----------
    p : int
        Interior half-width (RHS bandwidth).
    q : int
        Boundary accuracy order.
    s : int
        LHS half-width (0 for explicit, 1 for tridiagonal compact).
    nextra : int
        Extra rows/columns for numerical optimization.
    nu : int
        Derivative order (1 or 2).

    Returns
    -------
    Dimensions
        Named tuple (r, t, R, T, X).
    """
    t = p + q + 1 + nextra
    r = q + 1 + nextra

    if nu == 1:
        r_eff = r
    elif nu == 2:
        r_eff = r - 1
    else:
        raise ValueError(f"Unsupported derivative order nu={nu}; must be 1 or 2")

    R = r_eff + 1
    T = t + 1
    X = R if nu == 2 else 0

    return Dimensions(r=r, t=t, R=R, T=T, X=X)


@dataclass(frozen=True)
class SchemeParams:
    """Scheme parameters from Table 1 of Brady & Livescu (2021).

    Parameters
    ----------
    p : int
        Interior half-width (RHS bandwidth).
    q : int
        Boundary accuracy order.
    s : int
        LHS half-width (0 for explicit, 1 for tridiagonal compact).
    nextra : int
        Extra rows/columns for numerical optimization.
    nu : int
        Derivative order (1 or 2).
    """

    p: int
    q: int
    s: int
    nextra: int
    nu: int

    def dims(self) -> Dimensions:
        """Compute stencil dimensions for this scheme."""
        return compute_dimensions(self.p, self.q, self.s, self.nextra, self.nu)


# Pre-defined schemes from Table 1
E2_1 = SchemeParams(p=1, q=1, s=0, nextra=1, nu=1)
E2_2 = SchemeParams(p=1, q=1, s=0, nextra=0, nu=2)
E4_1 = SchemeParams(p=2, q=3, s=0, nextra=0, nu=1)
E4_2 = SchemeParams(p=2, q=3, s=0, nextra=0, nu=2)


# ---------------------------------------------------------------------------
# 20.5b — Uniform boundary stencil derivation
# ---------------------------------------------------------------------------


@dataclass
class UniformResult:
    """Result of uniform boundary stencil derivation.

    Attributes
    ----------
    B_u : Matrix
        r_eff x t uniform boundary coefficient matrix.  Entries are rational
        in the free alpha symbols (if any).
    interior : list
        Interior stencil coefficients (length 2*p+1), as Rational values.
    alpha_symbols : list[Symbol]
        Free alpha symbols remaining after conservation.
    p : int
        Interior half-width.
    q : int
        Boundary accuracy order.
    nu : int
        Derivative order.
    """

    B_u: Matrix
    interior: list
    alpha_symbols: list
    p: int
    q: int
    nu: int


def _build_uniform_vandermonde(
    i: int, t: int, n_eqs: int, nu: int
) -> tuple[Matrix, Matrix]:
    """Build the Vandermonde-like Taylor system for uniform grid row *i*.

    Parameters
    ----------
    i : int
        Row centre (grid point index).
    t : int
        Number of grid-point columns.
    n_eqs : int
        Number of Taylor matching equations (= max(q+1, nu+1)).
    nu : int
        Derivative order.

    Returns
    -------
    (V, rhs) where V is n_eqs x t and rhs is n_eqs x 1.
    """
    V = Matrix(
        n_eqs,
        t,
        lambda k, j: Rational((j - i) ** k, factorial(k)),
    )
    rhs = Matrix(n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0))
    return V, rhs


def derive_e2_uniform_boundary(
    nu: int,
    alpha_symbols: list[Symbol] | None = None,
) -> UniformResult:
    """Derive the E2 uniform boundary stencil.

    This is a temporary inline derivation for E2 schemes (p=1, q=1).
    When Module 20.3 (``boundary.py``) is completed, replace with a call
    to the general boundary derivation.  The ``UniformResult`` interface
    remains the same.

    Parameters
    ----------
    nu : int
        Derivative order (1 or 2).
    alpha_symbols : list of Symbol, optional
        Free alpha symbols to use.  For nu=1 must have length 4.
        Ignored (may be ``None``) for nu=2 (no free parameters).

    Returns
    -------
    UniformResult
    """
    p, q = 1, 1
    nextra = 1 if nu == 1 else 0
    dims = compute_dimensions(p, q, 0, nextra, nu)
    t = dims.t
    r_eff = dims.r if nu == 1 else dims.r - 1
    n_eqs = max(q + 1, nu + 1)

    if nu == 2:
        # E2_2: 1 row × 3 columns, fully determined (no free params).
        interior = [Rational(1), Rational(-2), Rational(1)]
        V, rhs = _build_uniform_vandermonde(0, t, n_eqs, nu)
        sol = V.solve(rhs)
        B_u = sol.T
        return UniformResult(
            B_u=B_u, interior=interior, alpha_symbols=[], p=p, q=q, nu=nu
        )

    if nu == 1:
        # E2_1: 3 rows × 4 columns, 4 free alpha symbols after conservation.
        interior = [Rational(-1, 2), Rational(0), Rational(1, 2)]

        if alpha_symbols is None:
            alpha_symbols = [Symbol(f"alpha_{k}") for k in range(4)]
        if len(alpha_symbols) != 4:
            raise ValueError(
                f"E2_1 requires exactly 4 alpha symbols, got {len(alpha_symbols)}"
            )

        n_free_per_row = t - n_eqs  # = 2

        # Temporary raw symbols (2 per row, 6 total).
        raw: list[list[Symbol]] = []
        for i in range(r_eff):
            raw.append([Symbol(f"_raw_{i}_{k}") for k in range(n_free_per_row)])

        # Solve each row's Taylor system, leaving the last n_free_per_row
        # columns (= the interior columns) as free parameters.
        rows: list[list] = []
        for i in range(r_eff):
            V, rhs = _build_uniform_vandermonde(i, t, n_eqs, nu)
            free_vals = Matrix([[s] for s in raw[i]])

            V_det = V[:, :n_eqs]
            V_free = V[:, n_eqs:]
            sol = V_det.solve(rhs - V_free * free_vals)

            rows.append(list(sol) + raw[i])

        B_u = Matrix(rows)

        # Conservation: sum_i B_u[i, j] = 0 for interior columns j >= p+1.
        # Eliminates the last row's raw symbols.
        subs: dict = {}
        for j in range(p + 1, t):
            col_sum = sum(B_u[row_i, j] for row_i in range(r_eff - 1))
            raw_idx = j - n_eqs
            subs[raw[-1][raw_idx]] = -col_sum

        B_u = B_u.subs(subs)

        # Map surviving raw symbols → caller-supplied alpha symbols.
        alpha_map: dict = {}
        idx = 0
        for i in range(r_eff - 1):
            for k in range(n_free_per_row):
                alpha_map[raw[i][k]] = alpha_symbols[idx]
                idx += 1

        B_u = B_u.subs(alpha_map)

        return UniformResult(
            B_u=B_u,
            interior=interior,
            alpha_symbols=list(alpha_symbols),
            p=p,
            q=q,
            nu=nu,
        )

    raise ValueError(f"Unsupported derivative order nu={nu}")
