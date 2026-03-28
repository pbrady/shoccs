"""TEMO (Truncation Error Matching Optimization) cut-cell stencil extension.

Implements the TEMO procedure from Brady & Livescu (2021) for deriving
psi-parameterized cut-cell boundary stencils from uniform boundary stencils.
"""

from dataclasses import dataclass
from typing import NamedTuple

from sympy import Matrix, Poly, Rational, Symbol, cancel, factorial
from sympy.polys.matrices import DomainMatrix


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


# ---------------------------------------------------------------------------
# 20.5e Phase 1 — QQ(psi) field utilities and linear solve
# ---------------------------------------------------------------------------


def make_psi_field(psi: Symbol):
    """Create the QQ(psi) fraction field and return (K, psi_elem).

    Parameters
    ----------
    psi : Symbol
        The SymPy symbol for psi.

    Returns
    -------
    (K, psi_elem)
        K is the QQ(psi) fraction field, psi_elem is psi as a field element.
    """
    from sympy import QQ

    K = QQ.frac_field(psi)
    psi_elem = K.from_sympy(psi)
    return K, psi_elem


def to_field_elem(expr, K):
    """Convert a SymPy expression (rational in psi only) to a QQ(psi) field element.

    Calls ``cancel(expr)`` once on input, then ``K.from_sympy()``.

    Parameters
    ----------
    expr : Expr
        SymPy expression that is rational in psi (no other symbols).
    K : FractionField
        The QQ(psi) field from ``make_psi_field``.

    Returns
    -------
    Field element in K.

    Raises
    ------
    ValueError
        If ``expr`` contains symbols other than psi.
    """
    expr = cancel(expr)
    # Check for unexpected symbols
    psi_sym = K.symbols[0]
    extra = expr.free_symbols - {psi_sym}
    if extra:
        raise ValueError(
            f"Expression contains non-psi symbols: {extra}. "
            f"Expected only {psi_sym}."
        )
    return K.from_sympy(expr)


def from_field_elem(elem, K):
    """Convert a QQ(psi) field element back to a SymPy expression.

    Parameters
    ----------
    elem : field element
        An element of the QQ(psi) field.
    K : FractionField
        The QQ(psi) field from ``make_psi_field``.

    Returns
    -------
    SymPy Expr
    """
    return K.to_sympy(elem)


def decompose_alpha_terms(expr, symbols: list[Symbol]) -> dict:
    """Decompose an expression that is linear in ``symbols`` but rational in psi.

    Returns ``{1: c_0, s_0: c_1, s_1: c_2, ...}`` where each ``c_k`` is a
    SymPy expression in psi only.

    Parameters
    ----------
    expr : Expr
        SymPy expression, polynomial (degree <= 1) in each symbol, rational in psi.
    symbols : list of Symbol
        The symbols to decompose over (alpha, beta, or a mix).

    Returns
    -------
    dict[Symbol | int, Expr]
        Mapping from each symbol (and the integer ``1`` for the constant term)
        to its psi-rational coefficient.

    Raises
    ------
    ValueError
        If any term is nonlinear in any symbol.
    """
    if not symbols:
        return {1: expr}

    # Use Poly to decompose into monomials over the given symbols.
    poly = Poly(expr, *symbols, domain="ZZ(psi)")
    result = {}
    for monom, coeff_raw in poly.as_dict().items():
        # monom is a tuple of exponents, e.g. (1, 0) for first symbol
        if max(monom) > 1:
            raise ValueError(
                f"Nonlinear term detected: exponents {monom} for symbols {symbols}"
            )
        # Convert domain element back to SymPy
        coeff = poly.domain.to_sympy(coeff_raw)
        if sum(monom) == 0:
            result[1] = coeff
        elif sum(monom) == 1:
            idx = monom.index(1)
            result[symbols[idx]] = coeff
        else:
            # Cross term like alpha_0 * alpha_1
            raise ValueError(
                f"Cross-term detected: exponents {monom} for symbols {symbols}"
            )

    return result


def solve_in_field(V_sympy: Matrix, rhs_sympy: Matrix, K, symbols: list[Symbol]):
    """Solve a square linear system in QQ(psi), with optional symbol-dependent RHS.

    This is the core linear solve used by ``solve_temo_row`` in 20.5d.

    Parameters
    ----------
    V_sympy : Matrix
        Square n x n SymPy matrix, entries rational in psi only.
    rhs_sympy : Matrix
        n x 1 SymPy column vector. Entries may be linear in the given symbols
        (alpha, beta) with psi-rational coefficients.
    K : FractionField
        The QQ(psi) field from ``make_psi_field``.
    symbols : list of Symbol
        Non-psi symbols that may appear linearly in rhs_sympy.  Empty list for
        the pure-psi case.

    Returns
    -------
    list of Expr
        Length-n list of SymPy expressions, each rational in psi and linear in
        the given symbols.
    """
    n = V_sympy.rows
    assert V_sympy.cols == n, "V must be square"
    assert rhs_sympy.rows == n and rhs_sympy.cols == 1

    # Build V as a DomainMatrix in K
    V_rows = []
    for i in range(n):
        V_rows.append([to_field_elem(V_sympy[i, j], K) for j in range(n)])
    V_dm = DomainMatrix(V_rows, (n, n), K)

    if not symbols:
        # Pure-psi case: convert rhs directly and solve once
        rhs_elems = [[to_field_elem(rhs_sympy[i, 0], K)] for i in range(n)]
        rhs_dm = DomainMatrix(rhs_elems, (n, 1), K)
        sol_dm = V_dm.lu_solve(rhs_dm)
        return [from_field_elem(row[0], K) for row in sol_dm.to_list()]

    # Symbol-dependent case: decompose RHS into per-symbol components,
    # solve each component separately, then reassemble.
    # Decompose each RHS entry
    decomposed = [decompose_alpha_terms(rhs_sympy[i, 0], symbols) for i in range(n)]

    # Collect all keys (1, sym_0, sym_1, ...)
    all_keys: set = set()
    for d in decomposed:
        all_keys.update(d.keys())

    # For each key, build a RHS column and solve
    solutions_per_key: dict = {}
    for key in sorted(all_keys, key=lambda k: (0, "") if k == 1 else (1, str(k))):
        rhs_col = []
        for i in range(n):
            coeff = decomposed[i].get(key, Rational(0))
            rhs_col.append([to_field_elem(coeff, K)])
        rhs_dm = DomainMatrix(rhs_col, (n, 1), K)
        sol_dm = V_dm.lu_solve(rhs_dm)
        solutions_per_key[key] = [
            from_field_elem(row[0], K) for row in sol_dm.to_list()
        ]

    # Reassemble: solution[j] = x_{j,1} + sum_s x_{j,s} * s
    result = []
    for j in range(n):
        expr = Rational(0)
        for key, sol_vec in solutions_per_key.items():
            if key == 1:
                expr += sol_vec[j]
            else:
                expr += sol_vec[j] * key
        result.append(expr)

    return result
