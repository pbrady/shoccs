"""TEMO (Truncation Error Matching Optimization) cut-cell stencil extension.

Implements the TEMO procedure from Brady & Livescu (2021) for deriving
psi-parameterized cut-cell boundary stencils from uniform boundary stencils.
"""

from dataclasses import dataclass
from typing import NamedTuple

from sympy import Integer, Matrix, Poly, Rational, S, Symbol, cancel, factorial
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

    Uses Eq. 11a from Brady & Livescu (2021) with D-R25 correction:
        t = p + q + 1 + nextra     (stencil width)
        r = p + 1 + nextra         (number of boundary rows)

    For cut-cell stencils:
        R = r_eff + 1, T = t + 1

    where r_eff = r for 1st derivatives, r_eff = r - 1 for 2nd derivatives
    (the last uniform boundary row overlaps with the first interior row).

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
    r = p + 1 + nextra

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
# 21.1a — General uniform boundary for TEMO
# ---------------------------------------------------------------------------


def derive_uniform_boundary_for_temo(
    scheme: SchemeParams,
    alpha_symbols: list[Symbol] | None = None,
) -> UniformResult:
    """Derive the uniform boundary stencil for any scheme using boundary.py.

    This is the general replacement for ``derive_e2_uniform_boundary``.
    It uses ``boundary.solve_boundary_row`` with TEMO-specific dimensions
    and the alpha distribution convention from the plan.

    Parameters
    ----------
    scheme : SchemeParams
        Scheme parameters (p, q, s, nextra, nu).
    alpha_symbols : list of Symbol, optional
        Free alpha symbols to use.  If None, creates alpha_0..alpha_{n-1}.

    Returns
    -------
    UniformResult
    """
    from stencil_gen.boundary import solve_boundary_row
    from stencil_gen.interior import derive_interior, full_gamma_array

    p, q, s, nextra, nu = scheme.p, scheme.q, scheme.s, scheme.nextra, scheme.nu
    dims = compute_dimensions(p, q, s, nextra, nu)
    t = dims.t
    r_eff = dims.r if nu == 1 else dims.r - 1
    n_free_per_row = t - (q + 1)

    # --- Count total alpha symbols ---
    if nextra == 0:
        # Early rows: 1 active each; last row: min(n_free, 2) active
        n_alpha = (r_eff - 1) * 1 + min(n_free_per_row, 2)
    else:
        # Early rows: n_free each; last row: phi placeholders (resolved by conservation)
        n_alpha = (r_eff - 1) * n_free_per_row

    if alpha_symbols is None:
        alpha_symbols = [Symbol(f"alpha_{k}") for k in range(n_alpha)]
    if len(alpha_symbols) != n_alpha:
        raise ValueError(
            f"Scheme requires exactly {n_alpha} alpha symbols, got {len(alpha_symbols)}"
        )

    # --- Build free symbol lists per row ---
    free_per_row: list[list] = []
    alpha_idx = 0

    if nextra == 0:
        # No conservation row
        for i in range(r_eff - 1):
            # Early rows: 1 active alpha, rest zero
            free = [alpha_symbols[alpha_idx]] + [S.Zero] * (n_free_per_row - 1)
            alpha_idx += 1
            free_per_row.append(free)
        # Last row: min(n_free, 2) active
        active = min(n_free_per_row, 2)
        last_free = [alpha_symbols[alpha_idx + k] for k in range(active)]
        last_free += [S.Zero] * (n_free_per_row - active)
        alpha_idx += active
        free_per_row.append(last_free)
    else:
        # Conservation-constrained last row
        for i in range(r_eff - 1):
            # Early rows: all n_free active
            free = [alpha_symbols[alpha_idx + k] for k in range(n_free_per_row)]
            alpha_idx += n_free_per_row
            free_per_row.append(free)
        # Last row: phi placeholders (will be resolved by conservation)
        from sympy import symbols as sym_symbols

        phi_syms = list(sym_symbols(f"phi_0:{n_free_per_row}"))
        free_per_row.append(phi_syms)

    # --- Solve each row's Taylor system ---
    rows: list[list] = []
    for i in range(r_eff):
        result = solve_boundary_row(i, t, q, nu, free_per_row[i])
        rows.append(result.coefficients)

    B_u = Matrix(rows)

    # --- Conservation step (nextra > 0 only) ---
    if nextra > 0:
        subs: dict = {}
        for j in range(p + 1, t):
            col_sum_upper = sum(B_u[row_i, j] for row_i in range(r_eff - 1))
            phi_idx = j - (q + 1)
            subs[phi_syms[phi_idx]] = -col_sum_upper
        B_u = B_u.subs(subs)

    # --- Interior coefficients ---
    interior_result = derive_interior(s, p, nu)
    interior = full_gamma_array(interior_result)

    return UniformResult(
        B_u=B_u,
        interior=interior,
        alpha_symbols=list(alpha_symbols),
        p=p,
        q=q,
        nu=nu,
    )


# ---------------------------------------------------------------------------
# 20.5c — Degenerate stencil B^d_l (psi=0 limit)
# ---------------------------------------------------------------------------


def build_degenerate_stencil(
    B_u: Matrix, interior_coeffs: list, p: int, q: int, nu: int
) -> Matrix:
    """Build the degenerate (psi=0) stencil B^d_l.

    At psi=0 the wall point coincides with x_0. The degenerate stencil is an
    (r+1) x (t+1) matrix that satisfies Design Principles 1 and 2 from
    Brady & Livescu (2021).

    Parameters
    ----------
    B_u : Matrix
        r_eff x t uniform boundary coefficient matrix from
        ``derive_e2_uniform_boundary``.
    interior_coeffs : list
        Interior stencil coefficients (length 2*p+1), as Rational values.
    p : int
        Interior half-width.
    q : int
        Boundary accuracy order.
    nu : int
        Derivative order (1 or 2).

    Returns
    -------
    Matrix
        (r_eff + 1) x (t + 1) degenerate stencil matrix.
        Column layout: [wall, x_0, x_1, ..., x_{t-1}].
    """
    r = B_u.rows  # r_eff
    t = B_u.cols
    R = r + 1
    T = t + 1
    n_eqs = max(q + 1, nu + 1)

    B_d = Matrix.zeros(R, T)

    # --- Rows 0..r-1: Design Principles 1 and 2 ---
    for i in range(r):
        # DP1: B_u cols 1..t-1 map to B_d cols 2..T-1
        for j in range(1, t):
            B_d[i, j + 1] = B_u[i, j]

        # DP2: split B_u[i,0] between wall (col 0) and x_0 (col 1)
        if nu == 1:
            # B^{d,2}: all rows — wall gets full weight, x_0 zeroed
            B_d[i, 0] = B_u[i, 0]
            B_d[i, 1] = Rational(0)
        elif nu == 2:
            if i == 0:
                # B^{d,1} row 0: wall zeroed, x_0 gets full weight
                B_d[i, 0] = Rational(0)
                B_d[i, 1] = B_u[i, 0]
            else:
                # B^{d,1} rows >= 1: wall gets full weight, x_0 zeroed
                B_d[i, 0] = B_u[i, 0]
                B_d[i, 1] = Rational(0)
        else:
            raise ValueError(f"Unsupported derivative order nu={nu}")

    # --- Row r (near-interior) ---

    # Step 1: zero variant column
    # nu=1: x_0 (col 1) zeroed for all rows including near-interior
    # nu=2: x_0 (col 1) zeroed for rows >= 1 (near-interior is row r >= 1)
    zeroed_col = 1
    B_d[r, zeroed_col] = Rational(0)

    # Check if the interior stencil fits entirely within the T-frame
    # boundary block without overlapping the zeroed column.
    # Interior stencil at x_r covers T-frame cols (r-p+1)..(r+p+1).
    # Fits if: (a) r > p (zeroed col 1 = x_0 is outside stencil range)
    #          (b) r + p + 1 <= t (rightmost stencil col within boundary block)
    interior_start = r - p + 1
    interior_end = r + p + 1
    can_embed_interior = (interior_start > zeroed_col) and (interior_end <= t)

    if can_embed_interior:
        # Direct embedding: the near-interior row is just the interior stencil
        # placed at the correct T-frame columns.  Wall (col 0) and x_0 (col 1)
        # are outside the stencil range and stay at 0.
        for k, coeff in enumerate(interior_coeffs):
            B_d[r, interior_start + k] = coeff
    else:
        # Conservation approach: use column-sum constraints from boundary rows
        # to fix interior columns, then solve Taylor system for the rest.
        # This is valid when the uniform boundary has conservation (nextra > 0)
        # or when the stencil doesn't fit and we need boundary-modified values.
        fixed_cols: set[int] = set()
        if r >= 2:
            for j in range(p + 2, T):
                val = -sum(B_d[i, j] for i in range(1, r))
                B_d[r, j] = val
                fixed_cols.add(j)

        # Taylor solve for remaining unknowns.
        # At psi=0, wall and x_0 coincide at the same position.
        # Deltas from row r centered at x_r: [-r, -r, 1-r, 2-r, ..., t-1-r]
        deltas = [Rational(-r)] * 2 + [Rational(j - r) for j in range(1, t)]

        known_cols = {zeroed_col} | fixed_cols
        unknown_cols = [j for j in range(T) if j not in known_cols]
        n_unk = len(unknown_cols)

        rhs = Matrix(
            n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0)
        )
        for k in range(n_eqs):
            for j in known_cols:
                rhs[k, 0] -= Rational(deltas[j] ** k, factorial(k)) * B_d[r, j]

        V = Matrix(
            n_eqs,
            n_unk,
            lambda k, uj: Rational(
                deltas[unknown_cols[uj]] ** k, factorial(k)
            ),
        )

        if n_unk <= n_eqs:
            V_sq = V[:n_unk, :]
            rhs_sq = rhs[:n_unk, :]
            sol = V_sq.solve(rhs_sq)

            for k in range(n_unk, n_eqs):
                residual = (
                    sum(V[k, uj] * sol[uj] for uj in range(n_unk))
                    - rhs[k, 0]
                )
                if cancel(residual) != 0:
                    raise RuntimeError(
                        f"Overdetermined system inconsistent at equation "
                        f"{k}: residual = {residual}"
                    )
        else:
            raise ValueError(
                f"Underdetermined near-interior row: {n_unk} unknowns, "
                f"{n_eqs} equations. Conservation should fix more columns."
            )

        for uj, j in enumerate(unknown_cols):
            B_d[r, j] = sol[uj]

    return B_d


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


# ---------------------------------------------------------------------------
# 20.5d — B_l(psi) general cut-cell stencil construction
# ---------------------------------------------------------------------------


def build_cut_cell_deltas(i: int, T: int, psi) -> list:
    """Build the T normalized deltas from row centre x_i for the cut-cell grid.

    Parameters
    ----------
    i : int
        Row index (grid point).
    T : int
        Total number of columns (including wall).
    psi : Symbol or Expr
        Fractional wall position.

    Returns
    -------
    list of length T
        ``[-(psi+i), -i, 1-i, 2-i, ..., (T-2)-i]``
    """
    return [-(psi + i)] + [Rational(j - i) for j in range(T - 1)]


def build_temo_vandermonde(i: int, T: int, q: int, nu: int, psi) -> tuple:
    """Build the non-uniform Vandermonde Taylor system for cut-cell row *i*.

    Parameters
    ----------
    i : int
        Row centre index.
    T : int
        Number of columns (including wall).
    q : int
        Boundary accuracy order.
    nu : int
        Derivative order.
    psi : Symbol or Expr
        Fractional wall position.

    Returns
    -------
    (V, rhs) where V is n_eqs x T, rhs is n_eqs x 1.
    n_eqs = max(q+1, nu+1).
    """
    n_eqs = max(q + 1, nu + 1)
    deltas = build_cut_cell_deltas(i, T, psi)
    V = Matrix(
        n_eqs,
        T,
        lambda k, j: deltas[j] ** k / factorial(k),
    )
    rhs = Matrix(n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0))
    return V, rhs


@dataclass
class RowSolveResult:
    """Result of solving a single TEMO row.

    Attributes
    ----------
    coeffs : list
        Length-T list of SymPy expressions (rational in psi, linear in
        alpha + beta symbols).
    beta_info : list[tuple[int, Symbol]]
        ``(col_index, beta_symbol)`` pairs for underdetermined columns.
        Empty for fully determined rows.
    """

    coeffs: list
    beta_info: list


@dataclass
class StencilResult:
    """Result of the full cut-cell stencil construction.

    Attributes
    ----------
    matrix : Matrix
        R x T SymPy Matrix of coefficients.
    beta_info : list[tuple[int, int, Symbol]]
        ``(row, col, symbol)`` for every beta parameter.
    beta_symbols : list[Symbol]
        All unique beta symbols (flat list).
    """

    matrix: Matrix
    beta_info: list
    beta_symbols: list


def _zeroed_col_for_row(i: int, nu: int) -> int:
    """Return the column index zeroed by the variant for row *i*.

    For nu=1 (1st derivative): x_0 (col 1) is zeroed for ALL rows.
    For nu=2 (2nd derivative): wall (col 0) for row 0, x_0 (col 1) for rows >= 1.
    """
    if nu == 1:
        return 1
    elif nu == 2:
        return 0 if i == 0 else 1
    else:
        raise ValueError(f"Unsupported nu={nu}")


def solve_uniform_limit(
    B_u: Matrix,
    interior: list,
    p: int,
    q: int,
    nu: int,
    nextra: int,
) -> Matrix:
    """Solve for B_l(1) — the cut-cell stencil at psi=1 (uniform limit).

    Returns the R x T matrix where R = r_eff + 1, T = t + 1.
    This is needed to provide Category A target values for the near-interior
    row in the general psi solve.

    Parameters
    ----------
    B_u : Matrix
        r_eff x t uniform boundary coefficient matrix.
    interior : list
        Interior stencil coefficients (length 2*p+1).
    p : int
        Interior half-width.
    q : int
        Boundary accuracy order.
    nu : int
        Derivative order.
    nextra : int
        Extra rows/columns for optimization.

    Returns
    -------
    Matrix
        R x T matrix of coefficients at psi=1.
    """
    r = B_u.rows  # r_eff
    t = B_u.cols
    R = r + 1
    T = t + 1
    n_eqs = max(q + 1, nu + 1)

    B_l_1 = Matrix.zeros(R, T)

    # Step 1: Boundary rows (i = 0..r-1)
    # Two cases based on which column is zeroed at psi=0:
    # (a) x_0 zeroed (nu=1 all rows, nu=2 rows>=1):
    #     Simple embedding: cols 1..T-1 = B_u[i, 0..t-1], wall = 0.
    #     At psi=1 Category A gives x_0 = B_u[i,0] (already embedded).
    # (b) wall zeroed (nu=2 row 0):
    #     Category A prescribes wall = B_u[i,0] at psi=1. The remaining
    #     columns must be solved from the Taylor system (not just B_u shifted).
    for i in range(r):
        zeroed = _zeroed_col_for_row(i, nu)
        if zeroed == 0:
            # Wall-zeroed variant: prescribe wall = B_u[i,0], solve for rest.
            B_l_1[i, 0] = B_u[i, 0]
            deltas_i = [Rational(-(1 + i))] + [Rational(j - i) for j in range(t)]
            V_i = Matrix(
                n_eqs, T,
                lambda k, j: Rational(deltas_i[j] ** k, factorial(k)),
            )
            rhs_i = Matrix(
                n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0)
            )
            # Move wall column to RHS
            rhs_adj = rhs_i.copy()
            for k in range(n_eqs):
                rhs_adj[k, 0] -= V_i[k, 0] * B_l_1[i, 0]
            # Solve for cols 1..T-1
            V_rem = V_i[:, 1:]
            sol_i = V_rem.solve(rhs_adj)
            for j in range(t):
                B_l_1[i, j + 1] = sol_i[j]
        else:
            # x_0-zeroed variant: simple embedding, wall = 0.
            for j in range(t):
                B_l_1[i, j + 1] = B_u[i, j]
            B_l_1[i, 0] = Rational(0)

    # Step 2-4: Near-interior row (row r)
    # Check if the interior stencil fits entirely within the T-frame
    # boundary block without overlapping the zeroed column (x_0 at col 1).
    # At psi=1: wall at col 0 is at offset -(1+r) from x_r, x_0 at col 1
    # is at offset -r.  Interior stencil covers offsets -p..+p.
    # Fits if: (a) r > p (both wall and x_0 are outside stencil range)
    #          (b) r + p + 1 <= t (rightmost stencil col within boundary block)
    interior_start = r - p + 1
    interior_end = r + p + 1
    can_embed_interior = (interior_start > 1) and (interior_end <= t)

    if can_embed_interior:
        # Direct embedding: near-interior row is the interior stencil
        # at T-frame cols (r-p+1)..(r+p+1).  All other cols are 0.
        for k, coeff in enumerate(interior):
            B_l_1[r, interior_start + k] = coeff
    else:
        # Conservation + Taylor solve approach (for schemes where the
        # interior stencil extends beyond the boundary block, e.g. E2_1).
        deltas = [Rational(-(1 + r))] + [Rational(j - r) for j in range(t)]
        V_full = Matrix(
            n_eqs,
            T,
            lambda k, j: Rational(deltas[j] ** k, factorial(k)),
        )
        rhs_full = Matrix(
            n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0)
        )

        # Conservation at psi=1 (all weights = 1).
        # sum_{i=0}^{R-1} B_l(1)[i, j] = 0 for interior columns.
        # A T-frame column j is conserved if:
        #   (a) no interior stencil reaches it: j < R-p, OR
        #   (b) uniform conservation applies: j >= p+2
        fixed_cols: set[int] = set()
        for j in range(2, T):
            if not (j < R - p or j >= p + 2):
                continue
            val = -sum(B_l_1[i, j] for i in range(r))
            B_l_1[r, j] = val
            fixed_cols.add(j)

        unknown_cols = [j for j in range(T) if j not in fixed_cols]

        rhs_reduced = rhs_full.copy()
        for k in range(n_eqs):
            for j in fixed_cols:
                rhs_reduced[k, 0] -= V_full[k, j] * B_l_1[r, j]

        V_reduced = Matrix(
            n_eqs,
            len(unknown_cols),
            lambda k, uj: V_full[k, unknown_cols[uj]],
        )

        n_unk = len(unknown_cols)
        if n_unk <= n_eqs:
            V_sq = V_reduced[:n_unk, :]
            rhs_sq = rhs_reduced[:n_unk, :]
            sol = V_sq.solve(rhs_sq)

            for k in range(n_unk, n_eqs):
                res = sum(
                    V_reduced[k, uj] * sol[uj] for uj in range(n_unk)
                )
                res -= rhs_reduced[k, 0]
                if cancel(res) != 0:
                    raise RuntimeError(
                        f"Inconsistent uniform-limit system at equation {k}"
                    )
        else:
            raise ValueError(
                f"Underdetermined uniform-limit row r: {n_unk} unknowns, "
                f"{n_eqs} equations"
            )

        for uj, j in enumerate(unknown_cols):
            B_l_1[r, j] = sol[uj]

    return B_l_1


def identify_prescribed_entries(
    i: int,
    r: int,
    t: int,
    nextra: int,
    nu: int,
    B_u: Matrix,
    B_l_1: Matrix,
    B_d: Matrix,
    psi,
    n_eqs: int,
) -> dict:
    """Identify Category A and limit-interpolation prescribed entries for row *i*.

    Category A prescribes the zeroed column as ``psi * target``.

    Limit interpolation prescribes extra columns (from ``nextra > 0``) as
    ``psi * B_l_1[i, j] + (1 - psi) * B_d[i, j]``, ensuring the stencil
    matches both the uniform (psi=1) and degenerate (psi=0) limits.  For
    boundary rows the two limits are identical so the prescription is constant;
    for the near-interior row it is psi-dependent.

    Parameters
    ----------
    i : int
        Row index (0 to R-1).
    r : int
        Number of uniform boundary rows (r_eff).
    t : int
        Uniform stencil width.
    nextra : int
        Extra rows/columns.
    nu : int
        Derivative order.
    B_u : Matrix
        r_eff x t uniform boundary matrix.
    B_l_1 : Matrix
        R x T uniform limit matrix from ``solve_uniform_limit``.
    B_d : Matrix
        R x T degenerate stencil from ``build_degenerate_stencil``.
    psi : Symbol
        The psi symbol.
    n_eqs : int
        Number of Taylor equations per row (``max(q+1, nu+1)``).

    Returns
    -------
    dict[int, Expr]
        ``{col_index: expr(psi)}`` for all prescribed columns.
    """
    T = B_l_1.cols
    prescribed: dict = {}
    zeroed = _zeroed_col_for_row(i, nu)

    # Category A: zeroed column
    if i < r:
        # Boundary row — target is alpha^u_{i,0}
        target = B_u[i, 0]
    else:
        # Near-interior row — target from B_l(1)
        target = B_l_1[i, zeroed]

    prescribed[zeroed] = psi * target

    # Limit interpolation for extra columns (nextra > 0).
    # These are the columns that would otherwise be underdetermined (beta
    # columns).  Prescribing them via psi-linear interpolation between B_l_1
    # and B_d ensures both psi limits are satisfied and eliminates the need
    # for beta symbols.
    free_cols = sorted(j for j in range(T) if j not in prescribed)
    n_free = len(free_cols)
    n_excess = n_free - n_eqs
    if n_excess > 0:
        # Same convention as the old beta selection: highest free columns.
        extra_cols = free_cols[-n_excess:]
        for j in extra_cols:
            prescribed[j] = psi * B_l_1[i, j] + (1 - psi) * B_d[i, j]

    return prescribed


def solve_temo_row(
    i: int,
    V: Matrix,
    rhs: Matrix,
    prescribed: dict,
    psi,
    K,
    symbols: list,
    beta_prefix: str = "beta",
) -> RowSolveResult:
    """Solve a single TEMO row, handling underdetermined systems via betas.

    Parameters
    ----------
    i : int
        Row index.
    V : Matrix
        n_eqs x T Vandermonde matrix.
    rhs : Matrix
        n_eqs x 1 RHS vector.
    prescribed : dict[int, Expr]
        Prescribed column entries ``{col: expr(psi)}``.
    psi : Symbol
        The psi symbol.
    K : FractionField
        QQ(psi) field.
    symbols : list[Symbol]
        Non-psi symbols (alpha from B_u) that may appear in prescribed values.
    beta_prefix : str
        Prefix for new beta symbols.

    Returns
    -------
    RowSolveResult
    """
    n_eqs = V.rows
    T = V.cols

    # Move prescribed columns to RHS
    rhs_adj = rhs.copy()
    for j, val in prescribed.items():
        for k in range(n_eqs):
            rhs_adj[k, 0] -= V[k, j] * val

    # Identify free columns (not prescribed)
    free_cols = sorted(j for j in range(T) if j not in prescribed)
    n_free = len(free_cols)

    beta_info: list[tuple[int, Symbol]] = []
    all_symbols = list(symbols)

    if n_free > n_eqs:
        # Underdetermined: introduce beta symbols for excess columns
        n_excess = n_free - n_eqs
        # Beta columns = last n_excess free columns (highest indices)
        beta_cols = free_cols[-n_excess:]
        solve_cols = free_cols[:n_eqs]

        for k, bc in enumerate(beta_cols):
            beta_sym = Symbol(f"{beta_prefix}_{i}_{k}")
            beta_info.append((bc, beta_sym))
            all_symbols.append(beta_sym)
            # Move beta column to RHS
            for row_k in range(n_eqs):
                rhs_adj[row_k, 0] -= V[row_k, bc] * beta_sym
    elif n_free == n_eqs:
        solve_cols = free_cols
    else:
        raise ValueError(
            f"Overdetermined TEMO row {i}: {n_free} free cols, {n_eqs} equations"
        )

    # Build square Vandermonde for solve columns
    V_sq = Matrix(n_eqs, n_eqs, lambda k, uj: V[k, solve_cols[uj]])

    # Solve using solve_in_field
    sol = solve_in_field(V_sq, rhs_adj, K, all_symbols)

    # Assemble full coefficient vector
    coeffs = [Rational(0)] * T
    for j, val in prescribed.items():
        coeffs[j] = val
    for uj, j in enumerate(solve_cols):
        coeffs[j] = sol[uj]
    for bc, beta_sym in beta_info:
        coeffs[bc] = beta_sym

    return RowSolveResult(coeffs=coeffs, beta_info=beta_info)


def construct_cut_cell_stencil(
    B_u: Matrix,
    interior: list,
    p: int,
    q: int,
    nu: int,
    nextra: int,
    psi: Symbol,
) -> StencilResult:
    """Construct the psi-parameterized cut-cell stencil B_l(psi).

    This is the central TEMO procedure: produces an R x T matrix of
    rational functions of psi (and alpha symbols from B_u).

    Extra columns (from ``nextra > 0``) are prescribed via limit
    interpolation between B_l(1) and B_d(0), ensuring both psi limits
    are satisfied.  This eliminates the need for beta symbols; all free
    parameters in the result originate from B_u's alpha symbols.

    Parameters
    ----------
    B_u : Matrix
        r_eff x t uniform boundary coefficient matrix.
    interior : list
        Interior stencil coefficients.
    p : int
        Interior half-width.
    q : int
        Boundary accuracy order.
    nu : int
        Derivative order.
    nextra : int
        Extra rows/columns.
    psi : Symbol
        The psi symbol.

    Returns
    -------
    StencilResult
    """
    r = B_u.rows
    R = r + 1
    T = B_u.cols + 1
    n_eqs = max(q + 1, nu + 1)

    K, _ = make_psi_field(psi)

    # Collect alpha symbols from B_u
    alpha_syms = sorted(B_u.free_symbols, key=lambda s: s.name)

    # Step 0: Compute the two limits needed for prescriptions
    B_l_1 = solve_uniform_limit(B_u, interior, p, q, nu, nextra)
    B_d = build_degenerate_stencil(B_u, interior, p, q, nu)

    # Solve each row
    all_beta_info: list[tuple[int, int, Symbol]] = []
    all_beta_symbols: list[Symbol] = []
    rows: list[list] = []

    for i in range(R):
        V, rhs_vec = build_temo_vandermonde(i, T, q, nu, psi)
        prescribed = identify_prescribed_entries(
            i, r, B_u.cols, nextra, nu, B_u, B_l_1, B_d, psi, n_eqs
        )
        result = solve_temo_row(
            i, V, rhs_vec, prescribed, psi, K, alpha_syms, beta_prefix="beta"
        )
        rows.append(result.coeffs)
        for col, sym in result.beta_info:
            all_beta_info.append((i, col, sym))
            all_beta_symbols.append(sym)

    matrix = Matrix(rows)
    return StencilResult(
        matrix=matrix, beta_info=all_beta_info, beta_symbols=all_beta_symbols
    )


# ---------------------------------------------------------------------------
# 20.5f — Neumann eta coefficients and output assembly
# ---------------------------------------------------------------------------


@dataclass
class CutCellResult:
    """Assembled cut-cell stencil output with all BC variants.

    Attributes
    ----------
    floating : Matrix
        R x T floating (no BC) stencil.
    dirichlet : Matrix
        (R-1) x T Dirichlet stencil (rows 1..R-1 of floating).
    neumann : Matrix or None
        R x T Neumann stencil (only for nu=2, X > 0).
    eta : list or None
        Length-R Neumann eta coefficients (only for nu=2, X > 0).
    dims : Dimensions
        (r, t, R, T, X) dimensions.
    alpha_symbols : list[Symbol]
        Free alpha symbols (from B_u).
    """

    floating: Matrix
    dirichlet: Matrix
    neumann: "Matrix | None"
    eta: "list | None"
    dims: Dimensions
    alpha_symbols: list


def _neumann_zeroed_col_for_row(i: int, nu: int) -> int:
    """Return the column index zeroed by the Neumann variant B^{d,0} for row *i*.

    For nu=2 (Neumann): row 0 zeros x_0 (col 1), rows >= 1 zero wall (col 0).
    This is the OPPOSITE of the floating variant B^{d,1}.
    """
    if nu == 2:
        return 1 if i == 0 else 0
    else:
        raise ValueError(f"Neumann only supported for nu=2, got nu={nu}")


def derive_uniform_neumann(
    interior: list,
    p: int,
    q: int,
    nu: int,
) -> tuple[Matrix, list]:
    """Derive the uniform Neumann stencil B^{uN}_l.

    Augments the Taylor system with a virtual derivative column for
    ``h * f'(x_wall)``, then solves with conservation on the uniform grid.

    Parameters
    ----------
    interior : list
        Interior stencil coefficients (length 2*p+1).
    p : int
        Interior half-width.
    q : int
        Boundary accuracy order.
    nu : int
        Derivative order (must be 2).

    Returns
    -------
    (B_uN, eta_u)
        B_uN: r_eff x t matrix (uniform Neumann stencil).
        eta_u: length-r_eff list of eta values.
    """
    if nu != 2:
        raise ValueError(f"Neumann only for nu=2, got {nu}")

    # For E2_2: r_eff=1, t=3
    # The uniform Neumann row is derived from an augmented Taylor system:
    # Row 0 on uniform grid [x_0, x_1, x_2] with virtual eta column.
    t = 2 * p + q
    r_eff = q  # For nu=2: r_eff = r - 1 where r = q + 1
    n_eqs = max(q + 1, nu + 1)

    B_uN = Matrix.zeros(r_eff, t)
    eta_u = []

    for i in range(r_eff):
        # Standard uniform Vandermonde for t columns
        deltas = [Rational(j - i) for j in range(t)]

        # Augmented system: T_cols + 1 eta column
        # V[k, j] = delta_j^k / k! for j = 0..t-1
        # V[k, t] = delta_0^{k-1} / (k-1)! for k >= 1, 0 for k=0
        # (derivative column for the wall at x_0 on uniform grid, delta_wall = -i)
        delta_wall = Rational(-i)

        V_aug = Matrix(n_eqs, t + 1, lambda k, j: (
            deltas[j] ** k / factorial(k) if j < t
            else (delta_wall ** (k - 1) / factorial(k - 1) if k >= 1 else Rational(0))
        ))
        rhs = Matrix(
            n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0)
        )

        # Conservation: for uniform Neumann, the last column (j=t-1) should be 0
        # because the interior stencil doesn't reach that far.
        # Apply: c[t-1] = 0 (conservation at rightmost column for single-row case)
        # Move last stencil column to RHS (it's 0, so no change to rhs)
        # Actually for single row, conservation is: sum over rows = 0 for interior cols
        # With r_eff=1, there's only one row plus the interior stencil.
        # For the uniform case, prescribe c[t-1] = 0 based on the worked example.

        # For E2_2 (r_eff=1, t=3): n_eqs=3, unknowns = t+1 = 4 (3 stencil + 1 eta)
        # Conservation: c[t-1] = 0
        prescribed_cols = {t - 1: Rational(0)}

        # Move prescribed to RHS
        rhs_adj = rhs.copy()
        for col_j, val in prescribed_cols.items():
            for k in range(n_eqs):
                rhs_adj[k, 0] -= V_aug[k, col_j] * val

        # Free columns: all except prescribed stencil cols (keep eta = col t)
        free_cols = [j for j in range(t + 1) if j not in prescribed_cols]
        V_free = Matrix(n_eqs, len(free_cols), lambda k, fj: V_aug[k, free_cols[fj]])

        sol = V_free.solve(rhs_adj)

        for fj, col_j in enumerate(free_cols):
            if col_j < t:
                B_uN[i, col_j] = sol[fj]
            else:
                eta_u.append(sol[fj])
        for col_j, val in prescribed_cols.items():
            if col_j < t:
                B_uN[i, col_j] = val

    return B_uN, eta_u


def build_neumann_vandermonde(
    i: int, T: int, q: int, nu: int, psi
) -> tuple[Matrix, Matrix]:
    """Build the augmented Neumann Vandermonde system for cut-cell row *i*.

    Extends ``build_temo_vandermonde`` with one extra column for
    ``eta_i * h * f'(x_wall)``.

    Parameters
    ----------
    i : int
        Row centre index.
    T : int
        Number of stencil columns (including wall).
    q : int
        Boundary accuracy order.
    nu : int
        Derivative order.
    psi : Symbol or Expr
        Fractional wall position.

    Returns
    -------
    (V_aug, rhs) where V_aug is n_eqs x (T+1), rhs is n_eqs x 1.
    The last column of V_aug is the virtual derivative column.
    """
    n_eqs = max(q + 1, nu + 1)
    deltas = build_cut_cell_deltas(i, T, psi)
    delta_wall = deltas[0]  # -(psi + i)

    V_aug = Matrix(n_eqs, T + 1, lambda k, j: (
        deltas[j] ** k / factorial(k) if j < T
        else (delta_wall ** (k - 1) / factorial(k - 1) if k >= 1 else Rational(0))
    ))
    rhs = Matrix(n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0))
    return V_aug, rhs


def solve_neumann_uniform_limit(
    B_uN: Matrix,
    eta_u: list,
    interior: list,
    p: int,
    q: int,
    nu: int,
    nextra: int,
) -> Matrix:
    """Solve for B_l_N(1) — the Neumann cut-cell stencil at psi=1.

    Returns an R x (T+1) matrix where the last column holds the eta values.

    Parameters
    ----------
    B_uN : Matrix
        r_eff x t uniform Neumann boundary stencil.
    eta_u : list
        Length-r_eff uniform Neumann eta values.
    interior : list
        Interior stencil coefficients (length 2*p+1).
    p : int
        Interior half-width.
    q : int
        Boundary accuracy order.
    nu : int
        Derivative order.
    nextra : int
        Extra rows/columns.

    Returns
    -------
    Matrix
        R x (T+1) matrix. Columns 0..T-1 are stencil coefficients,
        column T is the eta value for each row.
    """
    r = B_uN.rows  # r_eff
    t = B_uN.cols
    R = r + 1
    T = t + 1
    n_eqs = max(q + 1, nu + 1)

    # Result: R x (T+1) — last column is eta
    B_l_N_1 = Matrix.zeros(R, T + 1)

    # Step 1: Boundary rows (i = 0..r-1)
    # Solve the augmented Taylor system at psi=1 with eta prescribed from
    # the uniform Neumann and conservation on the rightmost stencil column.
    for i in range(r):
        deltas_i = [Rational(-(1 + i))] + [Rational(j - i) for j in range(t)]
        delta_wall_i = deltas_i[0]

        V_aug = Matrix(
            n_eqs, T + 1,
            lambda k, j, _di=deltas_i, _dw=delta_wall_i: (
                Rational(_di[j] ** k, factorial(k)) if j < T
                else (Rational(_dw ** (k - 1), factorial(k - 1)) if k >= 1
                      else Rational(0))
            ),
        )
        rhs_i = Matrix(
            n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0)
        )

        # Prescribe eta = eta_u[i] and c[T-1] = 0 (conservation at rightmost col)
        fixed_i: dict[int, Rational] = {T: eta_u[i], T - 1: Rational(0)}

        rhs_adj = rhs_i.copy()
        for k in range(n_eqs):
            for col_j, val in fixed_i.items():
                rhs_adj[k, 0] -= V_aug[k, col_j] * val

        free_cols = [j for j in range(T + 1) if j not in fixed_i]
        V_free = Matrix(
            n_eqs, len(free_cols),
            lambda k, fj, _fc=free_cols: V_aug[k, _fc[fj]],
        )
        sol = V_free.solve(rhs_adj)
        for fj, col_j in enumerate(free_cols):
            B_l_N_1[i, col_j] = sol[fj]
        for col_j, val in fixed_i.items():
            B_l_N_1[i, col_j] = val

    # Step 2-4: Near-interior row (row r)
    deltas_r = [Rational(-(1 + r))] + [Rational(j - r) for j in range(t)]
    delta_wall_r = deltas_r[0]

    V_aug_r = Matrix(
        n_eqs, T + 1,
        lambda k, j, _dr=deltas_r, _dw=delta_wall_r: (
            Rational(_dr[j] ** k, factorial(k)) if j < T
            else (Rational(_dw ** (k - 1), factorial(k - 1)) if k >= 1 else Rational(0))
        ),
    )
    rhs_r = Matrix(
        n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0)
    )

    # Step 3: Conservation at psi=1 for interior stencil columns.
    fixed_cols: set[int] = set()
    for j in range(2, T):
        if not (j <= R - p or j >= p + 2):
            continue
        val = -sum(B_l_N_1[i, j] for i in range(r))
        B_l_N_1[r, j] = val
        fixed_cols.add(j)

    # Unknowns: wall (0), x_0 (1), unfixed stencil cols, eta (T)
    unknown_cols = [j for j in range(T + 1) if j not in fixed_cols]

    # Step 4: Solve augmented Taylor for near-interior row unknowns.
    rhs_reduced = rhs_r.copy()
    for k in range(n_eqs):
        for j in fixed_cols:
            rhs_reduced[k, 0] -= V_aug_r[k, j] * B_l_N_1[r, j]

    V_reduced = Matrix(
        n_eqs, len(unknown_cols),
        lambda k, uj, _uc=unknown_cols: V_aug_r[k, _uc[uj]],
    )

    n_unk = len(unknown_cols)

    # If still underdetermined, apply additional constraints:
    # 1. Conservation on rightmost stencil column (c[T-1])
    # 2. eta = 0 for near-interior row (no wall-derivative coupling needed
    #    at the uniform limit — the near-interior stencil is the interior stencil)
    while n_unk > n_eqs:
        if T - 1 not in fixed_cols:
            val = -sum(B_l_N_1[i, T - 1] for i in range(r))
            B_l_N_1[r, T - 1] = val
            fixed_cols.add(T - 1)
        elif T not in fixed_cols:
            # Prescribe eta = 0 for the near-interior row at psi=1
            B_l_N_1[r, T] = Rational(0)
            fixed_cols.add(T)
        else:
            raise ValueError(
                f"Underdetermined Neumann uniform-limit row r: "
                f"{n_unk} unknowns, {n_eqs} equations"
            )

        unknown_cols = [j for j in range(T + 1) if j not in fixed_cols]
        rhs_reduced = rhs_r.copy()
        for k in range(n_eqs):
            for j in fixed_cols:
                rhs_reduced[k, 0] -= V_aug_r[k, j] * B_l_N_1[r, j]
        V_reduced = Matrix(
            n_eqs, len(unknown_cols),
            lambda k, uj, _uc=unknown_cols: V_aug_r[k, _uc[uj]],
        )
        n_unk = len(unknown_cols)

    if n_unk <= n_eqs:
        V_sq = V_reduced[:n_unk, :]
        rhs_sq = rhs_reduced[:n_unk, :]
        sol = V_sq.solve(rhs_sq)

        for k in range(n_unk, n_eqs):
            res = sum(V_reduced[k, uj] * sol[uj] for uj in range(n_unk))
            res -= rhs_reduced[k, 0]
            if cancel(res) != 0:
                raise RuntimeError(
                    f"Inconsistent Neumann uniform-limit at equation {k}"
                )

    for uj, j in enumerate(unknown_cols):
        B_l_N_1[r, j] = sol[uj]

    return B_l_N_1


def identify_neumann_prescribed_entries(
    i: int,
    r: int,
    t: int,
    nextra: int,
    nu: int,
    B_uN: Matrix,
    B_l_N_1: Matrix,
    B_d_N: Matrix,
    psi,
    n_eqs: int,
) -> dict:
    """Identify prescribed entries for Neumann variant B^{d,0}.

    Uses the opposite zeroed column convention from the floating variant:
    row 0 zeros x_0, rows >= 1 zero wall.

    Parameters
    ----------
    i : int
        Row index.
    r : int
        Number of uniform boundary rows (r_eff).
    t : int
        Uniform stencil width.
    nextra : int
        Extra rows/columns.
    nu : int
        Derivative order (must be 2).
    B_uN : Matrix
        r_eff x t uniform Neumann boundary matrix.
    B_l_N_1 : Matrix
        R x (T+1) Neumann uniform limit matrix.
    B_d_N : Matrix
        R x (T+1) Neumann degenerate stencil.
    psi : Symbol
        The psi symbol.
    n_eqs : int
        Number of Taylor equations per row.

    Returns
    -------
    dict[int, Expr]
        ``{col_index: expr(psi)}`` for all prescribed columns in the
        augmented (T+1)-column frame.
    """
    T_aug = B_l_N_1.cols  # T + 1
    T = T_aug - 1
    prescribed: dict = {}
    zeroed = _neumann_zeroed_col_for_row(i, nu)

    # Category A: zeroed column.
    # For Neumann, always use B_l_N_1 for the target because the Neumann
    # uniform limit involves solving the augmented Taylor system — the values
    # differ from a simple B_uN embedding.
    target = B_l_N_1[i, zeroed]
    prescribed[zeroed] = psi * target

    # Prescribe eta (col T) via limit interpolation for ALL rows.
    # For boundary rows, limits are typically identical (eta is constant).
    # For near-interior rows, limits differ, giving a psi-dependent eta.
    prescribed[T] = psi * B_l_N_1[i, T] + (1 - psi) * B_d_N[i, T]

    # Limit interpolation for extra stencil columns (nextra > 0)
    free_cols = sorted(j for j in range(T_aug) if j not in prescribed)
    n_free = len(free_cols)
    n_excess = n_free - n_eqs
    if n_excess > 0:
        stencil_free = [j for j in free_cols if j < T]
        n_stencil_excess = min(n_excess, len(stencil_free))
        if n_stencil_excess > 0:
            extra_cols = stencil_free[-n_stencil_excess:]
            for j in extra_cols:
                prescribed[j] = psi * B_l_N_1[i, j] + (1 - psi) * B_d_N[i, j]

    return prescribed


def _build_neumann_degenerate(
    B_uN: Matrix,
    eta_u: list,
    interior: list,
    p: int,
    q: int,
    nu: int,
) -> Matrix:
    """Build the Neumann degenerate stencil (psi=0) in augmented frame.

    Returns an R x (T+1) matrix where the last column is eta.
    Uses Neumann variant B^{d,0}: row 0 zeros x_0, rows >= 1 zero wall.

    Parameters
    ----------
    B_uN : Matrix
        r_eff x t uniform Neumann boundary stencil.
    eta_u : list
        Length-r_eff uniform Neumann eta values.
    interior : list
        Interior stencil coefficients.
    p : int
        Interior half-width.
    q : int
        Boundary accuracy order.
    nu : int
        Derivative order (must be 2).

    Returns
    -------
    Matrix
        R x (T+1) degenerate Neumann stencil.
    """
    r = B_uN.rows
    t = B_uN.cols
    R = r + 1
    T = t + 1
    n_eqs = max(q + 1, nu + 1)

    B_d_N = Matrix.zeros(R, T + 1)

    # Rows 0..r-1: DP1 + Neumann variant B^{d,0}
    for i in range(r):
        # Cols 1..T-1 from B_uN (shifted)
        for j in range(1, t):
            B_d_N[i, j + 1] = B_uN[i, j]

        zeroed = _neumann_zeroed_col_for_row(i, nu)
        if zeroed == 1:
            # x_0 zeroed (row 0): wall gets B_uN[i,0], x_0 = 0
            B_d_N[i, 0] = B_uN[i, 0]
            B_d_N[i, 1] = Rational(0)
        elif zeroed == 0:
            # wall zeroed (rows >= 1): wall = 0, x_0 gets B_uN[i,0]
            B_d_N[i, 0] = Rational(0)
            B_d_N[i, 1] = B_uN[i, 0]

        # eta column
        B_d_N[i, T] = eta_u[i]

    # Row r (near-interior): solve augmented Taylor at psi=0
    # Deltas at psi=0: wall coincides with x_0, so delta = [-r, -r, 1-r, ..., t-1-r]
    deltas_r = [Rational(-r), Rational(-r)] + [Rational(j - r) for j in range(1, t)]
    delta_wall_r = Rational(-r)

    V_aug_r = Matrix(
        n_eqs, T + 1,
        lambda k, j, _dr=deltas_r, _dw=delta_wall_r: (
            Rational(_dr[j] ** k, factorial(k)) if j < T
            else (Rational(_dw ** (k - 1), factorial(k - 1)) if k >= 1 else Rational(0))
        ),
    )
    rhs_r = Matrix(
        n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0)
    )

    # Neumann variant for near-interior row (i >= 1): wall zeroed
    B_d_N[r, 0] = Rational(0)

    # Conservation at psi=0 (w_0=0): for interior stencil cols j >= 2
    # sum_{i=1}^{R-1} B_d_N[i,j] = 0 (row 0 drops out since w_0=0)
    fixed_cols: set[int] = set()
    fixed_cols.add(0)  # wall zeroed

    if r >= 2:
        for j in range(2, T):
            val = -sum(B_d_N[ii, j] for ii in range(1, r))
            B_d_N[r, j] = val
            fixed_cols.add(j)

    # Conservation at rightmost column: c[T-1] = 0
    if T - 1 not in fixed_cols:
        B_d_N[r, T - 1] = Rational(0)
        fixed_cols.add(T - 1)

    # Solve for remaining unknowns in near-interior row
    unknown_cols = [j for j in range(T + 1) if j not in fixed_cols]

    rhs_reduced = rhs_r.copy()
    for k in range(n_eqs):
        for j in fixed_cols:
            rhs_reduced[k, 0] -= V_aug_r[k, j] * B_d_N[r, j]

    V_reduced = Matrix(
        n_eqs, len(unknown_cols),
        lambda k, uj, _uc=unknown_cols: V_aug_r[k, _uc[uj]],
    )

    sol = V_reduced.solve(rhs_reduced)
    for uj, j in enumerate(unknown_cols):
        B_d_N[r, j] = sol[uj]

    return B_d_N


def construct_neumann_stencil(
    B_u: Matrix,
    B_uN: Matrix,
    eta_u: list,
    interior: list,
    p: int,
    q: int,
    nu: int,
    nextra: int,
    psi: Symbol,
) -> tuple[Matrix, list]:
    """Construct the psi-parameterized Neumann cut-cell stencil.

    Parameters
    ----------
    B_u : Matrix
        r_eff x t uniform boundary coefficient matrix (floating).
    B_uN : Matrix
        r_eff x t uniform Neumann boundary stencil.
    eta_u : list
        Length-r_eff uniform Neumann eta values.
    interior : list
        Interior stencil coefficients.
    p : int
        Interior half-width.
    q : int
        Boundary accuracy order.
    nu : int
        Derivative order (must be 2).
    nextra : int
        Extra rows/columns.
    psi : Symbol
        The psi symbol.

    Returns
    -------
    (neumann_main, eta)
        neumann_main: R x T SymPy Matrix of stencil coefficients.
        eta: length-R list of eta expressions (rational in psi).
    """
    r = B_uN.rows
    R = r + 1
    T = B_uN.cols + 1
    n_eqs = max(q + 1, nu + 1)

    K, _ = make_psi_field(psi)

    # Collect alpha symbols
    alpha_syms = sorted(B_u.free_symbols, key=lambda s: s.name)

    # Compute limits
    B_l_N_1 = solve_neumann_uniform_limit(
        B_uN, eta_u, interior, p, q, nu, nextra
    )
    B_d_N = _build_neumann_degenerate(B_uN, eta_u, interior, p, q, nu)

    rows: list[list] = []

    for i in range(R):
        V_aug, rhs_vec = build_neumann_vandermonde(i, T, q, nu, psi)

        prescribed = identify_neumann_prescribed_entries(
            i, r, B_uN.cols, nextra, nu, B_uN, B_l_N_1, B_d_N, psi, n_eqs
        )

        # Use solve_temo_row on the augmented system (T+1 columns)
        result = solve_temo_row(
            i, V_aug, rhs_vec, prescribed, psi, K, alpha_syms, beta_prefix="nbeta"
        )
        rows.append(result.coeffs)

    # Split into main stencil (R x T) and eta (R x 1)
    neumann_main = Matrix(R, T, lambda i, j: rows[i][j])
    eta = [rows[i][T] for i in range(R)]

    return neumann_main, eta


def assemble_cut_cell_result(
    floating: Matrix,
    neumann_main: "Matrix | None",
    eta: "list | None",
    dims: Dimensions,
    alpha_symbols: list,
) -> CutCellResult:
    """Assemble the full cut-cell stencil result with all BC variants.

    Parameters
    ----------
    floating : Matrix
        R x T floating stencil from ``construct_cut_cell_stencil``.
    neumann_main : Matrix or None
        R x T Neumann stencil (only for nu=2).
    eta : list or None
        Length-R eta values (only for nu=2).
    dims : Dimensions
        Scheme dimensions.
    alpha_symbols : list[Symbol]
        Free alpha symbols.

    Returns
    -------
    CutCellResult
    """
    R = dims.R
    # Dirichlet = rows 1..R-1 of floating
    dirichlet = floating[1:, :]

    return CutCellResult(
        floating=floating,
        dirichlet=dirichlet,
        neumann=neumann_main,
        eta=eta,
        dims=dims,
        alpha_symbols=alpha_symbols,
    )
