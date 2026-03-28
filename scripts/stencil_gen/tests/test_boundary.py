"""Tests for boundary stencil derivation pipeline (20.3a–20.3g).

Test naming convention:
  - test_taylor_*       : 20.3a (Taylor system builder)
  - test_solve_row_*    : 20.3b (single-row boundary solver)
  - test_conservation_* : 20.3d (conservation constraint solver)
  - test_E4u_*          : 20.3e (E4u end-to-end validation)
  - test_E6u_*          : 20.3f (E6u end-to-end validation)
  - test_E8u_*          : 20.3g (E8u end-to-end validation)
"""

import pytest
from sympy import Matrix, Rational, S, Symbol, cancel, symbols

from stencil_gen.taylor_system import build_taylor_system
from stencil_gen.boundary import solve_boundary_row, BoundaryRow


# ---------------------------------------------------------------------------
# Pipeline helper and fixtures (deferred imports for isolation)
# ---------------------------------------------------------------------------

def _run_pipeline(p, nu=1, s=0):
    """Run derive_boundary + conservation, return full pipeline results."""
    from stencil_gen.boundary import derive_boundary
    from stencil_gen.conservation import build_conservation_system, solve_conservation
    result = derive_boundary(p=p, nu=nu, s=s)
    equations, w_syms, last_free = build_conservation_system(
        result.r, result.t, p, result.rows, result.interior_coeffs)
    solution_dict, updated_rows = solve_conservation(
        equations, w_syms, last_free, result.all_free_params, result.rows)
    return updated_rows, solution_dict, w_syms, result


def _interior_contribution(j, r, p, interior_coeffs):
    """Compute the sum of interior stencil contributions to column j."""
    ic = S.Zero
    for m in range(max(0, j - r - p), j - r + p + 1):
        if m >= 0:
            idx = j - (r + m) + p
            if 0 <= idx <= 2 * p:
                ic += interior_coeffs[idx]
    return ic


@pytest.fixture(scope="module")
def e4u_pipeline():
    """Run E4u pipeline once, reuse across all test_E4u_* functions."""
    return _run_pipeline(p=2)


@pytest.fixture(scope="module")
def e6u_pipeline():
    return _run_pipeline(p=3)


@pytest.fixture(scope="module")
def e8u_pipeline():
    return _run_pipeline(p=4)


# ---------------------------------------------------------------------------
# 20.3a -- Taylor system tests
# ---------------------------------------------------------------------------


def test_taylor_E4u_row0_shape():
    """V shape is (4, 5) for E4u_1 row 0 (i=0, t=5, q=3, nu=1)."""
    V, rhs = build_taylor_system(0, 5, 3, 1)
    assert V.shape == (4, 5)
    assert rhs.shape == (4, 1)


def test_taylor_E4u_row0_entries():
    """V entries match the worked example for E4u_1 row 0."""
    V, rhs = build_taylor_system(0, 5, 3, 1)

    # k=0 row: (j-0)^0 / 0! = 1 for all j
    for j in range(5):
        assert V[0, j] == 1

    # k=1 row: (j-0)^1 / 1! = j
    for j in range(5):
        assert V[1, j] == j

    # k=2 row: j^2 / 2
    assert V[2, 0] == 0
    assert V[2, 1] == Rational(1, 2)
    assert V[2, 2] == 2
    assert V[2, 3] == Rational(9, 2)
    assert V[2, 4] == 8

    # k=3 row: j^3 / 6
    assert V[3, 0] == 0
    assert V[3, 1] == Rational(1, 6)
    assert V[3, 2] == Rational(4, 3)
    assert V[3, 3] == Rational(9, 2)
    assert V[3, 4] == Rational(32, 3)


def test_taylor_E4u_row0_rhs():
    """rhs = [0, 1, 0, 0]^T for nu=1."""
    V, rhs = build_taylor_system(0, 5, 3, 1)
    assert rhs == Matrix([0, 1, 0, 0])


def test_taylor_E4u_row1_spot():
    """Spot-check V for E4u_1 row 1 (i=1)."""
    V, rhs = build_taylor_system(1, 5, 3, 1)
    assert V.shape == (4, 5)
    # k=1: (j - 1)^1 / 1!
    assert V[1, 0] == -1
    assert V[1, 1] == 0
    assert V[1, 2] == 1
    assert V[1, 3] == 2
    assert V[1, 4] == 3


def test_taylor_E8u_row0_shape():
    """V shape is (8, 11) for E8u_1 row 0 (i=0, t=11, q=7, nu=1)."""
    V, rhs = build_taylor_system(0, 11, 7, 1)
    assert V.shape == (8, 11)
    assert rhs.shape == (8, 1)


# ---------------------------------------------------------------------------
# 20.3b -- Single-row boundary solver tests
# ---------------------------------------------------------------------------

a0, a1, a2, a3, a4 = symbols("alpha_0 alpha_1 alpha_2 alpha_3 alpha_4")


def test_solve_row_E4u_row0():
    """E4u row 0: 1 free param (alpha_0), 5 coefficients."""
    result = solve_boundary_row(i=0, t=5, q=3, nu=1, free_symbols=[a0])
    assert result.row_index == 0
    assert len(result.coefficients) == 5
    assert result.free_params == [a0]
    expected = [
        (6 * a0 - 11) / 6,
        3 - 4 * a0,
        (12 * a0 - 3) / 2,
        -(12 * a0 - 1) / 3,
        a0,
    ]
    for got, exp in zip(result.coefficients, expected):
        assert cancel(got - exp) == 0, f"{got} != {exp}"


def test_solve_row_E4u_row1():
    """E4u row 1: 1 free param (alpha_1), 5 coefficients."""
    result = solve_boundary_row(i=1, t=5, q=3, nu=1, free_symbols=[a1])
    assert result.row_index == 1
    assert len(result.coefficients) == 5
    assert result.free_params == [a1]
    expected = [
        (3 * a1 - 1) / 3,
        -(8 * a1 + 1) / 2,
        6 * a1 + 1,
        -(24 * a1 + 1) / 6,
        a1,
    ]
    for got, exp in zip(result.coefficients, expected):
        assert cancel(got - exp) == 0, f"{got} != {exp}"


def test_solve_row_zero_padded():
    """E6u row 0: 2 free slots, second is zero-padded."""
    result = solve_boundary_row(i=0, t=8, q=5, nu=1, free_symbols=[a0, S.Zero])
    assert len(result.coefficients) == 8
    assert result.coefficients[7] == S.Zero
    assert result.coefficients[6] == a0


def test_solve_row_two_free():
    """E6u row 3 (penultimate): 2 active free params."""
    result = solve_boundary_row(
        i=3, t=8, q=5, nu=1, free_symbols=[a3, a4]
    )
    assert len(result.coefficients) == 8
    assert result.coefficients[6] == a3
    assert result.coefficients[7] == a4


# ---------------------------------------------------------------------------
# 20.3d -- Conservation constraint solver tests
# ---------------------------------------------------------------------------


def test_conservation_weight_count_E4u(e4u_pipeline):
    """Verify 3 weight symbols exist and are solved."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    assert len(w_syms) == 3
    for w in w_syms:
        assert w in solution_dict


def test_conservation_placeholders_resolved_E4u(e4u_pipeline):
    """Verify last row has no phi_* symbols remaining."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    last_row = updated_rows[2]
    allowed = set(result.all_free_params)
    for coeff in last_row.coefficients:
        assert coeff.free_symbols <= allowed, (
            f"Unexpected symbols in last row: {coeff.free_symbols - allowed}"
        )


def test_conservation_redundant_column_E4u(e4u_pipeline):
    """Verify the redundant column (t-1=4) sums to zero."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    w_exprs = [solution_dict[w] for w in w_syms]
    col_sum = sum(w * row.coefficients[4]
                  for w, row in zip(w_exprs, updated_rows))
    col_sum += _interior_contribution(4, result.r, 2, result.interior_coeffs)
    assert cancel(col_sum) == 0


# ---------------------------------------------------------------------------
# 20.3e -- E4u_1 end-to-end validation tests
# ---------------------------------------------------------------------------

# Alpha symbols for E4u
_a0, _a1 = symbols("alpha_0 alpha_1")

# Alpha values from E4u_1.t.cpp
_alpha_vals_e4 = {
    _a0: -0.7733323791884821,
    _a1:  0.1623961700641681,
}


def test_E4u_taylor_shape_and_entries(e4u_pipeline):
    """Taylor system shape and specific entries for E4u row 0."""
    V, rhs = build_taylor_system(0, 5, 3, 1)
    assert V.shape == (4, 5)
    assert V[0, 0] == 1
    assert V[1, 1] == 1
    assert V[2, 3] == Rational(9, 2)
    assert V[3, 4] == Rational(32, 3)


def test_E4u_row0_symbolic(e4u_pipeline):
    """Row 0 symbolic coefficients match E4u_1.cpp lines 80-84."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    row = updated_rows[0]
    expected = [
        (6 * _a0 - 11) / 6,
        3 - 4 * _a0,
        (12 * _a0 - 3) / 2,
        -(12 * _a0 - 1) / 3,
        _a0,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 0 coeff {i}: {got} != {exp}"


def test_E4u_row1_symbolic(e4u_pipeline):
    """Row 1 symbolic coefficients match E4u_1.cpp lines 85-89."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    row = updated_rows[1]
    expected = [
        (3 * _a1 - 1) / 3,
        -(8 * _a1 + 1) / 2,
        6 * _a1 + 1,
        -(24 * _a1 + 1) / 6,
        _a1,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 1 coeff {i}: {got} != {exp}"


def test_E4u_row2_symbolic(e4u_pipeline):
    """Row 2 (conservation-constrained) symbolic coefficients match E4u_1.cpp lines 90-94."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    row = updated_rows[2]
    expected = [
        -(168 * _a1 + 54 * _a0 - 11) / 138,
        (112 * _a1 + 36 * _a0 - 15) / 23,
        -(336 * _a1 + 108 * _a0 + 1) / 46,
        (336 * _a1 + 108 * _a0 + 47) / 69,
        -(28 * _a1 + 9 * _a0 + 2) / 23,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 2 coeff {i}: {got} != {exp}"


def test_E4u_numerical_floating(e4u_pipeline):
    """Numerical evaluation (floating, h=2) matches E4u_1.t.cpp."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    h = 2
    expected_float = [
        -1.3033328562609077, 3.046664758376964, -3.069997137565446,
        1.713331425043631, -0.38666618959424104,
        -0.08546858163458262, -0.5747923401283361, 0.9871885101925043,
        -0.4081256734616695, 0.08119808503208405,
        0.0923093909615862, -0.5359042305130115, 0.3038563457695172,
        0.13076243615365518, 0.00897605762825287,
    ]
    computed = []
    for row in updated_rows:
        for coeff in row.coefficients:
            val = float(coeff.xreplace(_alpha_vals_e4)) / h
            computed.append(val)
    assert len(computed) == len(expected_float)
    for i, (got, exp) in enumerate(zip(computed, expected_float)):
        assert abs(got - exp) < 1e-12, f"Floating coeff {i}: {got} != {exp}"


def test_E4u_numerical_dirichlet(e4u_pipeline):
    """Numerical evaluation (Dirichlet, h=0.5) matches E4u_1.t.cpp."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    h = 0.5
    # Dirichlet drops row 0, uses rows 1 and 2
    expected_dirichlet = [
        -0.3418743265383305, -2.2991693605133445, 3.9487540407700172,
        -1.632502693846678, 0.3247923401283362,
        0.3692375638463448, -2.143616922052046, 1.2154253830780688,
        0.5230497446146207, 0.03590423051301148,
    ]
    computed = []
    for row in updated_rows[1:]:  # skip row 0
        for coeff in row.coefficients:
            val = float(coeff.xreplace(_alpha_vals_e4)) / h
            computed.append(val)
    assert len(computed) == len(expected_dirichlet)
    for i, (got, exp) in enumerate(zip(computed, expected_dirichlet)):
        assert abs(got - exp) < 1e-12, f"Dirichlet coeff {i}: {got} != {exp}"


def test_E4u_conservation_column_sums(e4u_pipeline):
    """Conservation verification: weighted column sums satisfy SBP."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    w_exprs = [solution_dict[w] for w in w_syms]
    t = result.t  # 5
    r = result.r  # 3
    p = 2

    for j in range(t):
        col_sum = sum(
            w * row.coefficients[j]
            for w, row in zip(w_exprs, updated_rows)
        )
        col_sum += _interior_contribution(j, r, p, result.interior_coeffs)
        if j == 0:
            # Column 0 sums to -1
            assert cancel(col_sum + 1) == 0, f"Column {j} SBP failed"
        else:
            # All other columns sum to 0
            assert cancel(col_sum) == 0, f"Column {j} SBP failed"


def test_E4u_polynomial_exactness(e4u_pipeline):
    """Polynomial exactness up to degree q=3."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    t = result.t  # 5

    for d in range(4):  # degrees 0, 1, 2, 3
        # Grid values f(j) = j^d for j = 0..t-1
        grid_vals = [Rational(j) ** d for j in range(t)]
        for row in updated_rows:
            i = row.row_index
            # Apply stencil: sum_j coeff_j * f(j)
            stencil_result = sum(
                c * fj for c, fj in zip(row.coefficients, grid_vals)
            )
            # Expected: d-th derivative of x^d at x=i
            if d == 0:
                expected = 0
            elif d == 1:
                expected = 1
            else:
                # d-th derivative of x^d w.r.t. first derivative = d * i^(d-1)
                expected = d * Rational(i) ** (d - 1)
            assert cancel(stencil_result - expected) == 0, (
                f"Poly exactness failed: d={d}, row={i}, "
                f"got {stencil_result}, expected {expected}"
            )
