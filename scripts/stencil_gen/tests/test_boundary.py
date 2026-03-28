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
