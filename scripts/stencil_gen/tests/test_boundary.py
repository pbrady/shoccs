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
from sympy import Matrix, Rational, S

from stencil_gen.taylor_system import build_taylor_system


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
