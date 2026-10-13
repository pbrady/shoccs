"""The axial and transverse monomial oracles on a hand-built transverse R row."""

from pathlib import Path

import numpy as np
import pytest

from pyshoccs.oracle import monomial_oracle, transverse_oracle
from pyshoccs.tables import Coo, Layout, Lines, RRecords, Tables

PSI = 0.3


def transverse_row_tables():
    """A 6 x 5 grid with one Rx record at (x_2 + psi h, y_1) and the legacy
    transverse formulation of its y-derivative in A1: the exact R value and two
    cross-line values linearly interpolated between the columns i = 2 and
    i = 3 under a one-sided (1, -2, 1) / h^2 difference. A0 holds the
    along-ray row through (x_R, x_2, x_1), exact for quadratics.
    """
    nx, ny = 6, 5
    n_D = nx * ny
    n_V = n_D + 1
    h = 0.25
    x = np.linspace(0, 1.25, nx)
    y = np.linspace(0, 1.0, ny)
    assert np.isclose(x[1] - x[0], h) and np.isclose(y[1] - y[0], h)

    def ic(i, j):
        return i * ny + j

    x_R = x[2] + PSI * h
    R = n_D
    w2, w3 = 1 - PSI, PSI

    # A1: y-derivative at the R point, rows of Bf_x (5) and Br_x (6)
    row = [R] * 5
    col = [R, ic(2, 2), ic(3, 2), ic(2, 3), ic(3, 3)]
    val = np.array([1.0, -2 * w2, -2 * w3, w2, w3]) / h**2
    part = [6, 5, 5, 5, 5]
    A1 = Coo(np.array(row, np.int64), np.array(col, np.int64), val, np.array(part, np.uint8),
             np.full(5, -1, np.int32))

    # A0: x-derivative at the R point from (x_R, x_2, x_1)
    a, b, c = x_R, x[2], x[1]
    cR, c2, c1 = (2 / ((a - b) * (a - c)), 2 / ((b - a) * (b - c)), 2 / ((c - a) * (c - b)))
    A0 = Coo(np.array([R, R, R], np.int64), np.array([ic(2, 1), ic(1, 1), R], np.int64),
             np.array([c2, c1, cR]), np.array([5, 5, 6], np.uint8), np.full(3, -1, np.int32))

    L = Layout((nx, ny, 1), n_D, (1, 0, 0), {"D": 0, "Rx": n_D, "Ry": n_V, "Rz": n_V}, n_V)
    R0 = RRecords(np.array([PSI]), np.array([[x_R, y[1], 0.0]]), np.array([[1.0, 0, 0]]),
                  np.zeros(1, np.uint8), np.array([[3, 1, 0]], np.int32), np.array([ic(3, 1)]),
                  np.zeros(1, np.int32), np.zeros(0, np.int64), np.array([0]))
    Rempty = RRecords(*(np.zeros(0) for _ in range(9)))
    meta = {"extents": [nx, ny, 1], "dims": 2, "h": [h, h, 0.0], "grid_bcs": [["dirichlet"] * 2] * 3,
            "object_bcs": ["floating"], "system": {"type": "heat", "diffusivity": 1.0}, "dt": 0.01,
            "mms": {"type": "gaussian", "center": [[0.5, 0.5]], "variance": [[1, 1]],
                    "amplitude": [1], "frequency": [1]}}
    taps = np.array([[1, -2, 1], [1, -2, 1], [0, 0, 0]], dtype=float) / h**2
    lines = [Lines(np.zeros((0, 7), np.int64), np.zeros((0, 2)), np.zeros(0, np.uint8))] * 3
    return Tables(Path("<memory>"), meta, L, x, y, np.zeros(1), [R0, Rempty, Rempty], lines,
                  np.arange(n_D), np.zeros(0, np.int64), taps, {}, [A0, A1, Coo.empty()],
                  [Coo.empty()] * 3, None, {}, {}, None)


def test_axial_oracle_on_the_r_row():
    t = transverse_row_tables()
    orc = monomial_oracle(t, kmax=3, include_closure=False)
    assert orc.gate_passed()
    for d in (0, 1):
        for k in range(3):
            rec = orc.select(cls="R", d=d, k=k)
            assert rec.size == 1 and rec["passed"][0], (d, k, rec)
            assert rec["coord"][0] == d and np.isclose(rec["psi"][0], PSI)
    # the one-sided (1, -2, 1) difference misses x^3 by exactly 6 h
    k3 = orc.select(cls="R", d=1, k=3)
    assert not k3["passed"][0]
    assert np.isclose(k3["residual"][0], 6 * 0.25)


def test_transverse_oracle_sees_the_linear_interpolation():
    t = transverse_row_tables()
    orc = transverse_oracle(t, mmax=2)
    lin = orc.select(cls="R_transverse", d=1, coord=0, k=1)
    quad = orc.select(cls="R_transverse", d=1, coord=0, k=2)
    assert lin.size == 1 and quad.size == 1
    assert lin["passed"][0]
    assert not quad["passed"][0]
    # exact R value minus the interpolated x^2: the error psi (1 - psi) h^2 / h^2
    assert np.isclose(quad["residual"][0], PSI * (1 - PSI))
    # the along-ray row in A0 has no transverse record in direction 0 beyond coord 1
    along = orc.select(cls="R_transverse", d=0, coord=1)
    assert along.size == 2 and np.all(along["passed"])
    s = orc.summary()
    assert s["R_transverse:k=2"]["n_fail"] == 1 and s["R_transverse:k=2"]["worst_dir"] == 1
    assert "R_transverse" in orc.format()


def test_transverse_oracle_on_along_ray_rows_only(synthetic_tables):
    # the synthetic export has two Rx rows, both along their ray in A0 and none in
    # A1: y^m is constant over such a row's columns, so every record passes
    orc = transverse_oracle(synthetic_tables)
    assert orc.table.size == 4
    assert np.all(orc.table["dir"] == 0) and np.all(orc.table["coord"] == 1)
    assert np.all(orc.table["passed"])
    assert orc.summary()["R_transverse:k=2"]["n_fail"] == 0
    combined = monomial_oracle(synthetic_tables).combined(orc)
    assert combined.gate_passed()
    assert set(combined.summary()) >= {"R:k=3", "D_closure:k=2", "R_transverse:k=1"}
