"""Role bits, interior masks and the exception CSR on known tables."""

from pathlib import Path

import numpy as np
import pytest

from pyshoccs.structure import (StructureError, StructuredOperator, apply_structured,
                                compare_with_export, derive_structure)
from pyshoccs.tables import Coo, Layout, Lines, RRecords, Tables


def two_crossing_lines():
    """A 5 x 5 grid with one x-line (j = 2) and one y-line (i = 2) of E2 rows.

    Each line has a dense row at both ends (part 0 and 2) and interior rows in
    between; the crossing point (2, 2) is interior in both directions. One
    extra "B-like" entry on the x-line's left dense row points at an R column.
    """
    nx = ny = 5
    n_D = nx * ny
    n_V = n_D + 1
    h = 0.25
    taps = np.array([[1, -2, 1], [1, -2, 1], [0, 0, 0]], dtype=float) / h**2

    def ic(i, j):
        return i * ny + j

    def line(d):
        row, col, val, part = [], [], [], []
        for p, pos in ((0, 1), (1, 2), (2, 3)):
            for jj, c in enumerate(taps[d]):
                i, j = (pos, 2) if d == 0 else (2, pos)
                r = ic(i, j)
                cc = ic(pos + jj - 1, 2) if d == 0 else ic(2, pos + jj - 1)
                row.append(r); col.append(cc); val.append(c); part.append(p)
        if d == 0:
            row.append(ic(1, 2)); col.append(n_D); val.append(7.0); part.append(3)
        return Coo(np.array(row, np.int64), np.array(col, np.int64), np.array(val),
                   np.array(part, np.uint8), np.full(len(row), 0, np.int32))

    L = Layout((nx, ny, 1), n_D, (1, 0, 0), {"D": 0, "Rx": n_D, "Ry": n_V, "Rz": n_V}, n_V)
    R0 = RRecords(np.array([0.5]), np.array([[0.3, 0.5, 0.0]]), np.array([[-1.0, 0, 0]]),
                  np.zeros(1, np.uint8), np.array([[0, 2, 0]], np.int32), np.array([ic(0, 2)]),
                  np.zeros(1, np.int32), np.zeros(0, np.int64), np.array([0]))
    Rempty = RRecords(*(np.zeros(0) for _ in range(9)))
    meta = {"extents": [nx, ny, 1], "dims": 2, "h": [h, h, 1.0], "grid_bcs": [["dirichlet"] * 2] * 3,
            "object_bcs": ["floating"], "system": {"type": "heat", "diffusivity": 1.0}, "dt": 0.01,
            "mms": {"type": "gaussian", "center": [[0.5, 0.5]], "variance": [[1, 1]],
                    "amplitude": [1], "frequency": [1]}}
    fluid = np.arange(n_D)
    faces = np.array([ic(0, j) for j in range(ny)] + [ic(4, j) for j in range(ny)]
                     + [ic(i, 0) for i in range(nx)] + [ic(i, 4) for i in range(nx)])
    lines = [Lines(np.zeros((1, 7), np.int64), np.full((1, 2), np.nan), np.zeros(1, np.uint8))] * 3
    return Tables(Path("<memory>"), meta, L, np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.zeros(1),
                  [R0, Rempty, Rempty], lines, fluid, faces, taps, {},
                  [line(0), line(1), Coo.empty()], [Coo.empty()] * 3, None, {}, {}, None)


def test_role_bits_and_masks_on_crossing_lines():
    t = two_crossing_lines()
    s = derive_structure(t)
    ic = lambda i, j: i * 5 + j  # noqa: E731
    assert s.role[ic(2, 2)] & 3 == 3          # interior in x and y
    assert s.role[ic(1, 2)] & 3 == 0          # dense row, no interior bit
    assert s.role[ic(3, 2)] & 3 == 0
    assert s.role[ic(2, 1)] & 3 == 0 and s.role[ic(2, 3)] & 3 == 0
    assert s.role[ic(2, 2)] & 8 and s.role[ic(0, 0)] & 16
    assert s.interior[0].sum() == 1 and s.interior[1].sum() == 1
    assert s.interior[2] is None
    # exception rows: the four dense rows; the crossing row has none
    assert list(s.exc_rows) == sorted([ic(1, 2), ic(3, 2), ic(2, 1), ic(2, 3)])
    assert s.exc.indices.dtype == np.int32
    # the B-like entry survives in exc on the left dense row
    r = list(s.exc_rows).index(ic(1, 2))
    cols = s.exc.indices[s.exc.indptr[r]:s.exc.indptr[r + 1]]
    assert t.layout.n_D in cols
    assert s.counts["exc_rows"] == 4 and s.counts["interior_rows_per_dir"] == [1, 1, 0]


def test_structured_apply_equals_csr_on_crossing_lines():
    t = two_crossing_lines()
    rng = np.random.default_rng(1)
    u = rng.uniform(-1, 1, t.layout.n_V)
    ref = t.to_scipy() @ u
    got = apply_structured(t, u)
    assert np.allclose(got, ref, rtol=1e-14, atol=1e-12)
    # the structured operator reuses its buffers across calls
    op = StructuredOperator(t)
    out = np.empty(t.layout.n_V)
    op.apply(u, out)
    op.apply(2 * u, out)
    assert np.allclose(out, 2 * ref, rtol=1e-14, atol=1e-12)


def test_tap_check_catches_wrong_interior_coefficient():
    t = two_crossing_lines()
    t.A[0].val[4] *= 1.5
    with pytest.raises(StructureError, match="interior coefficient"):
        derive_structure(t)
    derive_structure(t, check_taps=False)


def test_synthetic_roles_and_cross_check(synthetic_tables):
    t = synthetic_tables
    s = derive_structure(t)
    ic = lambda i, j: i * 5 + j  # noqa: E731
    for j in (1, 2, 3):
        assert s.role[ic(2, j)] & 1 and s.role[ic(3, j)] & 1
        assert not (s.role[ic(1, j)] & 1) and not (s.role[ic(4, j)] & 1)
    for i in (1, 2, 3, 4):
        assert s.role[ic(i, 2)] & 2 and not (s.role[ic(i, 1)] & 2)
    assert not np.any(s.role[ic(5, 0):] & 8)   # the slab is solid
    rep = compare_with_export(t, s)
    assert rep["role"]["present"] and rep["role"]["match"], rep
    assert not rep["exc"]["present"]
    # the Dirichlet R row has no exception row, the two floating ones do
    assert t.layout.n_D + 2 not in s.exc_rows
    assert t.layout.n_D in s.exc_rows and t.layout.n_D + 1 in s.exc_rows


def test_synthetic_structured_equals_scipy(synthetic_tables):
    t = synthetic_tables
    u = t.oracle["rand_V"]
    assert np.allclose(apply_structured(t, u), t.to_scipy() @ u, rtol=1e-14, atol=1e-12)
