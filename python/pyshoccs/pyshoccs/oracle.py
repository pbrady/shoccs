"""The monomial oracle of spec Section 3, check 3.

Axial oracle: for direction d and k in 0..3 the vector ``x_d^k`` over V (grid
coordinate for D, intersection position for R) is pushed through the rows of
``A_d`` and compared with ``k (k-1) x_d^(k-2)`` at the row's own location (0
for k < 2). A row passes when ``|residual| <= 16 eps sum_j |a_j x_j^k|``.
R rows are the gate (k <= 2; k = 3 is reported, not gated); fluid D closure
rows (left and right dense rows, rows receiving B entries) get the same test
as a diagnostic.

Transverse oracle (reported, not gated): the R rows of ``A_d`` applied to
``x_c^m`` for every other active coordinate c and m = 1, 2, whose exact second
x_d-derivative is 0. The axial monomials are constant across the grid lines a
transverse R row interpolates between, so only these see the interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tables import Tables, SEGMENT_NAMES

EPS = np.finfo(np.float64).eps
CLASSES = ("R", "D_closure", "R_transverse")
ROW_DTYPE = np.dtype([
    ("dir", np.int8), ("coord", np.int8), ("row", np.int64), ("segment", "U2"), ("cls", "U12"),
    ("k", np.int8), ("value", np.float64), ("expected", np.float64),
    ("residual", np.float64), ("tol", np.float64), ("passed", np.bool_),
    ("psi", np.float64), ("psi_min", np.float64), ("nnz", np.int32),
])


@dataclass
class OracleResult:
    """One record per (row, direction, monomial); ``psi`` is the row's own record
    psi (NaN on D rows), ``psi_min`` the smallest psi among its R columns."""

    table: np.ndarray
    kmax: int

    def select(self, cls: str | None = None, kmax: int | None = None, k: int | None = None,
               d: int | None = None, coord: int | None = None) -> np.ndarray:
        t = self.table
        m = np.ones(t.size, dtype=bool)
        if cls is not None:
            m &= t["cls"] == cls
        if kmax is not None:
            m &= t["k"] <= kmax
        if k is not None:
            m &= t["k"] == k
        if d is not None:
            m &= t["dir"] == d
        if coord is not None:
            m &= t["coord"] == coord
        return t[m]

    def worst(self, **sel) -> np.void | None:
        t = self.select(**sel)
        if t.size == 0:
            return None
        ratio = t["residual"] / np.maximum(t["tol"], np.finfo(np.float64).tiny)
        return t[int(np.argmax(ratio))]

    def summary(self) -> dict:
        out = {}
        for cls in CLASSES:
            for k in np.unique(self.table["k"]).tolist():
                t = self.select(cls=cls, k=k)
                if t.size == 0:
                    continue
                w = self.worst(cls=cls, k=k)
                out[f"{cls}:k={k}"] = {
                    "n": int(t.size),
                    "n_fail": int((~t["passed"]).sum()),
                    "max_residual": float(t["residual"].max()),
                    "max_ratio": float((t["residual"] / np.maximum(t["tol"], 1e-300)).max()),
                    "worst_row": int(w["row"]),
                    "worst_segment": str(w["segment"]),
                    "worst_dir": int(w["dir"]),
                    "worst_coord": int(w["coord"]),
                    "worst_psi": float(w["psi"]),
                    "worst_psi_min": float(w["psi_min"]),
                }
        return out

    def gate_passed(self) -> bool:
        g = self.select(cls="R", kmax=2)
        return bool(np.all(g["passed"])) if g.size else True

    def combined(self, other: "OracleResult") -> "OracleResult":
        return OracleResult(np.concatenate([self.table, other.table]), max(self.kmax, other.kmax))

    def format(self) -> str:
        lines = [f"{'class':12s} {'k':>2s} {'rows':>7s} {'fail':>6s} {'max residual':>13s} "
                 f"{'max ratio':>10s}  worst row"]
        for key, s in self.summary().items():
            cls, kk = key.split(":k=")
            lines.append(f"{cls:12s} {kk:>2s} {s['n']:7d} {s['n_fail']:6d} {s['max_residual']:13.3e} "
                         f"{s['max_ratio']:10.3g}  V[{s['worst_row']}] {s['worst_segment']} "
                         f"dir {s['worst_dir']} coord {s['worst_coord']} psi {s['worst_psi']:.3e} "
                         f"psi_min {s['worst_psi_min']:.3e}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------------


def _own_psi(tables: Tables, rows: np.ndarray) -> np.ndarray:
    """psi of the row's own record for R rows, NaN for D rows."""
    L = tables.layout
    psi = np.full(rows.size, np.nan)
    seg = L.segment_of(rows)
    for r in range(3):
        m = seg == r + 1
        if np.any(m):
            psi[m] = tables.R[r].psi[rows[m] - L.r_offset(r)]
    return psi


def _psi_min(tables: Tables, d: int) -> np.ndarray:
    """Per V row: the smallest psi among the R columns of A_d it touches (NaN when none)."""
    L = tables.layout
    c = tables.A[d]
    psi_min = np.full(L.n_V, np.inf)
    is_Rcol = c.col >= L.n_D
    if np.any(is_Rcol):
        rcol = c.col[is_Rcol]
        seg = L.segment_of(rcol) - 1
        psi = np.empty(rcol.size)
        for r in range(3):
            m = seg == r
            if np.any(m):
                psi[m] = tables.R[r].psi[rcol[m] - L.r_offset(r)]
        np.minimum.at(psi_min, c.row[is_Rcol], psi)
    psi_min[np.isinf(psi_min)] = np.nan
    return psi_min


def _records(d: int, coord: int, rows: np.ndarray, segnames: np.ndarray, cls: np.ndarray | str,
             k: int, value: np.ndarray, expected: np.ndarray, tol: np.ndarray,
             psi: np.ndarray, psi_min: np.ndarray, nnz: np.ndarray) -> np.ndarray:
    t = np.empty(rows.size, dtype=ROW_DTYPE)
    t["dir"] = d
    t["coord"] = coord
    t["row"] = rows
    t["segment"] = segnames
    t["cls"] = cls
    t["k"] = k
    t["value"] = value
    t["expected"] = expected
    res = np.abs(value - expected)
    t["residual"] = res
    t["tol"] = tol
    t["passed"] = res <= tol
    t["psi"] = psi
    t["psi_min"] = psi_min
    t["nnz"] = nnz
    return t


# ----------------------------------------------------------------------------
# The two oracles
# ----------------------------------------------------------------------------


def monomial_oracle(tables: Tables, kmax: int = 3, include_closure: bool = True,
                    dirs=None) -> OracleResult:
    """Axial monomials ``x_d^k`` on the R rows (and fluid D closure rows) of ``A_d``."""
    L = tables.layout
    n_D, n_V = L.n_D, L.n_V
    loc = tables.locations
    dirs = list(L.active_dims) if dirs is None else list(dirs)
    recs = []
    for d in dirs:
        c = tables.A[d]
        if c.nnz == 0:
            continue
        A = tables.to_scipy(dirs=(d,))
        absA = abs(A)
        nnz_row = np.diff(A.indptr)
        psi_min = _psi_min(tables, d)
        R_rows = np.unique(c.row[c.row >= n_D])
        rows = [R_rows]
        cls = [np.full(R_rows.size, "R")]
        if include_closure:
            closure = np.unique(c.row[(c.row < n_D) & np.isin(c.part, (0, 2, 3))])
            closure = closure[tables.is_fluid_D[closure] & ~tables.is_dirichlet_face_D[closure]]
            rows.append(closure)
            cls.append(np.full(closure.size, "D_closure"))
        rows = np.concatenate(rows)
        cls = np.concatenate(cls)
        if rows.size == 0:
            continue
        x = loc[:, d]
        xr = x[rows]
        segnames = np.array(SEGMENT_NAMES)[L.segment_of(rows)]
        psi = _own_psi(tables, rows)
        for k in range(kmax + 1):
            xk = np.ones(n_V) if k == 0 else x ** k
            y = (A @ xk)[rows]
            tol = 16 * EPS * (absA @ np.abs(xk))[rows]
            if k < 2:
                expected = np.zeros(rows.size)
            else:
                expected = k * (k - 1) * (np.ones(rows.size) if k == 2 else xr ** (k - 2))
            recs.append(_records(d, d, rows, segnames, cls, k, y, expected, tol,
                                 psi, psi_min[rows], nnz_row[rows]))
    table = np.concatenate(recs) if recs else np.empty(0, dtype=ROW_DTYPE)
    return OracleResult(table, kmax)


def transverse_oracle(tables: Tables, mmax: int = 2, dirs=None) -> OracleResult:
    """Transverse monomials ``x_c^m`` (c != d) on the R rows of ``A_d``; exact value 0."""
    L = tables.layout
    n_D = L.n_D
    loc = tables.locations
    dirs = list(L.active_dims) if dirs is None else list(dirs)
    recs = []
    for d in dirs:
        c = tables.A[d]
        if c.nnz == 0:
            continue
        rows = np.unique(c.row[c.row >= n_D])
        if rows.size == 0:
            continue
        A = tables.to_scipy(dirs=(d,))
        absA = abs(A)
        nnz_row = np.diff(A.indptr)[rows]
        psi = _own_psi(tables, rows)
        psi_min = _psi_min(tables, d)[rows]
        segnames = np.array(SEGMENT_NAMES)[L.segment_of(rows)]
        zero = np.zeros(rows.size)
        for cc in L.active_dims:
            if cc == d:
                continue
            x = loc[:, cc]
            for m in range(1, mmax + 1):
                xm = x ** m
                y = (A @ xm)[rows]
                tol = 16 * EPS * (absA @ np.abs(xm))[rows]
                recs.append(_records(d, cc, rows, segnames, "R_transverse", m, y, zero, tol,
                                     psi, psi_min, nnz_row))
    table = np.concatenate(recs) if recs else np.empty(0, dtype=ROW_DTYPE)
    return OracleResult(table, mmax)


def describe_row_entries(tables: Tables, d: int, row: int) -> str:
    """The raw entries of one row of A_d, for debugging a failing oracle row."""
    c = tables.A[d]
    m = c.row == row
    L = tables.layout
    loc = tables.locations
    lines = [f"A{d} row {L.describe_row(row)} at x_{d} = {loc[row, d]!r}: {int(m.sum())} entries"]
    from .structure import part_name

    for col, val, part in zip(c.col[m], c.val[m], c.part[m]):
        extra = ""
        if col >= L.n_D:
            r = int(L.segment_of(col)) - 1
            extra = f" psi={tables.R[r].psi[col - L.r_offset(r)]:.6e}"
        lines.append(f"  {part_name(part):10s} col {L.describe_row(int(col)):40s} x_{d}={loc[col, d]!r} "
                     f"coef {val!r}{extra}")
    return "\n".join(lines)
