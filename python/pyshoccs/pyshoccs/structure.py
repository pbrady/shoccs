"""The structured tables of spec Section 4, derived from the raw COO.

- ``role`` (u8 per D point): bit d set iff the point is an interior-circulant
  row of a line in direction d (an ``A{d}_part == 1`` entry on that row);
  bit 3 fluid; bit 4 Dirichlet grid face.
- the interior mask per direction, ``(nx, ny, nz)`` booleans from bit d;
- ``exc``: one CSR over V holding every entry that is not an interior tap of a
  row interior in that entry's direction (dense left and right rows, B, every
  Bf and Br row), duplicates summed, int32 columns, rows sorted and unique.

``apply_structured`` is then: per direction an axis stencil on a zero-padded
copy of the D block, multiplied by the direction's interior mask, summed over
directions, plus the exception CSR added on its rows. R rows have no bits, so
they receive exactly their exception row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .tables import Tables, PART_NAMES

ROLE_FLUID = 8
ROLE_DIRICHLET = 16


class StructureError(RuntimeError):
    """The COO does not have the shape the structured form assumes."""


@dataclass
class Structure:
    role: np.ndarray
    interior: list[np.ndarray | None]
    taps: np.ndarray
    exc_rows: np.ndarray
    exc: sp.csr_matrix
    counts: dict

    @property
    def half(self) -> int:
        return self.taps.shape[1] // 2

    @property
    def width(self) -> int:
        return self.taps.shape[1]

    def exc_full(self, n_V: int) -> sp.csr_matrix:
        """The exception CSR as a full-height (n_V, n_V) matrix."""
        coo = self.exc.tocoo()
        rows = self.exc_rows[coo.row]
        return sp.csr_matrix((coo.data, (rows, coo.col)), shape=(n_V, n_V))


def _check_interior_taps(tables: Tables, d: int, role_bit_rows: np.ndarray) -> None:
    """Verify that the part == 1 entries of direction d are exactly the taps."""
    c = tables.A[d]
    sel = c.part == 1
    if not np.any(sel):
        return
    w = tables.taps.shape[1]
    half = w // 2
    stride = tables.layout.strides[d]
    row = c.row[sel]
    col = c.col[sel]
    val = c.val[sel]
    n_D = tables.layout.n_D
    if row.max() >= n_D or col.max() >= n_D:
        bad = int(np.argmax((row >= n_D) | (col >= n_D)))
        raise StructureError(
            f"A{d}: interior entry {bad} touches an R row or column: "
            f"{tables.layout.describe_row(int(row[bad]))} <- V[{col[bad]}]")
    off = col - row
    j, rem = np.divmod(off, stride)
    j = j + half
    ok = (rem == 0) & (j >= 0) & (j < w)
    if not np.all(ok):
        bad = int(np.argmax(~ok))
        raise StructureError(
            f"A{d}: interior entry on {tables.layout.describe_row(int(row[bad]))} has column "
            f"offset {off[bad]} which is not (j - {half}) * stride {stride} for j in 0..{w - 1}")
    counts = np.bincount(row, minlength=n_D)
    rows_hit = np.flatnonzero(counts)
    if np.any(counts[rows_hit] != w):
        bad = int(rows_hit[np.argmax(counts[rows_hit] != w)])
        raise StructureError(
            f"A{d}: {tables.layout.describe_row(bad)} has {counts[bad]} interior entries, "
            f"expected the full tap width {w}")
    want = tables.taps[d][j]
    diff = np.abs(val - want)
    tol = 4 * np.finfo(np.float64).eps * np.abs(want)
    if np.any(diff > tol):
        bad = int(np.argmax(diff - tol))
        raise StructureError(
            f"A{d}: interior coefficient on {tables.layout.describe_row(int(row[bad]))} tap j={j[bad]} "
            f"is {val[bad]!r}, taps.npy says {want[bad]!r} (|diff| {diff[bad]:.3e})")


def derive_structure(tables: Tables, check_taps: bool = True) -> Structure:
    """Derive role bits, interior masks and the exception CSR from the raw COO."""
    L = tables.layout
    n_D, n_V = L.n_D, L.n_V
    role = np.zeros(n_D, dtype=np.uint8)
    interior: list[np.ndarray | None] = [None, None, None]

    for d in range(3):
        c = tables.A[d]
        sel = c.part == 1
        if not np.any(sel):
            continue
        rows = np.unique(c.row[sel])
        if rows.max() >= n_D:
            raise StructureError(f"A{d}: an interior (part 1) entry sits on an R row")
        if check_taps:
            _check_interior_taps(tables, d, rows)
        role[rows] |= np.uint8(1 << d)
        interior[d] = ((role & (1 << d)) != 0).reshape(L.extents)
    role[tables.fluid_idx] |= np.uint8(ROLE_FLUID)
    role[tables.dirichlet_face_idx] |= np.uint8(ROLE_DIRICHLET)

    # Exception entries: everything that is not an interior tap of a row that
    # is interior in that entry's direction. R rows have no bits.
    rows_l, cols_l, vals_l = [], [], []
    for d in range(3):
        c = tables.A[d]
        if c.nnz == 0:
            continue
        bit = np.zeros(c.nnz, dtype=bool)
        is_D = c.row < n_D
        bit[is_D] = (role[c.row[is_D]] & (1 << d)) != 0
        keep = ~((c.part == 1) & bit)
        rows_l.append(c.row[keep])
        cols_l.append(c.col[keep])
        vals_l.append(c.val[keep])
    if rows_l:
        rows = np.concatenate(rows_l)
        cols = np.concatenate(cols_l)
        vals = np.concatenate(vals_l)
    else:
        rows = np.zeros(0, np.int64)
        cols = np.zeros(0, np.int64)
        vals = np.zeros(0)
    full = sp.coo_matrix((vals, (rows, cols)), shape=(n_V, n_V)).tocsr()
    full.sum_duplicates()
    full.sort_indices()
    exc_rows = np.flatnonzero(np.diff(full.indptr)).astype(np.int64)
    exc = full[exc_rows]
    exc = sp.csr_matrix((exc.data, exc.indices.astype(np.int32), exc.indptr.astype(np.int32)),
                        shape=(exc_rows.size, n_V))
    exc.sort_indices()

    seg = L.segment_of(exc_rows) if exc_rows.size else np.zeros(0, np.int64)
    counts = {
        "n_D": n_D,
        "n_V": n_V,
        "interior_rows_per_dir": [int(((role & (1 << d)) != 0).sum()) for d in range(3)],
        "fluid": int(tables.fluid_idx.size),
        "dirichlet_faces": int(np.unique(tables.dirichlet_face_idx).size),
        "exc_rows": int(exc_rows.size),
        "exc_rows_D": int((seg == 0).sum()),
        "exc_rows_R": int((seg > 0).sum()),
        "exc_nnz": int(exc.nnz),
        "exc_max_row_nnz": int(np.diff(exc.indptr).max()) if exc_rows.size else 0,
        "tap_width": int(tables.taps.shape[1]),
    }
    return Structure(role, interior, tables.taps.copy(), exc_rows, exc, counts)


class StructuredOperator:
    """``out = A u`` in the structured form with preallocated work buffers."""

    def __init__(self, tables: Tables, structure: Structure | None = None):
        self.tables = tables
        self.s = structure if structure is not None else derive_structure(tables)
        L = tables.layout
        self.ext = L.extents
        self.n_D = L.n_D
        self.n_V = L.n_V
        half = self.s.half
        self.half = half
        self.dirs = [d for d in L.active_dims if self.s.interior[d] is not None]
        pad = tuple((half, half) if d in self.dirs else (0, 0) for d in range(3))
        self.pad = pad
        pshape = tuple(self.ext[d] + pad[d][0] + pad[d][1] for d in range(3))
        self._padded = np.zeros(pshape)
        self._center = tuple(slice(pad[d][0], pad[d][0] + self.ext[d]) for d in range(3))
        self._acc = np.empty(self.ext)
        self._tmp = np.empty(self.ext)
        self._total = np.empty(self.ext)
        self._exc_out = np.empty(self.s.exc_rows.size)
        self.masks = {d: self.s.interior[d].astype(np.float64) for d in self.dirs}
        self.windows = {}
        for d in self.dirs:
            wins = []
            for j in range(self.s.width):
                shift = j - half
                sl = list(self._center)
                sl[d] = slice(pad[d][0] + shift, pad[d][0] + shift + self.ext[d])
                wins.append(tuple(sl))
            self.windows[d] = wins
        self._exc_mv = _csr_matvec_factory(self.s.exc)

    def apply(self, u: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty(self.n_V)
        s = self.s
        D = u[: self.n_D].reshape(self.ext)
        self._padded[self._center] = D
        total = self._total
        total.fill(0.0)
        acc, tmp = self._acc, self._tmp
        for d in self.dirs:
            taps = s.taps[d]
            wins = self.windows[d]
            np.multiply(self._padded[wins[0]], taps[0], out=acc)
            for j in range(1, s.width):
                np.multiply(self._padded[wins[j]], taps[j], out=tmp)
                acc += tmp
            acc *= self.masks[d]
            total += acc
        out[: self.n_D] = total.ravel()
        out[self.n_D:] = 0.0
        if s.exc_rows.size:
            self._exc_mv(u, self._exc_out)
            out[s.exc_rows] += self._exc_out
        return out


def _csr_matvec_factory(m: sp.csr_matrix):
    """``y[:] = m @ x`` into a preallocated ``y``, using scipy's kernel when available."""
    try:
        from scipy.sparse import _sparsetools

        indptr, indices, data = m.indptr, m.indices, m.data
        nr, nc = m.shape

        def mv(x, y):
            y.fill(0.0)
            _sparsetools.csr_matvec(nr, nc, indptr, indices, data, x, y)

        probe_x = np.zeros(nc)
        probe_y = np.zeros(nr)
        mv(probe_x, probe_y)
        return mv
    except Exception:  # noqa: BLE001

        def mv(x, y):
            y[:] = m @ x

        return mv


def apply_structured(tables: Tables, u: np.ndarray, structure: Structure | None = None) -> np.ndarray:
    """Convenience wrapper around ``StructuredOperator``."""
    return StructuredOperator(tables, structure).apply(np.asarray(u, dtype=np.float64))


# ----------------------------------------------------------------------------
# Comparison with the exporter's own tables
# ----------------------------------------------------------------------------


def _exporter_exc(tables: Tables, n_V: int) -> tuple[sp.csr_matrix | None, str]:
    """Rebuild the exporter's exception CSR from whatever ``exc_*.npy`` files exist."""
    f = tables.exc_files
    if not f:
        return None, "no exc_*.npy files"
    indptr = f.get("rowptr", f.get("indptr"))
    indices = f.get("col", f.get("indices", f.get("cols")))
    data = f.get("val", f.get("data", f.get("vals", f.get("values"))))
    rows = f.get("rows", f.get("row"))
    if indptr is not None and indices is not None and data is not None:
        nr = indptr.size - 1
        m = sp.csr_matrix((data.astype(np.float64), indices.astype(np.int64), indptr.astype(np.int64)),
                          shape=(nr, n_V))
        if nr == n_V:
            return m, "full-height CSR (exc_rowptr/exc_col/exc_val)"
        if rows is not None and rows.size == nr:
            coo = m.tocoo()
            full = sp.csr_matrix((coo.data, (rows[coo.row], coo.col)), shape=(n_V, n_V))
            return full, "compact CSR (exc_rows/exc_rowptr/exc_col/exc_val)"
        return None, f"CSR with {nr} rows but no matching exc_rows array"
    if rows is not None and indices is not None and data is not None and rows.size == indices.size:
        full = sp.csr_matrix((data.astype(np.float64), (rows.astype(np.int64), indices.astype(np.int64))),
                             shape=(n_V, n_V))
        return full, "COO triplets (exc_row/exc_col/exc_val)"
    return None, f"unrecognised exc file set {sorted(f)}"


def compare_with_export(tables: Tables, structure: Structure | None = None) -> dict:
    """Compare the Python-derived role and exc with ``role.npy`` and ``exc_*.npy``.

    Returns a dict with ``role`` and ``exc`` sub-reports; each says whether the
    exporter's table was present, whether it matches, and the worst row.
    """
    s = structure if structure is not None else derive_structure(tables)
    L = tables.layout
    report: dict = {"role": {}, "exc": {}}

    if tables.role is None:
        report["role"] = {"present": False}
    else:
        diff = tables.role != s.role
        n = int(diff.sum())
        rep = {"present": True, "match": n == 0, "n_diff": n}
        if n:
            bad = int(np.argmax(diff))
            rep["worst"] = {
                "row": bad,
                "where": L.describe_row(bad),
                "python": int(s.role[bad]),
                "exporter": int(tables.role[bad]),
                "python_bits": _role_bits(int(s.role[bad])),
                "exporter_bits": _role_bits(int(tables.role[bad])),
            }
            per_bit = {}
            for b, name in ((0, "interior_x"), (1, "interior_y"), (2, "interior_z"),
                            (3, "fluid"), (4, "dirichlet")):
                per_bit[name] = int((((tables.role ^ s.role) >> b) & 1).sum())
            rep["diff_per_bit"] = per_bit
        report["role"] = rep

    m, how = _exporter_exc(tables, L.n_V)
    if m is None:
        report["exc"] = {"present": bool(tables.exc_files), "note": how}
    else:
        mine = s.exc_full(L.n_V)
        m.sum_duplicates()
        m.sort_indices()
        mine.sort_indices()
        rows_e = np.flatnonzero(np.diff(m.indptr))
        rows_p = s.exc_rows
        d = (m - mine)
        d.sum_duplicates()
        absd = np.abs(d.data) if d.nnz else np.zeros(0)
        rep = {
            "present": True,
            "layout": how,
            "rows_exporter": int(rows_e.size),
            "rows_python": int(rows_p.size),
            "rows_only_exporter": int(np.setdiff1d(rows_e, rows_p).size),
            "rows_only_python": int(np.setdiff1d(rows_p, rows_e).size),
            "nnz_exporter": int(m.nnz),
            "nnz_python": int(mine.nnz),
            "max_abs_diff": float(absd.max()) if absd.size else 0.0,
        }
        scale = float(np.abs(mine.data).max()) if mine.nnz else 1.0
        rep["match"] = (rep["rows_only_exporter"] == 0 and rep["rows_only_python"] == 0
                        and rep["max_abs_diff"] <= 1e-15 * scale)
        if absd.size and rep["max_abs_diff"] > 0:
            dc = d.tocoo()
            k = int(np.argmax(np.abs(dc.data)))
            rep["worst"] = {"row": int(dc.row[k]), "where": L.describe_row(int(dc.row[k])),
                            "col": int(dc.col[k]), "diff": float(dc.data[k])}
        report["exc"] = rep
    return report


def _role_bits(v: int) -> list[str]:
    names = []
    for b, name in ((0, "interior_x"), (1, "interior_y"), (2, "interior_z"),
                    (3, "fluid"), (4, "dirichlet")):
        if v & (1 << b):
            names.append(name)
    return names


def part_name(p: int) -> str:
    return PART_NAMES.get(int(p), str(p))
