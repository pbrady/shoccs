"""Load a ``shoccs-export`` directory and expose the exact operator over V.

Everything follows ``docs/reference/pyshoccs-export.md``: the packed vector V is
``[D | Rx | Ry | Rz]`` with D in C order (z fastest), the R segments in the
generation order of ``object_geometry`` (never re-sorted here), and the operator
of each direction d is held as raw COO entries over V in the order the C++
matrices store them, duplicates not summed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

SEGMENT_NAMES = ("D", "Rx", "Ry", "Rz")
PART_NAMES = {0: "O_left", 1: "O_interior", 2: "O_right", 3: "B",
              5: "Bf_x", 6: "Br_x", 7: "Bf_y", 8: "Br_y", 9: "Bf_z", 10: "Br_z"}

ORACLE_VECTORS = (
    "u0_V", "rhs_u0_V", "rand_V", "lap_rand_V", "rhs_rand_V",
    "sol_t0_V", "src_t0_V", "sol_t1_V", "src_t1_V", "u1_V",
)


class ExportFormatError(RuntimeError):
    """An export file is missing or does not match the contract."""


def default_export_dir() -> Path | None:
    """The export directory the tests and tools fall back to.

    ``PYSHOCCS_EXPORT_DIR`` wins; otherwise ``.scratch/export/heat51`` under the
    repository root (four levels above this file). Returns None when neither
    exists.
    """
    env = os.environ.get("PYSHOCCS_EXPORT_DIR")
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    repo = Path(__file__).resolve().parents[3]
    p = repo / ".scratch" / "export" / "heat51"
    return p if p.is_dir() else None


# ----------------------------------------------------------------------------
# npy loading with contract checks
# ----------------------------------------------------------------------------


def load_npy(path: str | Path) -> np.ndarray:
    """Load one ``.npy`` file (no pickles) as a C-contiguous array."""
    path = Path(path)
    if not path.is_file():
        raise ExportFormatError(f"missing export file {path}")
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        raise ExportFormatError(f"cannot read {path}: {type(e).__name__}: {e}") from e
    return np.ascontiguousarray(arr)


def expect(arr: np.ndarray, name: str, dtype=None, ndim: int | None = None,
           shape: tuple | None = None) -> np.ndarray:
    """Check dtype and shape of a loaded array, naming the file in the error.

    ``shape`` entries of None are wildcards. The dtype check accepts any
    integer dtype for an integer expectation and any float dtype for a float
    one, because the contract fixes widths but a consumer only needs the kind;
    the exact dtype is still reported.
    """
    if dtype is not None:
        want = np.dtype(dtype)
        if arr.dtype.kind != want.kind:
            raise ExportFormatError(
                f"{name}: dtype {arr.dtype} does not match the contract ({want})")
    if ndim is not None and arr.ndim != ndim:
        raise ExportFormatError(
            f"{name}: ndim {arr.ndim} (shape {arr.shape}) does not match the contract ({ndim})")
    if shape is not None:
        if arr.ndim != len(shape):
            raise ExportFormatError(
                f"{name}: shape {arr.shape} does not match the contract {shape}")
        for got, want in zip(arr.shape, shape):
            if want is not None and got != want:
                raise ExportFormatError(
                    f"{name}: shape {arr.shape} does not match the contract {shape}")
    return arr


def _optional(path: Path, fallback: np.ndarray) -> np.ndarray:
    return load_npy(path) if path.is_file() else fallback


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Sizes and offsets of the packed vector V."""

    extents: tuple[int, int, int]
    n_D: int
    n_R: tuple[int, int, int]
    offsets: dict[str, int]
    n_V: int

    @property
    def strides(self) -> tuple[int, int, int]:
        nx, ny, nz = self.extents
        return (ny * nz, nz, 1)

    @property
    def active_dims(self) -> tuple[int, ...]:
        return tuple(d for d in range(3) if self.extents[d] > 1)

    def r_offset(self, d: int) -> int:
        return self.offsets[SEGMENT_NAMES[d + 1]]

    def segment_slice(self, name: str) -> slice:
        i = SEGMENT_NAMES.index(name)
        start = self.offsets[name]
        length = self.n_D if i == 0 else self.n_R[i - 1]
        return slice(start, start + length)

    @property
    def segment_slices(self) -> dict[str, slice]:
        return {name: self.segment_slice(name) for name in SEGMENT_NAMES}

    def segment_of(self, rows) -> np.ndarray:
        """Segment code per packed row: 0 D, 1 Rx, 2 Ry, 3 Rz."""
        rows = np.asarray(rows)
        bounds = np.array([self.offsets["Rx"], self.offsets["Ry"], self.offsets["Rz"]])
        return np.searchsorted(bounds, rows, side="right").astype(np.int64)

    def segment_name(self, row: int) -> str:
        return SEGMENT_NAMES[int(self.segment_of(row))]

    def local_index(self, row: int) -> int:
        """Index inside its segment (D index, or position in R(d))."""
        return int(row) - self.offsets[self.segment_name(row)]

    def describe_row(self, row: int) -> str:
        seg = self.segment_name(row)
        loc = self.local_index(row)
        if seg == "D":
            nx, ny, nz = self.extents
            i, rem = divmod(loc, ny * nz)
            j, k = divmod(rem, nz)
            return f"V[{row}] = D[{loc}] (i,j,k)=({i},{j},{k})"
        return f"V[{row}] = {seg}[{loc}]"


@dataclass
class RRecords:
    """The intersection records of one ray direction (spec Section 2.1)."""

    psi: np.ndarray
    position: np.ndarray
    normal: np.ndarray
    ray_outside: np.ndarray
    solid_coord: np.ndarray
    solid_ic: np.ndarray
    shape_id: np.ndarray
    dirichlet_idx: np.ndarray
    non_dirichlet_idx: np.ndarray

    @property
    def count(self) -> int:
        return int(self.psi.shape[0])


@dataclass
class Lines:
    """Line fragments of one direction: ``lines{d}``, ``lines{d}_psi``, ``lines{d}_skipped``."""

    lines: np.ndarray
    psi: np.ndarray
    skipped: np.ndarray

    @property
    def count(self) -> int:
        return int(self.lines.shape[0])


@dataclass
class Coo:
    """Raw COO entries of one direction's operator over V, in stored order."""

    row: np.ndarray
    col: np.ndarray
    val: np.ndarray
    part: np.ndarray
    line: np.ndarray

    @property
    def nnz(self) -> int:
        return int(self.row.shape[0])

    @staticmethod
    def empty() -> "Coo":
        z = np.zeros(0, dtype=np.int64)
        return Coo(z, z.copy(), np.zeros(0), np.zeros(0, dtype=np.uint8),
                   np.full(0, -1, dtype=np.int32))


@dataclass
class Tables:
    """Everything in one export directory, with the exact operator."""

    export_dir: Path
    meta: dict[str, Any]
    layout: Layout
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    R: list[RRecords]
    lines: list[Lines]
    fluid_idx: np.ndarray
    dirichlet_face_idx: np.ndarray
    taps: np.ndarray
    walls: dict[str, np.ndarray]
    A: list[Coo]
    N: list[Coo]
    role: np.ndarray | None
    exc_files: dict[str, np.ndarray]
    oracle: dict[str, np.ndarray]
    stats_u1: list[float] | None
    _csr_cache: dict = field(default_factory=dict, repr=False)

    # -- convenience accessors ------------------------------------------------

    @property
    def dims(self) -> int:
        return int(self.meta.get("dims", len(self.layout.active_dims)))

    @property
    def extents(self) -> tuple[int, int, int]:
        return self.layout.extents

    @property
    def h(self) -> np.ndarray:
        return np.asarray(self.meta["h"], dtype=np.float64)

    @property
    def diffusivity(self) -> float:
        return float(self.meta["system"]["diffusivity"])

    @property
    def t0(self) -> float:
        return float(self.meta.get("t0", 0.0))

    @property
    def dt(self) -> float:
        return float(self.meta["dt"])

    @property
    def grid_bcs(self) -> list[list[str]]:
        return self.meta["grid_bcs"]

    @property
    def object_bcs(self) -> list[str]:
        return self.meta["object_bcs"]

    @property
    def has_neumann(self) -> bool:
        return any("neumann" in [str(s).lower() for s in pair] for pair in self.grid_bcs)

    @property
    def n_V(self) -> int:
        return self.layout.n_V

    @property
    def n_D(self) -> int:
        return self.layout.n_D

    @property
    def case_name(self) -> str:
        return self.export_dir.name

    def r_packed(self, d: int, local: np.ndarray) -> np.ndarray:
        """Packed V indices of positions ``local`` in R(d)."""
        return self.layout.r_offset(d) + np.asarray(local, dtype=np.int64)

    @cached_property
    def locations(self) -> np.ndarray:
        """(n_V, 3) coordinates: grid point for D, intersection position for R."""
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        d = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        parts = [d] + [self.R[k].position.reshape(-1, 3) for k in range(3)]
        loc = np.concatenate(parts, axis=0)
        if loc.shape[0] != self.layout.n_V:
            raise ExportFormatError(
                f"locations: {loc.shape[0]} points but n_V = {self.layout.n_V}")
        return np.ascontiguousarray(loc)

    @cached_property
    def is_fluid_D(self) -> np.ndarray:
        m = np.zeros(self.layout.n_D, dtype=bool)
        m[self.fluid_idx] = True
        return m

    @cached_property
    def is_dirichlet_face_D(self) -> np.ndarray:
        m = np.zeros(self.layout.n_D, dtype=bool)
        m[self.dirichlet_face_idx] = True
        return m

    # -- the exact operator ---------------------------------------------------

    def coo_concat(self, dirs=(0, 1, 2)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.concatenate([self.A[d].row for d in dirs]) if dirs else np.zeros(0, np.int64)
        cols = np.concatenate([self.A[d].col for d in dirs]) if dirs else np.zeros(0, np.int64)
        vals = np.concatenate([self.A[d].val for d in dirs]) if dirs else np.zeros(0)
        return rows, cols, vals

    def to_scipy(self, dirs=(0, 1, 2)) -> sp.csr_matrix:
        """CSR over V of ``sum_{d in dirs} A_d``, duplicates summed, indices sorted."""
        key = tuple(dirs)
        if key not in self._csr_cache:
            n = self.layout.n_V
            rows, cols, vals = self.coo_concat(key)
            m = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
            m.sum_duplicates()
            m.sort_indices()
            self._csr_cache[key] = m
        return self._csr_cache[key]

    def exact_apply(self, u: np.ndarray, dirs=(0, 1, 2)) -> np.ndarray:
        """Sum the raw entries in stored order, direction by direction.

        This is the reference for sum-order questions: per direction the O
        entries come first, then B, then Bf/Br, as the C++ matrices store them.
        """
        u = np.asarray(u, dtype=np.float64)
        out = np.zeros(self.layout.n_V)
        for d in dirs:
            c = self.A[d]
            if c.nnz:
                np.add.at(out, c.row, c.val * u[c.col])
        return out

    def summary(self) -> str:
        L = self.layout
        lines = [f"export {self.export_dir}",
                 f"  extents {L.extents}, n_D {L.n_D}, n_R {L.n_R}, n_V {L.n_V}",
                 f"  grid_bcs {self.grid_bcs}, object_bcs {self.object_bcs}",
                 f"  scheme {self.meta.get('scheme')}",
                 f"  fluid {self.fluid_idx.size}, dirichlet faces {self.dirichlet_face_idx.size}"]
        for d in range(3):
            parts, counts = np.unique(self.A[d].part, return_counts=True)
            desc = ", ".join(f"{PART_NAMES.get(int(p), p)}={c}" for p, c in zip(parts, counts))
            lines.append(f"  A{d}: nnz {self.A[d].nnz} ({desc}); N{d}: nnz {self.N[d].nnz}")
        lines.append(f"  oracle vectors: {sorted(self.oracle)}")
        if self.role is not None:
            lines.append(f"  exporter role.npy present; exc files {sorted(self.exc_files)}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def _load_layout(meta: dict) -> Layout:
    ext = meta.get("extents")
    if ext is None:
        raise ExportFormatError("meta.json: missing 'extents'")
    ext = [int(e) for e in ext] + [1] * (3 - len(ext))
    n_D = int(meta.get("n_D", ext[0] * ext[1] * ext[2]))
    if n_D != ext[0] * ext[1] * ext[2]:
        raise ExportFormatError(f"meta.json: n_D {n_D} != prod(extents) {ext}")
    n_R = meta.get("n_R", [0, 0, 0])
    n_R = tuple(int(v) for v in n_R) + (0,) * (3 - len(n_R))
    off = meta.get("offsets")
    if off is None:
        off = {"D": 0, "Rx": n_D, "Ry": n_D + n_R[0], "Rz": n_D + n_R[0] + n_R[1]}
    off = {k: int(v) for k, v in off.items()}
    n_V = int(meta.get("n_V", n_D + sum(n_R)))
    want = {"D": 0, "Rx": n_D, "Ry": n_D + n_R[0], "Rz": n_D + n_R[0] + n_R[1]}
    if off != want or n_V != n_D + sum(n_R):
        raise ExportFormatError(
            f"meta.json: offsets {off} / n_V {n_V} inconsistent with n_D {n_D}, n_R {n_R}")
    return Layout(tuple(ext), n_D, n_R, off, n_V)


def _load_records(root: Path, d: int, n: int) -> RRecords:
    p = f"R{d}_"
    f64_empty = np.zeros(0)
    rec = RRecords(
        psi=expect(_optional(root / f"{p}psi.npy", f64_empty), f"{p}psi", np.float64, shape=(n,)),
        position=expect(_optional(root / f"{p}position.npy", np.zeros((0, 3))),
                        f"{p}position", np.float64, shape=(n, 3)),
        normal=expect(_optional(root / f"{p}normal.npy", np.zeros((0, 3))),
                      f"{p}normal", np.float64, shape=(n, 3)),
        ray_outside=expect(_optional(root / f"{p}ray_outside.npy", np.zeros(0, np.uint8)),
                           f"{p}ray_outside", np.uint8, shape=(n,)),
        solid_coord=expect(_optional(root / f"{p}solid_coord.npy", np.zeros((0, 3), np.int32)),
                           f"{p}solid_coord", np.int32, shape=(n, 3)),
        solid_ic=expect(_optional(root / f"{p}solid_ic.npy", np.zeros(0, np.int64)),
                        f"{p}solid_ic", np.int64, shape=(n,)),
        shape_id=expect(_optional(root / f"{p}shape_id.npy", np.zeros(0, np.int32)),
                        f"{p}shape_id", np.int32, shape=(n,)),
        dirichlet_idx=expect(_optional(root / f"{p}dirichlet_idx.npy", np.zeros(0, np.int32)),
                             f"{p}dirichlet_idx", np.int32, ndim=1).astype(np.int64),
        non_dirichlet_idx=expect(_optional(root / f"{p}non_dirichlet_idx.npy", np.zeros(0, np.int32)),
                                 f"{p}non_dirichlet_idx", np.int32, ndim=1).astype(np.int64),
    )
    both = np.concatenate([rec.dirichlet_idx, rec.non_dirichlet_idx])
    if n and (np.unique(both).size != n or both.min(initial=0) < 0 or both.max(initial=-1) >= n):
        raise ExportFormatError(
            f"R{d}: dirichlet_idx ({rec.dirichlet_idx.size}) and non_dirichlet_idx "
            f"({rec.non_dirichlet_idx.size}) do not partition 0..{n - 1}")
    return rec


def _load_lines(root: Path, d: int) -> Lines:
    lines = expect(_optional(root / f"lines{d}.npy", np.zeros((0, 7), np.int64)),
                   f"lines{d}", np.int64, shape=(None, 7))
    n = lines.shape[0]
    psi = expect(_optional(root / f"lines{d}_psi.npy", np.full((n, 2), np.nan)),
                 f"lines{d}_psi", np.float64, shape=(n, 2))
    skipped = expect(_optional(root / f"lines{d}_skipped.npy", np.zeros(n, np.uint8)),
                     f"lines{d}_skipped", np.uint8, shape=(n,))
    return Lines(lines, psi, skipped)


def _load_coo(root: Path, prefix: str, n_V: int, with_part: bool) -> Coo:
    rp = root / f"{prefix}_row.npy"
    if not rp.is_file():
        return Coo.empty()
    row = expect(load_npy(rp), f"{prefix}_row", np.int64, ndim=1).astype(np.int64)
    nnz = row.shape[0]
    col = expect(load_npy(root / f"{prefix}_col.npy"), f"{prefix}_col", np.int64, shape=(nnz,)).astype(np.int64)
    val = expect(load_npy(root / f"{prefix}_val.npy"), f"{prefix}_val", np.float64, shape=(nnz,))
    if with_part:
        part = expect(load_npy(root / f"{prefix}_part.npy"), f"{prefix}_part", np.uint8, shape=(nnz,))
        line = expect(_optional(root / f"{prefix}_line.npy", np.full(nnz, -1, np.int32)),
                      f"{prefix}_line", np.int32, shape=(nnz,))
    else:
        part = np.full(nnz, 255, np.uint8)
        line = np.full(nnz, -1, np.int32)
    if nnz:
        if row.min() < 0 or row.max() >= n_V:
            bad = int(np.argmax((row < 0) | (row >= n_V)))
            raise ExportFormatError(f"{prefix}: row {row[bad]} at entry {bad} outside V (n_V {n_V})")
        if with_part and (col.min() < 0 or col.max() >= n_V):
            bad = int(np.argmax((col < 0) | (col >= n_V)))
            raise ExportFormatError(f"{prefix}: col {col[bad]} at entry {bad} outside V (n_V {n_V})")
        if not np.all(np.isfinite(val)):
            bad = int(np.argmax(~np.isfinite(val)))
            raise ExportFormatError(f"{prefix}: non-finite coefficient at entry {bad} (row {row[bad]})")
    return Coo(row, col, val, part, line)


def _load_stats(path: Path) -> list[float] | None:
    if not path.is_file():
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "stats" in data:
            data = data["stats"]
        else:
            data = [v for v in data.values() if isinstance(v, (int, float))]
    vals = [float(v) for v in data]
    if len(vals) != 11:
        raise ExportFormatError(f"{path}: expected 11 numbers, found {len(vals)}")
    return vals


def load_tables(export_dir: str | Path) -> Tables:
    """Load an export directory. Raises ``ExportFormatError`` with the file name on a mismatch."""
    root = Path(export_dir)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise ExportFormatError(f"missing {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    layout = _load_layout(meta)
    nx, ny, nz = layout.extents

    x = expect(load_npy(root / "x.npy"), "x", np.float64, shape=(nx,))
    y = expect(load_npy(root / "y.npy"), "y", np.float64, shape=(ny,))
    z = expect(_optional(root / "z.npy", np.zeros(1)), "z", np.float64, shape=(nz,))

    R = [_load_records(root, d, layout.n_R[d]) for d in range(3)]
    lines = [_load_lines(root, d) for d in range(3)]

    fluid_idx = expect(load_npy(root / "fluid_idx.npy"), "fluid_idx", np.int32, ndim=1).astype(np.int64)
    if fluid_idx.size and (fluid_idx.min() < 0 or fluid_idx.max() >= layout.n_D):
        raise ExportFormatError("fluid_idx: index outside D")
    if fluid_idx.size > 1 and np.any(np.diff(fluid_idx) <= 0):
        raise ExportFormatError("fluid_idx: not strictly ascending")
    dir_idx = expect(_optional(root / "dirichlet_face_idx.npy", np.zeros(0, np.int32)),
                     "dirichlet_face_idx", np.int32, ndim=1).astype(np.int64)
    if dir_idx.size and (dir_idx.min() < 0 or dir_idx.max() >= layout.n_D):
        raise ExportFormatError("dirichlet_face_idx: index outside D")

    taps = expect(load_npy(root / "taps.npy"), "taps", np.float64, shape=(3, None))
    if taps.shape[1] % 2 != 1:
        raise ExportFormatError(f"taps: width {taps.shape[1]} is not odd (2p+1)")

    walls = {}
    for p in sorted(root.glob("wall_*.npy")):
        walls[p.stem] = load_npy(p)

    A = [_load_coo(root, f"A{d}", layout.n_V, with_part=True) for d in range(3)]
    for d in range(3):
        if layout.extents[d] == 1 and A[d].nnz:
            raise ExportFormatError(f"A{d}: {A[d].nnz} entries for a direction of extent 1")
    N = [_load_coo(root, f"N{d}", layout.n_V, with_part=False) for d in range(3)]

    role = None
    if (root / "role.npy").is_file():
        role = expect(load_npy(root / "role.npy"), "role", np.uint8, shape=(layout.n_D,))
    exc_files = {p.stem[len("exc_"):]: load_npy(p) for p in sorted(root.glob("exc_*.npy"))}

    oracle = {}
    for name in ORACLE_VECTORS:
        p = root / f"{name}.npy"
        if p.is_file():
            oracle[name] = expect(load_npy(p), name, np.float64, shape=(layout.n_V,))
    stats_u1 = _load_stats(root / "stats_u1.json")

    return Tables(root, meta, layout, x, y, z, R, lines, fluid_idx, dir_idx, taps, walls,
                  A, N, role, exc_files, oracle, stats_u1)
