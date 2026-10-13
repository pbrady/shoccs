"""The five parity checks of spec Section 3 (phase-1 gate 1).

Row classes over V, derived from the export alone:

- ``solid_D``: D points not in ``fluid_idx``;
- ``dirichlet_D``: D points in ``dirichlet_face_idx``;
- ``clamp_D``: fluid D rows that are closure rows (dense left or right rows, or
  rows with a B entry) of a line whose object end has psi outside
  ``[1e-4, 1 - 1e-4]``;
- ``fluid_D``: the remaining fluid D rows ("away from the clamps");
- ``R_dirichlet``: R points of a Dirichlet shape (no rows);
- ``R``: the other R points.

Each check returns a ``CheckResult`` naming the worst row, its class and the
per-class maxima, so a failure can be traced to one row of one segment.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .heat import HeatSystem, STATS_NAMES
from .oracle import monomial_oracle, transverse_oracle, OracleResult
from .structure import Structure, derive_structure, StructuredOperator, compare_with_export
from .tables import Tables, load_tables, SEGMENT_NAMES

EPS = np.finfo(np.float64).eps
PSI_CLAMP = (1e-4, 1.0 - 1e-4)
CLASS_NAMES = ("solid_D", "dirichlet_D", "fluid_D", "clamp_D", "R_dirichlet", "R")


@dataclass
class RowClasses:
    code: np.ndarray
    clamp_h: np.ndarray
    counts: dict

    def name(self, row: int) -> str:
        return CLASS_NAMES[int(self.code[row])]

    def mask(self, *names: str) -> np.ndarray:
        m = np.zeros(self.code.size, dtype=bool)
        for n in names:
            m |= self.code == CLASS_NAMES.index(n)
        return m


def classify_rows(tables: Tables) -> RowClasses:
    L = tables.layout
    n_D, n_V = L.n_D, L.n_V
    code = np.full(n_V, CLASS_NAMES.index("fluid_D"), dtype=np.int8)
    code[:n_D][~tables.is_fluid_D] = CLASS_NAMES.index("solid_D")
    code[:n_D][tables.is_dirichlet_face_D] = CLASS_NAMES.index("dirichlet_D")
    clamp_h = np.zeros(n_V)
    h = tables.h
    lo, hi = PSI_CLAMP
    for d in range(3):
        c = tables.A[d]
        if c.nnz == 0:
            continue
        flagged = np.zeros(c.nnz, dtype=bool)
        # dense rows of a line whose object end is clamped
        has_line = c.line >= 0
        for part, side in ((0, 0), (2, 1)):
            m = has_line & (c.part == part)
            if np.any(m):
                psi = tables.lines[d].psi[c.line[m], side]
                bad = np.isfinite(psi) & ((psi < lo) | (psi > hi))
                flagged[np.flatnonzero(m)[bad]] = True
        # rows with a B entry whose R column is clamped
        isB = (c.part == 3) & (c.col >= n_D) & (c.row < n_D)
        if np.any(isB):
            col = c.col[isB]
            seg = L.segment_of(col) - 1
            psi = np.empty(col.size)
            for r in range(3):
                mm = seg == r
                if np.any(mm):
                    psi[mm] = tables.R[r].psi[col[mm] - L.r_offset(r)]
            bad = (psi < lo) | (psi > hi)
            flagged[np.flatnonzero(isB)[bad]] = True
        rows = np.unique(c.row[flagged])
        rows = rows[(rows < n_D) & (code[rows] == CLASS_NAMES.index("fluid_D"))]
        code[rows] = CLASS_NAMES.index("clamp_D")
        clamp_h[rows] = np.maximum(clamp_h[rows], h[d])
    for d in range(3):
        code[tables.r_packed(d, tables.R[d].non_dirichlet_idx)] = CLASS_NAMES.index("R")
        code[tables.r_packed(d, tables.R[d].dirichlet_idx)] = CLASS_NAMES.index("R_dirichlet")
    counts = {n: int((code == i).sum()) for i, n in enumerate(CLASS_NAMES)}
    return RowClasses(code, clamp_h, counts)


@dataclass
class CheckResult:
    name: str
    passed: bool
    n_rows: int = 0
    n_fail: int = 0
    worst_row: int = -1
    worst_class: str = ""
    worst_where: str = ""
    worst_diff: float = 0.0
    worst_tol: float = 0.0
    max_abs: float = 0.0
    max_rel: float = 0.0
    scale: float = 0.0
    per_class: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    seconds: float = 0.0

    def message(self) -> str:
        s = (f"{self.name}: {'pass' if self.passed else 'FAIL'}; {self.n_fail}/{self.n_rows} rows over "
             f"tolerance; max |diff| {self.max_abs:.3e} ({self.max_rel:.3e} of scale {self.scale:.3e}); "
             f"worst {self.worst_where} class {self.worst_class}: |diff| {self.worst_diff:.3e} "
             f"vs tol {self.worst_tol:.3e}")
        if self.notes:
            s += "; " + "; ".join(self.notes)
        return s

    def to_dict(self) -> dict:
        return asdict(self)


def compare_rows(name: str, got: np.ndarray, ref: np.ndarray, tol: np.ndarray,
                 classes: RowClasses, layout, rows: np.ndarray | None = None) -> CheckResult:
    """Row-wise ``|got - ref| <= tol`` with the worst row by ratio and per-class maxima."""
    if rows is None:
        rows = np.arange(got.size)
    g, r, t = got[rows], ref[rows], tol[rows]
    diff = np.abs(g - r)
    bad = ~np.isfinite(g)
    diff[bad] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(t > 0, diff / np.where(t > 0, t, 1.0), np.where(diff > 0, np.inf, 0.0))
    fail = diff > t
    scale = float(np.abs(r).max()) if r.size else 0.0
    res = CheckResult(name=name, passed=not bool(fail.any()), n_rows=int(rows.size),
                      n_fail=int(fail.sum()), scale=scale)
    if rows.size:
        w = int(np.argmax(ratio))
        res.worst_row = int(rows[w])
        res.worst_class = classes.name(res.worst_row)
        res.worst_where = layout.describe_row(res.worst_row)
        res.worst_diff = float(diff[w])
        res.worst_tol = float(t[w])
        res.max_abs = float(diff.max())
        res.max_rel = float(res.max_abs / scale) if scale > 0 else res.max_abs
        codes = classes.code[rows]
        for i, cname in enumerate(CLASS_NAMES):
            m = codes == i
            if not np.any(m):
                continue
            k = int(np.argmax(np.where(m, ratio, -1.0)))
            res.per_class[cname] = {
                "n": int(m.sum()), "n_fail": int((fail & m).sum()),
                "max_abs": float(diff[m].max()),
                "max_ratio": float(ratio[m].max()),
                "worst_row": int(rows[k]), "worst_where": layout.describe_row(int(rows[k])),
            }
    if bad.any():
        res.notes.append(f"{int(bad.sum())} non-finite values in the Python result")
    return res


# ----------------------------------------------------------------------------
# Tolerances
# ----------------------------------------------------------------------------


def operator_tolerance(tables: Tables, A, classes: RowClasses, max_u: float) -> np.ndarray:
    """Per-row tolerance of check 1 for ``A u`` with ``|u| <= max_u``.

    Fluid rows away from the clamps and R rows: ``1e-13 |row|_1 max|u|``.
    Clamp-adjacent rows: the larger of that and ``64 eps max|coef| h``. Solid,
    Dirichlet and Dirichlet-R rows: 0 (the C++ never writes them).
    """
    absA = abs(A)
    row1 = np.asarray(absA.sum(axis=1)).ravel()
    rowmax = absA.max(axis=1).toarray().ravel()
    tol = 1e-13 * row1 * max_u
    clamp = classes.mask("clamp_D")
    tol[clamp] = np.maximum(tol[clamp], 64 * EPS * rowmax[clamp] * classes.clamp_h[clamp])
    tol[classes.mask("solid_D", "dirichlet_D", "R_dirichlet")] = 0.0
    return tol


# ----------------------------------------------------------------------------
# The checks
# ----------------------------------------------------------------------------


def _need(tables: Tables, *names: str) -> None:
    missing = [n for n in names if n not in tables.oracle]
    if missing:
        raise FileNotFoundError(f"export {tables.export_dir} lacks {missing}")


def check_operator(tables: Tables, classes: RowClasses | None = None) -> CheckResult:
    """Check 1: ``to_scipy() @ rand_V == lap_rand_V`` row by row."""
    t0 = time.perf_counter()
    _need(tables, "rand_V", "lap_rand_V")
    classes = classes or classify_rows(tables)
    A = tables.to_scipy()
    u = tables.oracle["rand_V"]
    got = A @ u
    tol = operator_tolerance(tables, A, classes, float(np.abs(u).max()))
    res = compare_rows("1 operator identity (A rand == lap_rand)", got, tables.oracle["lap_rand_V"],
                       tol, classes, tables.layout)
    # the gate is the D rows; R rows are reported but judged by check 3
    gate = classes.mask("solid_D", "dirichlet_D", "fluid_D", "clamp_D")
    r_fail = int((np.abs(got - tables.oracle["lap_rand_V"]) > tol)[~gate].sum())
    d_fail = res.n_fail - r_fail
    res.passed = d_fail == 0
    if r_fail:
        res.notes.append(f"{r_fail} R rows differ from the C++ apply (export fidelity, not gated here)")
    res.seconds = time.perf_counter() - t0
    return res


def check_rhs(tables: Tables, system: HeatSystem, classes: RowClasses | None = None) -> list[CheckResult]:
    """Check 2: the Python RHS on ``rand_V`` and ``u0_V`` at t0 against the C++ graph."""
    classes = classes or classify_rows(tables)
    A = tables.to_scipy()
    out = []
    for uname, rname in (("rand_V", "rhs_rand_V"), ("u0_V", "rhs_u0_V")):
        t0 = time.perf_counter()
        _need(tables, uname, rname)
        u = tables.oracle[uname]
        got = system.rhs(u, system.t0)
        src = np.zeros(tables.layout.n_V)
        src[system.src_idx] = system.source_selected(system.t0).copy()
        tol = system.k * operator_tolerance(tables, A, classes, float(np.abs(u).max()))
        tol += 1e-13 * np.abs(src)
        tol[classes.mask("solid_D", "dirichlet_D", "R_dirichlet")] = 0.0
        res = compare_rows(f"2 rhs identity (rhs({uname}) == {rname})", got, tables.oracle[rname],
                           tol, classes, tables.layout)
        res.seconds = time.perf_counter() - t0
        out.append(res)
    return out


def check_oracle(tables: Tables, classes: RowClasses | None = None) -> tuple[CheckResult, OracleResult]:
    """Check 3: the monomial oracle; gate on R rows for k <= 2.

    The k = 3 rows, the fluid D closure rows and the transverse monomials
    (``x_c^m``, c != d, on the R rows) are reported in the notes and the
    summary, not gated.
    """
    t0 = time.perf_counter()
    classes = classes or classify_rows(tables)
    orc = monomial_oracle(tables, kmax=3, include_closure=True)
    orc = orc.combined(transverse_oracle(tables, mmax=2))
    gate = orc.select(cls="R", kmax=2)
    res = CheckResult(name="3 monomial oracle (R rows, k <= 2)", passed=orc.gate_passed(),
                      n_rows=int(gate.size), n_fail=int((~gate["passed"]).sum()))
    w = orc.worst(cls="R", kmax=2)
    if w is not None:
        res.worst_row = int(w["row"])
        res.worst_class = classes.name(res.worst_row)
        res.worst_where = f"{tables.layout.describe_row(res.worst_row)} dir {int(w['dir'])} k={int(w['k'])}"
        res.worst_diff = float(w["residual"])
        res.worst_tol = float(w["tol"])
        res.max_abs = float(gate["residual"].max())
        res.scale = float(np.abs(gate["expected"]).max())
        res.max_rel = res.max_abs / res.scale if res.scale > 0 else res.max_abs
    res.per_class = orc.summary()
    k3 = orc.select(cls="R", k=3)
    if k3.size:
        res.notes.append(f"k=3 (not gated): {int((~k3['passed']).sum())}/{k3.size} R rows over tolerance")
    dc = orc.select(cls="D_closure", kmax=2)
    if dc.size:
        res.notes.append(f"D closure diagnostic k<=2: {int((~dc['passed']).sum())}/{dc.size} rows over tolerance")
    for m in (1, 2):
        tr = orc.select(cls="R_transverse", k=m)
        if tr.size == 0:
            continue
        w = tr[int(np.argmax(tr["residual"]))]
        res.notes.append(
            f"transverse x_c^{m} (not gated): {int((~tr['passed']).sum())}/{tr.size} R rows over tolerance, "
            f"max residual {float(w['residual']):.3e} at {tables.layout.describe_row(int(w['row']))} "
            f"dir {int(w['dir'])} coord {int(w['coord'])} psi {float(w['psi']):.4f}")
    res.seconds = time.perf_counter() - t0
    return res, orc


def check_step(tables: Tables, system: HeatSystem, classes: RowClasses | None = None) -> list[CheckResult]:
    """Check 4: one RK4 step from ``u0_V`` against ``u1_V`` and ``stats_u1.json``."""
    t0 = time.perf_counter()
    _need(tables, "u0_V", "u1_V")
    classes = classes or classify_rows(tables)
    u0 = tables.oracle["u0_V"]
    u1_ref = tables.oracle["u1_V"]
    u1 = system.rk4_step(u0, system.t0, system.dt)
    tol = np.full(u1.size, 1e-12 * float(np.abs(u1_ref).max()))
    res = compare_rows("4 step identity (rk4(u0) == u1)", u1, u1_ref, tol, classes, tables.layout)
    res.seconds = time.perf_counter() - t0
    out = [res]

    t1 = time.perf_counter()
    st = CheckResult(name="4b stats identity (stats(u1) == stats_u1.json)", passed=True)
    if tables.stats_u1 is None:
        st.passed = True
        st.notes.append("stats_u1.json absent, skipped")
    else:
        mine = system.stats(u1, system.t0 + system.dt)
        ref = tables.stats_u1
        worst = 0.0
        details = {}
        for i, name in enumerate(STATS_NAMES):
            a, b = mine[i], ref[i]
            is_index = name.endswith("_ic")
            if is_index:
                ok = a == b
                if not ok:
                    # a tie between two locations is not a failure
                    ok = _tied_index(system, u1, name, int(a), int(b))
                    if ok:
                        st.notes.append(f"{name}: Python {int(a)} vs C++ {int(b)} (tied error values)")
                diff = abs(a - b)
            else:
                diff = abs(a - b)
                ok = diff <= 1e-12 * max(abs(b), 1e-300) or diff <= 1e-14
                worst = max(worst, diff / max(abs(b), 1e-300))
            details[name] = {"python": a, "cpp": b, "ok": bool(ok)}
            if not ok:
                st.passed = False
                st.n_fail += 1
                if diff >= st.worst_diff:
                    st.worst_diff = float(diff)
                    st.worst_where = name
        st.n_rows = len(STATS_NAMES)
        st.max_rel = worst
        st.per_class = details
    st.seconds = time.perf_counter() - t1
    out.append(st)
    return out


def _tied_index(system: HeatSystem, u1: np.ndarray, name: str, a: int, b: int) -> bool:
    t = system.t0 + system.dt
    sol = system.solution(t)
    if name == "domain_ic":
        rows = (a, b)
    else:
        d = "xyz".index(name[1])
        off = system.L.r_offset(d)
        rows = (off + a, off + b)
    ea, eb = (abs(u1[r] - sol[r]) for r in rows)
    return abs(ea - eb) <= 1e-12 * max(ea, eb, 1e-300)


def check_structure(tables: Tables, structure: Structure | None = None,
                    classes: RowClasses | None = None) -> CheckResult:
    """Check 5: the structured form equals ``to_scipy()`` on ``rand_V``."""
    t0 = time.perf_counter()
    _need(tables, "rand_V")
    classes = classes or classify_rows(tables)
    structure = structure or derive_structure(tables)
    A = tables.to_scipy()
    u = tables.oracle["rand_V"]
    ref = A @ u
    got = StructuredOperator(tables, structure).apply(u)
    row1 = np.asarray(abs(A).sum(axis=1)).ravel()
    tol = 1e-14 * row1 * float(np.abs(u).max())
    res = compare_rows("5 structural identity (structured rand == A rand)", got, ref, tol,
                       classes, tables.layout)
    res.seconds = time.perf_counter() - t0
    return res


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def run_all(export_dir: str | Path, json_path: str | Path | None = None,
            backend: str = "scipy", verbose: bool = True) -> dict:
    """Run the five checks, print a table and write JSON. Returns the report dict."""
    tables = load_tables(export_dir)
    classes = classify_rows(tables)
    structure = derive_structure(tables)
    system = HeatSystem(tables, backend=backend, structure=structure)
    results: list[CheckResult] = []
    errors: dict[str, str] = {}

    def run(fn, *a):
        try:
            r = fn(*a)
        except Exception as e:  # noqa: BLE001
            errors[fn.__name__] = f"{type(e).__name__}: {e}"
            results.append(CheckResult(name=fn.__name__, passed=False, notes=[errors[fn.__name__]]))
            return None
        if isinstance(r, tuple):
            results.append(r[0])
            return r[1]
        if isinstance(r, list):
            results.extend(r)
        else:
            results.append(r)
        return r

    run(check_operator, tables, classes)
    run(check_rhs, tables, system, classes)
    orc = run(check_oracle, tables, classes)
    run(check_step, tables, system, classes)
    run(check_structure, tables, structure, classes)
    xcheck = compare_with_export(tables, structure)

    report = {
        "export_dir": str(Path(export_dir).resolve()),
        "case": tables.case_name,
        "backend": backend,
        "extents": list(tables.layout.extents),
        "n_V": tables.layout.n_V,
        "row_classes": classes.counts,
        "structure_counts": structure.counts,
        "all_passed": all(r.passed for r in results),
        "checks": [r.to_dict() for r in results],
        "oracle_summary": orc.summary() if orc is not None else None,
        "exporter_cross_check": xcheck,
        "errors": errors,
    }
    if verbose:
        print(tables.summary())
        print(f"row classes: {classes.counts}")
        print(format_table(results))
        if orc is not None:
            print("oracle by class and k:")
            print(orc.format())
        print(f"exporter cross-check: role {xcheck['role']}, exc {xcheck['exc']}")
        print("ALL PASSED" if report["all_passed"] else "SOME CHECKS FAILED")
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=1, default=_json_default)
        if verbose:
            print(f"wrote {json_path}")
    return report


def format_table(results: list[CheckResult]) -> str:
    lines = [f"{'check':58s} {'result':6s} {'fail/rows':>12s} {'max abs':>10s} {'max rel':>10s}  worst row (class)"]
    for r in results:
        lines.append(f"{r.name[:58]:58s} {'pass' if r.passed else 'FAIL':6s} {r.n_fail:>5d}/{r.n_rows:<6d} "
                     f"{r.max_abs:10.3e} {r.max_rel:10.3e}  {r.worst_where} ({r.worst_class})")
        for n in r.notes:
            lines.append(f"    note: {n}")
    return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main(argv=None) -> int:
    import argparse

    from .tables import default_export_dir

    ap = argparse.ArgumentParser(description="pyshoccs parity checks against a shoccs-export directory")
    ap.add_argument("export_dir", nargs="?", default=None)
    ap.add_argument("--json", default=None, help="write the report here (default parity_<case>.json)")
    ap.add_argument("--backend", default="scipy", choices=("scipy", "structured"))
    args = ap.parse_args(argv)
    export = Path(args.export_dir) if args.export_dir else default_export_dir()
    if export is None or not Path(export).is_dir():
        print("no export directory (pass one, or set PYSHOCCS_EXPORT_DIR)", file=sys.stderr)
        return 2
    json_path = args.json or f"parity_{Path(export).name}.json"
    report = run_all(export, json_path, backend=args.backend)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
