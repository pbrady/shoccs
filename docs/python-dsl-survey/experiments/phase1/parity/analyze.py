"""Independent cross-checks of the phase-1 exports, beyond pyshoccs.parity.

For each export directory given on the command line:

1. row accounting: every row with an entry is a non-Dirichlet fluid D row or
   an R row, and the C++ eager Laplacian is exactly 0 everywhere else;
2. structural cross-check done bitwise: the exporter's role.npy and
   exc_rows/rowptr/col/val against the Python derivation (same rows, same
   sorted columns, identical values), plus the counts;
3. fp64 sum-order spread between exact_apply (stored order), to_scipy,
   the Python apply_structured and the C++ eager lap_rand_V;
4. the monomial oracle broken down by R segment, direction and k, with the
   residual divided by h, the worst rows (psi, position, ray), the k = 3
   passes, and the closure kind (wall or object, which of the two object
   closure rows) at which the D closure rows stop being exact.

Writes analysis_<case>.json next to this file and prints a summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pyshoccs.oracle import monomial_oracle, describe_row_entries
from pyshoccs.parity import classify_rows
from pyshoccs.structure import derive_structure, StructuredOperator
from pyshoccs.tables import load_tables, SEGMENT_NAMES

EPS = np.finfo(np.float64).eps


def rel_scale(tables, A, u):
    row1 = np.asarray(abs(A).sum(axis=1)).ravel()
    return row1 * float(np.abs(u).max())


def row_accounting(tables, classes):
    L = tables.layout
    n_D = L.n_D
    rows_with_entries = np.unique(np.concatenate([tables.A[d].row for d in range(3)]))
    code = classes.code
    names = np.array(["solid_D", "dirichlet_D", "fluid_D", "clamp_D", "R_dirichlet", "R"])
    per_class_rows = {n: int((code[rows_with_entries] == i).sum()) for i, n in enumerate(names)}
    lap = tables.oracle["lap_rand_V"]
    zero_classes = classes.mask("solid_D", "dirichlet_D", "R_dirichlet")
    out = {
        "rows_with_entries": int(rows_with_entries.size),
        "rows_with_entries_by_class": per_class_rows,
        "fluid_or_R_rows_without_entries": int((classes.mask("fluid_D", "clamp_D", "R").sum()
                                                 - per_class_rows["fluid_D"] - per_class_rows["clamp_D"]
                                                 - per_class_rows["R"])),
        "lap_rand_nonzero_on_zero_classes": int(np.count_nonzero(lap[zero_classes])),
        "entries_on_zero_classes": int(per_class_rows["solid_D"] + per_class_rows["dirichlet_D"]
                                       + per_class_rows["R_dirichlet"]),
        "dirichlet_face_list_len": int(tables.dirichlet_face_idx.size),
        "dirichlet_face_unique": int(np.unique(tables.dirichlet_face_idx).size),
        "solid_on_dirichlet_face": int((~tables.is_fluid_D & tables.is_dirichlet_face_D).sum()),
    }
    return out


def structural_bitwise(tables, structure):
    L = tables.layout
    f = tables.exc_files
    e_rows = f["rows"].astype(np.int64)
    e_ptr = f["rowptr"].astype(np.int64)
    e_col = f["col"].astype(np.int64)
    e_val = f["val"]
    p_rows = structure.exc_rows.astype(np.int64)
    p = structure.exc
    p.sort_indices()
    rep = {
        "role_bitwise_equal": bool(np.array_equal(tables.role, structure.role)),
        "exc_rows_equal": bool(np.array_equal(e_rows, p_rows)),
        "exc_rows_sorted_unique_exporter": bool(np.all(np.diff(e_rows) > 0)),
        "exc_rowptr_equal": bool(np.array_equal(e_ptr, p.indptr.astype(np.int64))),
        "exc_col_dtype": str(f["col"].dtype),
        "exc_rowptr_dtype": str(f["rowptr"].dtype),
    }
    # compare per row after sorting the exporter's columns (it writes them sorted already)
    cols_equal = True
    vals_bitwise = True
    max_rel_val = 0.0
    if rep["exc_rows_equal"] and rep["exc_rowptr_equal"]:
        order = np.lexsort((e_col, np.repeat(np.arange(e_rows.size), np.diff(e_ptr))))
        ec = e_col[order]
        ev = e_val[order]
        cols_equal = bool(np.array_equal(ec, p.indices.astype(np.int64)))
        if cols_equal:
            vals_bitwise = bool(np.array_equal(ev.view(np.int64), p.data.view(np.int64)))
            scale = np.maximum(np.abs(ev), np.abs(p.data))
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(scale > 0, np.abs(ev - p.data) / np.where(scale > 0, scale, 1), 0.0)
            max_rel_val = float(r.max()) if r.size else 0.0
        rep["exporter_cols_already_sorted"] = bool(np.array_equal(order, np.arange(order.size)))
    rep["exc_cols_equal"] = cols_equal
    rep["exc_vals_bitwise_equal"] = vals_bitwise
    rep["exc_vals_max_rel_diff"] = max_rel_val
    rep["counts"] = {
        "interior_rows_per_dir_python": structure.counts["interior_rows_per_dir"],
        "interior_rows_per_dir_exporter": [int(((tables.role >> d) & 1).sum()) for d in range(3)],
        "exc_rows_python": int(p_rows.size),
        "exc_rows_exporter": int(e_rows.size),
        "exc_nnz_python": int(p.nnz),
        "exc_nnz_exporter": int(e_col.size),
        "exc_rows_D": int((p_rows < L.n_D).sum()),
        "exc_rows_R": int((p_rows >= L.n_D).sum()),
        "exc_max_row_nnz": int(np.diff(p.indptr).max()),
        "taps": tables.taps.tolist(),
    }
    # interior count identity: interior rows * width == number of part-1 entries
    w = tables.taps.shape[1]
    rep["interior_entries_identity"] = [
        int((tables.A[d].part == 1).sum()) == structure.counts["interior_rows_per_dir"][d] * w
        for d in range(3)]
    return rep


def sum_order_spread(tables, structure):
    u = tables.oracle["rand_V"]
    A = tables.to_scipy()
    vecs = {
        "exact_apply": tables.exact_apply(u),
        "to_scipy": A @ u,
        "apply_structured": StructuredOperator(tables, structure).apply(u),
        "cpp_eager": tables.oracle["lap_rand_V"],
    }
    scale_row = rel_scale(tables, A, u)
    gmax = float(np.abs(vecs["cpp_eager"]).max())
    names = list(vecs)
    out = {"global_scale": gmax, "pairs": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = np.abs(vecs[a] - vecs[b])
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(scale_row > 0, d / np.where(scale_row > 0, scale_row, 1), np.where(d > 0, np.inf, 0))
            out["pairs"][f"{a} vs {b}"] = {
                "max_abs": float(d.max()),
                "max_rel_global": float(d.max() / gmax),
                "max_rel_row": float(r.max()),
                "n_rows_differ": int(np.count_nonzero(d)),
            }
    return out


def closure_kind(tables, d, row):
    """'wall' if the dense row belongs to a line end at a wall, else 'object:<offset>'.

    offset is the number of grid cells between the row and the object end of its
    line (1 for the row next to the R point, 2 for the next one).
    """
    c = tables.A[d]
    m = (c.row == row) & (c.part != 1) & (c.line >= 0)
    if not np.any(m):
        return "none"
    parts = np.unique(c.part[m])
    line_ix = int(c.line[m][0])
    ln = tables.lines[d].lines[line_ix]
    stride, start_ic, end_ic, start_R, _, end_R, _ = (int(v) for v in ln)
    kinds = []
    for part in parts:
        if part == 0:
            if start_R < 0:
                kinds.append("wall_L")
            else:
                off = (row - start_ic) // stride + 1
                kinds.append(f"object_L:{off}")
        elif part == 2:
            if end_R < 0:
                kinds.append("wall_R")
            else:
                off = (end_ic - row) // stride + 1
                kinds.append(f"object_R:{off}")
    return "+".join(kinds)


def oracle_breakdown(tables, h):
    L = tables.layout
    orc = monomial_oracle(tables, kmax=3, include_closure=True)
    t = orc.table
    out = {"by_segment_dir_k": {}, "worst_k3_R": [], "worst_k3_D": [], "k3_R_pass": [],
           "D_closure_first_failing_k": {}}
    for cls in ("R", "D_closure"):
        for seg in ("D", "Rx", "Ry", "Rz"):
            for d in range(3):
                for k in range(4):
                    m = (t["cls"] == cls) & (t["segment"] == seg) & (t["dir"] == d) & (t["k"] == k)
                    if not np.any(m):
                        continue
                    s = t[m]
                    hd = h[d] if h[d] > 0 else np.nan
                    key = f"{cls}/{seg}/dir{d}/k{k}"
                    ratio = s["residual"] / np.maximum(s["tol"], 1e-300)
                    out["by_segment_dir_k"][key] = {
                        "n": int(s.size), "n_fail": int((~s["passed"]).sum()),
                        "max_residual": float(s["residual"].max()),
                        "max_residual_over_h": float(s["residual"].max() / hd),
                        "median_residual_over_h": float(np.median(s["residual"]) / hd),
                        "max_ratio_to_tol": float(ratio.max()),
                        "along_ray": bool(seg != "D" and "xyz".index(seg[1]) == d),
                    }
    # worst k = 3 rows, R and D closure
    for cls, key in (("R", "worst_k3_R"), ("D_closure", "worst_k3_D")):
        m = (t["cls"] == cls) & (t["k"] == 3)
        s = t[m]
        order = np.argsort(-s["residual"])[:8]
        for w in s[order]:
            row = int(w["row"])
            d = int(w["dir"])
            rec = {"row": row, "where": L.describe_row(row), "dir": d, "residual": float(w["residual"]),
                   "residual_over_h": float(w["residual"] / h[d]), "tol": float(w["tol"]),
                   "nnz": int(w["nnz"]), "psi_min": float(w["psi_min"])}
            if row >= L.n_D:
                r = int(L.segment_of(row)) - 1
                i = row - L.r_offset(r)
                rec.update({"ray": SEGMENT_NAMES[r + 1], "record": i, "psi": float(tables.R[r].psi[i]),
                            "position": tables.R[r].position[i].tolist(),
                            "along_ray": bool(r == d)})
            else:
                rec["closure_kind"] = closure_kind(tables, d, row)
            out[key].append(rec)
    # which R rows pass at k = 3
    m = (t["cls"] == "R") & (t["k"] == 3) & t["passed"]
    for w in t[m][:50]:
        row = int(w["row"])
        r = int(L.segment_of(row)) - 1
        i = row - L.r_offset(r)
        out["k3_R_pass"].append({"row": row, "ray": SEGMENT_NAMES[r + 1], "record": i, "dir": int(w["dir"]),
                                 "psi": float(tables.R[r].psi[i]), "residual": float(w["residual"]),
                                 "nnz": int(w["nnz"]), "along_ray": bool(r == int(w["dir"]))})
    out["k3_R_pass_count"] = int(m.sum())
    # D closure: first failing k per closure kind
    m = t["cls"] == "D_closure"
    s = t[m]
    kinds = {}
    for w in s:
        key = (int(w["dir"]), int(w["row"]))
        kinds.setdefault(key, closure_kind(tables, key[0], key[1]))
    kind_arr = np.array([kinds[(int(w["dir"]), int(w["row"]))] for w in s])
    for kind in sorted(set(kind_arr)):
        mk = kind_arr == kind
        res = {}
        for k in range(4):
            mm = mk & (s["k"] == k)
            res[f"k{k}"] = {"n": int(mm.sum()), "n_fail": int((~s["passed"][mm]).sum()),
                            "max_residual": float(s["residual"][mm].max()) if mm.any() else 0.0}
        first = next((k for k in range(4) if res[f"k{k}"]["n_fail"] > 0), None)
        res["first_failing_k"] = first
        out["D_closure_first_failing_k"][kind] = res
    # residual / h distribution for k = 3 R rows (first-order check)
    m = (t["cls"] == "R") & (t["k"] == 3)
    s = t[m]
    rh = np.array([w["residual"] / h[int(w["dir"])] for w in s])
    out["k3_R_residual_over_h"] = {"min": float(rh.min()), "median": float(np.median(rh)),
                                  "p90": float(np.percentile(rh, 90)), "max": float(rh.max())}
    return out, orc


def main(argv):
    here = Path(__file__).resolve().parent
    for arg in argv:
        export = Path(arg)
        tables = load_tables(export)
        case = export.name
        h = tables.h
        classes = classify_rows(tables)
        structure = derive_structure(tables)
        rep = {"case": case, "h": h.tolist(), "n_V": tables.layout.n_V,
               "row_classes": classes.counts}
        rep["row_accounting"] = row_accounting(tables, classes)
        rep["structural_bitwise"] = structural_bitwise(tables, structure)
        rep["sum_order_spread"] = sum_order_spread(tables, structure)
        rep["oracle"], orc = oracle_breakdown(tables, h)
        # dump of the worst k = 3 R row and D closure row
        wr = rep["oracle"]["worst_k3_R"][0]
        wd = rep["oracle"]["worst_k3_D"][0]
        rep["oracle"]["worst_k3_R_entries"] = describe_row_entries(tables, wr["dir"], wr["row"])
        rep["oracle"]["worst_k3_D_entries"] = describe_row_entries(tables, wd["dir"], wd["row"])
        with open(here / f"analysis_{case}.json", "w") as f:
            json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"== {case} h={h.tolist()} n_V={tables.layout.n_V}")
        print("row accounting:", json.dumps(rep["row_accounting"]))
        sb = rep["structural_bitwise"]
        print("structural bitwise:", {k: v for k, v in sb.items() if k != "counts"})
        print("counts:", json.dumps(sb["counts"]))
        print("sum-order spread (global scale %.3e):" % rep["sum_order_spread"]["global_scale"])
        for k, v in rep["sum_order_spread"]["pairs"].items():
            print(f"  {k:36s} max_abs {v['max_abs']:.3e} rel_global {v['max_rel_global']:.3e} "
                  f"rel_row {v['max_rel_row']:.3e} rows_differ {v['n_rows_differ']}")
        print("oracle by segment/dir/k:")
        for k, v in rep["oracle"]["by_segment_dir_k"].items():
            print(f"  {k:28s} n {v['n']:6d} fail {v['n_fail']:6d} max_res {v['max_residual']:.3e} "
                  f"res/h {v['max_residual_over_h']:.3e} (median {v['median_residual_over_h']:.3e}) "
                  f"ratio {v['max_ratio_to_tol']:.3g} along_ray {v['along_ray']}")
        print("k3 R residual/h:", rep["oracle"]["k3_R_residual_over_h"])
        print("k3 R pass count:", rep["oracle"]["k3_R_pass_count"])
        for p in rep["oracle"]["k3_R_pass"][:10]:
            print("   pass:", p)
        print("worst k3 R rows:")
        for w in rep["oracle"]["worst_k3_R"][:5]:
            print("  ", w)
        print("worst k3 D rows:")
        for w in rep["oracle"]["worst_k3_D"][:5]:
            print("  ", w)
        print("D closure first failing k by kind:")
        for kind, v in rep["oracle"]["D_closure_first_failing_k"].items():
            print(f"  {kind:14s} first_failing_k {v['first_failing_k']} "
                  + " ".join(f"k{k}:{v[f'k{k}']['n_fail']}/{v[f'k{k}']['n']}" for k in range(4)))
        print(rep["oracle"]["worst_k3_R_entries"])
        print(rep["oracle"]["worst_k3_D_entries"])
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
