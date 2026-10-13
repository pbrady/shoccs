"""Print the raw entries of selected rows and the signed-zero audit of exc_val.

usage: inspect_rows.py EXPORT_DIR [d:row ...]
With no rows: one transverse R row per (segment, direction) pair, the k = 3
passing R rows of a 3D export, and the exc_val entries whose bit pattern differs
between the exporter and the Python derivation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pyshoccs.oracle import describe_row_entries, monomial_oracle
from pyshoccs.structure import derive_structure
from pyshoccs.tables import load_tables, SEGMENT_NAMES


def main(argv):
    tables = load_tables(Path(argv[0]))
    L = tables.layout
    if len(argv) > 1:
        for spec in argv[1:]:
            d, row = (int(v) for v in spec.split(":"))
            print(describe_row_entries(tables, d, row))
        return
    orc = monomial_oracle(tables, kmax=3, include_closure=False)
    t = orc.table
    k3 = t[t["k"] == 3]
    seen = set()
    for w in k3:
        r = int(L.segment_of(int(w["row"]))) - 1
        d = int(w["dir"])
        key = (r, d, int(w["nnz"]), bool(w["passed"]))
        if key in seen:
            continue
        seen.add(key)
        print(f"--- {SEGMENT_NAMES[r + 1]} row in A{d}, nnz {w['nnz']}, k=3 residual {w['residual']:.4e} "
              f"(res/h {w['residual'] / tables.h[d]:.4f}) passed={bool(w['passed'])}")
        print(describe_row_entries(tables, d, int(w["row"])))
    # along-ray residual / h as a function of psi
    print("--- along-ray k=3 residual/h vs psi")
    rows = []
    for w in k3:
        r = int(L.segment_of(int(w["row"]))) - 1
        if r != int(w["dir"]):
            continue
        i = int(w["row"]) - L.r_offset(r)
        rows.append((float(tables.R[r].psi[i]), float(w["residual"] / tables.h[r])))
    rows.sort()
    for psi, c in rows[:: max(1, len(rows) // 16)]:
        print(f"  psi {psi:.4f}  res/h {c:.4f}")
    # the object closure D rows: residual / h vs psi
    orc2 = monomial_oracle(tables, kmax=3, include_closure=True)
    t2 = orc2.table
    dc = t2[(t2["cls"] == "D_closure") & (t2["k"] == 3) & ~t2["passed"]]
    print(f"--- object closure D rows failing k=3: {dc.size}; residual/h vs psi_min")
    pairs = sorted((float(w["psi_min"]), float(w["residual"] / tables.h[int(w["dir"])])) for w in dc)
    for psi, c in pairs[:: max(1, len(pairs) // 16)]:
        print(f"  psi {psi:.4f}  res/h {c:.4f}")
    # signed zeros in exc_val
    s = derive_structure(tables)
    f = tables.exc_files
    ev = f["val"]
    pv = s.exc.data
    if ev.size == pv.size:
        diff = ev.view(np.int64) != pv.view(np.int64)
        print(f"--- exc_val bit-pattern differences: {int(diff.sum())}")
        for k in np.flatnonzero(diff)[:10]:
            print(f"  entry {k}: exporter {ev[k]!r} python {pv[k]!r}")
        ez = int(np.count_nonzero(ev == 0))
        print(f"    stored zeros in exporter exc_val: {ez}, in python: {int(np.count_nonzero(pv == 0))}")


if __name__ == "__main__":
    main(sys.argv[1:])
