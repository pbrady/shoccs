"""Characterise the 3D transverse rows whose x_c^2 residual is not psi (1 - psi)
although they have no neighbouring R column: ratio histogram, normal component,
whether the row's two interpolated taps sit on different grid lines in the
third coordinate, and two example rows.

usage: deviating_rows.py EXPORT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pyshoccs.oracle import transverse_oracle, describe_row_entries
from pyshoccs.tables import load_tables


def main(argv):
    tables = load_tables(Path(argv[0]))
    L = tables.layout
    n_D = L.n_D
    tr = transverse_oracle(tables, mmax=2).select(cls="R_transverse", k=2)
    rows = []
    for w in tr:
        row = int(w["row"])
        d = int(w["dir"])
        r = int(L.segment_of(row)) - 1
        if r == d:
            continue
        i = row - L.r_offset(r)
        c = tables.A[d]
        m = (c.row == row) & (c.col >= n_D) & (c.col != row)
        if np.any(m):
            continue
        psi = float(w["psi"])
        pred = psi * (1 - psi)
        ratio = float(w["residual"]) / pred if pred > 0 else np.nan
        nd = abs(tables.R[r].normal[i, d])
        # the third coordinate: do the D columns of this row span more than one plane?
        cols = c.col[(c.row == row) & (c.col < n_D)]
        third = [k for k in range(3) if k not in (r, d)][0]
        ext = L.extents
        idx = np.stack(np.unravel_index(cols, ext), axis=1)
        planes = np.unique(idx[:, third]).size
        rows.append((row, d, r, i, psi, ratio, nd, planes, int(w["coord"])))
    rows = np.array(rows, dtype=object)
    ratio = rows[:, 5].astype(float)
    print(f"{tables.case_name}: {rows.shape[0]} rows without a neighbouring R column")
    exact = np.isclose(ratio, 1.0, atol=1e-9)
    print(f"  ratio == 1: {int(exact.sum())}; flat normal among them: {int((rows[exact, 6].astype(float) <= 1e-12).sum())}")
    dev = ~exact
    print(f"  deviating: {int(dev.sum())}; flat normal among them: {int((rows[dev, 6].astype(float) <= 1e-12).sum())}")
    print(f"  deviating ratios: min {np.nanmin(ratio[dev]):.4f} median {np.nanmedian(ratio[dev]):.4f} max {np.nanmax(ratio[dev]):.4f}")
    print(f"  deviating rows whose coord == third axis (monomial across the planes): "
          f"{int(sum(1 for q in rows[dev] if q[8] not in (q[1], q[2])))} of {int(dev.sum())}")
    print(f"  deviating rows with D columns on more than one plane of the third coordinate: "
          f"{int((rows[dev, 7].astype(int) > 1).sum())}")
    shown = 0
    for q in rows[dev]:
        if shown >= 2:
            break
        print(f"  example: ratio {q[5]:.4f} psi {q[4]:.4f} normal[d] {q[6]:.3e} planes {q[7]} coord {q[8]}")
        print(describe_row_entries(tables, int(q[1]), int(q[0])))
        shown += 1


if __name__ == "__main__":
    main(sys.argv[1:])
