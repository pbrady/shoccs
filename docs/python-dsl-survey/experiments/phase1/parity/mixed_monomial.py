"""Transverse monomial diagnostic for the R rows.

For direction d and a transverse coordinate c != d, push x_c^m (m = 1, 2, 3)
through A_d and look at the R rows. The exact second x_d-derivative of x_c^m is
0, so any residual measures what the cross-line interpolation of the R-row
formulation does to a function that varies across the lines. Also x_c x_d^2
(expected 2 x_c at the row) as the mixed case.

usage: mixed_monomial.py EXPORT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pyshoccs.tables import load_tables, SEGMENT_NAMES

EPS = np.finfo(np.float64).eps


def main(argv):
    tables = load_tables(Path(argv[0]))
    L = tables.layout
    loc = tables.locations
    n_D = L.n_D
    dims = list(L.active_dims)
    print(f"== {tables.case_name} h={tables.h.tolist()}")
    for d in dims:
        A = tables.to_scipy(dirs=(d,))
        absA = abs(A)
        R_rows = np.unique(tables.A[d].row[tables.A[d].row >= n_D])
        for c in dims:
            if c == d:
                continue
            xc = loc[:, c]
            xd = loc[:, d]
            for label, f, expected in (
                (f"x{c}", xc, np.zeros(R_rows.size)),
                (f"x{c}^2", xc ** 2, np.zeros(R_rows.size)),
                (f"x{c}^3", xc ** 3, np.zeros(R_rows.size)),
                (f"x{c}*x{d}^2", xc * xd ** 2, 2.0 * xc[R_rows]),
            ):
                y = (A @ f)[R_rows]
                tol = 16 * EPS * (absA @ np.abs(f))[R_rows]
                res = np.abs(y - expected)
                fail = res > tol
                # split by segment of the row and whether it has R columns of another ray
                segs = L.segment_of(R_rows)
                line = f"  A{d} on {label:12s}: R rows {R_rows.size:5d} fail {int(fail.sum()):5d} max res {res.max():.3e}"
                for s in sorted(set(segs.tolist())):
                    m = segs == s
                    line += f" | {SEGMENT_NAMES[s]} fail {int(fail[m].sum())}/{int(m.sum())} max {res[m].max():.3e}"
                print(line)
                if label == f"x{c}^2" and fail.any():
                    # compare the residual with psi (1 - psi) of the row's own record
                    rows = R_rows[fail]
                    segs_f = segs[fail]
                    psi = np.array([tables.R[int(s) - 1].psi[int(r) - L.r_offset(int(s) - 1)]
                                    for r, s in zip(rows, segs_f)])
                    pred = psi * (1 - psi)
                    ratio = res[fail] / np.where(pred > 0, pred, 1)
                    print(f"      residual / (psi (1 - psi)) over the failing rows: min {ratio.min():.4f} "
                          f"median {np.median(ratio):.4f} max {ratio.max():.4f}")
                    worst = np.argsort(-res[fail])[:3]
                    for w in worst:
                        r = int(rows[w])
                        s = int(segs_f[w]) - 1
                        i = r - L.r_offset(s)
                        print(f"      worst {L.describe_row(r)} psi {psi[w]:.4f} position {tables.R[s].position[i].tolist()} "
                              f"res {res[fail][w]:.4e} value {y[fail][w]:.4e}")


if __name__ == "__main__":
    main(sys.argv[1:])
