"""Verify two row-level claims about the transverse R rows.

1. A transverse R row passes the axial k = 3 test exactly when its record's
   normal has no component in the derivative direction (the interior stencil
   branch of interp_deriv_coefficients).
2. The x_c^2 residual equals psi (1 - psi) exactly when the row has no R column
   of another record (interior interpolant); the rows with a neighbouring R
   column (wall interpolant) are the ones that deviate.

usage: verify_claims.py EXPORT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pyshoccs.oracle import monomial_oracle, transverse_oracle
from pyshoccs.tables import load_tables


def main(argv):
    tables = load_tables(Path(argv[0]))
    L = tables.layout
    n_D = L.n_D

    def record(row):
        r = int(L.segment_of(row)) - 1
        return r, row - L.r_offset(r)

    def has_other_R_column(d, row):
        c = tables.A[d]
        m = (c.row == row) & (c.col >= n_D) & (c.col != row)
        return bool(np.any(m))

    ax = monomial_oracle(tables, kmax=3, include_closure=False).select(cls="R", k=3)
    agree = 0
    total = 0
    for w in ax:
        r, i = record(int(w["row"]))
        d = int(w["dir"])
        if r == d:
            continue
        total += 1
        flat_normal = abs(tables.R[r].normal[i, d]) <= 1e-12
        if flat_normal == bool(w["passed"]):
            agree += 1
    print(f"claim 1 ({tables.case_name}): transverse k=3 pass <=> normal[d] == 0 on {agree}/{total} rows")

    tr = transverse_oracle(tables, mmax=2).select(cls="R_transverse", k=2)
    interior_exact = 0
    interior_total = 0
    wall_total = 0
    wall_deviating = 0
    for w in tr:
        row = int(w["row"])
        r, i = record(row)
        d = int(w["dir"])
        if r == d:
            continue
        pred = float(w["psi"]) * (1 - float(w["psi"]))
        dev = abs(float(w["residual"]) - pred)
        if has_other_R_column(d, row):
            wall_total += 1
            wall_deviating += dev > 1e-9
        else:
            interior_total += 1
            interior_exact += dev <= 1e-9
    print(f"claim 2 ({tables.case_name}): residual == psi(1-psi) on {interior_exact}/{interior_total} "
          f"interior-interpolant rows; {wall_deviating}/{wall_total} wall-interpolant rows deviate")


if __name__ == "__main__":
    main(sys.argv[1:])
