"""Gate (5) of docs/PYTHON_DSL_SURVEY.md Section 4.2, computed from an export.

Compact-table footprint (docs/reference/pyshoccs-export.md Section 4, as the
exporter writes it in Section 2.5) against the legacy per-direction footprint of
block metadata plus coefficients plus CSR values and columns, not counting the
legacy full-height row pointers. Inputs are the export's own arrays, so every
number here is computed, not measured.

compact = n_D bytes of role bits
        + 4 B per exception row (exc_rows, int32)
        + 8 B per row pointer (exc_rowptr, int64, n_exc + 1 entries)
        + 12 B per exception nonzero (8 B value, 4 B int32 column)
        + 8 B per tap

legacy, per direction d with extent > 1, kept lines L_d = lines{d} minus the
lines with lines{d}_skipped set:
        meta_d   = 13 ints x 4 B x L_d            (src/matrices/inner_block_meta.hpp)
        coeffs_d = 8 B x (part-0 entries + part-2 entries + 3 x L_d)
                   (every left and right dense coefficient, one 3-tap interior per line)
        CSR      = 16 B per nonzero of parts 3 and 5 to 10 (8 B value, 8 B int64 column:
                   B, Bf_x, Br_x, Bf_y, Br_y, Bf_z, Br_z)

Usage: python -I gate5_memory.py <export dir> [<export dir> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np


def footprint(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    n_D = meta["n_D"]
    role = np.load(d / "role.npy")
    exc_rows = np.load(d / "exc_rows.npy")
    exc_rowptr = np.load(d / "exc_rowptr.npy")
    exc_val = np.load(d / "exc_val.npy")
    taps = np.load(d / "taps.npy")
    assert role.shape == (n_D,) and role.dtype == np.uint8
    assert exc_rowptr.shape == (exc_rows.size + 1,)
    compact = {
        "role bytes": int(role.nbytes),
        "exc_rows (4 B each)": 4 * int(exc_rows.size),
        "exc_rowptr (8 B each)": 8 * int(exc_rowptr.size),
        "exc nonzeros (12 B each)": 12 * int(exc_val.size),
        "taps (8 B each)": 8 * int(taps.size),
    }
    compact_total = sum(compact.values())

    legacy = {}
    legacy_total = 0
    per_dir = []
    for dd in range(3):
        if meta["extents"][dd] == 1:
            continue
        lines = np.load(d / f"lines{dd}.npy")
        skipped = np.load(d / f"lines{dd}_skipped.npy")
        part = np.load(d / f"A{dd}_part.npy")
        kept = int(lines.shape[0] - int(skipped.sum()))
        n0 = int(np.count_nonzero(part == 0))
        n1 = int(np.count_nonzero(part == 1))
        n2 = int(np.count_nonzero(part == 2))
        csr = int(np.count_nonzero(np.isin(part, [3, 5, 6, 7, 8, 9, 10])))
        meta_b = 13 * 4 * kept
        coef_b = 8 * (n0 + n2 + 3 * kept)
        csr_b = 16 * csr
        per_dir.append(
            {
                "d": dd,
                "lines": int(lines.shape[0]),
                "skipped": int(skipped.sum()),
                "kept lines": kept,
                "part 0 (left dense)": n0,
                "part 1 (interior)": n1,
                "part 2 (right dense)": n2,
                "CSR entries (parts 3, 5..10)": csr,
                "nnz total": int(part.size),
                "meta_d bytes": meta_b,
                "coeffs_d bytes": coef_b,
                "CSR val+col bytes": csr_b,
            }
        )
        legacy_total += meta_b + coef_b + csr_b
    legacy["per direction"] = per_dir
    return {
        "export": str(d.name),
        "n_D": n_D,
        "n_V": meta["n_V"],
        "n_exc": int(exc_rows.size),
        "nnz_exc": int(exc_val.size),
        "compact": compact,
        "compact total bytes": compact_total,
        "legacy": legacy,
        "legacy total bytes": legacy_total,
        "ratio compact / legacy": compact_total / legacy_total,
    }


def main(argv):
    for arg in argv:
        r = footprint(arg)
        print(f"## {r['export']}: n_D {r['n_D']}, n_V {r['n_V']}, exception rows {r['n_exc']}, exception nnz {r['nnz_exc']}")
        print()
        print("| compact part | bytes |")
        print("|---|---|")
        for k, v in r["compact"].items():
            print(f"| {k} | {v:,} |")
        print(f"| total | {r['compact total bytes']:,} |")
        print()
        print("| direction | lines | skipped | kept lines | part 0 | part 1 | part 2 | CSR entries | nnz | meta_d B | coeffs_d B | CSR val+col B |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in r["legacy"]["per direction"]:
            print(
                f"| {p['d']} | {p['lines']:,} | {p['skipped']:,} | {p['kept lines']:,} | {p['part 0 (left dense)']:,} | "
                f"{p['part 1 (interior)']:,} | {p['part 2 (right dense)']:,} | {p['CSR entries (parts 3, 5..10)']:,} | "
                f"{p['nnz total']:,} | {p['meta_d bytes']:,} | {p['coeffs_d bytes']:,} | {p['CSR val+col bytes']:,} |"
            )
        print(f"| legacy total | | | | | | | | | | | {r['legacy total bytes']:,} |")
        print()
        print(
            f"compact {r['compact total bytes']:,} B ({r['compact total bytes'] / 1e6:.3f} MB), "
            f"legacy {r['legacy total bytes']:,} B ({r['legacy total bytes'] / 1e6:.3f} MB), "
            f"ratio {r['ratio compact / legacy']:.4f}"
        )
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
