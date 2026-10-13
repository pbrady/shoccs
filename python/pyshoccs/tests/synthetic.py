"""A tiny synthetic export directory in the format of ``docs/reference/pyshoccs-export.md``.

The case is a 7 x 5 grid on [0, 1]^2 with Dirichlet walls and a vertical slab
of solid points at i >= 5. Every x-line j = 1..3 ends at an intersection record
Rx[j-1] at psi = 0.4; the records of shapes 0 (j = 1, 2) are floating, the one
of shape 1 (j = 3) is Dirichlet. Rows are E2 three-point second differences;
the closure rows next to the slab and the Rx rows use the three-point formula
through (x_3, x_4, x_R), exact for quadratics, so the monomial oracle passes
for k <= 2 and fails for k = 3 as a real one-sided closure would.

The oracle vectors are produced by pyshoccs itself, so a parity run on this
directory tests the plumbing of every check, not the C++.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NX, NY, NZ = 7, 5, 1
PSI = 0.4
K = 1.0 / 30.0
MMS = {"type": "gaussian", "center": [[0.5, 0.5]], "variance": [[0.3, 0.4]],
       "amplitude": [2.0], "frequency": [0.5]}


def three_point_second_derivative(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Coefficients of f''(a) from f(a), f(b), f(c); exact for quadratics."""
    return (2.0 / ((a - b) * (a - c)), 2.0 / ((b - a) * (b - c)), 2.0 / ((c - a) * (c - b)))


def build_arrays() -> dict:
    nx, ny, nz = NX, NY, NZ
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    z = np.zeros(1)
    hx, hy = x[1] - x[0], y[1] - y[0]
    n_D = nx * ny * nz
    sx, sy = ny * nz, nz

    def ic(i, j, k=0):
        return i * sx + j * sy + k

    n_Rx = 3
    off_Rx = n_D
    n_V = n_D + n_Rx
    x_R = x[5] - PSI * hx

    # geometry
    solid = np.zeros((nx, ny), dtype=bool)
    solid[5:, :] = True
    fluid_idx = np.array([ic(i, j) for i in range(nx) for j in range(ny) if not solid[i, j]], dtype=np.int32)
    faces = []
    faces += [ic(0, j) for j in range(ny)]
    faces += [ic(nx - 1, j) for j in range(ny)]
    faces += [ic(i, 0) for i in range(nx)]
    faces += [ic(i, ny - 1) for i in range(nx)]
    dirichlet_face_idx = np.array(faces, dtype=np.int32)

    R0 = dict(
        psi=np.full(n_Rx, PSI),
        position=np.array([[x_R, y[j], 0.0] for j in (1, 2, 3)]),
        normal=np.array([[-1.0, 0.0, 0.0]] * n_Rx),
        ray_outside=np.zeros(n_Rx, np.uint8),
        solid_coord=np.array([[5, j, 0] for j in (1, 2, 3)], dtype=np.int32),
        solid_ic=np.array([ic(5, j) for j in (1, 2, 3)], dtype=np.int64),
        shape_id=np.array([0, 0, 1], dtype=np.int32),
        dirichlet_idx=np.array([2], dtype=np.int32),
        non_dirichlet_idx=np.array([0, 1], dtype=np.int32),
    )

    taps = np.array([[1.0, -2.0, 1.0] / hx**2, [1.0, -2.0, 1.0] / hy**2, [0.0, 0.0, 0.0]])

    # A0: x-lines
    row, col, val, part, line = [], [], [], [], []
    lines0, lines0_psi, lines0_skip = [], [], []
    for j in range(ny):
        L = len(lines0)
        skipped = j in (0, ny - 1)
        endR = j - 1 if 1 <= j <= 3 else -1
        lines0.append([sx, ic(0, j), ic(4, j), -1, -1, endR, 0 if endR >= 0 else -1])
        lines0_psi.append([np.nan, PSI if endR >= 0 else np.nan])
        lines0_skip.append(1 if skipped else 0)
        if skipped:
            continue
        # left dense row i = 1 at the Dirichlet wall
        for jj, c in enumerate(taps[0]):
            row.append(ic(1, j)); col.append(ic(jj, j)); val.append(c); part.append(0); line.append(L)
        # interior rows
        for i in (2, 3):
            for jj, c in enumerate(taps[0]):
                row.append(ic(i, j)); col.append(ic(i + jj - 1, j)); val.append(c); part.append(1); line.append(L)
        # right dense row i = 4 through (x_3, x_4, x_R)
        c4, c3, cR = three_point_second_derivative(x[4], x[3], x_R)
        row.append(ic(4, j)); col.append(ic(3, j)); val.append(c3); part.append(2); line.append(L)
        row.append(ic(4, j)); col.append(ic(4, j)); val.append(c4); part.append(2); line.append(L)
        row.append(ic(4, j)); col.append(off_Rx + endR); val.append(cR); part.append(3); line.append(-1)
    # Rx rows of A0 (non-Dirichlet records only): f'' at x_R from (x_R, x_4, x_3)
    for r in (0, 1):
        j = r + 1
        cR, c4, c3 = three_point_second_derivative(x_R, x[4], x[3])
        prow = off_Rx + r
        row.append(prow); col.append(ic(4, j)); val.append(c4); part.append(5); line.append(-1)
        row.append(prow); col.append(ic(3, j)); val.append(c3); part.append(5); line.append(-1)
        row.append(prow); col.append(prow); val.append(cR); part.append(6); line.append(-1)
    A0 = dict(row=np.array(row, np.int64), col=np.array(col, np.int64), val=np.array(val),
              part=np.array(part, np.uint8), line=np.array(line, np.int32))

    # A1: y-lines i = 1..4 (i = 0 and 6 are Dirichlet wall lines, 5 and 6 solid)
    row, col, val, part, line = [], [], [], [], []
    lines1, lines1_psi, lines1_skip = [], [], []
    for i in range(nx):
        L = len(lines1)
        lines1.append([sy, ic(i, 0), ic(i, ny - 1), -1, -1, -1, -1])
        lines1_psi.append([np.nan, np.nan])
        skipped = i in (0, nx - 1) or solid[i, 0]
        lines1_skip.append(1 if skipped else 0)
        if skipped:
            continue
        for j, p in ((1, 0), (2, 1), (3, 2)):
            for jj, c in enumerate(taps[1]):
                row.append(ic(i, j)); col.append(ic(i, j + jj - 1)); val.append(c); part.append(p); line.append(L)
    A1 = dict(row=np.array(row, np.int64), col=np.array(col, np.int64), val=np.array(val),
              part=np.array(part, np.uint8), line=np.array(line, np.int32))

    role = np.zeros(n_D, np.uint8)
    role[np.unique(A0["row"][A0["part"] == 1])] |= 1
    role[np.unique(A1["row"][A1["part"] == 1])] |= 2
    role[fluid_idx] |= 8
    role[dirichlet_face_idx] |= 16

    h = [hx, hy, 1.0]
    dt = 0.5 * min(hx, hy) ** 2 / (4 * K)
    meta = {
        "export_version": 1, "config": "synthetic", "script": "", "dims": 2,
        "extents": [nx, ny, nz], "domain_min": [0.0, 0.0, 0.0], "domain_max": [1.0, 1.0, 0.0],
        "h": h, "n_D": n_D, "n_R": [n_Rx, 0, 0],
        "offsets": {"D": 0, "Rx": off_Rx, "Ry": n_V, "Rz": n_V}, "n_V": n_V,
        "grid_bcs": [["dirichlet", "dirichlet"], ["dirichlet", "dirichlet"], ["floating", "floating"]],
        "object_bcs": ["floating", "dirichlet"],
        "scheme": {"type": "E2", "order": 2, "p": 1},
        "system": {"type": "heat", "diffusivity": K},
        "mms": MMS, "t0": 0.0, "dt": dt, "seed": 1, "git_commit": "synthetic",
    }
    return dict(meta=meta, x=x, y=y, z=z, R0=R0, taps=taps, fluid_idx=fluid_idx,
                dirichlet_face_idx=dirichlet_face_idx, A0=A0, A1=A1,
                lines0=np.array(lines0, np.int64), lines0_psi=np.array(lines0_psi),
                lines0_skipped=np.array(lines0_skip, np.uint8),
                lines1=np.array(lines1, np.int64), lines1_psi=np.array(lines1_psi),
                lines1_skipped=np.array(lines1_skip, np.uint8), role=role)


def write_synthetic_export(root: str | Path, with_oracle: bool = True) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    a = build_arrays()
    with open(root / "meta.json", "w") as f:
        json.dump(a["meta"], f, indent=1)
    for name in ("x", "y", "z", "taps", "fluid_idx", "dirichlet_face_idx", "role",
                 "lines0", "lines0_psi", "lines0_skipped", "lines1", "lines1_psi", "lines1_skipped"):
        np.save(root / f"{name}.npy", a[name])
    for k, v in a["R0"].items():
        np.save(root / f"R0_{k}.npy", v)
    for d in (0, 1):
        for k, v in a[f"A{d}"].items():
            np.save(root / f"A{d}_{k}.npy", v)
    if with_oracle:
        from pyshoccs.heat import HeatSystem
        from pyshoccs.tables import load_tables

        t = load_tables(root)
        sysm = HeatSystem(t, backend="scipy")
        rng = np.random.default_rng(a["meta"]["seed"])
        rand = rng.uniform(-1.0, 1.0, t.layout.n_V)
        t0, dt = sysm.t0, sysm.dt
        u0 = sysm.initialize(t0)
        vec = {
            "rand_V": rand,
            "lap_rand_V": t.exact_apply(rand),
            "rhs_rand_V": sysm.rhs(rand, t0).copy(),
            "u0_V": u0,
            "rhs_u0_V": sysm.rhs(u0, t0).copy(),
            "sol_t0_V": sysm.solution(t0),
            "src_t0_V": sysm.source(t0),
            "sol_t1_V": sysm.solution(t0 + dt / 2),
            "src_t1_V": sysm.source(t0 + dt / 2),
        }
        u1 = sysm.rk4_step(u0, t0, dt).copy()
        vec["u1_V"] = u1
        for k, v in vec.items():
            np.save(root / f"{k}.npy", np.ascontiguousarray(v, dtype=np.float64))
        with open(root / "stats_u1.json", "w") as f:
            json.dump({"stats": sysm.stats(u1, t0 + dt)}, f)
    return root


if __name__ == "__main__":
    import sys

    print(write_synthetic_export(sys.argv[1] if len(sys.argv) > 1 else "synthetic_export"))
