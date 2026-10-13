#!/usr/bin/env python3
"""Time the Python array arm on one exported case.

For an export directory this script times, per backend, the operator apply,
one RHS, one RK4 step with its stats (what the shipped C++ step does once per
step) and the step's parts (operator, source, boundary update, stats, stage
arithmetic), as the median of N repetitions after a warm-up, and writes JSON
with machine information and the native C++ bar of Part IV E for the matching
case.

Backends: ``scipy`` (CSR matvec), ``structured`` (axis stencil plus exception
CSR, NumPy), and behind the ``bench`` extra ``torch-eager``, ``torch-compile``
(CPU, float64) and ``mlx`` (CPU stream, float64, ``mx.compile``), all three on
the structured form. ``torch-mps`` (eager) and ``mlx-gpu`` (``mx.compile``) run
the same structured form in float32 on the Apple GPU. Every MLX and GPU backend
needs a Metal device, which the Bash sandbox hides: ``import mlx.core`` fails
there even for the CPU stream, and torch reports no MPS device.

NumPy runs single-threaded unless ``--threads`` says otherwise; the thread
variables are set before numpy is imported. ``--runs N`` repeats the whole
measurement N times in one process and writes one JSON per run (``_runN``
inserted before the suffix of ``--out``).

Usage (from python/pyshoccs):
  uv run python bench/bench_heat.py --export ../../.scratch/export/heat51 \
      --out ../../.scratch/phase1/py_arm/heat51_1t_run1.json
  uv run python bench/bench_heat.py --export ... --backends cpu --threads 4
  uv run python bench/bench_heat.py --export ... --backends mlx,gpu --runs 2   (outside the sandbox)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _pre_parse_threads(argv) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--threads", type=int, default=1)
    ns, _ = pre.parse_known_args(argv)
    return max(1, ns.threads)


THREADS = _pre_parse_threads(sys.argv[1:])
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = str(THREADS)
# torch.compile writes its inductor cache under /tmp by default; keep it in TMPDIR
import tempfile  # noqa: E402

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(tempfile.gettempdir(), "torchinductor_pyshoccs"))

import numpy as np  # noqa: E402
import scipy  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pyshoccs.heat import HeatSystem, RK_C, RK_W, STATS_NAMES  # noqa: E402
from pyshoccs.structure import StructuredOperator, derive_structure  # noqa: E402
from pyshoccs.tables import load_tables, default_export_dir  # noqa: E402

# ----------------------------------------------------------------------------
# The native C++ bar (Part IV E), microseconds unless named *_ms
# ----------------------------------------------------------------------------

CPP_BAR_PROVENANCE = (
    "docs/python-dsl-survey/05-measurements.md Part IV E, native macOS build measured 2026-10-13: "
    "Apple M4 (4 performance cores), Apple clang 21.0.0, Kokkos 5.1.1 OpenMP via llvm-openmp 20.1.8, "
    "Release. Stage figures are bench_rhs_cutcell medians (--benchmark_min_time=0.3s) in microseconds; "
    "step_ms is the end-to-end median Wall_ms of the shipped RK4 step. 1t = one thread "
    "(--kokkos-num-threads=1 / OMP_NUM_THREADS=1); 4t = four threads with libomp defaults; "
    "4t_spin = four threads with KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active. Single runs."
)

CPP_BAR = {
    (51, 51, 1): {
        "case": "heat51",
        "rhs_submit_us": {"1t": 21.0, "4t": 309.0, "4t_spin": 54.1},
        "fill_source_us": {"1t": 59.8, "4t": 45.6, "4t_spin": 47.4},
        "update_boundary_us": {"1t": 24.4, "4t": 89.7, "4t_spin": 32.7},
        "stats_us": {"1t": 26.9, "4t": 86.0, "4t_spin": 33.0},
        "step_ms": {"1t": 0.856, "4t": 2.26, "4t_spin": 0.266},
    },
    (201, 201, 1): {
        "case": "heat201",
        "rhs_submit_us": {"1t": 217.0, "4t": 358.0, "4t_spin": 228.0},
        "fill_source_us": {"1t": 922.0, "4t": 296.0, "4t_spin": 682.0},
        "update_boundary_us": {"1t": 351.0, "4t": 184.0, "4t_spin": 312.0},
        "stats_us": {"1t": 377.0, "4t": 191.0, "4t_spin": 319.0},
        "step_ms": {"1t": 6.82, "4t": 4.05, "4t_spin": 2.17},
    },
    (401, 401, 1): {
        "case": "heat401",
        "rhs_submit_us": {"1t": 827.0, "4t": 575.0, "4t_spin": 797.0},
        "step_ms": {"1t": 26.96, "4t": 10.09, "4t_spin": 8.18},
    },
    (65, 65, 65): {
        "case": "heat3d65",
        "rhs_submit_us": {"1t": 2065.0, "4t": 961.0, "4t_spin": 1810.0},
        "fill_source_us": {"1t": 7985.0, "4t": 2429.0, "4t_spin": 5250.0},
        "update_boundary_us": {"1t": 2813.0, "4t": 906.0, "4t_spin": 2076.0},
        "stats_us": {"1t": 2960.0, "4t": 955.0, "4t_spin": 2045.0},
        "step_ms": {"1t": 58.4, "4t": 19.96, "4t_spin": 17.49},
    },
}

NUMPY_BACKENDS = ("scipy", "structured")
TORCH_CPU_BACKENDS = ("torch-eager", "torch-compile")
MLX_CPU_BACKENDS = ("mlx",)
GPU_BACKENDS = ("torch-mps", "mlx-gpu")
ALIASES = {
    "numpy": NUMPY_BACKENDS,
    "cpu": NUMPY_BACKENDS + TORCH_CPU_BACKENDS,
    "all": NUMPY_BACKENDS + TORCH_CPU_BACKENDS + MLX_CPU_BACKENDS,
    "gpu": GPU_BACKENDS,
}


class Skip(Exception):
    pass


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------


def timeit(fn, reps: int, warm: int, sync=lambda: None) -> dict:
    for _ in range(warm):
        fn()
        sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {
        "ms_median": statistics.median(ts) * 1e3,
        "ms_min": ts[0] * 1e3,
        "ms_p90": ts[int(0.9 * (len(ts) - 1))] * 1e3,
        "reps": reps,
    }


def split(T: dict) -> dict:
    """The step as the sum of its parts (4 stages, stats once) against the measured step.

    ``rhs_glue`` is what an RHS costs beyond the operator and the source
    evaluation: the scale by k, the scatter-add of the source and the zero fill
    of the Dirichlet rows. ``axpy`` is the stage arithmetic of the integrator,
    measured directly where the backend exposes it (NumPy); for the functional
    backends it is inside the remainder.
    """
    g = lambda k: T[k]["ms_median"]  # noqa: E731
    op, rhs, src, bc, st, step = (g(k) for k in ("operator", "rhs", "source", "boundary", "stats", "step"))
    parts = {"operator_x4": 4 * op, "source_x4": 4 * src, "rhs_glue_x4": 4 * (rhs - op - src),
             "boundary_x4": 4 * bc, "stats_x1": st}
    if "axpy" in T:
        parts["axpy_x1"] = g("axpy")
    parts["sum_of_parts"] = sum(parts.values())
    parts["step"] = step
    parts["remainder"] = step - parts["sum_of_parts"]
    return parts


def correctness(u1: np.ndarray, stats: list[float], u_ref: dict, u_n: np.ndarray | None = None) -> dict:
    def rel(a, b):
        scale = float(np.abs(b).max()) or 1.0
        return float(np.abs(np.asarray(a, dtype=np.float64) - b).max() / scale)

    out = {}
    if "u1_V" in u_ref:
        out["rel_err_u1_vs_cpp"] = rel(u1, u_ref["u1_V"])
    if "u1_numpy" in u_ref:
        out["rel_err_u1_vs_numpy"] = rel(u1, u_ref["u1_numpy"])
    if u_n is not None and "u_n_numpy" in u_ref:
        out["n_steps"] = u_ref["n_steps"]
        out["rel_err_un_vs_numpy"] = rel(u_n, u_ref["u_n_numpy"])
        out["abs_err_un_vs_numpy"] = float(np.abs(np.asarray(u_n, dtype=np.float64) - u_ref["u_n_numpy"]).max())
    if "stats_u1" in u_ref and u_ref["stats_u1"] is not None:
        ref = u_ref["stats_u1"]
        out["stats_max_rel_err_vs_cpp"] = max(
            abs(a - b) / max(abs(b), 1e-300) for a, b, name in zip(stats, ref, STATS_NAMES)
            if not name.endswith("_ic"))
        out["stats_index_match"] = all(a == b for a, b, name in zip(stats, ref, STATS_NAMES)
                                       if name.endswith("_ic"))
    return out


# ----------------------------------------------------------------------------
# NumPy / SciPy backends
# ----------------------------------------------------------------------------


def stage_axpy(u0, dt, out, k, rk, tmp) -> None:
    """The slot arithmetic of ``rk4_step_generic`` with the RHS and boundary calls removed."""
    rk.fill(0.0)
    k.fill(0.0)
    np.copyto(out, u0)
    for i in range(4):
        if i > 0:
            np.multiply(k, dt * RK_C[i], out=tmp)
            np.add(u0, tmp, out=out)
        np.multiply(k, dt * RK_W[i], out=tmp)
        rk += tmp
    np.add(u0, rk, out=out)


def numpy_multi_step(sysm: HeatSystem, u0: np.ndarray, n: int) -> np.ndarray:
    u, t = u0.copy(), sysm.t0
    for _ in range(n):
        u = sysm.rk4_step(u, t, sysm.dt).copy()
        t += sysm.dt
    return u


def bench_numpy_backend(tables, structure, backend: str, reps: int, warm: int, u_ref: dict,
                        n_steps: int) -> dict:
    sysm = HeatSystem(tables, backend=backend, structure=structure)
    direct = HeatSystem(tables, backend=backend, structure=structure, hoist=False)
    n = tables.layout.n_V
    rand = tables.oracle.get("rand_V", np.random.default_rng(0).uniform(-1, 1, n))
    u0 = tables.oracle.get("u0_V", sysm.initialize())
    t0, dt = sysm.t0, sysm.dt
    out = np.empty(n)
    ub = u0.copy()
    rec = {"backend": backend, "status": "ok", "timings": {}}
    T = rec["timings"]
    T["operator"] = timeit(lambda: sysm.apply_operator(rand, out), reps, warm)
    T["rhs"] = timeit(lambda: sysm.rhs(rand, t0, out), reps, warm)
    T["source"] = timeit(lambda: sysm.source_selected(t0), reps, warm)
    T["boundary"] = timeit(lambda: sysm.update_boundary(ub, t0), reps, warm)
    T["stats"] = timeit(lambda: sysm.stats(u0, t0), reps, warm)
    T["axpy"] = timeit(lambda: stage_axpy(u0, dt, out, sysm._k, sysm._rk, sysm._tmp), reps, warm)
    T["step_nostats"] = timeit(lambda: sysm.rk4_step(u0, t0, dt, out), reps, warm)

    def step_with_stats():
        sysm.rk4_step(u0, t0, dt, out)
        return sysm.stats(out, t0 + dt)

    T["step"] = timeit(step_with_stats, reps, warm)
    small = max(5, reps // 4)
    T["source_direct"] = timeit(lambda: direct.source_selected(t0), small, 2)
    T["boundary_direct"] = timeit(lambda: direct.update_boundary(ub, t0), small, 2)
    T["stats_direct"] = timeit(lambda: direct.stats(u0, t0), small, 2)

    def step_direct():
        direct.rk4_step(u0, t0, dt, out)
        return direct.stats(out, t0 + dt)

    T["step_direct"] = timeit(step_direct, small, 2)
    rec["split_ms"] = split(T)
    u1 = sysm.rk4_step(u0, t0, dt).copy()
    u_n = numpy_multi_step(sysm, u0, n_steps) if n_steps else None
    rec.update(correctness(u1, sysm.stats(u1, t0 + dt), u_ref, u_n))
    rec["note"] = ("hoisted MMS: source and boundary values from precomputed spatial factors "
                   "(the *_direct timings evaluate the gaussian per point every call); "
                   "step = rk4_step plus stats, step_nostats = rk4_step alone; "
                   "axpy = the stage arithmetic alone on the same buffers")
    return rec


# ----------------------------------------------------------------------------
# Shared structured-form tables for torch and MLX
# ----------------------------------------------------------------------------


class StructuredArrays:
    """The structured tables as plain NumPy arrays, ready for a tensor backend."""

    def __init__(self, tables, structure):
        sysm = HeatSystem(tables, backend="structured", structure=structure)
        op: StructuredOperator = sysm.op
        self.sysm = sysm
        self.ext = tables.layout.extents
        self.n_D = tables.layout.n_D
        self.n_V = tables.layout.n_V
        self.n_R = self.n_V - self.n_D
        self.half = op.half
        self.width = structure.width
        self.dirs = op.dirs
        self.pad = op.pad
        self.windows = op.windows
        self.taps = {d: structure.taps[d].copy() for d in self.dirs}
        self.masks = {d: op.masks[d] for d in self.dirs}
        exc = structure.exc
        nnz = np.diff(exc.indptr)
        W = int(nnz.max()) if nnz.size else 1
        n_exc = exc.shape[0]
        self.ell_cols = np.zeros((n_exc, W), dtype=np.int64)
        self.ell_vals = np.zeros((n_exc, W), dtype=np.float64)
        for r in range(n_exc):
            a, b = exc.indptr[r], exc.indptr[r + 1]
            self.ell_cols[r, : b - a] = exc.indices[a:b]
            self.ell_vals[r, : b - a] = exc.data[a:b]
        self.exc_rows = structure.exc_rows.astype(np.int64)
        self.ell_width = W
        self.ell_fill = float(nnz.sum() / (n_exc * W)) if n_exc else 1.0
        self.src_idx = sysm.src_idx.astype(np.int64)
        self.bc_idx = np.unique(sysm.bc_idx).astype(np.int64)
        self.keep = np.ones(self.n_V)
        self.keep[self.bc_idx] = 0.0
        self.f = sysm.mms.frequency.copy()
        self.k = sysm.k
        self.Msrc = np.ascontiguousarray(sysm.src_eval.M.T)          # (n_src, 2n)
        self.Gstat = np.ascontiguousarray(sysm.src_eval.G.T)         # (n_src, n)
        from pyshoccs.mms import HoistedGaussian

        bc_eval = HoistedGaussian(sysm.mms, tables.locations[self.bc_idx], sysm.k)
        self.Gbc = np.ascontiguousarray(bc_eval.G.T)                 # (n_bc, n)
        self.seg_end = sysm._seg_end.astype(np.int64)
        self.fluid_idx = sysm.fluid_idx
        self.nondir_local = [tables.R[d].non_dirichlet_idx for d in range(3)]
        self.t0, self.dt = sysm.t0, sysm.dt
        self.rand = tables.oracle.get("rand_V", np.random.default_rng(0).uniform(-1, 1, self.n_V))
        self.u0 = tables.oracle.get("u0_V", sysm.initialize())


def stats_from_parts(sa: StructuredArrays, err: np.ndarray, su: np.ndarray) -> list[float]:
    """The eleven numbers from |u - sol| and u on the src points (host side)."""
    ends = sa.seg_end
    nf = int(ends[0])
    if nf:
        j = int(np.argmax(err[:nf]))
        err_d, idx_d = float(err[j]), float(sa.fluid_idx[j])
        u_min, u_max = float(su[:nf].min()), float(su[:nf].max())
    else:
        err_d = idx_d = u_min = u_max = 0.0
    ce, ci = [0.0] * 3, [0.0] * 3
    for d in range(3):
        a, b = int(ends[d]), int(ends[d + 1])
        if b == a:
            continue
        j = int(np.argmax(err[a:b]))
        ce[d], ci[d] = float(err[a + j]), float(sa.nondir_local[d][j])
        u_min, u_max = min(u_min, float(su[a:b].min())), max(u_max, float(su[a:b].max()))
    return [max(err_d, *ce), u_min, u_max, err_d, idx_d, ce[0], ci[0], ce[1], ci[1], ce[2], ci[2]]


# ----------------------------------------------------------------------------
# torch
# ----------------------------------------------------------------------------


class TorchHeat:
    def __init__(self, sa: StructuredArrays, compiled: bool, threads: int,
                 device: str = "cpu", dtype: str = "float64"):
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise Skip(f"torch import failed: {e!s:.200}")
        self.torch = torch
        torch.set_num_threads(threads)
        if device == "mps" and not torch.backends.mps.is_available():
            raise Skip("torch reports no MPS device (the Bash sandbox hides Metal)")
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.sync = torch.mps.synchronize if device == "mps" else (lambda: None)
        self.sa = sa
        self.compiled = compiled
        if device == "cpu":
            self.name = "torch-compile" if compiled else "torch-eager"
        else:
            self.name = f"torch-{device}" + ("-compile" if compiled else "")
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)  # noqa: E731
        I = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int64, device=self.device)  # noqa: E731
        self.T = T
        self.taps = {d: [float(c) for c in sa.taps[d]] for d in sa.dirs}
        self.masks = {d: T(sa.masks[d]) for d in sa.dirs}
        # F.pad takes pairs from the last dimension backwards
        self.padspec = tuple(p for d in (2, 1, 0) for p in sa.pad[d])
        self.ell_cols = I(sa.ell_cols)
        self.ell_vals = T(sa.ell_vals)
        self.exc_rows = I(sa.exc_rows)
        self.src_idx = I(sa.src_idx)
        self.bc_idx = I(sa.bc_idx)
        self.keep = T(sa.keep)
        self.f = T(sa.f)
        self.k = float(sa.k)
        self.Msrc = T(sa.Msrc)
        self.Gstat = T(sa.Gstat)
        self.Gbc = T(sa.Gbc)
        self.zR = torch.zeros(sa.n_R, dtype=self.dtype, device=self.device)
        self.nf = int(sa.seg_end[0])
        self.step_fn = torch.compile(self._step) if compiled else self._step
        self.rhs_fn = torch.compile(self._rhs) if compiled else self._rhs
        self.op_fn = torch.compile(self._operator) if compiled else self._operator

    def host(self, x) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _operator(self, u):
        torch, sa = self.torch, self.sa
        D = u[: sa.n_D].view(sa.ext)
        Dp = torch.nn.functional.pad(D, self.padspec)
        total = None
        for d in sa.dirs:
            taps, wins = self.taps[d], sa.windows[d]
            acc = Dp[wins[0]] * taps[0]
            for j in range(1, sa.width):
                acc = acc + Dp[wins[j]] * taps[j]
            acc = acc * self.masks[d]
            total = acc if total is None else total + acc
        du = torch.cat([total.reshape(-1), self.zR])
        vals = (self.ell_vals * u[self.ell_cols]).sum(dim=1)
        return du.index_add(0, self.exc_rows, vals)

    def _rhs(self, u, t):
        torch = self.torch
        du = self._operator(u) * self.k
        ft = self.f * t
        c = torch.cat([-self.f * torch.sin(ft), -self.k * torch.cos(ft)])
        du = du.index_add(0, self.src_idx, torch.mv(self.Msrc, c))
        return du * self.keep

    def _update_boundary(self, u, t):
        vals = self.torch.mv(self.Gbc, self.torch.cos(self.f * t))
        return (u * self.keep).index_add(0, self.bc_idx, vals)

    def _step(self, u0, t):
        sa, torch = self.sa, self.torch
        dt = sa.dt
        rk = None
        u = u0
        for i in range(4):
            ti = t + dt * RK_C[i]
            if i > 0:
                u = self._update_boundary(u0 + (dt * RK_C[i]) * k, ti)
            k = self._rhs(u, ti)
            rk = (dt * RK_W[i]) * k if rk is None else rk + (dt * RK_W[i]) * k
        u1 = self._update_boundary(u0 + rk, t + dt)
        su = u1[self.src_idx]
        sol = torch.mv(self.Gstat, torch.cos(self.f * (t + dt)))
        err = (su - sol).abs()
        return u1, err, su

    def multi_step(self, u0, t0, n: int) -> np.ndarray:
        u, t = u0, t0
        for _ in range(n):
            u, _, _ = self.step_fn(u, t)
            t = t + self.sa.dt
        self.sync()
        return self.host(u)

    def run(self, reps, warm, u_ref, n_steps) -> dict:
        torch, sa = self.torch, self.sa
        rand = self.T(sa.rand)
        u0 = self.T(sa.u0)
        t0 = torch.tensor(sa.t0, dtype=self.dtype, device=self.device)
        rec = {"backend": self.name, "status": "ok", "timings": {}}
        Tm = rec["timings"]
        t_first = time.perf_counter()
        self.step_fn(u0, t0)
        self.sync()
        rec["first_step_s"] = time.perf_counter() - t_first
        sync = self.sync
        Tm["operator"] = timeit(lambda: self.op_fn(rand), reps, warm, sync)
        Tm["rhs"] = timeit(lambda: self.rhs_fn(rand, t0), reps, warm, sync)
        ff = torch.cat([self.f, self.f])
        Tm["source"] = timeit(lambda: torch.mv(self.Msrc, ff), reps, warm, sync)
        Tm["boundary"] = timeit(lambda: self._update_boundary(u0, t0), reps, warm, sync)

        def stats_only():
            su = u0[self.src_idx]
            sol = torch.mv(self.Gstat, torch.cos(self.f * t0))
            err = (su - sol).abs()
            return stats_from_parts(sa, self.host(err), self.host(su))

        Tm["stats"] = timeit(stats_only, reps, warm, sync)

        def step():
            u1, err, su = self.step_fn(u0, t0)
            return stats_from_parts(sa, self.host(err), self.host(su))

        Tm["step"] = timeit(step, reps, warm, sync)
        rec["split_ms"] = split(Tm)
        u1, err, su = self.step_fn(u0, t0)
        u_n = self.multi_step(u0, t0, n_steps) if n_steps else None
        rec.update(correctness(self.host(u1).copy(), stats_from_parts(sa, self.host(err), self.host(su)),
                               u_ref, u_n))
        rec["device"] = str(self.device)
        rec["dtype"] = str(self.dtype).replace("torch.", "")
        rec["note"] = ("structured form; exception rows as a zero-padded ELL gather-dot "
                       f"(width {sa.ell_width}, fill {sa.ell_fill:.2f}); functional RK4 step"
                       + ("; torch.compile over the whole step, RHS and operator" if self.compiled else
                          "; eager") + f"; {rec['device']} {rec['dtype']}"
                       + f"; intra-op threads {torch.get_num_threads()}"
                       + ("; one torch.mps.synchronize per timed call" if rec["device"] == "mps" else ""))
        return rec


# ----------------------------------------------------------------------------
# MLX
# ----------------------------------------------------------------------------


_MLX_IMPORT: dict = {}


def import_mlx():
    """Import mlx.core once per process.

    A failed import (no Metal device under the sandbox) leaves nanobind's type
    registry half filled, and a second attempt aborts the interpreter, so the
    outcome of the first attempt is kept and reused.
    """
    if "module" not in _MLX_IMPORT:
        try:
            import mlx.core as mx

            _MLX_IMPORT["module"] = mx
        except Exception as e:  # noqa: BLE001
            _MLX_IMPORT["module"] = None
            _MLX_IMPORT["error"] = f"{e!s:.200}"
    if _MLX_IMPORT["module"] is None:
        raise Skip(f"mlx import failed: {_MLX_IMPORT['error']}")
    return _MLX_IMPORT["module"]


class MLXHeat:
    def __init__(self, sa: StructuredArrays, device: str = "cpu", dtype: str = "float64"):
        mx = import_mlx()
        self.mx = mx
        mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
        self.dtype = getattr(mx, dtype)
        self.np_dtype = np.float64 if dtype == "float64" else np.float32
        try:
            mx.eval(mx.ones((4,), dtype=self.dtype) * 2)
        except Exception as e:  # noqa: BLE001
            raise Skip(f"MLX {dtype} on {device} unsupported: {e!s:.200}")
        self.sa = sa
        self.device = device
        self.name = "mlx" if device == "cpu" else "mlx-gpu"
        T = lambda a: mx.array(np.asarray(a, dtype=self.np_dtype), dtype=self.dtype)  # noqa: E731
        I = lambda a: mx.array(np.asarray(a, dtype=np.int64))  # noqa: E731
        self.T = T
        self.taps = {d: [float(c) for c in sa.taps[d]] for d in sa.dirs}
        self.masks = {d: T(sa.masks[d]) for d in sa.dirs}
        self.padspec = [tuple(int(p) for p in sa.pad[d]) for d in range(3)]
        self.ell_cols = I(sa.ell_cols)
        self.ell_vals = T(sa.ell_vals)
        self.exc_rows = I(sa.exc_rows)
        self.src_idx = I(sa.src_idx)
        self.bc_idx = I(sa.bc_idx)
        self.keep = T(sa.keep)
        self.f = T(sa.f)
        self.k = float(sa.k)
        self.Msrc = T(sa.Msrc)
        self.Gstat = T(sa.Gstat)
        self.Gbc = T(sa.Gbc)
        self.zR = mx.zeros((sa.n_R,), dtype=self.dtype)
        mx.eval(*self.masks.values(), self.ell_cols, self.ell_vals, self.exc_rows, self.src_idx,
                self.bc_idx, self.keep, self.f, self.Msrc, self.Gstat, self.Gbc, self.zR)
        self.step_fn = mx.compile(self._step)
        self.rhs_fn = mx.compile(self._rhs)
        self.op_fn = mx.compile(self._operator)

    def _operator(self, u):
        mx, sa = self.mx, self.sa
        D = u[: sa.n_D].reshape(sa.ext)
        Dp = mx.pad(D, self.padspec)
        total = None
        for d in sa.dirs:
            taps, wins = self.taps[d], sa.windows[d]
            acc = Dp[wins[0]] * taps[0]
            for j in range(1, sa.width):
                acc = acc + Dp[wins[j]] * taps[j]
            acc = acc * self.masks[d]
            total = acc if total is None else total + acc
        du = mx.concatenate([total.reshape(-1), self.zR])
        vals = (self.ell_vals * u[self.ell_cols]).sum(axis=1)
        return du.at[self.exc_rows].add(vals)

    def _rhs(self, u, t):
        mx = self.mx
        du = self._operator(u) * self.k
        ft = self.f * t
        c = mx.concatenate([-self.f * mx.sin(ft), -self.k * mx.cos(ft)])
        du = du.at[self.src_idx].add(self.Msrc @ c)
        return du * self.keep

    def _update_boundary(self, u, t):
        vals = self.Gbc @ self.mx.cos(self.f * t)
        return (u * self.keep).at[self.bc_idx].add(vals)

    def _step(self, u0, t):
        sa, mx = self.sa, self.mx
        dt = sa.dt
        rk = None
        u = u0
        for i in range(4):
            ti = t + dt * RK_C[i]
            if i > 0:
                u = self._update_boundary(u0 + (dt * RK_C[i]) * k, ti)
            k = self._rhs(u, ti)
            rk = (dt * RK_W[i]) * k if rk is None else rk + (dt * RK_W[i]) * k
        u1 = self._update_boundary(u0 + rk, t + dt)
        su = u1[self.src_idx]
        sol = self.Gstat @ mx.cos(self.f * (t + dt))
        err = mx.abs(su - sol)
        return u1, err, su

    def multi_step(self, u0, t0, n: int) -> np.ndarray:
        mx = self.mx
        u, t = u0, t0
        for _ in range(n):
            u, _, _ = self.step_fn(u, t)
            t = t + self.sa.dt
            mx.eval(u, t)
        return np.array(u)

    def run(self, reps, warm, u_ref, n_steps) -> dict:
        mx, sa = self.mx, self.sa
        rand = self.T(sa.rand)
        u0 = self.T(sa.u0)
        t0 = mx.array(sa.t0, dtype=self.dtype)
        mx.eval(rand, u0, t0)
        rec = {"backend": self.name, "status": "ok", "timings": {}}
        Tm = rec["timings"]
        t_first = time.perf_counter()
        mx.eval(*self.step_fn(u0, t0))
        rec["first_step_s"] = time.perf_counter() - t_first
        Tm["operator"] = timeit(lambda: mx.eval(self.op_fn(rand)), reps, warm)
        Tm["rhs"] = timeit(lambda: mx.eval(self.rhs_fn(rand, t0)), reps, warm)
        ff = mx.concatenate([self.f, self.f])
        Tm["source"] = timeit(lambda: mx.eval(self.Msrc @ ff), reps, warm)
        Tm["boundary"] = timeit(lambda: mx.eval(self._update_boundary(u0, t0)), reps, warm)

        def stats_only():
            su = u0[self.src_idx]
            sol = self.Gstat @ mx.cos(self.f * t0)
            err = mx.abs(su - sol)
            mx.eval(err, su)
            return stats_from_parts(sa, np.array(err), np.array(su))

        Tm["stats"] = timeit(stats_only, reps, warm)

        def step():
            u1, err, su = self.step_fn(u0, t0)
            mx.eval(u1, err, su)
            return stats_from_parts(sa, np.array(err), np.array(su))

        Tm["step"] = timeit(step, reps, warm)
        rec["split_ms"] = split(Tm)
        u1, err, su = self.step_fn(u0, t0)
        mx.eval(u1, err, su)
        u_n = self.multi_step(u0, t0, n_steps) if n_steps else None
        rec.update(correctness(np.array(u1), stats_from_parts(sa, np.array(err), np.array(su)), u_ref, u_n))
        rec["device"] = self.device
        rec["dtype"] = str(self.dtype).replace("mlx.core.", "")
        rec["note"] = (f"structured form on the MLX {self.device} stream in {rec['dtype']}; exception rows "
                       f"as a zero-padded ELL gather-dot (width {sa.ell_width}, fill {sa.ell_fill:.2f}); "
                       "mx.compile over the whole functional RK4 step; one mx.eval per timed call")
        return rec


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def machine_info(threads: int) -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mac_ver": platform.mac_ver()[0],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": threads,
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                       "KMP_BLOCKTIME", "OMP_WAIT_POLICY")},
    }
    for mod in ("torch", "mlx"):
        try:
            import importlib.metadata as md

            info[mod] = md.version(mod)
        except Exception:  # noqa: BLE001
            info[mod] = "unavailable"
    sc = {}
    for key in ("hw.ncpu", "hw.perflevel0.physicalcpu", "machdep.cpu.brand_string", "hw.memsize"):
        try:
            r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
            sc[key] = r.stdout.strip() if r.returncode == 0 else "unavailable"
        except Exception:  # noqa: BLE001
            sc[key] = "unavailable"
    info["sysctl"] = sc
    try:
        info["loadavg_at_start"] = os.getloadavg()
    except Exception:  # noqa: BLE001
        pass
    return info


def cpp_bar_for(extents) -> dict:
    bar = CPP_BAR.get(tuple(int(e) for e in extents))
    if bar is None:
        return {"case": None, "note": "no native C++ figure for these extents in Part IV E",
                "provenance": CPP_BAR_PROVENANCE}
    return dict(bar, provenance=CPP_BAR_PROVENANCE)


def ratios(rec: dict, bar: dict) -> dict:
    out = {}
    if rec.get("status") != "ok" or bar.get("case") is None:
        return out
    T = rec["timings"]
    for setting in ("1t", "4t_spin", "4t"):
        r = {}
        if "rhs_submit_us" in bar:
            r["rhs_over_cpp_rhs_submit"] = T["rhs"]["ms_median"] * 1e3 / bar["rhs_submit_us"][setting]
            r["operator_over_cpp_rhs_submit"] = T["operator"]["ms_median"] * 1e3 / bar["rhs_submit_us"][setting]
            r["step_over_4_cpp_rhs_submits"] = T["step"]["ms_median"] * 1e3 / (4 * bar["rhs_submit_us"][setting])
        if "step_ms" in bar:
            r["step_over_cpp_step"] = T["step"]["ms_median"] / bar["step_ms"][setting]
        if "fill_source_us" in bar:
            r["source_over_cpp_fill_source"] = T["source"]["ms_median"] * 1e3 / bar["fill_source_us"][setting]
            r["update_boundary_over_cpp"] = T["boundary"]["ms_median"] * 1e3 / bar["update_boundary_us"][setting]
            r["stats_over_cpp"] = T["stats"]["ms_median"] * 1e3 / bar["stats_us"][setting]
        out[setting] = r
    return out


def print_record(rec: dict) -> None:
    if rec.get("status") != "ok":
        print(f"  {rec['backend']:16s} {rec['status']}: {rec.get('reason', '')}")
        return
    T = rec["timings"]
    cells = "  ".join(f"{k} {T[k]['ms_median']:.4f}" for k in ("operator", "rhs", "source", "boundary", "stats", "step"))
    err = rec.get("rel_err_u1_vs_cpp", rec.get("rel_err_u1_vs_numpy", float("nan")))
    err_n = rec.get("rel_err_un_vs_numpy", float("nan"))
    print(f"  {rec['backend']:16s} ms: {cells}  | rel err u1 {err:.2e}, after {rec.get('n_steps', 0)} steps {err_n:.2e}")


def run_backends(backends, tables, structure, bar, reps, warm, u_ref, n_steps) -> dict:
    sa = None
    results = {}
    for b in backends:
        try:
            if b in NUMPY_BACKENDS:
                rec = bench_numpy_backend(tables, structure, b, reps, warm, u_ref, n_steps)
            else:
                if sa is None:
                    sa = StructuredArrays(tables, structure)
                if b == "torch-eager":
                    rec = TorchHeat(sa, False, THREADS).run(reps, warm, u_ref, n_steps)
                elif b == "torch-compile":
                    rec = TorchHeat(sa, True, THREADS).run(reps, warm, u_ref, n_steps)
                elif b == "torch-mps":
                    rec = TorchHeat(sa, False, THREADS, "mps", "float32").run(reps, warm, u_ref, n_steps)
                elif b == "mlx":
                    rec = MLXHeat(sa).run(reps, warm, u_ref, n_steps)
                elif b == "mlx-gpu":
                    rec = MLXHeat(sa, "gpu", "float32").run(reps, warm, u_ref, n_steps)
                else:
                    raise ValueError(f"unknown backend {b}")
        except Skip as e:
            rec = {"backend": b, "status": "skipped", "reason": str(e)}
        except Exception as e:  # noqa: BLE001
            import traceback

            rec = {"backend": b, "status": "failed", "reason": f"{type(e).__name__}: {e!s:.400}",
                   "traceback": traceback.format_exc(limit=4)}
        rec["ratios_to_cpp"] = ratios(rec, bar)
        results[b] = rec
        print_record(rec)
    return results


def parse_backends(spec: str) -> list[str]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        out.extend(ALIASES.get(item, (item,)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=None, help="export directory (default PYSHOCCS_EXPORT_DIR or .scratch/export/heat51)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1, help="NumPy/BLAS and torch thread count")
    ap.add_argument("--backends", default="numpy",
                    help="comma list of scipy,structured,torch-eager,torch-compile,mlx,torch-mps,mlx-gpu "
                         "or the aliases numpy (scipy,structured), cpu (numpy plus torch), all (cpu plus mlx), "
                         "gpu (torch-mps,mlx-gpu)")
    ap.add_argument("--steps", type=int, default=10,
                    help="chained steps for the drift check against the fp64 SciPy path (0 to skip)")
    ap.add_argument("--runs", type=int, default=1, help="repeat the whole measurement this many times")
    ap.add_argument("--out", default=None, help="JSON output path (with --runs > 1, _runN is inserted)")
    args = ap.parse_args(argv)

    export = Path(args.export) if args.export else default_export_dir()
    if export is None or not Path(export).is_dir():
        print("no export directory", file=sys.stderr)
        return 2
    backends = parse_backends(args.backends)

    tables = load_tables(export)
    structure = derive_structure(tables)
    bar = cpp_bar_for(tables.layout.extents)
    info = machine_info(THREADS)
    print(f"export {export}: extents {tables.layout.extents}, n_V {tables.layout.n_V}, "
          f"exc rows {structure.counts['exc_rows']}, threads {THREADS}, reps {args.reps}, runs {args.runs}")
    print(f"C++ bar: {bar.get('case')} {json.dumps({k: v for k, v in bar.items() if k not in ('provenance',)})}")

    u_ref = {k: tables.oracle[k] for k in ("u1_V",) if k in tables.oracle}
    u_ref["stats_u1"] = tables.stats_u1
    ref_sys = HeatSystem(tables, backend="scipy", structure=structure)
    u0 = tables.oracle.get("u0_V", ref_sys.initialize())
    u_ref["u1_numpy"] = ref_sys.rk4_step(u0, ref_sys.t0, ref_sys.dt).copy()
    if args.steps:
        u_ref["n_steps"] = args.steps
        u_ref["u_n_numpy"] = numpy_multi_step(ref_sys, u0, args.steps)

    for run in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"run {run} of {args.runs}")
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        t_run = time.perf_counter()
        results = run_backends(backends, tables, structure, bar, args.reps, args.warm, u_ref, args.steps)
        out = {
            "export_dir": str(Path(export).resolve()),
            "case": tables.case_name,
            "extents": list(tables.layout.extents),
            "n_V": tables.layout.n_V,
            "n_D": tables.layout.n_D,
            "structure_counts": structure.counts,
            "machine": info,
            "settings": {"reps": args.reps, "warm": args.warm, "threads": THREADS,
                         "steps_check": args.steps, "run": run, "runs": args.runs},
            "cpp_bar": bar,
            "results": results,
            "started": started,
            "wall_s": time.perf_counter() - t_run,
        }
        if args.out:
            p = Path(args.out)
            if args.runs > 1:
                p = p.with_name(f"{p.stem}_run{run}{p.suffix}")
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(out, f, indent=1, default=str)
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
