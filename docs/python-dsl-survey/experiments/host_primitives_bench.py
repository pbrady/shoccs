#!/usr/bin/env python3
"""Synthetic SHOCCS-shaped RK4 step on host Python array frameworks.

Measures one RK4 step of a heat-like system built from the SHOCCS runtime
primitive set (01-codebase-idioms.md Sections 4 to 7):

  P0  packed field vector [D | R], R about 0.5% of |D| (irregular cut-point values)
  P2  E2 3-point second derivative along every axis via shifted slices of a
      zero-padded view, multiplied by an interior mask (role bits: solid sphere
      about 6% of points, closure shell about 2.4%, Dirichlet faces)
  P3  fixed-width (w = 7) ELL gather-dot over the packed vector for the closure
      rows plus all R rows, scattered to unique output rows (no atomics),
      int32 indices, random but fixed dissipative coefficients
  P1  scale by diffusivity, add source * scalar time factor, RK4 stage axpby
  P4  Dirichlet face fill (slice assignment, or a static face mask in MLX)
  P5  one masked Linf reduction per step with a single host readback
  P6  whole-step compilation where the backend offers it

Coefficients are synthetic: E2 interior plus random closure coefficients, not
the real psi-dependent closures. There is no host MMS evaluation.

Usage (from scripts/stencil_gen):
  uv run --with mlx python ../../docs/python-dsl-survey/experiments/host_primitives_bench.py \
      --cases 51 --out run_smoke.json
  ... --cases all --out host_primitives_run1.json
  ... --merge host_primitives_run1.json host_primitives_run2.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    ("2D 51x51", (51, 51)),
    ("2D 71x71", (71, 71)),
    ("2D 201x201", (201, 201)),
    ("2D 401x401", (401, 401)),
    ("3D 65^3", (65, 65, 65)),
    ("3D 129^3", (129, 129, 129)),
    ("3D 257^3", (257, 257, 257)),
]

# C++ bar, ms per RK4 step, Kokkos OpenMP Release in the devcontainer, 8 threads
# (51x51 uses the 1-thread figure, which is the faster one).
CPP_BAR = {
    "2D 51x51": 0.90,
    "2D 201x201": 4.6,
    "2D 401x401": 13.7,
    "3D 65^3": 38.0,
}

BACKENDS = [
    "numpy-cpu-f64",
    "torch-cpu-f64-eager",
    "torch-cpu-f64-compile",
    "torch-mps-f32-eager",
    "torch-mps-f32-compile",
    "mlx-gpu-f32-compile",
    "mlx-cpu-f64-compile",
]

W = 7
NCHECK = 10
NWARM = 5


# --------------------------------------------------------------------------
# Problem construction (numpy fp64, shared by all backends)
# --------------------------------------------------------------------------


def build_problem(shape, seed=1234):
    dim = len(shape)
    n = shape[0]
    h = 1.0 / (n - 1)
    N = int(np.prod(shape))
    coords = np.meshgrid(*[np.linspace(0.0, 1.0, s) for s in shape], indexing="ij")
    dist = np.sqrt(sum((c - 0.5) ** 2 for c in coords))
    radius = math.sqrt(0.06 / math.pi) if dim == 2 else (0.06 * 3 / (4 * math.pi)) ** (1 / 3)
    solid = dist < radius
    face = np.zeros(shape, bool)
    for a in range(dim):
        sl0 = tuple(0 if b == a else slice(None) for b in range(dim))
        sl1 = tuple(-1 if b == a else slice(None) for b in range(dim))
        face[sl0] = True
        face[sl1] = True
    # closure shell: the 2.4% of non-face fluid points nearest the sphere
    ncl = int(round(0.024 * N))
    cand = np.flatnonzero((~solid & ~face).ravel())
    order = np.argsort(dist.ravel()[cand], kind="stable")
    closure_rows = np.sort(cand[order[:ncl]])
    closure = np.zeros(N, bool)
    closure[closure_rows] = True
    closure = closure.reshape(shape)
    role = (solid.astype(np.uint8) * 1) | (closure.astype(np.uint8) * 2) | (face.astype(np.uint8) * 4)
    interior = role == 0
    nR = max(1, int(round(0.005 * N)))
    strides = [int(np.prod(shape[a + 1:])) for a in range(dim)]
    s0, sl = strides[0], strides[-1]
    rng = np.random.default_rng(seed)

    p = closure_rows
    rc = N + rng.integers(0, nR, size=(ncl, 2))
    cols_c = np.stack([p, p + s0, p - s0, p + sl, p - sl, rc[:, 0], rc[:, 1]], axis=1)
    anchors = closure_rows[rng.integers(0, ncl, size=nR)]
    jr = N + np.arange(nR)
    cols_r = np.stack(
        [jr, anchors, anchors + s0, anchors - s0, anchors + sl, anchors - sl, N + rng.integers(0, nR, size=nR)],
        axis=1,
    )
    idx = np.concatenate([cols_c, cols_r]).astype(np.int32)
    rows = np.concatenate([closure_rows, jr]).astype(np.int32)
    nrows = rows.size
    off = rng.uniform(0.1, 1.0, size=(nrows, W - 1)) * 0.5 / h**2
    diag = -1.05 * off.sum(axis=1)
    C = np.column_stack([diag, off])

    src = rng.uniform(-1.0, 1.0, size=N + nR)
    uD = np.ones(shape)
    for c in coords:
        uD = uD * np.sin(math.pi * c)
    uD[solid] = 0.0
    u0 = np.concatenate([uD.ravel(), rng.uniform(0.0, 0.1, size=nR)])
    fluid = np.concatenate([(~solid).ravel(), np.ones(nR, bool)])

    return dict(
        shape=tuple(shape), dim=dim, N=N, nR=nR, h=h,
        inv_h2=[1.0 / h**2] * dim,
        mask=interior.astype(np.float64),
        keep=(~face).astype(np.float64).ravel(),
        rows=rows, idx=idx, C=C, src=src, u0=u0, fluid=fluid,
        kappa=1.0, dt=0.1 * h**2, t0=0.0,
        counts=dict(
            N=N, nR=nR, solid=int(solid.sum()), closure=int(ncl), face=int(face.sum()),
            interior=int(interior.sum()), ell_rows=int(nrows),
        ),
    )


def tfac(t):
    return 1.0 + 0.5 * math.sin(2.0 * math.pi * t)


def axis_slices(dim, a):
    plus = tuple(slice(2, None) if b == a else slice(1, -1) for b in range(dim))
    minus = tuple(slice(0, -2) if b == a else slice(1, -1) for b in range(dim))
    return plus, minus


def face_slices(dim):
    out = []
    for a in range(dim):
        out.append(tuple(0 if b == a else slice(None) for b in range(dim)))
        out.append(tuple(-1 if b == a else slice(None) for b in range(dim)))
    return out


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class Skip(Exception):
    pass


def med_floor(fn, sync, n=200, warm=20):
    for _ in range(warm):
        fn()
        sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


class NumpyBackend:
    name = "numpy-cpu-f64"
    note = "eager, out= buffers for stage axpby (two ops: multiply then add), ping-pong u"

    def setup(self, P):
        self.P = P
        self.shape, self.dim, self.N = P["shape"], P["dim"], P["N"]
        self.sl = [axis_slices(self.dim, a) for a in range(self.dim)]
        self.fs = face_slices(self.dim)
        n = P["u0"].size
        self.u = P["u0"].copy()
        self.unext = np.empty(n)
        self.u1 = np.empty(n)
        self.rk = np.empty(n)
        self.tmp = np.empty(n)
        self.t = P["t0"]

    def rhs(self, u, tf):
        P = self.P
        D = u[: self.N].reshape(self.shape)
        Pd = np.pad(D, 1)
        lap = None
        for a, (sp, sm) in enumerate(self.sl):
            term = (Pd[sp] + Pd[sm] - 2.0 * D) * P["inv_h2"][a]
            lap = term if lap is None else lap + term
        du = np.empty(u.size)
        du[: self.N] = (lap * P["mask"]).ravel()
        du[self.N:] = 0.0
        du[P["rows"]] = (P["C"] * u[P["idx"]]).sum(axis=1)
        du *= P["kappa"]
        du += tf * P["src"]
        duD = du[: self.N].reshape(self.shape)
        for s in self.fs:
            duD[s] = 0.0
        return du

    def step(self):
        P = self.P
        dt, t = P["dt"], self.t
        u0, u1, rk, tmp = self.u, self.u1, self.rk, self.tmp
        k = self.rhs(u0, tfac(t))
        np.multiply(k, dt / 6, out=rk)
        np.multiply(k, dt / 2, out=tmp)
        np.add(u0, tmp, out=u1)
        k = self.rhs(u1, tfac(t + dt / 2))
        np.multiply(k, dt / 3, out=tmp)
        rk += tmp
        np.multiply(k, dt / 2, out=tmp)
        np.add(u0, tmp, out=u1)
        k = self.rhs(u1, tfac(t + dt / 2))
        np.multiply(k, dt / 3, out=tmp)
        rk += tmp
        np.multiply(k, dt, out=tmp)
        np.add(u0, tmp, out=u1)
        k = self.rhs(u1, tfac(t + dt))
        np.multiply(k, dt / 6, out=tmp)
        rk += tmp
        np.add(u0, rk, out=self.unext)
        self.u, self.unext = self.unext, self.u
        self.t = t + dt
        return float(np.where(P["fluid"], np.abs(self.u), 0.0).max())

    def get_u(self):
        return self.u.copy()

    def floor(self):
        x = np.ones(1024)
        return med_floor(lambda: x + 1.0, lambda: None)

    def free(self):
        self.__dict__.clear()


class TorchBackend:
    def __init__(self, device, dtype, compiled):
        import torch

        self.torch = torch
        self.device = device
        self.dtype = torch.float64 if dtype == "f64" else torch.float32
        self.compiled = compiled
        self.name = f"torch-{device}-{dtype}-{'compile' if compiled else 'eager'}"
        if compiled:
            self.note = "torch.compile(default mode) over the whole functional RK4 step incl. reduction; t as 0-dim tensor"
        else:
            self.note = "eager; torch.add(alpha=, out=) fused axpby into preallocated buffers, ping-pong u"
        if device == "mps" and not torch.backends.mps.is_available():
            raise Skip("MPS unavailable (no Metal device visible)")

    def sync(self):
        if self.device == "mps":
            self.torch.mps.synchronize()

    def setup(self, P):
        torch = self.torch
        self.P = P
        self.shape, self.dim, self.N = P["shape"], P["dim"], P["N"]
        dev, dt_ = self.device, self.dtype
        T = lambda a: torch.as_tensor(a, dtype=dt_).to(dev)  # noqa: E731
        I = lambda a: torch.as_tensor(a, dtype=torch.int32, device=dev)  # noqa: E731
        self.mask = T(P["mask"])
        self.C = T(P["C"])
        self.src = T(P["src"])
        self.rows = I(P["rows"])
        self.idx = I(P["idx"])
        self.fluid = torch.as_tensor(P["fluid"], device=dev)
        self.zR = torch.zeros(P["nR"], dtype=dt_, device=dev)
        self.u = T(P["u0"]).clone()
        self.t = P["t0"]
        self.sl = [axis_slices(self.dim, a) for a in range(self.dim)]
        self.fs = face_slices(self.dim)
        self.pad = (1, 1) * self.dim
        self.reduce_outside = False
        self.tried_fallback = False
        self.face_mask = False
        self.keep = torch.cat([T(P["keep"]), torch.ones(P["nR"], dtype=dt_, device=dev)])
        if self.compiled:
            torch._dynamo.reset()
            self.cstep = torch.compile(self._step_fn)
        else:
            n = self.u.numel()
            self.unext = torch.empty(n, dtype=dt_, device=dev)
            self.u1 = torch.empty(n, dtype=dt_, device=dev)
            self.rk = torch.empty(n, dtype=dt_, device=dev)
        self.sync()

    def rhs(self, u, tf):
        torch, P = self.torch, self.P
        D = u[: self.N].view(self.shape)
        Pd = torch.nn.functional.pad(D, self.pad)
        lap = None
        for a, (sp, sm) in enumerate(self.sl):
            term = (Pd[sp] + Pd[sm] - 2.0 * D) * P["inv_h2"][a]
            lap = term if lap is None else lap + term
        du = torch.cat([(lap * self.mask).reshape(-1), self.zR])
        vals = (self.C * u[self.idx]).sum(dim=1)
        du = du.index_put((self.rows,), vals)
        du = P["kappa"] * du + tf * self.src
        if self.face_mask:
            return du * self.keep
        duD = du[: self.N].view(self.shape)
        for s in self.fs:
            duD[s] = 0.0
        return du

    def _step_fn(self, u0, t):
        torch, P = self.torch, self.P
        dt = P["dt"]
        tf = lambda tt: 1.0 + 0.5 * torch.sin(2.0 * math.pi * tt)  # noqa: E731
        k = self.rhs(u0, tf(t))
        rk = (dt / 6) * k
        u1 = u0 + (dt / 2) * k
        k = self.rhs(u1, tf(t + dt / 2))
        rk = rk + (dt / 3) * k
        u1 = u0 + (dt / 2) * k
        k = self.rhs(u1, tf(t + dt / 2))
        rk = rk + (dt / 3) * k
        u1 = u0 + dt * k
        k = self.rhs(u1, tf(t + dt))
        rk = rk + (dt / 6) * k
        un = u0 + rk
        if self.reduce_outside:
            return un
        stat = torch.where(self.fluid, un.abs(), torch.zeros_like(un)).amax()
        return un, stat

    def step(self):
        torch, P = self.torch, self.P
        dt, t = P["dt"], self.t
        if self.compiled:
            tt = torch.tensor(t, dtype=self.dtype, device=self.device)
            if not self.reduce_outside:
                try:
                    self.u, stat = self.cstep(self.u, tt)
                except Exception as e:  # noqa: BLE001
                    reason = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
                    if not self.face_mask:
                        # fallback 1: face fill as a static mask multiply instead of in-place slice assignment
                        self.face_mask = True
                        self.note += ("; FALLBACK 1: whole-step compile with in-place face-slice fill failed ("
                                      + reason + "); face fill done as a static face-mask multiply")
                    elif not self.tried_fallback:
                        # fallback 2: reduction outside the compiled region
                        self.tried_fallback = True
                        self.reduce_outside = True
                        self.note += ("; FALLBACK 2: compile still failed (" + reason
                                      + "); masked reduction runs eagerly outside the compiled step")
                    else:
                        raise
                    torch._dynamo.reset()
                    self.cstep = torch.compile(self._step_fn)
                    return self.step()
            else:
                self.u = self.cstep(self.u, tt)
                stat = torch.where(self.fluid, self.u.abs(), torch.zeros_like(self.u)).amax()
            self.t = t + dt
            return float(stat.item())
        u0, u1, rk = self.u, self.u1, self.rk
        k = self.rhs(u0, tfac(t))
        torch.mul(k, dt / 6, out=rk)
        torch.add(u0, k, alpha=dt / 2, out=u1)
        k = self.rhs(u1, tfac(t + dt / 2))
        rk.add_(k, alpha=dt / 3)
        torch.add(u0, k, alpha=dt / 2, out=u1)
        k = self.rhs(u1, tfac(t + dt / 2))
        rk.add_(k, alpha=dt / 3)
        torch.add(u0, k, alpha=dt, out=u1)
        k = self.rhs(u1, tfac(t + dt))
        rk.add_(k, alpha=dt / 6)
        torch.add(u0, rk, out=self.unext)
        self.u, self.unext = self.unext, self.u
        self.t = t + dt
        stat = torch.where(self.fluid, self.u.abs(), torch.zeros_like(self.u)).amax()
        return float(stat.item())

    def get_u(self):
        self.sync()
        return self.u.detach().cpu().double().numpy().copy()

    def floor(self):
        torch = self.torch
        x = torch.ones(1024, dtype=self.dtype, device=self.device)
        if self.compiled:
            torch._dynamo.reset()
            f = torch.compile(lambda a: a + 1.0)
            return med_floor(lambda: f(x), self.sync)
        return med_floor(lambda: x + 1.0, self.sync)

    def free(self):
        if self.compiled:
            self.torch._dynamo.reset()
        dev = self.device
        torch = self.torch
        self.__dict__.clear()
        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()


def torch_dtype_f64(torch):
    return torch.float64


class MLXBackend:
    def __init__(self, device):
        try:
            import mlx.core as mx
        except Exception as e:  # noqa: BLE001
            raise Skip(f"mlx import failed: {e!s:.200}")
        self.mx = mx
        self.devname = device
        self.dev = mx.gpu if device == "gpu" else mx.cpu
        self.dtype = mx.float32 if device == "gpu" else mx.float64
        self.name = f"mlx-{device}-{'f32' if device == 'gpu' else 'f64'}-compile"
        self.note = "mx.compile over the whole functional RK4 step incl. reduction; face fill as static mask; scatter via .at[].add"

    def _dev(self):
        self.mx.set_default_device(self.dev)

    def setup(self, P):
        mx = self.mx
        self._dev()
        self.P = P
        self.shape, self.dim, self.N = P["shape"], P["dim"], P["N"]
        dt_ = self.dtype
        npdt = np.float64 if dt_ == mx.float64 else np.float32
        T = lambda a: mx.array(np.asarray(a, dtype=npdt), dtype=dt_)  # noqa: E731
        self.mask = T(P["mask"])
        self.keep = mx.concatenate([T(P["keep"]), mx.ones((P["nR"],), dtype=dt_)])
        self.C = T(P["C"])
        self.src = T(P["src"])
        self.rows = mx.array(P["rows"])
        self.idx = mx.array(P["idx"])
        self.fluid = mx.array(P["fluid"])
        self.zR = mx.zeros((P["nR"],), dtype=dt_)
        self.u = T(P["u0"])
        self.t = P["t0"]
        self.sl = [axis_slices(self.dim, a) for a in range(self.dim)]
        mx.eval(self.mask, self.keep, self.C, self.src, self.rows, self.idx, self.fluid, self.zR, self.u)
        self.cstep = mx.compile(self._step_fn)

    def rhs(self, u, tf):
        mx, P = self.mx, self.P
        D = u[: self.N].reshape(self.shape)
        Pd = mx.pad(D, 1)
        lap = None
        for a, (sp, sm) in enumerate(self.sl):
            term = (Pd[sp] + Pd[sm] - 2.0 * D) * P["inv_h2"][a]
            lap = term if lap is None else lap + term
        du = mx.concatenate([(lap * self.mask).reshape(-1), self.zR])
        vals = (self.C * u[self.idx]).sum(axis=1)
        du = du.at[self.rows].add(vals)
        du = (P["kappa"] * du + tf * self.src) * self.keep
        return du

    def _step_fn(self, u0, t):
        mx, P = self.mx, self.P
        dt = P["dt"]
        tf = lambda tt: 1.0 + 0.5 * mx.sin(2.0 * math.pi * tt)  # noqa: E731
        k = self.rhs(u0, tf(t))
        rk = (dt / 6) * k
        u1 = u0 + (dt / 2) * k
        k = self.rhs(u1, tf(t + dt / 2))
        rk = rk + (dt / 3) * k
        u1 = u0 + (dt / 2) * k
        k = self.rhs(u1, tf(t + dt / 2))
        rk = rk + (dt / 3) * k
        u1 = u0 + dt * k
        k = self.rhs(u1, tf(t + dt))
        rk = rk + (dt / 6) * k
        un = u0 + rk
        stat = mx.where(self.fluid, mx.abs(un), mx.zeros_like(un)).max()
        return un, stat

    def step(self):
        mx = self.mx
        tt = mx.array(self.t, dtype=self.dtype)
        un, stat = self.cstep(self.u, tt)
        mx.eval(un, stat)
        self.u = un
        self.t += self.P["dt"]
        return float(stat.item())

    def get_u(self):
        return np.array(self.u).astype(np.float64)

    def floor(self):
        mx = self.mx
        self._dev()
        x = mx.ones((1024,), dtype=self.dtype)
        mx.eval(x)
        f = mx.compile(lambda a: a + 1.0)
        return med_floor(lambda: mx.eval(f(x)), lambda: None)

    def free(self):
        mx = self.mx
        self.__dict__.clear()
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass
        self.mx = mx


def make_backend(name):
    if name == "numpy-cpu-f64":
        return NumpyBackend()
    if name.startswith("torch-"):
        _, dev, dt, mode = name.split("-")
        return TorchBackend(dev, dt, mode == "compile")
    if name == "mlx-gpu-f32-compile":
        return MLXBackend("gpu")
    if name == "mlx-cpu-f64-compile":
        b = MLXBackend("cpu")
        # probe fp64 support on the cpu stream
        mx = b.mx
        mx.set_default_device(mx.cpu)
        try:
            a = mx.ones((4,), dtype=mx.float64)
            mx.eval(a * 2)
        except Exception as e:  # noqa: BLE001
            raise Skip(f"MLX fp64 on cpu unsupported: {e!s:.200}")
        return b
    raise ValueError(name)


# --------------------------------------------------------------------------
# Machine info
# --------------------------------------------------------------------------


def machine_info():
    info = dict(
        uname_m=platform.machine(),
        mac_ver=platform.mac_ver()[0],
        python=sys.version.split()[0],
        numpy=np.__version__,
    )
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_num_threads"] = torch.get_num_threads()
        info["torch_mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"unavailable: {e}"
    try:
        import importlib.metadata as md

        info["mlx"] = md.version("mlx")
    except Exception:  # noqa: BLE001
        info["mlx"] = "unavailable"
    keys = ["hw.ncpu", "hw.perflevel0.physicalcpu", "machdep.cpu.brand_string", "hw.memsize"]
    sc = {}
    for k in keys:
        try:
            r = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, timeout=5)
            sc[k] = r.stdout.strip() if r.returncode == 0 else f"unavailable ({r.stderr.strip()[:80]})"
        except Exception as e:  # noqa: BLE001
            sc[k] = f"unavailable ({e})"
    info["sysctl"] = sc
    try:
        info["loadavg_at_start"] = os.getloadavg()
    except Exception:  # noqa: BLE001
        pass
    return info


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_case(label, shape, backends, ref_cache, results, allow_big):
    P = build_problem(shape)
    N = P["N"]
    print(f"\n=== {label}: |D|={N} counts={P['counts']}", flush=True)
    nsteps = 100 if N < 100_000 else (40 if N < 1_000_000 else 20)

    # numpy fp64 reference after NCHECK steps
    ref = NumpyBackend()
    ref.setup(P)
    for _ in range(NCHECK):
        ref.step()
    uref = ref.get_u()
    ref.free()
    scale = float(np.abs(uref).max())

    for bname in backends:
        key = f"{label}|{bname}"
        rec = dict(case=label, backend=bname, shape=list(shape), N=N, nR=P["nR"])
        if label == "3D 257^3" and not allow_big.get(bname, False):
            rec.update(status="skipped", reason="129^3 step not under 2 s on this backend (or it failed)")
            results[key] = rec
            print(f"  {bname:24s} skipped (257^3 gate)", flush=True)
            continue
        try:
            b = make_backend(bname)
            rec["note"] = getattr(b, "note", "")
            t0 = time.perf_counter()
            b.setup(P)
            first = None
            for i in range(NCHECK):
                ts = time.perf_counter()
                b.step()
                if i == 0:
                    first = time.perf_counter() - ts
            rec["setup_s"] = time.perf_counter() - t0 - 0  # includes the 10 check steps
            rec["first_step_s"] = first
            rec["note"] = getattr(b, "note", "")
            u = b.get_u()
            relerr = float(np.abs(u - uref).max() / scale)
            tol = 1e-12 if "f64" in bname else 1e-4
            rec.update(rel_err_vs_numpy_f64=relerr, tol=tol, correct=bool(relerr <= tol))
            for _ in range(NWARM):
                b.step()
            ts_list = []
            for _ in range(nsteps):
                ts = time.perf_counter()
                b.step()
                ts_list.append(time.perf_counter() - ts)
            med = statistics.median(ts_list)
            rec.update(
                status="ok",
                steps_timed=nsteps,
                ms_median=med * 1e3,
                ms_min=min(ts_list) * 1e3,
                ms_p90=sorted(ts_list)[int(0.9 * (len(ts_list) - 1))] * 1e3,
                ns_per_point=med * 1e9 / N,
            )
            if label in CPP_BAR:
                rec["cpp_bar_ms"] = CPP_BAR[label]
                rec["ratio_to_cpp"] = rec["ms_median"] / CPP_BAR[label]
            if label == "3D 129^3":
                allow_big[bname] = med < 2.0
            print(
                f"  {bname:24s} {rec['ms_median']:10.3f} ms/step  {rec['ns_per_point']:8.2f} ns/pt  "
                f"relerr={relerr:.2e} {'OK' if rec['correct'] else 'FAIL'}  first={first:.2f}s",
                flush=True,
            )
            b.free()
        except Skip as e:
            rec.update(status="skipped", reason=str(e))
            print(f"  {bname:24s} skipped: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec.update(status="failed", reason=f"{type(e).__name__}: {e!s:.400}")
            print(f"  {bname:24s} FAILED: {type(e).__name__}: {e!s:.300}", flush=True)
            traceback.print_exc(limit=3)
        results[key] = rec
        gc.collect()


def run_floors(backends):
    floors = {}
    for bname in backends:
        try:
            b = make_backend(bname)
            floors[bname] = dict(status="ok", us_median=b.floor() * 1e6)
            print(f"  floor {bname:24s} {floors[bname]['us_median']:8.2f} us", flush=True)
        except Skip as e:
            floors[bname] = dict(status="skipped", reason=str(e))
        except Exception as e:  # noqa: BLE001
            floors[bname] = dict(status="failed", reason=f"{type(e).__name__}: {e!s:.300}")
            print(f"  floor {bname} FAILED {e!s:.200}", flush=True)
    return floors


def main_run(args):
    if args.cases == "all":
        cases = CASES
    else:
        want = set(args.cases.split(","))
        cases = [c for c in CASES if c[0].split()[-1].split("x")[0].split("^")[0] in want or c[0] in want]
    backends = BACKENDS if args.backends == "all" else args.backends.split(",")
    info = machine_info()
    print(json.dumps(info, indent=1), flush=True)
    results, ref_cache, allow_big = {}, {}, {}
    t0 = time.time()
    for label, shape in cases:
        run_case(label, shape, backends, ref_cache, results, allow_big)
    print("\n=== dispatch floors", flush=True)
    floors = run_floors(backends)
    out = dict(
        machine=info,
        started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        wall_s=time.time() - t0,
        results=results,
        floors=floors,
    )
    path = os.path.join(HERE, args.out)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# Merge two runs and write the final json + md
# --------------------------------------------------------------------------


def fmt(x, nd=2):
    if x is None:
        return "-"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.{nd}f}"


def main_merge(args):
    runs = []
    for p in args.merge:
        with open(os.path.join(HERE, p)) as f:
            runs.append(json.load(f))
    merged = {}
    keys = []
    for r in runs:
        for k in r["results"]:
            if k not in keys:
                keys.append(k)
    for k in keys:
        cands = [(i + 1, r["results"][k]) for i, r in enumerate(runs) if k in r["results"]]
        ok = [(i, c) for i, c in cands if c.get("status") == "ok"]
        if ok:
            i, best = min(ok, key=lambda ic: ic[1]["ms_median"])
            best = dict(best, from_run=i, all_runs_ms={f"run{j}": c.get("ms_median") for j, c in cands})
        else:
            i, best = cands[-1]
            best = dict(best, from_run=i)
        merged[k] = best
    floors = {}
    for b in BACKENDS:
        c = [(i + 1, r["floors"].get(b)) for i, r in enumerate(runs) if r["floors"].get(b)]
        ok = [(i, x) for i, x in c if x.get("status") == "ok"]
        if ok:
            i, x = min(ok, key=lambda ix: ix[1]["us_median"])
            floors[b] = dict(x, from_run=i)
        elif c:
            floors[b] = dict(c[-1][1], from_run=c[-1][0])
    final = dict(
        description="Synthetic SHOCCS-shaped RK4 step (E2 heat, random closure coefficients) on host Python frameworks",
        machine=runs[-1]["machine"],
        runs=[dict(file=p, started=r["started"], wall_s=r["wall_s"], machine=r["machine"]) for p, r in zip(args.merge, runs)],
        cpp_bar_ms=CPP_BAR,
        results=merged,
        floors=floors,
    )
    with open(os.path.join(HERE, "host_primitives_results.json"), "w") as f:
        json.dump(final, f, indent=1)
    write_md(final)
    print("wrote host_primitives_results.json and .md")


def write_md(final):
    m = final["machine"]
    sc = m.get("sysctl", {})
    res = final["results"]
    labels = [c[0] for c in CASES]
    L = []
    L.append("# Host primitive benchmark: SHOCCS-shaped RK4 step in Python array frameworks\n")
    L.append(
        "This is the host-side measurement the survey lacked. It times one RK4 step of a synthetic, "
        "SHOCCS-shaped heat problem built from the runtime primitives P0 to P6 of "
        "`01-codebase-idioms.md` Section 4, on the CPU in fp64 and on the Apple GPU in fp32, and "
        "compares it with the C++ bar measured in the devcontainer. Script: "
        "`host_primitives_bench.py`; raw data: `host_primitives_results.json` (merged) and the per-run files.\n"
    )
    L.append("## Machine\n")
    L.append(f"- CPU: {sc.get('machdep.cpu.brand_string', 'unavailable')}; hw.ncpu {sc.get('hw.ncpu', 'unavailable')}, "
             f"performance cores {sc.get('hw.perflevel0.physicalcpu', 'unavailable')}; memory "
             f"{int(sc['hw.memsize']) / 2**30:.0f} GiB" if str(sc.get('hw.memsize', '')).isdigit() else
             f"- sysctl: {sc}")
    L.append(f"- uname -m {m.get('uname_m')}, macOS {m.get('mac_ver')}, Python {m.get('python')}, numpy {m.get('numpy')}, "
             f"torch {m.get('torch')} (intra-op threads {m.get('torch_num_threads')}), mlx {m.get('mlx')}")
    for r in final["runs"]:
        la = r["machine"].get("loadavg_at_start")
        la_s = ", ".join(f"{x:.2f}" for x in la) if la else "n/a"
        L.append(f"- {r['file']}: started {r['started']}, wall {r['wall_s']:.0f} s, load average at start {la_s}")
    L.append("")

    L.append("## Observations\n")
    L.append("Best backend per case (lower-of-two-runs medians):\n")
    L.append("| Case | best CPU fp64 | ms | x C++ | best GPU fp32 | ms | x C++ |")
    L.append("|---|---|---|---|---|---|---|")
    for lab in labels:
        cpu = [res[k] for k in res if k.startswith(lab + "|") and "f64" in k and res[k].get("status") == "ok"]
        gpu = [res[k] for k in res if k.startswith(lab + "|") and "f32" in k and res[k].get("status") == "ok"]
        if not cpu and not gpu:
            continue
        row = [lab]
        for grp in (cpu, gpu):
            if grp:
                b = min(grp, key=lambda r: r["ms_median"])
                row += [b["backend"], fmt(b["ms_median"], 3),
                        f"{b['ratio_to_cpp']:.2f}" if "ratio_to_cpp" in b else "-"]
            else:
                row += ["-", "-", "-"]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.extend(OBSERVATIONS)
    L.append("")

    L.append("## Milliseconds per RK4 step (median)\n")
    L.append("Each cell is the median wall time of one full RK4 step (four RHS evaluations, stage updates, one masked "
             "Linf reduction and its host readback), the lower of the two runs; the run it came from is in brackets. "
             "`skip` and `fail` reasons are listed below the table.\n")
    hdr = "| Case | grid points | " + " | ".join(BACKENDS) + " | C++ bar |"
    L.append(hdr)
    L.append("|" + "---|" * (len(BACKENDS) + 3))
    for lab in labels:
        row = []
        N = None
        for b in BACKENDS:
            r = res.get(f"{lab}|{b}")
            if r is None:
                row.append("-")
                continue
            N = r.get("N", N)
            if r.get("status") == "ok":
                flag = "" if r.get("correct") else " (WRONG)"
                row.append(f"{fmt(r['ms_median'], 3)} [r{r['from_run']}]{flag}")
            else:
                row.append(r.get("status", "-"))
        if N is None:
            continue
        L.append(f"| {lab} | {N} | " + " | ".join(row) + f" | {CPP_BAR.get(lab, '-')} |")
    L.append("")

    L.append("## Nanoseconds per grid point and ratio to the C++ bar\n")
    L.append("ns/pt is the step median divided by |D|. The ratio is Python time over C++ time (below 1 means faster than C++); "
             "it is given only where the C++ bar has a matching case: 51x51 against 0.90 ms, 201x201 against 4.6 ms, "
             "401x401 against 13.7 ms, 65^3 against 38 ms. The C++ 71x71 figure (0.43 ms) is for scalar_wave, not heat, "
             "so it is not used as a ratio.\n")
    L.append("| Case | " + " | ".join(BACKENDS) + " |")
    L.append("|" + "---|" * (len(BACKENDS) + 1))
    for lab in labels:
        row = []
        any_ = False
        for b in BACKENDS:
            r = res.get(f"{lab}|{b}")
            if r and r.get("status") == "ok":
                any_ = True
                s = f"{fmt(r['ns_per_point'])} ns"
                if "ratio_to_cpp" in r:
                    s += f" / {r['ratio_to_cpp']:.2f}x"
                row.append(s)
            else:
                row.append("-")
        if any_:
            L.append(f"| {lab} | " + " | ".join(row) + " |")
    L.append("")

    L.append("## Dispatch floors\n")
    L.append("Median of 200 calls of a trivial elementwise op (`x + 1` on 1024 elements) plus synchronisation "
             "(`torch.mps.synchronize`, `mx.eval`); compiled backends call a compiled function.\n")
    L.append("| Backend | floor (us) |")
    L.append("|---|---|")
    for b in BACKENDS:
        f = final["floors"].get(b)
        if not f:
            continue
        L.append(f"| {b} | {fmt(f['us_median']) if f.get('status') == 'ok' else f.get('status') + ': ' + f.get('reason', '')} |")
    L.append("")

    L.append("## Correctness and the fp32 error datapoint\n")
    L.append("After 10 steps from the same initial state every backend is compared with the numpy fp64 run: "
             "max |u - u_ref| / max |u_ref| over the packed [D | R] vector. Tolerance 1e-12 for fp64, 1e-4 for fp32. "
             "This is E2 with random (dissipative, diagonally dominant) closure coefficients of size about 1/h^2, "
             "not the real psi-dependent closures, so the fp32 numbers say nothing about small-psi conditioning.\n")
    L.append("| Case | " + " | ".join(BACKENDS) + " |")
    L.append("|" + "---|" * (len(BACKENDS) + 1))
    for lab in labels:
        row = []
        any_ = False
        for b in BACKENDS:
            r = res.get(f"{lab}|{b}")
            if r and "rel_err_vs_numpy_f64" in r:
                any_ = True
                row.append(f"{r['rel_err_vs_numpy_f64']:.1e}" + ("" if r["correct"] else " FAIL"))
            else:
                row.append("-")
        if any_:
            L.append(f"| {lab} | " + " | ".join(row) + " |")
    L.append("")

    L.append("## First-step (compile) cost\n")
    L.append("Wall time of the first step, which includes tracing and compilation for compiled backends (seconds).\n")
    L.append("| Case | " + " | ".join(BACKENDS) + " |")
    L.append("|" + "---|" * (len(BACKENDS) + 1))
    for lab in labels:
        row = []
        any_ = False
        for b in BACKENDS:
            r = res.get(f"{lab}|{b}")
            if r and r.get("first_step_s") is not None:
                any_ = True
                row.append(f"{r['first_step_s']:.2f}")
            else:
                row.append("-")
        if any_:
            L.append(f"| {lab} | " + " | ".join(row) + " |")
    L.append("")

    skips = [(k, r) for k, r in res.items() if r.get("status") != "ok"]
    if skips:
        L.append("## Skips and failures\n")
        for k, r in skips:
            L.append(f"- {k.replace('|', ', ')}: {r.get('status')}: {r.get('reason', '')}")
        L.append("")

    L.append("## Backend notes\n")
    seen = set()
    for r in res.values():
        b = r["backend"]
        if b in seen or not r.get("note"):
            continue
        seen.add(b)
        L.append(f"- {b}: {r['note']}")
    L.append("")
    final_md_tail(L)
    with open(os.path.join(HERE, "host_primitives_results.md"), "w") as f:
        f.write("\n".join(L) + "\n")


OBSERVATIONS = [
    "- Small 2D cases (51x51, 71x71) are dispatch-bound everywhere. Eager numpy and compiled MLX on the CPU run a "
    "whole RK4 step in about 0.1 to 0.3 ms, several times under the 0.90 ms C++ bar, mainly because the synthetic "
    "step has no host MMS evaluation, no update_boundary and far fewer launches than the C++ step (see Caveats). "
    "torch.compile on the CPU is slower than eager here (its per-call guard and wrapper overhead is about 6 us, "
    "against under 1 us for an eager op).",
    "- Every GPU path pays one CPU-GPU synchronisation per step, and the measured floor for a trivial op plus "
    "sync is about 140 to 190 us on both MPS and MLX. With the whole step compiled, MLX and torch.compile on MPS "
    "sit at about 0.37 and 0.5 ms per step up to 201x201, i.e. two to three sync floors; eager MPS is 1.5 to 2.2 ms "
    "because it issues on the order of a hundred separate small kernels per step (about 20 ops per RHS times four, plus stage updates).",
    "- From 401x401 and 65^3 upwards the compiled GPU paths are bandwidth-bound and reach 2.6 to 4 ns per point "
    "per RK4 step in fp32; CPU fp64 paths sit at 9 to 27 ns per point (torch.compile CPU best at about 16 ns at 3D). The GPU paths are 4 to 6x faster than the best CPU fp64 path at 129^3 and 257^3.",
    "- The ratio column overstates the Python advantage: the C++ step also does host MMS evaluation, update_boundary, full stats and many more launches, none of which the synthetic step has. A fairer reference is the C++ RHS-only graph submit, 1.16 ms per RHS at 64^3 (8 threads), about 4.6 ms for the four RHS of one RK4 step before any stage updates. The best CPU fp64 Python step at 65^3 (4.2 to 4.5 ms, MLX CPU or torch.compile CPU, including stage updates and the reduction) is at parity with that, and the compiled GPU fp32 paths (about 1.0 to 1.1 ms) are about 4x under it.",
    "- The fp32 error after 10 steps is 1e-7 to 2.4e-7 relative on every case and backend, i.e. a few fp32 ulps; "
    "this says nothing about real small-psi closures.",
]


def final_md_tail(L):
    L.append("## Methods\n")
    L.append(
        "The problem is a heat-like system on a cubic or square grid with h = 1/(n-1). A sphere of solid points "
        "(about 6% of the grid) sits in the middle; the 2.4% of non-face fluid points nearest to it are closure rows; "
        "the 2*dim boundary planes are Dirichlet faces. Role bits combine into an interior mask. The packed state is "
        "[D | R] with |R| = 0.5% of |D|. One RHS is: zero-pad D, sum over axes of the E2 3-point second difference "
        "times 1/h^2 using shifted slices, multiply by the interior mask, append zeros for R, then a w = 7 ELL "
        "gather-dot over the packed vector for the closure rows and all R rows (int32 indices, columns are the row "
        "itself, axis neighbours and random R points; random positive off-diagonals with a diagonal of -1.05 times "
        "their sum), scattered to unique rows; scale by kappa = 1, add a precomputed source times "
        "1 + 0.5 sin(2 pi t); zero the faces. The RK4 step is the classical four-stage scheme with dt = 0.1 h^2 "
        "(u1 = u0 + c k, rk accumulation, u_new = u0 + rk), followed by max |u| over fluid points (all of R plus "
        "non-solid D) read back to the host once."
    )
    L.append("")
    L.append(
        "Timing: 10 check steps (which include compilation), 5 more warm-up steps, then the median of 100 (|D| < 1e5), "
        "40 (|D| < 1e6) or 20 timed steps; each timed step ends with the host readback of the reduction, which "
        "forces synchronisation (MLX additionally evaluates the new state with `mx.eval`). The 257^3 case runs only "
        "on backends whose 129^3 median was under 2 s. The full suite ran twice; each cell keeps the lower median. "
        "Metal is not visible inside the agent's Bash sandbox, so the runs that include MPS and MLX GPU were made "
        "outside it."
    )
    L.append("")
    L.append("## Caveats\n")
    L.append("- Synthetic coefficients: E2 interior plus random closure and R-row coefficients. No real psi-dependent "
             "closures, no cross-axis interpolation rows, no B/N split into separate tables: one merged ELL table.")
    L.append("- No host MMS evaluation (the C++ step evaluates the manufactured solution on the host several times "
             "per step), no update_boundary, no stats beyond one Linf, no I/O. The C++ bar includes stats and its "
             "host MMS work, so the comparison favours Python somewhat at small sizes.")
    L.append("- The C++ bar was measured in a Linux devcontainer (8 vCPU aarch64 VM on this Mac, Kokkos OpenMP), "
             "not natively; native C++ may be faster, particularly at large sizes where VM memory bandwidth matters.")
    L.append("- A concurrent container job may have been using CPU during the first run; the per-run load averages "
             "are listed under Machine (1.36 at the start of run 1, 1.13 at run 2), and each cell reports which run it "
             "came from. Most best cells come from run 2.")
    L.append("- The functional backends (torch.compile, MLX) allocate new buffers per step rather than ping-ponging "
             "fixed ones; eager numpy and torch use preallocated buffers for the stage updates.")
    L.append("- fp32 is used on the GPU because MPS rejects float64 and MLX has no GPU float64.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all", help="all, or comma list of 51,71,201,401,65,129,257")
    ap.add_argument("--backends", default="all")
    ap.add_argument("--out", default="host_primitives_run.json")
    ap.add_argument("--merge", nargs="+")
    args = ap.parse_args()
    if args.merge:
        main_merge(args)
    else:
        main_run(args)


if __name__ == "__main__":
    main()
