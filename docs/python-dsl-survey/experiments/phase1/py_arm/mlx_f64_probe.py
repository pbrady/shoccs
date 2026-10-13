"""Which MLX CPU operations keep float64? Compare each op of the bench step with NumPy float64.

Run outside the sandbox (mlx.core needs a Metal device even for the CPU stream):
  uv run --project python/pyshoccs python .scratch/phase1/py_arm/mlx_f64_probe.py
"""

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
rng = np.random.default_rng(1)
n = 4096
u = rng.uniform(-1, 1, n)
M = rng.uniform(-1, 1, (n, 2))
c = rng.uniform(-1, 1, 2)
idx = rng.integers(0, n, (300, 12))
rows = np.unique(rng.integers(0, n, 300))
vals = rng.uniform(-1, 1, rows.size)
t = 0.7234


def T(a):
    return mx.array(np.asarray(a, dtype=np.float64), dtype=mx.float64)


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


cases = {}
x = T(u)
cases["multiply by python float"] = (x * 0.0333333, u * 0.0333333)
cases["add"] = (x + x * 0.5, u + u * 0.5)
cases["cos"] = (mx.cos(x), np.cos(u))
cases["sin"] = (mx.sin(x), np.sin(u))
cases["exp"] = (mx.exp(x), np.exp(u))
cases["abs"] = (mx.abs(x), np.abs(u))
cases["matmul (n,2)@(2,)"] = (T(M) @ T(c), M @ c)
cases["matmul (2,n)@(n,)"] = (T(M.T) @ x, M.T @ u)
cases["gather 2d index"] = (x[mx.array(idx)], u[idx])
cases["gather-dot sum(axis=1)"] = ((T(M[idx, 0]) * x[mx.array(idx)]).sum(axis=1), (M[idx, 0] * u[idx]).sum(axis=1))
z = mx.zeros((n,), dtype=mx.float64)
cases["scatter-add .at[].add"] = (z.at[mx.array(rows)].add(T(vals)), np.add.at(np.zeros(n), rows, vals) or np.bincount(rows, vals, n))
cases["pad"] = (mx.pad(x.reshape(64, 64), [(1, 1), (1, 1)])[1:-1, 1:-1].reshape(-1), u)
cases["concatenate"] = (mx.concatenate([x, z]), np.concatenate([u, np.zeros(n)]))
cases["reshape slices (stencil)"] = (
    (x.reshape(64, 64)[:, :-2] - 2 * x.reshape(64, 64)[:, 1:-1] + x.reshape(64, 64)[:, 2:]).reshape(-1),
    (u.reshape(64, 64)[:, :-2] - 2 * u.reshape(64, 64)[:, 1:-1] + u.reshape(64, 64)[:, 2:]).reshape(-1))
f = mx.compile(lambda a: mx.cos(a) * 0.5 + a)
cases["compiled cos*0.5+x"] = (f(x), np.cos(u) * 0.5 + u)
g = mx.compile(lambda a, s: mx.cos(a * s))
cases["compiled cos(x*t) with scalar array"] = (g(x, mx.array(t, dtype=mx.float64)), np.cos(u * t))
cases["cos(x*t) with scalar array, eager"] = (mx.cos(x * mx.array(t, dtype=mx.float64)), np.cos(u * t))
cases["scalar array t itself"] = (mx.array(t, dtype=mx.float64), np.array(t))
cases["f * t (1-vector times 0-d)"] = (T([0.5]) * mx.array(t, dtype=mx.float64), np.array([0.5 * t]))

for name, (a, b) in cases.items():
    mx.eval(a)
    print(f"{name:42s} dtype {str(a.dtype):16s} rel err {rel(a, b):.2e}")
