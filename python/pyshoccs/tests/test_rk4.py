"""The RK4 step on one-point ODEs: amplification polynomial and stage times."""

import numpy as np

from pyshoccs.heat import rk4_step_generic, RK_C, RK_W


def step(rhs, u0, t, dt, update_boundary=lambda u, t: None):
    n = u0.size
    out, k, rk, tmp = (np.empty(n) for _ in range(4))
    return rk4_step_generic(rhs, update_boundary, u0, t, dt, out, k, rk, tmp).copy()


def test_linear_amplification_polynomial():
    lam, dt = -2.3, 0.37
    z = lam * dt

    def rhs(u, t, out):
        out[:] = lam * u

    u1 = step(rhs, np.array([1.0]), 0.0, dt)
    poly = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    assert abs(u1[0] - poly) <= 4e-16 * abs(poly) + 1e-16


def test_stage_times_make_cubic_quadrature_exact():
    """u' = t^2 integrates exactly under RK4 (Simpson weights at 0, 1/2, 1/2, 1)."""
    dt, t0 = 0.8, 0.3

    def rhs(u, t, out):
        out[:] = t * t

    u1 = step(rhs, np.array([0.0]), t0, dt)
    exact = ((t0 + dt) ** 3 - t0**3) / 3
    assert abs(u1[0] - exact) <= 1e-15


def test_boundary_update_runs_before_stages_one_to_three_and_after():
    calls = []

    def rhs(u, t, out):
        out[:] = 0.0

    def ub(u, t):
        calls.append(t)
        u[:] = t

    u1 = step(rhs, np.array([5.0]), 1.0, 0.5, ub)
    assert calls == [1.0 + 0.5 * RK_C[1], 1.0 + 0.5 * RK_C[2], 1.0 + 0.5 * RK_C[3], 1.5]
    assert u1[0] == 1.5
    assert abs(sum(RK_W) - 1.0) < 1e-16
