"""The gaussian manufactured solution family, vectorised over points.

Matches ``src/mms/gauss2d.cpp`` and ``gauss3d.cpp``. Per term i,

    value     = A_i exp(-0.5 sum_a (x_a - c_ia)^2 / v_ia^2) cos(f_i t)
    ddt       = -A_i f_i exp(...) sin(f_i t)
    gradient  = -value (x_a - c_ia) / v_ia^2
    laplacian = value sum_a ((x_a - c_ia)^2 - v_ia^2) / v_ia^4

summed over the terms, with a running over the first ``dims`` coordinates. The
C++ laplacian is the algebraically expanded polynomial form of the same
expression, so agreement is to rounding (about 1e-14 relative).

``Gaussian.spatial`` exposes the time-independent factors G_i(x) = A_i exp(...)
and Q_i(x) = sum_a ((x_a - c_ia)^2 - v_ia^2) / v_ia^4, which the heat system
hoists out of the time loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Gaussian:
    dims: int
    center: np.ndarray
    variance: np.ndarray
    amplitude: np.ndarray
    frequency: np.ndarray

    def __post_init__(self):
        self.center = _pad3(np.asarray(self.center, dtype=np.float64), 0.0)
        self.variance = _pad3(np.asarray(self.variance, dtype=np.float64), 1.0)
        self.amplitude = np.atleast_1d(np.asarray(self.amplitude, dtype=np.float64))
        self.frequency = np.atleast_1d(np.asarray(self.frequency, dtype=np.float64))
        n = self.center.shape[0]
        if not (self.variance.shape[0] == n == self.amplitude.size == self.frequency.size):
            raise ValueError("gaussian: center, variance, amplitude and frequency lengths differ")
        if self.dims not in (1, 2, 3):
            raise ValueError(f"gaussian: dims must be 1, 2 or 3, got {self.dims}")

    @classmethod
    def from_meta(cls, meta: dict) -> "Gaussian":
        m = meta["mms"]
        kind = str(m.get("type", "gaussian")).lower()
        if kind != "gaussian":
            raise ValueError(f"manufactured solution type {kind!r} is not supported here")
        return cls(int(meta["dims"]), m["center"], m["variance"], m["amplitude"], m["frequency"])

    @property
    def n_terms(self) -> int:
        return int(self.center.shape[0])

    # -- time-independent factors ----------------------------------------------

    def _pts(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(-1, 1) if self.dims == 1 else p.reshape(1, -1)
        if p.shape[1] < self.dims:
            raise ValueError(f"points have {p.shape[1]} coordinates, need {self.dims}")
        return p

    def spatial(self, pts) -> tuple[np.ndarray, np.ndarray]:
        """``(G, Q)`` of shape (n_terms, N): G_i = A_i exp(...), Q_i the laplacian factor."""
        p = self._pts(pts)
        n = self.n_terms
        G = np.empty((n, p.shape[0]))
        Q = np.empty((n, p.shape[0]))
        for i in range(n):
            arg = np.zeros(p.shape[0])
            q = np.zeros(p.shape[0])
            for a in range(self.dims):
                dx = p[:, a] - self.center[i, a]
                v = self.variance[i, a]
                arg += -0.5 * np.power(dx, 2) * np.power(v, -2.0)
                q += (np.power(dx, 2) - v * v) / (v ** 4)
            G[i] = np.exp(arg) * self.amplitude[i]
            Q[i] = q
        return G, Q

    # -- the four fields -------------------------------------------------------

    def value(self, t: float, pts) -> np.ndarray:
        G, _ = self.spatial(pts)
        return np.cos(t * self.frequency) @ G

    def ddt(self, t: float, pts) -> np.ndarray:
        G, _ = self.spatial(pts)
        return (-self.frequency * np.sin(t * self.frequency)) @ G

    def laplacian(self, t: float, pts) -> np.ndarray:
        G, Q = self.spatial(pts)
        return np.cos(t * self.frequency) @ (G * Q)

    def gradient(self, t: float, pts) -> np.ndarray:
        """(N, 3); components beyond ``dims`` are zero."""
        p = self._pts(pts)
        G, _ = self.spatial(p)
        c = np.cos(t * self.frequency)
        out = np.zeros((p.shape[0], 3))
        for i in range(self.n_terms):
            for a in range(self.dims):
                dx = p[:, a] - self.center[i, a]
                out[:, a] += -G[i] * c[i] * dx * np.power(self.variance[i, a], -2.0)
        return out

    def source(self, t: float, pts, diffusivity: float) -> np.ndarray:
        """The heat source ``ddt - k laplacian``."""
        return self.ddt(t, pts) - diffusivity * self.laplacian(t, pts)


class HoistedGaussian:
    """Value and heat source at a fixed point set with the time factors hoisted.

    ``value(t, out)`` writes ``sum_i cos(f_i t) G_i`` and ``source(t, out)``
    writes ``sum_i (-f_i sin(f_i t) G_i - k cos(f_i t) G_i Q_i)`` into ``out``
    with one small matrix-vector product each and no per-call allocation
    beyond the coefficient vector.
    """

    def __init__(self, ms: Gaussian, pts, diffusivity: float):
        G, Q = ms.spatial(pts)
        self.f = ms.frequency.copy()
        self.k = float(diffusivity)
        self.G = np.ascontiguousarray(G)
        self.M = np.ascontiguousarray(np.concatenate([G, G * Q], axis=0))
        self.n = int(G.shape[1])
        self._c = np.empty(2 * self.f.size)

    def value(self, t: float, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty(self.n)
        np.dot(np.cos(t * self.f), self.G, out=out)
        return out

    def source(self, t: float, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty(self.n)
        nt = self.f.size
        c = self._c
        c[:nt] = -self.f * np.sin(t * self.f)
        c[nt:] = -self.k * np.cos(t * self.f)
        np.dot(c, self.M, out=out)
        return out


def _pad3(a: np.ndarray, fill: float) -> np.ndarray:
    a = np.atleast_2d(a)
    if a.shape[1] > 3:
        raise ValueError("gaussian parameters have more than three coordinates")
    if a.shape[1] < 3:
        pad = np.full((a.shape[0], 3 - a.shape[1]), fill)
        a = np.concatenate([a, pad], axis=1)
    return np.ascontiguousarray(a)
