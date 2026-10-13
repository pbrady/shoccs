"""The heat system on the packed vector V, mirroring ``src/systems/heat.cpp``.

- ``source(t)``: ``ddt - k lap`` of the manufactured solution at V locations;
- ``rhs(u, t)``: ``k (A u) + src(t)`` on fluid D points and non-Dirichlet R
  points, then 0 on Dirichlet grid faces and Dirichlet-object R points (the fill
  is applied last, as ``heat::build_rhs_graph`` does);
- ``update_boundary(u, t)``: Dirichlet grid faces and Dirichlet-object R points
  set to the manufactured solution at t;
- ``stats(u, t)``: the eleven numbers of ``compute_scalar_stats``;
- ``rk4_step(u0, t, dt)``: exactly ``integrators::rk4``.

Two operator backends: ``scipy`` (one CSR matvec over V) and ``structured``
(``structure.StructuredOperator``). Work vectors are preallocated so a step
allocates nothing beyond small coefficient arrays.
"""

from __future__ import annotations

import numpy as np

from .mms import Gaussian, HoistedGaussian
from .structure import Structure, StructuredOperator, _csr_matvec_factory
from .tables import Tables

RK_C = (0.0, 0.5, 0.5, 1.0)
RK_W = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
STATS_NAMES = ("Linf", "min", "max", "domain_Linf", "domain_ic",
               "Rx_Linf", "Rx_ic", "Ry_Linf", "Ry_ic", "Rz_Linf", "Rz_ic")


class ScipyOperator:
    """``out = A u`` with the summed CSR over V."""

    name = "scipy"

    def __init__(self, tables: Tables):
        self.A = tables.to_scipy()
        self.n_V = tables.layout.n_V
        self._mv = _csr_matvec_factory(self.A)

    def apply(self, u: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty(self.n_V)
        self._mv(u, out)
        return out


def make_operator(tables: Tables, backend: str, structure: Structure | None = None):
    if backend == "scipy":
        return ScipyOperator(tables)
    if backend == "structured":
        op = StructuredOperator(tables, structure)
        op.name = "structured"
        return op
    raise ValueError(f"unknown operator backend {backend!r} (scipy or structured)")


def rk4_step_generic(rhs, update_boundary, u0: np.ndarray, t: float, dt: float,
                     out: np.ndarray, k: np.ndarray, rk: np.ndarray, tmp: np.ndarray) -> np.ndarray:
    """The classical RK4 step of ``src/temporal/rk4.cpp`` on preallocated buffers.

    ``rhs(u, t, out)`` writes the system RHS into ``out``; ``update_boundary(u, t)``
    edits ``u`` in place. Stage i > 0 evaluates ``u0 + dt c_i k`` after a
    boundary update; the final state is ``u0 + sum_i dt w_i k_i`` followed by
    ``update_boundary(t + dt)``.
    """
    rk.fill(0.0)
    k.fill(0.0)
    np.copyto(out, u0)
    for i in range(4):
        if i > 0:
            np.multiply(k, dt * RK_C[i], out=tmp)
            np.add(u0, tmp, out=out)
            update_boundary(out, t + dt * RK_C[i])
        rhs(out, t + dt * RK_C[i], k)
        np.multiply(k, dt * RK_W[i], out=tmp)
        rk += tmp
    np.add(u0, rk, out=out)
    update_boundary(out, t + dt)
    return out


class HeatSystem:
    def __init__(self, tables: Tables, backend: str = "scipy", hoist: bool = True,
                 structure: Structure | None = None, mms: Gaussian | None = None):
        if tables.has_neumann:
            raise NotImplementedError(
                f"grid_bcs {tables.grid_bcs}: Neumann walls are not supported by pyshoccs.heat")
        self.tables = tables
        self.L = tables.layout
        self.n_V = self.L.n_V
        self.n_D = self.L.n_D
        self.k = tables.diffusivity
        self.t0 = tables.t0
        self.dt = tables.dt
        self.backend = backend
        self.hoist = hoist
        self.mms = mms if mms is not None else Gaussian.from_meta(tables.meta)
        self.op = make_operator(tables, backend, structure)
        loc = tables.locations
        self.loc = loc

        fluid = tables.fluid_idx.astype(np.int64)
        nondir = [tables.r_packed(d, tables.R[d].non_dirichlet_idx) for d in range(3)]
        dirr = [tables.r_packed(d, tables.R[d].dirichlet_idx) for d in range(3)]
        self.fluid_idx = fluid
        self.nondir_R = nondir
        self.dir_R = dirr
        # rows that receive the source (fluid D, then non-Dirichlet Rx, Ry, Rz)
        self.src_idx = np.concatenate([fluid] + nondir)
        # rows filled by the boundary update (Dirichlet faces, then Dirichlet R)
        self.bc_idx = np.concatenate([tables.dirichlet_face_idx.astype(np.int64)] + dirr)
        # segment boundaries inside src_idx, reused by stats
        self._seg_end = np.cumsum([fluid.size] + [a.size for a in nondir])

        self.src_eval = HoistedGaussian(self.mms, loc[self.src_idx], self.k)
        self.bc_eval = HoistedGaussian(self.mms, loc[self.bc_idx], self.k)
        self.all_eval = None

        n = self.n_V
        self._src = np.empty(self.src_idx.size)
        self._bc = np.empty(self.bc_idx.size)
        self._stat_u = np.empty(self.src_idx.size)
        self._stat_sol = np.empty(self.src_idx.size)
        self._stat_err = np.empty(self.src_idx.size)
        self._k = np.zeros(n)
        self._rk = np.zeros(n)
        self._tmp = np.empty(n)
        self._gather = np.empty(self.src_idx.size)

    # -- manufactured solution -------------------------------------------------

    def solution(self, t: float) -> np.ndarray:
        """MMS value at every V location (allocates; for checks, not the hot path)."""
        return self.mms.value(t, self.loc)

    def source(self, t: float) -> np.ndarray:
        """MMS source ``ddt - k lap`` at every V location (allocates)."""
        return self.mms.source(t, self.loc, self.k)

    def source_selected(self, t: float, out: np.ndarray | None = None) -> np.ndarray:
        """The source on ``src_idx`` only, hoisted unless ``hoist=False``."""
        out = self._src if out is None else out
        if self.hoist:
            self.src_eval.source(t, out)
        else:
            out[:] = self.mms.source(t, self.loc[self.src_idx], self.k)
        return out

    def boundary_values(self, t: float, out: np.ndarray | None = None) -> np.ndarray:
        out = self._bc if out is None else out
        if self.hoist:
            self.bc_eval.value(t, out)
        else:
            out[:] = self.mms.value(t, self.loc[self.bc_idx])
        return out

    def initialize(self, t: float | None = None) -> np.ndarray:
        """``heat::initialize`` then ``update_boundary``: MMS on fluid D and all R, 0 on solid."""
        t = self.t0 if t is None else t
        sol = self.solution(t)
        u = np.zeros(self.n_V)
        u[self.fluid_idx] = sol[self.fluid_idx]
        u[self.n_D:] = sol[self.n_D:]
        self.update_boundary(u, t)
        return u

    # -- the system ------------------------------------------------------------

    def apply_operator(self, u: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        return self.op.apply(u, out)

    def rhs(self, u: np.ndarray, t: float, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty(self.n_V)
        self.op.apply(u, out)
        out *= self.k
        self.source_selected(t)
        out[self.src_idx] += self._src
        out[self.bc_idx] = 0.0
        return out

    def update_boundary(self, u: np.ndarray, t: float) -> np.ndarray:
        self.boundary_values(t)
        u[self.bc_idx] = self._bc
        return u

    def rk4_step(self, u0: np.ndarray, t: float, dt: float, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty(self.n_V)
        return rk4_step_generic(self.rhs, self.update_boundary, u0, t, dt,
                                out, self._k, self._rk, self._tmp)

    def stats(self, u: np.ndarray, t: float) -> list[float]:
        """``compute_scalar_stats``: Linf, min, max, domain Linf and index, then per R segment."""
        if self.hoist:
            self.src_eval.value(t, self._stat_sol)
        else:
            self._stat_sol[:] = self.mms.value(t, self.loc[self.src_idx])
        np.take(u, self.src_idx, out=self._stat_u)
        np.subtract(self._stat_u, self._stat_sol, out=self._stat_err)
        np.abs(self._stat_err, out=self._stat_err)
        ends = self._seg_end
        nf = int(ends[0])
        if nf > 0:
            e = self._stat_err[:nf]
            j = int(np.argmax(e))
            err_d = float(e[j])
            idx_d = float(self.fluid_idx[j])
            u_min = float(self._stat_u[:nf].min())
            u_max = float(self._stat_u[:nf].max())
        else:
            err_d, idx_d, u_min, u_max = 0.0, 0.0, 0.0, 0.0
        comp_err = [0.0, 0.0, 0.0]
        comp_idx = [0.0, 0.0, 0.0]
        for d in range(3):
            a, b = int(ends[d]), int(ends[d + 1])
            if b == a:
                continue
            e = self._stat_err[a:b]
            j = int(np.argmax(e))
            comp_err[d] = float(e[j])
            comp_idx[d] = float(self.tables.R[d].non_dirichlet_idx[j])
            u_min = min(u_min, float(self._stat_u[a:b].min()))
            u_max = max(u_max, float(self._stat_u[a:b].max()))
        err = max(err_d, comp_err[0], comp_err[1], comp_err[2])
        return [err, u_min, u_max, err_d, idx_d,
                comp_err[0], comp_idx[0], comp_err[1], comp_idx[1], comp_err[2], comp_idx[2]]

    def stats_named(self, u: np.ndarray, t: float) -> dict[str, float]:
        return dict(zip(STATS_NAMES, self.stats(u, t)))
