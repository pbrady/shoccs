# Cross-cutting criteria: differentiability, mixed precision, distributed memory

Summarised in Part II, [02-framework-survey.md](02-framework-survey.md) section 9.9, alongside the other comparators.

| | |
|---|---|
| Verdict | reference_only. None of the three criteria is decisive; differentiability is a modest tie-breaker for JAX. |
| Fit for SHOCCS | Moderate for differentiability (speculative future need), low for mixed precision, low for distributed memory. |
| CPU fp64 parity | Available in every shortlisted option; score 6/10. |
| Apple GPU | No fp64 on any path, differentiable or not; score 3/10. |
| Risk | 6/10. Enzyme+Kokkos, DaCe AD maturity and the fp32 performance claims are unproven. |

Verifier-adjusted overall score: 5/10 (unchanged after verification).

## Status as of 2026-10-12

| Item | Version / state | Verification |
|---|---|---|
| JAX | 0.11.2, uploaded 2026-09-17 | confirmed (PyPI JSON) |
| Devito | 4.8.23, 2026-08-07 | version and date confirmed; MPI support not re-verified in this pass |
| Numba | 0.68.0 listed as latest; upload date 2026-09-30 not read | partly unverified |
| PyTorch | 2.14.1 latest; upload date not seen | confirmed (version only) |
| DaCe | stable 1.0.2 (2025-03-20); pre-release 2.0.0a10 (2026-10-03) | confirmed from PyPI |
| metaljax | 0.11.9 (2026-09-29), experimental, requires jax>=0.11,<0.12, tested with 0.11.2 | confirmed |
| Enzyme (EnzymeAD/Enzyme) | issue #1650 opened 2024-01-31, still open | confirmed |
| Enzyme-JAX | reported pushed to May 2026 by an aggregator | unverified |
| Halide AD | background knowledge only | unverified |

## Backends and fp64

| Backend | fp64 | Differentiation | Distributed |
|---|---|---|---|
| CPU, all options | Yes (JAX needs jax_enable_x64; PyTorch and Numba fp64 for float64 arrays) | JAX and PyTorch native forward and reverse AD. Numba has none built in (a 2023 Numba-Enzyme prototype, current state unknown). Devito has no general AD but supports hand-derived adjoint operators (unverified). DaCe AD is reverse-mode with ILP checkpointing (arXiv 2509.02197); stable-release status unestablished. | Devito MPI; Numba plus mpi4py for explicit halos; Kokkos+MPI is the incumbent |
| CUDA | Yes | JAX, PyTorch, DaCe, Devito cover it | Mature (NCCL, JAX sharding, Devito MPI plus OpenACC); not relevant to the Apple-first goal |
| Apple GPU (Metal, MPS) | No. PyTorch MPS rejects float64 and MLX fp64 is CPU only (per 00-apple-gpu-facts.md; not re-verified in this pass). metaljax documents f64 compute as unsupported, with METALJAX_F64=downcast emulating in f32. | Autograd works on MPS for supported ops, in fp32 or lower | Effectively N/A |

A differentiable Apple GPU run is therefore fp32 or df64 at best, with closure rows evaluated in fp64 on the host. High-order cut-cell closures have small-psi cancelling coefficients (E4_1 clamps psi to [1e-4, 1-1e-4]), so fp32 closure rows are not acceptable.

## Mapping for the hard runtime shapes

| Shape | Differentiability | MPI | Support |
|---|---|---|---|
| Parameter BO (alpha, epsilon, tension) | Mostly eigenvalue metrics of small matrices; gradients via JAX/PyTorch on coefficient assembly or SymPy (already in the pipeline) | n/a | workaround |
| Packed field vector [D, Rx, Ry, Rz] | Single pytree leaf, ideal | Needs an ownership map per rank for R segments | workaround |
| Masked axis stencil | Differentiable via slices and where; transpose is the reversed-coefficient stencil | Halo exchange on the decomposed axis | native |
| ELL gather-dot-scatter | Gather is differentiable, VJP is scatter-add (small cost at 0.5 to 2.5% of points) | Rows crossing rank boundaries need duplicated halo points; Devito does not model it | workaround |
| Face-plane assign | Indexed update (JAX .at[].set, PyTorch index_put) | Boundary ranks only | native |
| Masked reduction | sum with where | allreduce; accumulate in fp64 | native |
| Whole-step compilation | JAX jit plus scan plus jax.checkpoint gives bounded-memory reverse mode; PyTorch has no MPS graph capture | Devito compiles the time loop including halos | workaround |
| Psi-dependent closure (I34 MMS, I-series) | Setup-time derivative, already done symbolically in SymPy; runtime AD only needed for shape optimisation | n/a | workaround |
| Distributed domain decomposition overall | n/a | Devito first-class but a poor fit for cut-cell rows; JAX shard_map needs hand-written exchange for ELL; PyTorch is collectives only; DaCe distributed-SDFG unverified | unsupported |

## Code sketch

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def rhs(u, p):              # p = (alpha, ...) stencil parameters
    d2 = (u[2:] - 2*u[1:-1] + u[:-2]) * p["inv_h2"]      # axis stencil
    out = jnp.zeros_like(u).at[1:-1].set(d2)
    # ELL closure: idx (nr,w), coef(alpha) (nr,w), rows (nr,)
    c = p["coef0"] + p["alpha"] * p["coef1"]
    clos = jnp.sum(c * u[p["idx"]], axis=1)               # gather-dot
    return out.at[p["rows"]].set(clos)                    # scatter

def rk4(u, p, dt):
    k1 = rhs(u,p); k2 = rhs(u+dt/2*k1,p)
    k3 = rhs(u+dt/2*k2,p); k4 = rhs(u+dt*k3,p)
    return u + dt/6*(k1+2*k2+2*k3+k4)

def loss(alpha, u0, p, n, dt):
    p = {**p, "alpha": alpha}
    step = jax.checkpoint(lambda u,_: (rk4(u,p,dt), None))
    u, _ = jax.lax.scan(step, u0, None, length=n)
    return jnp.sum(u**2)

g = jax.jit(jax.grad(loss), static_argnums=(3,))   # d loss / d alpha
```

The sketch was not executed as part of this survey.

## Performance assessment

The C++ bar is measured (`05-measurements.md`, devcontainer, 2026-10-12): the shipped heat.lua RK4 step is 0.90 ms at 51x51 (59% of it host manufactured-solution evaluation), the fixed-up step is estimated at about 0.15 ms, and the 65^3 RHS graph submit is 798 us at 8 threads and 1867 us at 1 thread. The forward Python side is measured on a synthetic step (`05-measurements.md` Section D, same M4): 0.085 to 0.15 ms per fused fp64 CPU step at 51x51 against 91 us for four C++ RHS submits; 4.2 to 4.5 ms at 65^3 against 3.2 ms (8 threads) and 7.5 ms (1 thread); the fp32 GPU step 0.35 to 0.5 ms at 51x51 (sync floor 167 to 181 us), 1.0 ms at 65^3 and 43.6 ms at 257^3. Everything below about reverse mode, fp32 storage and MPI is still inference, not measurement.

- Small 2D (51^2): JAX jit plus scan adds no forward cost; JAX was not in the Section D run, but the comparable fused CPU fp64 paths there sit at the discretization bar. Reverse is roughly 2 to 4x forward, plus checkpoint recomputation, so a reverse step would be about 0.2 to 0.6 ms against the fixed-up estimate of 0.15 ms. PyTorch eager reverse is dispatch-bound (0.75 us per eager op measured forward), likely 5 to 10x slower than the C++ bar. A Devito adjoint is comparable to forward speed but not general.
- Large 3D (65^3, measured RHS 798 us at 8 threads and 1867 us at 1 thread): forward is bandwidth-bound and the measured compiled CPU fp64 paths from Python are within 1.4x of the eight-thread submits. Reverse-mode memory is the limiter without checkpointing (4 stages x steps x 4 fields).
- fp32 storage at 64^3: probably a small gain, since the roughly 2 MB per field state is likely cache resident on M-class chips. At 256^3 (134 MB per buffer) it might give 1.6 to 2x on DRAM-bound sweeps, but fp32 RK4 accumulation floors accuracy near 1e-7, below the 1e-10 or better needed for E4/E6/E8 convergence studies. The one measured datapoint is consistent with that floor: the fp32 GPU paths in `05-measurements.md` Section D disagree with fp64 by 1e-7 to 2.4e-7 relative after 10 steps on synthetic E2 closures. No external evidence either way.
- MPI: on one machine at 64^3 a decomposition adds halo cost and gains nothing. At 256^3 it matters only for multi-node memory.

## What the DSL designs should copy or avoid

Copy:

1. Keep all irregularity in setup-time tables so every runtime kernel is a pure function of arrays. AD then becomes a transposition rule: the axis stencil transpose is a reversed-coefficient stencil, and the ELL gather-dot transpose is scatter-add over the same table.
2. Declare each kernel with an explicit VJP/adjoint rule and a per-array dtype policy (storage dtype, accumulation dtype, closure rows always fp64).
3. Carry a decomposition-ready layout: explicit halo width per axis and an ownership map for R segments.
4. Support checkpointing at the step boundary for reverse mode.

Avoid:

- Promising fp32 storage as a default.
- Building a distributed runtime or a general AD engine in the first version.
- Devito-style generate-everything MPI for the irregular rows.

Keep Kokkos+MPI as the escape hatch.

## Deciding fact

None of the three criteria is decisive for the shortlist. Only differentiability separates options: JAX is the only one with whole-step reverse mode through gather and scatter on CPU fp64 today, and that is a nice-to-have. fp32 storage is rejected on accuracy grounds, and MPI is not needed for this group. The criteria give a modest tie-breaker to JAX and to a thin-waist DSL with built-in VJP rules, and do not rescue Metal or PyTorch MPS.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| JAX 0.11.2 uploaded 2026-09-17 | confirmed | PyPI JSON; whether a newer release exists since was not checked |
| Devito 4.8.23 on 2026-08-07 with MPI | confirmed (version and date only) | PyPI JSON; MPI support not re-verified here |
| Numba 0.68.0 on 2026-09-30 | uncertain | Version confirmed as latest; date unreadable; no-built-in-AD not re-verified |
| PyTorch 2.14.1 | confirmed (version only) | PyPI JSON; upload date not seen |
| Enzyme+Kokkos not an established path | confirmed, corrected | Issue #1650 (opened 2024-01-31, still open). Correction: the reporter differentiated a plain for-loop; the issue does not clearly show parallel_for working. Parallel_reduce failed on Kokkos internals. Canonical repo is EnzymeAD/Enzyme. |
| DaCe AD and distributed SDFG | uncertain | AD (reverse mode, ILP checkpointing) supported by arXiv 2509.02197; distributed-SDFG unverified; AD in a stable release not established |
| metaljax fp64 undocumented | refuted | Documented: f64 compute unsupported, ops in f64 fail at compile time, METALJAX_F64=downcast emulates in f32, keep jax_enable_x64 off. Reinforces the no-fp64 conclusion. |
| No Apple GPU path offers fp64 | uncertain | Consistent with the metaljax README; PyTorch MPS and MLX statements not re-verified |
| fp32 gain small at 64^3; fp32 RK4 floors near 1e-7 | uncertain | Self-declared inference, no measurement |
| Halide AD, Devito adjoint pattern, Enzyme-JAX activity | unverified | Background knowledge or aggregator only; not counted below |

Counts: 4 confirmed (JAX, Devito version, PyTorch version, Enzyme+Kokkos), 1 refuted (metaljax fp64), 4 uncertain (Numba date, DaCe, Apple no-fp64, fp32 inference).

## Sources

- https://pypi.org/pypi/jax/json
- https://pypi.org/pypi/devito/json
- https://pypi.org/pypi/numba/json
- https://pypi.org/pypi/torch/json
- https://pypi.org/pypi/dace/json
- https://pypi.org/pypi/metaljax/json
- https://arxiv.org/html/2509.02197v1
- https://github.com/EnzymeAD/Enzyme/issues/1650
- https://pretalx.enzyme.csail.mit.edu/enzymecon-2023/talk/9BHKSP
- https://gittrend.io/repo/EnzymeAD/Enzyme-JAX (aggregator, unverified)
- docs/python-dsl-survey/00-apple-gpu-facts.md
- docs/python-dsl-survey/01-codebase-idioms.md
