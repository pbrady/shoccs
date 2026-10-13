# JAX (XLA / PJRT) as a SHOCCS solver backend or DSL substrate

Survey with adversarial verification applied. Status date: 2026-10-12. Version checked: jax/jaxlib 0.11.2 (released 2026-09-17).

## Verdict

| Item | Verdict |
|---|---|
| Fit for SHOCCS | Strong candidate for a CPU/CUDA reference backend and DSL lowering target; all 51 catalog idioms map (unstructured parts become host-built tables). |
| CPU parity | Plausibly meets or beats C++ (unverified: local synthetic benchmarks only; needs a real-geometry check on the same machine). |
| Apple GPU | Poor. jax-metal is abandoned; jax-mps is an Alpha, single-maintainer, fp32-only plugin, measured no faster than CPU fp64. |
| fp64 | Native on CPU and CUDA/ROCm (enable via `jax.enable_x64`). None on any Apple path. |
| Risk | Medium. Fusion behaviour (barriers, scatter copies) is heuristic; the Apple goal rests on a hobby plugin. Verifier-adjusted overall score: 6.5 / 10. |

## Status as of 2026-10-12

- Latest release: jax/jaxlib 0.11.2 (2026-09-17). Monthly cadence: 0.10.0 (Apr 16), 0.10.1 (May 20), 0.10.2 (Jun 17), 0.11.0 (Jul 16), 0.11.1 (Aug 17), 0.11.2 (Sep 17). The 0.9.x dates (Feb-Mar 2026) were not re-verified. Whether a later release exists on PyPI was not checked.
- `jax.experimental.sparse` is officially "no longer being actively developed" and "not recommended for use in performance-critical applications" (confirmed in docs). All sparse coupling must be hand-written gathers plus `segment_sum` or `.at[].add`.
- `jax.experimental.enable_x64` was deprecated in 0.8.0 in favour of the `jax.enable_x64` context manager (confirmed).
- jaxlib 0.11.2 wheels ship C++ FFI extension headers. Correction: the changelog names `collectives.h` and `record.h` for out-of-tree plugins; this is narrower than "general custom-call FFI headers", which shipped earlier. `jax.ffi` itself is an established feature.
- Apple Metal: jax-metal 0.1.1 (2024-10-08) is the last release. A tracking issue (#26968) reported incompatibility with jax>=0.5.1 in March 2025. On 2026-01-20 a JAX maintainer closed it saying the team is no longer working on it, and mentions were removed from the install docs. Nit: the "needs jax<=0.5.0" limit is a community workaround (jax/jaxlib 0.5.0), not a declared pin; PyPI metadata says jaxlib>=0.4.34 and mentions `ENABLE_PJRT_COMPATIBILITY=1`.
- Community jax-mps 0.11.0 (2026-09-15, classifier Alpha, MLX-based PJRT plugin): about 95.2% pass on JAX upstream tests with exclusions (float64, sub-byte and 8-bit floats, multi-device, Pallas/Mosaic). CI only checks compilation because hosted runners lack Apple GPUs. StableHLO bytecode targets jaxlib 0.11.x.

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| XLA:CPU | Production | Native (`jax_enable_x64`) | Multithreaded; full op coverage incl. gather/scatter, `while`/`fori_loop`/`scan`, `optimization_barrier`, `pure_callback`. |
| XLA:GPU CUDA | Production (flagship) | Native, hardware rate | Gather/scatter lower to native kernels; executable replayed as CUDA graph/thunks. Not benchmarked here. |
| XLA:GPU ROCm | Production | Native | Not evaluated. |
| TPU | Production | Emulated, slow | Not relevant. |
| jax-metal (Apple, MPSGraph) | Dead | No (float64/complex unsupported; exact error string unverified) | Do not use. |
| jax-mps 0.11.0 (community, MLX) | Alpha | No ("float64 not supported on Metal") | Single maintainer, pinned to jaxlib 0.11.x. Works with jax 0.11.2 in a local test only. |
| Pallas | GPU/TPU kernels | n/a | On CPU runs only in interpret mode; unsupported on jax-mps. |
| `jax.ffi` | CPU/CUDA custom calls | n/a | Escape hatch for hand-written C++ kernels; not on jax-mps. |

Any Apple-GPU run is fp32 only unless double-float emulation is hand-written in `jnp` (estimated 4-10x cost, unverified). The psi-closure precision question under fp32 is framework-independent.

## Idiom mapping

Support values: native, workaround. "Setup only" means the work happens once on the host in NumPy and does not affect step performance.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | Pytree/NamedTuple state {D:(nx,ny,nz), Rx, Ry, Rz}; slots become values flowing through the jitted step. Optional packing into one 1-D array with static offsets. | Zero cost. |
| I02 | native | Host NumPy: vectorised quadratic roots, filter, `np.lexsort` to (slow,fast,t), emitting SoA arrays. Snap/clamp/nextafter must be reproduced bit-exactly. | Setup only; far faster than per-ray virtual calls. |
| I03 | native | Host NumPy fragment table (start_ic, len, stride, end kinds) and dirichlet_line mask; used to build masks/ELL tables. | Setup only. |
| I04 | native | Host NumPy bool masks (fluid, dir, obj_dir) uploaded once as jit args or closed-over constants. | Setup only; a mask beats a 63 MB int32 gather list at 256^3. |
| I05 | native | Python floats baked in as compile-time constants (interior taps, uniform closures, RBF solve via numpy.linalg). | Free; folds into fused kernels. |
| I06 | native | `sympy.lambdify(psi, F, 'numpy')` or a retargeted CSE printer over a psi[nR] array in fp64 on host. | Setup; one batched call. |
| I07 | workaround | Host Python/Numba with `np.searchsorted` on per-pencil fragment tables instead of the linear interp_line scan; emits padded (idx, coef) ELL rows. Data-dependent branching does not belong in jit. | Setup only; faster than O(abs(R)) scan. |
| I08 | native | Host NumPy: group lines by (left_kind, right_kind, bc); emit closure ELL tables (row_ids, idx, C) plus masks. | Setup only. |
| I09 | workaround | Host NumPy produces Bf/Br as ELL over a concatenated [D;R(r)] source, padding idx to a dummy slot with C=0. | Setup only. |
| I10 | native | `np.lexsort` or `scipy.sparse.coo_array`; keep compact COO/ELL at runtime, never full-height row pointers. | Setup only. |
| I11 | native | `np.cumsum`/`np.concatenate` plus deduplicated closure tables; one device upload per table. | Setup only. |
| I12 | native | Python config (dict/TOML/Python) replaces Lua; coefficient fields and IC built with jnp/NumPy. | Setup only. |
| I13 | native | `D.reshape(nx,ny,nz)` is free (C order matches). Per-axis ops via `lax.slice_in_dim`/`jnp.roll`/`lax.conv`; coordinates via `jnp.meshgrid`/broadcasting. | Free; XLA picks loop order. |
| I14 | native | R order fixed by host lexsort; all tables indexed consistently. | N/A (correctness contract). |
| I15 | native | Functional values replace spans and scratch vectors; in-place behaviour via `donate_argnums` at the step boundary. | Neutral to positive. |
| I16 | workaround | No ragged strided-segment primitive. Reformulate as a whole-grid axis stencil (I17), overwrite closure rows with the ELL gather-dot-scatter (I18/I19), then apply masks. vmap over a padded (n_lines, Lmax) gather is possible but wasteful. | Plausibly faster than C++ (coalesced whole-grid stencil); measured 1.6 ns/pt fused 3-axis RHS at 64^3 (unverified local). |
| I17 | native | Sum of shifted `lax.slice_in_dim` slices times constant taps, `jnp.pad` (or `lax.conv_general_dilated`). XLA fuses axes into one loop fusion. | Bandwidth-bound; measured 0.8 ns/pt at 256^3 (13 ms) without scatters (unverified local). |
| I18 | native | `jnp.einsum('rt,t...->r...', Cwall, D[:T])` on first/last T planes then `dynamic_update_slice`/`.at[:R].set`, or fold wall rows into the closure ELL. | Negligible. |
| I19 | native | ELL gather-dot-scatter: `duf.at[rows].set(jnp.sum(C*flat[idx],1), unique_indices=True)`. | Each scatter breaks fusion and adds a D-sized copy; measured 13 to 35 ms at 256^3 with closure + B scatters; merge into one table. |
| I20 | workaround | Avoid `jax.experimental.sparse`. Use `x[idx] * w` then `segment_sum(num_segments=static, indices_are_sorted=True)` or `.at[rows].add`. | Fused gather + scatter inside one executable. |
| I21 | native | `duf.at[B_rows].add(B_w * R[B_cols])` with compact COO of about R*abs(R) entries. | Avoids the 134 MB row-pointer sweep at 256^3. |
| I22 | native | Broadcast axpy on X slabs or `where` on a static face mask; vanishes from the trace without Neumann BC. | Negligible. |
| I23 | native | `dR = jnp.sum(Bf_C * src[Bf_idx], axis=1)` with `src = concatenate([D.ravel(), R])`; padded ELL. | Tiny; fuses. |
| I24 | native | Merged into the I23 ELL via the concatenated source; no separate chain. | Free. |
| I25 | native | Disappears: the output is built as an expression. | Eliminated. |
| I26 | native | Multiply in the expression; XLA fuses it. | Eliminated as a pass. |
| I27 | native | `du = Ox(u)+Oy(u)+Oz(u)` plus scatters, functional; eq_t partial coverage made explicit via `where(mask, new, old)` or zero default. | Removes the multi-kernel critical path. |
| I28 | native | Return a tuple or stack on a leading axis (`jnp.stack`, or vmap over direction with static tables). | No scratch fields when fused with I29. |
| I29 | native | `gGx*dux + gGy*duy + gGz*duz` in jnp. | Free under fusion. |
| I30 | native | `jnp.where(fluid_mask, du + src, du)`; `.at[obj_idx].add(src_R[obj_idx], unique_indices=True)` for small R sets. | Fused masked FMA; faster than a 63 MB index gather. |
| I31 | native | `jnp.where(dir_mask, 0.0, du)` or `u.at[0,:,:].set(val)`; object Dirichlet via `where(obj_dir_mask, sol_R, uR)`. | Negligible. |
| I32 | native | `jax.jit` of the whole RK4 step (or K steps via `fori_loop`/`scan`) with `donate_argnums`. Put `lax.optimization_barrier` between RK stages (see risks). | Measured 51x51: RHS 8.5-14 us, step 39-74 us (unverified local); C++ step baseline is only an estimate. |
| I33 | native | Plain Python function; eager mode for debugging; fences become `block_until_ready`. | Eager 3.9 us/op on CPU; use jit in production. |
| I34 | native | Closed-form MMS in jnp over broadcast coordinates and an (nR,3) array, t traced; `jax.grad`/`jax.hessian` or sympy-generated jnp for derivatives. | One fused pass replaces about 37k virtual calls per step. |
| I35 | workaround | Lua per-point callbacks replaced by user-written jnp/sympy array expressions. `jax.pure_callback` works but costs about 24 us per call on CPU plus a host round trip on GPU. | Array expression fast; callback is a perf trap. |
| I36 | native | Source is a device-side expression of traced t inside the RHS. | Removes per-stage host sync. |
| I37 | native | `jax.tree.map` axpby inside the jitted step; XLA fuses across leaves. | Fused; no separate launches. |
| I38 | native | Functional state passing with `donate_argnums=0`. | Free. |
| I39 | native | `where(dir_mask, sol(t), u)` on face slices/R points inside jit. | Removes full-field host evals. |
| I40 | native | Hoisted to setup masks. | Eliminated. |
| I41 | native | `jnp.max(jnp.where(fluid, jnp.abs(u-sol), -inf))`, `argmax`, min/max fused; read back every K steps. | One extra pass; readback about one dispatch. |
| I42 | native | Host-side dump check; `np.asarray(u)` (zero-copy on CPU) then raw float64 and XDMF from Python. | Equal or better. |
| I43 | native | Host Python loop around the jitted step or K-step chunk; Euler adaptive dt from a device reduction in loop state. | About 2 us per call; amortised by chunks. |
| I44 | native | Explicit matrix from the ELL/stencil tables via scipy.sparse, or `jax.jacfwd` of the linear RHS (dense, fine to n about 1e4), then `numpy.linalg.eig`. | Diagnostic; jacfwd gives operator export for free. |
| I45 | native | NumPy one-liners as test oracles. | N/A. |
| I46 | native | jnp expressions; XLA fusion replaces expression templates. | Equal or better. |
| I47 | native | Leading component axis (5,nx,ny,nz) with batched stencils/ELL (vmap or broadcasting); physics in jnp. | Good if fusion cooperates; barriers/remat may be needed to avoid huge flux temporaries or recomputation. |
| I48 | native | Unchanged NumPy/SciPy, or `jax.jacfwd` of the real cut-cell RHS. | N/A. |
| I49 | native | Retarget sympy lambdify/printer to 'jax' or 'numpy' modules. | Setup only. |
| I50 | workaround | Stays SciPy fp64 on host. JAX contributes `jax.linearize`/`jvp` as a matrix-free `LinearOperator` or `jacfwd` for dense export. No non-Hermitian sparse eigensolver in JAX. | Host-bound as today. |
| I51 | native | In-process jitted solver replaces subprocess runs; torch/botorch coexist, arrays move via dlpack. | Removes per-point process and Kokkos init cost. |

Idioms left out: none are impossible. The workaround set (I07, I09, I16, I20, I35, I50) all reduce to host-built tables or host SciPy.

## Code sketches

Sketches come from the survey and were not executed as part of this write-up. Treat them as illustrative.

Per-line block matvec with per-line dense closures (I16-I19):

```python
import jax, jax.numpy as jnp
# setup (host NumPy): taps[d] (w,), clos[d] = dict(rows (n,), idx (n,T), C (n,T)) over flat D,
# int32, rows unique; row_mask[d] = rows of direction d that O writes (fluid, non-Dirichlet line)
def axis_stencil(u, ax, taps):
    n, w = u.shape[ax], len(taps); h = w // 2
    acc = sum(c * jax.lax.slice_in_dim(u, j, n - w + 1 + j, axis=ax) for j, c in enumerate(taps))
    pad = [(0, 0)] * u.ndim; pad[ax] = (h, h)
    return jnp.pad(acc, pad)          # interior everywhere; ends/cut rows overwritten next

def deriv(u3, d, taps, clos, row_mask):
    out = axis_stencil(u3, d, taps).ravel()
    f = u3.ravel()
    vals = jnp.sum(clos['C'] * f[clos['idx']], axis=1)   # psi-dependent wall+object rows
    out = out.at[clos['rows']].set(vals, unique_indices=True, indices_are_sorted=True)
    return jnp.where(row_mask.ravel(), out, 0.0).reshape(u3.shape)  # explicit eq_t coverage

def laplacian_D(u3, T):   # T: static per-direction tables (closed over -> constants)
    return sum(deriv(u3, d, *T[d]) for d in range(u3.ndim))
```

O/B CSR cut-cell coupling to R points (I21, I23, I24):

```python
import jax.numpy as jnp
# setup emits compact COO/ELL (int32), padded slots point at a dummy index with coef 0
# B[d]  = (rows (m,), cols (m,), w (m,))              R_d -> D
# BF[d][r] = (idx (nR_r, K), C (nR_r, K))  over src=[D.ravel(); R_r] (K = 12-16 for r!=d)
def cut_coupling(duD, u, d, B, BF):
    rows, cols, w = B[d]
    duD = duD.ravel().at[rows].add(w * u.R[d][cols]).reshape(duD.shape)
    dR = []
    for r in range(3):
        if u.R[r].shape[0] == 0: dR.append(u.R[r]); continue   # static shape -> pruned
        src = jnp.concatenate([u.D.ravel(), u.R[r]])
        idx, C = BF[d][r]
        dR.append(jnp.sum(C * src[idx], axis=1))               # Bf_r + Br_r merged
    return duD, dR
# laplacian: sum over d of D parts and of dR[r] parts; all inside one jit
```

Boundary application: masked source, Dirichlet faces/objects, update_boundary (I30, I31, I39):

```python
import jax.numpy as jnp
# masks built once on host: fluid (nx,ny,nz) bool, dirD (nx,ny,nz) bool,
# objdir[r] (nR_r,) bool, nondir[r] = ~objdir[r]; X,Y,Z broadcast coords; PR[r] (nR_r,3)
def apply_rhs_bcs(du, t, mms):
    sD = mms.source(X, Y, Z, t)                                  # fused closed form
    D = jnp.where(fluid, du.D + sD, du.D)
    D = jnp.where(dirD, 0.0, D)
    R = [jnp.where(nondir[r], du.R[r] + mms.source_pts(PR[r], t), 0.0) for r in range(3)]
    return State(D, R)

def update_boundary(u, t, mms):
    D = u.D
    D = D.at[0].set(mms.value(X[0], Y[0], Z[0], t))                # face slice (P4)
    D = D.at[-1].set(mms.value(X[-1], Y[-1], Z[-1], t))
    D = D.at[:, 0].set(mms.value(X[:, 0], Y[:, 0], Z[:, 0], t))    # boundary-only eval
    R = [jnp.where(objdir[r], mms.value_pts(PR[r], t), u.R[r]) for r in range(3)]
    return State(D, R)
```

RK4 with pytree axpby, donation, K-step chunk (I37, I38, I41):

```python
import jax, jax.numpy as jnp
from functools import partial
ax = lambda a, c, b: jax.tree.map(lambda x, y: x + c * y, a, b)
bar = jax.lax.optimization_barrier          # stop XLA:CPU recomputing nested stencils

def rk4(u, t, dt, P):
    k1 = bar(rhs(u, t, P))
    k2 = bar(rhs(update_boundary(ax(u, dt/2, k1), t + dt/2, P), t + dt/2, P))
    k3 = bar(rhs(update_boundary(ax(u, dt/2, k2), t + dt/2, P), t + dt/2, P))
    k4 = bar(rhs(update_boundary(ax(u, dt, k3), t + dt, P), t + dt, P))
    acc = jax.tree.map(lambda a, b, c, d: a + 2*b + 2*c + d, k1, k2, k3, k4)
    return update_boundary(ax(u, dt/6, acc), t + dt, P)

@partial(jax.jit, static_argnums=3, donate_argnums=0)
def chunk(u, t0, dt, K, P):
    def body(i, c):
        u, _ = c; u = rk4(u, t0 + i*dt, dt, P)
        return u, stats(u, t0 + (i+1)*dt, P)    # masked max|u-sol| etc., stays on device
    return jax.lax.fori_loop(0, K, body, (u, stats(u, t0, P)))
# host: u, s = chunk(u, t, dt, K, P); check float(s.linf) every K steps; dump via np.asarray(u.D)
```

## Performance assessment

All numbers below are local measurements by the survey author (jax 0.11.2, XLA:CPU fp64, 10-core Apple-silicon Mac) on synthetic tables of the right shape (3-point E2 stencils, ELL closures, B/Bf), not a faithful port. The verifier could not confirm any of them from public sources, and the C++ step baseline is an estimate (0.3-1 ms) from a Linux container, about 2x noisy. Treat as indicative, not established.

Small 2D (51x51, 71x71):

| Quantity | JAX (local) | C++ reference |
|---|---|---|
| RHS at 51x51 | 14 us | about 30-40 us |
| RHS at 71x71 | 23 us | n/a |
| Full RK4 step, 51x51 | 66-74 us | estimated 300-1000 us |
| Full RK4 step, 71x71 | 124-131 us | n/a |
| Stencil+mask-only step with barriers, 51x51 | 39 us | n/a |

The whole step is one executable with about 2 us dispatch and a vectorised MMS, which is why a 3-10x gain over C++ is plausible. It is not proven.

Large 3D:

| Quantity | JAX (local) | C++ reference |
|---|---|---|
| Fused RHS, stencil+source+masks, 256^3 | 13 ms (0.8 ns/pt) | about 30-110 ms bar |
| Same at 64^3 | 1.5-1.6 ns/pt | n/a |
| RHS with ELL scatter tables, 256^3 | 35 ms (2.1 ns/pt) | n/a |
| RHS with ELL scatter tables, 64^3 | 424 us | 1646 us |
| RK4 step with barriers | 1.7 ms (64^3), 86 ms (256^3) | about 4 x 60 ms plus slot ops |

Dispatch overhead: jitted call 1.9-2.4 us, eager op 3.9 us, `pure_callback` about 24 us. Once the step (or K steps via `fori_loop`) is a single executable, dispatch is negligible on CPU.

Fusion: `jit` plus `fori_loop`/`scan` compiles RHS, BCs, slot updates and stats into one executable; `donate_argnums` handles reuse. Reported pitfall (unverified, local): without `optimization_barrier` between RK stages, XLA:CPU fused nested stencil producers into consumers and recomputed them, making the step cost 15x (64^3) to 33x (256^3) the RHS instead of about 4-6x, a 3-5x slowdown. Each scatter costs a D-sized copy, so merge all row overwrites per output into one table (reported 2.4% rows raising the RHS from 13 to 35 ms at 256^3).

Compile time: roughly 0.1-2 s per (shapes, table sizes, geometry), observed qualitatively. Passing tables as arguments avoids recompiles when only values change; changing |R| recompiles. The persistent compilation cache can amortise this.

Apple GPU (jax-mps, fp32, local): sync dispatch gives RHS 320 us (51x51), 588 us (64^3), 20 ms (256^3) and step 322 us, 1.24 ms, 86 ms. With `JAX_MPS_ASYNC_DISPATCH=1`: RHS 72 us, 251 us, 22 ms and step 207 us, 927 us, 92 ms. That is 3-5x slower than CPU fp64 at 2D shipped sizes and no faster at 256^3. The runs executed outside the Seatbelt sandbox because it hides the Metal device.

## Suitability as a DSL substrate

Strong on CPU/CUDA, weak on Apple.

Strengths:
- The front end (geometry, fragments, sympy closures, ELL tables) stays host NumPy and lowers to a small runtime vocabulary: shifted-slice axis stencil, ELL gather-dot-scatter, masked `where`, closed-form pointwise expressions, pytree axpby.
- `jit` plus `fori_loop` give whole-step compilation, and donation removes slot and pointer-baking machinery.
- `jax.jacfwd`/`jax.linearize` give exact operator export for eigen/pseudospectra work (I44, I50).
- `vmap` handles multi-component Euler batching; `grad` enables alpha optimisation through the real solver.
- `jax.ffi` is an escape hatch for hand-written C++ kernels (CPU/CUDA only).

Weaknesses:
1. Scatters break fusion, so the DSL must merge row overwrites per output.
2. XLA fusion heuristics can over-fuse and recompute stencil chains; the DSL must place barriers/remat at stage boundaries.
3. Limited control of loop order and tiling on CPU; Pallas does not target CPU efficiently, so deeper tuning means C++ FFI.
4. Static shapes: geometry changes recompile.
5. No supported Metal backend and no fp64 on any Apple path.

Recommended role: the reference/CPU/CUDA backend of a DSL whose IR is backend-neutral, with Apple GPU served by a separate lowering (MLX, Metal, Taichi, Warp) once the fp32 question is settled.

Interop: `np.asarray(jax_array)` is zero-copy on CPU; DLPack interoperates with torch (CPU/CUDA) and CuPy; jax-mps to torch-mps DLPack is untested and unlikely zero-copy. Calling JAX from C++ has no supported embedded API beyond the PJRT C API (export StableHLO via `jax.export` and run through a PJRT client, feasible but heavy) or embedding Python.

## Risks

1. Apple GPU: jax-metal is dead; jax-mps is a single-maintainer Alpha plugin tied to jaxlib 0.11.x with compile-only CI, so the MPS goal rests on a hobby project.
2. No fp64 on any Apple path, blocking fp64 parity on MPS regardless of plugin quality.
3. Over-fusion on XLA:CPU needs `optimization_barrier`; the effect size is unverified and heuristic, so it may change across XLA releases. Add perf regression tests.
4. Scatter materialisation can double or triple bandwidth at 256^3 unless tables are merged.
5. `jax.experimental.sparse` is unmaintained; all sparse coupling is hand-rolled.
6. Static shapes: any geometry or table-size change recompiles; moving bodies would need padded max-size tables.
7. Opaque Python callbacks (about 24 us each, host sync on GPU) are a trap if the DSL allows non-jnp MMS or BC code.
8. Fast-moving API with deprecations every few releases; pin versions.
9. All CPU performance claims rest on synthetic local benchmarks; validate on real cut-cell geometry against the actual C++ baseline on the same machine.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| jax-metal unmaintained: last release 0.1.1 (Oct 2024), incompatible with jax>=0.5.1, issue closed 2026-01-20, removed from docs | Confirmed | PyPI shows 0.1.1 on 2024-10-08; issue #26968 shows the Mar 2025 incompatibility report and the Jan 2026 closure comment. Nit: the jax<=0.5.0 limit is a community workaround, not a declared pin. |
| No Apple-GPU JAX path supports float64 | Confirmed | jax-mps PyPI page excludes float64 as "not supported on Metal". Apple's jax-metal page was not re-fetched, so its exact error string is unverified. |
| jax-mps 0.11.0 (Alpha, MLX) runs fp32 RK4 with gather/scatter on Apple GPU, no faster than CPU fp64 | Uncertain | Release date, Alpha classifier, MLX basis, 95.2% pass rate and compile-only CI confirmed. The timings are local and not publicly verifiable. |
| XLA:CPU RK4 step about 40-75 us at 51x51, RHS 1.6 ns/pt at 64^3, beats C++ | Uncertain | Local benchmarks only; the cited jax.jit docs do not support performance numbers; C++ step baseline is an estimate; tables synthetic. Plausible, not established. |
| Missing `optimization_barrier` causes 3-5x slowdown from recomputation | Uncertain | The API exists; the effect is a local observation, likely XLA-version dependent. |
| `jax.experimental.sparse` not actively developed, not for performance-critical use | Confirmed | Quoted from the sparse docs. |
| Latest JAX 0.11.2 (2026-09-17); jaxlib ships C++ FFI headers | Confirmed, with correction | Changelog dates match. Headers named are `collectives.h` and `record.h` for out-of-tree plugins; narrower than the survey wording. Later PyPI releases not checked. |
| `jax.experimental.enable_x64` deprecated in 0.8+ for `jax.enable_x64` | Confirmed | Changelog entry for 0.8.0. |
| Release cadence 0.9.x Feb-Mar, 0.10.x Apr-Jun, 0.11.x Jul-Sep 2026 | Confirmed (0.9.x unseen) | 0.10.x and 0.11.x dates match the changelog. |
| jax-mps "pins to one jaxlib minor" | Roughly confirmed | Docs say StableHLO bytecode targets jaxlib 0.11.x and the build pins LLVM/StableHLO to 0.11.1; working with jax 0.11.2 rests on the local test. |
| jax-metal "needs jax<=0.5.0" | Corrected | Not a declared pin; see nit above. |
| pure_callback about 24 us per call | Unverified | Local measurement only. |

## Sources

- https://docs.jax.dev/en/latest/changelog.html
- https://pypi.org/project/jax-metal/
- https://developer.apple.com/metal/jax/
- https://github.com/jax-ml/jax/issues/26968
- https://pypi.org/project/jax-mps/
- https://github.com/tillahoffmann/jax-mps
- https://docs.jax.dev/en/latest/jax.experimental.sparse.html
- https://docs.jax.dev/en/latest/buffer_donation.html
- https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
- https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html
- Cited by the survey but not checked by the verifier: https://developer.apple.com/forums/thread/815139, https://github.com/google/jax/pull/2936, https://www.gitclear.com/open_repos/google/jax/release/jax-v0.10.2
- Local benchmarks (unverified, not reproduced here): .scratch/jaxbench/{bench.py,bench2.py,bench3.py,mps.py,cb.py} in this worktree; C++ baseline benchmarks/baselines/baseline-20260320-024030.json.
