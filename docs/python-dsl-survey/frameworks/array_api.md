# Python Array API Standard as a Portability Layer for SHOCCS

> **Verdict box**
> - **Fit for SHOCCS:** Good as an eager vocabulary and conformance target; not usable alone as the execution layer. Overall score 4/10.
> - **CPU parity:** Not met in eager mode (dispatch-bound at small 2D, extra temporaries at large 3D); plausible only if an external compiler (jax.jit, torch.compile) fuses the whole step.
> - **Apple GPU:** None from the standard itself. torch-MPS is eager fp32 with open Array API failures; MLX is not an Array API backend; jax-metal is unverified and likely stale.
> - **fp64:** Available on NumPy, CuPy, torch CPU/CUDA, JAX (with x64 enabled). Unavailable on every Apple GPU path (MPS, MLX GPU, Metal).
> - **Risk:** Standard itself is low risk (stable, slow moving). Performance and Apple GPU risk are not reduced by it.

Status as of 2026-10-12. Scope: the standard plus array-api-compat, array-api-strict and array-api-extra, considered as a layer under a SHOCCS DSL.

## Status and version

| Item | Value | Verification |
|---|---|---|
| Latest standard revision | 2025.12 | Confirmed on the spec changelog (adds broadcast_shapes, isin, linalg.eig/eigvals, searchsorted scalar update). The changelog has no release dates, so the label is not date-verified. |
| array-api-compat | 1.14.0 (2026-02-26) targets 2025.12; 1.15.0 (2026-06-07) exists | Both confirmed on the compat changelog. The docs index text still says 2024.12 (stale). |
| array-api-strict | 2.5 defaults to 2025.12 | Unverified. |
| array-api-extra | Provides `at`, `apply_where` and similar | Current `at()` semantics not verified. |
| Nature | Spec plus thin shims, not an execution engine | Consistent with all spec pages read. |

## Backends and fp64

| Backend | Via | CPU | CUDA | Apple GPU | fp64 |
|---|---|---|---|---|---|
| NumPy 2.x | native / compat | yes | no | no | yes |
| CuPy | compat | no | yes | no | yes (consumer GPU throughput penalty) |
| PyTorch | compat | yes | yes | MPS, eager only | CPU/CUDA yes; MPS no (SciPy #20700 attributes most failures to missing float64) |
| JAX | native (jax.numpy) | yes | yes | jax-metal (unverified, reported stale) | yes only with `jax_enable_x64`; Metal no |
| MLX | not wrapped by array-api-compat | CPU stream | no | Metal, fp32/fp16/bf16 | CPU only; emulation declined by maintainers (issue closed Dec 2025) |
| array-api-strict | reference | yes (NumPy) | no | no | yes; conformance checker, not a backend |
| Dask, ndonnx, Sparse | compat | n/a | n/a | n/a | irrelevant to this workload |

Any Apple GPU run is fp32 unless double-float is hand written. That is an accuracy risk for small-psi cut-cell closures.

## Idiom mapping

Support values: native (standard ops suffice), workaround (needs array-api-extra, host code or padding), unsupported (outside the standard). "Setup" means done once before time stepping.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | Dict/pytree of 4 arrays, or packed D, Rx, Ry, Rz array via concat and static slices | Equal to C++ (setup) |
| I02 | native | Host: sqrt for quadratic roots, boolean masks, sort/argsort (no lexsort, use stable argsort per key or composite key); nextafter IS in the standard since 2024.12 | Faster than C++ virtual-call loop; setup |
| I03 | workaround | Host: cumulative_sum over open/close events, nonzero, searchsorted; emit start/len arrays | Setup, fine |
| I04 | native | zeros(bool) plus boolean set, or nonzero for index list | Setup |
| I05 | native | Python floats, asarray constants; host linalg solve for RBF system | Setup |
| I06 | native | Elementwise pow, where, clip over psi arrays, or sympy lambdify with modules=xp; compute in fp64 on host then cast | Setup, far faster than per-point C++ |
| I07 | workaround | Host searchsorted over sorted R keys, take_along_axis, where for interior/wall branch; pad data-dependent footprint to fixed width | Setup, beats O(R^2) scan |
| I08 | workaround | Group lines by (left_kind, right_kind, bc), build closure tables with stack/concat; mostly host code | Setup |
| I09 | workaround | Batched I06/I07 output with tensordot or broadcast multiply and sum over taps, giving ELL tables | Setup |
| I10 | workaround | No sparse types, no lexsort or bincount; use host scipy.sparse or keep COO/ELL tables | Setup; ELL needs no CSR |
| I11 | native | cumulative_sum plus concat; dedupe closures with unique_inverse | Setup |
| I12 | native | Pure Python front end replaces Lua | Setup |
| I13 | native | reshape to (nx,ny,nz) is a view; strided axes by slicing; linspace and broadcasting for coordinates | Equal; strided-axis cache behaviour same as C++ |
| I14 | native | Keep R ordering in host tables, verify with argsort checks | No runtime cost |
| I15 | native | Functional state; scratch buffers become temporaries inside a compiled function; JAX arrays are immutable so use `xpx.at` or `where` for writes | Fusion removes scratch; eager adds an allocation per op |
| I16 | workaround | Shifted slices for interior, ELL gather-dot for closures, where(mask,new,old) for coverage; no single primitive | Eager NumPy 2 to 5x slower than fused loop at large sizes; compiled may match C++ (13.7 ns/pt at 64^3 is the C++ figure); small 2D dispatch-bound unless compiled |
| I17 | native | sum_j c[j]*shifted slice via slicing and concat with zeros (no conv, no pad in the standard) | Eager 3 to 6x slower than C++ at bandwidth-bound sizes; compiled near parity |
| I18 | native | tensordot(C, x[..., first/last T planes]) then slice assign or where | Near parity (tiny) |
| I19 | workaround | take on flattened u with an [n_rows, Tmax] index table (the survey assumed 1D indices only; 2024.12 added integer array indexing, so multi-dimensional indices may work, unverified per backend), multiply by C, sum last axis, scatter via `xpx.at(out)[idx].set` or where on a full-length mask | Gather cheap; scatter launch-bound at small sizes, negligible at 256^3 |
| I20 | workaround | ELL/COO: take(x, idx) * C, reshape, sum(axis=1), scatter-add via `xpx.at(y)[rows].add(v)` (dispatch semantics unverified); unique rows make set-then-add safe | np.add.at slow on NumPy; unique rows allow fast fancy-index path; compiled fine |
| I21 | workaround | Compact COO of about R*abs(R) entries: take(R_buf, col) * w then `xpx.at(du_D)[row].add(v)`; rows unique | Faster than C++ full-height CSR sweep (134 MB row pointers at 256^3) |
| I22 | native | Slab axpy u3[k,:,:] += extra[k]*g_face via slicing (functional: concat or xpx.at) | Parity; no-op without Neumann BC |
| I23 | workaround | ELL gather-dot over concat([u_D,u_R]): take, reshape (n,K), multiply C, sum axis 1, add to du_R | Launch-bound small; cheap with fusion |
| I24 | native | Same ELL over concatenated [D;R] source, merged with I23 | Same as I23; merging removes dependency chain |
| I25 | native | zeros_like, or build output as one expression to avoid zero fill | Free under fusion; one extra pass eager |
| I26 | native | Scalar multiply folded into expression | Free under fusion |
| I27 | native | Python addition of directional terms; where for eq semantics | Needs fusion to avoid 3 temporaries per laplacian |
| I28 | native | Return tuple of three arrays | Fused with consumer under jit; else 3 scratch fields |
| I29 | native | gGx*dux + gGy*duy + gGz*duz | Parity fused; three temporaries eager |
| I30 | native | du + where(fluid_mask, src, 0), or xpx.at for tiny R sets | Dense masked add costs 2 to 6% redundant work but coalesced |
| I31 | native | Slice assignment on reshaped D, or where(face_mask, value, u) for functional backends | Negligible |
| I32 | unsupported | No graph or capture concept in the standard (absence of evidence, not an explicit statement). Delegate to jax.jit, torch.compile, mx.compile | Entirely compiler dependent; eager at about 300 ops per RK4 step is 0.3 to 1.5 ms dispatch on CPU |
| I33 | native | Plain Python function | Per-op dispatch (1 to 5 us NumPy, 5 to 20 us torch-MPS, catalog estimates) replaces fences |
| I34 | native | Vectorised closed form with exp, cos, pow over broadcast grids; (nR,3) for R points | Faster than C++ per-point virtual calls (about 37k per step); no fp64 on Apple GPU |
| I35 | unsupported | No per-point host callbacks; replace with array expressions or sympy-generated vector functions | Removing the callback is a win |
| I36 | workaround | Pass t as 0-d array, evaluate source in the same step; fusion external | Parity once compiled |
| I37 | native | u + c*r on packed arrays; fuse BC and accumulate into stage expression | Eager about 9 passes per RK4 step, memory-bound at 256^3; fused equals C++ |
| I38 | native | Functional rebinding, no copy; donation is JAX-specific | Better than C++ (zero copies) |
| I39 | native | where(dir_mask, sol_t_on_boundary, u), evaluating boundary points only via take | Better than C++ |
| I40 | native | Hoist to setup | Better than C++ |
| I41 | native | max/min/argmax of where(mask, abs(u-sol), -inf); per-step readback is a sync | One extra pass; check every K steps on GPU |
| I42 | native | Compute error under dump-due branch; device to host via asarray | Parity or better |
| I43 | native | Python while loop around a compiled step | Negligible host cost |
| I44 | workaround | linalg.eig/eigvals (2025.12) are dense only; export operator to scipy.sparse on host, use ARPACK | Diagnostic only |
| I45 | native | NumPy one-liners | N/A |
| I46 | native | Ordinary array expressions | Same as I25 to I29 |
| I47 | workaround | Leading component axis; pointwise physics with sqrt, where; derivatives batched over axis 0; needs P1 to P3 plus fusion | Fusion dependent; 2 GB flux temporaries at 256^3 if eager |
| I48 | native | Host NumPy/SciPy as is | N/A |
| I49 | native | Reuse sympy layer, retarget printer to xp | Setup |
| I50 | unsupported | Sparse eigs and SVD pseudospectra not in standard; keep SciPy on host | Unchanged |
| I51 | native | In-process Python call replaces subprocess loop | Strictly better |

## Code sketches

Per-line block matvec with strides and per-line dense closures (I16). The scatter relies on array-api-extra.

```python
def deriv_axis(u3, ax, c, w, interior_mask, cl):
    # u3 = xp.reshape(u, (nx,ny,nz)); c = taps; w = len(c)
    h = w // 2
    acc = c[h] * u3
    for j in range(1, h + 1):
        acc = acc + c[h+j] * shift(u3, +j, ax) + c[h-j] * shift(u3, -j, ax)
    out = xp.where(interior_mask, acc, 0.0)   # eq semantics made explicit
    # closures: ELL table, rows unique, window = flat indices into u
    uf = xp.reshape(u3, (-1,))
    win = xp.reshape(xp.take(uf, cl.idx_flat), (cl.n, cl.T))
    rows = xp.sum(win * cl.C, axis=1)
    of = xp.reshape(out, (-1,))
    of = xpx.at(of)[cl.out_idx].set(rows)      # array-api-extra scatter
    return xp.reshape(of, u3.shape)
# shift(a, j, ax) = concat of slice and zeros (no pad in the standard)
```

Cut-cell coupling to R points (I21, I23, I24):

```python
def cut_terms(uD, uR, du_D, du_R, t):
    # B: R{dir} -> D, compact COO (row, col, w), rows unique per entry
    v = xp.take(uR[dir], t.B_col) * t.B_w
    du_D = xpx.at(du_D)[t.B_row].add(v)
    # Bf + Br merged over concatenated source [D; R(r)], ELL padded to K
    src = xp.concat([uD, uR[r]])
    g = xp.reshape(xp.take(src, t.E_idx_flat), (t.nR, t.K))
    du_R = du_R + xp.sum(g * t.E_C, axis=1)   # padded idx -> dummy slot, C=0
    return du_D, du_R
```

Gather/scatter boundary application (I30, I31):

```python
def apply_bc(u, du, src, t, m):
    # dense masked forms (fluid_mask, dir_mask are precomputed bool arrays)
    du = du + xp.where(m.fluid, src, 0.0)         # source add at fluid points
    du = xp.where(m.dirichlet, 0.0, du)           # RHS zero on Dirichlet set
    sol_b = mms_at(m.bpos, t)                     # boundary-only evaluation
    u = xpx.at(u)[m.bidx].set(sol_b)              # update_boundary
    return u, du
```

RK4 on a packed buffer (I37, I38):

```python
def rk4_step(u, t, dt, rhs, bc):
    # u packed: one array [D|Rx|Ry|Rz], so one op instead of four
    k1 = rhs(u, t)
    u2 = bc(u + 0.5*dt*k1, t + 0.5*dt); k2 = rhs(u2, t + 0.5*dt)
    u3 = bc(u + 0.5*dt*k2, t + 0.5*dt); k3 = rhs(u3, t + 0.5*dt)
    u4 = bc(u + dt*k3,     t + dt);     k4 = rhs(u4, t + dt)
    return bc(u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4), t + dt)
# wrap with jax.jit / torch.compile; the standard offers no fusion itself
```

## Performance assessment

All figures below are the survey's estimates; none were measured here.

| Regime | Assessment |
|---|---|
| Small 2D (2.6k to 5k points) | Eager code is dispatch-bound. About 300 ops per RK4 step at 1 to 5 us (NumPy) gives 0.3 to 1.5 ms per step, comparable to the unmeasured C++ estimate of 0.3 to 1 ms but with no headroom. torch-MPS at 5 to 20 us per op would be 5 to 10x slower. Meeting the bar needs a compiler outside the standard. |
| Large 3D (256^3) | Bandwidth-bound. Eager creates a temporary per op, so shifted-slice stencils cost roughly 3 to 6x the memory traffic of a fused loop, against a 16 to 31 ms C++ bound. Fused, parity is plausible and compact COO/ELL tables beat full-height CSR sweeps. |
| Dispatch overhead | NumPy 1 to 5 us, torch CPU 5 to 10 us, torch-MPS 5 to 20 us per call; C++ graph nodes about 0.8 us. array-api-compat adds Python indirection per op (unmeasured). |
| Fusion | Not in the standard. Obtainable by tracing with jax.jit, torch.compile, or mx.compile (MLX not wrapped). Data-dependent shapes (nonzero, boolean-mask getitem, unique_*) break tracing, so the DSL must use static-shape ops: where, take, fixed-width ELL. |
| JIT warmup | Backend specific, seconds per (shape, geometry), amortised over 20 to 2650 steps. |

Apple GPU detail. The standard adds no path. Options: (1) torch-MPS through compat, eager fp32, 5 to 20 us per op, no graph capture claimed (unverified), SciPy issue #20700 (open, July 2026 activity) documents failures, mostly float64 related plus an MPS take_along_dim crash; Inductor on MPS is at best a prototype (a mirrored `codegen/mps.py` calls itself an early prototype for elementwise ops), production state unconfirmed; torch.mps now exposes `compile_shader` and `load_metallib`, giving a hand-written Metal kernel route from torch. (2) MLX: Metal GPU and mx.compile, fp32 only on GPU, no Array API wrapper. (3) jax-metal 0.1.1: reported stale, unverified. (4) Hand-written Metal kernels emitted by the DSL, bypassing the standard.

## Suitability as a DSL substrate

Good as vocabulary and conformance test, weak as an execution substrate.

- Strengths: narrow static-shape-friendly op set (where, take, take_along_axis, reductions, tensordot, elementwise) covering setup, stencil, masked update and gather halves of the hot path. array-api-strict can prove a lowering uses only standard functions.
- Gaps: no scatter/index_add (array-api-extra `at` papers over it per backend), no conv or stencil primitive, no einsum, no sparse, no pad, no lexsort or bincount, no compile or fusion, no streams, no per-device dtype policy.
- Recommendation: the DSL owns its own IR (stencil, ELL gather, masked update) and lowers to Array API code for the portable reference path, and to backend specific code (jax.jit, C/OpenMP stencil, Metal kernel) for the fast path. Do not rely on the standard for the "no performance hit" constraint.
- Interop: DLPack (`from_dlpack`) gives zero-copy between NumPy, CuPy, torch and JAX where devices match. The standard does not cover C++ calls; use nanobind/pybind11, and wrap Kokkos host Views with an explicit pointer shim.

## Risks

1. No fusion or compilation in the standard; performance relies on external compilers, which defeats the portability premise.
2. Scatter-add missing; `xpx.at` is a third-party shim whose duplicate and mutation semantics differ per backend (unverified).
3. MLX not an Array API backend; torch-MPS has open Array API failures; jax-metal status unverified.
4. No fp64 on any Apple GPU path; fp32 closures at small psi may be inaccurate.
5. Dispatch overhead at 2.6k to 5k points makes eager lose to the C++ graph unless the whole step is traced.
6. Data-dependent shapes break tracing, forcing padded fixed-width tables.
7. Standard evolves slowly; scatter or conv are unlikely soon.
8. Compat and strict versions track standard revisions; pinning may be needed.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| No scatter-add, no conv; take and take_along_axis exist; 2025 adds only broadcast_shapes, isin, eig, eigvals | Confirmed | Spec changelog v2025.12 and v2024.12; 2025 blog (2025-03-26). Release dates for 2025.12 not on changelog. |
| array-api-compat 1.14.0 (2026-02-26) targets 2025.12; 1.15.0 (2026-06-07) real; MLX not supported | Confirmed | Compat changelog; docs list NumPy, CuPy, PyTorch, Dask, JAX, ndonnx, Sparse. Docs index text says 2024.12 (stale). |
| MLX GPU has no float64; emulation declined | Confirmed | github.com/ml-explore/mlx issue 1905, closed Dec 2025. |
| PyTorch MPS has no float64 | Confirmed | SciPy #20700 describes lacking float64/complex128. Forum maintainer advice not re-fetched. |
| SciPy #20700 documents Array API failures on torch-MPS | Confirmed | Open since 2024-05-12, activity through July 2026. |
| Standard defines no JIT, graph capture or fusion | Uncertain | Nothing of the kind appears in any page read; no explicit statement found. |
| torch.compile/Inductor on MPS unsupported | Uncertain | 2.1-era thread says unsupported; Inductor MPS codegen exists as prototype; PyTorch 2.14 blog (2026-09-17) does not announce compile support. |
| jax-metal 0.1.1 stale and fails on recent JAX | Uncertain | Not re-verified; Metal lacking f64 is supported by the MLX issue. |
| I02: nextafter not in the standard | Refuted | Added in v2024.12; use the standard function. |
| I19: take takes only 1D indices | Corrected | 2024.12 added integer array indexing; compat 1.13.0 notes negative-index support in torch.take and take_along_axis. Flattening still safe. |
| xpx.at add/set dispatch semantics per backend | Unverified | array-api-extra behaviour not checked. |
| array-api-strict 2.5 defaults to 2025.12 | Unverified | Not checked. |
| Torch-MPS has no graph capture | Unverified | Not checked. |

## Sources

- https://data-apis.org/array-api/latest/changelog.html
- https://data-apis.org/blog/array_api_v2025_release/
- https://data-apis.org/blog/array_api_v2024_release/
- https://data-apis.org/array-api/latest/API_specification/indexing_functions.html
- https://data-apis.org/array-api-compat/changelog.html
- https://data-apis.org/array-api-compat/
- https://data-apis.org/array-api-strict/changelog.html
- https://pypi.org/project/array-api-compat/
- https://github.com/ml-explore/mlx/issues/1905
- https://github.com/ml-explore/mlx/issues/48
- https://www.mintlify.com/ml-explore/mlx/api/data-types
- https://github.com/scipy/scipy/issues/20700
- https://discuss.pytorch.org/t/float64-tensors-on-mps/224325
- https://discuss.pytorch.org/t/jitting-fails-on-mps-device/192079
- https://pytorch.org/blog/pytorch-2-14-release-blog/
- https://docs.pytorch.org/docs/2.14/mps.html
- https://pypi.org/project/jax-metal
- https://developer.apple.com/forums/thread/815139
- https://github.com/data-apis/array-api/issues/808
