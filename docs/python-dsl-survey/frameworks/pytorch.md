# PyTorch (eager, torch.compile/Inductor, MPS) as a SHOCCS substrate

> **Verdict box**
> - **Fit for SHOCCS:** Good for CPU and CUDA; usable on Apple GPU only as an fp32 large-3D option. Every runtime idiom maps to dense torch ops once setup flattens the cut-cell geometry into masks, a uniform axis stencil and compact ELL tables.
> - **CPU parity:** Large 3D likely meets or beats C++ (bandwidth bound, fused). Small shipped 2D cases are marginal (1-2x of C++), per unverified local measurements.
> - **Apple GPU:** Partial. Eager MPS ops and Inductor-MPS Metal codegen exist, plus `torch.mps.compile_shader`; no graph replay, so small sizes are dispatch bound.
> - **fp64:** CPU and CUDA full. MPS none (confirmed): fp32/fp16/bf16 only.
> - **Risk:** Low for CPU/CUDA (huge, maintained, already a locked dependency); medium-high for the MPS path (fp32 accuracy of cut-cell closures unknown, Inductor-MPS has open correctness bugs).

## Status as of 2026-10-12

- Latest release: PyTorch 2.14.1, announced 2026-09-30 (verified). Release cadence is one per two months in 2026 (verified in the 2.11 blog). The 2.12 (May 13) and 2.13 (July) dates are unverified.
- The project lockfile pins torch 2.11.0 (CPU wheel). Local measurements used 2.11.0 on macOS arm64.
- The 2.11 release listed an MPS operator expansion. Verified content is modest: some distribution functions, erfcx, grid_sampler_2d and baddbmm extensions, async error reporting. The claim that 2.13 added FlexAttention on MPS is unverified and irrelevant to SHOCCS.
- MPS was not reachable from the Bash sandbox, so no local MPS timings exist.

## Backends and fp64

| Backend | fp64 | Execution path | Notes |
|---|---|---|---|
| CPU | Full, native, vectorized | ATen kernels (OpenMP); Inductor emits fused C++/OpenMP loops; `cpp_wrapper=True` removes the Python wrapper | Gather, `index_add_`, `scatter_add_` all native in fp64. `torch.sparse.mm` works but is opaque to Inductor. |
| CUDA | Full (rate depends on GPU) | Inductor generates Triton; `mode="reduce-overhead"` uses CUDA graphs | Strongest backend; a free side benefit, not the target hardware. CUDA-graph details are unverified (doc page was a redirect stub), though widely documented. |
| MPS (Apple GPU) | None (confirmed). Moving fp64 to MPS raises "MPS framework doesn't support float64" | Eager MPSGraph plus Metal kernels; Inductor-MPS emits Metal source (`torch/_inductor/codegen/mps.py`); `torch.mps.compile_shader` and `load_metallib` for hand-written MSL | No graph capture or replay API on the 2.14 `torch.mps` page (strong inference). Sparse on MPS is not a dependable path. |
| ROCm / XPU | Per hardware | Wheels exist | Not relevant. |

fp64 detail: an Apple DTS engineer (Sep 2025) asked users to file enhancement requests for fp64 in the MPS framework and pointed to MLX. A PyTorch forum reply (2026-06-23) says the GPU lacks native fp64 and emulation would likely be too slow. The survey's phrase "Metal exposes no double type" is plausible but was not stated in those threads. Double-float emulation would have to be hand-written MSL via `compile_shader`, and Inductor cannot generate it.

## Idiom mapping

Support levels: native (direct torch ops), workaround (needs restructuring or host code), unsupported. "Setup" means host NumPy/SymPy work done once. Performance entries for CPU are estimates built on unverified local measurements unless noted.

| Id | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | Pytree of tensors per field, or one packed 1-D tensor `D\|Rx\|Ry\|Rz` with static offset views | Same as C++; packing cuts eager op count 4x |
| I02 | native | Host NumPy/torch-CPU fp64: vectorized quadratic roots, filter, lexsort into (slow, fast, t) order, SoA tensors, same 1e-12 snap/clamp | Setup only; likely faster than C++ virtual-call loop |
| I03 | workaround | Host NumPy per-pencil grouping with `searchsorted` and cumulative ray parity to emit a fragment table | Setup only |
| I04 | native | Bool masks built in NumPy, uploaded once; optional index tensors | Setup only; masks favor fused dense updates |
| I05 | native | SymPy constants become Python floats baked into the traced graph | Setup only |
| I06 | native | `sympy.lambdify(psi, block, "numpy")` over the psi array in fp64 with clamps re-added; cast to fp32 for MPS | Setup only |
| I07 | workaround | Host fragment lookup via `searchsorted` replaces O(\|R\|) scan, emits padded ELL (idx, coef) | Setup only; O(\|R\| log n) |
| I08 | workaround | Host grouping by (left_kind, right_kind, bc); dedup closure table plus ELL closure rows with interior/uncovered masks | Setup only |
| I09 | workaround | Host NumPy batched over R points; ELL tables for Bf/Br over concatenated `[D; R(r)]` source | Setup only |
| I10 | native | `np.lexsort` or scipy COO to CSR for export; runtime keeps compact COO/ELL | Setup only; removes full-height row pointers |
| I11 | native | `np.cumsum`/`np.concatenate` into a dedup closure table; one `torch.as_tensor` upload per table | Setup only; avoids ~700k allocations at 256^3 |
| I12 | native | Python front end (TOML/YAML/Python) replaces Lua; coefficient fields as tensor expressions | Setup only |
| I13 | native | `u.view(nx,ny,nz)`; shifted slices per axis; `view(-1)` for ELL gathers; broadcast 1-D coordinates | Free (views) |
| I14 | native | Reproduce lexsort order exactly at setup; all tables indexed consistently | No runtime cost; correctness hazard only |
| I15 | native | Functional tensors replace spans; temporaries fused away under compile, or preallocated `out=` buffers in eager | Equal or better |
| I16 | workaround | No ragged strided-segment primitive. Decompose into uniform axis stencil over whole grid, ELL gather-dot plus `index_put` overwrite of closure rows, and `where(mask)` for uncovered rows. Alternative: one custom kernel (C++ extension on CPU, `compile_shader` on MPS) | CPU compiled: estimated to match or beat C++ at 64^3+ (coalesced slices fix slow-axis locality). Small 2D: launch bound, ~2-4 ops per direction at ~0.6 us each eager (unverified measurement) |
| I17 | native | Explicit shifted slices of a padded view (`F.pad` plus `narrow`) or `conv1d` after permute; Inductor fuses taps and directions | Memory bound; ~1-2 ns/pt per axis at large N (local measurement, unverified) |
| I18 | native | `einsum('rt,t...->r...', Cwall, x3[:T])` on first/last T planes, slice-assigned into the output slab | Negligible |
| I19 | workaround | ELL gather-dot `(C * xflat[idx]).sum(1)` then `index_put` (overwrite) or `index_add_`; rows unique. Inductor may lower `index_add` to an atomic store loop | Tiny row counts; gather bound at 256^3 (a few ms per RHS on CPU, estimate) |
| I20 | workaround | Avoid `torch.sparse` on MPS; compact ELL/COO with gather plus `index_add_`. On CPU `torch.sparse.mm(csr, dense)` works in fp64 but is not fusable | Better than C++ full-height sweeps since only non-empty rows are touched |
| I21 | native | `du_flat.index_add_(0, B_rows, B_w * uR[B_cols])` with compact COO | Much faster than the C++ full-height sweep at 256^3; 1-2 ops at small sizes |
| I22 | native | Broadcast axpy on slabs; omitted at trace time when no Neumann face | Removes empty C++ sweeps |
| I23 | native | One ELL over `torch.cat([uD_flat, uRr])`, padded idx pointing at a zero slot | 1 gather plus 1 reduce |
| I24 | native | Merged into the I23 ELL by concatenating the R(r) source (removes Bf to Br chain) | Free once merged |
| I25 | native | Build `du` as an expression and let Inductor fuse; eager `zero_()` on a packed buffer is 1 op | Disappears under compile |
| I26 | native | Scale folded into the output expression | Fused, free |
| I27 | native | Functional sum of directional terms; ordering constraints vanish; partial `eq_t` coverage made explicit with `where(mask, new, old)` | 1-3 fused kernels under compile |
| I28 | native | Return a tuple or stack of 3 derivatives; fuse into the I29 consumer so they are never materialized | Better than C++ at large N (no 3x scratch) |
| I29 | native | Pointwise `gGx*dux + gGy*duy + gGz*duz` | Fused, bandwidth optimal |
| I30 | native | `du + fluid_mask*src` or `torch.where`; small masks for R sets | About free versus a C++ gather pass |
| I31 | native | `u3[0,:,:] = val` in eager, or `torch.where(dir_mask, sol_t, u)` inside the compiled step | Negligible |
| I32 | workaround | CPU: `torch.compile(step, fullgraph=True)`, no graph replay. CUDA: `mode="reduce-overhead"`. MPS: Inductor fusion only, no command-buffer replay | CPU: fixed per-call overhead dominates at heat.lua size (local, unverified: ~70-90 us per toy RK4 step). MPS: ~10-50 us per compiled call plus a sync (community estimate, unmeasured) |
| I33 | native | Plain Python function; CPU is synchronous; MPS syncs on `.item()`/`.cpu()` | ~0.6 us per op CPU (unverified local); ~5-20 us per op MPS (community, unmeasured) |
| I34 | native | Closed-form MMS as torch expressions over broadcast coordinates and an (nR,3) position tensor, generated from SymPy | One fused kernel versus 37k virtual calls |
| I35 | unsupported | Per-point Lua callbacks cannot be traced. Replace with SymPy/torch expression MMS or precomputed tables | N/A; replacement is faster |
| I36 | native | Pass `t` as a 0-d tensor to avoid recompiles; source joins the fused graph | Removes per-stage host sync |
| I37 | native | Packed state `u1 = u0 + c*k`, or `torch._foreach_*` over buffer lists | ~0.6 us per packed op eager; fused when compiled |
| I38 | native | Rebind Python references, or `u0.copy_(u1)` when static buffers are needed | Free or 1 memcpy |
| I39 | native | Evaluate exact solution only on face slices and R Dirichlet subset, then `where`/slice assign in the step | Large algorithmic win over C++ full-field evaluation |
| I40 | native | Hoist masks and index tensors to setup | Removes ~20-30 allocations per step |
| I41 | native | `torch.where(mask, (u-sol).abs(), -inf)` then `aminmax`/`argmax` on device; read back every K steps | CPU about free; MPS one sync per readback |
| I42 | native | Compute error field only when a dump is due; `.cpu().numpy().tofile`; XDMF from Python | Faster than C++ (gated) |
| I43 | native | Host Python loop around the compiled step; dt from a formula or device reduction | Few us per step; negligible |
| I44 | native | Assemble operator from the same tables into scipy.sparse (or `torch.func.vmap/jacrev` for small n), then `numpy.linalg.eigvals` | Diagnostic |
| I45 | native | NumPy/torch one-liners as test oracles | N/A |
| I46 | native | Native torch pointwise expressions; Inductor fuses arbitrary trees | Equal or better |
| I47 | native | Leading component axis `(5,nx,ny,nz)`; stencils and ELL broadcast over it; fluxes fused pointwise | Good compiled (avoids multi-GB flux temporaries); eager CPU materializes them |
| I48 | native | Reuse tables to build scipy.sparse or `torch.sparse_coo/csr` on CPU | Diagnostic |
| I49 | native | Already Python/SymPy; add `lambdify` with numpy/torch modules or a torch printer | Setup |
| I50 | native | Stays on host scipy (ARPACK, dense SVD) with a `LinearOperator` over the torch tables | Unchanged |
| I51 | native | In-process torch solver replaces the subprocess bridge; torch/botorch already locked | Removes per-point process startup (large BO win) |

Only I35 is unsupported. Workarounds (I03, I07, I08, I09, I16, I19, I20, I32) all reduce to host-side setup or the ELL gather plus `index_put` pattern.

## Code sketches

Sketches are illustrative and untested as written.

Per-line block matvec, one direction (I16 to I19):

```python
# setup (host): c = interior taps (w=2P+1); interior_mask[D] bool
# ELL closure rows: row_ids[n], idx[n,T], C[n,T]
def axis_op(u3, d, c, interior_mask, row_ids, idx, C):
    P = (len(c) - 1) // 2
    pad = [0, 0] * 3
    pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = P      # F.pad orders dims last-first
    up = torch.nn.functional.pad(u3[None], pad)[0]
    n = u3.shape[d]
    acc = sum(c[j] * up.narrow(d, j, n) for j in range(len(c)))
    out = torch.where(interior_mask, acc, 0.0).reshape(-1)   # total definition
    vals = (C * u3.reshape(-1)[idx]).sum(1)                  # closure dot products
    return out.index_put((row_ids,), vals, accumulate=False).view_as(u3)
```

Cut-cell coupling to R points (I21, I23, I24):

```python
def cut_coupling(du_flat, duR, u_flat, uR_dir, uR_r, B_rows, B_cols, B_w, F_idx, F_w):
    du_flat = du_flat.index_add(0, B_rows, B_w * uR_dir[B_cols])
    src = torch.cat([u_flat, uR_r, u_flat.new_zeros(1)])   # last slot is the zero pad
    duR = duR + (F_w * src[F_idx]).sum(dim=1)
    return du_flat, duR
# CPU fp64 fuses under compile; CUDA uses Triton gather; MPS fp32 only, avoid torch.sparse.
```

Boundary application (I30, I31, I39):

```python
def rhs_boundary(du3, duR, src3, srcR, nondir_R):
    du3 = du3 + torch.where(fluid_mask, src3, 0.0)
    du3 = torch.where(dir_mask, 0.0, du3)
    duR = torch.where(nondir_R, duR + srcR, 0.0)
    return du3, duR

def update_boundary(u3, uR, t, bnd_flat_idx, bnd_xyz, Rdir_idx, Rdir_pos):
    val = mms(*bnd_xyz.unbind(1), t)                 # evaluate only on boundary points
    u = u3.reshape(-1).index_put((bnd_flat_idx,), val)
    uR = uR.index_put((Rdir_idx,), mms(*Rdir_pos.unbind(1), t))
    return u.view_as(u3), uR
# t is a 0-d tensor so torch.compile does not recompile per stage.
```

Packed-state RK4 (I32, I37, I38):

```python
def rk4_step(u0, t, dt):
    k1 = rhs(u0, t)
    u1 = update_boundary_packed(u0 + 0.5 * dt * k1, t + 0.5 * dt)
    k2 = rhs(u1, t + 0.5 * dt)
    u1 = update_boundary_packed(u0 + 0.5 * dt * k2, t + 0.5 * dt)
    k3 = rhs(u1, t + 0.5 * dt)
    u1 = update_boundary_packed(u0 + dt * k3, t + dt)
    k4 = rhs(u1, t + dt)
    u = u0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return update_boundary_packed(u, t + dt)

step = torch.compile(rk4_step, fullgraph=True, dynamic=False)  # CUDA: mode="reduce-overhead"
for n in range(nsteps):
    u = step(u, t, dt); t += dt
    if n % K == 0: stats = masked_stats(u, t).cpu()    # one sync every K steps
```

## Performance assessment

All timings in this section come from the survey's local runs (torch 2.11, macOS arm64, 4 threads, sandboxed, libomp shm warning) and are unverified by any public source. Re-measure natively before deciding.

**Dispatch overhead.** CPU eager is about 0.6 us per op on 1k doubles, comparable to the Kokkos 0.7-0.8 us launch floor. A compiled call has about 5.5 us fixed overhead plus per-kernel OpenMP cost. MPS eager is about 5-20 us per op (community figure, not measured) plus blocking syncs on readback, and MPS has no graph replay. CUDA with `reduce-overhead` reduces a step to about one graph launch.

**Small 2D (shipped heat.lua / scalar_wave.lua sizes).** A 51x51 toy RHS took 27-28 us eager and 124-133 us per RK4 step eager. `torch.compile` did not help (128-146 us per step; 77-91 us for a pure-stencil variant even with `cpp_wrapper=True`) because roughly 70 us of fixed per-call and per-kernel overhead dominates. A real RHS is estimated at 40-70 ops, so 40-70 us per RHS versus a C++ bar of about 30-40 us, i.e. 1-2x of C++. Vectorized MMS (I34) recovers some ground since C++ is estimated to spend 100-300 us per step on host MMS. On MPS at this size, hundreds of ops per step at 5-20 us each would be 5-10x slower than C++. The "no performance loss" constraint is only marginally met on CPU here.

**Large 3D (about 256^3).** Bandwidth bound. Measured compiled 5-point 2D RK4 is about 4.5 ns/pt/step at 1024^2 (about 1.1 ns/pt per RHS); eager is 2.6x slower because of temporaries. For 256^3 with a compiled step reading D about once per axis, the estimate is 20-40 ms per RHS on CPU against a C++ bar of 30-110 ms. Compact ELL tables remove the C++ 134 MB row-pointer sweeps. MPS fp32 halves bytes and could plausibly be 2-5x faster than CPU C++ at 256^3 with a well-fused step, but this is unmeasured and fp32 only.

**Fusion.** On CPU, `fullgraph=True` traced a full RK4 step (4 RHS including gather and `index_add`) as one graph with 0 graph breaks, per `torch._dynamo.explain` (local, unverified). CUDA adds graph capture. Inductor-MPS fuses to Metal but is less mature. `torch.sparse` ops and `compile_shader` kernels are opaque to Inductor, so a DSL should emit dense, gather and `index_add` ops to stay fusable.

**JIT warmup.** 0.4-4.7 s cold, 0.1-1.6 s warm cache. Specialized to shape and geometry (`dynamic=False`); any table length change recompiles. Negligible for long runs, a tax on short BO runs.

## Suitability as a DSL substrate

Strong for CPU and CUDA, weak but usable for MPS.

- A DSL lowering to dense torch ops (shifted-slice stencils, ELL gather-dot plus `index_put`/`index_add`, `where` masks, packed-state updates) gets Inductor fusion and fp64 on CPU and CUDA without a hand-written backend. It also gets CUDA graphs, autograd, `torch.func` (jacrev/vmap for operator export) and AOTInductor export to C++ (CPU/CUDA; MPS support uncertain).
- The DSL's job shrinks to setup: fragment tables, closure dedup and ELL emission from SymPy, and emitting a pure Python step function. This absorbs the cut-cell unstructuredness at setup rather than in kernels.
- Interop is good: zero-copy NumPy and DLPack on CPU/CUDA, `torch.utils.cpp_extension` and the stable libtorch ABI for wrapping Kokkos operators as `torch.library.custom_op` (traceable via `register_fake` but opaque to Inductor).
- Weak points: (1) small-size overhead means a DSL could only win at 2.6k points by emitting a single hand-fused C++ kernel and bypassing torch; (2) MPS needs its own Metal codegen via `compile_shader` for critical kernels, losing cross-kernel fusion, plus fp32 only; (3) Inductor is ML oriented (guards, recompiles on shape change, opaque custom ops).
- Versus JAX: more imperative (easier in-place and debugging) and a real MPS backend, but no `lax.scan`-style whole-time-loop compilation. Unrolling K steps into one graph did not improve per-step time in the survey's test.

## Risks

1. No fp64 on MPS (confirmed). Whether small-psi high-order closures (E4_1 clamped to [1e-4, 1-1e-4], RBF variants) stay accurate and stable in fp32 is unanswered.
2. Small-size CPU parity is marginal (unverified local data); heat.lua and scalar_wave.lua may land at 1-2x of C++.
3. Inductor-MPS maturity: open silent-correctness issue #169738 (AdaptiveMaxPool under Inductor on MPS, opened 2025-12-06, still open) and reported expected-failure tests (not independently checked). Regressions across the two-monthly cadence are plausible; no graph replay.
4. Shape- and geometry-specialized compilation: new geometry or table length recompiles (0.1-5 s).
5. Custom kernels (`compile_shader`, C++ extensions) are opaque to Inductor and break fusion around them.
6. Bit-for-bit parity with C++ is unlikely (reduction order, FMA, total `where` semantics replacing stale `eq_t` values). Validation must be tolerance based; the `t-laplacian` reference is itself currently failing.
7. Measurement caveat: MPS unavailable and a libomp warning in the sandbox; compiled small-size overhead may be partly environmental.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| MPS does not support float64; Apple has not committed to fp64 | Confirmed | Apple forum thread 797778 (error text; DTS asks for enhancement request, points to MLX); PyTorch forum reply 2026-06-23. Caveat: threads discuss the MPS framework, not Metal's type system. |
| `torch.mps` exposes synchronize, Event, profiler, compile_shader, load_metallib; no graph capture/replay | Confirmed | 2.14 `torch.mps` page lists these and no graph API. Absence elsewhere is an inference. |
| `compile_shader` compiles Metal source at runtime, kernels callable from Python | Confirmed | 2.14 `torch.mps` page. |
| Inductor MPS backend exists with open correctness bugs and expected-failure tests | Confirmed (bug), partly unchecked | Issue #169738 open, silent correctness, MPS-labeled. Existence of `codegen/mps.py` was checked locally by the survey only; expected-failure tests not checked online. |
| Latest release 2.14.1 (2026-09-30), two-month cadence, 2.11 expanded MPS ops | Confirmed | dev-discuss announcement; 2.11 blog. 2.12/2.13 dates unverified. The 2.11 expansion is modest in substance. |
| Local CPU figures (0.6 us/op, 5.5 us compiled overhead, 0 graph breaks, 2.6x at 1024^2, 70 us fixed per step) | Unverified | Local sandboxed measurements, 4 threads, libomp warning; not checkable from public sources. |
| `reduce-overhead` uses CUDA graphs and needs static inputs | Unverified | Fetched doc page was a redirect stub; widely documented but not confirmed here. |
| 2.13 added FlexAttention on MPS | Unverified | Not in the 2.11 blog; irrelevant to the score. |
| Apple pointed to MLX, which also lacks GPU fp64 | Unverified (partly) | Thread links an MLX float64 feature request; current MLX status not confirmed. |
| "Metal exposes no double type" | Unverified wording | Threads say the MPS framework lacks float64; the Metal-level statement is not in them. |

Adjusted overall score: 6.5 (unchanged by verification).

## Sources

- https://dev-discuss.pytorch.org/t/pytorch-2-14-1-general-availability/3455
- https://dev-discuss.pytorch.org/t/pytorch-2-13-0-general-availability/3412
- https://dev-discuss.pytorch.org/t/pytorch-2-12-0-general-availability/3385
- https://pytorch.org/blog/pytorch-2.11-release/
- https://docs.pytorch.org/docs/2.14/mps.html
- https://docs.pytorch.org/docs/2.14/notes/mps.html
- https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html
- https://github.com/pytorch/pytorch/issues/169738
- https://developer.apple.com/forums/thread/797778
- https://discuss.pytorch.org/t/float64-tensors-on-mps/224325
- https://developer.apple.com/forums/thread/709663
- https://discuss.pytorch.org/t/questions-about-pytorch-2-0-and-mps/168552
- https://newreleases.io/project/pypi/torch/release/2.9.0
- https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
- Local measurement scripts (unverified): .scratch/torchbench/{b,c,d,e}.py
