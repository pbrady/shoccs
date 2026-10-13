# Apple MLX as a SHOCCS Python/DSL Substrate

> **Verdict box**
> - **Fit for SHOCCS:** Partial. Every runtime idiom can be expressed, but MLX is at best an optional fp32 Metal backend under a DSL, not the primary substrate.
> - **CPU parity:** Mixed. fp64 exp/sin/cos use fp32-accuracy approximations (#3047, open), which stands. The fp64 CPU path was inferred to be single-core for built-in ops (#2185, unbenchmarked); measured: 0.085 ms per synthetic SHOCCS-shaped RK4 step at 51x51 (the fastest small-2D path of any backend) and 4.2 ms at 65^3 on CPU fp64 with `mx.compile`, see `05-measurements.md` Section D.
> - **Apple GPU:** Good integration (native Metal, unified memory, `mx.compile`, custom kernels) but fp32/fp16/bf16 only.
> - **fp64:** CPU only. None on Metal, and GPU emulation was declined (#1905, Dec 2025). CUDA fp64 coverage is undocumented (unverified).
> - **Risk:** Medium to high. Pre-1.0 API, ML-first roadmap, scientific-computing requests declined or deprioritized, no sparse support.

Verifier-adjusted overall score: **4.5 / 10**.

## Status (as of 2026-10-12)

- Version checked: MLX v0.32.3, released 29 Sep 2026. Earlier releases: v0.32.2 (25 Aug), v0.32.1 (18 Aug), v0.32.0 (07 Jul), v0.31.2 (22 Apr). The releases page omits the year; 2026 is assumed.
- Pre-1.0 with releases every few weeks. Mature for ML on Apple silicon (LLM inference, training, quantized matmul), thin for scientific computing.
- Install matrix: macOS >= 14 (Metal), Linux CUDA via `mlx[cuda12]` / `mlx[cuda13]` (SM >= 7.5, glibc >= 2.35), Linux CPU-only via `mlx[cpu]` (glibc >= 2.35).
- No first-party sparse support. A sparse-matmul request (#2728) was labeled wontfix in Mar 2026 and closed May 2026. Third-party `mlx-sparse` exists (early beta, announced May 2026).

## Backends and fp64

| Backend | Platforms | fp64 | Notes |
|---|---|---|---|
| Metal | macOS >= 14, Apple silicon | None. Emulation declined (#1905, closed 3 Dec 2025; maintainers advise fp64 on CPU). | MLX's own Metal kernels (not PyTorch MPS). Unified memory, lazy graph, elementwise fusion via `mx.compile`, custom kernels via `mx.fast.metal_kernel`. |
| CPU | macOS arm64, Linux | Yes (`float64` dtype). | Maintainer statement in #2185 says no built-in multithreading for custom primitives; single-core behavior for built-in ops is inferred, not benchmarked. Measured: 0.085 ms per synthetic step at 51x51 and 4.2 ms at 65^3 on CPU fp64 (`05-measurements.md` Section D), so whatever its threading the path is not slow at these sizes. BLAS/LAPACK calls can thread. fp64 exp, sin, cos, erf, erfinv use fp32 SIMD approximations (#3047, open, low priority, deferred until a Highway migration). `mx.array` converts numpy float64 to float32 unless `dtype=mx.float64` is passed (observed in the Section D run). |
| CUDA | Linux, NVIDIA SM >= 7.5, CUDA 12/13 | Unverified. Double specializations appear to exist for elementwise ops, but no per-op documentation was found. | Active development through 2026, ML-first. fp32 matmul defaults to TF32 (see #3860 in the verification log). |
| Other | None for scientific use | Third-party R (Rmlx) and Elixir (EMLX) bindings document no GPU fp64. | No ROCm backend. Bindings for C++, Swift, C. |

Silent dtype hazard: NumPy float64 arrays convert to MLX float32 by default, so dtype must be passed explicitly (unverified in this pass, taken from the survey of the NumPy usage docs).

## Idiom mapping

Support levels: native, workaround, unsupported. "Setup only" means the work stays in NumPy/SymPy on the host and MLX is not involved at runtime. Performance statements are unmeasured estimates.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | Dataclass or dict of 4 arrays `{D:(nx,ny,nz), Rx, Ry, Rz}`, or one packed 1-D array with static offsets. Slots become Python state through a compiled step. | Parity. Packing cuts launch count about 4x. |
| I02 | workaround | Host NumPy fp64 (vectorized roots, filter, `np.lexsort`), then upload. | Setup only. |
| I03 | workaround | Host NumPy merge-join per pencil into SoA fragment tables, converted to masks and ELL tables. | Setup only. |
| I04 | native | Bool masks and int32 index arrays built in NumPy, uploaded once. | Setup only; masks fuse at runtime. |
| I05 | workaround | Host NumPy/SymPy constants baked in as Python floats or captured arrays. | Setup only. |
| I06 | workaround | `sympy.lambdify(psi, F, 'numpy')` on host fp64 psi, cast on upload. Evaluate in fp64 on host first because small-psi cancellation in fp32 is risky. | Setup only. |
| I07 | workaround | Host Python/Numba with `searchsorted` per pencil; emit padded (idx, coef) ELL tables. | Setup only. |
| I08 | workaround | Host NumPy grouped by (left_kind, right_kind, bc); emit interior masks plus ELL closure table. | Setup only. |
| I09 | workaround | Host NumPy; ELL tables over a concatenated `[D; R(r)]` source. | Setup only. |
| I10 | workaround | No sparse type. Keep compact COO/ELL arrays instead of CSR. | Setup only; removes the full-height row pointer. |
| I11 | workaround | `np.cumsum`/`np.concatenate` plus dedup to a closure table; one upload. | Setup only. |
| I12 | workaround | Python config (dict, TOML, or Lua via lupa) replaces sol2; NumPy evaluates coefficient fields. | Setup only. |
| I13 | native | `u.reshape(nx,ny,nz)` (row-major, z fastest, as in C++). Shifted slices are strided views feeding fused elementwise kernels. | Parity. |
| I14 | workaround | Reproduce the (slow, fast, t) lexsort in NumPy or define an explicit permutation; index tables consistently at setup. | No runtime cost; correctness hazard only. |
| I15 | native | Work buffers become intermediates inside the compiled function. No in-place mutation; outputs are returned. | Parity or better when fused; intermediates after gathers are materialized. |
| I16 | workaround | Decompose into fused shifted-slice stencil, `mx.where(interior_mask)`, ELL closure gather-dot, and scatter (`.at[rows].add`). Alternative: one `mx.fast.metal_kernel` per direction replicating the TeamPolicy (GPU only). | Large 3D Metal: stencil bandwidth-bound and fused, closure adds 1 to 3 kernels per direction. CPU: roughly 2-4x slower than 4-core Kokkos at 64^3+ (estimate). |
| I17 | native | `c0*u[:-2]+c1*u[1:-1]+c2*u[2:]` via slices; fused by `mx.compile`. Writing back to the full grid needs `mx.pad`, concatenate or `slice_update`, an extra unfused copy. Avoid via padded-halo layout or a custom kernel. `conv1d` is ML-shaped and not useful for 3 to 9 taps. | Metal fp32: near bandwidth. CPU fp64: one core, slower than multi-thread C++ in bandwidth-bound regimes (inferred). |
| I18 | native | Wall closures as `mx.einsum('rt,t...->r...', Cw, u3[:T])`, placed with `slice_update` or pad. | Negligible volume; 1 to 2 extra kernels each (launch-bound when small). |
| I19 | workaround | ELL gather-dot `(C * mx.take(src, idx)).sum(-1)` then `du.at[rows].add(vals)`. Rows are unique. | Small (about 2.4% of rows). 2 to 3 unfused kernels per table; dominates launch count at small 2D sizes. |
| I20 | workaround | No CSR. Use ELL or COO gather plus scatter-add. | Better than C++ full-height sweep at large N; launch-bound at small N. |
| I21 | workaround | Compact COO: `duD = duD.at[rowsB].add(wB * Rdir[colsB])`. | Large gain at 256^3 (no row-pointer sweep); 2 small kernels. |
| I22 | native | Broadcast axpy on face slabs via `slice_update` or masked `where`. Drop entirely when no Neumann BC. | Negligible. |
| I23 | workaround | ELL gather-dot over `concatenate([uD_flat, Rr])`. | Tiny; one gather and one reduction kernel (reductions are not fused). |
| I24 | workaround | Merged into the I23 table by indexing the R part of the concatenated source. | No extra kernel; removes dependency chain. |
| I25 | native | Disappears: output built as a sum; `where` with mask replaces zero-fill. | Better than C++ (no pass). |
| I26 | native | Fold k* into the fused output expression. | Free under compile. |
| I27 | native | Functional sum `du = Ox+Oy+Oz+B+N`. | Elementwise parts fused. |
| I28 | native | Return a tuple of three arrays in the compiled function; consumer dot fuses with final stage. | Saves scratch only where no gather or scatter boundary intervenes. |
| I29 | native | `gx*dux+gy*duy+gz*duz` fused elementwise. | Bandwidth parity or better. |
| I30 | native | `du + mx.where(fluid_mask, src, 0)`; `.at[idx].add` for tiny R sets. | Dense masked add fused; better than C++ gather at large N. |
| I31 | native | `mx.where(dir_mask, value, u)` or `slice_update` on face planes; R Dirichlet via `where` on an R mask. | Negligible. |
| I32 | workaround | `mx.compile` over the whole RK4 step plus one `mx.eval` per step. No CUDA-graph replay; compile caching plus batched encoding is the substitute. Pointer-stability constraint disappears. | About 15-30 kernels per RHS, 60-120 per step (estimate). Roughly 0.2-1 ms per step at small 2D sizes, comparable to estimated C++ 0.3-1 ms, not clearly better. |
| I33 | native | Plain Python functions; laziness means no fences until eval. | Eager adds about 1-10 us per op (estimate); compile amortizes. |
| I34 | native | Closed-form MMS as array math over broadcast coordinates, with `t` as an `mx.array` input. | Fast on Metal. On CPU fp64, exp/cos are fp32-accuracy (#3047), giving about 1e-7 relative error: unacceptable for convergence studies. Evaluate in NumPy instead. |
| I35 | workaround | Replace Lua callbacks with SymPy-generated array expressions (lambdify). | Removes serial blocker. |
| I36 | native | Pass `t` as a traced scalar array; source computed in the compiled step. | Removes host sync. |
| I37 | native | `u1 = u0 + c*k` fused elementwise via `tree_map`. | Fully fused. |
| I38 | native | Functional state passing; no copies. | Free. |
| I39 | native | `mx.where(bnd_mask, g(t), u)` with g evaluated in the compiled step. | Large win over C++ host allocations. |
| I40 | native | Hoisted to setup as constant mask arrays. | Free. |
| I41 | native | `mx.max(mx.where(mask, mx.abs(u-sol), -inf))`, `mx.argmax`; read back with `.item()` every K steps. | Reductions not fused; `.item()` forces eval and sync. |
| I42 | native | Gate on dump-due; `np.array(u, copy=False)` zero-copy view after eval (unverified), write with NumPy. | Host I/O dominated. |
| I43 | native | Host Python loop around compiled step; one `mx.eval` per step. | About 10-50 us Python overhead per step (estimate). |
| I44 | workaround | Assemble `scipy.sparse` matrix from the same tables, then `numpy.linalg.eigvals`. | Diagnostic; host fp64. |
| I45 | native | NumPy one-liners as oracles. | n/a |
| I46 | native | Fused elementwise via `mx.compile`. | Parity or better. |
| I47 | workaround | Leading component axis `(5, ...)`; pointwise physics fused; batched stencil via slices; R-point slip via `normals[nR,3]`. | Metal fp32 fast but accuracy question sharpens; CPU fp64 loses at scale. |
| I48 | workaround | Stays NumPy/SciPy. | n/a |
| I49 | workaround | Stays SymPy; retarget printer to lambdify with an MLX-capable namespace or NumPy. | Setup only. |
| I50 | unsupported | No sparse non-Hermitian eigensolvers in MLX; stays scipy/ARPACK on host. | n/a |
| I51 | native | In-process MLX solver replaces the subprocess bridge; DLPack sharing with torch (unverified). | Removes process-per-point cost. |

Idioms left out entirely: I50 only. Idioms that are workarounds because MLX has no sparse type, no in-place mutation, or no TeamPolicy analogue: I10, I16, I19-I21, I23, I24, I32.

## Code sketches

Sketches are untested illustrations of the survey's mapping, not verified code.

Per-line block matvec in one direction (I16 to I19):

```python
import mlx.core as mx
# setup (NumPy -> mx): taps c[w], interior mask m_int (nx,ny,nz) bool,
# closure ELL: rows[n] (flat D out), idx[n,T] (flat D in), C[n,T]
def axis_op(u3, axis, c, m_int, rows, idx, C):
    w = len(c); h = w // 2; n = u3.shape[axis]
    sl = lambda a, b: tuple(slice(a, b) if d == axis else slice(None) for d in range(3))
    s = sum(c[j] * u3[sl(j, n - w + 1 + j)] for j in range(w))  # fused elementwise
    pad = [(h, h) if d == axis else (0, 0) for d in range(3)]
    s = mx.pad(s, pad)                                  # separate copy kernel
    s = mx.where(m_int, s, 0.0).reshape(-1)             # drop closure/solid/Dirichlet rows
    vals = (C * mx.take(u3.reshape(-1), idx)).sum(-1)   # gather + reduce kernels
    return s.at[rows].add(vals)                         # scatter (rows unique)

@mx.compile
def lap_D(u3, P):  # P: per-direction tables captured as inputs
    return sum(axis_op(u3, d, *P[d]) for d in range(3))
```

Cut-cell coupling to R points via compact COO/ELL over [D; R] (I21, I23, I24):

```python
import mlx.core as mx
# B (R_dir -> D): rowsB[nb] flat D, colsB[nb] into R_dir, wB[nb]
# Bf+Br merged (D|R_r -> R_r): idxF[nR_r, K] into concat source, WF[nR_r, K] (pad: idx->0, W->0)
def cut_coupling(uD, R, duD, duR, tabs):
    for d, (rowsB, colsB, wB) in enumerate(tabs['B']):
        duD = duD.at[rowsB].add(wB * R[d][colsB])        # duplicates accumulate correctly
    newR = []
    for r in range(3):
        if R[r].size == 0:
            newR.append(duR[r]); continue
        src = mx.concatenate([uD, R[r]])                 # [D ; R_r] source
        idxF, WF = tabs['F'][r]                          # all derivative dirs pre-summed at setup
        newR.append(duR[r] + (WF * src[idxF]).sum(-1))   # gather-dot, no Bf->Br chain
    return duD, newR
```

Boundary application with masks and tiny R lists (I30, I31, I39):

```python
import mlx.core as mx
# setup: fluid_m (D bool), dirD_m (D bool), nondirR_m[r], dirR_m[r] (R bools)
# xD, yD, zD broadcast coords; posR[r] (nR,3)
def mms(x, y, z, t, p):                                   # fusable closed form
    r2 = (x - p.cx)**2 + (y - p.cy)**2 + (z - p.cz)**2
    return p.a * mx.exp(-r2 * p.ivar) * mx.cos(p.om * t)   # CPU fp64 exp/cos ~fp32 acc (#3047)

def apply_rhs_bcs(du, src):
    du['D'] = mx.where(dirD_m, 0.0, du['D'] + mx.where(fluid_m, src['D'], 0.0))
    for r in range(3):
        du['R'][r] = mx.where(dirR_m[r], 0.0,
                              du['R'][r] + mx.where(nondirR_m[r], src['R'][r], 0.0))
    return du

def update_boundary(u, t, p):
    gD = mms(xD, yD, zD, t, p)
    u['D'] = mx.where(dirD_m, gD, u['D'])
    for r in range(3):
        P = posR[r]
        u['R'][r] = mx.where(dirR_m[r], mms(P[:, 0], P[:, 1], P[:, 2], t, p), u['R'][r])
    return u
```

Whole-step compiled RK4 with scalar algebra on slots (I32, I37, I38):

```python
import mlx.core as mx
from mlx.utils import tree_map
def axpy(a, x, y):
    return tree_map(lambda xi, yi: yi + a * xi, x, y)     # fused elementwise

@mx.compile
def rk4_step(u0, t, dt):
    k1 = rhs(u0, t)
    u = update_boundary(axpy(dt / 2, k1, u0), t + dt / 2, P)
    k2 = rhs(u, t + dt / 2)
    u = update_boundary(axpy(dt / 2, k2, u0), t + dt / 2, P)
    k3 = rhs(u, t + dt / 2)
    u = update_boundary(axpy(dt, k3, u0), t + dt, P)
    k4 = rhs(u, t + dt)
    acc = tree_map(lambda a, b, c, d: a + 2 * b + 2 * c + d, k1, k2, k3, k4)
    return update_boundary(axpy(dt / 6, acc, u0), t + dt, P)

t = mx.array(0.0, dtype=mx.float64)   # dtype explicit: numpy f64 -> f32 default
for step in range(nsteps):
    u = rk4_step(u, t, dt); t = t + dt
    mx.eval(u, t)                     # one command-buffer flush per step
    if step % K == 0: err = stats(u, t).item()
```

Note: on Metal the `float64` array in the last sketch is not available; the Metal path must use float32.

## Performance assessment

All numbers below are unmeasured estimates, as the survey states. The verifier found no benchmarks.

**Fusion.** Confirmed from `mlx/compile.cpp`: `is_fusable` returns true only for unary, binary, ternary and broadcast primitives. `is_reduction` is defined but never used. Fused groups are limited to 24 arrays and depth 11. Gathers, scatters, reductions, convolutions and copy-producing slices stay separate kernels. A single-kernel whole RHS therefore needs a hand-written `mx.fast.metal_kernel` (or CUDA equivalent), which can participate in a compiled graph as an opaque primitive.

**Small 2D (heat, scalar wave; about 2.6k to 5k points).**
- Metal: a compiled RK4 step still issues dozens to over 100 kernels. At about 2 to 10 us per dispatch this is roughly 0.2 to 1 ms per step, versus an estimated C++ 0.3 to 1 ms. Parity is possible, not assured; a 2x loss is plausible. fp32 only.
- CPU fp64: per-op overhead (about 1 to 5 us) times the unfused kernel count gives a similar 0.2 to 1 ms. Vectorized MMS helps but its accuracy is degraded by #3047.

**Large 3D (256^3).**
- Metal fp32: bandwidth-bound fused stencils on half-size data; an RHS of about 5 to 20 ms might beat the 30 to 110 ms fp64 C++ bar, but only by giving up fp64. `pad`/`slice_update` copies add about one extra D pass per axis unless a halo layout or custom kernel is used.
- CPU fp64: the earlier inference was 1.5 to 4x slower than multi-core Kokkos OpenMP if built-in ops are single-threaded (#2185, not benchmarked). Measured on a synthetic step (`05-measurements.md` Section D): 4.2 ms at 65^3 against 3.2 ms for four eight-thread Kokkos RHS submits and 7.5 ms at one thread, 36.2 ms at 129^3 and 300 ms at 257^3 (1.1x the best torch.compile CPU path at 257^3, 272 ms). The no-performance-loss bar is open, not failed, until the real tables are run.

**Dispatch overhead.** Eager mode builds a graph node per op (about 1 us) and encodes kernels at eval. Compiled mode pays tracing once per shape; replay still dispatches every unfused primitive. No whole-step graph replay object exists.

**JIT warmup.** First `mx.compile` call traces and generates Metal source (C++ compiled by the system compiler on CPU); expect roughly 0.1 to several seconds per distinct shape or dtype, with recompiles on shape, dtype or input-count changes. Amortized over 20 to 2650 steps this is acceptable.

## Suitability as a DSL substrate

MLX works as a GPU-only fp32 lowering target under a custom DSL and poorly as the single substrate.

- Strengths: clean functional graph API, compile caching, unified memory, C++ core usable from both sides, and custom Metal/CUDA kernel escape hatches whose source a DSL could generate. A DSL would naturally emit one custom kernel per direction (stencil plus closure table plus mask, replacing the TeamPolicy) and use MLX ops for the rest.
- Weaknesses: the fp64 CPU path, which is the parity reference, is likely single-threaded with inaccurate transcendentals, so the DSL needs a separate CPU backend (Numba, C++/Kokkos, or JAX). Because fusion stops at gathers and reductions, the DSL would write most hot kernels itself, reducing MLX to a kernel launcher and memory manager.
- Recommended role: an optional Metal backend of a DSL whose primary fp64 backend is something else.

Interop notes (unverified in the verification pass): `mx.array(np)` copies; `np.array(mx_arr, copy=False)` is a zero-copy view after eval; `torch.as_tensor` over DLPack is zero-copy for GPU-resident buffers; the core is a C++ library, so a C++ host can call it directly; calling existing Kokkos code from MLX would need a custom Primitive or a host round-trip.

## Risks

1. No fp64 on Apple GPU, with emulation declined (#1905). The MPS goal becomes an fp32 accuracy bet on small-psi cut-cell closures and on Euler.
2. CPU backend multithreading is absent for custom primitives and probably for built-ins (inferred); large 3D fp64 runs likely lose to multi-core Kokkos.
3. CPU fp64 exp/sin/cos/erf use fp32 approximations (#3047, open, low priority, blocked on Highway migration), silently corrupting MMS convergence studies.
4. `mx.compile` fuses only elementwise ops, so launch counts at small 2D stay high and the latency bar is not assuredly met.
5. Silent precision defaults: NumPy float64 to MLX float32 (unverified), and the TF32 matmul default (#3860), which affects both CUDA and Apple silicon.
6. No sparse support (#2728 wontfix); all CSR coupling must be hand-rolled as ELL/COO.
7. Pre-1.0 API churn and an ML-driven roadmap; scientific requests (GPU fp64, CPU threading, fp64 SIMD accuracy) were declined or deprioritized.
8. Hand-written Metal kernels give the best Metal performance but are GPU-only, have no CPU fallback, and form a second code path to validate.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| No fp64 on Metal; GPU emulation declined (#1905, Dec 2025) | Confirmed | Issue closed 3 Dec 2025 by a maintainer who said the project will not implement it; emulation would be slow, PyTorch MPS has the same limit, higher precision belongs on CPU. GitHub close reason is "completed", not "not planned"; wording is a decline. |
| CPU has no built-in multithreading (#2185, May 2025), so fp64 CPU elementwise/gather runs on one core | Uncertain | Maintainer (16 May 2025) said no built-in multithreading for custom primitives and suggested streams or a thread pool. Issue author noted most built-in CPU ops appear single-threaded. Issue open, low priority; a contributor proposed a thread-pool helper on 3 Jul 2026 with no maintainer response. Single-core conclusion for built-in ops is inferred; no benchmark found. BLAS/LAPACK can thread. The performance consequence drawn from it is softened by measurement: 0.085 ms per synthetic step at 51x51 and 4.2 ms at 65^3 on CPU fp64 (`05-measurements.md` Section D); the threading itself was not measured. |
| CPU fp64 exp/sin/cos/erf/erfinv use fp32 approximations (#3047), open, low priority, blocked on Highway | Confirmed | Opened 23 Jan 2026; labels bug and low priority. On 16 Aug 2026 a maintainer said the project plans to migrate to Highway (PR #3019) and is not accepting non-critical SIMD changes. Correction: opened Jan 2026; "Aug 2026" is the latest status. |
| `mx.compile` fuses only unary/binary/ternary/broadcast; `is_reduction` unused; limits 24 arrays, depth 11 | Confirmed | `compile.cpp`: `is_fusable` = unary or binary or ternary or broadcast; `max_compile_depth=11`, `max_compile_arrays=24`; `is_reduction` never called. |
| Latest release v0.32.3 (29 Sep); v0.32.0 07 Jul; v0.31.2 22 Apr | Confirmed | Releases page lists v0.32.3 (Latest), v0.32.2, v0.32.1, v0.32.0, v0.31.2. Year omitted on the page. Install docs are for 0.32.3. |
| Install matrix: macOS >= 14; `mlx[cuda12]`/`[cuda13]` SM >= 7.5; `mlx[cpu]`; glibc >= 2.35 | Confirmed | Install page lists these requirements. |
| No first-party sparse; #2728 closed won't-fix; `mlx-sparse` exists | Confirmed | #2728 labeled wontfix 26 Mar 2026, closed 5 May 2026; maintainer cited JAX sparse no longer actively developed. `mlx-sparse` early beta announced 21 May 2026. |
| CUDA fp32 matmul silently runs as TF32 (#3860) | Confirmed, with correction | `MLX_ENABLE_TF32` defaults to 1 on CUDA; about 2.9e-4 relative error on a 512x512 GEMM versus 2.1e-7 with it off. Correction: issue closed 4 Aug 2026 as a documentation fix (PR #3894), and the title was broadened because the default also applies on Apple silicon (M5-class data). The survey's CUDA-only framing was incomplete. |
| NumPy f64 to MLX f32 default; `np.array(copy=False)` zero-copy; torch DLPack zero-copy; `metal_kernel` features (grid, templates, `init_value`, `atomic_outputs`, strided inputs) | Unverified | Verifier did not fetch the NumPy usage or custom-kernel docs. Not refuted; carries no weight either way. |
| `mx.fast.cuda_kernel` and `precompiled_cuda_kernel` exist | Unverified | Existence not checked. |
| CUDA fp64 elementwise "double specializations" | Unverified | The survey itself marks it so; no per-op documentation found. |
| #3078: no thread-safe multi-model concurrency | Unverified | Not checked. |
| Performance figures (0.2-1 ms per step; 1.5-4x slower than Kokkos; 5-20 ms Metal RHS) | Unverified (estimates) | Survey admits these are unmeasured; no benchmarks exist. |

## Sources

- https://github.com/ml-explore/mlx/releases
- https://ml-explore.github.io/mlx/build/html/install.html
- https://ml-explore.github.io/mlx/build/html/usage/compile.html
- https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/compile.cpp
- https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- https://ml-explore.github.io/mlx/build/html/usage/numpy.html
- https://ml-explore.github.io/mlx/build/html/python/ops.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.at.html
- https://github.com/ml-explore/mlx/issues/1905
- https://github.com/ml-explore/mlx/issues/2185
- https://github.com/ml-explore/mlx/issues/3047
- https://github.com/ml-explore/mlx/issues/2856
- https://github.com/ml-explore/mlx/issues/2728
- https://github.com/ml-explore/mlx/issues/3860
- https://github.com/ml-explore/mlx/issues/3078
- https://mlx-sparse.readthedocs.io/en/latest/supported.html
- https://developer.apple.com/forums/thread/797778
