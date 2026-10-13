# Taichi (taichi-lang) for SHOCCS

## Verdict box

| Item | Verdict |
|---|---|
| Fit for SHOCCS | Plausible optional code-generation backend (CPU f64, CUDA f64, Metal f32); not a foundation. Imperative kernels express the cut-cell irregularity (ELL closures, COO couplings, gather lists) directly. |
| CPU parity | Likely parity at 256^3 (bandwidth-bound); at small 2D (30-40 us per RHS bar) at risk unless the whole RHS is hand-fused into one kernel. All overhead figures are unmeasured. |
| Apple GPU | Native Metal backend emitting MSL (not MPS/MPSGraph). f32 and int only, no sparse SNodes, no known graph replay, PyTorch MPS interop unverified and probably absent. |
| fp64 | Yes on CPU (LLVM) and CUDA. No on Metal (confirmed unsupported). Vulkan only on devices with fp64; MoltenVK on Apple has none. |
| Risk | High long-term: latest release v1.7.4 is 14 months old at 2026-10-12, 878 open issues. Maintenance mode is an inference, not a stated status. |

Verifier-adjusted overall score: 5 / 10 (the survey said 5.5).

## Status as of 2026-10-12

Latest release is v1.7.4 (2025-07-31). PyPI lists nothing newer; the previous release was v1.7.3 (2024-12-23). The GitHub repository page showed 878 open issues and 47 open PRs (cached summary).

The v1.7.2 to v1.7.4 releases are mostly packaging and bug fixes: Python 3.13 wheels, Python below 3.9 dropped, manylinux2014 wheels dropped, macOS minimum raised, faster compile key generation in 1.7.3. v1.7.0 (2023-11-27) was a real feature release (stabilized real_func, argument packs, dynamic CUDA VRAM, GGUI on Metal, CC backend removed). The CC backend removal therefore dates from 2023, not from 2025 packaging work. The cadence supports "slow-moving"; "maintenance mode" is an inference because the repository states no status. The README reportedly still lists Python 3.6 to 3.10, a docs staleness signal.

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| CPU (LLVM x64, arm64) | Mature; macOS arm64 wheels exist | Yes (`ti.init(default_fp=ti.f64)`) | Outermost loop parallelised on a thread pool. Offline cache avoids re-JIT. SIMD quality vs hand-tuned C++ unmeasured. |
| CUDA (LLVM-PTX) | Mature | Yes, including f64 atomics | Listed as tier 2 in the C-API docs. Graph capture claims unverified (see log). |
| Metal | Native MSL backend, not MPS | No (type table marks f64 unsupported) | Sparse SNodes removed in v1.4.0 (deprecated v1.3.0). Dense fields, ndarrays, atomics, ordinary kernels work. Metal AOT/TiRT runtime since v1.5.0 is experimental (tier 2). |
| Vulkan | Tier 1 in the C-API docs | Device dependent; none via MoltenVK on Apple | Primary AOT/graph target. |
| OpenGL 4.3+ | Present | Not assessed | |
| WASM | Experimental | Not assessed | |
| CC | Removed in v1.7.0 | n/a | |

Double-float emulation on Metal is not provided and would have to be hand-written in kernels. An f64 reference solve stays on CPU or CUDA.

## Idiom mapping

Support values: native, workaround, unsupported. "Setup only" means host-side work outside the timestep loop. Performance expectations are architectural estimates; none were benchmarked.

| ID | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | One `ti.ndarray(ti.f64, ...)` per buffer, or a packed ndarray with static offsets; the dict is host Python. | Equal to C++; setup only. |
| I02 | workaround | Host NumPy/Numba ray casting emitting SoA arrays; a kernel over pencils x shapes would need atomic counters and a sort for variable-length hits. | Setup only. |
| I03 | workaround | Host NumPy merge-join emitting start, length, stride, closure-id arrays. | Setup only. |
| I04 | native | NumPy masks or index arrays uploaded via `from_numpy`. | Setup only. |
| I05 | native | Python constants; closure-captured floats or `ti.template()` become compile-time constants. | Setup only; baked constants as fast as C++. |
| I06 | workaround | Host lambdify, or a `@ti.func` of the CSE'd rational expressions per R point; a Taichi SymPy printer target is needed. | Setup only; run on CPU in f64 for small-psi cancellation. |
| I07 | workaround | Host Python/Numba producing a padded table for data-dependent footprints. | Setup only. |
| I08 | workaround | Host NumPy grouping into line tables, deduplicated closure table, ELL row tables. | Setup only. |
| I09 | workaround | Host NumPy, or a kernel over R points writing fixed-width ELL rows. | Setup only. |
| I10 | native | `np.lexsort` or `scipy.sparse`; Taichi only receives compact COO/ELL arrays. | Setup only. |
| I11 | native | `np.cumsum` + `np.concatenate`, uploaded to ndarrays. | Setup only. |
| I12 | native | Python front end replaces the Lua config. | Setup only. |
| I13 | native | `ti.ndarray` of shape (nx,ny,nz) iterated with `ti.ndrange`; stride access is index arithmetic. The "no padding, flat-indexable, z fastest" claim is unverified. | Equal to C++ (expected). |
| I14 | native | Convention only: one array per R(dir) with an explicit index table. | No cost. |
| I15 | native | Ndarrays passed by reference (`ti.types.ndarray()`), kernels write in place; scratch buffers are ndarrays. | Equal to C++. On Metal all buffers live in GPU memory. |
| I16 | workaround | No ragged per-line team primitive. Three kernels per direction: masked interior, wall closure on boundary planes, ELL object closure. Fluid fragments become masks. | CPU: estimated at or better than C++ (C++ one-team-per-line is 13.7 ns/pt at 64^3 slow axis; a collapsed outer loop with contiguous inner loop might reach 1-3 ns/pt, unmeasured). Launch count rises about 3x per direction; tiny 2D needs one fused kernel. |
| I17 | native | `for i,j,k in ti.ndrange(...)` with masked, `ti.static`-unrolled tap sum. | About C++ and memory-bound at 256^3; LLVM codegen quality vs Kokkos/clang unmeasured. |
| I18 | native | Small plane kernel with `ti.static` loops over closure rows and taps; closure matrices as constants or a small field. | Equal to C++; tiny. |
| I19 | workaround | ELL gather-dot kernel with index and weight tables, scatter to `out[q]`. | Equal work, one more launch; about 0.5-8 ns per gathered element (survey estimate). |
| I20 | workaround | Row-parallel CSR kernel, or better compact ELL/COO with `ti.atomic_add` where rows repeat. | Roughly C++; avoids the full-height row-pointer sweep. |
| I21 | native | Compact COO loop over nnz; unique rows need no atomics. | Faster than C++, which sweeps 16.8M row pointers per call. |
| I22 | native | Face-slab kernel adding the Neumann extra term. | Faster than C++; no-op without Neumann BC. |
| I23 | native | Fixed-width ELL gather-dot over R(r) rows. | Equal to C++; launch-bound at small sizes. |
| I24 | native | Same ELL kernel with R(r) indices, or merged into I23 over a concatenated source. | Equal or better. |
| I25 | native | Fold into the first writer (assign instead of `+=`) or `field.fill(0)`. | Cheap; avoiding the fill is a bonus. |
| I26 | native | Fuse into the operator kernel's final write, or a flat elementwise loop. | Equal to C++. |
| I27 | workaround | Single kernel summing three directional contributions per point is possible (plain imperative code); eq-semantics on uncovered points must be made explicit by the DSL. | Equal to C++; fusing 3 axes could cut memory traffic about 3x (estimate). |
| I28 | native | One kernel writing three outputs, or three kernels. | Equal to C++. |
| I29 | native | Fused elementwise kernel over `ti.ndrange`. | Equal to C++. |
| I30 | native | Mask-multiplied FMA, or indexed loop over a sorted unique index array. | Equal to C++; mask form does 2-6% redundant work but coalesces on GPU. |
| I31 | native | Slice loops over a face via `ti.ndrange` on two axes, or masked where; R lists are indexed loops. | Equal to C++. |
| I32 | workaround | No automatic whole-RHS fusion. Write one large `@ti.kernel` or `@ti.func` chain so the RHS is one Python-to-runtime call; each top-level loop is still its own offloaded task. No jit/scan of the whole step. The AOT graph API (`ti.graph.GraphBuilder`) exists but is not a dynamic-Metal replay mechanism (unverified). | CPU: a few microseconds per call plus about 1 us per task, probably near the 30-40 us bar (unmeasured). Metal: each task is a compute-pass dispatch with substantial fixed cost (unmeasured). |
| I33 | native | Ordinary Python function calling kernels; GPU calls are asynchronous, `ti.sync()` fences. | Per-call Python overhead (unverified, microseconds to tens of microseconds) exceeds C++ unless calls are grouped. |
| I34 | native | `@ti.func` using `ti.exp`, `ti.cos`, `ti.pow` in a fused kernel; time as a float argument. | Faster than C++ virtual host calls (expected). |
| I35 | unsupported | Kernels cannot call Python callbacks. Replace with a `@ti.func` or SymPy-generated function, or precompute on the host. | Not applicable. |
| I36 | native | Source as a `@ti.func` in the RHS kernel with `t` as scalar argument. | Better than C++; no host round trip. |
| I37 | native | Fused axpby kernels over a flat packed array, one launch per stage. | Equal to C++ per pass (0.25-0.4 ns/elem); fewer launches if packed. |
| I38 | native | Ndarray swap on the Python side or a copy kernel; no baked-pointer constraint outside the AOT graph. | Free if swapped. |
| I39 | native | Boundary-only kernel evaluating the exact solution on faces and Dirichlet R points. | Much faster than C++, which evaluates the whole field. |
| I40 | native | Hoist to setup; Python holds arrays. | Free. |
| I41 | workaround | `ti.atomic_max/min` into a 0-d ndarray; argmax needs a second pass or packed value-index trick; read back via `to_numpy()` or scalar return. | Equal on CPU; atomic contention and device-to-host sync cost on GPU. |
| I42 | native | `to_numpy()` then NumPy writes; gate the error-field kernel on dump-due. | Equal to C++; I/O bound. |
| I43 | native | Python `while` loop around kernels. | Negligible at tens to thousands of steps. |
| I44 | native | Host SciPy/NumPy assembly; Taichi not involved. | Not applicable. |
| I45 | native | NumPy. | Not applicable. |
| I46 | native | Direct elementwise kernels; no expression templates needed. | Equal to C++. |
| I47 | workaround | Leading component axis (ncomp,nx,ny,nz), pointwise `@ti.func` for flux and primitives, 15 directional derivatives batched over the component axis, normal-projection kernel for Euler walls. | Roughly C++ on CPU; flux written inside the stencil kernel avoids temporaries. f32 Euler on Metal is an accuracy risk. |
| I48 | native | NumPy/SciPy unchanged. | Not applicable. |
| I49 | native | Unchanged SymPy; retarget the printer or use lambdify. | Setup only. |
| I50 | native | Host SciPy eigensolves; Taichi has no eigensolver, so the DSL must export matrices. | Not applicable. |
| I51 | native | In-process Python solver replaces the subprocess loop. Torch can coexist but shares no GPU memory space with Taichi on Metal (unverified). | Better than C++ for the BO loop; no process start per design point. |

## Code sketches

The sketches are illustrative. The verifier found two of the survey's originals invalid; they are corrected or labelled below. None have been compiled or run.

### Per-line block matvec (I16, I17, I18, I19)

The survey's version contained a malformed line (`u.flat(...) if False else ... uf[...]`) and an undefined `uf`. This version assumes the caller also passes flat 1-D views `uf` and `duf` of the same buffers; whether Taichi permits two ndarray arguments aliasing one buffer is unverified.

```python
@ti.kernel
def dxx(u: ti.types.ndarray(), du: ti.types.ndarray(),
        uf: ti.types.ndarray(), duf: ti.types.ndarray(),   # flat views (aliasing unverified)
        inner: ti.types.ndarray(), c: ti.types.ndarray(),
        wall_W: ti.types.ndarray(),
        cl_row: ti.types.ndarray(), cl_idx: ti.types.ndarray(),
        cl_C: ti.types.ndarray(), nclos: ti.i32):
    # (a) interior, whole grid, masked; x is the strided axis
    for i, j, k in ti.ndrange(NX, NY, NZ):
        if inner[i, j, k] != 0:
            s = 0.0
            for t in ti.static(range(W)):
                s += c[t] * u[i + t - P, j, k]
            du[i, j, k] += s
    # (b) wall closure on the first plane, uniform across lines
    for j, k in ti.ndrange(NY, NZ):
        for r in ti.static(range(R)):
            s = 0.0
            for t in ti.static(range(T)):
                s += wall_W[r, t] * u[t, j, k]
            du[r, j, k] = s
    # (c) per-fragment object closures, one ELL row each
    for q in range(nclos):
        s = 0.0
        for t in ti.static(range(TMAX)):
            s += cl_C[q, t] * uf[cl_idx[q, t]]
        duf[cl_row[q]] = s   # overwrites interior result; inner mask excludes closure rows
```

### O/B CSR cut-cell coupling to R points (I21, I23, I24)

```python
# compact COO for B (R->D); ELL for Bf (D->R) and Br (R->R)
@ti.kernel
def cut_coupling(uD: ti.types.ndarray(), uR: ti.types.ndarray(),
                 duD: ti.types.ndarray(), duR: ti.types.ndarray(),
                 b_row: ti.types.ndarray(), b_col: ti.types.ndarray(),
                 b_w: ti.types.ndarray(), nb: ti.i32,
                 f_idx: ti.types.ndarray(), f_w: ti.types.ndarray(),
                 r_idx: ti.types.ndarray(), r_w: ti.types.ndarray(),
                 nR: ti.i32):
    for e in range(nb):                 # rows unique, no atomics
        duD[b_row[e]] += b_w[e] * uR[b_col[e]]
    for q in range(nR):                 # one thread per R point
        s = 0.0
        for k in ti.static(range(KF)):  # D -> R, padded with weight 0
            s += f_w[q, k] * uD[f_idx[q, k]]
        for k in ti.static(range(KR)):  # R -> R self-coupling, merged
            s += r_w[q, k] * uR[r_idx[q, k]]
        duR[q] += s
```

### Gather/scatter boundary application (I30, I31)

Pseudo-code: `exact`, `coord_x`, `coord_y` are undefined helpers that would be `@ti.func`s.

```python
@ti.kernel
def apply_bc_and_source(du: ti.types.ndarray(), src: ti.types.ndarray(),
                        fluid_idx: ti.types.ndarray(), nfl: ti.i32,
                        dir_idx: ti.types.ndarray(), ndir: ti.i32,
                        u: ti.types.ndarray(), t: ti.f64):
    for q in range(nfl):                # sorted unique, no atomics
        i = fluid_idx[q]
        du[i] += src[i]
    for q in range(ndir):               # Dirichlet: rhs = 0
        du[dir_idx[q]] = 0.0
    for q in range(ndir):               # state: exact value, boundary only
        i = dir_idx[q]
        u[i] = exact(coord_x(i), coord_y(i), t)   # hypothetical @ti.func
# On GPU, prefer du[I] += mask[I]*src[I] over ndrange for coalescing.
```

### Packed scalar algebra with RK4 (I37, I38)

```python
# pack D|Rx|Ry|Rz into one flat f64 ndarray per slot: u0, u1, rk, k
@ti.kernel
def assign_lc(d: ti.types.ndarray(), a: ti.types.ndarray(),
              b: ti.types.ndarray(), c: ti.f64):
    for i in d:
        d[i] = a[i] + c * b[i]

@ti.kernel
def acc(d: ti.types.ndarray(), b: ti.types.ndarray(), c: ti.f64):
    for i in d:
        d[i] += c * b[i]

def rk4(u0, u1, rk, k, t, dt):
    rk.fill(0); u1.copy_from(u0)
    for s, (cs, ca) in enumerate([(0.0, 1/6), (.5, 1/3), (.5, 1/3), (1., 1/6)]):
        if s: assign_lc(u1, u0, k, cs * dt); update_bc(u1, t + cs * dt)
        rhs(u1, k, t + cs * dt)         # one kernel, ideally
        acc(rk, k, ca * dt)
    assign_lc(u1, u0, rk, 1.0); update_bc(u1, t + dt)
# about 9 launches per step with packing; the loop is Python-side
```

## Performance assessment

All figures in this section are architectural estimates. The survey ran no benchmarks, and the verifier found no source for the launch-overhead claims.

**Small 2D (2.6k to 5k points).** The C++ bar is about 30-40 us per RHS (about 0.8 us per graph node). Taichi would need the entire RHS in one `@ti.kernel` of roughly 10-15 top-level loops. Each top-level loop is still a separate offloaded task, so thread-pool dispatch for such small loops may consume a few microseconds each and land near or above the bar. A per-operation eager design with a Python call per op would plausibly lose by 5-10x. On Metal, per-dispatch fixed cost is higher and no replay mechanism is known. This regime is the main threat to the "no performance hit" constraint.

**Large 3D (256^3).** Bandwidth-bound. A collapsed parallel outer loop with a contiguous inner loop should be competitive with the C++ one-team-per-line kernel (13.7 ns/pt at 64^3 slow axis), possibly better. LLVM codegen quality versus Kokkos/clang is unmeasured. CUDA should approach peak bandwidth. Metal f32 halves bytes per point but raises an accuracy question for small-psi closures.

**Dispatch overhead.** Each Python kernel call crosses the Python/runtime boundary with ndarray argument marshalling. The microsecond-to-tens-of-microseconds figure is unverified. Calling `@ti.func` from kernels is inlined and avoids Python cost but not per-task cost.

**JIT warm-up.** First call of each kernel compiles via LLVM or MSL, typically hundreds of milliseconds to seconds; large fused kernels with unrolled loops sit at the long end (unverified estimate). The offline cache removes this on later runs. Passing geometry as ndarray data rather than baked constants avoids per-geometry recompiles.

**Fusion.** Taichi provides none across kernels, no step tracing, no XLA-like compiler. Ordering between top-level loops inside one kernel is preserved, so accumulate chains are safe. Fusion of zero-fill, axes, closures, scale, source, and Dirichlet fill must be done by the DSL code generator. The "no fusion" statement is the survey's reading of the kernel model and is unverified.

## Suitability as a DSL substrate

Strengths: imperative loops map ragged data (ELL closures, COO couplings, gather lists, per-line tables) without contortion, unlike array-programming frameworks. `ti.static` unrolling makes tap counts and closure sizes compile-time constants. A generator can emit one big kernel per RHS for manual fusion. CPU and CUDA give f64. The DSL could emit Taichi source by string templates or Python AST.

Weaknesses: no automatic fusion, per-task dispatch cost threatening the small-2D bar, no f64 and no sparse on Metal, no known graph replay on Metal, a restrictive Python AST frontend (generated code must stay in a conservative subset), harder debugging of JIT kernels than C++, and a slow-moving project.

Recommendation: a backend abstraction in the DSL with C++/Kokkos CPU as the parity backend, and Taichi as an optional experimental backend for Metal f32 and CUDA. A poor choice as the single dependency.

## Risks

1. Project health: no release since 2025-07-31, 878 open issues; cut-cell kernel bugs may not be fixed upstream.
2. No f64 on Metal; accuracy of small-psi high-order closures in f32 is unproven.
3. Sparse SNodes removed on Metal in v1.4.0; sparse structures must be hand-built from dense arrays.
4. No whole-RHS fusion or confirmed graph replay on dynamic Metal; small 2D cases at risk from dispatch and Python call overhead.
5. Restricted Python frontend subset; JIT debugging is harder than C++.
6. Metal AOT/TiRT is experimental (tier 2; Vulkan is the only tier-1 backend), so a C++ embedding path is shaky.
7. Seconds of JIT warm-up for large fused kernels, mitigated by the offline cache.
8. CPU parity and per-task overhead claims rest on architecture, not measurement. A benchmark spike on a 2D RHS is needed before any decision.
9. Taichi cannot call the existing Kokkos C++ from kernels; calling Taichi from C++ is only via the experimental AOT/TiRT path.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| Latest release v1.7.4, 2025-07-31, nothing newer on PyPI | Confirmed | PyPI lists 1.7.4 (Jul 31, 2025) above 1.7.3 (Dec 23, 2024); page may be cached. |
| Sparse SNodes removed on Metal in v1.4.0, deprecated v1.3.0 | Confirmed | v1.4.0 notes: "Support for sparse SNodes on the Metal backend has been removed." Docs: Dynamic removed v1.3.0, Pointer/Bitmasked v1.4.0. |
| Metal backend lacks f64 | Confirmed | Type docs support table marks f64 unsupported on Metal; 2020 changelog says Metal only supports 32-bit types. |
| Metal AOT runtime (TiRT) added v1.5.0, experimental | Confirmed | v1.5.0 notes (2023-03-27); C-API docs list Metal tier 2, Vulkan only tier 1. |
| 878 open issues | Confirmed | Repo page: 878 issues, 47 PRs (cached summary). |
| CC backend removed in v1.7.0 | Confirmed | v1.7.0 notes (2023-11-27). Survey wrongly grouped it with 2025 packaging work. |
| v1.7.x was packaging work implying maintenance mode | Uncertain | 1.7.2 to 1.7.4 are packaging/bug fixes, but 1.7.0 was a feature release; no explicit status statement in the repo. |
| "Dropped driver support" in v1.7.4 | Not supported | Notes show manylinux2014 and old macOS dropped, nothing on drivers. Omitted from this document. |
| No cross-kernel fusion or whole-step graph on Metal; AOT graph not equivalent | Uncertain | No evidence either way. C-API compute graph stable since 1.4.0 with Vulkan primary; docs/graph URL returned 404. `ti.graph.GraphBuilder` name and "CUDA and Vulkan" list unconfirmed. |
| Per-call Python launch overhead of microseconds to tens of microseconds | Uncertain | No source; survey flags it as unmeasured. |
| Per-task dispatch cost, CPU parity, JIT times | Unverified | Survey estimates only; no benchmarks. |
| ndarrays flat-indexable with no padding; NumPy zero-copy; torch interop on Metal | Unverified | Not checked. |
| Code sketch for I16 | Refuted as written | Malformed line and undefined `uf`; corrected above, still uncompiled. |
| Code sketch for I30 | Pseudo-code | Undefined `exact`, `coord_x`, `coord_y` helpers; labelled as such. |

## Sources

- https://pypi.org/project/taichi/
- https://github.com/taichi-dev/taichi/releases
- https://github.com/taichi-dev/taichi/releases/tag/v1.2.0
- https://github.com/taichi-dev/taichi/releases/tag/v1.4.0
- https://github.com/taichi-dev/taichi
- https://docs.taichi-lang.org/docs/type
- https://docs.taichi-lang.org/docs/sparse
- https://docs.taichi-lang.org/docs/taichi_core
- https://docs.taichi-lang.org/docs/hello_world
- https://taichi-zh.readthedocs.io/zh-cn/master/type.html
- https://developer.apple.com/forums/thread/709663
