# NVIDIA Warp as a SHOCCS substrate

> **Verdict box**
> - **Fit for SHOCCS:** Good kernel-level expressiveness for cut-cell idioms; not sufficient as the single substrate.
> - **CPU parity:** Poor. CPU launches run serially and synchronously on one host thread (confirmed in the FAQ).
> - **Apple GPU:** None. macOS arm64 is "CPU Only"; no Metal or MPS backend in docs or changelog.
> - **fp64:** Supported on CPU and CUDA, but `wp.float` is fp32, so the DSL must declare float64 everywhere.
> - **Risk:** Medium to high. NVIDIA-controlled and CUDA-first, monthly releases with API churn, CPU graph capture is experimental.

## Status as of 2026-10-12

- Latest release: **1.18.0 (2026-10-05)**. Earlier: 1.17.0 (2026-08-31), 1.16.0 (2026-08-03), 1.15.0 (2026-07-07), 1.14.0 (2026-06-01). The original survey named 1.17.0 as latest; verification corrected this.
- A 1.16.1 bugfix dated 2026-08-24 was claimed in the survey but could not be found in the visible (truncated) changelog. Treat as unverified.
- Monthly feature cadence. Python kernels compile to C++ (LLVM/Clang, CPU) or CUDA (NVRTC). Mainly used for simulation and robotics.
- CPU graph capture (`wp.capture_begin/end/capture_launch`) has existed since 1.13.0 and was extended in 1.14.0 and 1.16.0. The docs state it is experimental and may change without a formal deprecation cycle.
- Since 1.17.0 there are experimental APIs for compiling Warp modules for external C++/CUDA consumers; 1.18.0 ships `warp.lib` and a `warp_clang.h` header so native C++ applications can link to Warp. APIC serialization (`.wrp` files) is part of the CPU graph feature. Both make the C++ interop story better than the survey first stated, but they are experimental and not evaluated here.

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| CPU (Linux, Windows, macOS arm64) | Supported, serial and synchronous per launch | Full float64 | No multithreading of a single launch. `warp.config.enable_cpu_blocks` runs cooperative fibers on one thread for tile kernels, so it adds no parallelism. |
| CUDA (Linux, Windows) | First-class: graphs, tiles, sparse BSR, mempools | Full float64, rate limited on consumer GPUs | Not relevant to an Apple-only workstation; useful for clusters. |
| Metal / MPS | None | Not applicable | Docs list macOS as CPU only. No Metal entry in the 1.14 to 1.18 changelog (older entries unread). No roadmap item found. |
| ROCm, SYCL, Vulkan | None | n/a | |
| Windows arm64 | Experimental | n/a | CPU; CUDA only via source builds. |

fp64 notes: `wp.float` and `float` alias fp32, and there are no implicit conversions, so every array, constant and local must be explicitly float64. The float64 vector and matrix types are expected from general knowledge and were not seen directly on the docs page fetched (partly verified).

## Idiom mapping

Support values: native, workaround, unsupported. "Setup" means the idiom runs once at initialization, so runtime performance is not a concern.

| Idiom | Support | How | Expected performance |
|---|---|---|---|
| I01 | native | Dict or dataclass of `wp.array(dtype=wp.float64)`, or one packed array with static offsets, or a `wp.struct` of 4 arrays. | Equal to C++; setup. |
| I02 | workaround | Host NumPy ray casting (optionally `wp.Mesh` and `mesh_query_ray`). Output SoA arrays. | Setup; faster than C++ virtual-call loop if vectorised. |
| I03 | workaround | NumPy merge-join and lexsort on host; Warp kernel per pencil possible. | Setup. |
| I04 | native | NumPy masks or index arrays uploaded as `wp.bool` or `wp.int32` arrays; face slices via 3D views. | Setup. |
| I05 | native | Python floats baked in as `wp.constant` or closure constants; dynamic kernels via captured constants. | Equal. |
| I06 | native | Closed-form psi in a `wp.func` or kernel; SymPy printer can be retargeted to Warp Python; or NumPy lambdify. | Setup; fp64 required. |
| I07 | workaround | Host NumPy `searchsorted`, or a branching kernel per R point. | Setup. |
| I08 | workaround | Host Python/NumPy grouped by closure kind; emit line and closure tables. | Setup. |
| I09 | workaround | Host NumPy, or one launch over R points writing ELL tables. | Setup. |
| I10 | native | Compact COO/ELL with `np.lexsort`; `warp.sparse.bsr_from_triplets` is an option. | Setup. |
| I11 | native | `np.cumsum` and concatenate, then upload. | Setup. |
| I12 | native | Plain Python config replaces Lua. | n/a |
| I13 | native | 3D arrays indexed `a[i,j,k]` (max 4 dims), launched with `dim=(nx,ny,nz)` and `wp.tid()`. Flat 1D with strides also works. | Equal on CUDA; single thread on CPU. |
| I14 | native | Same SoA R arrays, ordering via `np.lexsort`. | Equal. |
| I15 | native | Pass arrays to kernels; `wp.struct` bundling of buffers; zero-copy NumPy view on CPU. | Equal. |
| I16 | workaround | Decompose into whole-grid interior stencil (I17) plus ELL closure rows (I18/I19), or a thread-per-line kernel with per-line metadata in a struct array (see sketch 1). Tile APIs are CUDA oriented. | CUDA: parity or better. CPU: serial, roughly 1/ncores of Kokkos OpenMP (estimate). |
| I17 | native | Kernel over (i,j,k) with an unrolled tap loop; taps as `wp.constant`; interior mask or bounds. | CUDA bandwidth bound, equal to hand CUDA. CPU: single thread, estimated 2 to 4x slower than multithreaded C++ at large 3D (unmeasured). |
| I18 | native | Kernel over face planes with small unrolled R x T matmul; fixed-size locals supported. | Negligible; equal. |
| I19 | workaround | ELL tables applied by one kernel over rows, overwriting stencil results. | Equal on CUDA; tiny on CPU. |
| I20 | native | `warp.sparse` BSR (`bsr_mv`) exists, but a custom ELL or COO kernel is simpler. 1x1 blocks work. | Comparable; CPU serial. |
| I21 | native | Compact COO kernel with `wp.atomic_add` (sketch 2), one thread per nonzero. | Likely faster than C++ full-height row sweep. |
| I22 | native | Plane kernel adding `extra[k]*g_face`. | Equal or better. |
| I23 | native | ELL gather-dot kernel, one thread per R row, padded index. | Equal; launch bound when small. |
| I24 | native | Merge into I23 kernel over concatenated source, or a second kernel. | Equal. |
| I25 | native | `array.zero_()` or write output directly in the stencil kernel. | Equal; fusion is manual. |
| I26 | native | Fold into stencil kernel or an elementwise kernel. | Equal. |
| I27 | workaround | One fused Laplacian kernel summing three axis contributions per point (no atomics), then closure and CSR kernels; ordering by launch order. | Equal or better than C++ by hand fusion. |
| I28 | native | One kernel writing three output arrays. | Equal. |
| I29 | native | Elementwise kernel or fused into gradient kernel. | Equal. |
| I30 | native | Mask or index-array kernel (sketch 3); unique indices so no atomics. | Equal. |
| I31 | native | Face kernels over 2D grids. | Equal. |
| I32 | workaround | CUDA: `wp.ScopedCapture` graph. CPU: experimental `capture_begin/end/capture_launch` replays from a C++ loop. Pointer stability applies. | No cross-kernel fusion; graph removes launch overhead only. CPU replay near C++ dispatch cost is a hypothesis (unmeasured). |
| I33 | native | Python functions calling `wp.launch`; each CPU launch is synchronous. | Eager Python overhead estimated tens of microseconds per launch (unmeasured). Loses at small sizes. |
| I34 | native | Kernel with closed-form sums using `wp.exp`, `wp.cos`, `wp.pow`; time as scalar argument. | CUDA far faster than host calls; CPU similar or better than virtual-call loop. |
| I35 | unsupported | No per-point Python callback inside kernels; replace with generated Warp kernel or precomputed table. | n/a (replaced) |
| I36 | native | Pass `t` as kernel argument. Under capture, `t` must be a device array or updated via parameter setting. | Equal or better. |
| I37 | native | Fused axpby kernel per buffer or one launch over a packed array (sketch 4). | Equal (bandwidth bound). |
| I38 | native | `wp.copy` or `array.assign`. | Equal. |
| I39 | native | Boundary kernel evaluating exact solution only on faces and Dirichlet R points. | Better than C++. |
| I40 | native | Hoist to setup. | Better. |
| I41 | native | `wp.utils.array_inner/array_sum`, or kernel with `wp.atomic_max/min` into a 1-element array; masked argmax needs two steps. | Equal. |
| I42 | native | Error kernel gated by dump interval; `arr.numpy()` then Python writes binary and XDMF. | Equal. |
| I43 | native | Python loop around launches or a captured graph. | Equal. |
| I44 | native | Assemble `scipy.sparse` from tables, then `numpy.linalg.eig`. | n/a (host diagnostic). |
| I45 | native | NumPy oracle. | n/a |
| I46 | workaround | No expression templates; users write fused kernels. A DSL layer must generate kernels (runtime kernel construction from source or closures). | Equal to hand-fused. |
| I47 | native | `wp.vec3d/vec4d` locals, structs for state, batched derivatives via extra launch dim. | Equal on CUDA. |
| I48 | native | NumPy/scipy twin. | n/a |
| I49 | native | SymPy to Warp-source printer; kernels are Python source. | Setup. |
| I50 | native | Host scipy. A BSR to scipy export helper may exist, but the name `warp.sparse.bsr_to_scipy` is unverified. | n/a |
| I51 | native | In-process Python solver; torch/jax/DLPack interop and differentiable `wp.Tape` for design loops. | n/a |

Unsupported idioms: I35 only. Workarounds are required for I02, I03, I07, I08, I09, I16, I19, I27, I32, I46.

## Code sketches

These are from the survey. They have not been compiled or run. The bracket syntax `wp.array3d[wp.float64]` used in the survey's idiom table is unverified; the docs show `wp.array3d(dtype=wp.float64)` in kernel signatures, which the sketches use.

Per-line block matvec with strides and per-line dense closures (thread per line, direct port of the Kokkos team kernel):

```python
@wp.struct
class Line:
    out0: int; in0: int; stride: int
    nl: int; nint: int; nr: int
    lcols: int; rcols: int
    lco: int; rco: int; rin0: int

@wp.kernel
def block_mv(lines: wp.array(dtype=Line), coef: wp.array(dtype=wp.float64),
             x: wp.array(dtype=wp.float64), b: wp.array(dtype=wp.float64),
             taps: wp.array(dtype=wp.float64), w: int):
    L = lines[wp.tid()]
    for r in range(L.nl):
        s = wp.float64(0.0)
        for j in range(L.lcols):
            s += coef[L.lco + r*L.lcols + j] * x[L.in0 + j*L.stride]
        b[L.out0 + r*L.stride] += s
    for r in range(L.nint):
        o = L.out0 + (L.nl + r)*L.stride
        s = wp.float64(0.0)
        for j in range(w):
            s += taps[j] * x[o + (j - w//2)*L.stride]
        b[o] += s
    for r in range(L.nr):
        s = wp.float64(0.0)
        for j in range(L.rcols):
            s += coef[L.rco + r*L.rcols + j] * x[L.rin0 + j*L.stride]
        b[L.out0 + (L.nl + L.nint + r)*L.stride] += s
# wp.launch(block_mv, dim=n_lines, inputs=[...]). On CPU this loop is serial; on CUDA prefer the
# whole-grid stencil kernel (interior) + ELL closure kernel for coalescing.
```

O/B CSR cut-cell coupling to R points (compact COO and ELL gather):

```python
@wp.kernel
def B_apply(row: wp.array(dtype=wp.int32), col: wp.array(dtype=wp.int32),
            w: wp.array(dtype=wp.float64), uR: wp.array(dtype=wp.float64),
            du: wp.array(dtype=wp.float64)):
    n = wp.tid()
    wp.atomic_add(du, row[n], w[n] * uR[col[n]])   # R -> D, compact COO

@wp.kernel
def Bf_apply(idx: wp.array2d(dtype=wp.int32), C: wp.array2d(dtype=wp.float64),
             uD: wp.array(dtype=wp.float64), duR: wp.array(dtype=wp.float64), K: int):
    q = wp.tid()
    s = wp.float64(0.0)
    for k in range(K):               # padded: idx -> dummy slot, C -> 0
        s += C[q, k] * uD[idx[q, k]]
    duR[q] += s                      # D -> R(r) rows, unique rows so no atomic
# wp.launch(B_apply, dim=nnz, ...); wp.launch(Bf_apply, dim=nR, ...)
```

Gather/scatter boundary application (Dirichlet zero, masked source add, faces):

```python
@wp.kernel
def src_add(mask: wp.array(dtype=wp.bool), src: wp.array(dtype=wp.float64),
            du: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if mask[i]:
        du[i] += src[i]

@wp.kernel
def dir_zero(idx: wp.array(dtype=wp.int32), du: wp.array(dtype=wp.float64)):
    du[idx[wp.tid()]] = wp.float64(0.0)   # flattened Dirichlet list

@wp.kernel
def face_x0(u: wp.array3d(dtype=wp.float64), val: wp.array2d(dtype=wp.float64)):
    j, k = wp.tid()
    u[0, j, k] = val[j, k]
# launch: dim=D.size, dim=len(idx), dim=(ny, nz)
```

Four-buffer scalar algebra with RK4 slot updates:

```python
@wp.struct
class Field:
    D: wp.array(dtype=wp.float64)
    Rx: wp.array(dtype=wp.float64)
    Ry: wp.array(dtype=wp.float64)
    Rz: wp.array(dtype=wp.float64)

@wp.kernel
def axpby(d: wp.array(dtype=wp.float64), a: wp.array(dtype=wp.float64),
          c: wp.float64, r: wp.array(dtype=wp.float64)):
    i = wp.tid()
    d[i] = a[i] + c * r[i]

def assign_lc(dst, a, c, r):             # one launch per non-empty buffer
    for n in ('D', 'Rx', 'Ry', 'Rz'):
        d = getattr(dst, n)
        if d.size:
            wp.launch(axpby, d.size, [d, getattr(a, n), c, getattr(r, n)])
# Capture one RHS + stage update with wp.ScopedCapture() and replay with wp.capture_launch(g).
# Packing D|Rx|Ry|Rz into one array cuts launches 4x.
```

## Performance assessment

All numbers below are estimates from the survey unless stated; none were measured.

**Small 2D (about 40 nodes per RHS, bar of 30 to 40 microseconds).** Eager `wp.launch` costs an estimated 10 to 50 microseconds per call, and the docs say Python dispatch dominates cheap CPU kernels. With about 40 launches per RHS this is likely 0.5 to 2 ms, which fails the bar by more than an order of magnitude. Experimental CPU graph capture moves the loop into C++ and could approach 1 to 3 microseconds per kernel, but that is a hypothesis and the feature is experimental. Compute itself is not the issue at 2.6k to 5k points.

**Large 3D (64^3 to 256^3).** CUDA is bandwidth bound and can match hand CUDA with a fused, coalesced stencil. On CPU, serial launches make this one-core memory bandwidth, plausibly 2 to 5x slower than Kokkos OpenMP on a multicore machine (estimate). Recovering parallelism would require splitting the domain across Python threads or processes; whether `wp.launch` releases the GIL is unverified.

**Dispatch overhead.** Pre-recorded `wp.Launch` objects (`record_cmd=True`) reduce parameter marshalling. CPU graph capture replays a recorded sequence in a C++ loop. CUDA graphs are mature. For sparse BSR, `nnz_sync()` cannot read a topology update recorded in the same CPU capture and compact `bsr_mm` needs a host readback on CUDA, so it is unavailable during capture (confirmed in docs); SHOCCS should avoid these paths by using fixed ELL/COO tables.

**Fusion.** No cross-kernel fusion is documented (the claim that none exists is plausible but not stated in the docs, so it is unverified). Each launch is a separate kernel, and graph capture removes launch overhead only, not memory passes. Fusion must be done by the kernel author or by a DSL that generates fused kernels. First-use compile takes seconds, then is cached on disk.

## Suitability as a DSL substrate

Strengths: kernels are ordinary Python source with typed structs, arrays up to 4D, fp64, atomics, local fixed arrays, branching and loops. Kernels can be generated at runtime from strings or closures, so a SHOCCS DSL could emit one fused kernel per RHS and sidestep the missing auto-fusion. Ragged lines, ELL gathers and COO scatters map naturally to explicit-index kernels, which is a better match than array-programming frameworks. `wp.Tape` differentiation and `warp.sparse` are extras relevant to optimisation loops.

Blockers: serial CPU launches defeat parity with Kokkos OpenMP on multicore at 3D sizes, and the absence of Metal fails the MPS goal. Eager dispatch overhead misses the small-2D bar unless experimental CPU graph capture holds up.

Conclusion: a strong design reference and a viable CUDA code-generation target. As a single substrate it would need a second backend for Apple GPU and a multithreaded CPU path. The new external-C++ module compilation (1.17/1.18) may be worth a follow-up evaluation as a way to embed generated kernels in the existing C++ solver, but it is untested here.

## Risks

- Serial, synchronous CPU launches (confirmed): no CPU parity with Kokkos OpenMP at mid and large 3D sizes.
- No Metal or MPS backend and no sign of one; writing one is a large effort.
- Eager launch overhead misses the small-2D bar; the mitigation (CPU graph capture) is experimental and its API may change without a deprecation cycle.
- No documented cross-kernel fusion; performance depends on the DSL generating fused kernels.
- NVIDIA-controlled and CUDA-first, so CPU and macOS may lag.
- fp64 pitfalls: `wp.float` is fp32 and there are no implicit casts, so silent precision loss is possible.
- JIT compile cost on first use and whenever geometry-dependent constants change.
- Monthly API churn.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| CPU kernel launches run serially and synchronously | Confirmed | FAQ: "CPU kernel launches currently run serially and synchronously." `enable_cpu_blocks` adds no parallelism. |
| macOS arm64 is CPU only, no Metal | Confirmed | Compatibility page lists macOS arm64 as "CPU Only"; FAQ agrees; no Metal in 1.14 to 1.18 changelog (older entries unread, changelog fetch truncated). Absence of evidence, but consistent. |
| Experimental CPU graph capture replays from a C++ loop and removes Python dispatch | Confirmed | Runtime docs mark it experimental; changelog: introduced 1.13.0, extended 1.14.0 and 1.16.0. |
| float64 supported, no implicit conversion, `wp.float` is fp32 | Confirmed (partly) | Runtime docs: float and int alias fp32 and int32; arrays limited to 4 dims. Float64 vec/mat types and the no-implicit-conversion text were not seen directly. |
| Latest release is 1.17.0 (2026-08-31) | Refuted | Latest is 1.18.0 (2026-10-05). 1.17.0 and its date are correct but not latest. |
| 1.16.1 bugfix on 2026-08-24 | Unverified | Not found in the visible, truncated changelog. |
| No automatic fusion across kernels | Uncertain | No official statement found either way; consistent with per-kernel launch model. |
| BSR limits under capture | Confirmed | Runtime docs: `nnz_sync()` and `status_sync()` limits; compact `bsr_mm` needs host readback on CUDA. CPU `bsr_mm` under capture not confirmed. |
| 1.17 adds experimental external C++/CUDA module compilation | Confirmed | Changelog; 1.18.0 adds `warp.lib` and `warp_clang.h`. Survey's "not a supported path" is outdated. |
| Windows arm64 experimental | Confirmed | Compatibility page. |
| `wp.launch` releases the GIL | Unverified | No doc evidence. |
| `warp.sparse.bsr_to_scipy` exists | Unverified | Not found in the changelog; use only as a pointer. |
| Launch overhead 10 to 50 microseconds; 2 to 4x slower than Kokkos; 15 to 30 GB/s | Unverified estimates | Unsourced survey estimates. |
| `wp.array3d[wp.float64]` bracket syntax | Unverified | Docs show `wp.array3d(dtype=wp.float64)`. |

Score adjustment: none. The decisive claims were confirmed, so the overall score remains 3.5 out of 10.

## Sources

- https://nvidia.github.io/warp/latest/project/changelog.html
- https://nvidia.github.io/warp/latest/user_guide/compatibility.html
- https://nvidia.github.io/warp/latest/user_guide/faq.html
- https://nvidia.github.io/warp/latest/user_guide/runtime.html
- https://github.com/NVIDIA/warp/issues?q=metal
- https://nvidia.github.io/warp/v1.16/api_reference/_generated/warp.launch_function.html
