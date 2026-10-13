# Contributing an Apple GPU backend to Kokkos (native Metal, or SYCL via AdaptiveCpp Metal)

> **Verdict:** Excluded. No Kokkos Metal or Apple-GPU backend exists, and building one is a multi-year compiler project.
> **Fit for SHOCCS:** Low. It does not serve the Python re-expression goal; useful only as a rejected alternative and a design reference.
> **CPU fp64 parity:** Unchanged (Kokkos Serial/OpenMP/Threads stay as they are). Score 8 of 10, but the Metal work adds nothing here.
> **Apple GPU:** Score 1 of 10. No hardware fp64 on Apple GPUs, so `real = double` cannot run natively.
> **Risk:** Very high (score 1 of 10). Experimental upstream stack, unsupported by Kokkos SYCL, estimated at years of effort.

Overall score (verifier-adjusted): 1.5. Other scores: idiom coverage 5, ergonomics 6.

## Status as of 2026-10-12

- Version checked: Kokkos v5.2.2 (tagged 2026-09-10). AdaptiveCpp `develop` Metal doc fetched 2026-10-12.
- No Kokkos Metal or Apple-GPU backend exists. The Kokkos requirements page lists only IntelLLVM as a SYCL compiler (minimum 2024.2.1 for 5.x, 2023.0.0 for 4.x). AdaptiveCpp and Metal are not mentioned.
- AdaptiveCpp's Metal backend is experimental and tested only on macOS 26 and Apple Silicon. It requires LLVM 15 or newer. It has no double (software emulation is only planned), emulates 64-bit atomics, commits regular SYCL events as separate command buffers (a coarse-grained-events extension reduces this), has no `sycl::stream` or printf, and restricts pointer-to-integer casts.
- A search snippet suggests AdaptiveCpp Metal USM support was being completed. This does not change the verdict.
- Unverified: absence of any Kokkos issue or PR about AdaptiveCpp or Metal (the issue tracker was searched only shallowly through web search).

## Backends and fp64

| Backend | Status | fp64 |
|---|---|---|
| CPU (Serial/OpenMP/Threads) | Works today; unchanged by this route and already the C++ bar | Full binary64 |
| CUDA | Works on NVIDIA; not the owner's target | Full binary64 |
| Native Kokkos::Metal | Does not exist. Would need execution space, memory space, parallel_for/reduce/scan, TeamPolicy, Graph, deep_copy, and a functor-to-AIR/MSL compilation path | None in hardware; software only |
| Kokkos SYCL on AdaptiveCpp Metal | No route today. Kokkos SYCL would need porting away from Intel oneAPI extensions, on an experimental stack | AdaptiveCpp: no double, emulation planned |
| Other (OpenMP offload, OpenCL) | No OpenMP offload to Apple GPUs; OpenCL on macOS is deprecated with no fp64 | None |

fp64 detail: AdaptiveCpp confirms that Apple Silicon GPUs lack hardware FP64. The broader claims that this holds for every chip from M1 to M5 and that MSL has no usable double were not independently checked against Apple documentation (unverified). If true, a Metal backend could run SHOCCS's `real = double` code only through df64 or software binary64 emulation, which would have to be built in the backend or compiler and does not exist upstream. This constraint is independent of how the backend is built.

## Mapping of the hard runtime shapes

| Idiom | How it maps | Support |
|---|---|---|
| Packed field vector [D\|Rx\|Ry\|Rz] | `Kokkos::View<real*>` with subviews. Trivial in Kokkos, and unified memory makes deep_copy cheap. Blocker: element type would have to be float | workaround |
| Masked axis stencil (3D, interior mask) | MDRangePolicy or flat RangePolicy with a mask check. A Metal backend would need MDRangePolicy tiling built from scratch | workaround |
| ELL gather-dot-scatter (closure rows) | RangePolicy over rows with a fixed-width inner loop. Atomics only if rows collide; 64-bit atomics are emulated in AdaptiveCpp Metal. Disjoint rows avoid the issue | workaround |
| Face-plane assignment | Strided or gather selection-descriptor kernels, plain parallel_for | native |
| Masked reduction | parallel_reduce with a mask. A new backend needs its own reduction implementation, and it would be fp32-only on Metal | workaround |
| Whole-step compilation (RK4 in one graph) | `create_graph<execution_space>` is already used by SHOCCS. A Metal Graph node could map to a command buffer or indirect command buffer. With AdaptiveCpp each regular event gets its own command buffer, which is likely poor for a launch-bound 51x51 step (inference; the coarse-grained-events extension may mitigate it) | workaround |
| Expression-template field algebra | Compiles to parallel_for functors. Needs functor-to-MSL compilation to handle templates, lambdas and Kokkos headers | workaround |
| TeamPolicy / hierarchical parallelism | Not used in SHOCCS's hot path; pure backend-completeness overhead | not applicable |

## Code sketch

```cpp
// What the user would write is unchanged host-style Kokkos; all cost is in the backend.
// Axis stencil + ELL closure, per RK stage (SHOCCS style):
Kokkos::parallel_for("axis", MDRangePolicy<execution_space, Rank<3>>({1,1,1},{nx-1,ny-1,nz-1}),
  KOKKOS_LAMBDA(int i,int j,int k){
    if (mask(i,j,k)) du(i,j,k) = c0*u(i,j,k) + c1*(u(i+1,j,k)-u(i-1,j,k)); });
Kokkos::parallel_for("ell", RangePolicy<execution_space>(0,nrows),
  KOKKOS_LAMBDA(int r){ real s=0;
    for (int w=0; w<W; ++w) s += val(r,w)*u(col(r,w));
    du(row(r)) = s; });
// Requires execution_space = Kokkos::Experimental::Metal (does not exist);
// with real=double it cannot run on any Apple GPU.
```

## Performance assessment

This cannot be measured against the C++ bar because no backend exists. The following is estimation, not measurement.

- Small 2D (51x51, about 30 to 40 us per RHS, launch-bound): a new Metal backend or AdaptiveCpp would likely be slower than the CPU host path, since each launch costs tens of microseconds. Even with graph or indirect-command-buffer batching, at best comparable to the CPU.
- Large 3D (64^3, about 1.7 ms per RHS, bandwidth-bound): in fp32 an M-series Pro/Max GPU could plausibly beat the CPU. With df64 or software binary64 the gain largely disappears. The roughly 10x or more emulation cost is an unsourced estimate (unverified). Emulated 64-bit atomics would also slow ELL scatter if rows collide.
- Net: no demonstrated advantage at either size, at the price of a multi-year backend.

## What the DSL designs should copy or avoid

Copy:
- Kokkos' separation of execution space, memory space and View layout.
- The Graph abstraction, where the whole RK step is a recorded DAG. This is the right way to beat launch overhead.
- The reducer interface with an explicit identity.

Avoid:
- Generic functor-to-device compilation, TeamPolicy and the general execution model. A purpose-built DSL needs only about 5 kernel shapes, so generating MSL from a small IR is a far smaller job than porting Kokkos.
- Per-event command buffers (the AdaptiveCpp lesson). Batch many kernels per command buffer or use indirect command buffers.
- Hard-coding `real = double`. Make precision a parameter of the IR so fp32 or df64 can be selected for the GPU while fp64 stays on CPU.

## Deciding fact

Even a perfectly built Kokkos Metal backend would have no hardware fp64, and SHOCCS is fp64 (`real = double`). Beyond that, the backend does not exist, and the only SYCL route (AdaptiveCpp Metal) is experimental, macOS 26 only, has no double, and is not a Kokkos-supported SYCL compiler (the requirements page lists only IntelLLVM).

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| Kokkos 5.x requirements list only IntelLLVM (min 2024.2.1) for SYCL; no AdaptiveCpp or Metal mention | Confirmed | Requirements page fetched: only IntelLLVM listed; no mention of AdaptiveCpp or Metal |
| AdaptiveCpp Metal: experimental, macOS 26 / Apple Silicon only, LLVM 15+, no double (emulation planned), emulated atomic64, one command buffer per event, no sycl::stream/printf, limited ptrtoint casts | Confirmed | Every limitation appears in the fetched doc. A coarse-grained events extension reduces per-event overhead (omitted from the original assessment) |
| Latest Kokkos release is v5.2.2 (2026-09-10) with no Apple/Metal backend | Confirmed | Releases page shows v5.2.2, 10 September, no Metal mention. Page has no detailed notes, so absence is weakly shown |
| No Apple GPU (M1 to M5) has fp64 and MSL has no usable double | Uncertain | AdaptiveCpp doc confirms Apple Silicon lacks hardware FP64. M1 to M5 range (M5 not checked) and the MSL claim were not checked against Apple docs |
| Kokkos SYCL relies on Intel oneAPI extensions; reduction/device-global/printf bugs on AMD (IWOCL 2025) | Uncertain | Search summaries support the oneAPI reliance and AMD being open in 2025. The specific bug list was not verified; the PDF was not fetched |
| No Kokkos issue or PR for AdaptiveCpp or Metal | Uncertain | Web search found none; the tracker was not searched directly |
| AdaptiveCpp Metal needs LLVM 15+ and metal-cpp | LLVM 15+ confirmed; metal-cpp unverified | metal-cpp not seen in the fetched summary |
| Df64/software binary64 costs roughly 10x or more; "effectively a compiler project" | Unverified | Estimates with no cited source |
| OpenMPTarget, OpenACC, HPX, SYCL backends each took funded teams years | Unverified | Unsourced comparison |
| Kokkos SYCL is "tested only with Intel's compiler" | Softened | Requirements page only shows IntelLLVM as supported, which is weaker than "tested only" |
| AdaptiveCpp per-event command buffers are the worst case for launch-bound steps | Inference | Not measured; mitigated by the coarse-grained-events extension |

Counts: 3 confirmed, 0 refuted, 3 uncertain (plus several unsourced estimates marked unverified above).

## Sources

- https://kokkos.org/kokkos-core-wiki/get-started/requirements.html
- https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-metal.md
- https://github.com/kokkos/kokkos/releases
- https://www.iwocl.org/wp-content/uploads/iwocl-2025-daniel-arndt-kokkos.pdf (not fetched directly)
- https://impact.ornl.gov/en/publications/experiences-with-implementing-kokkos-sycl-backend/
- https://github.com/kokkos/kokkos/issues
- docs/python-dsl-survey/00-apple-gpu-facts.md
