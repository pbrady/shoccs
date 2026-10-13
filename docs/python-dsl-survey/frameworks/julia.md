# Julia (KernelAbstractions.jl + Metal.jl) as the non-Python comparator

| | |
|---|---|
| Verdict | Reference only (design inspiration, not an adoption candidate). Verifier-adjusted overall score 4.5 / 10. |
| Fit for SHOCCS | Low as a replacement strategy, moderate as a reference design. Adopting it means a second-language rewrite of the C++ core, which contradicts the Python re-expression goal. |
| CPU fp64 parity | Full Float64 support on CPU. Credible via native `@inbounds @simd` loops with Threads or Polyester. The new KA 0.10.0 CPU backend (OpenCL on PoCL) is unmeasured for launch-bound small cases. |
| Apple GPU | Float32, Float16 and BFloat16 only. No Float64 kernels, and no df64 package was found (unverified absence). Same ceiling as MLX and PyTorch MPS. |
| Risk | KA 0.10.0 is a breaking release published today and rebuilt its CPU backend. Metal.jl still carries "known limitations". Thin maintainer pool for a numerical-PDE codebase. No measurements were made. |

## Status as of 2026-10-12

| Package | Version | Notes | Verification |
|---|---|---|---|
| KernelAbstractions.jl | v0.10.0 (release page dated 12 Oct; year not shown on page) | Breaking. CPU backend is now OpenCL running on PoCL. `CPU(; static=true)`, `isgpu`, `@ka_code_llvm` and `@ka_code_typed` removed. `GPU` deprecated. Backends implement KernelInterface 0.4. Release notes also mention `PrivateArray` and `@Const` being constified at launch. | Confirmed |
| Metal.jl | v1.11.1 (dated 18 Sep; year not shown, page footer 2026) | 1.11.0 added `cbrt(Float32)` and 64-bit atomic min/max. 1.10.1 added ordered atomics, KA atomics and fences (Metal 3.2+). 1.11.1 fixes a macOS 27 crash. Blog post 2026-07-01 (1.10) added macOS 14 minimum, native matmul, MPS linear solvers, FFT and BFloat16. | Confirmed |
| PythonCall.jl / juliacall | v0.9.36 (dated 18 Sep) | One Julia thread by default since 0.9.32 (14 May). Multithreading crash fixes in 0.9.35 (08 Jun). Mixed Python/Julia threaded use is still maturing. | Confirmed |

The 2026 JuliaGPU ecosystem post (2026-08-03) lists CUDA, ROCm, oneAPI, Metal, OpenCL and CPU as KA targets and calls Metal.jl "still under active development with some known limitations". Its wording is generic and does not mention the 0.10 CPU change.

## Backends and fp64

| Backend | Status | Float64 |
|---|---|---|
| CPU via KA | Works, but 0.10.0 routes it through OpenCL/PoCL. New and unmeasured for a 30-40 us/RHS launch-bound budget. | Full |
| CPU via native Julia (`Threads.@threads`, Polyester.jl, `@simd`) | Safer route for fp64 CPU work. | Full |
| CUDA.jl | Most mature backend. A JuliaCon 2026 abstract reportedly says KA lacks warp ops on extended types, vectorized memory access and inter-workgroup ordering (unverified, abstract not fetched). | Native |
| Metal.jl (Apple GPU) | KA kernels run in Float32, Float16, BFloat16. `MtlArray(A)` preserves the input eltype while `mtl` auto-converts Float64 to Float32 (confirmed from the Metal.jl array docs). | No. Apple hardware and the Metal API are widely understood to lack double, but the cited docs page does not state this and the Apple forum thread was not fetched, so this rests on prior knowledge. The exact compile error text "unsupported use of double value" is unverified. |
| AMDGPU.jl, oneAPI.jl, OpenCL.jl | Not relevant to the Apple target. | n/a |

Metal.jl releases through 1.10 and 1.11 kept fixing stray Float64 uses in intrinsics (`cbrt`, `Base._cpow`), which shows the Float32-only constraint is enforced one function at a time.

A df64 number type could be written by hand as an isbits struct with operator overloads, and multiple dispatch makes that easy. It would carry float32 exponent range and flushed denormals. No package supplying it was found, but this is an absence claim from a limited search and is unverified.

## Mapping of the hard runtime shapes

| Idiom | Julia approach | Support |
|---|---|---|
| Packed field vector `[D\|Rx\|Ry\|Rz]` | One flat vector with static offsets, viewed as four `@view` slices. Views are isbits and pass into kernels without allocation. | Native |
| Masked axis stencil (constant coefficients, 3D, interior mask) | `@kernel` over `@index(Global, Cartesian)` with 3 to 9 taps, `@Const` input and a mask test. On CPU a plain `@inbounds @simd` loop. Tap count as `Val{N}` so the loop unrolls. | Native |
| ELL gather-dot-scatter for closure rows (0.5 to 2.5 percent of points) | Second kernel over closure rows with fixed width W. Distinct output per row means no atomics. Accumulating rows need `Atomix.@atomic`; Metal.jl 1.10.1 added KA atomics (confirmed), but Float32 atomic add on Metal being the weak spot is unsourced. Prefer deterministic ELL with no scatter-add. | Native |
| Face-plane assignment | 2D-range kernel or strided view broadcast (`u[1,:,:] .= ...`), works on Array and MtlArray. | Native |
| Masked reductions (error norms, max dt) | `mapreduce` / `sum` with a mask; Metal.jl implements reductions; AcceleratedKernels.jl offers a KA-based portable reduce. Read back every K steps. | Native |
| Whole-step compilation (RK4 launch fusion, graph replay) | Julia JITs the whole host step function, so on CPU an RK step is ordinary compiled code with no tracer. No automatic fusion across KA launches and no CUDA-graph equivalent in Metal.jl. Batching work into one Metal command buffer may be possible but was not verified. On Metal, a 60 to 120 launch step stays launch-bound. | Workaround |
| RK slot arithmetic and elementwise (I37/I38/I46) | Fused broadcasts such as `u1 .= u0 .+ c .* k`. | Native |
| CSR boundary coupling (I10/I20) | SparseArrays on CPU; hand-written ELL on Metal. Do not rely on GPU sparse libraries. | Workaround |
| MMS closed forms and source callbacks (I34/I35/I36) | Julia functions compile into kernels as closures over isbits captures. Float32 transcendentals work on Metal. Host-side fp64 evaluation remains for convergence studies. | Native |
| Geometry, ray casting, merge-join, config (I02-I09, I14) | Plain Julia host code. Means porting the C++ geometry; the Lua config would need replacing. | Native |
| SymPy stencil derivation (I48/I49), sparse eigensolver (I50) | Keep the Python SymPy pipeline and call it through juliacall, or export coefficient tables (JSON/HDF5). Symbolics.jl, Arpack.jl and KrylovKit exist, but re-porting is not justified. | Workaround |

## Code sketch

This sketch is illustrative and was not compiled or run.

```julia
using KernelAbstractions
import KernelAbstractions as KA

# axis stencil with compile-time tap count N, interior mask, constant coeffs
@kernel function axis_stencil!(du, @Const(u), @Const(mask), w::NTuple{N,T}, ::Val{S}) where {N,T,S}
    I = @index(Global, Cartesian)
    @inbounds if mask[I]
        acc = zero(T)
        h = N >> 1
        @unroll for k in 1:N                     # unrolled at compile time
            acc += w[k] * u[I + (k - 1 - h) * S]  # S = CartesianIndex unit step along axis
        end
        du[I] = acc
    end
end

# ELL gather-dot for closure rows; distinct output row per thread => no atomics
@kernel function ell_closure!(du, @Const(u), @Const(rowidx), @Const(col), @Const(val))
    r = @index(Global)
    @inbounds begin
        acc = zero(eltype(du))
        for k in 1:size(col, 2)
            acc += val[r, k] * u[col[r, k]]
        end
        du[rowidx[r]] = acc
    end
end

# launch (backend = CPU() / MetalBackend() / CUDABackend())
be = KA.get_backend(du)
axis_stencil!(be, (16, 8, 4))(du, u, mask, w, Val(S); ndrange = size(u))
ell_closure!(be, 64)(du, u, rowidx, col, val; ndrange = length(rowidx))
KA.synchronize(be)
```

## Performance assessment

No measurement was made. Everything below is reputation or inference.

| Case | Assessment |
|---|---|
| Small 2D (51x51, about 30-40 us/RHS C++ bar, launch-bound) | Credible on CPU only with native `@simd` loops and Polyester or Threads, not through KA launch machinery. Whole steps are compiled ahead of time, so there is no interpreter overhead and 0.3 to 1 ms per RK4 step is plausible. Thread spawn and task overhead of roughly 1 to 10 us can be comparable to the C++ budget. The KA 0.10.0 PoCL/OpenCL CPU path has unmeasured per-launch cost; that it exceeds C++ Kokkos Serial or OpenMP is inference, not evidence. On Metal, Float32 plus tens of us per launch means 60 to 120 launches per step cannot beat C++ at this size. |
| Large 3D (64^3, about 1.7 ms/RHS, bandwidth-bound) | Well-written Julia stencil loops are reputed to land within about 0 to 20 percent of C/Fortran (unverified, no benchmark checked). On Metal a Float32 kernel is bandwidth-bound and could beat the CPU on Pro/Max chips, but Float32 changes the numerics of a high-order cut-cell scheme. The 2025 arXiv KA portability paper (2507.16710) reportedly shows portable KA kernels at a fraction of vendor-tuned peak on some workloads; specific numbers were not verified. |

## What the DSL designs should copy or avoid

Copy:

1. The `@kernel` plus `ndrange` plus `get_backend` single-source model, with the backend chosen by array type.
2. Tap counts and axis steps as compile-time constants (`Val{N}`) so stencils unroll.
3. `@Const` read-only arguments and isbits views for packed fields.
4. A precision-generic element type `T` so one kernel runs Float64 on CPU and Float32 or a df64 struct on Metal.
5. Broadcast fusion for RK slot arithmetic.

Avoid:

1. Making the CPU path depend on a portability layer's launch machinery. KA 0.10.0 moving CPU to OpenCL/PoCL is an example of backend churn disturbing a CPU baseline.
2. Silent Float64 to Float32 conversion on upload (the `mtl` convenience path). A DSL should reject it or require an explicit precision policy.
3. Relying on cross-launch fusion or graph replay that the framework does not provide. The DSL must supply it itself.
4. Abstraction gaps (no warp ops, no vectorized loads, per the unverified JuliaCon abstract) that force backend-specific escapes.

Julia's one real advantage, a JIT that compiles an entire RK step to native code, is also available from Numba or a C++ code generator without adopting Julia.

## Deciding fact

Metal.jl, like every Apple GPU stack, offers no Float64 kernels, so Julia gives no Apple GPU advantage for an fp64 solver. Its only unique value is a CPU JIT, which would require rewriting the core in a second language and abandoning the Python goal, while KA's CPU backend has just been rebuilt on OpenCL/PoCL.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| KA v0.10.0 (12 Oct): CPU backend is OpenCL on PoCL; `CPU(static=true)`, `isgpu`, `@ka_code_llvm`, `@ka_code_typed` removed; `GPU` deprecated; KernelInterface 0.4 | Confirmed | Release tag page matches every detail. Year not shown. PoCL launch-cost concern is inference only. |
| Metal.jl v1.11.1 (18 Sep); 1.11.0 added `cbrt(Float32)` and 64-bit atomic min/max; 1.10.1 added KA atomics and fences | Confirmed | Releases page: #953, #866, #889, #875; `_cpow` issue #441 closed in 1.10.0. Year not shown. |
| Metal.jl 1.10 blog (2026-07-01): macOS 14, matmul, linear solvers, FFT, BFloat16 | Confirmed | All items present; post is silent on Float64. |
| `MtlArray(A)` keeps Float64 while `mtl` converts to Float32 | Confirmed | Metal.jl array docs. Docs do not state Float64 is unsupported nor quote the compile error. |
| Apple GPUs and the Metal API have no Float64; error text "unsupported use of double value" | Uncertain | Cited Apple forum thread not fetched; error string absent from the cited docs. Consistent with well-known behavior. |
| GPU ecosystem post lists CUDA, ROCm, oneAPI, Metal, OpenCL, CPU as KA targets; Metal "still under active development with some known limitations" | Confirmed | Quote matches the 2026-08-03 post. |
| PythonCall v0.9.36 (18 Sep); one thread default since 0.9.32; crash fixes in 0.9.35 | Confirmed | Releases page. Year not shown. |
| JuliaCon 2026 abstract: KA lacks warp ops, vectorized access, inter-workgroup ordering | Uncertain | Not fetched. |
| No Julia df64 package for Metal | Uncertain | Absence claim, no search done. |
| Julia stencil CPU performance within 0 to 20 percent of C; arXiv 2507.16710 figures | Uncertain | Self-declared reputation; not checked. |
| Metal command-buffer batching; Float32 atomic add weakness on Metal | Unverified | Flagged by verifier as unsourced or unchecked; not scored. |
| Release years for Metal.jl, KA, PythonCall | Unverified | Inferred from 2026 footer and today's date; not scored. |

Counts over the ten scored claims: 6 confirmed, 0 refuted, 4 uncertain. Score change: none; verifier-adjusted overall stays 4.5.

## Sources

- https://metal.juliagpu.org/dev/api/array/
- https://developer.apple.com/forums/thread/709663 (not fetched)
- https://juliagpu.org/post/2026-07-01-metal-1.10/
- https://github.com/JuliaGPU/Metal.jl/releases
- https://github.com/JuliaGPU/KernelAbstractions.jl/releases
- https://github.com/JuliaGPU/KernelAbstractions.jl/releases/tag/v0.10.0
- https://juliagpu.org/post/2026-08-03-gpu_ecosystem/
- https://pretalx.com/juliacon-2026/talk/EFQ8YD/ (not fetched)
- https://github.com/JuliaPy/PythonCall.jl/releases
- https://arxiv.org/html/2507.16710v1 (figures not verified)
- docs/python-dsl-survey/00-apple-gpu-facts.md (survey fp64 table)
