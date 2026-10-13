# Portable GPU routes to Apple GPUs: wgpu-py/WebGPU, Vulkan via MoltenVK (Kompute), tinygrad METAL

> **Verdict:** tier2 (overall 4/10 after verification). Experimental fp32 or double-float path only.
> **Fit for SHOCCS:** low as a core substrate, moderate as a design reference.
> **CPU fp64 parity:** poor (2/10). No native CPU compute route worth using in wgpu or Kompute; tinygrad has a CPU backend with fp64 but it is not a Kokkos-parity route.
> **Apple GPU:** reachable by all three (Metal via wgpu-native, MoltenVK, direct METAL), but fp32/fp16 only; no fp64.
> **Risk:** Python per-dispatch overhead at launch-bound sizes, fp32 breaks the 1e-10 test tolerances, kp is dormant, tinygrad targets ML workloads.

## Status as of 2026-10-12

| Package | Version | Date | State |
|---|---|---|---|
| wgpu-py | 0.32.0 (Python >=3.11) | 2026-07-19 | Active. Prior releases 0.31.1 (2026-06-23), 0.31.0 (2026-03-02). Wheels bundling wgpu-native are expected; 0.32.0 wheel names were not checked (inferred from earlier releases). |
| kp (Kompute Python) | 0.9.0 | 2024-01-20 | Stale, about 21 months without a release. |
| tinygrad | 0.14.0 (Python >=3.11) | 2026-08-24 | Very active; METAL is a first-class backend, but focus is ML/LLM. |
| MoltenVK | docs state Vulkan 1.4 layered over Metal | n/a | Active. |

## Backends and fp64

| Route | CPU | CUDA / NVIDIA | Apple GPU | fp64 |
|---|---|---|---|---|
| wgpu-py | None for compute worth using (software Vulkan only; unverified) | Vulkan/DX12 on NVIDIA, no CUDA | Metal via wgpu-native, WGSL translated by naga | WGSL has no f64. SHADER_F64 is native-only and Vulkan-only (confirmed), so unavailable on Metal. |
| Kompute (kp) | Software Vulkan only (unverified) | Vulkan on NVIDIA | Vulkan on MoltenVK, SPIR-V to MSL | Allowed where the device reports shaderFloat64. On Apple, Metal has no double type; MoltenVK not reporting it is inferred, not documented (unverified). |
| tinygrad | CPU backend (clang/LLVM JIT); fp64 works there (unverified in source) | CUDA and NV backends | Direct METAL backend, ctypes into Metal, no pyobjc (confirmed) | f64 is a dtype; Metal rejecting it is very likely but not located in source (unverified). |

Apple GPUs and Metal lack fp64 (Apple developer forum). A hand-written double-float (hi/lo fp32 pair) kernel is feasible in all three, but none ships one. Community estimates of emulated fp64 cost (over 4x add, over 27x multiply) are unverified.

## Mapping of the hard runtime shapes

| Idiom | wgpu-py / Kompute | tinygrad | Support |
|---|---|---|---|
| Packed field vector [D\|Rx\|Ry\|Rz] | One flat storage buffer or kp.Tensor, offsets as uniforms or push constants | One Tensor with slice/shrink views | native |
| Masked axis stencil (constant-coefficient interior with mask) | One thread per point, coefficients in a const array, mask via bounds test | Sum of shifted slices times coefficients with a where() mask, fused lazily | native |
| ELL gather-dot-scatter closure rows | Second small kernel with a fixed-width loop over an index table; deterministic if rows are disjoint | Advanced indexing lowers to comparison-reduce and can be very slow; scatter_reduce is awkward | workaround |
| Face-plane assignment | Small kernel with a plane descriptor | Strided slice setitem, may trigger copies | native |
| Masked reductions (min/max/maxloc) | Two-stage shared-memory reduction; no portable subgroup ops or float atomics in WGSL | Built-in max/min/where, argmax for maxloc | workaround |
| Whole-step compilation (RK4 as one graph) | wgpu: record all dispatches into one encoder, one submit, no cross-stage fusion. Kompute: kp.Sequence in one command buffer | TinyJit capture and replay; pointwise chains fuse, stencil chains stay separate kernels (ICB/batch execution on Metal unverified) | workaround |
| RK slot axpby | Trivial extra kernel or folded into RHS kernel | Fuses automatically | native |
| COO to CSR build, ray casting, line fragmentation | numpy/scipy at setup, upload static tables | same | not applicable |
| Eigen / stability analysis | CPU numpy/scipy, fp64 required | same | not applicable |

## Code sketch (illustrative, unverified)

The sketch was not checked against current wgpu-py docs. Known suspect points: `(k-2u)` underflows in u32 and relies on wraparound (use i32 offsets), runtime indexing `s[p.axis]` on a vec3 should be rechecked, and the uniform struct layout (vec3<u32>, u32, f32) was not checked for alignment. Treat it as a shape, not working code.

```python
import numpy as np, wgpu
dev = wgpu.utils.get_default_device()   # Metal via wgpu-native on Apple silicon
WGSL = """
@group(0) @binding(0) var<storage, read>       u  : array<f32>;
@group(0) @binding(1) var<storage, read_write> du : array<f32>;
@group(0) @binding(2) var<storage, read>       msk: array<u32>;   // 1 = interior
struct P { n: vec3<u32>, axis: u32, h: f32 }
@group(0) @binding(3) var<uniform> p: P;
const C = array<f32,5>(1.0/12.0,-8.0/12.0,0.0,8.0/12.0,-1.0/12.0); // E4 d1
@compute @workgroup_size(8,8,4)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (any(g >= p.n)) { return; }
  let s = vec3<u32>(1u, p.n.x, p.n.x*p.n.y);   // strides
  let i = g.x*s.x + g.y*s.y + g.z*s.z;
  if (msk[i] == 0u) { return; }
  var acc = 0.0;
  for (var k=0u; k<5u; k++) { acc += C[k]*u[i + (k-2u)*s[p.axis]]; }
  du[i] += acc / p.h;
}"""
shader = dev.create_shader_module(code=WGSL)
# bind groups/pipeline built once; per RK stage: encoder.begin_compute_pass(),
# set_pipeline, dispatch_workgroups(...) x3 axes + ELL kernel, then ONE queue.submit
# for the whole RK4 step; upload/readback via dev.queue.write_buffer / read_buffer (copies).
```

## Performance assessment

No benchmark was run; everything below is an architectural estimate.

- **Small 2D (51x51, 30-40 us per RHS, 0.3-1 ms per RK4 step on C++/Kokkos, launch-bound):** none of the three meets the bar. Python-to-wgpu-native dispatch costs tens of microseconds each; an RK4 step has dozens of dispatches, so roughly 1 to several ms per step at best, and per-step readback adds a sync. Kompute adds Sequence overhead and the MoltenVK translation layer. tinygrad TinyJit replay has the lowest per-kernel Python cost, so about 1 ms per RK4 step is plausible but not at C++ parity.
- **Large 3D (64^3, about 1.7 ms per RHS on CPU, bandwidth-bound):** Apple GPU bandwidth is high, so a hand-written fp32 stencil kernel could plausibly beat the host figure by a good factor. But fp32 changes the numerics for E4-E8 schemes: stability-eigenvalue and convergence-order tests at 1e-10 tolerances would not hold.
- **Double-float on fp32:** about 2x data and 10x flops; since the stencil is bandwidth-bound, likely 2-4x slower than fp32 (estimate).

## What the DSL designs should copy or avoid

| Copy | Avoid |
|---|---|
| Explicit command-encoder recording: the whole RK step as one pre-recorded submission with static bind groups | Python-per-dispatch overhead on the hot path |
| TinyJit capture-and-replay, lazy fusion of pointwise chains, free slice views | Generic indexing for ELL closure rows (compiles to comparison-reduce) |
| Feature-flag approach to precision: declare required features, fail loudly when SHADER_F64 is absent | Relying on WGSL portable-core features that lack f64, float atomics and subgroup ops |
| Emit stencil and ELL kernel source (MSL/WGSL/SPIR-V) directly, fuse the RK step into one command buffer, make double-float an explicit lowering option, keep a native CPU fp64 reference | Depending on a dormant Python package (kp 0.9.0, Jan 2024) |

## Deciding fact

No Apple-GPU path in any of these three can hold fp64: WGSL has no f64, SHADER_F64 is Vulkan-only in wgpu, Metal has no double type so MoltenVK cannot be expected to supply Float64 (inferred), and tinygrad METAL is expected to be fp32/fp16 only (unverified in source). Combined with Python per-dispatch overhead at the launch-bound 51x51 case, none can match the C++/Kokkos bar, so they are at most fp32 or double-float experimental paths for large 3D.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| wgpu-py 0.32.0 is latest, released 2026-07-19, wheels bundle wgpu-native | confirmed (bundling for 0.32.0 inferred) | GitHub releases API and PyPI; earlier wheels include macOS arm64 |
| wgpu SHADER_F64 is Vulkan-only and native-only; WGSL has no f64 | confirmed (WGSL point is general knowledge, not re-fetched) | docs.rs wgpu Features |
| Metal/Apple GPUs have no fp64, so MoltenVK cannot expose shaderFloat64 | uncertain | Apple forum confirms Metal limit; MoltenVK guide is silent on shaderFloat64, so the MoltenVK conclusion is inference |
| MoltenVK is a Vulkan 1.4 layer on Metal | confirmed | MoltenVK user guide wording. The example feature shaderBufferInt64Atomics was not found in the fetched page and is unverified |
| kp 0.9.0 released 2024-01-20, stale | confirmed | PyPI JSON |
| tinygrad 0.14.0 on 2026-08-24, Python >=3.11, METAL backend via ctypes, no pyobjc | confirmed | PyPI JSON and ops_metal.py. Weak-dtype/ContextVar details and the "over 1000 commits" figure were not checked |
| tinygrad METAL does not support float64 | uncertain | No fp64 handling found in ops_metal.py or device.py; the device.py citation in the assessment looks wrong |
| Emulated fp64 over 4x add and over 27x multiply; MLX maintainers declined emulation | uncertain | Secondhand, issue not fetched |
| Code sketch API calls and WGSL correctness | uncertain | Not checked; u32 underflow and uniform layout concerns noted above |

Counts: 5 confirmed, 0 refuted, 4 uncertain. Unchecked side items (wgpu-py headless compute, CPU compute "none", TinyJit ICB/batch on Metal) are marked unverified in the text above.

## Sources

- https://pypi.org/pypi/wgpu/json
- https://api.github.com/repos/pygfx/wgpu-py/releases?per_page=3
- https://github.com/pygfx/wgpu-py/releases
- https://packages.msys2.org/base/mingw-w64-python-wgpu
- https://docs.rs/wgpu/latest/wgpu/struct.Features.html
- https://pypi.org/pypi/tinygrad/json
- https://raw.githubusercontent.com/tinygrad/tinygrad/master/tinygrad/runtime/ops_metal.py
- https://github.com/tinygrad/tinygrad/releases
- https://docs.tinygrad.org/env_vars
- https://pypi.org/pypi/kp/json
- https://github.com/KhronosGroup/MoltenVK/blob/main/Docs/MoltenVK_Runtime_UserGuide.md
- https://github.com/KhronosGroup/MoltenVK/issues/1567
- https://developer.apple.com/forums/thread/709663
- https://upd.dev/ml-explore/mlx/issues/1905
