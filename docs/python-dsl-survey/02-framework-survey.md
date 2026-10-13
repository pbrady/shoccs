# Part II: Survey of Python Frameworks for a SHOCCS Port

Status date: 2026-10-12. This part compares 18 Python execution options against the 51-idiom catalog of the SHOCCS C++ solver (ids I01 to I51). It also shows how the four hardest idioms look in each option. Each framework was surveyed in its own file under `docs/python-dsl-survey/frameworks/`, and an independent pass then checked the load-bearing claims. This part summarises those files. Wherever it states a fact about a framework, the source is the linked detailed file and its verification log, plus the primary URL given in that file. Where a claim was not checked, this part says so. Section 9 adds nine further comparators and routes that a completeness review asked for (PyFR, loopy, MLIR-based substrates, portable GPU APIs, Julia, Pythran/Cython/Codon, Apple-silicon CPU fp64 headroom, a Kokkos Apple backend, and the cross-cutting criteria of `04-crosscutting-criteria.md`). Fresh C++ measurements made on 2026-10-12 are in `05-measurements.md` and summarised in section 2.1.

## 1. How to read this survey

### 1.1 Scope and evidence

- **Options surveyed (18):** JAX, PyTorch, Apple MLX, Taichi, Devito, pystencils, Numba, NVIDIA Warp, DaCe, GT4Py, Halide, PyKokkos, Mojo/MAX, Triton, a NumPy/SciPy assembled-sparse baseline, the Python Array API standard, direct Metal from Python, and CuPy/cuPyNumeric.
- **Evidence levels.** Each detailed file has a verification log. Claims are marked *confirmed* (checked against a current primary source), *refuted* or *corrected*, or *uncertain/unverified*. Section 8 summarises the outcomes.
- **Performance numbers.** Almost every framework performance figure in the survey is either an estimate or a local, synthetic, single-host measurement. None is a benchmark of a Python port of the real cut-cell solver against the C++ code on the same machine. The C++ reference points now come from `05-measurements.md` (section 2.1), measured on 2026-10-12 inside the devcontainer (Docker Desktop, 8 vCPU, Kokkos OpenMP, Release) on the owner's Mac. There is still no native macOS build.
  - RHS graph submit on heat.lua 51x51 with its sphere: 22.7 us at 1 thread. At 71x71: 34.5 us; scalar_wave 71x71: 30.7 us (both 1 thread).
  - Launch floor: 0.55 us per tiny Kokkos kernel at 1 thread.
  - Shipped RK4 step (io off): 0.90 ms for heat 51x51 (either thread count) and 0.43 ms for scalar_wave 71x71 (1 thread). The estimated fixed-up C++ step is about 0.15 ms at 51x51 (an estimate; section 2.1).
  - These supersede the March 2026 baseline (`benchmarks/baselines/baseline-20260320-024030.json`: about 32 us per 16^3 heat RHS and about 0.7 to 0.8 us per tiny kernel) and the earlier unmeasured estimate of 0.3 to 1 ms per RK4 step. Several detailed files still compare against those older figures; this part adds the new numbers next to such comparisons.
  - Treat every comparison of a framework figure against these numbers as indicative: the framework figures come from other runs, often on synthetic tables.
- **CuPy file.** `frameworks/cupy_legate.md` has been regenerated (verification: 7 confirmed, 0 refuted, 2 uncertain). Sections 4.2 and 5.18 draw on it.
- **Comparators.** The nine files summarised in section 9 are not counted among the 18 options above. Their scores appear in a separate matrix in section 3.

### 1.2 Criteria

| Criterion | Question asked |
|---|---|
| CPU parity | Can a fp64 CPU run match the C++/Kokkos code, both at the shipped small 2D sizes (2.6k to 5k points, launch bound) and at large 3D (64^3 to 256^3, bandwidth bound)? This is the primary bar (see section 2). |
| Apple GPU | Is there a maintained path to Apple GPUs (Metal, MPS or MLX)? How mature is it? |
| fp64 on GPU | Does the framework run fp64 on any GPU? On Apple GPUs the answer is always no (section 2). |
| Idiom coverage | How many catalog idioms map natively, need a workaround, or are unsupported? |
| Whole-RHS fusion | Can one RHS, or a whole RK4 step, become one compiled unit (one call, few memory passes)? Or must the DSL hand-fuse? |
| Dispatch overhead | What is the per-call or per-launch cost from Python, compared with the Kokkos launch floor of 0.55 us per tiny kernel at 1 thread (`05-measurements.md`; the March 2026 baseline gave 0.7 to 0.8 us)? |
| DSL-substrate fitness | How well would the framework serve as a lowering target for a SHOCCS DSL with its own IR? |
| Risk | Project health, API churn, vendor control, experimental status. |

### 1.3 Scoring

- Scores are 0 to 10 and come from each detailed file after the verification pass.
- For risk, **a higher score means lower risk** (safer long term).
- The overall score is the verifier-adjusted holistic score, **not** a weighted mean of the sub-scores. Several overall scores were lowered by the verifier when the evidence behind them turned out to be weak (section 8).
- Support labels in the idiom tables are native (N), workaround (W) and unsupported (U):
  - **native:** expressible with the framework's own primitives.
  - **workaround:** needs restructuring or host-side code.
  - **unsupported:** cannot be expressed. The idiom must be replaced.
- **Labelling caveat.** The detailed files label setup idioms (I02 to I12, I14) inconsistently. Some call host NumPy work "native" (JAX, Numba, NumPy/SciPy), others call the same work "workaround" (Devito, MLX). The underlying fact is the same everywhere: geometry and table construction run once in host NumPy, SymPy or Numba. Raw label counts are therefore a weak signal. Section 4 adds a count restricted to the 30 hot-path idioms (the 29 that Part I marks hot plus I47, which Part I marks hot for the future Euler system).

### 1.4 The four hard idioms

The catalog has 51 idioms, but four of them carry the cut-cell "unstructuredness" that defeated earlier attempts. Each framework section below says how it expresses them.

| Id | Hard idiom | Catalog ids | Where it lives in the C++ code |
|---|---|---|---|
| H1 | Ragged per-line block matvec. Each line has a dense left closure, a circulant interior and a dense right closure. Lines have different lengths and strides, and one Kokkos team serves one line. | I16 to I19 | `src/matrices/block.hpp:118-178` (`matvec_functor`, `TeamPolicy` plus `TeamThreadRange` plus `ThreadVectorRange`); per-line metadata is the 13-int struct in `src/matrices/inner_block_meta.hpp:9-23`; graph node at `src/matrices/block.hpp:197-209` |
| H2 | Cut-cell CSR coupling between the D grid and the R (cut-point) buffers: B, N, Bf and Br. | I20 to I24 | Full-height CSR row loop as a graph node at `src/matrices/csr.hpp:47-60` and eager form at `src/matrices/csr.cpp:37-51`; both sweep every row pointer whether or not the row has entries |
| H3 | Boundary application: masked or gathered source add, Dirichlet fill, and per-stage `update_boundary` with an exact-solution evaluation. | I30, I31, I39 (and I34, I36, I40) | Dirichlet gather list built per call at `src/systems/heat.cpp:195-205`; source scatter node at `src/systems/heat.cpp:237-240`; host source fill before every graph submit at `src/systems/system.cpp:62-63`; serial path for Lua MMS at `src/systems/detail/scalar_system_utils.hpp:14-16` |
| H4 | Whole-step execution: Kokkos Graph capture with pointer baking and fences, plus RK4 slot algebra. | I32, I37, I38 | Graph build at `src/systems/heat.cpp:213-285`, `instantiate` at `src/systems/heat.cpp:287`, `submit` plus fence at `src/systems/heat.cpp:292-293` |

## 2. Constraints that apply to every framework: Apple GPU, fp64 and Kokkos-Metal

These facts come from `00-apple-gpu-facts.md`. Its verifier confirmed all the load-bearing fp64 and Kokkos claims against primary sources and rated it 7.5/10, docking points for unverified secondary claims. The facts constrain every row of the comparison, so they come first.

| Fact | Status | Source |
|---|---|---|
| No Apple GPU (M1 to M5) has hardware fp64. MSL has no usable `double`. | Confirmed. AdaptiveCpp's Metal page states "Apple Silicon GPUs do not have hardware support for double-precision floating point". The MSL spec was not re-fetched. | https://adaptivecpp.github.io/AdaptiveCpp/install-metal/ |
| PyTorch MPS (2.14.1, 30 Sep 2026) has no float64. | Confirmed (mostly indirect evidence). | https://github.com/pytorch/pytorch/releases ; https://developer.apple.com/forums/thread/797778 ; https://discuss.pytorch.org/t/float64-tensors-on-mps/224325 |
| MLX float64 is CPU only. The request for GPU emulation (#1905) was closed on 3 Dec 2025 as not to be implemented. | Confirmed | https://github.com/ml-explore/mlx/issues/1905 |
| jax-metal stopped at 0.1.1 (8 Oct 2024) and lists float64 as unsupported. The JAX tracking issue was closed on 20 Jan 2026 because the team stopped work. | Confirmed | https://pypi.org/project/jax-metal/ ; https://github.com/jax-ml/jax/issues/26968 |
| Taichi's Metal backend has no f64, i64 or f16. | Confirmed | https://docs.taichi-lang.org/docs/type |
| Halide's Metal codegen raises a hard error on 64-bit float literals. | Confirmed in the Halide verification | https://github.com/halide/Halide/blob/main/src/CodeGen_Metal_Dev.cpp |
| Kokkos 5.x has no Metal or Apple-GPU backend. Its SYCL backend is documented only with IntelLLVM (oneAPI) 2024.2.1. | Confirmed. Third-party Metal efforts were not searched in the Kokkos tracker. | https://kokkos.org/kokkos-core-wiki/get-started/requirements.html |
| AdaptiveCpp has an experimental SYCL-to-Metal backend, tested on macOS 26 only, with no `double` (soft-double planned). No Kokkos-on-AdaptiveCpp port exists. | Backend status confirmed. The cited PR numbers are unverified. | https://adaptivecpp.github.io/AdaptiveCpp/install-metal/ |
| fp64 emulation: hand-written double-float (df64) gives about 48-bit mantissa with fp32 exponent range. Correctly rounded software binary64 is reported at about 13x fp32 cost. | df64 properties are standard. The 13x figure (warpx-metal) is unverified. | https://github.com/Lulzx/warpx-metal (unverified) |
| A GPU op plus synchronisation on the owner's M4 costs 167 us (MLX, compiled), 175 us (torch MPS eager) and 181 us (torch MPS compiled). An earlier unverified forum figure put an empty-kernel Metal round trip at about 120 us on an M2 Max. | Measured on 2026-10-12 (`05-measurements.md` Section D, dispatch floors: median of 200 calls of a 1024-element op plus sync). The 120 us figure remains a single unverified forum report. | `05-measurements.md` Section D; https://developer.apple.com/forums/thread/767853 |
| A single CPU thread on the survey host sustained about 96 GB/s on a 16.8M-element fp64 axpy. | Unverified local measurement on an unidentified chip. The verifier flagged it for recheck. For comparison, the devcontainer run of 2026-10-12 measured a 1M-element axpby at 463 us on 1 thread (0.44 ns per element, 67 GiB/s) and 385 us on 8 threads. | Local measurement, see `00-apple-gpu-facts.md`; `05-measurements.md` |
| PyTorch sparse CSR on MPS (PR #196205) is open and leaves add, addmm, addmv and mm uncovered. | Confirmed | https://github.com/pytorch/pytorch/pull/196205 |

Consequences for the rest of this part:

1. **"MPS like PyTorch" cannot mean fp64.** Every Apple-GPU path in this survey is fp32. The only alternative is a df64 kernel layer that we would write ourselves in Metal; no framework provides a df64 dtype. Whether small-psi high-order cut-cell closures stay accurate and stable in fp32 is an open numerical question. It is independent of the framework chosen.
2. **Kokkos offers no route to Metal.** The C++ code can stay on CPU Kokkos. Any Apple-GPU target would be a separate backend of a DSL or code generator.
3. **The primary bar is a fused, whole-step compiled fp64 CPU path.** At the shipped 2D sizes the data fits in L2. The measured GPU sync floor on the owner's M4 is 167 to 181 us (`05-measurements.md` Section D; an earlier unverified forum figure said about 120 us), seven to eight times the measured 22.7 us C++ RHS at 51x51 (1 thread, section 2.1); a whole-step-compiled fp32 GPU step is 0.35 to 0.5 ms there against 0.085 ms for the best fused CPU fp64 step from Python. A GPU only competes at those sizes if many RK steps or many independent cases are encoded per command buffer. At 65^3 and above the compiled GPU paths are 3 to 6x faster than the best CPU fp64 path in fp32 (1.0 against 4.2 ms at 65^3; 43.6 against 272 ms at 257^3), on a base M4. The 2 to 4x window at 256^3 on Pro/Max chips in fp32 or df64 remains an inference from rated bandwidths.
4. **Dense index ops are the portable form of the cut-cell coupling.** Gather, scatter-add, compact COO and padded ELL tables built on the host work everywhere. Framework sparse types do not: they are absent, unmaintained, or unusable on MPS.

### 2.1 The measured C++ bar

`05-measurements.md` holds fresh C++ measurements made on 2026-10-12 inside the project devcontainer (Docker Desktop on the owner's Apple-silicon Mac, 8 vCPU, Kokkos 5.1.1 OpenMP, Release). There is still no native macOS build, so these are Linux-container numbers on Mac hardware. They supersede the March 2026 baseline wherever a matching case exists.

| Case | RHS graph submit, 1 thread | RHS graph submit, 8 threads | Shipped RK4 step, io off |
|---|---|---|---|
| heat.lua 51x51 with its sphere | 22.7 us | 87.8 us | 0.90 ms (either thread count) |
| heat 71x71 | 34.5 us | 92.0 us | not run |
| scalar_wave.lua 71x71 (two spheres) | 30.7 us | 86.1 us | 0.43 ms (1 thread) |
| heat 201x201 | 198 us | 152 us | 4.6 ms (8 threads) |
| heat 401x401 | 772 us | 422 us | 13.7 ms (8 threads) |
| heat 65^3 with sphere | 1867 us | 798 us | 37.9 ms (8 threads) |

Other headline facts from the same run:

- **Step decomposition (51x51, 1 thread).** fill_source takes 132 us per stage (59% of the step), update_boundary 34 us per stage (15%), the four RHS submits 10%, and stats 32 us (4%). The shipped step is dominated by host-side manufactured-solution evaluation, not by the discretization.
- **Launch floor and bandwidth.** 0.55 us per tiny kernel at 1 thread. A 1M-element axpby takes 463 us at 1 thread (0.44 ns per element, 67 GiB/s) and 385 us at 8 threads.
- **Threads.** Eight threads are slower than one below about 40k points, so a serial path for small cases is part of the bar.
- **Fixed-up C++.** The "fixed-up C++" is a frozen list of exactly four changes: (1) hoist the object-BC gather lists out of the per-call path; (2) compact B and N tables with no full-height row pointers; (3) manufactured-solution evaluation once per step and only where needed, with time factors hoisted (fill_source and update_boundary); (4) packed [D | Rx | Ry | Rz] buffers. Its step is estimated at about 0.15 ms at 51x51 and about 5.5 ms at 65^3. Both are estimates derived in `05-measurements.md`, not measurements.

How this part uses the bar. The discretization bar for a replacement's kernels is the RHS column: 23 us at 51x51 and 31 to 35 us at 71x71 (1 thread). A replacement's whole step should be compared with the estimated fixed-up step, not with today's shipped step. The framework figures in section 5 come from other runs, mostly on synthetic tables, so every comparison with this bar stays indicative.

**The measured host-side bar (`05-measurements.md` Section D).** On 2026-10-12 a synthetic SHOCCS-shaped RK4 step (Part I primitives P0 to P6, E2 interior, random closure coefficients, no host MMS, no update_boundary, one Linf readback) was timed natively on the owner's M4 (torch 2.11.0 with MPS, 4 intra-op threads; MLX 0.32.3; numpy 2.4.4) on seven backends, outside the sandbox that hides Metal. Against four C++ RHS submits (4 x 22.7 us = 91 us at 51x51, 1 thread; 4 x 798 us = 3.2 ms at 65^3, 8 threads; 4 x 1867 us = 7.5 ms at 65^3, 1 thread):

| Case | numpy CPU f64 | torch CPU f64 eager | torch CPU f64 compile | torch MPS f32 compile | MLX GPU f32 compile | MLX CPU f64 compile | four C++ RHS submits |
|---|---|---|---|---|---|---|---|
| 51x51 | 0.103 ms | 0.149 ms | 0.168 ms | 0.501 ms | 0.353 ms | 0.085 ms | 0.091 ms (1t) |
| 201x201 | 0.674 ms | 1.406 ms | 0.523 ms | 0.530 ms | 0.376 ms | 0.622 ms | 0.61 ms (8t) |
| 65^3 | 5.33 ms | 4.61 ms | 4.49 ms | 1.09 ms | 1.01 ms | 4.24 ms | 3.2 ms (8t), 7.5 ms (1t) |
| 257^3 | 448 ms | 340 ms | 272 ms | 43.6 ms | 57.2 ms | 300 ms | not measured |

Dispatch floors (trivial op plus sync): numpy 0.46 us, torch eager 0.75 us, torch.compile CPU 6.4 us, MLX CPU compiled 8.7 us, MLX GPU 167 us, torch MPS 175 to 181 us. fp32 error after 10 steps: 1e-7 to 2.4e-7 relative, on synthetic closures only. Reading: a fused fp64 CPU step from Python sits at the discretization bar at 51x51 (0.9x to 1.6x of four submits) and within 1.4x of eight-thread Kokkos at 65^3, so CPU parity is plausible but not shown with real tables or in a native C++ comparison; the Apple GPU in fp32 loses at the shipped sizes and wins 4 to 6x at 129^3 and 257^3. Caveats: synthetic coefficients, no MMS or update_boundary, a container C++ bar, possible CPU contention in run 1.

## 3. Master comparison matrix

Sorted by overall score.
- Numbers are 0 to 10 sub-scores from the detailed files.
- Risk: higher means safer.
- "est." means unmeasured estimate. "local" means a local synthetic measurement nobody else has reproduced.

| Framework | CPU parity | Apple GPU | fp64 on GPU | Idiom coverage | Whole-RHS fusion | Dispatch overhead | DSL-substrate fitness | Risk | Overall |
|---|---|---|---|---|---|---|---|---|---|
| JAX 0.11.2 | 7.5: plausible at both sizes (local, synthetic) | 2: jax-metal dead; jax-mps Alpha, fp32 | CUDA yes; Apple no | 8.5 | Automatic: `jit` over RK4 step or K steps (`fori_loop`), needs `optimization_barrier` | About 2 us per jitted call (local) | High: CPU/CUDA reference lowering, `jacfwd` operator export | 7 | 6.5 |
| PyTorch 2.14.1 | 7: large 3D likely; small 2D 1 to 2x of C++ (local) | 4: MPS eager plus Inductor-MPS, fp32, no graph replay | CUDA yes; MPS no | 8 | Automatic on CPU/CUDA via Inductor (`fullgraph` RK4 step); no time-loop compile | About 0.6 us eager op; about 70 us fixed per compiled small step (local) | High CPU/CUDA, medium MPS | 8 | 6.5 |
| Numba 0.67.0 | 9: parity expected (est.) | 0: none | CUDA via numba-cuda (no macOS wheels); Apple no | 8 | Manual: whole RHS or step in one `njit` function; generator writes fused loops | 0.2 to 1 us per call (est.) | High: CPU emitter that keeps ragged loops | 6.5 | 6.5 |
| Halide 21.0.0 | 8: parity or better via fusion (est.) | 4: own Metal runtime, fp32 only | CUDA yes; Metal no | 7 | Automatic, schedule controlled: one `Pipeline` per RK4 step | Tens of us per Callable (est.) | High CPU/CUDA; manual schedules | 7 | 6.3 |
| DaCe 2.0.0-alpha10 | 8: plausible (unmeasured) | 1: none | CUDA/HIP yes | 8 | Automatic in principle: one SDFG per step (unverified) | 5 to 50 us per call (unsourced est.) | High: SDFG is close to the needed IR | 5 | 6 |
| Devito 4.8.23 | 8.5: 7.2 us per fused RK4 step at 51x51 (local, serial) | 0.5: none | NVIDIA/AMD via OpenMP offload/OpenACC | 7 | Automatic: whole multi-step time loop in one C function | `apply()` 0.2 to 0.8 ms (local), so batch steps | Medium-high: SymPy native; DSL owns tables | 7 | 5.5 |
| Taichi 1.7.4 | 6.5: large 3D likely; small 2D at risk (est.) | 4.5: native MSL backend, fp32 | CUDA yes; Metal no | 6.5 | Manual: one big kernel; no cross-kernel fusion | Unmeasured | Medium: imperative kernels fit tables | 3 | 5 |
| pystencils 2.1 | 7.5: large 3D likely; small 2D at risk (est.) | 1: none found | CUDA/HIP yes | 6 | Partial: per AssignmentCollection; 5 to 10 kernels per RHS | Few us per kernel (est.) | Medium: SymPy leaf kernels, C++ export via sfg | 5 | 5 |
| Mojo 1.0 / MAX 26.6 | 8: architectural argument, unmeasured | 3: MAX graphs on M1 to M5 in nightlies; custom-kernel path undocumented | CUDA expected, unconfirmed; Apple no | 6 | Manual: generator emits whole-RHS Mojo function | Native call on CPU | Medium: codegen target, not a DSL host | 4 | 5 |
| NumPy/SciPy (assembled CSR) | 5: small 2D plausible; large 3D 3 to 8x slower (est.) | 0: none | n/a | 8 | None | 1 to 5 us per call (est.) | Reference: oracle and table producer | 9 | 5 |
| Direct Metal (MLX kernels, pyobjc, metal-cpp) | 3: GPU only, no CPU path | 8: best Apple-GPU access | None | 7 | Manual: encode step in one command buffer | pyobjc 10 to 50 us per dispatch (est.); metal-cpp a few us (est.) | Apple-GPU backend layer under a DSL | 5 | 5 |
| MLX 0.32.3 | 3: fp64 CPU likely single-core for built-ins (inferred) | 6.5: native Metal, unified memory, fp32 | None | 6.5 | Elementwise only (confirmed in `compile.cpp`) | 1 to 10 us per op (est.) | Low-medium: optional fp32 Metal backend | 5 | 4.5 |
| GT4Py 1.2.3 | 6: large 3D plausible; small 2D doubtful (est.) | 1: none | CUDA/ROCm yes | 5 | Within a `field_operator`; 3 to 6 kernels per RHS | Unquantified | Low-medium: optional CPU/CUDA backend | 5 | 4 |
| Triton 3.8.0 | 2: no mainline CPU backend | 4: out-of-tree only (PR #126 open, triton-msl alpha) | CUDA elementwise yes; Apple no | 6 | Within one kernel only | 10 to 30 us per launch (est.) | Low-medium: CUDA/ROCm emitter | 5 | 4 |
| Array API standard 2025.12 | 4: eager loses; parity needs an external compiler | 2: none from the standard | Backend dependent; Apple no | 5 | None in the standard | Backend dependent | Reference vocabulary and conformance test | 7 | 4 |
| Warp 1.18.0 | 3: CPU launches serial and synchronous (confirmed) | 0: macOS is CPU only | CUDA yes | 8 | Manual; no cross-kernel fusion | 10 to 50 us per launch (est.) | Low as substrate; good design reference | 6 | 3.5 |
| PyKokkos (no release) | 5: large 3D near parity, small 2D poor (est.) | 0.5: none | CUDA/HIP yes | 6 | None (PK_FUSION is naive lazy fusion) | Tens of us per launch (est.) | Low: prototyping only | 3 | 3 |
| CuPy 14.2.0 / cuPyNumeric | 2: no CPU backend; cuPyNumeric end of life | 0: none | CUDA yes | 6 | `cupy.fuse` elementwise; RawKernel manual | Eager launches (unquantified) | CUDA-only seam | 4 | 3 |

**Comparators (section 9).** The nine comparator files were scored on fewer criteria, so they are listed in a second matrix rather than added to the one above, whose scores are unchanged. The role column marks every row as a comparator. "n/s" means the file gives no numeric score for that criterion; the words after it summarise what the file says instead.

| Comparator | Role | CPU parity | Apple GPU | fp64 on GPU | Risk | Overall | Verdict |
|---|---|---|---|---|---|---|---|
| Apple-silicon CPU fp64 (NEON, SME, AMX, Accelerate) | comparator (hardware target and baseline, not a framework) | 9: 3x to 20x over Kokkos host (modelled, unmeasured) | 2: no fp64 GPU | Apple no | 8 | 7 | shortlist |
| Pythran / Cython / Codon | comparator (CPU emitters) | n/s: near parity expected (est.) | n/s: none | Codon NVIDIA only; Apple no | n/s: low for CPU | 6 | tier2 |
| Cross-cutting criteria (differentiability, mixed precision, MPI) | comparator (criteria, not a framework) | 6 | 3 | Apple no | 6 | 5 | reference only |
| loopy | comparator (kernel generator) | 6: plausible for one fused step function (est.) | 1: no Metal target found | Apple no (OpenCL deprecated) | 4 | 4.5 | tier2 |
| Julia (KernelAbstractions.jl + Metal.jl) | comparator (non-Python) | n/s: full Float64 on CPU, credible (unmeasured) | n/s: Float32, Float16, BFloat16 only | Apple no | n/s: KA 0.10.0 breaking release | 4.5 | reference only |
| PyFR backends | comparator (architecture reference) | n/s: plausible on OpenMP (est.) | n/s: Metal backend fp32 only | Metal no (ValueError on double) | n/s: medium to high | 4 | reference only |
| wgpu-py / MoltenVK (Kompute) / tinygrad METAL | comparator (portable Apple-GPU routes) | 2 | n/s: reachable, fp32/fp16 only | Apple no | n/s: per-dispatch overhead, dormant kp | 4 | tier2 |
| MLIR: xDSL stencil dialect, IREE | comparator (compiler substrate) | n/s: plausible via LLVM (unmeasured) | n/s: none usable | Apple no | n/s: high | 3.5 | reference only |
| Kokkos Apple backend (Metal or AdaptiveCpp) | comparator (rejected route) | 8: existing Kokkos CPU, unchanged | 1 | none | 1 | 1.5 | excluded |

What the matrix says:
- No option scores well on both CPU parity and Apple GPU. The best CPU candidates (Numba 9, Devito 8.5, DaCe/Halide/Mojo 8) score 0 to 4 on Apple GPU.
- The best Apple-GPU access (direct Metal 8, MLX 6.5) has no competitive fp64 CPU path.
- Every Apple-GPU path is fp32.
- The comparators change none of this. The highest comparator score (Apple-silicon CPU, 7) belongs to hardware, not a framework, and its speedups are modelled, not measured. No comparator offers fp64 on an Apple GPU.

## 4. Idiom-by-framework support matrix

### 4.1 Ten frameworks, all 51 idioms

Columns are the ten options most relevant to the shortlist. Labels are taken from each detailed file's idiom table.
- N native, W workaround, U unsupported.
- **Hot** means the catalog marks the idiom as on the time-stepping path.

| Id | Idiom (short) | Hot | JAX | PyTorch | Numba | Halide | DaCe | Devito | Taichi | pystencils | MLX | Direct Metal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I01 | Registry slot/buffer allocation | | N | N | N | N | N | N | N | N | N | N |
| I02 | Ray casting to R records | | N | N | N | W | W | W | W | U | W | W |
| I03 | Line fragmentation, dirichlet_line | | N | W | N | W | W | W | W | W | W | W |
| I04 | Index-set construction | | N | N | N | N | N | N | N | N | N | N |
| I05 | psi-independent stencil constants | | N | N | N | N | N | N | N | N | W | N |
| I06 | psi-dependent closure evaluation | | N | N | N | W | W | W | W | N | W | W |
| I07 | Cross-direction interpolation | | W | W | W | W | W | W | W | W | W | W |
| I08 | domain_discretization assembly | | N | W | N | W | W | W | W | W | W | W |
| I09 | cut_discretization Bf/Br rows | | W | W | N | W | W | W | W | W | W | W |
| I10 | COO to CSR build | | N | N | N | N | N | W | N | N | W | N |
| I11 | Block flattening to device arrays | | N | N | N | N | N | W | N | N | W | N |
| I12 | Config and system setup (Lua) | | N | N | N | N | N | W | N | N | W | N |
| I13 | Flat C-order D index space | yes | N | N | N | N | N | N | N | N | N | N |
| I14 | R ordering contract | | N | N | N | N | N | W | N | N | W | N |
| I15 | Span bridge, work buffers | yes | N | N | N | N | N | N | N | N | N | N |
| I16 | Ragged line block matvec (H1) | yes | W | W | N | W | W | W | W | W | W | W |
| I17 | Interior circulant axis stencil | yes | N | N | N | N | N | N | N | N | N | N |
| I18 | Wall closures | yes | N | N | N | N | N | N | N | N | N | N |
| I19 | Object closures (psi-dependent) | yes | N | W | N | W | W | W | W | W | W | W |
| I20 | Generic CSR row kernel (H2) | yes | W | W | N | W | N | W | W | W | W | W |
| I21 | B: R to D coupling | yes | N | N | N | W | N | W | N | W | W | W |
| I22 | N: Neumann wall data | yes | N | N | N | N | N | N | N | N | N | N |
| I23 | Bf: D to R rows | yes | N | N | N | W | N | W | N | W | W | W |
| I24 | Br: R to R self-coupling | yes | N | N | N | W | N | W | N | W | W | W |
| I25 | Zero-fill before accumulation | yes | N | N | N | N | N | N | N | N | N | N |
| I26 | Whole-scalar scale | yes | N | N | N | N | N | N | N | N | N | N |
| I27 | Accumulation policy and ordering | yes | N | N | N | N | N | N | W | N | N | W |
| I28 | Gradient fan-out | yes | N | N | N | N | N | N | N | N | N | N |
| I29 | Variable-coefficient dot product | yes | N | N | N | N | N | N | N | N | N | N |
| I30 | Masked add/assign at gather sets (H3) | yes | N | N | N | N | N | N | N | N | N | N |
| I31 | Dirichlet face/object fill (H3) | yes | N | N | N | N | N | N | N | N | N | N |
| I32 | Graph capture and submit (H4) | yes | N | W | W | N | N | N | W | W | W | W |
| I33 | Eager apply path | yes | N | N | N | N | N | N | N | N | N | N |
| I34 | Closed-form MMS evaluation | yes | N | N | N | N | N | N | N | N | N | N |
| I35 | Lua per-point MMS callback | yes | W | U | W | N | U | U | U | U | W | U |
| I36 | Per-stage host source fill | yes | N | N | N | N | N | N | N | N | N | W |
| I37 | RK4 / Euler slot combinations (H4) | yes | N | N | N | N | N | N | N | N | N | N |
| I38 | deep_copy_slot | yes | N | N | N | N | N | N | N | N | N | N |
| I39 | update_boundary per stage (H3) | yes | N | N | N | N | N | N | N | N | N | N |
| I40 | Per-call gather-list rebuild | yes | N | N | N | N | N | N | N | N | N | N |
| I41 | Masked MinMax/MaxLoc stats | yes | N | N | N | W | N | W | W | W | N | W |
| I42 | Error field and dumps | yes | N | N | N | N | N | W | N | N | N | N |
| I43 | Outer time loop, dispatch | | N | N | N | N | N | N | N | N | N | N |
| I44 | Eigenvalue diagnostic | | N | N | N | N | W | W | N | W | W | N |
| I45 | Reference serial matvecs | | N | N | N | N | N | W | N | N | N | N |
| I46 | Expression-template kernels | | N | N | N | N | N | N | N | N | N | N |
| I47 | Euler scaffold | yes | N | N | N | W | N | N | W | W | W | W |
| I48 | Python dense/sparse assembly | | N | N | N | N | W | W | N | U | W | N |
| I49 | Python symbolic coefficient layer | | N | N | N | N | W | N | N | N | W | N |
| I50 | Stability-analysis cascade | | W | N | N | N | W | W | N | U | U | N |
| I51 | cpp_bridge and BO driver | | N | N | N | N | N | N | N | N | N | N |
| | **Totals N / W / U** | | 45 / 6 / 0 | 42 / 8 / 1 | 48 / 3 / 0 | 37 / 14 / 0 | 38 / 12 / 1 | 28 / 22 / 1 | 37 / 13 / 1 | 33 / 14 / 4 | 27 / 23 / 1 | 33 / 17 / 1 |
| | **Hot idioms (30) not native** | | 3 | 5 | 2 | 8 | 3 | 9 | 8 | 10 | 9 | 12 |

Reading notes:

- **Labels are not normalised across files.** Halide marks I35 native because it replaces the Lua callback with a generated expression (the NumPy/SciPy file does the same; section 4.2 relabels it). Every other framework needs the same replacement but calls it a workaround or unsupported. Under the JAX/PyTorch rule Halide's hot count would be 9, not 8; this table keeps the file's label. Devito and MLX call host NumPy setup a workaround, while JAX and Numba call it native.
- **"Native" for I50 and I44 means "stays in host SciPy".** No framework supplies a non-Hermitian sparse eigensolver or pseudospectra (section 6).
- **The hot-path count is the more useful row.** Numba (2), JAX (3) and DaCe (3) are lowest. Explicit-loop and SDFG models keep H1 and H2 close to their C++ form. Array and stencil frameworks have to decompose them.

### 4.2 The other eight frameworks

These options were not shortlisted, so the table lists only the non-native idioms. Every idiom not listed is native in that file, with one exception noted below the table (I35 for NumPy/SciPy).

| Framework | Unsupported | Workaround | Hot idioms (30) not native |
|---|---|---|---|
| Warp | I35 | I02, I03, I07, I08, I09, I16, I19, I27, I32, I46 | 5 |
| GT4Py | I02, I03, I35, I48, I49, I50 | I04, I06, I07, I08, I09, I14, I16, I19, I20, I21, I32, I41, I44, I47 | 8 |
| PyKokkos | I07, I32, I35 | I02, I03, I04, I05, I06, I08, I09, I10, I11, I12, I16, I34, I36, I44, I47 | 6 |
| Mojo/MAX | I35, I50 | I02, I06, I07, I10, I16, I19, I32, I34, I44, I47, I48, I49 | 6 |
| Triton | I35 | I02, I03, I06, I07, I08, I09, I16, I19, I20, I27, I32, I36, I47 | 8 |
| NumPy/SciPy | I32 | I07, I08, I09, I16, I17, I35 (relabelled, see below), I47 | 5 |
| Array API | I32, I35, I50 | I03, I07, I08, I09, I10, I16, I19, I20, I21, I23, I36, I44, I47 | 9 |
| CuPy | I35, I50 | I02, I03, I06, I07, I08, I09, I14, I16, I19, I27, I32, I36, I47 | 7 |

Notes on this table:

- **I35 is relabelled for NumPy/SciPy.** `frameworks/numpy_scipy.md` marks I35 native because a Python callable on arrays replaces the Lua callback. JAX labels the same replacement a workaround and PyTorch unsupported. So that the hot-path count is comparable with JAX and PyTorch, this part labels it a workaround for NumPy/SciPy, which raises the count from 4 to 5. The rule used here: a framework that needs the Lua per-point callback replaced by a generated array expression does not get a native label for I35. Halide in 4.1 keeps its file label (native); under this rule its hot count would be 9, not 8.
- **CuPy** labels come from `frameworks/cupy_legate.md`: 36 native, 13 workaround, 2 unsupported. Its seven non-native hot idioms are I16, I19, I27, I32, I35, I36 and I47.

## 5. Framework sections

Each section gives:
- a verdict;
- one line per hard idiom (H1 to H4, section 1.4);
- one code sketch;
- what is left out;
- a link to the detailed file.

The sketches are trimmed from the detailed files unless marked otherwise. None was compiled or run as part of this write-up. Each detailed file has all four sketches.

### 5.1 JAX (XLA), overall 6.5

**Verdict.** JAX 0.11.2 (2026-09-17) is a strong CPU/CUDA reference backend: all 51 idioms map, with the unstructured parts becoming host-built tables. Its local measurements on synthetic tables have it beating C++ at small 2D:
- RHS 14 us at 51x51;
- RK4 step 66 to 74 us.

The C++ step figure the file compared them against (0.3 to 1 ms) was an estimate, so the verifier treats the gain as plausible, not established. The C++ step has since been measured at 0.90 ms for the shipped heat.lua 51x51 case, and the estimated fixed-up C++ step is about 0.15 ms (section 2.1; an estimate). The JAX RHS figure (14 us) is below the measured 22.7 us C++ RHS at 51x51 (1 thread). The JAX figures still come from synthetic tables in a separate run, so the comparison stays indicative. Two local Apple GPU datasets now exist and should be read together: `frameworks/jax.md` records jax-mps fp32 runs (step 207 to 322 us at 51x51 and 86 to 92 ms at 256^3, 3 to 5x slower than CPU fp64 at the shipped sizes), and `05-measurements.md` Section D measures the same shape of step on the same machine through torch MPS and MLX (0.35 to 0.5 ms at 51x51, 43.6 to 57.2 ms at 257^3), with the GPU sync floor at 167 to 181 us; JAX itself was not in the Section D run. JAX has no credible Apple GPU path:
- jax-metal is abandoned (https://github.com/jax-ml/jax/issues/26968, confirmed).
- jax-mps 0.11.0 is an Alpha, single-maintainer, fp32-only MLX plugin whose CI only compiles (https://pypi.org/project/jax-mps/, confirmed).
- `jax.experimental.sparse` is officially no longer developed (https://docs.jax.dev/en/latest/jax.experimental.sparse.html, confirmed).

| Hard idiom | How |
|---|---|
| H1 | Whole-grid shifted-slice stencil (`lax.slice_in_dim` plus `jnp.pad`), closure rows overwritten with an ELL gather-dot and `.at[rows].set`, then a `where` mask for coverage. Workaround. |
| H2 | Compact COO `.at[rows].add` for B; one padded ELL over a concatenated `[D; R]` source merges Bf and Br. Native. |
| H3 | `jnp.where` on precomputed fluid and Dirichlet masks; face slices via `.at[0].set` with boundary-only MMS evaluation. Native. |
| H4 | `jax.jit` of the RK4 step or K steps via `fori_loop`, `donate_argnums`, and `optimization_barrier` between stages. Without barriers, XLA:CPU was reported to recompute stencil producers (3 to 5x slower, local, unverified). Native. |

```python
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
        return u, stats(u, t0 + (i+1)*dt, P)    # stays on device, read every K steps
    return jax.lax.fori_loop(0, K, body, (u, stats(u, t0, P)))
```

**Left out.**
- Nothing is impossible, but six idioms need workarounds (I07, I09, I16, I20, I35, I50).
- No Apple GPU fp64 and no maintained Metal plugin.
- Each scatter costs a D-sized copy and breaks fusion (local observation).
- Geometry changes recompile because shapes are static.
- `pure_callback` costs about 24 us per call (local).

Detailed file: [frameworks/jax.md](frameworks/jax.md).

### 5.2 PyTorch (eager, torch.compile/Inductor, MPS), overall 6.5

**Verdict.** On CPU and CUDA, PyTorch 2.14.1 (announced 2026-09-30, https://dev-discuss.pytorch.org/t/pytorch-2-14-1-general-availability/3455, confirmed) maps every runtime idiom except I35 to dense ops, and Inductor fuses a full RK4 step on CPU with zero graph breaks (local, unverified). Large 3D CPU parity is likely. Small 2D is marginal: about 70 us of fixed overhead per compiled step puts it at 1 to 2x of C++ (local, unverified). The file's comparison used an older, higher C++ RHS estimate. Against the measured bar (section 2.1), its toy 51x51 RHS (27 to 28 us eager, local) is close to the 22.7 us C++ RHS, and its toy RK4 step (128 to 146 us compiled, local) is well below the measured 0.90 ms shipped step and close to the estimated fixed-up 0.15 ms (an estimate). Torch is already a locked dependency of `scripts/stencil_gen` for BoTorch. The MPS backend is the only mainstream Apple-GPU route in the survey, with these limits:
- fp32 only (confirmed);
- no graph replay on the 2.14 `torch.mps` page (confirmed as listed, absence elsewhere inferred);
- an Inductor-MPS backend with an open silent-correctness bug (https://github.com/pytorch/pytorch/issues/169738, confirmed).

Measured on the owner's Mac (`05-measurements.md` Section D, torch 2.11.0, synthetic SHOCCS-shaped step): CPU fp64 eager 0.149 ms and compiled 0.168 ms at 51x51 against 91 us for four C++ RHS submits; 4.5 ms compiled at 65^3 against 3.2 ms (8 threads), with 4 intra-op threads; eager dispatch 0.75 us per op and 6.4 us per compiled call, so torch.compile is slower than eager at the shipped sizes. MPS fp32: eager 1.5 to 2.2 ms per step at the small sizes (about a hundred kernels per step), compiled 0.50 ms at 51x51 held by a 175 to 181 us sync floor, 1.09 ms at 65^3 and 43.6 ms at 257^3 (the fastest path there). torch.compile on MPS emitted invalid Metal for an in-place face-slice fill (InductorError: SyntaxError) and needed a static mask fallback: a measured instance of Inductor-MPS immaturity beside the open correctness issue above.

| Hard idiom | How |
|---|---|
| H1 | `F.pad` plus `narrow` shifted slices, `torch.where` for coverage, `index_put` overwrite of ELL closure rows. Alternative: a C++ extension or `torch.mps.compile_shader` kernel. Workaround. |
| H2 | `index_add` with compact COO for B; ELL gather-dot over `torch.cat([u_flat, uR, zero])`. Avoid `torch.sparse` on MPS. Native. |
| H3 | `torch.where` masks; `index_put` of boundary-only MMS values; `t` as a 0-d tensor so compile does not respecialise. Native. |
| H4 | `torch.compile(step, fullgraph=True)` per step on CPU; `mode="reduce-overhead"` CUDA graphs (unverified here); MPS has fusion only, no replay. Workaround. |

```python
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

**Left out.**
- I35 (no traced per-point callbacks).
- fp64 on MPS.
- Graph replay on MPS.
- Whole-time-loop compilation: unrolling K steps did not help in the survey's test.
- `torch.sparse` and `compile_shader` kernels are opaque to Inductor.

Detailed file: [frameworks/pytorch.md](frameworks/pytorch.md).

### 5.3 Numba, overall 6.5

**Verdict.** Numba 0.67.0 (11 Aug 2026, https://numba.readthedocs.io/en/stable/release/0.67.0-notes.html, confirmed) is the best fit found for the CPU half of a DSL. Explicit `njit`/`prange` loops keep the per-line ragged structure as ordinary control flow, and a whole RHS or RK4 step compiles into one function with no per-op dispatch. Its weaknesses:
- No Apple GPU path (absence unverified at tracker level).
- numba-cuda ships no macOS wheels (https://pypi.org/project/numba-cuda/, confirmed).
- Strided-axis SIMD needs loop reordering by the generator.
- SVML is unsupported (confirmed in the 0.66 notes).
- JIT compile time for large generated closures.

All performance figures are unmeasured estimates.

| Hard idiom | How |
|---|---|
| H1 | `prange` over the line table with left/interior/right loops per line: a direct port of the TeamPolicy kernel. For large 3D, `prange` over slow pencils with a unit-stride inner loop. Native. |
| H2 | Compact COO loop for B; ELL loop over R rows reading `u_D` or `u_R` by index offset. No atomics, since rows are unique. Native. |
| H3 | `prange` over fluid and Dirichlet index lists; face loops evaluate the MMS at boundary points only. Native. |
| H4 | No graph object, and none needed: the RHS, the RK4 step, or the whole time loop is one `njit` function. Labelled workaround because there is no capture API. |

```python
@njit(parallel=True, fastmath=True, cache=True)
def block_apply(meta, coef, c, x, b):
    for l in prange(meta.shape[0]):
        ro, co, st, lr, lc, lo, ir = meta[l,0], meta[l,1], meta[l,2], meta[l,3], meta[l,4], meta[l,5], meta[l,6]
        rr, rc, ro2, rco = meta[l,9], meta[l,10], meta[l,11], meta[l,12]
        for r in range(lr + ir + rr):
            s = 0.0
            if r < lr:
                for j in range(lc): s += coef[lo + r*lc + j] * x[co + j*st]
            elif r < lr + ir:
                o = ro + r*st
                for j in range(3): s += c[j] * x[o + (j-1)*st]   # taps specialised by codegen
            else:
                for j in range(rc): s += coef[ro2 + (r-lr-ir)*rc + j] * x[rco + j*st]
            b[ro + r*st] = s
```

**Left out.**
- Apple GPU execution of any idiom (a backend gap, not an idiom gap).
- I35 as a true per-point Python callback (objmode is far too slow).
- Automatic SIMD on strided axes.
- Cross-loop fusion: the generator must write fused loops.

Correction from verification: `numba.pycc` AOT is deprecated, not removed.

Detailed file: [frameworks/numba.md](frameworks/numba.md).

### 5.4 Halide (Python bindings), overall 6.3

**Verdict.** Halide 21.0.0 (2025-09-16, confirmed via GitHub API and PyPI) is a strong CPU/CUDA execution substrate whose central feature is whole-step fusion. One `Pipeline` per RK4 step would replace roughly 280 launches. Pipelines stay geometry independent because the cut-cell irregularity becomes host-built masks, a row-id map and padded ELL tables, read through gathers rather than scatters. Its Metal target is fp32 only: `CodeGen_Metal_Dev.cpp` errors on 64-bit float literals (confirmed). All performance numbers are estimates. One survey sketch used a nonexistent API (`hl.sum_taps`) and was fixed.

| Hard idiom | How |
|---|---|
| H1 | Per-point gather: `select(rid >= 0, closure, select(interior, stencil, old))` with an RDom over the ELL width. No scatter. Workaround. |
| H2 | ELL gather `sum_k W(k,q) * s(COL(k,q))` over the concatenated `[D|Rx|Ry|Rz]` state; B folded into the closure ELL. Workaround. |
| H3 | `select` on coordinate Vars for faces and on an R mask for object Dirichlet points, instead of scatter updates. Native. |
| H4 | One `hl.Pipeline` over the RK4 step with `compute_root` at RHS boundaries, compiled to a Callable. Native. |

```python
k = hl.RDom([hl.Range(0, TMAX)])
ub = hl.BoundaryConditions.repeat_edge(u)
interior = sum(c[j] * ub(x + (j - P), y, z) for j in range(2 * P + 1))  # x axis shown
r = hl.max(rid(x, y, z), 0)
closure = hl.sum(C(k, r) * uf(IDX(k, r)))        # uf: flat alias of [D|Rx|Ry|Rz]
du = hl.Func('du')
du[x, y, z] = hl.select(rid(x, y, z) >= 0, closure,
                        hl.select(mask(x, y, z) != 0, interior, u(x, y, z)))
du.parallel(y).vectorize(x, 8)
```

**Left out.**
- Scatter update definitions (serial on CPU, atomic on GPU; replaced by masked gathers).
- Ragged CSR with per-row extents (use padded ELL instead).
- Any sparse type.
- A time-loop construct.
- fp64 on Metal.
- Weak reductions on Metal.

Unverified items:
- the per-call Callable overhead;
- JIT latency;
- whether a persistent JIT cache exists.

Detailed file: [frameworks/halide.md](frameworks/halide.md).

### 5.5 DaCe, overall 6

**Verdict.** DaCe's SDFG model is the closest IR match to the SHOCCS idioms:
- maps;
- indirect memlets;
- dynamic map ranges;
- write-conflict-resolution scatter.

A whole step could in principle compile to one fused C++/OpenMP library. There is no Metal, OpenCL or Apple GPU backend in the docs or release notes (https://spcldace.readthedocs.io/en/latest/optimization/gpu.html, confirmed as a negative result). The latest release is v2.0.0-alpha10 (3 Oct, year inferred), and its GPU codegen is being rewritten (https://github.com/spcl/dace/releases, confirmed). Most CPU-side claims are unverified, including:
- indirect indexing in the Python frontend;
- whole-program fusion;
- call overhead;
- compile time.

| Hard idiom | How |
|---|---|
| H1 | Decomposed full-grid stencil map plus an ELL map that overwrites closure rows; or a per-line map with dynamic-range inner maps. Workaround. |
| H2 | COO map with write-conflict-resolution add for B; ELL map with indirect reads for Bf and Br. Native (unverified in the frontend). |
| H3 | Index-list maps and slice assignment. Native. |
| H4 | Whole RK4 step as one `@dace.program` compiled to one C++ library (plausible from the design, unverified against docs). Native. |

```python
@dace.program
def cut_coupling(u: dace.float64[ND], uR: dace.float64[NR], du: dace.float64[ND],
                 duR: dace.float64[NR], rowD: dace.int32[NB], colR: dace.int32[NB],
                 wB: dace.float64[NB], idx: dace.int32[NR, K], W: dace.float64[NR, K],
                 selfc: dace.float64[NR]):
    for n in dace.map[0:NB]:
        du[rowD[n]] += wB[n] * uR[colR[n]]       # indirect write, wcr add
    for r in dace.map[0:NR]:
        acc = selfc[r] * uR[r]                   # Br diagonal
        for k in range(K):
            acc += W[r, k] * u[idx[r, k]]        # padded entries: W=0, idx=0
        duR[r] += acc
```

**Left out.**
- I35.
- Apple GPU (adding one would be a multi-month compiler project, per the surveyor's estimate).
- Eigen and stability tooling stays in host SciPy.
- Compile latency and a run-time C++ toolchain hurt parameter sweeps.

Detailed file: [frameworks/dace.md](frameworks/dace.md).

### 5.6 Devito, overall 5.5

**Verdict.** Devito v4.8.23 (07 Aug 2026, https://github.com/devitocodes/devito/releases, confirmed) is a SymPy-based stencil compiler. It can fuse a whole multi-stage RK4 time loop, including indirect ELL/COO cut-cell couplings, into one C function. Local serial measurements:
- 7.2 us per fused RK4 step at 51x51 with 2000 steps per `apply()`;
- 0.45 ns/pt/step for a 7-point 256^3 update.

The detailed file compares the 51x51 figure with a C++ estimate of 0.3 to 1 ms per step. The C++ step has since been measured at 0.90 ms (shipped heat.lua 51x51), and the estimated fixed-up C++ step is about 0.15 ms (section 2.1; an estimate). The Devito test is serial, uses synthetic tables, and omits the full closure ELL, in-kernel MMS and Dirichlet SubDomain equations, so the speedup is indicative only. There is no Metal or Apple GPU path; Apple silicon appears only as CPU platforms in `archinfo.py` (confirmed). The default dtype is float32, so fp64 must be requested explicitly.

| Hard idiom | How |
|---|---|
| H1 | Masked constant-tap `Eq` per direction, plus closure rows as an indirect `Inc` over an ELL `(r, k)` pair. Devito emits `omp atomic` for the indirect write. Workaround. |
| H2 | Indirect `Inc` over custom Dimensions. The Bf loop index is the output, so no atomics. B is an indirect write, so atomics under OpenMP/ACC. Workaround. |
| H3 | Mask Functions and `SubDomain` equations with a symbolic `g(t, x, y, z)`. Native. |
| H4 | Whole RK4 time loop in one `Operator`. Stage equations need `implicit_dims=(time,)` or they are silently hoisted (reproduced locally). Native. |

```python
lap_int = (mi_x*u.dx2(weights=cx) + mi_y*u.dy2(weights=cy) + mi_z*u.dz2(weights=cz))
r, k = Dimension('r'), Dimension('k')
ox, oy, oz = (mk('o'+a, (r,), (nrow,), np.int32) for a in 'xyz')
ix, iy, iz = (mk('i'+a, (r, k), (nrow, Tmax), np.int32) for a in 'xyz')  # pad: dummy cell, C=0
C = mk('C', (r, k), (nrow, Tmax), np.float64)
eqs = [Eq(du, kappa*lap_int, implicit_dims=(time,)),
       Inc(du[ox[r], oy[r], oz[r]], kappa*C[r, k]*u[ix[r, k], iy[r, k], iz[r, k]],
           implicit_dims=(time,))]          # emitted as '#pragma omp atomic update'
```

**Left out.**
- I35.
- Apple GPU.
- A matrix-export mode, so eigen tooling needs a parallel scipy assembly.
- First-class reductions (stats run between `apply()` chunks).
- Calling `apply()` per RHS, at 0.2 to 0.8 ms each (local).

Unverified: whether native CUDA/HIP/SYCL codegen is commercial-only.

Detailed file: [frameworks/devito.md](frameworks/devito.md).

### 5.7 Taichi, overall 5

**Verdict.** Imperative `@ti.kernel` loops fit the cut-cell irregularity well, and CPU and CUDA run f64. Constraints:
- **Metal:** f32 only (https://docs.taichi-lang.org/docs/type, confirmed); sparse SNodes removed in v1.4.0 (confirmed); the Metal AOT runtime is experimental.
- **Fusion:** none across kernels, so the DSL would hand-fuse the whole RHS into one kernel.
- **Releases:** the latest is v1.7.4 (2025-07-31, https://pypi.org/project/taichi/, confirmed), 14 months old, with 878 open issues. "Maintenance mode" is an inference.
- **Code sketches:** two survey sketches were invalid and were corrected or labelled.

| Hard idiom | How |
|---|---|
| H1 | Three loops in one kernel per direction: masked interior over `ti.ndrange`, wall closure on boundary planes with `ti.static` unrolling, ELL object closure. Workaround. |
| H2 | One kernel: COO loop for B and ELL loops for Bf and Br over R points. Native. |
| H3 | Indexed loops over sorted fluid and Dirichlet lists; boundary-only exact solution in a `@ti.func`. Native (sketch is pseudo-code). |
| H4 | Hand-fused large kernel per RHS; each top-level loop is still a separate task. The `ti.graph` API name and backend list are unverified. Workaround. |

```python
@ti.kernel
def cut_coupling(uD: ti.types.ndarray(), uR: ti.types.ndarray(),
                 duD: ti.types.ndarray(), duR: ti.types.ndarray(),
                 b_row: ti.types.ndarray(), b_col: ti.types.ndarray(),
                 b_w: ti.types.ndarray(), nb: ti.i32,
                 f_idx: ti.types.ndarray(), f_w: ti.types.ndarray(),
                 r_idx: ti.types.ndarray(), r_w: ti.types.ndarray(), nR: ti.i32):
    for e in range(nb):                 # rows unique, no atomics
        duD[b_row[e]] += b_w[e] * uR[b_col[e]]
    for q in range(nR):
        s = 0.0
        for k in ti.static(range(KF)):  # D -> R, padded with weight 0
            s += f_w[q, k] * uD[f_idx[q, k]]
        for k in ti.static(range(KR)):  # R -> R, merged
            s += r_w[q, k] * uR[r_idx[q, k]]
        duR[q] += s
```

**Left out.**
- I35.
- f64 and sparse structures on Metal.
- Known graph replay on Metal.
- Cross-kernel fusion.
- PyTorch MPS interop (unverified, probably absent).

Detailed file: [frameworks/taichi.md](frameworks/taichi.md).

### 5.8 pystencils (and lbmpy), overall 5

**Verdict.** pystencils 2.1 (2026-09-29; 2.0 GA 2026-07-02; https://pypi.org/pypi/pystencils/json, confirmed) generates C++ for CPU (SIMD, OpenMP) and CUDA/HIP via CuPy, all with fp64. Its main attraction is that `stencil_gen`'s SymPy closures lower directly into Assignments. Limitations:
- **Cut-cell structure:** has to become interior stencils plus index-list kernels via `CreateKernelConfig.index_field`. That option is exclusive with ghost layers and runs in unspecified order (confirmed).
- **Apple GPU:** no Metal target was found. This is absence of evidence only.
- **Small 2D:** Python dispatch (a few us per kernel, estimated) threatens it unless the step is exported to C++ through pystencils-sfg (confirmed to exist).
- **Sketches:** they use 1.x spellings.

| Hard idiom | How |
|---|---|
| H1 | Whole-grid Assignment for the interior plus an index-list kernel that overwrites closure rows. Encoding ragged segments as index lists loses vectorisation. Workaround. |
| H2 | Padded ELL index-list kernels; no CSR type. Workaround. |
| H3 | Mask Field multiplier and `iteration_slice` face kernels; object points via an index kernel. Native. |
| H4 | No graph capture. Fuse per AssignmentCollection (5 to 10 kernels per RHS), or emit a C++ host function via pystencils-sfg. Workaround. |

```python
fluid = ps.fields('fluid: double[3D]'); dirm = ps.fields('dirm: double[3D]')
src, du = ps.fields('src,du: double[3D]')
expr = (1 - dirm.center) * (du.center + fluid.center * src.center)
k = ps.create_kernel(ps.Assignment(du.center, expr))
k_face = ps.create_kernel(ps.Assignment(u.center, sol.center),
                          iteration_slice=ps.make_slice[0:1, :, :])   # 1.x spelling
# object points: index-list kernel du_R[idx[i]] = 0 over the Dirichlet object list
```

**Left out.**
- I02, I35, I48 and I50 are unsupported.
- No ragged strided-segment primitive.
- No confirmed reduction primitive (unchecked).
- No Apple GPU.

Correction from verification: lbmpy 2.1 was yanked (pin 2.1.1).

Detailed file: [frameworks/pystencils.md](frameworks/pystencils.md).

### 5.9 Mojo / MAX (Modular), overall 5

**Verdict.**
- **Releases:** Mojo 1.0 shipped on 11 Aug 2026 with Modular 26.5; MAX 26.6 (17 Sep 2026) is current (https://max.modular.com/changelog, confirmed).
- **CPU:** a generated whole-RHS Mojo function should match C++ through MLIR/LLVM. This is an architectural argument; nothing was benchmarked.
- **Apple GPU:** real but young and ML-focused. MAX graphs run on M1 to M5 in nightlies (https://forum.modular.com/t/max-models-can-now-run-on-apple-silicon-gpus/3283, confirmed). A user custom-kernel path is undocumented. Atomics status is unknown after Mar 2026.
- **DSL role:** Mojo is not a Python superset, so it would be a codegen target, not a DSL host.
- **Ownership:** Modular is confirmed as part of Qualcomm (https://forum.modular.com/t/qualcomm-to-acquire-modular/3271). The survey's sketches use closure syntax that 26.6 changed.

| Hard idiom | How |
|---|---|
| H1 | Hand-written per-direction kernel: `parallelize` over pencils, SIMD over the fast index, fragment table for offsets. No TeamPolicy analogue. Workaround. |
| H2 | Plain loops over compact COO and a padded ELL over `[D; R]`, parallelised per R row. Native. |
| H3 | Vectorised masked `select` for the fluid add; slab loops for faces; index-list loops for R. Native. |
| H4 | One compiled Mojo function per step on CPU. `createGraphBuilder` is CUDA/HIP only, so there is no capture on Metal (confirmed). Workaround. |

```mojo
# Pre-26.6 closure syntax: will not compile on 26.6 as written.
fn apply_Bf[K: Int](duR: UnsafePointer[Float64], uD: UnsafePointer[Float64],
                    uR: UnsafePointer[Float64], idx: UnsafePointer[Int32],
                    C: UnsafePointer[Float64], nR: Int, nD: Int):
  @parameter
  fn row(q: Int):
    var acc = 0.0
    @parameter
    for k in range(K):
      var i = Int(idx[q*K + k])
      acc += C[q*K + k] * (uD[i] if i < nD else uR[i - nD])
    duR[q] += acc
  parallelize[row](nR)
```

**Left out.**
- I35 and I50.
- fp64 on Apple GPU.
- Sparse, FFT and eigensolver libraries.
- Automatic fusion of hand-written loops.

Detailed file: [frameworks/mojo.md](frameworks/mojo.md).

### 5.10 NumPy/SciPy baseline (assembled sparse operator), overall 5

**Verdict.** The baseline assembles the whole cut-cell operator into one `scipy.sparse` CSR over the packed `D|Rx|Ry|Rz` vector. This turns all six cut-cell irregularities into matrix rows. It is a mature, fp64, low-risk CPU path. Its best roles are:
- correctness oracle;
- producer of the index tables other backends consume;
- engine for stability analysis of the real cut-cell operator.

Performance:
- Small 2D parity is plausible, and one measurement supports it: an eager NumPy synthetic step (shifted-slice stencil plus ELL gather, not the assembled CSR) costs 0.103 ms at 51x51 against 91 us for four C++ RHS submits, with a 0.46 us dispatch floor (`05-measurements.md` Section D).
- Large 3D: the same eager path costs 5.3 ms at 65^3 against 3.2 ms (8 threads) and 7.5 ms (1 thread) of C++ submits, and 448 ms at 257^3 (measured, synthetic). The assembled-CSR form is unmeasured and likely loses 3 to 8x single-threaded, because CSR moves an estimated 3 to 5x more bytes per point than a structured stencil and nothing fuses.
- csr_matvec is believed single-threaded, but that is unverified for 1.18.

Current SciPy is 1.18.1 (https://discuss.scientific-python.org/t/ann-scipy-1-18-1-release/2554, a correction to the survey).

| Hard idiom | How |
|---|---|
| H1 | Rows of one global CSR, `y = A @ u`. Optionally structured slicing for the interior plus CSR for closures. Workaround. |
| H2 | Blocks of the same global CSR: B columns in the R range, Bf and Br rows in the R range. No dependency chain. Native. |
| H3 | Face slices, `np.add(..., where=fluid_mask)`, gather lists built once, boundary-only MMS. Native. |
| H4 | No graph, JIT or fusion. The fallback is a function with few NumPy calls (about 20 per RK4 step plus 4 SpMVs). Unsupported. |

```python
import scipy.sparse as sp
off = {'D': 0, 'x': D, 'y': D + nRx, 'z': D + nRx + nRy}
rB  = rows_next_to_cut;  cB  = off[dir] + obj_coord;  wB  = w_B      # B: R -> D
rBf = off[r] + q_idx;    cBf = d_cols;                wBf = w_Bf     # Bf: D -> R(r)
rBr = off[r] + q_idx;    cBr = off[r] + q_cols;       wBr = w_Br     # Br: R(r) -> R(r)
rows = np.r_[rO, rB, rBf, rBr]; cols = np.r_[cO, cB, cBf, cBr]; w = np.r_[wO, wB, wBf, wBr]
A = sp.coo_array((w, (rows, cols)), shape=(N, N)).tocsr()   # duplicates summed
y = A @ u                         # D rows and R rows in one SpMV
```

**Left out.**
- I32.
- Any GPU or MPS path.
- Fusion of elementwise passes.
- Threaded SpMV in SciPy itself (unverified).

Detailed file: [frameworks/numpy_scipy.md](frameworks/numpy_scipy.md).

### 5.11 Direct Metal from Python, overall 5

**Verdict.** This option covers pyobjc Metal, metalcompute, MLX `mx.fast.metal_kernel` and metal-cpp via nanobind. It is an execution substrate, not a PDE framework. It gives the best Apple-GPU access in the survey because we write the MSL ourselves, so every idiom except I35 is expressible. Constraints:
- **Precision:** the GPU path is fp32 or hand-written df64.
- **No CPU or CUDA path for MSL.**
- **Small 2D:** at the shipped sizes the GPU loses to the CPU, now measured: the GPU op-plus-sync floor is 167 to 181 us against a 22.7 us C++ RHS at 51x51 (1 thread, section 2.1), and a whole-step-compiled fp32 step through MLX or torch MPS costs 0.35 to 0.5 ms against 0.085 ms for the best fused CPU fp64 step from Python (`05-measurements.md` Section D). Only batching independent cases per dispatch changes that.
- **MLX custom kernels:** inputs are read-only and nothing is done in place upstream (https://github.com/ml-explore/mlx/issues/2547, confirmed).
- **metalcompute:** stale at 0.2.9, January 2025 (https://pypi.org/pypi/metalcompute/json, confirmed).

| Hard idiom | How |
|---|---|
| H1 | One MSL thread per (line, row) reading the 13-int meta record (a faithful port), or a fast-index-major interior kernel plus an ELL closure kernel. Workaround. |
| H2 | COO kernel for B with unique rows; ELL kernel over a concatenated `[D; R]` buffer for Bf and Br. Workaround. |
| H3 | Index-list kernels with indices uploaded once; masked `where` for the fluid source. Native. |
| H4 | Encode the RHS or step into one command buffer and commit once. An indirect command buffer is reachable from pyobjc or metal-cpp, not MLX. `mx.compile` cannot replay custom kernels (medium confidence). Workaround. |

```python
src = '''
uint r = thread_position_in_grid.x, l = thread_position_in_grid.y;
constant int* m = meta + l*13;
if ((int)r >= m[3] + m[6] + m[9]) return;
int s = m[2], out = m[0] + r*s;
float acc = 0.f;
if ((int)r < m[3])        { for (int j=0;j<m[4];j++) acc += coef[m[5]+r*m[4]+j]*x[m[1]+j*s]; }
else if ((int)r < m[3]+m[6]) { int w=m[8]; for (int j=0;j<w;j++) acc += coef[m[7]+j]*x[out+(j-w/2)*s]; }
else { int rr=r-m[3]-m[6]; for (int j=0;j<m[10];j++) acc += coef[m[11]+rr*m[10]+j]*x[m[12]+j*s]; }
y[out] = acc;
'''
k = mx.fast.metal_kernel('blk', ['meta', 'coef', 'x'], ['y'], src)
y = k(inputs=[meta, coef, x], grid=(maxrows, nL, 1), threadgroup=(64, 1, 1),
      output_shapes=[x.shape], output_dtypes=[mx.float32], init_value=0)[0]
```

**Left out.**
- I35.
- fp64 on the GPU.
- Any CPU or CUDA execution of the same kernels.
- In-place updates and replayable graphs through MLX.
- A metal-cpp extension would mean owning C++ maintenance.

Detailed file: [frameworks/direct_metal.md](frameworks/direct_metal.md).

### 5.12 Apple MLX, overall 4.5

**Verdict.** MLX 0.32.3 (29 Sep 2026, https://github.com/ml-explore/mlx/releases, confirmed) can express every runtime idiom but fails both hard constraints:
- **No GPU fp64:** Metal has none, and emulation was declined (#1905, confirmed).
- **CPU fp64, one blocker softened and one standing:**
  - Single-core execution of built-in ops was inferred from a maintainer statement about custom primitives (https://github.com/ml-explore/mlx/issues/2185, uncertain). Measured: compiled MLX CPU fp64 ran the synthetic SHOCCS-shaped step in 0.085 ms at 51x51, the fastest small-2D path of any backend, and in 4.2 ms at 65^3, within 1.4x of eight-thread Kokkos RHS submits (`05-measurements.md` Section D). Whatever its threading, the CPU fp64 path is not slow at these sizes; it is 1.3x the best torch path at 257^3 (300 against 272 ms).
  - fp64 `exp`, `sin`, `cos` and `erf` use fp32-accuracy approximations (https://github.com/ml-explore/mlx/issues/3047, confirmed, open), which would corrupt MMS convergence studies. This blocker stands.
  - `mx.array` silently converts a numpy float64 array to float32 unless the dtype is passed (observed in the Section D run).
- **Fusion:** `mx.compile` fuses only elementwise ops, with limits of 24 arrays and depth 11 (confirmed in https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/compile.cpp).
- **Sparse:** no first-party support (#2728, wontfix, confirmed).
- **GPU timings (measured, `05-measurements.md` Section D):** MLX GPU fp32 with `mx.compile` over the whole step: 0.353 ms at 51x51 and 0.376 ms at 201x201 (held by a 167 us op-plus-sync floor), 1.01 ms at 65^3 (the fastest path there), 57.2 ms at 257^3; fp32 error 1.3e-7 to 2.4e-7 relative after 10 steps on synthetic closures.

| Hard idiom | How |
|---|---|
| H1 | Shifted slices fused by `mx.compile`, then `mx.pad` (an extra unfused copy), `mx.where`, and an ELL gather (`mx.take`) plus `.at[rows].add`. Or one `mx.fast.metal_kernel` per direction. Workaround. |
| H2 | `.at[rowsB].add` for B; ELL gather-dot over `mx.concatenate([uD, R[r]])`. Workaround. |
| H3 | `mx.where` with precomputed masks; `slice_update` on face planes. Native. |
| H4 | `mx.compile` over the RK4 step plus one `mx.eval` per step. No graph replay; gathers, scatters and reductions stay separate kernels. Workaround. |

```python
def axis_op(u3, axis, c, m_int, rows, idx, C):
    w = len(c); h = w // 2; n = u3.shape[axis]
    sl = lambda a, b: tuple(slice(a, b) if d == axis else slice(None) for d in range(3))
    s = sum(c[j] * u3[sl(j, n - w + 1 + j)] for j in range(w))  # fused elementwise
    s = mx.pad(s, [(h, h) if d == axis else (0, 0) for d in range(3)])  # separate copy
    s = mx.where(m_int, s, 0.0).reshape(-1)
    vals = (C * mx.take(u3.reshape(-1), idx)).sum(-1)   # gather + reduce kernels
    return s.at[rows].add(vals)                         # scatter, rows unique
```

**Left out.**
- I50.
- fp64 on Metal.
- Multithreaded and accurate fp64 on CPU.
- Sparse types.
- Fusion beyond elementwise ops.

Correction from verification: the TF32 matmul default (#3860) was closed as a documentation fix, and it also affects Apple silicon.

Detailed file: [frameworks/mlx.md](frameworks/mlx.md).

### 5.13 GT4Py, overall 4

**Verdict.** GT4Py 1.2.3 (5 Oct 2026, https://pypi.org/project/gt4py/, confirmed) fuses stencils on CPU, NVIDIA and AMD. It has a stable `gt4py.cartesian` and an experimental `gt4py.next` that models unstructured access through neighbour tables. Recent churn (confirmed from release notes):
- the DaCe pin moved from 2.0.0a5 to a10 within 1.2.x, with a8 excluded;
- `field[...]` assignment was removed in 1.2.2;
- the imperative gtfn backend was removed in 1.2.3.

There is no Metal or macOS GPU support anywhere (confirmed, absence of evidence). The gather-only model and the API spellings in the sketches are unverified recollections.

| Hard idiom | How |
|---|---|
| H1 | Whole-grid masked `field_operator` plus closure rows via a neighbour table and a reverse table, combined with `where`. A literal ragged block matvec is not expressible. Workaround. |
| H2 | `neighbor_sum` over padded ELL connectivities for Bf, Br and B. No scatter or atomic add (believed, unverified). Workaround. |
| H3 | `where(fluid, ...)` and `where(dirich, 0, ...)`; domain-restricted program calls for faces. Native. |
| H4 | A program with several operators compiles to one generated function, but D and R live on different Dimensions, so expect 3 to 6 kernels per RHS. No capture. Workaround. |

```python
@gtx.field_operator      # pseudocode: API spellings unverified
def bc_and_src(du: gtx.Field[[DDim], float64], src: gtx.Field[[DDim], float64],
               fluid: gtx.Field[[DDim], bool], dirich: gtx.Field[[DDim], bool],
               k: float64) -> gtx.Field[[DDim], float64]:
    out = k*du + where(fluid, src, 0.0)      # masked add, no index list
    return where(dirich, 0.0, out)           # Dirichlet fill as mask
```

**Left out.**
- I02, I03, I35, I48, I49 and I50 are unsupported.
- Scatter (believed).
- Global reductions inside operators.
- Apple GPU.

The claim that Cartesian is in maintenance mode was refuted and dropped.

Detailed file: [frameworks/gt4py.md](frameworks/gt4py.md).

### 5.14 Triton, overall 4

**Verdict.**
- **CUDA/ROCm:** Triton 3.8.0 (28 Aug, year inferred; https://github.com/triton-lang/triton/releases, confirmed) is a mature tile-kernel DSL. Everything hot in SHOCCS maps to it, and fused kernels could beat the C++ team-per-line path on large CUDA runs.
- **CPU:** mainline has no backend. triton-cpu is an experimental Linux-only fork (https://github.com/triton-lang/triton-cpu, uncertain). This fails the CPU bar.
- **Apple GPU:** out of tree only.
  - triton-ext PR #126 is open and unmerged (https://github.com/triton-lang/triton-ext/pull/126, confirmed).
  - triton-msl 0.3.0 is alpha and targets Triton 3.7.0, not 3.8 or 3.9 (https://github.com/Janssena/triton-msl, a correction to the survey).
- **Launch overhead:** 10 to 30 us per launch, an unverified estimate.

| Hard idiom | How |
|---|---|
| H1 | 2D tile kernel for the interior with constexpr taps and an interior mask, then a `closure_rows` ELL kernel that overwrites. Workaround. |
| H2 | COO kernel with `tl.atomic_add` (only for duplicate safety); ELL kernel over `[D; R]` selected by `idx >= ND`. Native. |
| H3 | Masked elementwise kernel for the source and Dirichlet zero; strided `face_set` kernel. Native. |
| H4 | No graph object. Use `torch.cuda.CUDAGraph` around the launches on CUDA. Nothing on Metal or CPU. Workaround. |

```python
@triton.jit
def closure_rows(x, out, row_out, idx, C, nrows, TMAX: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < nrows
    t = tl.arange(0, TMAX)[None, :]
    ii = tl.load(idx + r[:, None] * TMAX + t, mask=rm[:, None], other=0)
    cc = tl.load(C + r[:, None] * TMAX + t, mask=rm[:, None], other=0.0)
    v = tl.sum(cc * tl.load(x + ii), axis=1)
    tl.store(out + tl.load(row_out + r, mask=rm), v, mask=rm)  # overwrites stencil result
```

**Left out.**
- I35.
- CPU parity.
- Mainline Apple support.
- Graph capture outside CUDA.

Unverified:
- fp64 `tl.dot` and fp64 atomics;
- the 19 to 34% M4 speedup claimed for PR #126.

Detailed file: [frameworks/triton.md](frameworks/triton.md).

### 5.15 Python Array API standard, overall 4

**Verdict.** The standard is a stable spec plus thin shims, not an execution engine (revision 2025.12; array-api-compat 1.14.0 and 1.15.0, https://data-apis.org/array-api-compat/changelog.html, confirmed). The supported backends are NumPy, CuPy, PyTorch, JAX and others. MLX is not one of them (confirmed). Most idioms map onto standard ops. The real gaps are:
- scatter-add (only the third-party array-api-extra `at` shim fills it, semantics unverified);
- pad, conv and sparse types;
- any fusion or compilation.

Parity therefore needs `jax.jit` or `torch.compile`, which sit outside the standard. Corrections from verification:
- `nextafter` is in the standard (since 2024.12);
- integer array indexing makes `take` more capable than the survey said.

| Hard idiom | How |
|---|---|
| H1 | Shifted slices built with `concat` and zeros (no `pad` in the standard), `where` for coverage, ELL via `take` plus `xpx.at(...).set`. Workaround. |
| H2 | `take` plus `xpx.at(du_D)[rows].add` for B; ELL over `concat([uD, uR])` for Bf and Br. Workaround. |
| H3 | `where` masks; `xpx.at(u)[bidx].set` for boundary values. Native. |
| H4 | Not in the standard; delegate to `jax.jit` or `torch.compile`. Unsupported. |

```python
def cut_terms(uD, uR, du_D, du_R, t):
    v = xp.take(uR[dir], t.B_col) * t.B_w
    du_D = xpx.at(du_D)[t.B_row].add(v)                # array-api-extra, not the standard
    src = xp.concat([uD, uR[r]])
    g = xp.reshape(xp.take(src, t.E_idx_flat), (t.nR, t.K))
    du_R = du_R + xp.sum(g * t.E_C, axis=1)            # padded idx -> dummy slot, C=0
    return du_D, du_R
```

**Left out.**
- I32, I35 and I50.
- Scatter-add, pad, conv, einsum, sparse, lexsort and bincount.
- Compilation.
- Any Apple GPU path.

Detailed file: [frameworks/array_api.md](frameworks/array_api.md).

### 5.16 NVIDIA Warp, overall 3.5

**Verdict.** Warp (latest 1.18.0, 2026-10-05; the survey's 1.17.0 was stale; https://nvidia.github.io/warp/latest/project/changelog.html) expresses nearly every idiom natively in explicit-index kernels. Two blockers are confirmed:
- CPU launches "run serially and synchronously" (https://nvidia.github.io/warp/latest/user_guide/faq.html), so multicore parity with Kokkos OpenMP is unlikely at 3D sizes.
- macOS arm64 is "CPU Only" (https://nvidia.github.io/warp/latest/user_guide/compatibility.html).

Experimental CPU graph capture (since 1.13) may remove Python dispatch. Experimental external C++ module support (1.17/1.18) improves interop. Neither was evaluated.

| Hard idiom | How |
|---|---|
| H1 | Thread-per-line kernel over a `wp.struct` line table (direct port), or a whole-grid stencil plus an ELL kernel. Serial on CPU. Workaround. |
| H2 | COO kernel with `wp.atomic_add` for B; ELL gather kernel per R row for Bf. Native. |
| H3 | Mask kernel, index-list kernel and face kernel. Native. |
| H4 | `wp.ScopedCapture` on CUDA; experimental CPU `capture_begin`/`capture_launch`. Neither fuses kernels. Workaround. |

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
    for k in range(K):
        s += C[q, k] * uD[idx[q, k]]
    duR[q] += s
```

**Left out.**
- I35.
- Multithreaded CPU execution of a launch.
- Apple GPU.
- Cross-kernel fusion.
- Implicit fp64: `wp.float` is fp32, so the DSL must declare float64 everywhere.

Detailed file: [frameworks/warp.md](frameworks/warp.md).

### 5.17 PyKokkos, overall 3

**Verdict.** PyKokkos translates type-annotated Python workunits into Kokkos C++, so large-grid kernels should match hand-written Kokkos. It has:
- no Metal path (confirmed in the README and policies docs);
- no graph replay or whole-RHS compile;
- only `RangePolicy` and `TeamPolicy` documented (https://kokkos.org/pykokkos/policies.html, confirmed);
- PyFuser, which gave about 1.8x on a GPU microbenchmark (https://kokkos.org/pykokkos/fusion.html, confirmed) and is not graph replay.

Project health is weak:
- no tagged releases;
- an open Kokkos 5.0 migration issue (#369);
- open macOS build bugs (#390, #413), all confirmed (https://github.com/kokkos/pykokkos).

SHOCCS pins Kokkos 5.1.1. The existing C++ kernels cannot be reused.

| Hard idiom | How |
|---|---|
| H1 | `TeamPolicy` workunit reading the 13-int meta View (compile status unverified), or separate `RangePolicy` interior and closure kernels. Workaround. |
| H2 | `RangePolicy` workunits for compact COO and fixed-width ELL. Native. |
| H3 | Masked workunit and index-fill workunit. Native. |
| H4 | No Kokkos Graph exposure found; `PK_FUSION=naive` is lazy fusion only. Unsupported. |

```python
@pk.workunit
def ell_gather(r: int, idx: pk.View2D[pk.int32], C: pk.View2D[pk.double], out: pk.View1D[pk.int32],
               srcD: pk.View1D[pk.double], srcR: pk.View1D[pk.double], nD: int, dst: pk.View1D[pk.double]):
    acc: float = 0.0
    for k in range(idx.extent(1)):             # padded to Tmax; padded C == 0
        i: int = idx[r][k]
        acc += C[r][k] * (srcD[i] if i < nD else srcR[i - nD])
    dst[out[r]] += acc
pk.parallel_for(nR, ell_gather, idx=Bf_idx, C=Bf_C, out=Rrows, srcD=uD, srcR=uR, nD=nD, dst=duR)
```

**Left out.**
- I07, I32 and I35.
- Apple GPU.
- Reuse of existing C++ kernels.
- Python callbacks in kernels (open request #394).

The claimed Threads/Serial backends were downgraded to unverified.

Detailed file: [frameworks/pykokkos.md](frameworks/pykokkos.md).

### 5.18 CuPy / cuPyNumeric (Legate), overall 3

**Verdict.** This section draws on `frameworks/cupy_legate.md` (regenerated 2026-10-12; verification 7 confirmed, 0 refuted, 2 uncertain). All 51 idioms are mapped there: 36 native, 13 workaround (host-built ELL/COO tables, fused elementwise kernels, `RawKernel` for per-line closures) and 2 unsupported (I35, I50). CuPy:
- **Scope:** a mature NumPy/SciPy-compatible library for NVIDIA CUDA, with `RawKernel`, `ElementwiseKernel` and `cupy.fuse`.
- **Sparse:** CuPy 14.2.0 added COO, CSR, CSC and DIA arrays with 64-bit indexing (https://newreleases.io/project/pypi/cupy/release/14.2.0, confirmed).
- **Platforms:**
  - ROCm 7.x is experimental, with no wheels from v13.4.0 on and no sparse support.
  - The install docs do not mention macOS or Metal (https://docs.cupy.dev/en/latest/install.html, confirmed).
- **No CPU backend.**

cuPyNumeric is confirmed end of life: "The final release is 26.06.01" (https://docs.nvidia.com/cupynumeric/26.06/user/differences.html). It also falls back to NumPy for unsupported dimensions or dtypes. One correction from verification: the int32-only sparse limit applies only to `spsolve`, `spilu` and older `spsolve_triangular`, not to SpMV.

| Hard idiom | How |
|---|---|
| H1 | `RawKernel` port, or decomposed interior plus ELL. `RawKernel` ignores array views, so the caller manages strides (confirmed). |
| H2 | `cupyx.scipy.sparse` CSR SpMV, or a `RawKernel` ELL gather. |
| H3 | `ElementwiseKernel` masked updates with automatic broadcasting (confirmed). |
| H4 | Eager launches. `cupy.cuda.Graph` is unverified. |

```python
# Written for this report as an illustration; untested. The detailed file has its own sketches.
import cupy as cp
ell = cp.RawKernel(r'''
extern "C" __global__
void ell_gather(const double* src, const int* idx, const double* C,
                const int* rows, double* out, int n, int K) {
  int q = blockDim.x * blockIdx.x + threadIdx.x;
  if (q >= n) return;
  double s = 0.0;
  for (int k = 0; k < K; ++k) s += C[q*K + k] * src[idx[q*K + k]];
  out[rows[q]] = s;                 // rows unique: no atomics
}''', 'ell_gather')
ell(((n + 255) // 256,), (256,), (src, idx, C, rows, out, cp.int32(n), cp.int32(K)))
```

**Left out.**
- I35 and I50.
- Any CPU execution.
- Apple GPU.
- Sparse on ROCm.

Detailed file: [frameworks/cupy_legate.md](frameworks/cupy_legate.md).

## 6. What is left out everywhere

These are the idioms and capabilities that **no** surveyed framework provides natively in the form the C++ code uses. They have to be owned by the DSL, or replaced, whatever backend is chosen.

1. **Cross-direction interpolation (I07).** All 18 detailed files mark it workaround, or unsupported (PyKokkos). The C++ `interp_line` scan with data-dependent footprints becomes host code: NumPy or Numba with `searchsorted` over fragment tables, emitting padded `(idx, coef)` ELL rows.
2. **The literal per-point Lua callback (I35).** Every framework needs it replaced by a closed-form array expression generated from SymPy, or by a precomputed table. A per-point host callback in the hot path is unsupported or a severe performance trap everywhere:
   - JAX `pure_callback` costs about 24 us per call (local).
   - Numba falls back to objmode.
   - The rest have no mechanism at all.
   The labels differ (the Halide and NumPy/SciPy files say native because their replacement is idiomatic; section 4.2 relabels NumPy/SciPy as a workaround so its hot count is comparable), but the work is the same.
3. **The stability-analysis cascade (I50) and eigen diagnostics (I44).** No framework supplies non-Hermitian sparse eigensolvers, ARPACK, SVD pseudospectra or a GKS sweep. "Native" in the tables means the work stays in host SciPy. Many backends also lack a matrix-export mode, so the DSL must assemble operators from its own tables:
   - Devito, pystencils and GT4Py have no export mode;
   - JAX and PyTorch can export via `jacfwd`/`jacrev` for small n.
4. **Geometry setup (I02, I03).** No framework has ray casting or line-fragment merge-join primitives. Every option does this once in host NumPy or Numba, even where the label says native.
5. **A ragged strided-segment primitive (the literal form of H1, I16).** No array or stencil framework has one. Most decompose H1 the same way: a whole-grid interior stencil, an ELL closure overwrite, and an explicit coverage mask. That set covers:
   - JAX, PyTorch, MLX and the Array API;
   - Halide, Devito, pystencils, GT4Py and Triton;
   - DaCe's preferred form.
   Only explicit-loop models express the per-line form literally: Numba (native), Warp, Mojo, DaCe's per-line map, direct Metal and the PyKokkos TeamPolicy port.
6. **A usable sparse type for the cut-cell coupling (H2).** The options either lack one or should not be relied on for it:
   - Absent: MLX (#2728 wontfix), Halide, pystencils, Devito, the Array API.
   - Unmaintained: JAX.
   - Unusable on MPS: PyTorch (PR #196205 open, matvec not covered).
   - Opaque to fusion: `torch.sparse` on CPU.

   The portable form is the same everywhere: compact COO for B and padded ELL over a concatenated `[D; R]` source for Bf and Br. This also removes the full-height row-pointer sweep of `src/matrices/csr.hpp:47-60`.
7. **Partial-coverage write semantics (I27).** The C++ `eq_t` policy leaves rows that no block covers holding stale values (`src/matrices/block.hpp:174`, `op(b_ptr[out_idx], dot)` only for covered rows). Every functional backend has to make this explicit with `where(mask, new, old)` or a zero default, and no backend reproduces it implicitly. Bit-for-bit parity with C++ is therefore unlikely, and validation has to be tolerance based. The reference test `t-laplacian` is itself currently failing (`docs/CLEANUP_PLAN.md` section 0a).
8. **fp64 on an Apple GPU.** No framework provides it, and no framework offers a df64 dtype (section 2).

## 7. Shortlist

### 7.1 Reasoning

The user's constraints rank as follows:
1. No performance loss against the C++ code. In practice that is fp64 on CPU, at both small 2D and large 3D.
2. An Apple-GPU target as a nice-to-have.

Section 2 shows that the second constraint can only be met in fp32, and section 3 shows that no single framework meets both. The host measurement (section 2.1, `05-measurements.md` Section D) sharpens the first constraint: a fused synthetic fp64 step on the CPU from Python sits at the four-submit discretization bar at 51x51 (compiled MLX CPU 0.085 ms, eager NumPy 0.103 ms, torch eager 0.149 ms against 91 us) and within 1.4x of eight-thread Kokkos at 65^3, so the CPU-parity question is now "with real tables and a native bar", not "at all"; and the Apple GPU in fp32 loses at the shipped sizes (sync floor 167 to 181 us) and wins 4 to 6x at 129^3 and 257^3. The shortlist is therefore a set of roles under a DSL whose IR the project owns, not a single framework:
- **CPU emitter:** must reach whole-step compilation with ragged structure intact.
- **Reference oracle:** must be fp64, simple, and able to export the operator.
- **Apple-GPU emitter:** fp32, experimental, gated on a precision study.

The idiom analysis also says what that IR must contain, because every backend converged on the same small vocabulary:
- masked constant-tap axis stencils;
- padded ELL gather-dot with overwrite or accumulate;
- compact COO scatter;
- `where` masks for boundary edits and coverage;
- closed-form pointwise expressions with `t` as a scalar;
- packed-state axpby;
- device-side reductions read back every K steps.

All geometry-dependent structure is host-built table data.

### 7.2 Tier 1: prototype first

| Role | Candidate | Why | Main open question |
|---|---|---|---|
| CPU emitter (loop form) | Numba | Highest CPU parity score (9) and fewest non-native hot idioms (2). Ragged per-line loops port directly (H1 native), the whole step is one function, and the project is mature with regular releases. | Unmeasured: prange fork-join at 2.6k points, strided-axis SIMD at 3D, JIT and cache cost for generated code. |
| CPU/CUDA emitter (array form) | JAX | Whole-step `jit` plus `fori_loop` and donation give the cleanest H4. It is one of two options with local, synthetic measurements below the C++ small-2D figures (RHS 14 us against the measured 22.7 us; RK4 step 66 to 74 us against the measured 0.90 ms and the estimated fixed-up 0.15 ms). The other is Devito (7.2 us per fused step at 51x51, serial, synthetic, originally compared against an estimated bar; section 7.3). `jacfwd` gives operator export for I44/I50. CUDA comes free. | Fusion heuristics (barriers, scatter copies) may change across XLA versions. Local numbers must be repeated on real geometry against C++ on the same machine. |
| Reference oracle and table producer | NumPy/SciPy assembled CSR | Lowest risk (9). Every irregularity becomes matrix rows. It is the natural correctness oracle for the other backends and the stability-analysis engine. | Not a performance path at large 3D (estimated 3 to 8x slower). |
| Apple-GPU emitter (experimental, fp32) | Direct Metal via MLX `mx.fast.metal_kernel`, moving to a metal-cpp extension if needed | Best Apple-GPU score (8). The DSL can emit MSL for the same table-driven kernels. Unified memory makes table sharing cheap. | fp32 accuracy of small-psi closures, which decides viability for any Apple path. Submit latency at small 2D is now measured: 167 to 181 us per op plus sync, a whole-step-compiled fp32 step 0.35 to 0.5 ms at 51x51 against 0.085 ms on the CPU (`05-measurements.md` Section D), so the small-2D case needs batching of independent cases per dispatch. MLX in-place and replay limits. |

The first benchmark spike should run Numba and JAX on the real heat RHS and RK4 step. It should compare them with the C++ code on the same Mac at 51x51, 71x71, 64^3 and 256^3, with NumPy/SciPy as the correctness oracle. Every framework performance number in this survey that matters for the decision is currently an estimate or a synthetic local measurement. The C++ side now exists as a devcontainer measurement (section 2.1), but not yet as a native macOS build.

### 7.3 Tier 2: evaluate if Tier 1 misses the bar

| Candidate | Why it is in Tier 2 rather than Tier 1 |
|---|---|
| PyTorch | Already a locked dependency. Offers CPU, CUDA and MPS from one API, plus `torch.mps.compile_shader` for hand-written MSL. Tier 2 because small-2D CPU parity measured marginal (about 70 us fixed per compiled step, local), MPS has no graph replay, and Inductor-MPS has open correctness bugs. A reasonable alternative to direct Metal as the Apple emitter host. |
| Halide | Whole-step `Pipeline` fusion and geometry-independent pipelines fit the table-driven IR well. Tier 2 because schedules are manual, all irregular access must be gather form, JIT latency is unknown, and no measurements exist. |
| Devito | The only option with a measured fused multi-step RK4 loop including cut-cell tables (7.2 us per step at 51x51, local, serial, synthetic; the file compared it with an estimated C++ bar, now measured at 0.90 ms with an estimated fixed-up 0.15 ms). It is SymPy native. Tier 2 because the indirect-write atomics, the `implicit_dims` hoisting trap, `apply()` overhead and the missing matrix export all push work back into the DSL. |
| DaCe | Closest IR match. Tier 2 because the CPU claims are unverified, it is in a v2 alpha with GPU codegen churn, and compile latency would hurt parameter sweeps. |

Section 9 adds two tier-2 comparators that sit beside these: Pythran or Cython as alternative CPU emitters to Numba (9.6), and loopy as a generator for one fused step function (9.2).

### 7.4 Not shortlisted

| Option | Reason |
|---|---|
| MLX (as primary) | fp32-accuracy fp64 transcendentals (#3047) and elementwise-only fusion. The inferred single-core CPU fp64 blocker is softened by measurement: compiled MLX CPU fp64 was the fastest small-2D path (0.085 ms at 51x51) and 4.2 ms at 65^3 (`05-measurements.md` Section D). Usable as the host for the Metal custom kernels above and as one of the three array-arm backends in the phase-1 experiment. |
| Taichi | Slow-moving project (no release in 14 months), no cross-kernel fusion, Metal f32 only. |
| pystencils | Useful SymPy leaf-kernel generator, but no Apple path, no confirmed reductions, and a young 2.x API. Could be revisited as a C++ export route via pystencils-sfg. |
| Mojo/MAX | Plausible CPU codegen target, but single-vendor, still breaking syntax after 1.0, and the Apple custom-kernel path is undocumented. Worth a watch item, not a commitment. |
| GT4Py | Gather-only model (believed), experimental `next` API with repeated breakage, no Apple path. |
| Triton | No CPU backend; Apple support is out of tree. Only relevant if CUDA becomes a target. |
| Array API standard | Not an execution layer. Useful as a conformance vocabulary for the reference backend. |
| Warp | Serial CPU launches, no Apple path. Good design reference only. |
| PyKokkos | No graph replay, no releases, macOS build bugs, no Metal. Adds little over the existing C++ Kokkos code. |
| CuPy / cuPyNumeric | CUDA only; cuPyNumeric is end of life. |

## 8. Verification outcomes

An independent pass checked the load-bearing claims of each detailed file against current primary sources. "Refuted" counts claims shown wrong as stated. That includes claims that were stale, corrected, or understated, as in the Mojo row.

The original pass recorded 161 claims. That figure includes the 11 claims of the Apple fact sheet (`00-apple-gpu-facts.md`), so the 18 framework files account for 150. Two rows have been recounted from their file logs since: the Halide and GT4Py logs each describe claims as refuted that the original tally did not count (see the rows). Adding those three refuted claims gives 164 for the original 19 surveys. The Apple fact sheet's 11 are its original claims (7 confirmed, 0 refuted, 4 uncertain); cross-check rows added to its log in editing are marked there and are not counted here. The nine comparator files of section 9 (the eight new framework files plus `04-crosscutting-criteria.md`; CuPy stays among the original 18) were verified on 2026-10-12 by the same method and are tallied in a second table below.

| Survey | Confirmed | Refuted | Uncertain | Score change | Notable corrections |
|---|---|---|---|---|---|
| Apple GPU facts | 7 | 0 | 4 | rated 7.5 | 11 original claims; the log has since gained cross-check rows added in editing (Halide fp64 literal error, MLX #3047 and others), marked there and not in this tally. MLX date corrected to v0.32.3 on 29 Sep. The 96 GB/s single-thread bandwidth, AdaptiveCpp PR numbers and warpx-metal 13x are unverified. The earlier unverified 120 us round trip has been superseded as a load-bearing figure by the measured 167 to 181 us sync floor (`05-measurements.md` Section D). |
| JAX | 6 | 0 | 3 | 7 to 6.5 | jax-metal's "needs jax<=0.5.0" is a community workaround, not a declared pin. The jaxlib 0.11.2 FFI headers are narrower than stated. All CPU speed claims remain local and synthetic. |
| PyTorch | 5 | 0 | 2 | 6.5 unchanged | The 2.11 MPS "comprehensive" operator expansion is modest in substance. The CUDA-graph doc page and 2.12/2.13 details are unverified. |
| MLX | 7 | 0 | 2 | 4.5 unchanged | TF32 default (#3860) is closed as a documentation fix and also affects Apple silicon. Single-core CPU for built-in ops is inferred only. |
| Taichi | 6 | 0 | 3 | 5.5 to 5 | CC backend removal dates from v1.7.0 (Nov 2023), not 2025 packaging work. Two code sketches were invalid. "Maintenance mode" is an inference. |
| Devito | 3 | 0 | 3 | 6 to 5.5 | SYCL support is not in the open-source registry. Release cadence is irregular (about 5.5-month gap). 7 us versus 300 to 1000 us is measurement against estimate (the C++ step has since been measured at 0.90 ms, section 2.1). |
| pystencils | 6 | 0 | 4 | 5.5 to 5 | lbmpy 2.1 was yanked. Python >=3.12 applies from 2.1 only. Index-list kernels use `CreateKernelConfig.index_field` in 2.x, not the 1.x name. |
| Numba | 5 | 1 | 3 | 7 to 6.5 | `numba.pycc` is deprecated, not removed (refuted). Free-threading is experimental, not absent. |
| Warp | 7 | 1 | 1 | 3.5 unchanged | Latest is 1.18.0, not 1.17.0 (refuted). The 1.16.1 entry could not be found. External C++ module compilation now exists (experimental). |
| DaCe | 5 | 0 | 4 | 6 unchanged | No Metal confirmed as a negative result. Indirect-access frontend support, whole-program fusion and call overhead are unverified. |
| GT4Py | 5 | 1 | 2 | 4 unchanged | "Cartesian is maintenance-only" was refuted in the file log and dropped (now counted as refuted; the original tally showed 0). The gather-only model and API spellings are unverified. |
| Halide | 4 | 2 | 5 | 6.5 to 6.3 | The `hl.sum_taps` sketch API does not exist (refuted in the file log, fixed). The stated Python 3.10+ floor is wrong; PyPI says >=3.9 (corrected in the log). Both are now counted as refuted; the original tally showed 0. The Anderson2021 GPU autoscheduler dates from v16, not v21. Metal fp64 evidence is stronger than stated (codegen error). |
| PyKokkos | 5 | 0 | 2 | 3 unchanged | Threads/Serial backends downgraded to unverified. Kokkos-has-no-Metal is absence of evidence. |
| Mojo/MAX | 5 | 1 | 4 | 5 unchanged | The Qualcomm acquisition is confirmed by Modular's own sources (the survey had understated it; counted as refuted). Sketches use closure syntax that 26.6 changed. |
| Triton | 3 | 0 | 4 | 4 unchanged | triton-msl targets Triton 3.7.0, not 3.8 or 3.9. triton-cpu also lists GPU targets. |
| NumPy/SciPy | 2 | 1 | 4 | 5 unchanged | SciPy 1.18.1 exists, so 1.18.0 is not current (refuted). sparse_dot_mkl accepts BSR too. |
| Array API | 5 | 0 | 3 | 4 unchanged | `nextafter` is in the standard. `take` and integer indexing are more capable than claimed. |
| Direct Metal | 6 | 0 | 3 | 5 unchanged | metalcompute is 0.2.9 (Jan 2025), not 0.2.4. The #2547 workaround is "write an extension", not a custom primitive. |
| CuPy / cuPyNumeric | 7 | 0 | 2 | 3 unchanged | cuPyNumeric end of life is better supported than stated (direct NVIDIA docs quote). The int32-only sparse limit is narrower (spsolve/spilu only). ROCm has no sparse support. |
| **Total, original 19 surveys** | **99** | **7** | **58** | | 164 claims after the Halide and GT4Py recount (the original tally was 99 / 4 / 58 = 161, of which 11 are the Apple fact sheet and 150 the 18 frameworks) |

Comparator files (section 9), verified 2026-10-12. Counts are those recorded for each file.

| Comparator | Confirmed | Refuted | Uncertain | Score change | Notable corrections |
|---|---|---|---|---|---|
| PyFR | 7 | 0 | 2 | 4 | Nothing refuted. Paper details, issue counts, MPS use and the sketch syntax are unverified. The file's closing tally line says 6 confirmed, but its log has seven confirmed rows (one partly). |
| loopy | 4 | 1 | 3 | 4.5 | The repository is more active than stated (commit on 2026-09-27; the "unreadable last commit" claim was refuted). No Metal target found, which is absence of evidence. |
| MLIR: xDSL / IREE | 6 | 1 | 3 | 3.5 unchanged | Upstream MLIR wheel PRs #160388 and #177143 were not open: one was merged then reverted, the other closed unmerged (refuted). |
| wgpu-py / MoltenVK / tinygrad | 5 | 0 | 4 | 4 | MoltenVK reporting no shaderFloat64 is inferred; tinygrad METAL rejecting fp64 was not found in source. The WGSL sketch has a u32 underflow. |
| Julia | 6 | 0 | 4 | 4.5 unchanged | The no-Float64 hardware claim rests on prior knowledge. PoCL launch cost, the JuliaCon abstract and the arXiv figures are unverified. |
| Pythran / Cython / Codon | 6 | 2 | 3 | 6.5 to 6 | The Cython prange max-reduction inference claim and the sketch's correctness were refuted. Performance parity is unmeasured. |
| Apple-silicon CPU fp64 | 4 | 0 | 4 | 8 to 7 | The 56 GFLOPS per-core fp64 NEON rate is inferred from the fp32 figure. The claim that Accelerate routes to SME on M4 is unsupported by its cited papers (the file marks that citation refuted; the tally counts it uncertain). |
| Kokkos Apple backend | 3 | 0 | 3 | 1.5 | Absence of Kokkos issues or PRs, the 10x emulation cost and the effort comparisons are unverified. |
| Cross-cutting criteria (`04-crosscutting-criteria.md`) | 4 | 1 | 4 | 5 unchanged | metaljax does document fp64 as unsupported (the "undocumented" claim was refuted). Enzyme+Kokkos issue #1650 is still open. |
| **Comparator subtotal** | **45** | **5** | **30** | | 80 claims checked |
| **Grand total** | **144** | **12** | **88** | | 244 claims (164 original after the recount, plus 80 comparator claims) |


Patterns across the verification:

- **Ecosystem facts held up.** Versions, release dates, backend lists, fp64 status and issue states were confirmed in nearly every case. The few refutations in the original surveys were stale versions (Warp, SciPy), one wrong deprecation status (Numba), one understated fact (Mojo), a nonexistent sketch API and a wrong Python floor (Halide) and an unsupported maintenance claim (GT4Py). The comparator files add five: a commit-date claim (loopy), the status of two wheel PRs (MLIR), a reduction-inference claim and a sketch's correctness (Pythran/Cython/Codon), and metaljax fp64 being called undocumented when it is documented (cross-cutting).
- **Performance claims did not get verified anywhere.** Every CPU-parity and dispatch figure is either an estimate or a local synthetic measurement. That is why five overall scores were lowered (JAX, Taichi, Devito, pystencils, Numba) and one was trimmed (Halide). Among the comparators, Apple-silicon CPU (8 to 7) and Pythran/Cython/Codon (6.5 to 6) were lowered for the same reason. No score was raised.
- **Code sketches are the weakest artefacts.** At least four files had sketches the verifier found invalid or out of date:
  - Taichi: two invalid;
  - Halide: a nonexistent API;
  - Mojo: pre-26.6 syntax;
  - GT4Py: pseudocode.
  pystencils sketches use 1.x spellings. Among the comparators, the Pythran/Cython/Codon sketch has probable defects (a copying `ravel()`, int32 against int64 indices, `collapse(2)` on loops that are not perfectly nested) and the WGSL sketch has a u32 underflow. Treat every sketch in this part as the shape of the algorithm, not working code.
- **Absence claims are mostly absence of evidence.** These rest on documentation and release notes, not exhaustive tracker searches:
  - no Metal backend in Numba, pystencils, DaCe, GT4Py or Kokkos;
  - no graph API in `torch.mps`.
  They are consistent across sources but not proven.

## 9. Additional comparators and routes considered

A completeness review asked for comparators and routes that the 18-option survey did not cover. Each has its own file, written and verified on 2026-10-12 by the same method as sections 5.1 to 5.18: a survey, then an independent check of the load-bearing claims. They are not counted among the 18 options of section 1.1. Scores are the verifier-adjusted overall scores from each file, and the sub-scores that exist are in the comparator matrix of section 3. Every performance figure in these files is an estimate or a model unless stated otherwise; the measured C++ bar they should be read against is in section 2.1.

| Comparator | Verdict | Overall | Deciding fact, in short | Verification (C / R / U) | File |
|---|---|---|---|---|---|
| PyFR backends | reference only | 4 | Pointwise kernels with gather/scatter views, no neighbour-offset stencil, no third-party API; Metal is fp32 and re-encoded per run | 7 / 0 / 2 | [frameworks/pyfr.md](frameworks/pyfr.md) |
| loopy | tier2 | 4.5 | No Apple-GPU path; on CPU only as a generator of one fused step function | 4 / 1 / 3 | [frameworks/loopy.md](frameworks/loopy.md) |
| MLIR: xDSL, MLIR Python bindings, IREE | reference only | 3.5 | No maintained Apple-GPU path; owning an MLIR toolchain costs more than it returns | 6 / 1 / 3 | [frameworks/mlir_xdsl_iree.md](frameworks/mlir_xdsl_iree.md) |
| wgpu-py/WebGPU, Vulkan via MoltenVK (Kompute), tinygrad METAL | tier2 | 4 | None can hold fp64 on an Apple GPU; Python per-dispatch overhead loses at 51x51 | 5 / 0 / 4 | [frameworks/webgpu_vulkan_tinygrad.md](frameworks/webgpu_vulkan_tinygrad.md) |
| Julia (KernelAbstractions.jl + Metal.jl) | reference only | 4.5 | No Float64 on Metal; its CPU JIT would mean a second-language rewrite | 6 / 0 / 4 | [frameworks/julia.md](frameworks/julia.md) |
| Pythran, Cython, Codon | tier2 | 6 | No Apple-GPU path; CPU emitters only, Pythran for a NumPy-subset DSL, Cython as fallback | 6 / 2 / 3 | [frameworks/pythran_cython_codon.md](frameworks/pythran_cython_codon.md) |
| Apple-silicon CPU fp64 headroom (NEON, SME, AMX, Accelerate) | shortlist | 7 | Stencil sweeps are bandwidth- or L2-bound on NEON alone; SME and AMX add nothing | 4 / 0 / 4 | [frameworks/apple_cpu_simd.md](frameworks/apple_cpu_simd.md) |
| Contributing a Kokkos Apple backend (Metal or AdaptiveCpp) | excluded | 1.5 | No hardware fp64 even if built; no backend exists; AdaptiveCpp Metal is experimental | 3 / 0 / 3 | [frameworks/kokkos_apple_backend.md](frameworks/kokkos_apple_backend.md) |
| Cross-cutting criteria: differentiability, mixed precision, MPI | reference only | 5 | None decisive; differentiability is a modest tie-breaker for JAX | 4 / 1 / 4 | [04-crosscutting-criteria.md](04-crosscutting-criteria.md) |

### 9.1 PyFR backends, overall 4 (reference only)

**What it is.** PyFR 3.1 (13 Mar 2026, BSD-3, develop commits through 8 Oct 2026) generates per-backend kernels (OpenMP, CUDA, HIP, OpenCL, Metal) from PyFR-Mako pointwise templates plus GiMMiK/libxsmm constant-matrix multiplies. Its OpenMP graph replays a whole run list in one C call, so CPU fp64 parity at small sizes is plausible (an estimate only).

**Deciding fact.** `pyfr.backends` offers only pointwise kernels with gather/scatter views and constant-matrix multiplies. It has no structured-grid neighbour-offset stencil and no supported third-party API, and its Metal backend rejects fp64 (ValueError on double) and re-encodes every kernel through PyObjC on every graph run. It is an architecture to copy, not a substrate for SHOCCS.

**Copy.** One backend-neutral pointwise kernel template language with small per-backend generators; views as a first-class argument kind (host-built index tables giving gather in and scatter out, the right abstraction for the closure, B, Bf and Br ELL tables); graph objects replayed by the cheapest native mechanism per backend, including a C run-list executor for OpenMP; external scalars such as time and coordinates in pointwise kernels, which moves MMS and boundary evaluation onto the device; an explicit fp32-only flag per backend that fails loudly.

**Avoid.** Re-encoding Metal kernels from Python on every run; omitting a neighbour-offset stencil primitive; no fusion across kernels; hard-coded AoSoA layouts; an internal backend API with no stability contract.

**Verification.** 7 confirmed, 0 refuted, 2 uncertain (release, fp32-only Metal, krunner, Metal graph and `gimmik-max-nnz` confirmed). All performance figures in the file are unmeasured estimates; for the measured Apple GPU floor on this workload shape (167 to 181 us per op plus sync, 0.35 to 0.5 ms per compiled fp32 step at 51x51) see `05-measurements.md` Section D. Detailed file: [frameworks/pyfr.md](frameworks/pyfr.md).

### 9.2 loopy, overall 4.5 (tier2)

**What it is.** Kloeckner's polyhedral kernel generator (latest release 2025.2 of 2025-07-30, MIT, Python 3.10 or later; a commit on 2026-09-27). It emits C, OpenCL, CUDA and ISPC. Releases are slow and the maintainer base is tiny. Documented limits include convex loop domains only, and footprint analysis errors on data-dependent indices by default, so the ELL gather cannot be prefetched.

**Deciding fact.** No Metal target was found (absence not proven), and Apple's OpenCL route is deprecated with no fp64, so loopy gives no Apple-GPU path. On CPU it is viable only as a generator for one fused step function: its per-call Python dispatch overhead (unmeasured) is comparable to the whole C++ RHS at 51x51, which measures 22.7 us at 1 thread (section 2.1; the file itself quotes an older, higher estimate).

**Copy.** Separation of the algorithm (domains, instructions, dependencies) from the schedule, so one specification lowers to several targets; a first-class dtype per array for precision-generic kernels; per-instruction predicates for the interior mask; a C-family target class as the model for adding an MSL target.

**Avoid.** Per-kernel dispatch through a Python-level executor; convexity restrictions that push data-driven closure loops outside the analysis; a heavy dependency stack with single-maintainer exposure.

**Verification.** 4 confirmed, 1 refuted (the "last commit unreadable" claim), 3 uncertain. CPU parity is plausible for the affine stencil only if the whole RK step is one generated C function; this is not benchmarked. Detailed file: [frameworks/loopy.md](frameworks/loopy.md).

### 9.3 MLIR-based substrates: xDSL, MLIR Python bindings, IREE, overall 3.5 (reference only)

**What it is.** xDSL v0.73.0 (9 Oct 2026) has a stencil dialect that fits the constant-offset axis stencil, and f64 CPU lowering through LLVM is plausible. ELL closure rows, masks and packed fields need custom lowerings outside the dialect. IREE (nightly 3.13.0rc20261012) has a Metal HAL that goes SPIR-V to MSL via SPIRV-Cross, with one-shot command buffers only and no documented fp64. The MLIR Python bindings have no official PyPI release.

**Deciding fact.** There is no maintained Apple-GPU path for xDSL, and IREE's Metal driver has no documented fp64 and no command-buffer replay. An MLIR toolchain therefore adds heavy maintenance cost without delivering either an fp64 CPU speedup over a template emitter or the Apple GPU goal.

**Copy.** A typed stencil-apply op with constant offsets and explicit bounds, so interior and boundary regions are bounds rather than masks; a separate ELL op for irregular closure rows; the whole step as one function; dtype as a pass-level parameter so a df64 lowering can be added later.

**Avoid.** Owning an MLIR/LLVM toolchain or depending on third-party wheels; dialect APIs that change every release; assuming a GPU backend exists because the IR is portable.

**Verification.** 6 confirmed, 1 refuted (the upstream wheel PRs #160388 and #177143 were not open: one was merged then reverted, the other closed unmerged), 3 uncertain. Performance claims are unmeasured. Detailed file: [frameworks/mlir_xdsl_iree.md](frameworks/mlir_xdsl_iree.md).

### 9.4 Portable GPU APIs: wgpu-py/WebGPU, Vulkan via MoltenVK (Kompute), tinygrad METAL, overall 4 (tier2)

**What it is.** Three non-MLX, non-pyobjc routes to the Apple GPU. wgpu-py 0.32.0 (2026-07-19) and tinygrad 0.14.0 (2026-08-24) are active; Kompute's Python package `kp` is stale at 0.9.0 (2024-01-20); MoltenVK is a Vulkan 1.4 layer over Metal.

**Deciding fact.** No Apple-GPU path in any of the three can hold fp64. WGSL has no f64, wgpu's SHADER_F64 is native-only and Vulkan-only, Metal has no double type (so MoltenVK is not expected to supply Float64, which is inferred), and tinygrad METAL is expected to be fp32/fp16 only (unverified in source). With Python per-dispatch overhead at the launch-bound 51x51 case, none is expected to match the C++/Kokkos bar (an estimate, not measured). They are at most fp32 or double-float experimental paths for large 3D, where fp32 also breaks the 1e-10 test tolerances. The measured Apple GPU reference for that regime is `05-measurements.md` Section D: 1.0 ms per compiled fp32 step at 65^3 and 43.6 ms at 257^3 through MLX and torch MPS, 4 to 6x under the best CPU fp64 path, with a 167 to 181 us sync floor that rules out the 51x51 case.

**Copy.** Explicit command-encoder recording, with the whole RK step as one pre-recorded submission and static bind groups; TinyJit-style capture and replay with lazy fusion of pointwise chains; a feature-flag approach to precision that declares required features and fails loudly when fp64 is absent.

**Avoid.** Python per-dispatch overhead on the hot path; generic indexing for ELL closure rows; relying on portable-core features that lack f64, float atomics and subgroup ops; depending on a dormant package (`kp`).

**Verification.** 5 confirmed, 0 refuted, 4 uncertain. The WGSL sketch is unverified and has a u32 underflow. Detailed file: [frameworks/webgpu_vulkan_tinygrad.md](frameworks/webgpu_vulkan_tinygrad.md).

### 9.5 Julia (KernelAbstractions.jl + Metal.jl), overall 4.5 (reference only)

**What it is.** The obvious non-Python comparator for a single-source CPU and Metal kernel language. KernelAbstractions.jl v0.10.0 is a breaking release dated 12 Oct whose CPU backend is now OpenCL on PoCL; Metal.jl v1.11.1 is dated 18 Sep. On Apple GPUs it runs Float32, Float16 and BFloat16. No measurements were made.

**Deciding fact.** Metal.jl, like every Apple GPU stack, offers no Float64 kernels, so Julia gives no Apple GPU advantage for an fp64 solver. Its only unique value is a whole-step CPU JIT, which would require rewriting the core in a second language and abandoning the Python goal, while KA's CPU backend has just been rebuilt on OpenCL/PoCL. The same JIT benefit is available from Numba or a generated C/C++ path.

**Copy.** The `@kernel` plus `ndrange` plus `get_backend` single-source model; tap counts and axis steps as compile-time constants (`Val{N}`) so stencils unroll; a precision-generic element type so one kernel runs Float64 on CPU and Float32 or a df64 struct on Metal; broadcast fusion for RK slot arithmetic.

**Avoid.** Making the CPU path depend on a portability layer's launch machinery; silent Float64-to-Float32 conversion on upload (a DSL should reject it or require an explicit precision policy); relying on cross-launch fusion or replay that the framework does not provide.

**Verification.** 6 confirmed, 0 refuted, 4 uncertain. The no-Float64 hardware claim rests on prior knowledge because the cited docs do not state it. Detailed file: [frameworks/julia.md](frameworks/julia.md).

### 9.6 Pythran, Cython and Codon, overall 6 (tier2)

**What it is.** Low-risk CPU emitters in the Numba tier: Pythran 0.19.0, Cython 3.3.0 and Codon v0.20.3, all actively released. Each compiles a Python subset or typed code to fast fp64 CPU loops. Expected near parity with the C++/Kokkos host path is an unmeasured estimate.

**Deciding fact.** None of the three has an Apple-GPU path (Codon's GPU backend is NVIDIA-only), so they qualify only as CPU emitters. Pythran fits a DSL that emits NumPy-subset source best, and Cython is the safe fallback.

**Copy.** Loop-form, constant-folded kernels with per-line [lo, hi) ranges instead of boolean masks; the whole RK step exported as one native function; fixed-width ELL arrays with unique scatter targets; hashing generated source and caching built modules in the DSL.

**Avoid.** NumPy fancy indexing and whole-array temporaries in hot paths; depending on one C++ toolchain on macOS; expecting any GPU path from these tools.

**Verification.** 6 confirmed, 2 refuted (the Cython prange max-reduction inference claim, and the sketch's correctness), 3 uncertain. The score was lowered from 6.5 to 6 for unmeasured performance and sketch defects. The sketch needs a compile-and-run check. Detailed file: [frameworks/pythran_cython_codon.md](frameworks/pythran_cython_codon.md).

### 9.7 Apple-silicon CPU fp64 headroom, overall 7 (shortlist)

**What it is.** Hardware, not a framework: the realistic target and baseline for the Python port. Apple-silicon CPU fp64 is native IEEE through NEON (2 fp64 lanes, 4 FMA pipes). M4 SME has two units, the P-cluster unit shared by about four threads, with fp64 FMOPA near 503 GFLOPS (P), a quarter of fp32; that suits dense GEMM, not banded stencils. Apple clang lacks OpenMP, so a port should ship its own thread pool.

**Deciding fact.** fp64 stencil sweeps at 0.8 to 1.25 flop/B are bandwidth-bound (256^3) or L2-bound (64^3) on NEON alone, so SME and AMX offer nothing. The whole win over Kokkos comes from fusion, layout and launch elimination, which any LLVM-based compiled path delivers. The file models 30 to 60 us per RHS at 64^3 and 4 to 5 ms at 256^3 on a base M4, roughly 3x to 20x faster than the Kokkos host path. These are unmeasured models; for reference, the measured devcontainer C++ RHS at 65^3 is 798 us at 8 threads (section 2.1), and the model has not been tested against it on native macOS.

**Copy.** A precision-generic IR whose CPU backend emits fused loop nests with NEON; whole-step fusion that folds RK slot arithmetic, scale, source and boundary conditions into the stencil sweep; a persistent thread pool with spin barriers sized to the P-core count; z-slab tiling; interior-span splitting so masks are evaluated only near fragment ends; the ELL closure applied while the slab is cache-hot.

**Avoid.** Relying on SME/SSVE or AMX; per-axis Accelerate/vDSP calls; per-op eager dispatch; a Python-visible OpenMP dependence; treating rated DRAM bandwidth as CPU-reachable on Pro/Max; hard-coding P-core counts.

**Verification.** 4 confirmed, 0 refuted, 4 uncertain. The 56 GFLOPS per-core fp64 NEON rate is inferred from the fp32 figure, not measured, and the claim that Accelerate routes to SME on M4 is unsupported by its cited papers. The score was lowered from 8 to 7. Detailed file: [frameworks/apple_cpu_simd.md](frameworks/apple_cpu_simd.md).

### 9.8 Contributing a Kokkos Apple backend, overall 1.5 (excluded)

**What it is.** The alternative to owning an MSL layer: add a Metal backend to Kokkos, or reach Metal through SYCL via AdaptiveCpp. As of Kokkos v5.2.2 (tagged 2026-09-10) there is no Kokkos Metal or Apple-GPU backend, and the Kokkos requirements page lists only IntelLLVM as a SYCL compiler. AdaptiveCpp's Metal backend is experimental, tested only on macOS 26, has no double (emulation only planned), and commits regular SYCL events as separate command buffers.

**Deciding fact.** Even a perfectly built Kokkos Metal backend would have no hardware fp64, and SHOCCS is fp64 (`real = double`). The backend also does not exist, and the only SYCL route is experimental and not a Kokkos-supported compiler. A native backend would be a multi-year compiler project.

**Copy.** Kokkos's separation of execution space, memory space and layout; the Graph abstraction, with the whole RK step as a recorded DAG; the reducer interface with an explicit identity.

**Avoid.** Generic functor-to-device compilation and the general execution model (a purpose-built DSL needs only a handful of kernel shapes); per-event command buffers; hard-coding `real = double` instead of making precision an IR parameter.

**Verification.** 3 confirmed, 0 refuted, 3 uncertain. The absence of Kokkos issues or PRs, the 10x emulation cost and the effort comparisons are unverified. Detailed file: [frameworks/kokkos_apple_backend.md](frameworks/kokkos_apple_backend.md).

### 9.9 Cross-cutting criteria: differentiability, mixed precision, distributed memory, overall 5 (reference only)

**What it is.** A scoring of three criteria the survey did not weigh. JAX 0.11.2 is the only option with whole-step reverse mode through gather and scatter on CPU fp64 today (jit, scan, checkpoint). Parameter Bayesian optimisation acts on small eigenvalue problems that SymPy or JAX already handle at setup time. metaljax documents fp64 as unsupported (with an optional fp32 downcast), so any differentiable Apple run is fp32 or df64 with closure rows kept in fp64 on the host. fp32 storage is rejected for convergence-grade runs because fp32 RK4 accumulation floors accuracy near 1e-7 (an unmeasured inference). MPI is unnecessary at the group's current sizes, with Kokkos+MPI as the escape hatch. Enzyme+Kokkos is not an established path (issue #1650 is still open).

**Deciding fact.** None of the three criteria is decisive. Only differentiability separates options, JAX leads it, and it is a nice-to-have. fp32 storage is rejected on accuracy grounds, MPI is not needed, and no Apple GPU path offers fp64.

**Copy.** Keep all irregularity in setup-time tables, so the adjoint of each runtime kernel is a transposition rule (a reversed-coefficient stencil; scatter-add over the same ELL table); declare each kernel with an explicit VJP rule and a per-array dtype policy (storage dtype, accumulation dtype, closure rows always fp64); keep layouts decomposition-ready (halo width per axis, an ownership map for R segments); support checkpointing at the step boundary.

**Avoid.** Promising fp32 storage as a default; building a distributed runtime or a general AD engine in the first version; Devito-style generate-everything MPI for the irregular rows.

**Verification.** 4 confirmed, 1 refuted (metaljax fp64 called undocumented when it is documented), 4 uncertain. DaCe AD is backed by arXiv 2509.02197, but its stable-release status is unknown. Detailed file: [04-crosscutting-criteria.md](04-crosscutting-criteria.md).

### 9.10 What the comparators change

- **The shortlist stands.** No comparator offers fp64 on an Apple GPU, and none beats the Tier 1 CPU candidates on evidence. Pythran and Cython are credible alternatives to Numba as the CPU emitter (tier 2), and loopy could generate one fused step function.
- **The CPU path has headroom on paper.** The Apple-silicon CPU file models large gains from fusion, layout and launch elimination alone. This is unmeasured and needs a native macOS run.
- **The design lessons converge.** Across PyFR, MLIR, the portable GPU APIs, Julia and Kokkos the same advice recurs: record the whole step once and replay it with at most one wait; keep irregular closure rows in a separate ELL op or view rather than forcing them into the stencil abstraction; make precision an IR parameter; fail loudly when fp64 is requested on a backend that lacks it.

## 10. Sources

Per-framework sources are listed in each detailed file. The primary sources cited directly in this part:

- Apple GPU and Kokkos: https://adaptivecpp.github.io/AdaptiveCpp/install-metal/ ; https://kokkos.org/kokkos-core-wiki/get-started/requirements.html ; https://github.com/pytorch/pytorch/pull/196205 ; https://developer.apple.com/forums/thread/797778 ; https://developer.apple.com/forums/thread/767853 (unverified) ; https://github.com/Lulzx/warpx-metal (unverified)
- JAX: https://docs.jax.dev/en/latest/changelog.html ; https://github.com/jax-ml/jax/issues/26968 ; https://pypi.org/project/jax-metal/ ; https://pypi.org/project/jax-mps/ ; https://docs.jax.dev/en/latest/jax.experimental.sparse.html
- PyTorch: https://dev-discuss.pytorch.org/t/pytorch-2-14-1-general-availability/3455 ; https://docs.pytorch.org/docs/2.14/mps.html ; https://github.com/pytorch/pytorch/issues/169738 ; https://discuss.pytorch.org/t/float64-tensors-on-mps/224325
- MLX: https://github.com/ml-explore/mlx/releases ; https://github.com/ml-explore/mlx/issues/1905 ; https://github.com/ml-explore/mlx/issues/2185 ; https://github.com/ml-explore/mlx/issues/3047 ; https://github.com/ml-explore/mlx/issues/2728 ; https://github.com/ml-explore/mlx/issues/2547 ; https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/compile.cpp
- Taichi: https://pypi.org/project/taichi/ ; https://docs.taichi-lang.org/docs/type ; https://github.com/taichi-dev/taichi/releases
- Devito: https://github.com/devitocodes/devito/releases ; https://github.com/devitocodes/devito/blob/main/devito/arch/archinfo.py
- pystencils: https://pypi.org/pypi/pystencils/json ; https://pycodegen.pages.i10git.cs.fau.de/docs/pystencils/2.0/api/generated/pystencils.codegen.config.CreateKernelConfig.index_field.html ; https://pycodegen.pages.i10git.cs.fau.de/pystencils-sfg/
- Numba: https://numba.readthedocs.io/en/stable/release/0.67.0-notes.html ; https://numba.readthedocs.io/en/stable/reference/deprecation.html ; https://pypi.org/project/numba-cuda/
- Warp: https://nvidia.github.io/warp/latest/project/changelog.html ; https://nvidia.github.io/warp/latest/user_guide/faq.html ; https://nvidia.github.io/warp/latest/user_guide/compatibility.html
- DaCe: https://github.com/spcl/dace/releases ; https://spcldace.readthedocs.io/en/latest/optimization/gpu.html
- GT4Py: https://pypi.org/project/gt4py/ ; https://github.com/GridTools/gt4py/releases ; https://gmd.copernicus.org/articles/19/713/2026/
- Halide: https://github.com/halide/Halide/releases ; https://pypi.org/pypi/halide/json ; https://github.com/halide/Halide/blob/main/src/CodeGen_Metal_Dev.cpp
- PyKokkos: https://github.com/kokkos/pykokkos ; https://kokkos.org/pykokkos/policies.html ; https://kokkos.org/pykokkos/fusion.html
- Mojo/MAX: https://max.modular.com/changelog ; https://max.modular.com/releases/v26.6/ ; https://forum.modular.com/t/max-models-can-now-run-on-apple-silicon-gpus/3283 ; https://forum.modular.com/t/qualcomm-to-acquire-modular/3271
- Triton: https://github.com/triton-lang/triton/releases ; https://github.com/triton-lang/triton-ext/pull/126 ; https://github.com/Janssena/triton-msl ; https://github.com/triton-lang/triton-cpu
- NumPy/SciPy: https://docs.scipy.org/doc/scipy/release/1.18.0-notes.html ; https://discuss.scientific-python.org/t/ann-scipy-1-18-1-release/2554
- Array API: https://data-apis.org/array-api/latest/changelog.html ; https://data-apis.org/array-api-compat/changelog.html ; https://github.com/scipy/scipy/issues/20700
- Direct Metal: https://pypi.org/pypi/metalcompute/json ; https://pypi.org/pypi/pyobjc-framework-Metal/json ; https://docs.pytorch.org/docs/main/mps.html
- Comparators (section 9): sources are listed in each comparator file and in `04-crosscutting-criteria.md`.
- CuPy / cuPyNumeric: https://docs.cupy.dev/en/latest/install.html ; https://docs.cupy.dev/en/latest/user_guide/kernel.html ; https://newreleases.io/project/pypi/cupy/release/14.2.0 ; https://docs.nvidia.com/cupynumeric/26.06/user/differences.html
- SHOCCS code referenced: `src/matrices/block.hpp`, `src/matrices/inner_block_meta.hpp`, `src/matrices/csr.hpp`, `src/matrices/csr.cpp`, `src/systems/heat.cpp`, `src/systems/system.cpp`, `src/systems/detail/scalar_system_utils.hpp`; C++ measurements `05-measurements.md` (2026-10-12, devcontainer) and the older baseline `benchmarks/baselines/baseline-20260320-024030.json`.
