# DaCe (spcl/dace) for SHOCCS

## Verdict

| Item | Assessment |
|---|---|
| Fit for SHOCCS | Best IR match among surveyed options for CPU work (maps, indirect memlets, dynamic ranges, wcr). Does not deliver the MPS goal. |
| CPU parity | Plausible parity or better via whole-step fusion. Unmeasured, treat as a hypothesis. |
| Apple GPU | None. No Metal, OpenCL or SYCL target in docs or release notes (confirmed as a negative result). |
| fp64 | Native on CPU and CUDA/HIP (not contradicted by sources). Apple GPUs lack hardware fp64, so any Metal path would be fp32. |
| Risk | Moderate: small research community, v2 alpha with GPU codegen being rewritten, C++ toolchain and compile latency at run time. |

## Status as of 2026-10-12

- Latest release seen: v2.0.0-alpha10 (pre-release, published 3 Oct; the releases page shows no year, so 2026 is inferred).
- Release notes include "New GPU Codegen Prep" items (alpha9 and alpha10) and "[GPU Codegen] Dynamic shared memory support", so the GPU codegen is under active change.
- bfloat16 and fp8 (e4m3fn/e5m2) types were added in alpha6 (14 Aug).
- FPGA code generation moved to a separate repository in the v2 alphas (alpha2 notes). The repo name "dace-fpga" was not checked.
- The latest stable 1.x version and the PyPI latest are unverified (the PyPI page did not load).
- Project origin: SPCL, ETH Zurich. Claims about ICON/GT4Py-adjacent use are vague and unsourced, so they are not relied on here.

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| CPU (C++/OpenMP via cmake) | Supported (README) | Native | Apple Silicon CPU needs an OpenMP-capable compiler (Homebrew libomp or gcc); build not verified. |
| NVIDIA CUDA | Supported (GPU docs) | Native | GPU_Device and GPU_ThreadBlock map schedules. |
| AMD HIP | Supported (GPU docs; selected through the DACE_compiler_cuda_backend option) | Native | Same flow as CUDA. Whether one class (CUDACodeGen) serves both and how targets register with TargetDispatcher is unverified. |
| GPU_ThreadBlock_Dynamic | Documented for uneven per-block work such as graph and sparse | n/a | Confirmed in GPU docs. |
| FPGA | Separate repo | n/a | Not relevant. |
| MPI codegen (dace.codegen.targets.mpi) | Unverified | n/a | Not relevant to the goal. |
| Metal / MPS | Not supported | Apple GPU: no hardware fp64 | Needs a new MSL code generator, Metal host runtime, new schedule/storage enums and xcrun toolchain support. Effort estimate (several engineer-months) is the surveyor's guess, not sourced. |

DaCe has no MPS interop: it does not call MPS kernels or PyTorch-MPS tensors.

## Idiom mapping

All performance expectations are unmeasured hypotheses unless noted. "Setup" means host-side work outside the timed path.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | Dict or tuple of dace arrays with symbolic sizes (nx, ny, nz, nR*), or one packed array with static offsets so slot ops become one map. | Equal to C++. |
| I02 | workaround | Ray casting on host in NumPy/Numba; pass SoA arrays (psi, solid_ic) to the SDFG. | Setup only. |
| I03 | workaround | Host NumPy merge-join emitting line start, length, stride, closure ids. | Setup only. |
| I04 | native | Host builds masks or int32 index lists; the SDFG consumes a 1-D index array in a map. | Equal to C++ (list); mask form also possible. |
| I05 | native | Host Python/SymPy; interior taps become constants or symbols, tap loops unrolled at compile time. | Setup only. |
| I06 | workaround | Evaluate on host (lambdify/CSE) in float64, or write the CSE output as a dace map. Prefer host. | Setup only. |
| I07 | workaround | Host Python/Numba with searchsorted; emit padded ELL tables. | Setup only. |
| I08 | workaround | Host NumPy grouping by (left_kind, right_kind, bc); emit closure table and ELL row table. | Setup only. |
| I09 | workaround | Host NumPy batched over R points; emit ELL tables. | Setup only. |
| I10 | native | scipy.sparse or np.lexsort on host; compact COO or ELL needs no row pointer. | Setup only. |
| I11 | native | np.cumsum and np.concatenate on host with closure-table dedup; SDFG takes flat coefficient and index arrays. | Setup only. |
| I12 | native | Plain Python replaces Lua config; NumPy state arrays. | Setup only. |
| I13 | native | 3-D C-order array A[nx,ny,nz] matching the C++ layout; maps over (i,j,k). MapInterchange and MapCollapse for loop order. | Equal to C++. |
| I14 | native | Convention only: R arrays are 1-D indexed by R ordinal, kept by host table builders. | No cost. |
| I15 | native | Explicit-size data containers; work buffers as transients that the compiler can fuse or remove. | Equal or better (hypothesis). |
| I16 | workaround | Decompose: full-grid interior stencil map plus a map over a closure ELL table that overwrites rows plus a mask for uncovered points. Alternative: per-line map with dynamic-range inner maps (sketch 1). | Decomposed form possibly 1.0 to 2x faster than C++ on CPU; per-line form about equal. Needs measurement. |
| I17 | native | One map per axis with three-tap expression, or slices. Memlets expose access pattern for fusion. | Equal to C++ (memory bound). |
| I18 | native | Small fixed maps over first/last R planes with constant-coefficient tasklets. | Equal to C++. |
| I19 | workaround | ELL gather-dot-scatter: map over rows, inner sum over Tmax columns. | Equal to C++. |
| I20 | native | CSR SpMV with dynamic-range inner map and wcr sum (canonical sample, samples/simple/spmv.py cited from memory; unverified). Prefer COO or ELL. | Equal to C++ on CPU. CUDA needs GPU_ThreadBlock_Dynamic or a sequential inner loop. |
| I21 | native | Compact COO map over nnz with wcr add (sketch 2). Removes the full-height row pointer. | Possibly faster than the C++ full-height sweep at large sizes. |
| I22 | native | Map over Neumann-face planes; compile out via Python-level `if` when absent. | Zero cost when absent. |
| I23 | native | ELL gather with indirect indexing u[idx[r,k]]. | Equal to C++. |
| I24 | native | Same ELL form over R, or merge into the I23 table over concatenated [D;R]. | Equal; merging removes one dependent kernel. |
| I25 | native | Not needed: initialize the output in the same map or let MapFusion merge zero-fill with first accumulate. | Possibly saves a pass. |
| I26 | native | Elementwise scale or fold into operator tasklets. | Equal or free when fused. |
| I27 | native | One map summing three directional stencils, then R-point updates; wcr where rows collide. | Possibly better than C++ (one read of u). |
| I28 | native | One map producing three outputs or a stacked (3,nx,ny,nz) array. | Equal; better if fused into consumer. |
| I29 | native | Elementwise tasklet gx*dux + gy*duy + gz*duz, fused with I28 by MapFusion. | Equal, better when fused. |
| I30 | native | Masked elementwise map or index-list map. | Equal to slightly better. |
| I31 | native | Slice assign u[0,:,:] = val, or index-list assign for object sets. | Equal to C++. |
| I32 | native | Whole RHS or RK4 step as one @dace.program compiled to one C++ library; no per-node launch cost on CPU. Single-library compilation is plausible from the design but not verified against docs. | Hypothesis: better than Kokkos graph on host. |
| I33 | native | Python function calling the @dace.program; no eager path needed. | Per-call overhead unmeasured; the 5 to 50 us figure and fast_call are unsourced estimates. |
| I34 | native | Elementwise map over the grid with coordinate vectors and math tasklets; t as scalar argument. | Likely faster than C++ virtual calls. |
| I35 | unsupported | A per-point Lua/Python callback cannot run inside a compiled SDFG; replace with a SymPy closed form. | N/A. |
| I36 | native | Fold source evaluation into the compiled RHS with t as scalar (I34). | Avoids host round trip. |
| I37 | native | Whole RK4 step in one program with stage arrays as transients; fused axpby maps (sketch 4). | Hypothesis: fewer passes than ~9 slot passes. |
| I38 | native | Swap array references or use transients; copy unnecessary. | Free. |
| I39 | native | Evaluate exact solution only on face/R sets via index-list maps or slice assign inside the step. | Possibly better than C++. |
| I40 | native | Hoist to host setup; index arrays are SDFG inputs. | Free. |
| I41 | native | Reduction with wcr max/min; arg-max needs a tuple wcr or two passes; return a small stats array. | Equal on CPU; CUDA readback sync is the cost. |
| I42 | native | Host NumPy writes; compute the error field only when a dump is due (Python branch around the call). | I/O bound. |
| I43 | native | Host Python loop calling the compiled step; sub-stepping inside a program also possible. | Equal at per-step granularity. |
| I44 | workaround | Host scipy.sparse and eig from the same tables; DaCe has no eigensolver. | N/A, diagnostic. |
| I45 | native | NumPy one-liners as test oracles. | N/A. |
| I46 | native | Python-frontend array expressions lowered to maps; MapFusion fuses chains. | Equal (memory bound). |
| I47 | native | Elementwise physics tasklets plus batched derivative maps over a component axis (5,nx,ny,nz); sqrt/pow map to libm. Fusing flux into derivative input avoids large temporaries. | Hypothesis: equal to hand C++. |
| I48 | workaround | Keep NumPy/SciPy; not a DaCe job. | N/A. |
| I49 | workaround | Keep SymPy on host; retarget the printer to emit a C++ tasklet body so existing CSE output is reusable. | Setup only. |
| I50 | workaround | Host scipy with a matrix-export function built from the same tables. | N/A. |
| I51 | native | In-process call into the compiled SDFG (loaded via ctypes) removes the subprocess-per-point loop; use symbols for sizes to avoid recompiles. | Likely much faster than the subprocess flow; compile cost per new symbol set remains. |

## Code sketches

These are untested sketches from the survey, not validated DaCe programs. The Python frontend's handling of dynamic ranges and the `tabs: ...` placeholder in sketch 4 would need real work.

Sketch 1, per-line block matvec with strides and per-line dense closures (I16):

```python
@dace.program
def block_matvec(u: dace.float64[ND], du: dace.float64[ND],
                 start: dace.int32[NL], n: dace.int32[NL], stride: dace.int32[NL],
                 lcl: dace.int32[NL], rcl: dace.int32[NL],
                 CL: dace.float64[NC, R, T], CR: dace.float64[NC, R, T],
                 c: dace.float64[W]):
    for l in dace.map[0:NL]:                     # one line per map iteration (OpenMP)
        s = stride[l]
        for r in dace.map[0:R]:                  # left closure rows
            acc = 0.0
            for t in range(T):
                acc += CL[lcl[l], r, t] * u[start[l] + t*s]
            du[start[l] + r*s] = acc
        for r in dace.map[R:n[l]-R]:             # dynamic-range interior
            du[start[l] + r*s] = (c[0]*u[start[l]+(r-1)*s]
                                  + c[1]*u[start[l]+r*s] + c[2]*u[start[l]+(r+1)*s])
        for r in dace.map[0:R]:                  # right closure rows
            acc = 0.0
            for t in range(T):
                acc += CR[rcl[l], r, t] * u[start[l] + (n[l]-T+t)*s]
            du[start[l] + (n[l]-R+r)*s] = acc
# Faster alternative: full-grid 3D stencil map + ELL closure overwrite.
```

Sketch 2, cut-cell coupling to R points, ELL gather and COO scatter (I19, I21, I23):

```python
@dace.program
def cut_coupling(u: dace.float64[ND], uR: dace.float64[NR], du: dace.float64[ND],
                 duR: dace.float64[NR], rowD: dace.int32[NB], colR: dace.int32[NB],
                 wB: dace.float64[NB], idx: dace.int32[NR, K], W: dace.float64[NR, K],
                 selfc: dace.float64[NR]):
    for n in dace.map[0:NB]:
        du[rowD[n]] += wB[n] * uR[colR[n]]       # indirect read + indirect write (wcr add)
    for r in dace.map[0:NR]:
        acc = selfc[r] * uR[r]                   # Br diagonal
        for k in range(K):
            acc += W[r, k] * u[idx[r, k]]        # padded entries: W=0, idx=0
        duR[r] += acc
```

Sketch 3, gather/scatter boundary application (I04, I30, I31, I39):

```python
@dace.program
def apply_bc(du: dace.float64[NX, NY, NZ], duR: dace.float64[NR],
             src: dace.float64[NX, NY, NZ], fluid: dace.int32[NF],
             dirR: dace.int32[NDR], u: dace.float64[NX, NY, NZ],
             solface: dace.float64[NY, NZ]):
    duf = np.reshape(du, [NX*NY*NZ]); srcf = np.reshape(src, [NX*NY*NZ])
    for m in dace.map[0:NF]:
        duf[fluid[m]] += srcf[fluid[m]]          # unique indices => no atomics
    for m in dace.map[0:NDR]:
        duR[dirR[m]] = 0.0
    du[0, :, :] = 0.0
    u[0, :, :] = solface
# mask form: for i,j,k in dace.map[...]: du[i,j,k] += fm[i,j,k]*src[i,j,k]
```

Sketch 4, whole RK4 step with multi-buffer slot updates (I32, I37, I38):

```python
@dace.program
def rk4_step(uD: dace.float64[ND], uRx: dace.float64[NRX], uRy: dace.float64[NRY],
             t: dace.float64, dt: dace.float64, tabs: ...):   # placeholder in the survey
    kD = np.empty_like(uD); kx = np.empty_like(uRx); ky = np.empty_like(uRy)
    aD = np.zeros_like(uD); ax = np.zeros_like(uRx); ay = np.zeros_like(uRy)
    sD = uD.copy(); sx = uRx.copy(); sy = uRy.copy()
    for s in range(4):
        update_bc(sD, sx, sy, t + cdt[s]*dt)
        rhs(sD, sx, sy, kD, kx, ky, t + cdt[s]*dt)
        aD += wgt[s]*dt*kD; ax += wgt[s]*dt*kx; ay += wgt[s]*dt*ky
        if s < 3:
            sD[:] = uD + cdt[s+1]*dt*kD
            sx[:] = uRx + cdt[s+1]*dt*kx; sy[:] = uRy + cdt[s+1]*dt*ky
    uD += aD; uRx += ax; uRy += ay
# Packing D|Rx|Ry into one array makes each axpby a single map.
```

## Performance assessment

None of the following was measured. No local benchmark exists for DaCe, and the C++ baseline numbers cited by the survey (13.7 ns/pt at 64^3 for the per-line block kernel, about 0.7 us Kokkos launch floor, about 2 ns/pt bandwidth estimate) come from the survey and were not re-checked.

- Small 2D (2.6k to 5k points): parity or better is possible only if the whole RK4 step is one program, which would remove roughly 280 launches per step. Risks: OpenMP fork/join on tiny maps (maps default to parallel schedules; small ones would need Sequential), and Python call overhead per step. Confidence: medium to low.
- Large 3D: bandwidth bound, so parity is expected if the host compiler vectorizes the generated loops. A fused three-axis stencil could beat the one-team-per-line block kernel. Stride-nx axes may need MapInterchange or MapTiling applied by hand; the docs describe the auto-optimizer as experimental (from memory, unverified).
- Dispatch overhead: zero per op inside a compiled SDFG is a design expectation. Per-call Python overhead (estimated 5 to 50 us, fast_call mitigation) is unsourced.
- Compile time and warmup: each new program or symbol set triggers parse, codegen, cmake and a C++ build, estimated 5 to 60 s, cached in .dacecache. Not measured. Precompute large CSE blocks on the host to avoid inflating generated code.
- Fusion: whole-RHS fusion via MapFusion and transient removal is a stated property of the SDFG design; not verified here against the docs.

## Suitability as a DSL substrate

Strong on CPU. The SDFG level (maps, memlets with indirect subsets, dynamic ranges, wcr, nested SDFGs, transformations) is the level a custom DSL would lower to. A DSL could build SDFGs through the Python API (dace.SDFG, add_map, add_tasklet, add_memlet_path), bypassing the AST frontend, whose corner cases are the main ergonomic risk. Generated C++ is inspectable, can embed existing CSE coefficient code as C++ tasklets, and is described as callable from C++ drivers (README, "integration with C/C++"). Interop claims about zero-copy NumPy and PyTorch tensor arguments are from memory and unverified.

Weaknesses: no Apple GPU, run-time C++ toolchain dependency, high compile latency (a problem for the Bayesian-optimization sweeps unless sizes are symbolic and cached), small community, and irregular GPU performance needing manual scheduling. Verdict: a good fit for "no CPU loss, full control of loops", a poor fit for MPS. An Apple GPU path would most plausibly come from a different framework (MLX or torch-MPS) fed the same ELL/COO table formulation.

## Risks

- No Metal/MPS backend and no sign of one; building one is a multi-month compiler project against a codegen being rewritten in the v2 alphas.
- v2.0.0-alpha10 is a pre-release; choosing between pinning 1.x and moving to 2.x is unresolved, and the stable 1.x version is unverified.
- Python-frontend gaps with complex indexing; safest path is programmatic SDFG construction at the cost of ergonomics.
- Compile latency and a required C++ compiler and cmake at run time.
- OpenMP overhead on tiny maps could lose to Kokkos at 2.6k to 5k points; unmeasured.
- Bus factor: research group project with limited long-term guarantees.
- Apple Silicon CPU builds need an OpenMP-capable compiler; not verified.
- Several figures (call overhead, auto-optimizer quality, tensor interop) are unsourced and must be verified before decisions rest on them.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| No Metal/Apple-GPU generator; targets are CPU, CUDA, HIP, FPGA | Confirmed (negative result) | GPU docs cover only CUDA and HIP; release notes never mention Metal; web search found only unrelated projects. DaCe issue tracker not searched. |
| Latest release v2.0.0-alpha10 (3 Oct) with "New GPU Codegen Prep" | Confirmed, year inferred | Releases page lists alpha10 and alpha9 GPU codegen items; no year shown. Stable 1.x and PyPI latest unverified. |
| bfloat16 and fp8 added in alpha6 | Confirmed | alpha6 (14 Aug) notes, #2455. |
| FPGA in a separate repo | Confirmed | alpha2 notes on moving FPGA codegen; repo name "dace-fpga" not checked. |
| GPU_ThreadBlock_Dynamic exists for irregular work | Confirmed | GPU optimization docs. |
| One CUDACodeGen serves CUDA and HIP; TargetDispatcher registration | Uncertain | Docs confirm HIP is used like CUDA via DACE_compiler_cuda_backend; class names and dispatcher not seen. |
| Indirect access, dynamic map ranges, wcr work in Python frontend (SpMV) | Uncertain | SpMV sample and frontend docs not fetched. |
| Whole-program single C++ library, whole-RHS fusion, 5 to 50 us call overhead, fast_call, compile seconds to minutes | Uncertain | No source fetched; surveyor labels figures as estimates. |
| fp64 native on CPU and CUDA/HIP; Apple GPUs lack hardware fp64 | Uncertain | Not contradicted by docs; Apple limitation is general knowledge, unchecked. |
| MPI codegen target, PyTorch tensor interop via __cuda_array_interface__, ICON/GT4Py-adjacent use | Unverified | Not checked; not relied on for decisions. |
| All "better than C++" perf expectations | Unverified | Unmeasured speculation; treat as hypotheses. |

## Sources

- https://github.com/spcl/dace
- https://github.com/spcl/dace/releases
- https://spcldace.readthedocs.io/en/latest/
- https://spcldace.readthedocs.io/en/latest/optimization/gpu.html
- https://spcldace.readthedocs.io/en/latest/source/dace.codegen.targets.html (cited by the survey; the verifier did not confirm reading it)
- https://xdsl.readthedocs.io/stable/reference/backend/mps/ (unrelated MPS backend found in the Metal search)
- https://pypi.org/project/dace/ (failed to load; version unverified)
