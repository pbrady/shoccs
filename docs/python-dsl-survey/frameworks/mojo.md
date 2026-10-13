# Mojo / MAX (Modular) for SHOCCS

## Verdict box

- **Fit for SHOCCS:** plausible CPU codegen backend, speculative Apple GPU backend. Best used as a code-generation target, not as a Python-superset DSL host.
- **CPU parity:** expected (both go through LLVM), but no performance number in this survey is measured.
- **Apple GPU:** young and aimed at ML inference. MAX graphs run on M1 to M5 in nightlies; a user custom-kernel path is undocumented and atomics (needed for scatter-add) are unconfirmed.
- **fp64:** full on CPU; hardware fp64 on CUDA/HIP is expected but not confirmed in current docs; none on Apple GPU (hardware limitation, not re-verified here from a primary source).
- **Risk:** medium to high. Single vendor (now confirmed part of Qualcomm), stability guarantee scope unclear, 26.6 still broke closure syntax, thin scientific ecosystem.

Overall score (verifier-adjusted): **5 / 10**. Sub-scores: cpu_parity 8, apple_gpu 3, idiom_coverage 6, ergonomics 5, long_term_risk 4.

## Status as of 2026-10-12

- Current release: MAX 26.6 (Sep 17, 2026), per the MAX changelog. Confirmed.
- Mojo 1.0 shipped Aug 11, 2026 with Modular 26.5 (confirmed by the 26.5 blog). 26.4 (Jun 18, 2026) carried Mojo 1.0 beta 2.
- The 26.5 blog says 1.x changes should be primarily additive. The survey's narrower claim that stability covers "only explicitly marked stable APIs" was not found in the fetched text and is unverified.
- 26.6 still made breaking closure changes: closures are passed as runtime values rather than compile-time capturing parameters, and capture lists such as `{imm}` replace `@__parameter` / `@__copy_capture`. Confirmed.
- Ownership: Modular is confirmed as part of Qualcomm (Modular forum post "Qualcomm to Acquire Modular"; "A Qualcomm Company" footer on the 26.5 blog). Per secondary sources the deal was announced around Jun 24, 2026 and closed around Jul 29, 2026. The earlier "reportedly acquired two weeks before 1.0" wording was understated and the timing imprecise.
- Licensing: the GitHub repo is Apache-2.0 with LLVM exceptions and lists the Mojo compiler as a component; MAX is under the Modular Community License; the README says compiler contributions are not yet accepted. A compiler source release on Aug 18, 2026 (attributed to ModCon) is supported by secondary sources (Phoronix, Jon Peddie); the ModCon attribution is unverified.
- Mojo is not a Python superset in practice: Python-like syntax with static typing, ownership and a different execution model.
- Not checked: 26.6 Mojo language release notes on mojolang.org.

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| CPU (arm64, x86) | Native AOT/JIT via MLIR/LLVM. SIMD types, `vectorize`, `parallelize`. | Full (`Float64`, `SIMD[DType.float64, N]`) | No Kokkos-like portability layer and no TeamPolicy; team/vector hierarchy is hand-written. MAX Graph also has a CPU backend. |
| NVIDIA CUDA | Mature. `DeviceContext`, shared memory, warp ops, tensor cores, atomics. | Hardware fp64 exists; not directly confirmed in current Modular docs (unverified). An old forum thread reported an fp64 memset compile error. | MAX graphs and custom Mojo ops. |
| AMD ROCm/HIP | Supported by MAX. | Not checked. | |
| Apple Metal (MPS-like goal) | Mojo lowers GPU functions to LLVM IR then AIR bitcode (MSL 3.2) using Xcode 16+ and macOS 15 SDK, via `MetalDeviceContext`. MAX graphs run on M1 to M5 in nightlies (Jun 2026 forum post; main testing M3 to M5). 26.6 adds `DeviceContext.create_event` / `DeviceEvent` (MTLSharedEvent) and `DeviceBuffer.unsafe_host_ptr` (CPU-addressable on unified memory, reads uncached). | None in Apple GPU hardware (widely documented; Apple forum, PyTorch MPS, MLX issue cited by the survey, not re-fetched in verification). No Modular source states Metal float64 works. | `createGraphBuilder` (graph capture) is CUDA/HIP only. Atomics, async_copy and GPU print were open as of the Dec 2025 to Mar 2026 forum thread; current status unverified. Needs macOS tooling, so no Linux container. |

Practical consequence: on Apple silicon the fp64 solver stays on the CPU. A hybrid layout (fp32 bulk stencil on GPU, fp64 closures and R points on CPU, sharing unified memory) is plausible but untested, and fp32 accuracy for small-psi closures remains an open question that is independent of Mojo.

## Idiom mapping

Support: native, workaround, unsupported. Performance statements are architectural expectations, not measurements.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | Mojo structs with `List[Float64]` or `UnsafePointer` buffers; or one packed D\|Rx\|Ry\|Rz allocation with static offsets. Python side exchanges NumPy / MAX Buffer via DLPack. | Equal to C++; setup only. |
| I02 | workaround | Keep in NumPy (vectorised quadratic roots plus lexsort), pass SoA arrays to Mojo. Bit-exact snap/clamp/nextafter port needs care; `math.nextafter` availability unverified. | Setup only. |
| I03 | native | Mojo loop over pencils producing fragment tables, or NumPy host. | Equal or better; setup only. |
| I04 | native | NumPy masks/indices uploaded as buffers; plain `List[Int32]` or bool masks in Mojo. | Setup only. |
| I05 | native | Compile-time parameters for interior taps; plain Float64 tables for closures; small RBF solve in NumPy. | Compile-time taps can beat C++ virtual dispatch. |
| I06 | workaround | Retarget the SymPy CSE printer (stencil_gen) to emit Mojo `fn` bodies, or keep evaluation in NumPy/lambdify and upload tables. Mojo has no sympy. | Setup only. |
| I07 | workaround | Host-side Python/Numba, or Mojo with searchsorted-style lookup. | Setup only; better than the O(\|R\|) scan. |
| I08 | native | Mojo loops over fragment table emitting line table, closure table and ELL rows; or NumPy. | Equal to C++. |
| I09 | native | Same as I08; compose deriv_coeff x interp_coeff in plain loops. | Equal to C++. |
| I10 | workaround | No scipy.sparse in Mojo. Build COO to ELL/CSR in NumPy (lexsort) or with stdlib sort. Prefer compact ELL. | Setup only. |
| I11 | native | Prefix-sum with plain loops, concatenate into one buffer, closure dedup. | Much better than ~700k Kokkos allocations. |
| I12 | native | Python front end (replacing Lua), calling Mojo via extension modules or running a Mojo program. | N/A. |
| I13 | native | Flat pointer indexing with explicit strides, or `LayoutTensor` row-major (nx,ny,nz). | Equal to C++. |
| I14 | native | Build R tables in Python lexsort order, or use explicit permutation arrays. | No cost. |
| I15 | native | `UnsafePointer[Float64]` or `Span`; `DeviceBuffer` on GPU. | Equal to C++. |
| I16 | workaround | Hand-written per-direction kernel: `parallelize` over pencils with SIMD over the fast index; fragment table for line offsets; no TeamPolicy. See sketch 1. | Parity with an equivalent C++ hand loop; possibly 2-5x better than the current one-team-per-line kernel on the slow axis if coalesced (estimate). |
| I17 | native | SIMD loads of shifted slices with compile-time taps; `vectorize` plus `parallelize` over outer index. | Parity with C++ -O3; memory-bound at large N. |
| I18 | native | Small fixed-size dense matvec on first/last planes, compile-time sizes per scheme, SIMD across the fast index. | Parity. |
| I19 | workaround | Precomputed ELL table (idx, C, out_idx) with gather-dot loop overwriting interior results at those rows. | Parity with C++ ELL loop. |
| I20 | native | Plain loops over CSR/ELL arrays; one thread per row on GPU. | Parity; better after compaction. |
| I21 | native | Compact COO (row, w, R-index) loop with `+=` into D. See sketch 2. | Much faster than full-height C++ CSR at large N; parity at small N. |
| I22 | native | Broadcast axpy over X face slabs; no-op without a Neumann face. | Parity or better. |
| I23 | native | ELL gather-dot padded to max nnz over source D concatenated with R. See sketch 2. | Parity. |
| I24 | native | Merge into the Bf ELL or run as a small separate loop. | Parity. |
| I25 | native | Fold into the fused kernel (output written once), or memset/fill. | Better than C++ only if the author writes the fused kernel; nothing fuses automatically. |
| I26 | native | Fold the scale into the fused output loop. | Parity or better. |
| I27 | native | One explicit fused loop summing Ox+Oy+Oz per point, or sequential accumulate passes; ordering by program order. | Parity; fusion is a manual choice. |
| I28 | native | Three output buffers in one pass, or a tuple of arrays. | Parity. |
| I29 | native | Fused elementwise loop, or a MAX custom op via `foreach`. | Parity; fusion with derivative output is manual. |
| I30 | native | Masked add loop (dense bool mask) or indexed gather loop. See sketch 3. | Parity. |
| I31 | native | Strided slice loops for faces; small index lists for R. See sketch 3. | Parity. |
| I32 | workaround | On CPU write rhs() as one compiled Mojo function, so no per-node launch cost. On GPU, MAX Graph fuses ops in the compiled graph; graph capture is not available on Metal. | Parity or better on CPU. Apple GPU submission cost not found measured. |
| I33 | native | Plain Mojo functions; no std::function. Fences map to `DeviceContext.synchronize` on GPU only. | Better than C++ eager. |
| I34 | workaround | Vectorised closed-form MMS in a Mojo kernel (`exp`/`cos`/`pow` with SIMD), or MAX graph elementwise ops. | Better than the C++ virtual-call path if SIMD libm is vectorised (unverified for Mojo). |
| I35 | unsupported | Mojo cannot call Lua cheaply. Replace with Python-defined MMS compiled to Mojo source (SymPy codegen) or a precomputed table. | N/A; removed by design. |
| I36 | native | Pass t as a scalar into the compiled fused RHS function. | Better than C++. |
| I37 | native | Fuse stage axpby into one SIMD loop over the packed buffer. See sketch 4. | Parity or better (about 36 launches to 1). |
| I38 | native | Pointer swap or memcpy; no pointer-stability constraint without graph baking. | Free. |
| I39 | native | Evaluate the exact solution on boundary sets only, into preallocated buffers. | Large win versus C++. |
| I40 | native | Hoist masks to setup. | Removes per-step allocation. |
| I41 | native | Masked max/argmax loop; parallel reduction via `parallelize` with per-thread partials. GPU needs an explicit reduction kernel (26.6 includes a Metal row-reduction determinism fix). | Parity on CPU; host readback cheap on unified memory. |
| I42 | native | Python file I/O or Mojo file API; XDMF via Python. | Equal; gating the error field makes it faster. |
| I43 | native | Python driver loop calling a Mojo step function, or an all-Mojo loop. | Negligible overhead. |
| I44 | workaround | Export tables to NumPy and use scipy/numpy eig on host. LAPACK via C FFI is possible. | Same as current. |
| I45 | native | Reference loops in Mojo or NumPy. | N/A. |
| I46 | native | Plain functions or compile-time parametric closures; no expression templates needed. | Parity. |
| I47 | workaround | Pointwise physics as SIMD functions; component axis as outer loop or leading `LayoutTensor` dimension. | Parity; fusion is manual. |
| I48 | workaround | Stay in NumPy/scipy; Mojo adds nothing. | N/A. |
| I49 | workaround | SymPy stays in Python; printer retargeted to emit Mojo source or NumPy/lambdify. | N/A. |
| I50 | unsupported | No ARPACK/SVD in Mojo. Stay in scipy fp64 on host; Mojo exports operator tables or a LinearOperator callback. | N/A. |
| I51 | native | In-process Python-Mojo call removes the subprocess loop; torch/botorch unchanged. | Large win (no process start per design point). |

## Code sketches

Important caveat from verification: these sketches use pre-26.6 `@parameter` closure syntax. 26.6 changed closures to runtime values with explicit capture lists, so they will not compile as written on 26.6. They are also not compiled or checked against 1.0 APIs (`LayoutTensor`, `parallelize`, `vectorize` names unchecked). Treat them as shape-of-the-algorithm only.

### 1. Per-line block matvec with strides and per-line dense closures (I16)

```mojo
# Mojo-style sketch; exact 1.0 / 26.6 syntax not compiled or verified.
fn block_matvec[W: Int, SIMD_W: Int](
    x: UnsafePointer[Float64], b: UnsafePointer[Float64],
    meta: UnsafePointer[Int32], coef: UnsafePointer[Float64],
    taps: InlineArray[Float64, W], n_lines: Int):
  @parameter
  fn per_line(l: Int):
    var m = meta + 13 * l
    var row0 = Int(m[0]); var col0 = Int(m[1]); var s = Int(m[2])
    var lr = Int(m[3]); var lc = Int(m[4]); var lco = Int(m[5])
    var ir = Int(m[6]); var rr = Int(m[9]); var rc = Int(m[10])
    var rco = Int(m[11]); var rcol = Int(m[12])
    for r in range(lr):          # left dense closure
      var acc = 0.0
      for j in range(lc): acc += coef[lco + r*lc + j] * x[col0 + j*s]
      b[row0 + r*s] = acc
    for r in range(lr, lr + ir):  # interior circulant
      var acc = 0.0
      for j in range(W): acc += taps[j] * x[row0 + r*s + (j - W//2)*s]
      b[row0 + r*s] = acc
    for r in range(rr):          # right dense closure
      var acc = 0.0
      for j in range(rc): acc += coef[rco + r*rc + j] * x[rcol + j*s]
      b[row0 + (lr + ir + r)*s] = acc
  parallelize[per_line](n_lines)  # for speed, tile so SIMD runs over the fast index
```

### 2. O/B CSR cut-cell coupling to R points as compact ELL (I21, I23, I24)

```mojo
# B: R{dir} -> D, compact COO; Bf: D -> R(r), ELL padded to K
fn apply_B(du: UnsafePointer[Float64], uR: UnsafePointer[Float64],
           rows: UnsafePointer[Int32], cols: UnsafePointer[Int32],
           w: UnsafePointer[Float64], nnz: Int):
  # rows unique per (R point, offset), no atomics needed
  for k in range(nnz):
    du[Int(rows[k])] += w[k] * uR[Int(cols[k])]

fn apply_Bf[K: Int](duR: UnsafePointer[Float64], uD: UnsafePointer[Float64],
                    uR: UnsafePointer[Float64], idx: UnsafePointer[Int32],
                    C: UnsafePointer[Float64], nR: Int, nD: Int):
  # source is concatenated [D ; R]: idx >= nD addresses R entries (Br merged in)
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

### 3. Gather/scatter boundary application (I30, I31)

```mojo
# Masked add at fluid points; dense bool mask version
fn masked_add(du: UnsafePointer[Float64], src: UnsafePointer[Float64],
              fluid: UnsafePointer[UInt8], n: Int):
  @parameter
  fn chunk[w: Int](i: Int):
    var m = fluid.load[width=w](i).cast[DType.bool]()
    var d = du.load[width=w](i)
    du.store(i, m.select(d + src.load[width=w](i), d))
  vectorize[chunk, 8](n)

# Dirichlet face x = 0 plane: contiguous slab
fn fill_x_face(u: UnsafePointer[Float64], g: UnsafePointer[Float64],
               i: Int, ny: Int, nz: Int):
  for q in range(ny * nz): u[i*ny*nz + q] = g[q]

# index-list form for the tiny R sets
fn scatter_set(dst: UnsafePointer[Float64], idx: UnsafePointer[Int32],
               val: UnsafePointer[Float64], n: Int):
  for k in range(n): dst[Int(idx[k])] = val[k]
```

### 4. Four-buffer scalar algebra with RK4 slot updates (I37)

```mojo
# Pack D|Rx|Ry|Rz into one buffer of length N; one fused pass per stage
fn rk4_stage(u1: UnsafePointer[Float64], u0: UnsafePointer[Float64],
             k: UnsafePointer[Float64], rk: UnsafePointer[Float64],
             c_assign: Float64, c_acc: Float64, first: Bool, N: Int):
  @parameter
  fn body[w: Int](i: Int):
    var kk = k.load[width=w](i)             # srhs from the RHS call
    rk.store(i, kk * c_acc if first else rk.load[width=w](i) + c_acc*kk)
    u1.store(i, u0.load[width=w](i) + c_assign * kk)
  vectorize[body, 8](N)

fn rk4_step(mut s: State, dt: Float64, t: Float64):
  rhs(s.u0, s.srhs, t)
  rk4_stage(s.u1, s.u0, s.srhs, s.rk, dt/2, dt/6, True, s.N)
  update_boundary(s.u1, t + dt/2)
  rhs(s.u1, s.srhs, t + dt/2)
  rk4_stage(s.u1, s.u0, s.srhs, s.rk, dt/2, dt/3, False, s.N)
  update_boundary(s.u1, t + dt/2)
  rhs(s.u1, s.srhs, t + dt/2)
  rk4_stage(s.u1, s.u0, s.srhs, s.rk, dt, dt/3, False, s.N)
  update_boundary(s.u1, t + dt)
  rhs(s.u1, s.srhs, t + dt)
  # final: u1 = u0 + rk + dt/6*srhs, then update_boundary(u1, t + dt)
```

## Performance assessment

None of this is measured; it rests on architectural reasoning.

**Small 2D (about 2.6k to 5k points).** Good if the whole RHS and step are compiled Mojo functions: there is no per-node dispatch, and roughly 280 launches collapse to a few loops. The survey's estimate is parity or better than the C++ figure of 0.3 to 1 ms/step. `parallelize` on a few thousand points may cost more than it saves, so run serial or parallelize only per line. Per-call Python-to-Mojo overhead is the unknown (Modular says it is investing in reducing it; this was not checked in verification), so call once per step or run the loop in Mojo.

**Large 3D.** Memory-bound. Both languages go through LLVM, so a coalesced fused axis stencil should reach the same bandwidth (about 60 to 120 GB/s on M-series unified memory, an estimate). Headroom comes from compact B/N tables and removing per-step host MMS evaluation. On the Apple GPU only fp32 is possible, so fp64 large-3D parity is CPU-only.

**Dispatch overhead.** CPU: a Mojo function call is a native call (nanoseconds). The per-invocation cost of `parallelize` is undocumented in what was read and is unverified. Apple GPU: enqueue goes through the Metal command queue; no measured per-launch cost was found, and Metal launches typically cost several to tens of microseconds, so small-2D on GPU would not beat the CPU.

**Fusion.** Mojo does not fuse hand-written loops; the author (or a generator) writes the fused loop, as in C++. MAX Graph fuses elementwise chains, offers eager mode with JIT by default in 26.6 (confirmed), and accepts Mojo custom ops via `foreach`, but nothing was found on stencil-style neighbour-access fusion, and whole-step compilation through a MAX graph is untested. The realistic strategy is a Python DSL that emits Mojo source for the whole RHS.

## Suitability as a DSL substrate

Mixed.

Pros:
- LLVM-level CPU speed means a DSL emitting whole-RHS Mojo source could reach C++ parity without a framework dispatch tax, which is the hardest bar (small 2D).
- Compile-time parameters suit scheme-specialised taps and closure sizes.
- One language targets CPU, CUDA, ROCm and Metal kernels.
- Python interop lets SymPy/NumPy setup stay in Python and share arrays. Interop mechanisms: Mojo functions can be built as Python extension modules; Mojo can import Python modules; MAX Buffer supports `to_numpy()` and `__dlpack__` (confirmed in 26.6 notes); NumPy pointers can be wrapped in `UnsafePointer`; C is reachable via `external_call` / `DLHandle`. There is no C++ template or Kokkos interop, so existing C++ cannot be reused as is.

Cons:
- Not a Python superset, so the DSL is a source-to-source generator (Python AST to Mojo text to `mojo build`), with compile latency and debugging through generated code.
- No tracing JIT or autodiff like JAX/Triton; fusion is the generator's job.
- Thin scientific ecosystem (no sparse, FFT, eigensolvers), so I44/I50 stay in SciPy.
- Generated code needs a pinned toolchain given 26.6 closure breakage.
- Apple GPU path is fp32-only and young.
- Single-vendor ownership (Qualcomm), MAX under the Modular Community License, compiler contribution closed.

Compared with a C++ codegen backend through Kokkos, Mojo adds a Metal route at the price of toolchain risk, and it does not remove the fp64 limitation that applies to every framework.

## Risks

- No fp64 on Apple GPU hardware, so the MPS goal needs fp32 or double-float emulation (several-x cost). Small-psi closure accuracy in fp32 is an open question independent of Mojo.
- Source-to-source generation is required; Python code does not run as Mojo.
- Stability: the scope of the 1.0 guarantee is unverified, and 26.6 still introduced breaking closure changes.
- Apple GPU support targets ML inference; atomics, async_copy, GPU print and debugger status are unconfirmed after Mar 2026. A host-write unified-memory bug was reported in Mar 2026. No documented user custom-kernel path was seen.
- Single-vendor risk: Qualcomm ownership is confirmed; the effect on Apple-GPU investment is unknown.
- Ecosystem gaps: no sparse or eigensolver libraries.
- No measured numbers for `parallelize` or Metal launch overhead; small-2D parity is an architectural argument.
- Python-to-Mojo call overhead and zero-copy semantics are still an improvement area; unified-memory host pointer reads are uncached.
- Code sketches are out of date relative to 26.6 closure syntax.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| MAX 26.6 (Sep 17, 2026) is latest; notes list Apple-GPU events, unified-memory host pointer; float64 mentioned only for a graph MLIR constant bug | confirmed | Changelog says latest is 26.6 (Sep 17, 2026). Notes: `create_event()` / `DeviceEvent` supported on Apple GPUs via MTLSharedEvent; `unsafe_host_ptr` CPU-addressable, uncached; "Fixed Graph MLIR text I/O corrupting float64 constants". Device-sync debug mode and bfloat16 fix not checked. |
| 26.6 has breaking closure changes; eager tensors use JIT by default; `__dlpack__` and `to_numpy` exist | confirmed | 26.6 notes: closures are runtime values, capture lists replace `@__parameter` / `@__copy_capture`; eager uses JIT by default; staging-buffer change covers `__dlpack__` / `to_numpy`. |
| MAX graphs run on Apple GPUs M1 to M5 in nightlies; `createGraphBuilder` not on Metal; no custom-kernel process described | confirmed | Modular forum post (Jun 27, 2026); `createGraphBuilder` only CUDA/HIP; main testing M3 to M5; nightlies, not a stable release. |
| Metal gaps: atomics, async_copy, GPU print/debug_assert, unified-memory host-write bug | confirmed (as of Mar 2026) | Forum thread updated to Mar 2026 lists these. SubBuffer and captured args not seen in the summary. Thread predates 26.6; current status unverified and some items on that page are superseded. |
| Mojo 1.0 shipped Aug 11, 2026 in Modular 26.5 | confirmed | 26.5 blog. |
| 1.0 stability covers only explicitly marked stable APIs | uncertain | Not in fetched text; blog says 1.x changes should be primarily additive. |
| Licensing: Apache-2.0 with LLVM exceptions, MAX under Modular Community License, compiler contributions not accepted | confirmed | GitHub README. README also lists the compiler as a repo component. |
| Compiler source released at ModCon Aug 18, 2026 | uncertain | Phoronix and Jon Peddie report a public release on Aug 18; 26.5 blog only promised open-sourcing in 2026; ModCon attribution unverified. |
| Modular acquired by Qualcomm about two weeks before 1.0, secondary source only | refuted (as stated) | Confirmed by Modular forum post and "A Qualcomm Company" footer. Announced about Jun 24, 2026, closed about Jul 29, 2026 per secondary sources. Reported value roughly $3.1 to 3.9B. |
| Apple GPUs/Metal have no hardware fp64, so no Mojo/MAX Metal path runs fp64 natively | uncertain | Not re-checked against Apple/PyTorch/MLX links in verification; no Modular page fetched mentions Metal float64 support. Widely known hardware fact. |
| Modular is investing in reducing Python-to-Mojo call overhead | uncertain | 26.4 blog not fetched. |
| Code sketches compile on current Mojo | refuted | Sketches use pre-26.6 `@parameter` closures; API names (`LayoutTensor`, `parallelize`, `vectorize`) and `math.nextafter` / vectorised `exp`/`cos` unchecked. |
| Performance parity claims (CPU, small 2D, large 3D, Metal launch cost) | unverified | Architectural reasoning only; no benchmarks. |

## Sources

- https://max.modular.com/changelog
- https://max.modular.com/releases/v26.6/
- https://forum.modular.com/t/max-models-can-now-run-on-apple-silicon-gpus/3283
- https://forum.modular.com/t/apple-silicon-gpu-support-in-mojo/2295
- https://forum.modular.com/t/qualcomm-to-acquire-modular/3271
- https://modular.com/blog/modular-26-4-sota-moe-serving-model-bringup-via-agent-skills-mojo-beta-2-and-more
- https://modular.com/blog/modular-26-5-mojo-1-0-is-here
- https://github.com/modular/modular
- https://www.phoronix.com/news/Modular-Mojo-Open-Source
- https://www.jonpeddie.com/news/qualcomm-opens-mojo-to-challenge-cuda/
- https://linuxiac.com/mojo-1-0-programming-language-officially-released/
- https://www.how2shout.com/news/mojo-1-0-release-modular-qualcomm.html
- https://developer.apple.com/forums/thread/709663
- https://discuss.pytorch.org/t/float64-tensors-on-mps/224325
- https://upd.dev/ml-explore/mlx/issues/1905
- https://forum.modular.com/t/gpu-float64-memset-support/586
