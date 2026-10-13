# Direct Metal from Python (pyobjc Metal, metalcompute, MLX mx.fast.metal_kernel, metal-cpp via nanobind)

> **Verdict**
> - **Fit for SHOCCS:** Good as the Apple-GPU backend layer under a DSL, poor as a standalone framework. We write every kernel ourselves.
> - **CPU parity:** None from Metal itself (GPU only). Small 2D sizes (2.6k-5k points, C++ RHS about 30-40 us) will very likely lose to the CPU loop (unmeasured estimate).
> - **Apple GPU:** Best access available (unified memory, command buffers, ICBs, heaps) via MSL, but every route needs us to own the kernels.
> - **fp64:** None on the Apple GPU. fp32 or double-float emulation only. fp64 exists only on CPU backends (numpy, C++, MLX CPU).
> - **Risk:** Medium. fp32 accuracy of small-psi closures is untested, and a metal-cpp extension is a permanent maintenance burden. Overall score 5 of 10.

## Status as of 2026-10-12

| Layer | Version checked | Notes |
|---|---|---|
| MLX | v0.32.3 (GitHub releases, tag dated 29 Sep, year not shown; 2026 inferred from the version sequence) | Metal and CUDA backends. v0.32.1 fixed a custom-kernel cache collision for same name and different source (#3833). v0.32.0 fixed a use-after-free when a custom Metal kernel is called with different dtypes in one graph (#3662). |
| pyobjc-framework-Metal | 12.2.2, uploaded 2026-08-11 (PyPI JSON) | Thin, complete Objective-C binding. Per-call bridge overhead (overhead size is an unmeasured estimate). |
| metalcompute (py-metal-compute) | 0.2.9, released 2025-01-21, macOS 14+ arm64 wheels (PyPI JSON) | Corrects the survey's "0.2.4". Looks stale and hobby-scale. No batching or ICB API was found (absence of evidence only). |
| metal-cpp and nanobind | Not checked | Writing our own extension is feasible but unassessed here. |

None of these is a PDE framework. Each is an execution substrate.

## Backends and fp64

| Backend | Support | fp64 | Notes |
|---|---|---|---|
| CPU | Not provided by Metal. MLX has a CPU backend with mx.float64, but it is not where parity is reached. mx.fast.metal_kernel does not run on MLX CPU (medium confidence). | Yes (numpy, numba, C++, MLX CPU) | Needs a separate substrate for the reference path. |
| CUDA | None for MSL. MLX has a CUDA backend (CUDA changes listed in v0.32.0 to v0.32.3). | n/a | mx.fast.cuda_kernel appears in a third-party Mintlify mirror of the MLX docs only. Unverified on official docs. |
| Metal / MPS | Primary target. We write MSL and dispatch it. MPS and MPSGraph are optional and add little for stencils. | No. Apple GPUs have no fp64 (the Apple forum thread was not re-fetched; MLX #1905 maintainer comments say PyTorch MPS also lacks double, which supports it indirectly). | MLX GPU fp64 emulation was declined: issue #1905 closed 2025-12-03 (confirmed on GitHub). The "mx.float64 CPU only" half was not re-verified from a primary source. |
| Other | Vulkan and WebGPU out of scope. | n/a | Apple Silicon only. |

Emulation: community double-float work (metal-float64) is cited at roughly 4x to over 27x cost. Unverified, speculative forum estimates.

Practical split: setup (psi closures, stencil coefficients, ray casting) in float64 on the CPU, cast coefficients to fp32 on upload, time-stepping state in fp32, with double-float hi+lo pairs only for accumulations that need it (for example the RK accumulator). Whether fp32 is acceptable for small-psi high-order closures is untested.

## Idiom mapping

Support values: native, workaround, unsupported. Because we write the kernels, "native" means directly expressible in MSL or host numpy. All performance entries are unmeasured expectations.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | One packed buffer or array per field with static offsets D, Rx, Ry, Rz (or a 4xN array). | Equal to C++. Packing cuts launches 4x. |
| I02 | workaround | Host numpy float64: vectorised quadratic roots over all lines and shapes, filter, lexsort, snap and clamp; np.nextafter for exact semantics. | Setup only. Likely faster than the C++ virtual-call loop. |
| I03 | workaround | numpy merge-join per pencil (searchsorted or segmented scan) emitting start, len, stride arrays. | Setup only, negligible. |
| I04 | native | Bool masks or int32 index arrays built in numpy, uploaded once. | Setup only. Hoisting the per-call rebuild (I40) beats current C++. |
| I05 | native | Python constants cast to fp32, baked into MSL or passed as a small constant buffer. | Setup only. |
| I06 | workaround | sympy.lambdify or a retargeted CSE printer (reuse scripts/stencil_gen/codegen.py with a numpy backend, re-add psi clamps), fp64 evaluation, cast at upload. | Setup only, much faster than per-point C++ calls. |
| I07 | workaround | Host numpy or numba with searchsorted fragment lookup, emitting padded (idx, coef) ELL tables. | Setup only, one-time cost. |
| I08 | workaround | numpy grouping of lines by (left_kind, right_kind, bc); emit a 13-int32 line table plus a deduplicated closure table. | Setup only. |
| I09 | workaround | numpy composition of derivative and interpolation weights into ELL tables idx[nR,Kmax], C[nR,Kmax]. | Setup only. |
| I10 | native | np.lexsort or scipy coo_matrix; keep compact COO or ELL, never a full-height row pointer. | Setup only. Avoids the 134 MB row pointer. |
| I11 | native | np.cumsum and np.concatenate, then newBufferWithBytes (pyobjc) or mx.array. | Setup only. |
| I12 | native | Python dicts or dataclasses replace the Lua front end. | Setup only. |
| I13 | native | Flat indexing in MSL with strides as constants; threads cover the fast index. | Parity or better if threads map across the fast index; bandwidth-bound at 256^3. |
| I14 | native | The R ordering contract is host data (slow, fast, t lexsort); all tables indexed consistently. | No runtime cost. |
| I15 | native | Work buffers allocated once as shared-storage MTLBuffers. | Equal. |
| I16 | workaround | Interior kernel plus closure ELL kernel plus masking, or one thread per (line, row) with a 13-int32 meta record (sketch 1). | Bandwidth-bound in fp32; plausibly 2-3 ms per axis at 256^3 (unmeasured). Current C++ is 13.7 ns/pt at 64^3. |
| I17 | native | MSL stencil: out[i] = sum_j c[j]*x[i+(j-w/2)*stride], taps as constants. | Bandwidth-bound, near optimal in fp32. |
| I18 | native | Small dense closure block as constants, on first and last planes (second launch or tail branch). | Negligible. |
| I19 | workaround | ELL gather-dot-scatter table that overwrites interior results, so it must run after the interior kernel. | Small (2.4% of rows); launch-bound at small sizes unless merged. |
| I20 | workaround | Replace CSR with ELL or COO; one generic gather-dot kernel; rows unique, no atomics. | Equal or better than C++. |
| I21 | workaround | Compact COO (row, col_R, w), row unique per entry. | Better than C++ (about 1e5 entries instead of a 16.8M-row sweep). |
| I22 | native | Broadcast axpy on X slab planes, skipped without Neumann BC. | Negligible. |
| I23 | workaround | ELL gather over a concatenated [D;R] source buffer. | Latency-bound at small sizes; fuse with I24. |
| I24 | workaround | Merge into the same ELL over the concatenated source, removing the Bf-to-Br dependency chain. | As I23. |
| I25 | native | Disappears: the first kernel writes the output; later kernels only accumulate where needed. | Better than C++ (saves a pass). |
| I26 | native | Fold the scale into the final write. | Better than C++. |
| I27 | workaround | One kernel summing the three directional stencils, or three dispatches in one command buffer with a serial dispatch type and fences. | Fused version reads D once; can beat C++ at large sizes. |
| I28 | native | One kernel with 3 outputs, or three independent dispatches in a concurrent encoder. | Parity. |
| I29 | native | Fuse the dot into the gradient kernel epilogue, no scratch. | Better than C++ at large sizes (avoids 400 MB scratch). |
| I30 | native | Dense masked FMA with a uint8 mask, or an indexed kernel for tiny R sets. | Parity or better, fusable. |
| I31 | native | Tiny face kernels or masked where() in the main epilogue. | Negligible. |
| I32 | workaround | Encode the RHS or step into one command buffer and commit once; optionally record an MTLIndirectCommandBuffer. ICB is reachable from pyobjc or metal-cpp, not MLX. metalcompute has no batching API found. mx.compile cannot replay custom kernels (medium confidence). | pyobjc encoding of about 40 dispatches estimated at 0.4-2 ms (unmeasured). A C++ extension with ICB may reach single-digit us per dispatch. Submit latency likely still exceeds the 30-40 us CPU RHS at 2601 points. |
| I33 | native | Plain Python function encoding the kernels; fences become waitUntilCompleted at sync points. | Slower than C++ unless batched. |
| I34 | native | Closed forms in MSL or numpy; SymPy-generated code evaluated per point. exp and cos exist in MSL. | Faster than C++ host virtual calls. |
| I35 | unsupported | Per-point Lua callbacks cannot run on the GPU. Replace with a SymPy array expression or precomputed table. | N/A, must be replaced. |
| I36 | workaround | Pass t as a kernel scalar (setBytes) and evaluate the source on device. | Removes the host sync; faster than C++. |
| I37 | native | Fused axpby over the packed state; RK4 stage update plus accumulate in one kernel. | Bandwidth-bound parity; dispatch-bound at small sizes. |
| I38 | native | Swap buffer handles in Python (ping-pong); no baked pointers. With an ICB, update bindings by index. | Better than C++ (no copies). |
| I39 | native | Small kernels over boundary sets only. | Better than C++ (no full-field evaluation). |
| I40 | native | Hoist to setup. | Better than C++. |
| I41 | workaround | simd_max plus threadgroup reduction, atomic_max on a uint-encoded float, or two-pass reduce; read back a small shared buffer every K steps. | Cheap on unified memory, but a per-step wait breaks pipelining. Check every K steps. |
| I42 | native | Error field in a kernel gated on dump interval; np.frombuffer on the shared buffer; XDMF written from Python. | Parity or better. |
| I43 | native | Python while loop around the encoded step. | Parity at large sizes only if each step is one command buffer. |
| I44 | native | scipy.sparse from the same tables plus numpy.linalg.eigvals, fp64 on host. | N/A, diagnostic. |
| I45 | native | NumPy one-liners as oracles. | N/A. |
| I46 | native | Fused MSL expression kernels from a small codegen (SymPy printer or string templates). | Parity. |
| I47 | workaround | Pointwise physics in MSL with a leading component axis; batched derivatives take a component index. fp32 with fp64 CPU reference. | Plausibly bandwidth-bound; shock accuracy in fp32 is an open question. |
| I48 | native | Stays in numpy and scipy on host. | N/A. |
| I49 | native | Stays in SymPy on host; retarget the printer to numpy. | N/A. |
| I50 | native | Host scipy in fp64 on the exported matrix. | N/A. |
| I51 | native | In-process Python solver replaces the subprocess loop; torch float64 CPU stays for BO. | Removes per-design-point process startup. |

## Code sketches

These are untested sketches from the survey, not validated code.

Sketch 1: per-line block matvec with strides and per-line dense closures (fused kernel, thread per row and line; MLX metal_kernel, fp32).

```python
# meta: int32[nL,13] = row_off,col_off,stride,Lr,Lc,Lco,Ir,Ico,w,Rr,Rc,Rco,Rcoloff
# coef: float[...] per-line or deduped via closure id
src = '''
uint r = thread_position_in_grid.x, l = thread_position_in_grid.y;
constant int* m = meta + l*13;
int nrow = m[3] + m[6] + m[9];
if ((int)r >= nrow) return;
int s = m[2], out = m[0] + r*s;
float acc = 0.f;
if ((int)r < m[3]) {                         // left dense closure
  for (int j=0;j<m[4];j++) acc += coef[m[5]+r*m[4]+j]*x[m[1]+j*s];
} else if ((int)r < m[3]+m[6]) {             // interior circulant
  int w=m[8];
  for (int j=0;j<w;j++) acc += coef[m[7]+j]*x[out+(j-w/2)*s];
} else {                                     // right dense closure
  int rr=r-m[3]-m[6];
  for (int j=0;j<m[10];j++) acc += coef[m[11]+rr*m[10]+j]*x[m[12]+j*s];
}
y[out] = acc;   // skipped rows keep prior value of y (zero-init)
'''
k = mx.fast.metal_kernel('blk',['meta','coef','x'],['y'],src)
y = k(inputs=[meta,coef,x], grid=(maxrows,nL,1), threadgroup=(64,1,1),
      output_shapes=[x.shape], output_dtypes=[mx.float32], init_value=0)[0]
# NOTE: for coalescing prefer a fast-index-major interior kernel; this one is the faithful port.
```

Sketch 2: CSR cut-cell coupling to R points as compact COO and ELL, with no full-height row pointer.

```python
# B: R{dir}->D, entries (row, col_R, w); rows unique -> no atomics.
# Bf/Br merged: source = concat[D;R], ELL rows over R points.
srcB = '''
uint e = thread_position_in_grid.x;
if (e >= nE) return;
du_out[row[e]] = du[row[e]] + w[e]*uR[col[e]];   // accumulate; unique rows
'''
srcELL = '''
uint q = thread_position_in_grid.x;
if (q >= nR) return;
float a = 0.f;
for (int k=0;k<K;k++) a += C[q*K+k]*ucat[idx[q*K+k]];  // padding: C=0, idx=0
du_R[q] = duR[q] + a;
'''
kB  = mx.fast.metal_kernel('B',['du','uR','row','col','w','nE'],['du_out'],srcB)
kEL = mx.fast.metal_kernel('ell',['duR','ucat','idx','C','nR','K'],['du_R'],srcELL)
ucat = mx.concatenate([u_D, u_Rx])      # source D;R
du_R = kEL(inputs=[duR,ucat,idx,C,nR,K], grid=(nR,1,1), threadgroup=(64,1,1),
           output_shapes=[(nR,)], output_dtypes=[mx.float32])[0]
```

Sketch 3: gather and scatter boundary application.

```python
src = '''
uint i = thread_position_in_grid.x;
if (i >= n) return;
uint p = idx[i];                 // unique indices -> no atomics
out[p] = val[p]*keep + (1.f-keep)*u[p];
'''
# dense alternative for fluid-masked source add, fusable:
#   du = mx.where(fluid_mask, du+src, du)
u3 = u.reshape(nx,ny,nz)
u3[0,:,:] = sol_face0       # MLX slice assignment (graph op)
# R Dirichlet set: u_R[dir_obj_idx] = sol_R[dir_obj_idx]
# Direct-Metal variant: one kernel per index list, indices uploaded once at setup.
```

Sketch 4: four-buffer scalar algebra with RK4 slot updates (packed state, one fused kernel per stage).

```python
src = '''
uint i = thread_position_in_grid.x;
if (i >= N) return;
float s = srhs[i];
rk_out[i] = rk[i] + c_acc*s;           // slot_accumulate
u1_out[i] = u0[i] + c_asg*s;           // slot_assign_lc for the next stage
'''
k = mx.fast.metal_kernel('rk4_stage',['u0','rk','srhs','c_acc','c_asg','N'],
                         ['rk_out','u1_out'],src)
def step(u0, t, dt):
    rk = mx.zeros_like(u0); u1 = u0
    for c_asg, c_acc, tt in [(dt/2,dt/6,t),(dt/2,dt/3,t+dt/2),(dt,dt/3,t+dt/2),(0,dt/6,t+dt)]:
        srhs = rhs(u1, tt)
        rk, u1n = k(inputs=[u0,rk,srhs,c_acc,c_asg,u0.size],grid=(u0.size,1,1),
                    threadgroup=(256,1,1),output_shapes=[u0.shape]*2,output_dtypes=[mx.float32]*2)
        u1 = apply_bc(u1n, tt+(dt/2 if c_asg==dt/2 else 0))
    return u0 + rk                     # lazy; mx.eval once per step
# mx.compile fuses surrounding elementwise ops but custom kernels stay separate dispatches.
```

## Performance assessment

All figures below are the survey's estimates. Nothing was measured, and the verification did not check them.

- **Small 2D (2.6k-5k points):** Expect a loss. The C++ RHS is about 30-40 us, mostly launch cost. GPU submit and completion latency is expected to exceed that, and pyobjc encoding adds more per dispatch. Only a C++ extension encoding the step once (ICB or fixed command buffer) is plausible, and it probably still cannot beat an in-cache CPU loop. Keep the CPU path for small 2D.
- **Large 3D (256^3):** The C++ bar is roughly 30-110 ms per RHS. A fused fp32 kernel set moves about half the bytes of fp64 and might reach 10-30 ms if unified-memory bandwidth is well used. Not apples to apples because fp64 accuracy is lost.
- **Dispatch overhead:** pyobjc about 10-50 us per dispatch in Python (estimate). metalcompute is per-call with no batching API found. MLX custom kernels are lazy graph nodes evaluated together at mx.eval, with per-op overhead of order 10 us (estimate). metal-cpp via nanobind could reach a few us, less with an ICB.
- **JIT warmup:** MSL compiles on first use, cached by name and source in MLX. Tens to hundreds of ms per kernel (estimate), amortised over 20-2650 steps. Geometry tables are plain buffers and do not recompile.
- **Fusion:** No automatic whole-RHS compiler exists here. mx.compile fuses elementwise ops but not across custom kernels or strided stencil access (medium confidence). Fusion is the DSL's job.

## Suitability as a DSL substrate

Right as the bottom layer of an Apple-GPU backend, wrong as the only backend. The DSL would hold the operator IR (axis stencil, closure ELL tables, COO couplings, masks, axpby, reductions) plus numpy float64 setup, then emit MSL per fused group and dispatch it. The catalog reduces to about six primitives (P1-P6), so emission is mechanical. The effort lies in the code generator, fusion planning, a command-buffer scheduler, fp32 validation, and separate CPU (numba or existing C++) and optional CUDA backends. MLX metal_kernel gets us running fast with lazy evaluation and unified memory but limits us on in-place updates and replayable graphs. A nanobind extension over metal-cpp removes those limits at the cost of maintaining C++ ourselves.

Interop: on Apple Silicon, shared-storage buffers avoid H2D and D2H copies, but synchronisation is needed before CPU reads. With pyobjc, MTLBuffer.contents() can be wrapped via memoryview and np.frombuffer; newBufferWithBytesNoCopy requires page alignment. torch.mps.compile_shader and load_metallib exist in the official PyTorch docs (confirmed); whether they work with MPS tensors without private APIs was not verified. np.array(mx_array) memory sharing is unverified. The existing Kokkos code cannot dispatch to Metal.

## Risks

1. fp32 only on GPU. Small-psi closures with large cancelling rational coefficients may lose accuracy or stability. This decides viability and is untested.
2. At shipped small 2D sizes GPU submit latency probably exceeds the whole C++ RHS, so the GPU path is large-3D only.
3. MLX custom kernels have read-only inputs, no true in-place writes, no replayable graph, and a small API. Recent cache and lifetime bug fixes (v0.32.0, v0.32.1) suggest the area is still maturing.
4. metalcompute is stale (last release Jan 2025) and hobby-scale. Not a safe foundation.
5. pyobjc is slow per call and verbose; fine for prototyping, wrong for a per-step hot path.
6. A metal-cpp plus nanobind extension means owning C++ and Metal build maintenance and Apple API churn.
7. No CPU or CUDA story from the same code. Kernel semantics must stay consistent between fp64 CPU and fp32 GPU.
8. Partial-coverage write semantics (eq on solid points) and the R ordering contract must be re-verified, and t-laplacian currently fails on the C++ side, so there is no trusted R-point reference.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| MLX v0.32.3 with Metal and CUDA backends; v0.32.1 fixed custom-kernel cache collision; v0.32.0 fixed use-after-free | Confirmed | GitHub releases: #3833 and #3662 cited. Tag dated 29 Sep with no year shown; 2026 is inferred. |
| MLX declined GPU fp64 emulation (#1905, closed Dec 2025) | Confirmed | Closed 2025-12-03 on GitHub; maintainer said emulation would be slow. The "mx.float64 CPU only" half was not re-verified. Survey cited an upd.dev mirror. |
| metal_kernel inputs read-only, no upstream in-place (#2547) | Confirmed, wording corrected | Closed by its author 2025-09-03 after using a fork. Maintainer said MLX has no true in-place ops, only donation of unreferenced inputs, and suggested a donate map or an extension. The survey's "custom primitive workaround" is corrected to "write an extension"; donation is not shipped. |
| metalcompute requires macOS 14+ and offers compile and run | Confirmed, version corrected | PyPI JSON: latest 0.2.9 (2025-01-21), macosx_14_0_arm64 wheels. Survey's "newest 0.2.4" was wrong. No batching or ICB found (absence of evidence). The 0.1.x to 0.2.x API change claim was not checked. |
| pyobjc-framework-Metal current at 12.2.2 | Confirmed | PyPI JSON: 12.2.2, uploaded 2026-08-11. ICB exposure and per-call overhead not checked. |
| Apple GPUs and Metal expose no fp64 | Unverified | Apple forum thread not re-fetched. Indirect support from MLX #1905 comments on PyTorch MPS. |
| mx.fast.cuda_kernel exists | Unverified | Only in a third-party Mintlify mirror of the MLX docs. |
| torch.mps.compile_shader exists | Confirmed | Listed in official torch.mps docs (2.7 through 2.14 snapshots), along with load_metallib. Use with tensors not verified. |
| metal-float64 emulation costs 4x to over 27x | Unverified | Not checked; speculative forum estimates. |
| All performance numbers (dispatch overhead, 256^3 speedups, bandwidth) | Unverified | Unmeasured estimates by the survey. |

Adjusted overall score: 5 (unchanged). Only minor factual errors were found.

## Sources

- https://github.com/ml-explore/mlx/releases
- https://github.com/ml-explore/mlx/issues/1905
- https://github.com/ml-explore/mlx/issues/2547
- https://pypi.org/pypi/metalcompute/json
- https://pypi.org/pypi/pyobjc-framework-Metal/json
- https://docs.pytorch.org/docs/main/mps.html
- https://www.mintlify.com/ml-explore/mlx/api/fast (third-party mirror)
- https://www.mintlify.com/ml-explore/mlx/cpp/metal-kernels (third-party mirror)
- https://developer.apple.com/forums/thread/709663 (not re-fetched)
- https://developer.apple.com/forums/thread/718737 (not re-fetched)
- https://discuss.pytorch.org/t/float64-tensors-on-mps/224325 (not re-fetched)
- https://developer.apple.com/videos/play/wwdc2025/315/ (not re-fetched)
