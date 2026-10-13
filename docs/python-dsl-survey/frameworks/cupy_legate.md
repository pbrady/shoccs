# CuPy and cuPyNumeric (Legate) as a SHOCCS solver backend or DSL substrate

Survey with adversarial verification applied. Status date: 2026-10-12. Version checked: CuPy v14.2.0. cuPyNumeric 26.06.x is end of life.

## Verdict

| Item | Verdict |
|---|---|
| Fit for SHOCCS | Credible CUDA-only GPU execution layer; 49 of 51 catalog idioms map natively or via host-built tables. I35 (Lua per-point callbacks) and I50 (non-Hermitian sparse eigs) are unsupported on device. |
| CPU parity | Poor. CuPy has no CPU execution path of its own; host work stays in NumPy/SciPy/Numba. cuPyNumeric runs on CPU via Legate but is end of life. |
| Apple GPU | None. Install docs and release notes list only NVIDIA CUDA and experimental AMD ROCm; no macOS, Apple Silicon or Metal mention. |
| fp64 | Native on NVIDIA GPUs (rate depends on SKU) and via NumPy on host. None on Apple GPUs, and CuPy has no Apple path in any case. |
| Risk | High for a primary substrate. Vendor lock-in to CUDA, sibling project EOL, eager launch overhead at small sizes. Verifier-adjusted overall score: 3 / 10 (unchanged). |

## Status as of 2026-10-12

- CuPy v14.2.0 is the version checked (release notes via newreleases.io). Install docs list NVIDIA CUDA GPUs with Compute Capability 3.0+, CUDA 12.0 to 13.4, Python 3.11 to 3.15. Windows wheels are also supported. The release page gives no exact date. Whether a later release exists was not checked.
- v14.2.0 added SciPy-compatible sparse arrays: COO, CSR, CSC and DIA, all 2-D, with COO and CSR also 1-D, supporting both 32-bit and 64-bit indexing. The release notes advise new user code to use sparse arrays rather than matrices (confirmed).
- AMD ROCm 7.x is experimental. Binary wheels are unavailable from v13.4.0 onward (AMD hosts ROCm 6.4 wheels separately). The ROCm limitations list includes no cuTENSOR and no sparse matrix support, so the sparse-based mappings (I10, I20, I21) would not work on ROCm. This was added by the verifier and was missing from the original survey.
- cuPyNumeric is a Legate-based distributed NumPy. NVIDIA's 26.06 docs state "This project has reached end of life and is no longer maintained or supported" and "The final release is 26.06.01" (confirmed from the primary NVIDIA docs; no independent announcement was found, so it is single-source in that sense). The survey's claim that 26.06.00 is the newest release on the Anaconda legate channel is unverified and probably outdated, since the final release is 26.06.01.
- cuPyNumeric falls back to base NumPy for unsupported dimensions or dtypes and does not support non-unit steps in index expressions such as `arr[::2]` (confirmed).

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| CuPy on NVIDIA CUDA | Production (primary, effectively only first-class) | Native; throughput depends on SKU (consumer cards are slow in fp64, general knowledge, not verified) | `ElementwiseKernel`, `ReductionKernel`, `RawKernel`/`RawModule` (full CUDA C), `cupy.fuse` for elementwise fusion, `cupyx.scipy.sparse` SpMV on cuSPARSE. `RawKernel` ignores views, so the caller manages strides; `ElementwiseKernel` handles broadcasting (confirmed). |
| CuPy on AMD ROCm 7.x | Experimental | Where the device supports it | No prebuilt wheels from v13.4.0, no sparse support. |
| CuPy on CPU | None | n/a | Use NumPy, SciPy or Numba on host. |
| CuPy on Metal / Apple GPU | None | No (Apple GPUs lack fp64) | Would need MLX, PyTorch MPS or Metal compute and a rewrite of the kernel layer. |
| cuPyNumeric (Legate) | End of life, final release 26.06.01 | CPU and GPU via Legate | NumPy fallback for unsupported dims/dtypes; no non-unit index steps. Poor long-term substrate. |

## Idiom mapping

Support values: native, workaround, unsupported. "Setup only" means the work happens once on the host and does not affect step performance. Perf entries are estimates from the catalog baseline, not CuPy measurements.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | `cupy.zeros` per buffer or one packed (4, n) array; field container as a dict of cupy arrays. | Equal to C++ per buffer; packing cuts launches 4x. |
| I02 | workaround | Ray casting stays host NumPy (vectorised quadratic roots, lexsort), emitted as SoA arrays then `cupy.asarray`. | Setup only. |
| I03 | workaround | Fragmentation is host NumPy; runtime consumes flat (start, len, stride, closure_id) arrays on device. | Setup only. |
| I04 | native | Boolean masks as cupy bool arrays; int32 index arrays for gathers. | Mask form is coalesced; gather costs 0.5 to 8 ns/elem (C++ baseline). |
| I05 | native | Stencil constants as Python floats or small cupy arrays; RBF solve on host numpy. | Setup only. |
| I06 | workaround | Host NumPy/SymPy lambdify over the psi array, cast to float64 cupy array; CUDA device code only if evaluated per step. | Setup only. |
| I07 | workaround | Host per-pencil `searchsorted`; emit padded (idx, coef) tables. | Setup only. |
| I08 | workaround | Host NumPy assembly grouped by (left_kind, right_kind, bc); emit deduplicated closure table. | Setup only. |
| I09 | workaround | Host NumPy assembly to ELL tables (idx, coef) uploaded to device. | Setup only. |
| I10 | native | `cupyx.scipy.sparse.csr_matrix` or `coo_matrix` built on host with SciPy, then transferred; keep compact COO rather than full-height row pointers. Not available on ROCm. | Compact COO avoids the 134 MB int64 row-pointer sweep at 256^3. |
| I11 | native | One packed device array for coefficients plus int32 metadata arrays; no per-line allocations. | Removes about 700k tiny allocations at 256^3. |
| I12 | native | Setup in Python; coefficient fields as cupy arrays. | Setup only. |
| I13 | native | `reshape(nx,ny,nz)` of flat D is a view; strided slices for x/y/z lines. | Bandwidth-bound; x-lines (stride ny*nz) are the uncoalesced direction for a naive kernel. |
| I14 | workaround | Keep the (slow,fast,t) R ordering host-side; carry explicit permutation tables if one global R space is chosen. | Setup only; a correctness hazard, not a performance one. |
| I15 | native | Cupy arrays passed by reference; no span type (use views); ownership is device-side. | No overhead. |
| I16 | workaround | `RawKernel` with one thread per (line, row); per-row metadata in int32/int64 device arrays; interior rows via the axis stencil (I17); closure rows via the ELL gather (I19). | Needs a coalesced thread mapping across the fast index to avoid the C++ one-team-per-strided-line penalty; GPU-only. |
| I17 | native | Sum of shifted slices `c[j]*u[...]` over the grid, or fused elementwise via `cupy.fuse`; mask at boundaries. | Memory-bound: about one read and one write of D per axis; about 2 to 3 ms per axis at 256^3 at 100 GB/s (estimate). |
| I18 | native | Small dense matmul on boundary planes via `einsum` or `cupy.matmul` on plane slices. | Negligible. |
| I19 | workaround | ELL gather-dot-scatter (padded idx and coef) with `cupyx.scatter_add` over unique output rows. | Tiny relative to D; launch-bound at small sizes. |
| I20 | native | `cupyx.scipy.sparse` CSR SpMV (cuSPARSE) or ELL gather-dot; no raw host pointers in kernels. Not available on ROCm. | Bandwidth-bound; at most 0.083 flop/B. |
| I21 | native | Compact COO/CSR of size about R*abs(R) to D; `csr_matrix @ uR` accumulated into duD. | Removes the full-height sweep; launch-bound at small sizes. |
| I22 | native | Broadcast axpy on the X face slabs (`duD[plane] += extra[k]*g_face`); no-op without Neumann BC. | Trivial. |
| I23 | native | ELL gather-dot with cupy indexing; `cupyx.scatter_add` into duR. | Launch-bound at small sizes; gather-bound at 256^3. |
| I24 | native | Merge into the Bf ELL over a concatenated [D;R] source. | Removes the Bf to Br dependency. |
| I25 | native | Fuse zero-fill into the expression (`out = a*x + b*y` without a separate zero pass). | Disappears under fusion (`cupy.fuse` or `ElementwiseKernel`). |
| I26 | native | Fold scale into the operator output expression. | Trivial. |
| I27 | workaround | Functional expression `du = Ox+Oy+Oz+B+N` with explicit total semantics; separate kernels otherwise. | Python-side ordering is easy; fusing the three directions needs a `RawKernel`. |
| I28 | native | Return a tuple of three cupy arrays from independent derivative calls. | Equal to C++ per call; extra scratch unless fused into the consumer. |
| I29 | native | `cupy.fuse(lambda gx,dx,gy,dy,gz,dz: gx*dx+gy*dy+gz*dz)`. | One fused elementwise kernel per RHS. |
| I30 | native | `du[fluid_mask] += src[fluid_mask]` or dense masked FMA `du += mask*src`; `cupyx.scatter_add` for tiny R sets. | Dense masked form is coalesced and fusable, with 2 to 6% redundant work. |
| I31 | native | Slice assign `u[0,:,:] = value`; object Dirichlet via small gather/scatter. | Negligible. |
| I32 | workaround | No CUDA-graph convenience layer in the Python API as used here. `cupy.cuda.Graph` may exist in recent CuPy but this is unverified (not found in the docs fetched by the verifier) and unverified for this workload. `cupy.fuse` and `RawKernel` fusion are the main levers. | Unverified; eager per-op launches cost several us each, so small 2-D cases are launch-bound. |
| I33 | native | Eager Python functions; sync only where a host value is needed. | Launch-bound at 2.6k to 5k points. |
| I34 | native | Closed-form MMS as cupy array math over broadcast coordinate grids; separable time factor `cos(wt)` precomputed. | Fused elementwise; likely faster than the C++ host virtual calls. |
| I35 | unsupported | Lua per-point callback has no device equivalent; replace with array-expression MMS (SymPy-generated). | Blocker unless the MMS is rewritten as array math. |
| I36 | workaround | Fill the source on device as an expression inside the RHS rather than a host fill. | Removes host sync once I34 is array math. |
| I37 | native | `cp.fuse(lambda u0,a,b,c,d,dt: u0+dt/6*(a+2*b+2*c+d))` and stage updates as fused expressions. | Bandwidth-bound; about 27 field touches per step at 256^3 unless fused. |
| I38 | native | Buffer swap by reference is natural in Python (no pointer baking). | Free. |
| I39 | native | `where(dir_mask, sol_t, u)` on boundary sets only, evaluated on boundary points. | Removes the full-field host evaluation. |
| I40 | native | Hoist object lists to setup; device-resident masks. | Removes per-step upload and sync. |
| I41 | native | `cupy.where(mask, abs(u-sol), -inf).max()` and `argmax`; read back every K steps. | One extra read of D per step; the per-step device-to-host sync is the real cost. |
| I42 | native | Gate error computation on the dump interval; `cupy.asnumpy` at dump time; write from NumPy. | D2H copy only on dump steps. |
| I43 | native | Host Python loop around a device step. | Loop overhead negligible. |
| I44 | native | `scipy.sparse` assembly from the same tables and `numpy.linalg.eigvals` on host. | Diagnostic only. |
| I45 | native | NumPy one-liners on host. | Test only. |
| I46 | native | Cupy elementwise expressions and `ElementwiseKernel`; alias handled by `out=`. | Launch floor per op; bandwidth-bound at 1M+ points. |
| I47 | workaround | Pointwise physics via `cp.fuse`; 15 flux fields fused into derivative inputs to avoid 2 GB temporaries at 256^3. Batched operator needs a leading component axis. | Bandwidth-bound. |
| I48 | native | `cupyx.scipy.sparse` Kronecker and dense blocks on device; host generation via NumPy. | Diagnostic. |
| I49 | native | SymPy codegen retargeted to a NumPy/CuPy printer; lambdify over psi array on host then upload. | Setup only. |
| I50 | unsupported | Non-Hermitian sparse eigs and SVD pseudospectra stay on host SciPy in fp64; no CUDA equivalent is used here (cupyx has limited eig support, not verified in detail). | Diagnostic, host-bound. |
| I51 | native | In-process Python solver removes the subprocess boundary; no CuPy-specific work. | Removes per-point process launch. |

All 51 idiom ids (I01 to I51) are covered. Workaround set: I02, I03, I06 to I09, I14, I16, I19, I27, I32, I36, I47. Unsupported: I35, I50.

## Code sketches

Sketches come from the survey and were not executed as part of this write-up. Treat them as illustrative.

Per-line block matvec with strides and per-line dense closures (I16). `RawKernel` ignores array views, so strides are passed explicitly:

```python
import cupy as cp
rowmv = cp.RawKernel(r'''
extern "C" __global__
void rowmv(const double* x, double* y, const long* out_idx, const long* col0,
           const long* stride, const int* ncols, const long* coff, const double* C,
           int n_items) {
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  if (t >= n_items) return;
  double acc = 0.0;
  for (int j = 0; j < ncols[t]; ++j)
    acc += C[coff[t] + j] * x[col0[t] + j*stride[t]];
  y[out_idx[t]] = acc;
}
''', 'rowmv')
def apply_rows(x, y, items):
    n = items.n_items
    rowmv(((n + 255) // 256,), (256,),
          (x, y, items.out_idx, items.col0, items.stride,
           items.ncols, items.coff, items.C, n))
```

O/B CSR coupling to R points (I21, I23):

```python
import cupy as cp
import cupyx.scipy.sparse as sp
B = sp.csr_matrix((w_B, v_B, u_B), shape=(nD, nR))   # compact R -> D
Bf = sp.csr_matrix((w_Bf, v_Bf, u_Bf), shape=(nR, nD))
Br = sp.csr_matrix((w_Br, v_Br, u_Br), shape=(nR, nR))
def apply_coupling(uD, uR, duD, duR):
    duD += B @ uR                  # cuSPARSE SpMV, accumulate
    duR += Bf @ uD + Br @ uR       # both read the pre-update state
    return duD, duR
```

Gather/scatter boundary application (I19, I23, I31):

```python
import cupy as cp
import cupyx
def dirichlet_zero(du, face_idx):
    du[face_idx] = 0.0
def apply_bf_ell(uD, idx, w, out_idx, duR):
    # idx: (nrow, K) int32, padded entries point at a dummy slot with w = 0
    vals = (w * uD[idx]).sum(axis=1)
    cupyx.scatter_add(duR, out_idx, vals)   # out_idx unique: no atomics needed
def update_boundary(u, sol, t, face_idx, face_vals_fn):
    u[face_idx] = face_vals_fn(face_idx, t)  # boundary-only evaluation
```

RK4 with fused slot updates (I37, I38):

```python
import cupy as cp
@cp.fuse()
def stage(u0, k, a):
    return u0 + a * k
@cp.fuse()
def final(u0, k1, k2, k3, k4, dt):
    return u0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
def rk4_step(u0, rhs, dt, t):
    k1 = rhs(u0, t)
    k2 = rhs(stage(u0, k1, 0.5 * dt), t + 0.5 * dt)
    k3 = rhs(stage(u0, k2, 0.5 * dt), t + 0.5 * dt)
    k4 = rhs(stage(u0, k3, dt), t + dt)
    return final(u0, k1, k2, k3, k4, dt)
```

## Performance assessment

No CuPy measurements exist for this codebase. Every figure below is an estimate derived from the catalog's C++ baseline or from general knowledge, and none was verified.

- Small 2-D (51x51, 71x71): launch-bound. Eager CuPy costs several microseconds per operation (estimate) against about 0.8 us per C++ graph node. Roughly 300 eager ops per RK4 step would be several times slower than the C++ path. Fusion with `cupy.fuse` plus a few `RawKernel`s is needed to approach parity.
- Large 3-D (256^3): bandwidth-bound on NVIDIA GPUs. The catalog's lower bound is about 16 to 31 ms per RHS, on the order of 2 to 3 ms per axis at 100 GB/s. A GPU is likely faster than the CPU C++ path here, but this was not measured.
- Dispatch: several us per eager kernel launch (estimate). CUDA graph capture via `cupy.cuda.Graph` is unverified for this workload and for the API itself.
- Whole-RHS fusion: partial. `cupy.fuse` fuses elementwise expressions only. `RawKernel`, cuSPARSE SpMV and `scatter_add` each break a fusion region, so a whole RHS is not one compiled kernel without hand-written `RawKernel`s.
- Interop: arrays expose `__cuda_array_interface__` and DLPack, so GPU exchange with other frameworks is zero-copy on device. There is no zero-copy with NumPy because host and device memory are separate. Loading compiled cubin/PTX via `RawModule` and linking host C++ via Cython or pybind11 are general knowledge, not verified. cuPyNumeric shares the NumPy API but not CuPy's memory model.

## Suitability as a DSL substrate

Reasonable as a CUDA-only execution layer, weak as the only substrate.

Strengths:
- NumPy-compatible API, `cupy.fuse`, the `RawKernel` escape hatch and `cupyx.scipy.sparse` make it a credible target for a DSL that emits array code.
- The `ElementwiseKernel` pattern is a clean design reference for a fused-map primitive.
- The cut-cell unstructured parts (ragged fragments, per-line closures, R-point coupling) stay host-side in setup and map to flat ELL/COO tables on device, the same pattern recommended for any substrate.

Weaknesses:
- No CPU path of its own and no Apple path.
- Fusion is elementwise only; scatters, SpMV and raw kernels break it.
- cuPyNumeric is end of life and falls back to NumPy for unsupported types, so it does not rescue the CPU gap.

Recommended role: keep the DSL IR backend-neutral, with the CPU path in NumPy/Numba and a GPU path behind a seam that could be CuPy or another framework.

## Risks

1. No Metal or Apple GPU backend, so the Apple nice-to-have is not reachable through CuPy.
2. cuPyNumeric is end of life after 26.06.01 (confirmed from NVIDIA's docs), so the Legate route is not viable long term.
3. No fp64 on Apple GPUs; any MPS or MLX port would run fp32, and small-psi cut-cell closures may lose accuracy.
4. Small 2-D cases are launch-bound in eager CuPy; without fusion or CUDA graphs the performance bar is missed, and CUDA graph support for this workload is unverified.
5. Sparse support is narrower than the original survey implied. The int32-only restriction applies to `spsolve` and `spilu` (and `spsolve_triangular` on CUDA before 12), not to SpMV or matmul, and v14.2.0 supports 64-bit indexing. The real sparse risk is that ROCm has no sparse support.
6. NVIDIA vendor lock-in for the GPU path; ROCm wheels are not shipped from v13.4.0 and ROCm is experimental.
7. All performance statements are estimates; none are CuPy measurements for this codebase.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| CuPy supports NVIDIA CUDA primary and ROCm experimental; install docs do not mention macOS, Apple Silicon or Metal | Confirmed | Install page lists CC 3.0+, CUDA 12.0 to 13.4, Python 3.11 to 3.15, ROCm 7.x experimental, macOS not mentioned. |
| `RawKernel` ignores views so the caller manages strides; `ElementwiseKernel` handles broadcasting | Confirmed | Kernel guide: `RawKernel` "ignores any views on CuPy arrays"; `ElementwiseKernel` does indexing with broadcasting automatically. |
| cuPyNumeric falls back to NumPy for unsupported dims/dtypes and does not support non-unit steps | Confirmed | Quoted from the NVIDIA 26.06 differences page. |
| cuPyNumeric 26.06 docs state end of life, final release 26.06.01 | Confirmed (survey rated it low confidence; raised) | Direct quotes from NVIDIA's docs. No independent announcement found by a generic search. |
| CuPy v14.2.0 added sparse arrays supporting COO, CSR, CSC, DIA | Confirmed | Release notes; 32-bit and 64-bit indexing supported; no exact release date on the page. |
| Some `cupyx.scipy.sparse` operations are int32-only and raise ValueError on int64 | Confirmed, but narrowed | Only `spsolve` and `spilu` (and `spsolve_triangular` on CUDA before 12) per the reference page. SpMV and matmul are not listed. The survey overstated the risk; corrected above. |
| cuPyNumeric 26.06.00 is the newest version on the Anaconda legate channel | Uncertain, probably wrong | Search snippet showed 26.06.00 and 26.01.00, but NVIDIA says the final release is 26.06.01. Anaconda page not fetched. Dropped from the status section. |
| ROCm wheels unavailable in v13.4+, ROCm 7.x experimental | Confirmed | Install page. It also lists no sparse matrix support on ROCm (added by verifier). |
| `cupy.cuda.Graph` exists in recent CuPy | Uncertain | Not mentioned in the kernel guide; API not checked. Marked unverified in I32 and Performance. |
| CuPy has no Apple GPU path | Confirmed | Follows from the install docs and release notes naming only CUDA and ROCm. |
| Dispatch cost "several us" per eager op; 16 to 31 ms per RHS at 256^3 | Unverified | Estimates from the catalog baseline, not measured in CuPy. |
| Compute Capability, Python and CUDA version ranges | Confirmed | Match the install page; no hallucinated versions found. |

Score change: none. Overall remains 3 / 10; corrections are minor (EOL better supported, sparse int64 risk narrower) and do not change the conclusion that CuPy meets neither the CPU-parity nor the Apple GPU goal. Original sub-scores: cpu_parity 2, apple_gpu 0, idiom_coverage 6, ergonomics 6, long_term_risk 4.

## Sources

- https://docs.cupy.dev/en/latest/install.html
- https://docs.cupy.dev/en/latest/user_guide/kernel.html
- https://docs.cupy.dev/en/latest/reference/scipy_sparse.html
- https://docs.nvidia.com/cupynumeric/26.06/user/differences.html
- https://newreleases.io/project/pypi/cupy/release/14.2.0
- https://rocm.blogs.amd.com/artificial-intelligence/cupy-v13/README.html
- Cited by the survey but not fetched by the verifier: https://anaconda.org/legate/cupynumeric/manage, https://docs.cupy.dev/en/v6.7.0/tutorial/kernel.html
