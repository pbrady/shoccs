# NumPy/SciPy Baseline (Assembled scipy.sparse CSR Operator)

| Verdict box | |
|---|---|
| Fit for SHOCCS | Good as a correctness oracle, prototype and index-table producer. Poor as the performance layer at large 3D. Idiom coverage is high (only I32 is unsupported). |
| CPU parity | Plausible at small 2D (estimate, unmeasured). Large 3D is likely 3-8x slower single-threaded (estimate, unmeasured). Overall score 5/10. |
| Apple GPU | None. SciPy is CPU only and has no MPS path. Score 0. |
| fp64 | Full fp64 on CPU. Metal/MPS has no fp64 (confirmed), so any GPU variant would be fp32 or emulated. |
| Risk | Very low long-term risk (mature library). Main risks are CSR bandwidth, serial SpMV, no fusion, and unmeasured performance claims. |

## Status as of 2026-10-12

- SciPy 1.18.0 was released 2026-06-19 (requires Python 3.12-3.14, NumPy >= 2.0). Correction from verification: 1.18.1, a bug-fix release, now exists, so 1.18.1 is the current release. The 1.18 notes section on sparse was not read.
- scipy.sparse offers csr_array and csr_matrix. The csr_matvec kernel is believed to be serial C++ with no GPU and no JIT. This is unverified for 1.18: the only evidence is a 2020 sparse_dot_mkl README.
- Role: very mature and stable. Fits small 2D as a production substrate. Large 3D needs a threaded or compiled kernel in addition.

## Backends and fp64

| Backend | Availability | fp64 | Notes |
|---|---|---|---|
| CPU (SciPy csr_matvec) | Native | Yes | Believed single-threaded (unverified for 1.18). |
| CPU multithreaded, MKL (sparse_dot_mkl) | x86 | Yes | Accepts CSR, CSC or BSR (corrected; the survey said CSR/CSC only). Unavailability on Apple Silicon is an unverified inference (package mentions a macOS .dylib loader; MKL has no known arm64 macOS build). |
| CPU multithreaded, Apple Accelerate Sparse | Apple CPU | Yes (double SparseMultiply exists) | Documented multithreaded by default; the threading text concerns the solvers, so SpMV threading is unconfirmed. Needs ctypes or a C shim. |
| CPU multithreaded, Numba prange / OpenMP CSR kernel | Any | Yes | Hand-written, not part of SciPy. |
| CUDA (CuPy cupyx.scipy.sparse) | Separate package | Yes | Near-identical API, cuSPARSE SpMV. Version not verified. |
| Metal / MPS | None | No | No SciPy binding. Metal lacks fp64 (PyTorch raises TypeError for float64 on MPS; Apple DTS asked for an enhancement request). The Apple Metal sparse BLAS claim was not checked. |

## Idiom mapping

Support values: native, workaround, unsupported. "Setup" means not on the hot path.

| Id | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | One packed ndarray u of length D+Rx+Ry+Rz with static offsets and views. | Equal to C++. Slot ops are one numpy call instead of four. |
| I02 | native | Vectorised ray casting, lexsort for (slow,fast,t) order, nextafter and 1e-12 snap/clamp reproduced. SoA outputs. | Setup only. Likely faster than the C++ virtual-call loop. |
| I03 | native | Fragment table via searchsorted or cumsum merge-join per pencil. | Setup, negligible. |
| I04 | native | Boolean masks or flatnonzero index arrays. Faces are slices of the 3D view. | Setup. Mask cheaper than an int32 gather list. |
| I05 | native | Python constants and fractions, np.linalg.solve for the RBF 10x10 system. | Setup. |
| I06 | native | sympy.lambdify (cse=True) or a retargeted CSE printer over a psi array. Re-add the psi clamps the generator lacks. | Setup, much faster than per-entry xreplace. |
| I07 | workaround | Host NumPy/Numba per-pencil searchsorted lookup, np.where for interior vs wall, COO output. | Setup. Hardest setup code but ports like host code. |
| I08 | workaround | Group lines by (left_kind, right_kind, bc), emit COO triplets in bulk. No inner_block_meta needed in a global CSR. | Setup. |
| I09 | workaround | Vectorised COO assembly of Bf/Br rows; composition via einsum or repeat. | Setup. |
| I10 | native | coo_array((w,(r,c))).tocsr() sums duplicates. Use int32 indices. | Setup. |
| I11 | native | Not needed; CSR is the flattened form. No coefficient dedup, so interior taps are stored per row. | Setup. Memory larger than C++ closure tables. |
| I12 | native | Python dicts or dataclasses replace Lua parsing. | Setup. |
| I13 | native | Free C-order reshape views. Flat index i*ny*nz+j*nz+k goes into CSR columns. | Equal to C++. |
| I14 | native | Preserve (slow,fast,t) lexsort. Concatenated index is D + offset_dir + i. | Setup. |
| I15 | native | Plain ndarrays with out= and preallocated scratch. | Equal with preallocated buffers. |
| I16 | workaround | Per-line block matvec becomes rows of one CSR over the concatenated vector, y = A @ u. Alternative: structured slicing for the interior plus CSR for closures. | Small 2D: roughly equal (estimate 10-20 us for 13k nnz). Large 3D: estimated 2-5x slower on bandwidth; serial SciPy a further 4-8x slower than threaded C++ (estimates). |
| I17 | workaround | CSR rows, or structured slices y[1:-1] = c0*x[:-2] + ... with out= chaining. | CSR estimated 3-5x the traffic of C++. Slices estimated 2-3x C++ for E2 (no fusion, single thread). |
| I18 | native | CSR rows, or einsum on first and last planes. | Same as C++ (tiny work). |
| I19 | native | CSR rows with psi-dependent coefficients. | Free (0.5-2.5% of rows). |
| I20 | native | A @ u. Private csr_matvec in _sparsetools accumulates into y (signature unverified). | Estimated 1-1.5 GFLOP/s, single thread. int32 row pointers use 4 B/row vs 8 B for C++ int64. |
| I21 | native | Rows of the concatenated CSR with column in R range. | Better than C++ (no full-height sweep). |
| I22 | native | Part of A if Neumann data is state, else slab axpy. | Better than C++. |
| I23 | native | Rows with row index in R range, column in D. | About equal to C++. |
| I24 | native | R-to-R columns in the same CSR block, no dependency chain. | Equal or better. |
| I25 | native | Not needed; y = A @ u writes everything. Mask for stale-value semantics. | Better than C++ (one pass saved). |
| I26 | native | Fold scale into A.data at setup, else np.multiply(y,k,out=y). | Free if folded. |
| I27 | native | One matrix holds dx+dy+dz and B/N/Bf/Br terms. | One SpMV vs 28-31 kernels. Better at small sizes; nnz is the sum over axes at large sizes. |
| I28 | native | Gx, Gy, Gz or one stacked 3N x N sparse matrix. | About equal for same nnz. Large sizes pay scratch memory; numpy cannot fuse. |
| I29 | native | np.multiply/np.add with out=, or einsum on stacked gradient. | Small: equal or better. Large 3D: 3-6 passes, estimated about 2x slower. |
| I30 | native | np.add(du, src, out=du, where=fluid_mask) or np.where. | Small: equal. Large: bandwidth bound, similar to C++. |
| I31 | native | Face slices and u[idx]=v gather lists built once. | Equal (negligible). |
| I32 | unsupported | No graph capture, JIT or fusion. Fallback is a hand-written function with a minimal number of calls. | Small 2D acceptable (1-5 us per numpy call). Large sizes lose from lack of fusion. |
| I33 | native | Plain Python; fences are meaningless. | Equal. |
| I34 | native | Vectorised NumPy over broadcast coordinate grids, restricted to boundary and source sets. lambdify output works. | Faster than C++ virtual calls at small sizes. At 256^3 restricted evaluation is required. |
| I35 | native | Python callable on arrays replaces the Lua callback. | Much faster than C++/Lua. |
| I36 | native | t is an ordinary float argument. | Equal or better. |
| I37 | native | In-place stage updates with out=. About 12-20 numpy calls per RK4 step. | Small: estimated 20-40 us per step. Large: no fusion, separate passes (estimated 1-2x C++ bandwidth cost). |
| I38 | native | np.copyto or swap references. | Better than C++. |
| I39 | native | Evaluate exact solution only at boundary points via precomputed indices. | Better than C++. |
| I40 | native | Hoisted to setup with persistent masks. | Better than C++. |
| I41 | native | Masked reductions, np.max(np.abs(...)), argmax. | Small: equal. Large: 2-3 passes, estimated about 2x slower than fused C++. |
| I42 | native | u.astype('<f8').tofile(f) matches the existing D,Rx,Ry,Rz file layout. XDMF via ElementTree. | Equal or better. |
| I43 | native | Python while-loop with dispatch dict or classes. | Negligible per-step overhead. |
| I44 | native | A.toarray() plus np.linalg.eigvals, or sparse.linalg.eigs. | Equal to the existing LAPACK path. Strongest use: assemble the real cut-cell operator for stability analysis. |
| I45 | native | np.dot and np.convolve one-liners. | Equal. |
| I46 | native | NumPy expressions with out=. No expression-template fusion. | Estimated 1-3x C++ at 1M doubles and above. |
| I47 | workaround | State as (5,N). Flux derivative as sparse @ dense multi-RHS. Pointwise physics in numpy. | Small 2D acceptable. Large 3D poor (about 2 GB flux temporaries, estimate). |
| I48 | native | What the existing Python code already does; replace its loops with vectorised COO assembly. | Equal. |
| I49 | native | Retarget CSE printer to NumPy or use lambdify. | Setup. |
| I50 | native | scipy.sparse.linalg.eigs/svds and np.linalg on the assembled operator. | Equal. Reference engine for stability analysis. |
| I51 | native | In-process Python solver called directly from botorch. | Removes per-point C++ process startup. |

## Code sketches

Per-line block matvec with strides and per-line dense closures, assembled once as CSR (I16-I19):

```python
import numpy as np, scipy.sparse as sp
# setup: rows/cols/w are accumulated per fragment line, global flat indices
def assemble_dir(lines, taps, wall_cl, psi_cl, stride, N):
    R, C, W = [], [], []
    for ln in lines:              # ln: start_ic, len, left, right (kind, closure, R index)
        ic = ln.start + stride*np.arange(ln.len)
        L, Rr = ln.left.rows, ln.right.rows
        for r in range(L):                      # left closure rows, dense LxT
            R += [ic[r]]*T; C += list(ic[:T]); W += list(ln.left.block[r])
        for r in range(L, ln.len-Rr):           # interior taps shared by every line
            R += [ic[r]]*len(taps); C += list(ic[r-P:r+P+1]); W += list(taps)
        for r in range(Rr):                     # right closure rows, already flipped
            R += [ic[-1-r]]*T; C += list(ic[-T:]); W += list(ln.right.block[r])
    return sp.coo_array((W,(R,C)), shape=(N,N)).tocsr()
# vectorise by grouping lines with equal (kinds, len); interior part via np.add.outer
A = Ax + Ay + Az                  # one matrix, duplicates summed; apply: y = A @ u
```

Cut-cell coupling to R points as blocks of one global matrix (I21, I23, I24):

```python
import scipy.sparse as sp
D, nRx, nRy, nRz = ...; off = {'D':0,'x':D,'y':D+nRx,'z':D+nRx+nRy}
# B: R_dir column -> fluid rows next to the cut (col = R index + offset)
rB = np.repeat(rows_next_to_cut, 1); cB = off[dir] + obj_coord; wB = w_B
# Bf: D -> R(r) rows, Br: R(r)->R(r), including r != dir derivative-of-interpolant columns
rBf = off[r] + q_idx; cBf = d_cols; wBf = w_Bf       # up to 12-16 nnz/row
rBr = off[r] + q_idx; cBr = off[r] + q_cols; wBr = w_Br
rows = np.r_[rO, rB, rBf, rBr]; cols = np.r_[cO, cB, cBf, cBr]; w = np.r_[wO, wB, wBf, wBr]
A = sp.coo_array((w, (rows, cols)), shape=(N, N)).tocsr()   # duplicates summed
y = A @ u                       # D rows and R rows in a single SpMV
```

Gather/scatter boundary application with masks and index arrays (I30, I31, I39):

```python
u3 = u[:D].reshape(nx, ny, nz)           # free view
du3 = du[:D].reshape(nx, ny, nz)
# Dirichlet face fill (affine slices) and object-point fill (gather list built once)
u3[0, :, :]  = sol_face(0, t);  u3[-1, :, :] = sol_face(-1, t)
u[dir_obj_idx] = sol_at(pos[dir_obj_idx], t)
# masked source add, fluid_mask is bool over D (setup), no per-call index rebuild
np.add(du[:D], src[:D], out=du[:D], where=fluid_mask)
np.add(du[D:], src[D:], out=du[D:], where=nondir_R_mask)
# zero at Dirichlet points of the RHS
du[dir_all_idx] = 0.0
```

Scalar algebra with RK4 slot updates on one packed array (I37, I38):

```python
# u packs D|Rx|Ry|Rz, so each slot op is ONE numpy call instead of 4
def rk4_step(u0, t, dt, rhs, bc, work):
    u1, k, acc = work.u1, work.k, work.acc
    np.copyto(u1, u0); acc[:] = 0.0
    for c_assign, c_acc, tt in ((0.0, dt/6, t), (dt/2, dt/3, t+dt/2),
                                (dt/2, dt/3, t+dt/2), (dt, dt/6, t+dt)):
        if c_assign:
            np.multiply(k, c_assign, out=u1); u1 += u0
            bc(u1, tt)                      # boundary points only
        rhs(u1, tt, out=k)                  # k = A @ u1 + src(tt), masked
        acc += c_acc * k                    # allocates one temp; use axpy for none
    np.add(u0, acc, out=u1); bc(u1, t+dt)
    return u1                               # swap references; no deep_copy needed
# the loop has about 20 numpy calls, then 4 SpMVs
```

## Performance assessment

All numbers below are estimates; none were measured. The C++ baseline is itself stale and noisy (Linux aarch64 container numbers).

- Small 2D: a 2601-row, roughly 13k-nnz SpMV is estimated at 10-20 us plus 5-10 us of Python/numpy overhead, against roughly 30-40 us for the C++ graph path. An RK4 step (about 4 SpMVs and 20-30 elementwise calls) is estimated at 150-300 us versus an unmeasured C++ estimate of 0.3-1 ms. The largest single win is vectorised MMS in place of 37k virtual calls. Verdict: unverified, plausible.
- Large 3D: single-threaded csr_matvec at an estimated 1-1.5 GFLOP/s gives roughly 0.1-0.25 s per RHS for a 256^3 E2 Laplacian (about 117M nnz) against the 30-110 ms C++ bar, a 3-8x deficit. CSR holds about 12 B/nnz, so about 1.4 GB per operator versus 22 MB of C++ closure tables. The 3-5x bytes-per-point penalty is arithmetic (12 B/nnz x about 7 nnz/row vs 16-24 B/pt structured), not measured. A threaded SpMV (Accelerate, OpenMP) recovers part of the gap but not the traffic difference. Structured slicing cuts memory but does not fuse (estimated 2-3x C++).
- Dispatch overhead: about 1-5 us per numpy call and 5-10 us of scipy wrapper overhead on A @ u (estimates). After assembly the RHS is a handful of calls, far below eager PyTorch/MPS (5-20 us per op x about 300 ops).
- Fusion and JIT: none in NumPy/SciPy. Assembly into one CSR is a partial substitute for the operator part only. Elementwise passes stay separate. Fusion needs Numba, JAX, torch.compile or MLX. No warmup cost; vectorised assembly takes seconds at 256^3 (estimate).

## Suitability as a DSL substrate

Good as the reference layer under a DSL, poor as the performance layer. A "materialise as scipy.sparse" backend would serve as the correctness oracle for other backends, enable stability and eigenvalue analysis of the real cut-cell operator, and produce the index tables every other backend consumes. A single global CSR absorbs all six cut-cell irregularities (fragments, R buffers, psi closures, B/Bf/Br coupling, gather lists, partial writes) into matrix rows. It hides structure, which costs bandwidth at large 3D. A DSL targeting the performance bar must also lower the regular interior to a compiled structured stencil kernel and keep only closure and R rows sparse. Interop is good: arrays are zero-copy ndarrays, shareable via pybind11/nanobind with C++/Kokkos kernels, exportable to Accelerate, MKL or cuSPARSE, and the packed D|Rx|Ry|Rz layout matches the existing binary dump.

## Risks

- Large 3D bandwidth: CSR moves an estimated 3-5x more bytes per point than a structured stencil.
- SciPy SpMV is believed serial. Threaded alternatives are platform dependent (Accelerate needs ctypes or a shim; sparse_dot_mkl on Apple Silicon unverified; pyamg adds no threading).
- Memory: about 1.4 GB per 3D E2 operator at 256^3, more for E4/E6/E8 or multi-component Euler.
- No fusion; elementwise passes add up.
- No GPU path, none via MPS; Metal lacks fp64.
- Small-size parity claims are estimates.
- Assembly hazards (R ordering, duplicate entries, right-end flip) cause silent corruption; check against the C++ operator.
- The t-laplacian failure on R-point values means the C++ reference for the Bf/Br path is not yet trustworthy.
- scipy.sparse._sparsetools.csr_matvec is a private API with unverified signature.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| SciPy 1.18.0 (2026-06-19) is the current release; Python 3.12-3.14, NumPy >= 2.0 | Refuted in part | Date and requirements confirmed. 1.18.1 bug-fix release exists, so 1.18.0 is not current. |
| csr_matvec is single-threaded in 2026; threaded sparse comes from MKL | Uncertain | Only a 2020 sparse_dot_mkl README; 1.18 not checked. |
| sparse_dot_mkl is CSR/CSC and float only, unavailable on Apple Silicon | Uncertain | Newer docs list CSR, CSC or BSR. PyPI lists a macOS .dylib loader; arm64 unavailability unverified. |
| Accelerate Sparse supports double and is multithreaded by default | Confirmed (partly) | Apple docs: multithreaded by default, VECLIB_MAXIMUM_THREADS=1 forces single; double SparseMultiply exists. Threading text concerns solvers; SpMV threading unconfirmed. |
| Metal/MPS has no fp64 | Confirmed | PyTorch forum (float64 on MPS TypeError), Apple developer forum threads. Later changes not checked. |
| CSR needs about 3-5x more bytes per point than structured stencil | Uncertain | Arithmetic estimate only. |
| Small 2D RK4 step 150-300 us comparable to C++ 0.3-1 ms | Uncertain | Self-labelled estimate, no measurement. |

## Sources

- https://docs.scipy.org/doc/scipy/release/1.18.0-notes.html
- https://zenodo.org/records/20764140
- https://discuss.scientific-python.org/t/ann-scipy-1-18-1-release/2554
- https://sourceforge.net/projects/scipy.mirror/files/v1.18.0/
- https://pypi.org/project/sparse-dot-mkl
- https://developer.apple.com/documentation/accelerate/sparse-solvers-library.md
- https://developer.apple.com/documentation/accelerate/sparsemultiply(_:_:_:_:)-76o5l.md
- https://developer.apple.com/documentation/accelerate/blas-library.md
- https://discuss.pytorch.org/t/float64-tensors-on-mps/224325
- https://developer.apple.com/forums/thread/709663
- https://www.developer.apple.com/forums/thread/797778
