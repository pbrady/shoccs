# Numba as a SHOCCS solver substrate

Status as of 2026-10-12. Version checked: Numba 0.67.0 (released 11 Aug 2026). Companion: numba-cuda 0.30.4 (3 Jul 2026).

## Verdict box

| Item | Verdict |
|---|---|
| Fit for SHOCCS | Strong as the CPU execution layer of a multi-backend DSL; not sufficient as the single substrate because it has no Apple GPU path. |
| CPU parity | Expected parity with C++/Kokkos, with possible gains after loop restructuring. All performance figures are unmeasured estimates (unverified). |
| Apple GPU | None found. No Metal/MPS backend; "no official plan" is unverified (absence of evidence, tracker not read). numba-cuda ships no macOS wheels. |
| fp64 | Full fp64 on CPU and on CUDA. Not applicable on Metal (no backend; Apple GPU hardware has no fp64 anyway). |
| Risk | Moderate: JIT compile time for large generated code, no automatic SIMD on strided axes, no SVML, small-team project, CUDA split into a separate package. |

Verifier-adjusted overall score: 6.5 / 10 (survey said 7).

## Status as of 2026-10-12

- Numba 0.67.0 released 11 Aug 2026. Adds NumPy 2.5 support and Windows ARM64 at Tier 1.5. The OpenMP threading layer now uses omp_set_max_active_levels when the runtime provides it, falling back to omp_set_nested.
- np.linalg.eig and eigvals now return complex results for real input (matches NumPy 2.5). Relevant to idiom I44.
- Numba 0.66.0 (30 Jun 2026) moved to llvmlite 0.48 / LLVM 22, added numba.typed.Set, and states Intel SVML remains unsupported (lost in 0.62.0).
- Python 3.14 free-threading builds were listed in the 0.66 notes as an experimental Tier 1.5 configuration. The 0.67 notes do not mention free-threading. The general Python support matrix was not read from the notes page, so "Python 3.14 supported" is only indirectly supported.
- Release cadence is roughly quarterly (0.65.1 in April, 0.66 in June, 0.67 in August).
- The in-tree CUDA target is deprecated in favour of NVIDIA's numba-cuda, which keeps the numba.cuda namespace.
- I did not check whether a release newer than 0.67.0 exists.

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| CPU (LLVM via llvmlite) | Primary, strong | Full (fp32 also) | @njit, fastmath, parallel=True with prange, cache=True. Threading layers tbb, omp, workqueue. |
| CUDA | Separate numba-cuda package (0.30.4) | Full, rate GPU dependent | Wheels: Windows x86-64, Linux x86-64, Linux ARM64; cu12 and cu13 extras; no macOS. Explicit @cuda.jit kernels with per-thread indexing. |
| Metal / MPS | None found (unverified absence) | N/A; Apple GPU lacks fp64 | Would need an MSL or AIR target plus a Metal runtime layer. One unverified third-party PyPI package claims a CUDA-to-Metal mapping; not evaluated, not recommended. |
| ROCm / HSA | Removed in earlier versions | N/A | Not relevant. |
| numba-dpex (Intel) | Separate project | N/A | Not relevant on macOS. |

## Idiom mapping

Support: native = expressible directly; workaround = needs restructuring. Performance statements are estimates unless marked otherwise.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | Tuple or list of ndarrays per field, or one packed array D\|Rx\|Ry\|Rz with static offsets; tuples of arrays reflect without copying. | Same as C++ (setup only). |
| I02 | native | Ray casting as @njit loop over pencils and shapes; np.nextafter as ufunc; 1e-12 snap/clamp; SoA outputs. | At least C++ parity (setup). |
| I03 | native | @njit merge-join over R records per pencil into preallocated int64 arrays. | C++ parity (setup). |
| I04 | native | NumPy bool masks or int32 index arrays; hoist object-BC lists to setup. | Negligible. |
| I05 | native | Plain Python/NumPy constants; np.linalg.solve for RBF; interior taps passed as tuple or closure constants. | Setup only. |
| I06 | native | Retarget SymPy CSE printer to Python source, njit it, vectorize over psi (or lambdify). Keep fastmath off or limited near poles at psi=0. | Setup only; likely faster than C++ pow(x,-1) path. |
| I07 | workaround | @njit with data-dependent branches; np.searchsorted for fragment lookup; padded ELL for variable footprints. | Setup only. |
| I08 | native | @njit loop over lines branching on wall/object and BC type; writes line table and deduplicated closure table. | Setup; likely faster than C++ virtual calls. |
| I09 | native | @njit loop over R points emitting ELL tables with derivative and interpolation rows inline. | Setup only. |
| I10 | native | np.lexsort, np.bincount, np.cumsum for row pointers; or compact COO/ELL; scipy.sparse for diagnostics. | Setup only. |
| I11 | native | np.cumsum plus np.concatenate with a deduplicated closure table and per-line closure id. | Setup only. |
| I12 | native | Python dicts or dataclasses replace Lua; convert to arrays/tuples before jitted code. | Setup only. |
| I13 | native | C-contiguous arrays; u.reshape(nx,ny,nz) is a free view; explicit strides in njit. | Same as C++. |
| I14 | native | Keep (slow,fast,t) R ordering in SoA arrays; debug assert in tests. | No runtime cost. |
| I15 | native | ndarrays reflect as pointer+shape+strides; preallocate work buffers and pass them in. | Same as C++ spans. |
| I16 | native | Central case: @njit(parallel=True, fastmath=True) with prange over the line table; per line left dense, interior stencil, right dense. For large 3D, prange over slow pencil index with unit-stride inner loop chosen per direction. | Host parity expected; inner loops vectorize only at stride 1; strided x-lines need restructuring for SIMD. Unmeasured. |
| I17 | native | Constant taps via tuple or generated source; vectorized by LLVM at stride 1; fastmath allows reassociation. | Parity with C++ -O3 expected; memory bound at large sizes. Unmeasured. |
| I18 | native | Small fixed R x T loops dispatched on closure id. | Parity. |
| I19 | native | Loop over object endpoints using ELL tables or per-line closure block. | Parity; tiny relative to D. |
| I20 | native | Hand-written CSR loop in njit; scipy.sparse data/indices/indptr passed as raw arrays. | Parity; faster with compact tables. |
| I21 | native | Compact COO loop du[row[k]] += w[k]*uR[col[k]] over about R*\|R\| entries; drop full-height row pointers. | Faster than full-height sweep at large N; parity at small N. |
| I22 | native | Face-slab loops only when a Neumann BC exists (build-time flag). | Faster than C++ empty sweep. |
| I23 | native | ELL loop over R rows with padded column index; gather-dot into du.R; serial or prange (rows unique, no atomics). | Parity. |
| I24 | native | Fuse into the Bf loop, reading D and R sources by index offset. | Parity or better. |
| I25 | native | Fuse zero-fill into first writer, or plain assignment; explicit du[:] = 0 where needed. | Better than C++ when fused. |
| I26 | native | Fold diffusivity multiply into operator output loop. | Saves one pass. |
| I27 | native | Sequential loops in one jitted function; du += directly; optional single fused pass over 3 directions. | Parity; fusion may beat C++. |
| I28 | native | Three line-kernel calls into three outputs, prange over line table each. | Parity. |
| I29 | native | One fused loop over i computing gx*dux + gy*duy + gz*duz. | Bandwidth bound; parity. |
| I30 | native | Loop over mask or index list: du[idx[i]] += src[idx[i]]. | Parity; gather about 30% more traffic (survey estimate). |
| I31 | native | Slice assignment on 3D view or njit loops over faces and gather lists. | Negligible cost. |
| I32 | workaround | No graph object and none needed: whole RHS or RK4 step as one @njit function calling other njit functions; LLVM inlines; Python loop calls it, or time loop inside njit. | Estimated better than C++ at small 2D (single dispatch). Unverified. |
| I33 | native | Plain Python calling njit kernels; no fences needed on CPU. | Roughly 0.2-1 us per call (unverified estimate); keep hot paths in one jitted function. |
| I34 | native | MMS as scalar njit code (Gaussian sums with exp, cos) in a parallel loop, or @vectorize(target='parallel'); SymPy generates derivatives. | Likely at or above C++; exp/cos use scalar libm because SVML is unsupported. |
| I35 | workaround | Replace Lua callback with njit-compiled callable or SymPy-generated code. A Python callback via objmode is possible but slow. | Equal to array path if njit-able; objmode callback far slower (unverified), keep out of hot path. |
| I36 | native | Pass t as scalar into the single jitted RHS; source evaluates in-jit, no host sync. | Faster than C++ (no fence or vector round trips). |
| I37 | native | Fused loops for stage update, BC, and accumulate. | Parity per pass; fewer touches per step. |
| I38 | native | Rotate Python references or slice-copy; no pointer baking. | Free if swapped. |
| I39 | native | Evaluate MMS only at boundary indices in an njit loop over faces and object lists. | Large win over full-field evaluation. |
| I40 | native | Hoist to setup as plain arrays. | Removes per-call cost. |
| I41 | native | njit loop with running max/min/argmax over mask; prange reductions recognise +=, max, min. | Parity; no device sync. |
| I42 | native | Compute error in-jit only when dump due; arr.tofile(); lxml or ElementTree for XDMF. | I/O dominated. |
| I43 | native | Python while loop calling jitted step, or whole loop inside njit. | Negligible overhead at 50-2650 steps. |
| I44 | native | Assemble operator via scipy.sparse or dense, then np.linalg.eigvals or scipy. Note 0.67 eig returns complex for real input. | Same LAPACK. |
| I45 | native | NumPy one-liners. | Not hot. |
| I46 | native | Array expressions in njit; with parallel=True, adjacent elementwise array expressions are fused; or explicit loops. | Parity with C++ expression templates expected. |
| I47 | native | Pointwise physics as njit scalar functions in one loop over points; leading component axis or tuple of arrays. | Parity or better (avoids flux temporaries). |
| I48 | native | NumPy and scipy; njit if Python loops matter. | Not hot. |
| I49 | native | Unchanged SymPy; retarget printer to Python source and exec into njit function. | Not hot. |
| I50 | native | Unchanged scipy host code. | Not hot. |
| I51 | native | In-process: call the jitted step directly, removing subprocess loop; cache=True. | Removes per-design-point process and Kokkos setup; JIT warmup once unless cached. |

Idioms left out: none on the CPU. The only gap is the Apple GPU target itself, which is not an idiom but a backend. I35 (true per-point Python callback) is effectively out of the hot path.

## Code sketches

These are survey sketches and have not been compiled or run. The RK4 sketch indexes heterogeneous tuples with a runtime index and leaves update_boundary and compute_rhs undefined, so it will need restructuring (for example unrolling the stage loop in generated source) before it types.

Per-line block matvec (I16/I17/I18/I19):

```python
@njit(parallel=True, fastmath=True, cache=True)
def block_apply(meta, coef, c, x, b, accumulate):
    for l in prange(meta.shape[0]):
        ro, co, st = meta[l,0], meta[l,1], meta[l,2]
        lr, lc, lo = meta[l,3], meta[l,4], meta[l,5]
        ir, io_ = meta[l,6], meta[l,7]
        rr, rc, ro2, rco = meta[l,9], meta[l,10], meta[l,11], meta[l,12]
        n = lr + ir + rr
        for r in range(n):
            s = 0.0
            if r < lr:
                for j in range(lc):
                    s += coef[lo + r*lc + j] * x[co + j*st]
            elif r < lr + ir:
                o = ro + r*st
                for j in range(3):  # taps specialised via generated source
                    s += c[j] * x[o + (j-1)*st]
            else:
                rl = r - lr - ir
                for j in range(rc):
                    s += coef[ro2 + rl*rc + j] * x[rco + j*st]
            if accumulate: b[ro + r*st] += s
            else:          b[ro + r*st] = s
```

Cut-cell coupling to R points (I21/I23/I24):

```python
@njit(fastmath=True, cache=True)
def cut_couplings(u_D, u_R, du_D, du_R,
                  b_row, b_col, b_w,         # compact COO: R(dir) -> D
                  f_idx, f_w, f_cnt):        # ELL over concat [D;R]: rows = |R|
    for k in range(b_row.shape[0]):
        du_D[b_row[k]] += b_w[k] * u_R[b_col[k]]
    nD = u_D.shape[0]
    for q in range(f_cnt.shape[0]):
        s = 0.0
        for k in range(f_cnt[q]):
            ix = f_idx[q, k]
            s += f_w[q, k] * (u_D[ix] if ix < nD else u_R[ix - nD])
        du_R[q] += s
```

Gather/scatter boundary application (I30/I31/I39):

```python
@njit(parallel=True, cache=True)
def apply_bcs(du_D, src_D, fluid_idx, dir_idx, u3, t, face_k, coords):
    for i in prange(fluid_idx.shape[0]):
        k = fluid_idx[i]; du_D[k] += src_D[k]
    for i in prange(dir_idx.shape[0]):
        du_D[dir_idx[i]] = 0.0
    # face value at time t, evaluated only on the boundary plane
    for j in prange(u3.shape[1]):
        for k in range(u3.shape[2]):
            u3[face_k, j, k] = mms(coords[0][face_k], coords[1][j], coords[2][k], t)
```

RK4 with slot updates (I37/I38):

```python
@njit(parallel=True, fastmath=True, cache=True)
def rk4_step(u0, u1, acc, rhs_buf, dt, t, args):
    # each field is a tuple (D, Rx, Ry, Rz)
    for s in range(4):
        c  = (0.5*dt, 0.5*dt, dt, 0.0)[s]
        w  = (dt/6, dt/3, dt/3, dt/6)[s]
        for b in range(4):
            n = u0[b].shape[0]
            for i in prange(n):
                u1[b][i] = u0[b][i] + c*rhs_buf[b][i] if s > 0 else u0[b][i]
        update_boundary(u1, t + c, args)
        compute_rhs(u1, rhs_buf, t + c, args)
        for b in range(4):
            for i in prange(acc[b].shape[0]):
                acc[b][i] += w * rhs_buf[b][i]
    for b in range(4):
        for i in prange(u0[b].shape[0]):
            u1[b][i] = u0[b][i] + acc[b][i]
```

## Performance assessment

All numbers in this section are estimates from the survey, not measurements, and the C++ baseline was not measured on this machine (unverified).

Small 2D (about 2.6k-5k points). The C++ RHS is quoted at about 32 us for roughly 40 launches. A single jitted RHS or RK step has one dispatch and fused loops, so 5-15 us per RHS is plausible. A prange region may cost 5-20 us of fork-join, comparable to the whole RHS, so use serial njit or one parallel region at these sizes. Matching or beating C++ is plausible but unproven.

Large 3D. Memory bound, so parity with C++/Kokkos is the expectation. LLVM vectorizes unit-stride loops; slow-axis strided stencils do not vectorize automatically, so the generator must reorder loops (prange over slow pencils, contiguous inner loop). The survey guesses 1.0x parity with a possible 1.5-2x gain after restructuring against the quoted 13.7 ns/pt Kokkos kernel at 64^3; this is unverified.

Dispatch overhead. Estimated 0.2-1 us per call plus 100-300 ns per array argument for unboxing (unverified). One call per step is negligible; tens of calls per RHS are not. Putting the time loop inside njit removes it, as does packing arrays into few arguments.

Fusion. Numba has no graph or whole-program fusion. It fuses elementwise array expressions within a function under parallel=True, but does not fuse separate explicit loops across stencil and pointwise passes. The DSL generator must write fused loops. In exchange, everything inside one njit function is compiled together and LLVM inlines callees, so there is no per-op overhead.

JIT warmup. Seconds to tens of seconds for large parallel functions, more for generated closures such as the roughly 1170-line E2_1 BC closure (survey figure). cache=True persists compiled code but is keyed on the source file, so dynamically generated modules need real files on disk and stable hashes. Warmup is amortized over production runs but hurts BO loops and CI without a warm cache. AOT via numba.pycc is deprecated, not removed (corrected below), and the docs say it will not be removed until a replacement exists; it is not a recommended route.

## Suitability as a DSL substrate

Strong for the CPU. A DSL can generate Python source (loops, index tables, specialized constants), exec it, and wrap it with njit(parallel=True, fastmath=True, cache=True). Explicit loops preserve the per-line structure, and the ragged cut-cell data (fragments, ELL tables, closure ids) uses ordinary control flow. This is the main reason Numba fits better than tracing array frameworks.

Limits the generator must handle: loop order and layout for SIMD on strided axes; compile time and cache keying for generated code; typing quirks (heterogeneous tuples of arrays, global freezing, recompiles on signature change); opaque typing error messages; scalar libm for transcendentals.

Recommended architecture: a DSL IR with several emitters. Numba is the CPU emitter. Apple GPU needs a separate emitter (MSL, MLX, or torch ops) on a different substrate, and would run in fp32 because Apple GPUs lack fp64. Unified memory on Apple silicon could allow CPU/GPU sharing, but Numba does not use it.

Interop. NumPy arrays pass zero-copy both ways; scipy and LAPACK work on the same memory. ctypes/cffi function pointers can be called from nopython code; @cfunc exposes a C pointer to C++ but is awkward. Calling Kokkos from Numba is impractical except via a C ABI shim.

## Risks

- No Apple GPU path (absence unverified at the tracker level). If MPS is a hard requirement, Numba covers half the goal.
- JIT compile time and cache invalidation for large generated functions; hurts BO loops and CI.
- Strided-axis SIMD is not automatic; large-3D parity depends on hand-restructured loops and is not benchmarked.
- CUDA moved to NVIDIA's numba-cuda with a separate cadence; signals where core-team focus lies.
- Intel SVML unsupported since 0.62.0, hurting vectorized exp/cos in the MMS.
- prange fork-join cost may equal a whole small-2D RHS.
- Version coupling to NumPy, Python, and llvmlite/LLVM (LLVM 22 in 0.66); upgrades can invalidate caches or change behavior (for example 0.67 eig returning complex).
- Free-threading support is experimental (Tier 1.5 in 0.66), not absent and not production.
- Small-team project; bus factor is moderate though 2026 releases are regular.
- Performance estimates are unmeasured; a microbenchmark on the target machine is required before committing.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| Latest Numba 0.67.0 (11 Aug 2026) with NumPy 2.5; Python 3.14 supported | Confirmed (Python 3.14 part indirect) | 0.67 notes: release date, NumPy 2.5, Windows ARM64 Tier 1.5. General Python matrix not on page. Existence of a newer release not checked. |
| 0.66.0 (30 Jun 2026) used llvmlite 0.48 / LLVM 22; SVML unsupported | Confirmed | 0.66 notes: llvmlite 0.48, LLVM 22, numba.typed.Set; SVML unsupported since 0.62.0. |
| numba-cuda 0.30.4 (3 Jul 2026), Linux and Windows wheels only, no macOS | Confirmed, with addition | PyPI: cu12/cu13 extras; Linux ARM64 wheels also exist; no macOS wheels. In-tree CUDA deprecated in 0.61. |
| No Metal/MPS backend and no official plan | Uncertain | Empty search only; numba/numba tracker not read. Plausible but unconfirmed. |
| OpenMP layer in 0.67 uses omp_set_max_active_levels | Confirmed | 0.67 notes; falls back to omp_set_nested. |
| np.linalg.eig returns complex for real input in 0.67 | Confirmed | 0.67 notes. |
| numba.pycc AOT deprecated and removed | Refuted (corrected) | Deprecation page: pending deprecation since 0.57.0, will not be removed until a replacement exists. Deprecated, not removed. |
| No free-threading support in 0.67 | Uncertain (misleading) | 0.67 notes are silent; 0.66 notes list 3.14 free-threading as experimental Tier 1.5. |
| Dispatch 0.2-1 us, unboxing 100-300 ns/array, prange 5-20 us, RHS 5-15 us | Uncertain | Self-declared estimates; no source. Benchmark before relying. |
| 1.5-2x gain after loop restructuring | Uncertain | Unmeasured estimate. |

## Sources

- https://numba.readthedocs.io/en/stable/release/0.67.0-notes.html
- https://numba.readthedocs.io/en/stable/release/0.66.0-notes.html
- https://numba.readthedocs.io/en/0.66.0/_sources/release/0.66.0-notes.rst.txt
- https://numba.readthedocs.io/en/stable/reference/deprecation.html
- https://pypi.org/project/numba-cuda/
- https://github.com/NVIDIA/numba-cuda
- https://numba.discourse.group/t/rfc-moving-the-cuda-target-to-a-new-package-maintained-by-nvidia/2628
- https://numba.readthedocs.io/en/stable/cuda/
