# Devito as a SHOCCS DSL substrate

| | |
|---|---|
| Fit for SHOCCS | Good for the regular stencil bulk and workable for cut-cell closures via indirect ELL/COO tables, on CPU and NVIDIA/AMD only. |
| CPU parity | Likely meets or beats C++/Kokkos in both regimes; local numbers are single-host, serial, and partly measured-vs-estimated (see log). |
| Apple GPU | None. No Metal, MPS or MSL path exists or is planned; Apple silicon is a CPU target only. |
| fp64 | Supported on all backends via explicit dtype=np.float64 (default is float32); moot on Apple GPU. |
| Risk | Moderate: no MPS, atomics on indirect writes, 0.2-0.8 ms apply() overhead, implicit_dims hoisting trap, GPU extras possibly commercial. |

Verifier-adjusted overall score: 5.5 / 10 (survey score 6.0).

## Status as of 2026-10-12

Latest release checked: Devito v4.8.23, 07 Aug 2026 (GitHub releases). Recent releases: v4.8.22 (22 Jun), v4.8.21 (23 Feb 2026), v4.8.20 (02 Sep 2025), back to v4.8.14 (19 Mar 2025). The cadence is irregular, with a gap of about 5.5 months between 4.8.20 and 4.8.21, so "every 1-3 months" is slightly generous. The project is actively developed and commercially backed by Devito Codes (commercial edition: DevitoPRO). Release-note feature claims (complex dtypes, interpolation-order API, ROCm 7, CUDA 12 images, Python 3.14 CI, MPI indexing revamp) were not checked against release notes and are unverified.

Devito is a SymPy-based stencil DSL lowered through its own compiler (Cluster IR, then IET) to C/C++ and JIT-compiled. Development is driven by seismic workloads (large regular grids, time loop inside the kernel, MPI domain decomposition). Indirect access works but is a secondary path.

The survey was run on macOS/Apple silicon, Python 3.14, Apple clang (serial C, no OpenMP). Scripts: .scratch/devito/scripts/t1.py to t5.py.

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| CPU C/C++ (serial) | Native | Yes | Measured locally. |
| CPU OpenMP | Native | Yes | parallel for, simd, blocking, CIRE; on macOS needs Homebrew llvm/libomp. |
| MPI | Native | Yes | Halo exchange and overlap modes; indirect/custom-Dimension tables are not decomposed automatically (unverified detail, from survey). |
| NVIDIA via OpenMP offload / OpenACC | Supported (open source) | Yes, full rate on datacenter GPUs | Registry confirmed; generated-code details are from local inspection. Indirect Inc lowers to atomic update. |
| AMD via OpenMP offload | Supported | Yes | ROCm 7 claim unverified. |
| Intel GPU | Partly confirmed | Yes | IntelDevice exists; OpenMP offload only. The survey's "SYCL" claim is NOT supported by the open-source registry (unverified). |
| Native CUDA/HIP/SYCL codegen | Not in open-source registry | n/a | Probably DevitoPRO only; unverified (no official feature matrix). |
| Metal / MPS / Apple GPU | Unsupported | n/a | Confirmed: archinfo has only AppleArm M1/M2/M3 CPU platforms; Device classes are Intel, NVIDIA, AMD. |

The library default dtype is float32, so every Grid, Function and TimeFunction must set dtype=np.float64 (int32 for index tables). fp64 RK4 matched a NumPy reference to 3e-16 (local, unverified by the verifier).

## Idiom mapping

Support: native, workaround (host-built tables or indirect indexing), unsupported. Most "setup only" rows stay in host NumPy.

| ID | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | One Function per buffer; R buffers as Function on a q Dimension; slots as separate Functions; TimeFunction buffer 2 for u0/u1. | No runtime cost. |
| I02 | workaround | Keep host NumPy/Numba ray casting emitting SoA psi, solid, normal, shape arrays. | Setup only. |
| I03 | workaround | Host builds fragment table; no ragged strided-segment primitive; flatten to masks plus ELL closure table. | Setup only. |
| I04 | native | Upload mask Functions (fluid, interior per direction, Dirichlet); SubDomain for faces; index Functions for object lists. | One extra stream per mask, fused. |
| I05 | native | u.dx2(weights=[...]) constant taps become literals. | Same as hand-coded constant taps. |
| I06 | workaround | Host sympy.lambdify over psi, reusing stencil_gen results, keeping clamps. | Setup only. |
| I07 | workaround | Host searchsorted fragment lookup emits padded (idx, coef) tables; SparseFunction interpolation cannot express fluid-aware one-sided wall interpolation. | Setup only. |
| I08 | workaround | Host groups lines by (left_kind, right_kind, bc); emits interior masks plus ELL closure table. | Setup only. |
| I09 | workaround | Host assembles Bf/Br rows into ELL tables over the R Dimension. | Setup only. |
| I10 | workaround | Not needed; compact COO/ELL replaces full-height row pointers. | Strictly better than C++ (removes about 134 MB/derivative sweeps at 256^3, survey estimate). |
| I11 | workaround | np.concatenate into a deduplicated closure table; one Function per table. | Setup only. |
| I12 | workaround | Python front end replaces Lua; coefficient fields as Functions from NumPy. | Setup only. |
| I13 | native | Grid is C order, last dimension fastest, matching ic(i,j,k); halo handled internally (Function.data excludes halo). | Native strided/SIMD loops. |
| I14 | workaround | R index q equals position in per-axis R list; consistent lexsort at setup. | None. |
| I15 | native | Work buffers are Functions owned by Devito. | No overhead. |
| I16 | workaround | Interior as masked constant-tap Eq per direction; wall/object closure rows as Inc over an ELL (r,k) pair; or thin-SubDomain Eqs for walls. Avoid a full-width per-point weights Function (2T-1 extra streams). | Interior near hand C (0.45 ns/pt/step, 7-point, 256^3, one core, local); closure ELL serial gather, atomics under OpenMP/GPU on about 2.4% of rows. |
| I17 | native | u.dx/dx2 with fd_order or explicit weights, times interior mask. | At or better than hand C. |
| I18 | native | SubDomain-restricted Eqs per closure row, or ELL; Dirichlet via masks. | Dense plane loops; one nest per closure row (R rows x 2 ends x 3 dirs). |
| I19 | workaround | ELL with psi-dependent rows via indirect Inc; PrecomputedSparseFunction.inject only for separable tensor-product weights. | Gather-bound, atomics under OpenMP/ACC; small share of RHS (survey estimate under 5%). |
| I20 | workaround | No sparse-matrix type; indirect Inc over custom Dimensions with int32 index and fp64 weight Functions. | Like a hand COO loop plus atomics under parallel backends. |
| I21 | workaround | Compact COO on Dimension b: Inc(du[Bx,By,Bz], Bw*uR[Bc]). | O(nnz); faster than current C++. |
| I22 | native | SubDomain Eqs on faces next to Neumann boundary; omitted when absent. | Negligible. |
| I23 | workaround | Eq(dR[q],0) then Inc over (q,k) ELL; loop index is output so no atomics. | Microseconds per RHS. |
| I24 | workaround | Inc(dR[q], wr[q,j]*uR[jc[q,j]]) in the same q loop. | Negligible. |
| I25 | native | Unneeded; du defined by an Eq, only closure/B terms are Inc. | Fused away. |
| I26 | native | Fold scale into expression via Constant. | Free. |
| I27 | native | One SymPy expression; Inc after the Eq in list order. | One loop nest for the D part. |
| I28 | native | Separate Eqs or inline into consumer. | Inlined: no scratch traffic. |
| I29 | native | Eq(rhs, gGx*u.dx + gGy*u.dy + gGz*u.dz); exact closure needs materialised components. | Fused. |
| I30 | native | Multiply by fluid mask Function; src symbolic in time and coordinates. | Dense masked FMA. |
| I31 | native | SubDomain Eqs with symbolic g(t,x,y,z), or masks; R Dirichlet via mask or indirect Eq. | Negligible. |
| I32 | native | Whole RHS and the whole RK4 time loop in one Operator; stage Eqs need implicit_dims=(time,) or they are hoisted (wrong results). | 7.2 us/RK4 step at 51x51 (2000 steps/apply), 18 us at 100 steps/apply (local). |
| I33 | native | Not needed; one apply runs many steps. | apply() costs 0.2-0.8 ms (local); never call per RHS. |
| I34 | native | MMS as SymPy expression in coordinates and time inside the Eq. | Fused in-kernel evaluation. |
| I35 | unsupported | No per-point host callbacks in an Operator; replace with SymPy MMS. | N/A (removed). |
| I36 | native | Source is an in-kernel expression of the time symbol. | Free. |
| I37 | native | Stage Eqs with implicit_dims=(time,); R buffers on the q Dimension. | Bandwidth-bound; no launches. |
| I38 | native | TimeFunction modulo buffering replaces deep_copy. | Free. |
| I39 | native | Dirichlet/Neumann data symbolic on SubDomains inside the time loop. | Negligible. |
| I40 | native | Static index/mask Functions built once. | Free. |
| I41 | workaround | Run K steps per apply, then NumPy or devito.norm/sumall; max/argmax and termination are not first-class. | One readback per K steps; D2H copy on GPU. |
| I42 | workaround | Between applies, NumPy plus Python I/O; optional ConditionalDimension snapshots. | Cost only on dumps. |
| I43 | native | Python loop of op.apply(time_m, time_M, dt) chunks. | 0.2-0.8 ms per chunk, amortised. |
| I44 | workaround | No matrix export; assemble scipy.sparse from the same host tables or probe with unit vectors. | Host only. |
| I45 | workaround | NumPy oracles. | N/A. |
| I46 | native | SymPy expressions are the programming model. | Fused. |
| I47 | native | Five TimeFunctions; inline nonlinear fluxes; derivatives of flux expressions with CIRE; cut-cell corrections need materialised flux temporaries. | Strong on regular part; extra temporaries at cut cells. |
| I48 | workaround | Stays in scipy. | N/A. |
| I49 | native | Same SymPy ecosystem; stencil_gen output feeds lambdify or in-kernel expressions. | Setup. |
| I50 | workaround | Needs scipy operator assembly as in I44. | Host only. |
| I51 | native | In-process Python solver removes the subprocess and Lua bridge. | Large saving per BO design point. |

## Code sketches

Locally tested by the survey; not independently re-run by the verifier.

Per-line block matvec with masked interior plus ELL closure (I16-I19):

```python
x, y, z = grid.dimensions
# interior: constant taps on the whole grid, masked by mi_d (0 on closure rows, solid, Dirichlet rows/lines)
lap_int = (mi_x*u.dx2(weights=cx) + mi_y*u.dy2(weights=cy) + mi_z*u.dz2(weights=cz))
# closure rows (wall + object, all 3 dirs) as one ELL table; rows unique per dir
r, k = Dimension('r'), Dimension('k')
mk = lambda n, dims, shp, dt: Function(name=n, dimensions=dims, shape=shp, dtype=dt)
ox, oy, oz = (mk('o'+a, (r,), (nrow,), np.int32) for a in 'xyz')
ix, iy, iz = (mk('i'+a, (r, k), (nrow, Tmax), np.int32) for a in 'xyz')  # pad: point at a dummy cell, C=0
C = mk('C', (r, k), (nrow, Tmax), np.float64)
eqs = [Eq(du, kappa*lap_int, implicit_dims=(time,)),
       Inc(du[ox[r], oy[r], oz[r]], kappa*C[r, k]*u[ix[r, k], iy[r, k], iz[r, k]],
           implicit_dims=(time,))]
# alt for wall closures: one Eq per closure row j on a thickness-1 SubDomain
# Devito emits '#pragma omp atomic update' for the indirect Inc (cannot prove row uniqueness)
```

O/B CSR cut-cell coupling to R points (I21-I24):

```python
q, kk, j, b = (Dimension(n) for n in ('q', 'kk', 'j', 'b'))
uR  = Function(name='uRx', dimensions=(q,), shape=(nR,), dtype=f64)
duR = Function(name='duRx', dimensions=(q,), shape=(nR,), dtype=f64)
fx, fy, fz = (Function(name='f'+a, dimensions=(q, kk), shape=(nR, KF), dtype=np.int32) for a in 'xyz')
fw = Function(name='fw', dimensions=(q, kk), shape=(nR, KF), dtype=f64)
rc = Function(name='rc', dimensions=(q, j), shape=(nR, KR), dtype=np.int32)
rw = Function(name='rw', dimensions=(q, j), shape=(nR, KR), dtype=f64)
Bx, By, Bz, Bc = (Function(name=n, dimensions=(b,), shape=(nB,), dtype=np.int32) for n in ('Bx','By','Bz','Bc'))
Bw = Function(name='Bw', dimensions=(b,), shape=(nB,), dtype=f64)
T = dict(implicit_dims=(time,))
eqs += [Eq(duR[q], 0., **T),
        Inc(duR[q], fw[q, kk]*u[fx[q, kk], fy[q, kk], fz[q, kk]], **T),  # loop index = output: no atomics
        Inc(duR[q], rw[q, j]*uR[rc[q, j]], **T),
        Inc(du[Bx[b], By[b], Bz[b]], Bw[b]*uR[Bc[b]], **T)]                # atomic under OpenMP/ACC
```

Boundary application, fluid source and Dirichlet (I30, I31, I39):

```python
tt = time*dt
X, Y = grid.dimensions; hx, hy = X.spacing, Y.spacing
sol = lambda xx, yy: amp*sp.exp(-((xx-cx)**2 + (yy-cy)**2)*iv2)*sp.cos(om*tt)
src_D = ddt_sol(X*hx, Y*hy, tt) - kappa*lap_sol(X*hx, Y*hy, tt)   # SymPy-generated MMS
eqs += [Eq(du, (1 - dirm)*(du + fluid*src_D), **T)]
eqs += [Eq(duR[q], (1 - dirR[q])*(duR[q] + srcR(xR[q], yR[q], tt)), **T)]
left = Left()  # a SubDomain subclass: {x: ('left', 1), y: y}
eqs += [Eq(S, sol(X*hx, Y*hy), subdomain=left, **T),
        Eq(SR[q], dirR[q]*sol(xR[q], yR[q]) + (1 - dirR[q])*SR[q], **T)]
```

RK4 slot updates (I37, I38), reported as matching NumPy to 3e-16:

```python
U  = TimeFunction(name='U', grid=grid, space_order=so, dtype=f64)
UR = TimeFunction(name='UR', dimensions=(grid.stepping_dim, q), shape=(2, nR), dtype=f64)
S, SR = Function(name='S', grid=grid, space_order=so, dtype=f64), Function(name='SR', dimensions=(q,), shape=(nR,), dtype=f64)
K  = [Function(name=f'K{i}', grid=grid, dtype=f64) for i in range(4)]
KR = [Function(name=f'KR{i}', dimensions=(q,), shape=(nR,), dtype=f64) for i in range(4)]
ts = grid.stepping_dim
eqs = [Eq(S, U), Eq(SR[q], UR[ts, q])]
for s, c in enumerate([0.5, 0.5, 1.0, None]):
    eqs += rhs(S, SR, K[s], KR[s])
    if c is not None:
        eqs += [Eq(S, U + c*dt*K[s]), Eq(SR[q], UR[ts, q] + c*dt*KR[s][q])]
eqs += [Eq(U.forward, U + dt/6*(K[0] + 2*K[1] + 2*K[2] + K[3])),
        Eq(UR[ts+1, q], UR[ts, q] + dt/6*(KR[0][q] + 2*KR[1][q] + 2*KR[2][q] + KR[3][q]))]
# CRITICAL: without implicit_dims Devito hoists time-invariant stage eqs out of the time loop
eqs = [e if e.lhs.function in (U, UR) else e.func(e.lhs, e.rhs, implicit_dims=(grid.time_dim,)) for e in eqs]
op = Operator(eqs)
for chunk in range(nchunks):
    op.apply(time_m=chunk*K, time_M=chunk*K + K - 1)
    stats = masked_stats(U.data[(chunk*K + K) % 2], UR.data[...])
```

## Performance assessment

All numbers below are local, single host, serial C on Apple silicon, from the survey. The verifier could not rerun them.

Small 2D (51x51 fp64): a full RK4 step with masked 5-point Laplacian, COO R-to-D coupling, D-to-R ELL and R-buffer stage algebra, fused in one Operator, took 7.2 us/step (2000 steps/apply) and 18 us/step (100 steps/apply). The survey compares this with a C++ estimate of 300-1000 us per step. That is a measurement against an estimate, not a like-for-like benchmark, so the claimed 5-20x speedup is indicative only. Not yet included in the test: full closure ELL, in-kernel MMS, Dirichlet SubDomain Eqs. The survey projects 15-50 us for a full heat step.

Large 3D: 0.45 ns/pt/step for a 7-point fp64 update at 128^3 and 256^3 on one core (about 36 GB/s), versus 6.3 ns/pt per RHS for C++ at 64^3 on 4 threads (different operation and thread count, so not directly comparable). With OpenMP expect near-STREAM bandwidth. Per-direction masks add one stream each. Closure and B rows (about 2.4% of rows) are cheap on CPU and a moderate, unmeasured penalty on GPU because of atomics.

Dispatch overhead: op.apply() costs 200 us (2 Functions) to 790 us (about 15 Functions) in Python argument preparation. Inside the generated C there is no per-op dispatch. Rule: batch K steps per apply.

Fusion: the whole multi-step time loop compiles into one C function. Two traps: stage Eqs need implicit_dims=(time,) or they are silently hoisted (reproduced locally, wrong results); indirect Inc becomes separate loop nests with atomics under OpenMP/ACC.

JIT: Operator construction plus C compilation takes seconds and is cached by code hash; table sizes are runtime arguments.

## Suitability as a DSL substrate

Strengths: SymPy-native (stencil_gen expressions, closures and MMS flow in directly); fuses a whole multi-stage time loop; hand-C-class bandwidth on the 3D bulk; MPI for the regular part; indirect indexing is enough for ELL/COO closure, B, Bf and Br tables.

Weaknesses: no ragged-segment, line-wise or matrix-export abstraction, so the SHOCCS DSL must own fragment flattening, closure-table construction and scipy assembly for eigen/stability tooling; indirect Inc always gets atomics; MPI with custom-Dimension tables needs manual per-rank localisation; the DSL must emit one Operator per run segment; no Metal backend.

Verdict: a credible fp64 CPU/NVIDIA/AMD execution layer under a custom DSL, where the frontend lowers geometry to masks, constant taps, ELL/COO tables and SubDomains and hands Devito one fused time-loop Operator. A separate backend is needed for Apple GPU (for example MLX, PyTorch-MPS or Taichi-Metal consuming the same host tables).

Interop: Function.data is a zero-copy NumPy view on CPU; external pointers cannot be wrapped without copy; GPU data is mapped at apply() boundaries; no general extern-function mechanism; generated C can be dumped (op.ccode) but embedding it in the C++ build is unofficial. These interop details are from the survey and unverified.

## Risks

1. No Metal/MPS backend and none planned (confirmed); Apple-GPU goal needs a second backend.
2. Silent wrong results if stage Eqs lack implicit_dims=(time,) (reproduced locally, unverified independently).
3. apply() overhead of 0.2-0.8 ms forces batching K steps per apply; stats readback every K steps.
4. Atomics on every indirect Inc; unmeasured GPU cost.
5. MPI does not automatically handle indirect/custom-Dimension tables.
6. Open-source GPU path depends on OpenMP-offload/OpenACC compilers; native CUDA/HIP/SYCL and advanced GPU optimisation may be commercial (unverified).
7. Default float32 silently applies to any Function created without dtype.
8. No matrix-export mode; parallel scipy assembly must be kept in sync with the tables.
9. Seismic-focused; cut-cell indirect patterns are at the edge of test coverage (recent releases still fix SparseFunction and SubDomain bugs; from survey).
10. macOS dev loop needs Homebrew llvm/libomp for OpenMP.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| Latest release v4.8.23 (07 Aug 2026); releases every 1-3 months | Confirmed, cadence corrected | Releases page lists v4.8.23 through v4.8.14; about 5.5-month gap between 4.8.20 and 4.8.21; years inferred from ordering. |
| Open-source backends are CPU C/CXX/OpenMP plus Device via OpenMP offload and OpenACC; no Metal; Apple only AppleArm M1/M2/M3; Device classes Intel/NVIDIA/AMD | Confirmed | devito/core/__init__.py and devito/arch/archinfo.py; GPU probing uses nvidia-smi, rocm-smi, sycl-ls, lshw, lspci only. |
| Native CUDA/HIP/SYCL codegen and advanced GPU optimisations are DevitoPRO only | Uncertain | Absent from open-source registry (consistent), vendor material suggests PRO features, no explicit feature matrix; SYCL not evidenced. Medium confidence. |
| Intel GPU works through OpenMP/SYCL | Uncertain (SYCL part unsupported) | IntelDevice exists, but no SYCL language registered; only OpenMP offload. |
| No Apple GPU / MPS / Metal path | Confirmed | No Metal code in core or archinfo; macOS references are CPU-side only. |
| Indirect indexing, atomics on indirect Inc, Differentiable weights, separable PrecomputedSparseFunction weights, float32 default, apply() overhead, implicit_dims hoisting, local performance numbers | Uncertain | Docs URL returned 404; rest on survey's own local runs, not rerun. Single-host, serial. |
| 7 us/step vs C++ 300-1000 us | Uncertain / weakened | Measurement compared with an estimate, not like for like. |
| Release-note feature list (complex dtypes, ROCm 7, CUDA 12, Python 3.14 CI, MPI revamp) | Unverified | Not checked against release notes. |

## Sources

- https://github.com/devitocodes/devito/releases
- https://www.devitoproject.org/
- https://github.com/devitocodes/devito/blob/main/devito/core/__init__.py
- https://github.com/devitocodes/devito/blob/main/devito/arch/archinfo.py
- https://github.com/devitocodes/devito/blob/main/devito/types/dimension.py
- https://github.com/devitocodes/devito/blob/main/devito/types/sparse.py
- https://github.com/devitocodes/devito/blob/main/devito/finite_differences/derivative.py
- https://github.com/devitocodes/devito/blob/main/devito/finite_differences/finite_difference.py
- https://nbviewer.jupyter.org/github/devitocodes/devito/blob/v4.0/examples/seismic/tutorials/07_DRP_schemes.ipynb
- https://nbviewer.org/github/devitocodes/devito/blob/v4.8.5/examples/userapi/05_conditional_dimension.ipynb
- https://nbviewer.org/github/devitocodes/devito/blob/v4.7.1/examples/performance/00_overview.ipynb
- https://www.intel.com/content/www/us/en/customer-spotlight/stories/devito-codes-customer-story.html
- https://www.amd.com/en/blogs/2023/devito-revolutionizes-high-performance-computing-f.html
- https://instinct.docs.amd.com/latest/_sources/isv-apps/devito.md
- Local experiments: .scratch/devito/scripts/t1.py to t5.py (Devito 4.8.23, Python 3.14, macOS Apple silicon, serial C)
