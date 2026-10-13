# pystencils (2.x) and lbmpy: survey for SHOCCS

| Verdict box | |
|---|---|
| Fit for SHOCCS | Plausible CPU/CUDA leaf-kernel layer under a DSL; not a full answer. Interior axis stencils fit well, ragged per-line closures and CSR coupling need restructuring into index-list kernels. |
| CPU parity | Likely at or above parity on large 3D (unmeasured); small 2D at risk from Python dispatch unless fused or exported to C++ via pystencils-sfg. |
| Apple GPU | None found. No Metal/MPS/MSL target (absence of evidence, not positively confirmed). A new backend would be needed. |
| fp64 | Native on CPU, CUDA and HIP. Not applicable on Apple GPU (no hardware fp64 on any framework). |
| Risk | Medium to high: 2.0 only about 3 months past GA, academic maintainers, LBM-focused community, several claims here unverified. Adjusted overall score 5.0 / 10. |

## Status as of 2026-10-12

- pystencils 2.1 (PyPI, 2026-09-29) is the latest. History: 2.0b1 2026-05-07, 2.0b2 2026-05-26, 2.0 GA 2026-07-02, 1.4 2025-10-23, 1.4.1 bugfix 2026-06-30. 2.1 requires Python >=3.12; 2.0 and its betas only required >=3.10 (correction to the survey).
- lbmpy 2.1.1 (2026-09-30). lbmpy 2.1 was yanked ("Invalid dependency specs"); pin 2.1.1 or later.
- pystencils 2.0 is a ground-up rewrite (new IR and backend). Target audience is lattice Boltzmann and phase-field work on regular grids. No evidence found of cut-cell or embedded-boundary finite-difference users.
- Unverified: that the i10git GitLab is archived and migrating to gitos.rrze.fau.de with GitHub mirrors. Verification found the i10git docs and issues still served and could not find the notice.

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| CPU (C++ JIT) | Confirmed | Native, default | Loop nests with ghost layers. Targets enum includes X86_SSE, X86_AVX, X86_AVX512 (+FP16), ARM_NEON (+FP16), ARM_SVE. Vectorization via cpu.vectorize.enable, OpenMP via cpu.openmp.enable (parallelizes the outermost loop). Quality of NEON/SVE on Apple Silicon unverified. |
| CUDA | Confirmed | Native | JIT and launch via CuPy; arrays may be CuPy or __cuda_array_interface__ objects. |
| HIP | Confirmed | Native | Via CuPy-ROCm; docs say setup is more complex (may need to build cupy from source). |
| SYCL | Confirmed experimental | Device dependent | Issue 126: finalization not on the 2.0 agenda; moving it to an extension package is only a proposal. "No JIT" is plausible but not confirmed from the text. No mature Apple-GPU SYCL implementation. |
| Metal / MPS | Not found (unverified absence) | Not available | 2.0 GPU docs list CUDA, HIP and a portable CurrentGPU target only. |
| OpenCL | Survey claim of removal is unverified | n/a | Apple OpenCL is deprecated and frozen at 1.2 regardless. |
| Kokkos | Does not exist | n/a | Issue 126 only says an accessor-based lowering protocol could support Kokkos Views and an extension package might host it. |
| Standalone C++ (pystencils-sfg) | Confirmed to exist | Native | Generator scripts register kernels and emit .hpp/.cpp with a CMake module (pystencilssfg_generate_target_sources); fields map to std::mdspan. PyPI availability and exact API of current releases unverified. |

## Idiom mapping

Support values: native, workaround, unsupported. Expected performance is a relative estimate against the current C++/Kokkos code and is unmeasured unless stated.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | Scalar as four independent Field objects, or packed 1D array with views for D and R. | Equal (setup). |
| I02 | unsupported | Keep host NumPy ray casting; no geometry primitives. | N/A, host. |
| I03 | workaround | Host NumPy fragment tables; no ragged-segment primitive, so masks, index lists or per-fragment calls. | Equal (setup). |
| I04 | native | Masks and index arrays in NumPy; consumed by an index-list kernel (CreateKernelConfig.index_field in 2.x; the 1.x name create_indexed_kernel is not in 2.x docs; struct-dtype details unverified). | Equal. |
| I05 | native | SymPy constants or floats become literals or kernel parameters. | Equal. |
| I06 | native | Existing SymPy CSE closure expressions fed as Assignments over psi data, or lambdified on host. Main strength of the framework. | Equal or better. |
| I07 | workaround | Host builds interpolation tables as ELL for an index-list kernel. | Equal (setup). |
| I08 | workaround | Host NumPy grouped by line kind. | Equal. |
| I09 | workaround | Host NumPy produces ELL tables. | Equal. |
| I10 | native | NumPy lexsort or scipy.sparse. | Equal. |
| I11 | native | np.concatenate, np.cumsum, dedup into closure table. | Equal or better. |
| I12 | native | Plain Python config replaces Lua. | Equal. |
| I13 | native | NumPy array shape (nx,ny,nz), C layout; arbitrary layouts and strides supported; offsets like u[1,0,0]. | Equal. |
| I14 | native | R ordering is array order; reproduce lexsort in NumPy. | Equal. |
| I15 | native | Fields bound at call time; work buffers are extra arguments. | Equal. |
| I16 | workaround | Interior axis stencil (I17) over the whole grid plus index-list closure kernels (I18/I19) overwriting closure rows. No fused ragged (start, stride, length) segment form; encoding it as an index list gives runtime-trip-count inner loops and loses vectorization. | Estimated 0.7x to 1.5x of C++ overall (unmeasured). |
| I17 | native | Assignment with taps u[-1,0,0] etc., ghost_layers or iteration slice. Core pystencils use case; blocking, vectorization, OpenMP available. | Estimated 1.0x to 2x faster than current Kokkos CPU kernel (unmeasured). |
| I18 | native | Separate kernels over boundary planes via iteration slice with explicit dense Assignments. | Equal. |
| I19 | workaround | Index-list kernel with a padded sum over compile-time T. Index-list kernels are the idiomatic route for irregular boundaries in this ecosystem (1.x evidence; 2.x via index_field). Note index_field is exclusive with ghost_layers and iteration_slice and iteration order is unspecified. | Equal (gather-bound, small). |
| I20 | workaround | Padded ELL index-list kernel; no native CSR. True variable-length CSR loop not confirmed as a feature. | Equal or better. |
| I21 | workaround | Compact ELL or COO index-list kernel, du[row] += w * uR[col]. | Estimated better (no full-height row-pointer sweep). |
| I22 | native | Slab assignment via iteration slice at the Neumann face with a 2D face field. | Equal or better. |
| I23 | workaround | Index-list gather from D, write to R. | Equal; launch overhead dominates at small sizes. |
| I24 | workaround | Concatenate sources into the I23 table or add a second small index kernel. | Equal. |
| I25 | native | Constant assignment, better fused into the accumulating kernel. | Better with fusion. |
| I26 | native | Fold factor into stencil Assignments symbolically. | Better. |
| I27 | native | du = Ox(u)+Oy(u)+Oz(u) as one Assignment, single read pass; per-direction closures need separate kernels. | Estimated better (one pass instead of three). |
| I28 | native | Several Assignments in one AssignmentCollection. | Equal or better. |
| I29 | native | Pointwise Assignment over gradient fields, fusable with the stencils. | Estimated better (no scratch fields). |
| I30 | native | Mask Field multiplier, sp.Piecewise, or index-list over fluid set. | Equal. |
| I31 | native | Face-slice kernels via iteration slice; object points via index kernel. | Equal. |
| I32 | workaround | No graph capture; fuse the RHS into few kernels instead. CUDA graphs would be manual via CuPy. | Estimated 4x to 10x slower per tiny kernel from Python unless exported via pystencils-sfg (unmeasured). |
| I33 | native | Python function calling compiled kernels. | Dominated by call overhead on small grids. |
| I34 | native | SymPy MMS with coordinate symbols compiled into a kernel (2.x coordinate symbol names unverified). | Estimated better than virtual-call evaluation. |
| I35 | unsupported | Lua callback goes away; use Python/NumPy or SymPy. | N/A. |
| I36 | native | t as scalar kernel parameter; source generated from SymPy. | Better. |
| I37 | native | AssignmentCollection per RK stage; pack D\|Rx\|Ry\|Rz into 1D arrays or use two kernels. | Equal or better. |
| I38 | native | Pointer swap in Python. | Better. |
| I39 | native | Face-slice kernels with SymPy exact solution. | Better. |
| I40 | native | Hoist in Python. | Better. |
| I41 | workaround | No reduction primitive confirmed in 2.x (unchecked); use NumPy/CuPy reductions. | Estimated 2x to 4x slower than fused MinMax at small sizes; negligible per step. |
| I42 | native | NumPy tofile and Python XML writing. | Equal. |
| I43 | native | Python loop. | Equal. |
| I44 | workaround | Assemble with scipy.sparse from the same tables; kernels cannot be exported as matrices. | Equal. |
| I45 | native | NumPy. | N/A. |
| I46 | native | SymPy expressions, the framework's native model. | Better. |
| I47 | workaround | Euler flux and primitive algebra easy in SymPy; multi-component fields via index dimensions. R-point normals and wall BCs remain hard as in every framework. | Equal or better. |
| I48 | unsupported | Dense/Kronecker assembly stays in NumPy/SciPy. | N/A. |
| I49 | native | stencil_gen CutCellResult Matrix blocks pass through codegen with no printer retargeting. Biggest advantage. | N/A (setup). |
| I50 | unsupported | Stays on host scipy; no matrix export. | N/A. |
| I51 | native | In-process Python solver replaces the subprocess loop. | Better. |

## Code sketches

All sketches are schematic. They use 1.x-style spellings (ps.fields, create_kernel with ghost_layers, iteration_slice, IterationSpace.over_range); in 2.x kernel options go through CreateKernelConfig, and sparse iteration through its index_field option. Check against the installed package.

### Per-line block matvec with strided interior and per-line closures (I16)

```python
import pystencils as ps, sympy as sp
u,du = ps.fields('u,du: double[3D]', layout='c')
# 1) interior: whole-grid constant-tap axis stencil along x
c = sp.symbols('c0:3')
interior = ps.Assignment(du.center, c[0]*u[-1,0,0]+c[1]*u[0,0,0]+c[2]*u[1,0,0])
k_int = ps.create_kernel(interior, ghost_layers=1)   # 2.x: create_kernel(asms, cfg)
# 2) closures: one index-list kernel over (row, col0..colT-1, coef0..coefT-1)
clos = ps.fields('clos: double[1D]')                 # packed ELL
row = ps.TypedSymbol('row', 'int64'); cols = [...]   # index-field members
acc = sum(C[k]*u_flat[col[k]] for k in range(T))     # u_flat: 1D alias of D
k_clos = ps.create_kernel(ps.Assignment(du_flat[row], acc))  # overwrites interior result
# host: k_int(u=u_arr, du=du_arr, c0=.., c1=.., c2=..)
#       k_clos(u_flat=u_arr.ravel(), du_flat=du_arr.ravel(), idx=ell_table)
# Rows near fragment ends are overwritten by closures or excluded with a mask Field.
```

### CSR cut-cell coupling to R points (I20 to I24)

```python
# Padded ELL tables built on host: Bf_idx[nR,K], Bf_w[nR,K]
uD  = ps.fields('uD: double[1D]'); duR = ps.fields('duR: double[1D]')
Bi  = ps.fields('Bi: int64[2D]');  Bw = ps.fields('Bw: double[2D]')
r = ps.TypedSymbol('r','int64')
acc = sum(Bw[0,k]*uDR[Bi[0,k]] for k in range(K))    # K compile-time, pad w=0, idx=0
asm = ps.Assignment(duDR[Bi_row], duDR[Bi_row] + acc)
kern = ps.create_kernel(asm, iteration_space=ps.IterationSpace.over_range(nR))  # 2.x name unverified
# R->D transpose: gather by (row_idx[nB], col_idx[nB], w[nB]); rows unique so no atomics.
```

### Boundary application (I30, I31)

```python
fluid = ps.fields('fluid: double[3D]'); dirm = ps.fields('dirm: double[3D]')
src, du = ps.fields('src,du: double[3D]')
expr = (1-dirm.center) * (du.center + fluid.center*src.center)
k = ps.create_kernel(ps.Assignment(du.center, expr))
k_face = ps.create_kernel(ps.Assignment(u.center, sol.center),
                          iteration_slice=ps.make_slice[0:1, :, :])
# Object points: index-list kernel du_R[idx[i]] = 0 over the Dirichlet object list.
```

### RK4 slot updates on packed buffers (I37)

```python
u0,u1,rk,srhs = ps.fields('u0,u1,rk,srhs: double[1D]')
c_a, c_b = sp.symbols('c_a c_b')
stage = [ps.Assignment(u1.center, u0.center + c_a*srhs.center),
         ps.Assignment(rk.center, rk.center + c_b*srhs.center)]
k_stage = ps.create_kernel(stage)
for (a,b) in [(dt/2,dt/6),(dt/2,dt/3),(dt,dt/3)]:
    k_stage(u0=U0,u1=U1,rk=RK,srhs=S,c_a=a,c_b=b)    # 1 call per stage
```

## Performance assessment

All figures below are engineering estimates; no benchmark was found or run.

- Small 2D: loses to C++ unless the whole step is exported as C++. Estimated 3 to 10 us per Python-to-kernel call; about 8 to 15 kernels per RHS and 50 to 80 calls per step gives roughly 0.3 to 0.8 ms per step, comparable to the estimated C++ step of 0.3 to 1 ms only if carefully fused. An unfused copy of the C++ graph (about 40 nodes per RHS) would be 3x to 5x slower.
- Large 3D: likely at or above parity. Bandwidth-bound axis stencils fuse into one pass and vectorize along the contiguous axis. Expect 1x to 2x over the 6.3 ns/pt measured at 64^3 and approach the 1.9 ns/pt bandwidth bound at 256^3 (extrapolation; no cut-cell measurement). Closure and ELL tables add 2 to 3 percent of rows.
- Dispatch overhead: a few microseconds per call, roughly 5x to 10x the 0.8 us C++ graph-node cost. No graph capture. On CUDA, each launch goes through CuPy (about 10 us or more).
- JIT warm-up: estimated 0.5 to 5 s per kernel, tens of seconds for dozens of kernels; poor for 20-step tests, fine for long runs. Caching details in 2.x unverified.
- Fusion: partial. Everything expressible as one AssignmentCollection over one iteration space fuses (three-axis laplacian, scale, source, RK stage). Interior, closure index lists and face slices stay separate, so the RHS is about 5 to 10 kernels. No whole-program compiler like jax.jit. pystencils-sfg can remove Python from the hot loop by emitting a C++ host function.

## Suitability as a DSL substrate

Strengths: SymPy input matches the existing stencil_gen assets, so DSL operators lower to Assignments without printer work; generated C++ loops with layout, slice and vectorization control; index-list kernels map onto ELL closure tables and R-point couplings; standalone C++ export via pystencils-sfg sidesteps Python dispatch and keeps the C++ driver option open.

Weaknesses: no ragged strided-segment primitive; no confirmed reductions or sparse types; no whole-step compiler; Python call overhead at small sizes; no Apple GPU. A DSL would own host setup (geometry, tables) and orchestration and use pystencils only for leaf kernels. A reasonable hybrid is DSL front end, SymPy lowering, pystencils CPU/CUDA kernels, plus a separate Metal path (for example MLX or a custom MSL printer) if Apple GPU is required. A Metal pystencils backend would need an MSL printer and a metal-cpp or PyObjC launch layer; weeks to months of work as an unmaintained fork.

## Risks

- No Metal/MPS target found; the Apple-GPU goal is unmet.
- 2.x is about 3 months past GA; API still settling and the sketches use 1.x names.
- Academic maintainership and LBM-focused community; infrastructure migration claim unverified.
- Fragment restructuring into interior plus closure index lists is a design change from the per-line team kernel.
- Python dispatch may miss the small-2D bar without fusion or C++ export.
- No confirmed reductions, CSR or graph capture.
- JIT warm-up and runtime C++ compiler requirement; Python >=3.12 for 2.1.
- Docs were partly unreadable to the survey; many performance numbers are estimates.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| 2.1 latest (2026-09-29), 2.0 GA 2026-07-02, Python >=3.12, lbmpy 2.1.1 2026-09-30 | Confirmed with corrections | PyPI JSON. Python >=3.12 starts at 2.1 (2.0 needed >=3.10). lbmpy 2.1 yanked. 1.4.1 exists (2026-06-30). |
| No Metal/MPS backend; OpenCL removed | Uncertain | 2.0 GPU docs list only CUDA, HIP, CurrentGPU; absence of evidence. OpenCL removal not re-verified. |
| SYCL experimental, not on 2.0 agenda, no JIT | Confirmed (no-JIT part unverified) | Issue 126. |
| No Kokkos backend; only hypothetical extension target | Confirmed | Issue 126. |
| GPU via CUDA and HIP with CuPy; HIP setup harder | Confirmed | 2.0 GPU manual and installation docs. |
| CPU SIMD (SSE/AVX/AVX512, NEON/SVE) and OpenMP | Confirmed | Target enum and cpu_optimization docs; Apple Silicon quality unverified. |
| Index-list kernels via create_indexed_kernel | Uncertain | 2.x uses CreateKernelConfig.index_field; exclusive with ghost_layers and iteration_slice; struct-dtype details unverified. |
| pystencils-sfg emits CMake-integrable C++ | Confirmed | sfg docs; PyPI status and API unverified. |
| i10git archived, migrating to gitos.rrze.fau.de | Uncertain | Could not find the notice; i10git pages still serve. |
| Python dispatch a few us per kernel | Uncertain | No benchmark; survey labels it an estimate. |
| No reduction primitive in 2.x | Unchecked | Not verified either way. |

## Sources

- https://pypi.org/pypi/pystencils/json
- https://pypi.org/pypi/lbmpy/json
- https://pypi.org/project/pystencils/
- https://i10git.cs.fau.de/pycodegen/pystencils/-/work_items/126
- https://pycodegen.pages.i10git.cs.fau.de/docs/pystencils/2.0/user_manual/gpu_kernels.html
- https://pycodegen.pages.i10git.cs.fau.de/docs/pystencils/2.0/user_manual/cpu_optimization.html
- https://pycodegen.pages.i10git.cs.fau.de/docs/pystencils/2.0/installation.html
- https://pycodegen.pages.i10git.cs.fau.de/docs/pystencils/2.0/api/generated/pystencils.codegen.target.Target.html
- https://pycodegen.pages.i10git.cs.fau.de/docs/pystencils/2.0/api/generated/pystencils.codegen.config.CreateKernelConfig.index_field.html
- https://pycodegen.pages.i10git.cs.fau.de/pystencils-sfg/
- https://pycodegen.pages.i10git.cs.fau.de/pystencils-sfg/getting_started.html
