# GT4Py (GridTools for Python): survey for SHOCCS

## Verdict box

- Fit for SHOCCS: partial. A plausible optional CPU/CUDA backend for a custom DSL, not a primary substrate. Overall score 4/10.
- CPU parity: plausible at large 3D (fused, tiled stencils); doubtful at small 2D unless the whole RK4 step is one compiled program. Unmeasured.
- Apple GPU: none. No Metal, MPS or macOS GPU support in README, extras, classifiers or 1.2.x release notes.
- fp64: yes on CPU, CUDA and ROCm. No Apple GPU path exists, and Apple GPUs lack fp64 anyway (unverified here).
- Risk: high. gt4py.next is still labeled experimental, 1.2.x releases kept breaking things, DaCe is pinned to an alpha series, and the cut-cell logic must be lowered to masks and tables outside GT4Py.

## Status as of 2026-10-12

Version checked: gt4py 1.2.3, published 5 Oct 2026 on PyPI. Development status Beta, Python >=3.12,<3.15 (excluding 3.13.10 and 3.14.1). Classifiers: OS POSIX, GPU :: NVIDIA CUDA. Extras: cartesian, cuda12, cuda13, jax, jax-cuda12, jax-cuda13, next, rocm6, rocm7, standard, testing. (Confirmed, PyPI.)

The project has two incompatible halves. gt4py.cartesian (GTScript on an I,J,K domain) is the stable branch. gt4py.next (Field, field_operator, connectivities) is described by the README as "still experimental" and "not compatible with the current stable version". Release notes show active changes in both branches, so the claim that Cartesian is in maintenance mode is not supported and is dropped.

Churn in the 1.2.x series (confirmed from release notes): DaCe pin moved from 2.0.0a5 (1.2.0) to a6 (1.2.1), a7 capped below a8 (1.2.2), and a10 (1.2.3). 1.2.2 removed the Cartesian field[...] assignment syntax. 1.2.3 removed the imperative gtfn backend and fixed a dace:gpu performance regression. Saying every 1.2.x release had breaking changes is an extrapolation; 1.2.0 and 1.2.1 were not shown to break anything.

## Backends and fp64

| Backend | Target | fp64 | Status and notes |
|---|---|---|---|
| Cartesian numpy, debug | CPU | yes | Confirmed in README. Reference use only. |
| Cartesian gt:cpu_ifirst, gt:cpu_kfirst | CPU (GridTools C++) | yes | Confirmed. Needs C++ toolchain at JIT time. kfirst matches a z-fastest layout. |
| Cartesian dace:cpu, dace:cpu_kfirst | CPU (DaCe) | yes | Confirmed. DaCe alpha pin. |
| Cartesian gt:gpu, dace:gpu | NVIDIA CUDA (ROCm via extras) | yes (hardware rate) | Confirmed. |
| next embedded | CPU, numpy or jax arrays | yes on numpy | Backend names for next are unverified (README lists none). |
| next gtfn_cpu, gtfn_gpu, dace_cpu, dace_gpu | C++/CUDA/HIP | yes | Names from the survey, unverified. The set of gtfn variants changed in 1.2.3. |
| ROCm 6/7 | AMD GPU | yes | Extras exist (confirmed). Auto-detect of ROCm arch in 1.2.2 per survey. |
| Metal / MPS / OpenCL | Apple GPU | n/a | Does not exist. |
| next embedded on jax under a third-party Metal plugin | Apple GPU | fp32 only | Speculative and unverified. The jax pytree (1.2.1) and Tracer dispatch (1.2.3, jax>=0.11) are confirmed, but plugin compatibility, fp32-only behavior and performance were not checked. |

fp64 on CPU and CUDA: dtype is a per-field or per-stencil parameter; this is from the survey and consistent with the project, but no document was read. A global precision switch in next is unverified.

## Idiom mapping

Support values: native, workaround, unsupported. Performance figures are the survey author's estimates and are unverified unless stated; no measurement was made. The gather-only model and the gt4py.next API spellings used below are unverified recollections.

| Idiom | Support | How | Expected perf (unverified) |
|---|---|---|---|
| I01 | native | One Field per buffer on distinct Dimensions; R buffers need their own Dimensions (not IJK storages in Cartesian). | Setup only |
| I02 | unsupported | Do in NumPy; upload psi and flags as 1-D fields. | Equal to NumPy, off hot path |
| I03 | unsupported | Host NumPy; convert fragments to masks, closure-row connectivities and coefficient fields (no ragged segments). | Setup only |
| I04 | workaround | Masks as bool fields used with where(); face sets as domain slices; gather lists as connectivity tables. | Mask form 2-6% extra work, fusable |
| I05 | native | Host constants captured as literals or scalar arguments. | Equal |
| I06 | workaround | Evaluate psi-dependent closure blocks on host (lambdified SymPy) into coefficient fields. | Setup only |
| I07 | workaround | Host NumPy produces padded neighbor tables; data-dependent footprints stay outside the DSL. | Setup only |
| I08 | workaround | Host grouping by (left_kind, right_kind, bc) producing closure tables. | Setup only |
| I09 | workaround | Host ELL tables for Bf and Br as neighbor-table offset providers. | Setup only |
| I10 | native | Not needed; ELL replaces CSR (scipy on host if CSR is kept). | Setup only |
| I11 | native | NumPy cumsum and concatenate; dedup closure table as a small field. | Setup only |
| I12 | native | Plain Python config instead of Lua. | Equal |
| I13 | native | Reshape to (nx,ny,nz) with IDim, JDim, KDim; shifts via offsets; choose z-fastest layout. | Equal if layout matches, else strided penalty |
| I14 | workaround | Keep SoA order and generate R<->D connectivities from the integer tables. | No runtime cost |
| I15 | native | Fields wrap numpy or cupy arrays zero-copy; temporaries are program-scoped. | Equal |
| I16 | workaround | Decompose: interior taps on the whole grid with a mask, then closure rows overwritten via a table. A literal ragged per-line block matvec is not expressible. | Survey says 0.8-1.5x C++ on CPU if fused; unverified. Unfused programs risk losing to a single fused C++ kernel |
| I17 | native | Shifts on IDim/JDim/KDim (next) or u[1,0,0] (Cartesian). | At or above hand C++ (expected, memory-bound) |
| I18 | native | Domain-restricted operators on wall planes. | Equal, tiny work |
| I19 | workaround | Closure rows as ELL coefficient fields plus neighbor table; map back to D with a reverse table and where(). | 0.5-1x C++ per touched row; launch-dominated at small N |
| I20 | workaround | neighbor_sum over padded ELL; no variable-length rows, no scatter or atomic add. | Comparable, with padding waste (12-16 nnz) |
| I21 | workaround | B as a D-side gather over a compact closure-row dimension. | Possibly faster than C++ full-height sweep |
| I22 | native | where(mask_slab, ...) or domain-restricted add. | Equal |
| I23 | native | ELL gather with neighbor_sum on RDim from a D-indexed source. | Equal per launch; many tiny launches unless fused |
| I24 | native | Same as I23 with an R to R table. | Equal |
| I25 | native | Disappears; output is an expression. | Better than C++ |
| I26 | native | Scalar multiply fused into the expression. | Better than C++ |
| I27 | native | Sum of dx, dy, dz, B and N in one field_operator; where(mask,new,old) for partial coverage. | Better if fused into one kernel |
| I28 | native | field_operator returning a tuple of three fields. | Equal |
| I29 | native | Pointwise expression fused with gradient outputs. | Better (fewer passes) |
| I30 | native | where(fluid_mask, du+src, du). | On par or better |
| I31 | native | Domain-restricted program calls or face masks. | Equal |
| I32 | workaround | A program with several operators compiles to one generated function. No CUDA-graph style capture exposed. Python call overhead per program is unquantified; compile the whole step. | Fusion substitutes for graphs |
| I33 | native | Python function calling programs. | Equal |
| I34 | native | Closed-form MMS as a field_operator with math functions; per-backend math coverage should be checked. | Expected faster than per-point virtual calls |
| I35 | unsupported | No per-point Python callbacks in compiled backends; use SymPy-generated closed form. | n/a |
| I36 | native | Source folded in with t as a scalar argument. | Removes host sync |
| I37 | native | Pointwise stage updates; whole RK4 can be one program. | Better when fused; call overhead matters at 2.6k points |
| I38 | native | Python reference swap. | Free |
| I39 | native | Domain-restricted boundary writes. | Better than C++ |
| I40 | native | Hoist to setup. | Free |
| I41 | workaround | No global reductions in field_operators (only neighbor reductions); do min/max with numpy or cupy on .ndarray outside. The claim that Cartesian has no reductions at all is unverified. | Equal on CPU; extra pass and sync on GPU |
| I42 | native | Error field as an operator; dump via NumPy. | Equal |
| I43 | native | Python loop. | Equal |
| I44 | workaround | Assemble matrices from the host tables with scipy.sparse. | Diagnostic only |
| I45 | native | NumPy oracles. | n/a |
| I46 | native | Field arithmetic inside operators. | Equal or better |
| I47 | workaround | Component axes as tuple fields or a data dimension; inline fluxes in derivative operators; dt reduction on host. | Good if fused |
| I48 | unsupported | Keep in NumPy/SciPy. | n/a |
| I49 | unsupported | Keep SymPy; retarget the printer to coefficient arrays. | n/a |
| I50 | unsupported | Host SciPy only. | n/a |
| I51 | native | In-process Python removes the subprocess boundary. | Faster iteration |

## Code sketches

Treat all sketches as pseudocode. The verifier noted that they use unqualified names (neighbor_sum, where, copy, Field), undefined offsets (XP, XM), a mixed-kind FieldOffset target, and API spellings (NeighborTableOffsetProvider, DimensionKind.LOCAL) that may have moved to newer Connectivity objects. They would likely not compile as written.

Per-line block matvec, shared interior taps plus closure rows as ELL overwrite:

```python
import gt4py.next as gtx
IDim=gtx.Dimension('I'); JDim=gtx.Dimension('J'); KDim=gtx.Dimension('K')
CRow=gtx.Dimension('CRow'); T=gtx.Dimension('T',kind=gtx.DimensionKind.LOCAL)
DDim=gtx.Dimension('D')   # flat D for closure gather
Ioff=gtx.FieldOffset('Ioff',source=IDim,target=(IDim,))
C2D=gtx.FieldOffset('C2D',source=DDim,target=(CRow,T))   # closure window cols
D2C=gtx.FieldOffset('D2C',source=CRow,target=(DDim,T))   # reverse (<=1 row per D)
@gtx.field_operator
def d2x(u:gtx.Field[[DDim],float64], m_int:gtx.Field[[DDim],bool],
        cC:gtx.Field[[CRow,T],float64], isC:gtx.Field[[DDim],bool],
        h2:float64)->gtx.Field[[DDim],float64]:
    # shared interior taps: strides ny*nz are an offset provider table 'XP','XM'
    interior=(u(XP)-2.0*u+u(XM))/h2
    clo=neighbor_sum(cC*u(C2D),axis=T)        # per-closure-row dense dot
    closD=neighbor_sum(clo(D2C),axis=T)       # unique row -> D (skip_value pads)
    return where(isC, closD, where(m_int, interior, 0.0))
# one table per direction; solid/Dirichlet rows get 0 via mask (explicit eq semantics)
```

O, B and Bf CSR cut-cell coupling as ELL gather:

```python
RDim=gtx.Dimension('R'); NB=gtx.Dimension('NB',kind=gtx.DimensionKind.LOCAL)
R2D=gtx.FieldOffset('R2D',source=DDim,target=(RDim,NB))   # Bf: R row -> D cols (padded to 16)
D2R=gtx.FieldOffset('D2R',source=RDim,target=(DDim,NB))   # B: D row -> R cols (compact rows)
R2R=gtx.FieldOffset('R2R',source=RDim,target=(RDim,NB))   # Br
@gtx.field_operator
def cut_terms(uD:gtx.Field[[DDim],float64], uR:gtx.Field[[RDim],float64],
              wBf:gtx.Field[[RDim,NB],float64], wBr:gtx.Field[[RDim,NB],float64],
              wB:gtx.Field[[DDim,NB],float64], hasB:gtx.Field[[DDim],bool]):
    dR = neighbor_sum(wBf*uD(R2D),axis=NB) + neighbor_sum(wBr*uR(R2R),axis=NB)
    dDB= where(hasB, neighbor_sum(wB*uR(D2R),axis=NB), 0.0)   # replaces full-height CSR
    return dR, dDB
# offset_provider={'R2D':NeighborTableOffsetProvider(tbl,RDim,DDim,16,has_skip_values=True),...}
# Gather-only (unverified): every output element pulls; padding uses skip_value=-1.
```

Gather/scatter boundary application (Dirichlet faces, fluid mask source add):

```python
@gtx.field_operator
def bc_and_src(du:gtx.Field[[DDim],float64], src:gtx.Field[[DDim],float64],
               fluid:gtx.Field[[DDim],bool], dirich:gtx.Field[[DDim],bool],
               k:float64)->gtx.Field[[DDim],float64]:
    out = k*du + where(fluid, src, 0.0)      # masked add, no index list
    return where(dirich, 0.0, out)           # Dirichlet fill as mask
# faces on the 3-D view are also domain slices:
@gtx.program
def set_face(u:gtx.Field[[IDim,JDim,KDim],float64], g:gtx.Field[[IDim,JDim,KDim],float64]):
    copy(g,out=u,domain={IDim:(0,1),JDim:(0,ny),KDim:(0,nz)})
# True scatter (index_add) is believed absent: use the inverse gather via a reverse table,
# or a where-mask over the full domain. Dense mask is 2-6% redundant but fusable.
```

Four-buffer scalar algebra with RK4 slot updates in one program:

```python
@gtx.field_operator
def axpy4(u0D:Field[[DDim],f64],u0x:Field[[RxDim],f64],u0y:Field[[RyDim],f64],
          kD:Field[[DDim],f64],kx:Field[[RxDim],f64],ky:Field[[RyDim],f64],c:f64):
    return u0D+c*kD, u0x+c*kx, u0y+c*ky          # tuple output, one kernel per dim group
@gtx.program
def rk4_step(u0D,u0x,u0y, s1D,s1x,s1y, accD,accx,accy, kD,kx,ky, dt:f64, t:f64, ...):
    rhs(u0D,u0x,u0y, t,        out=(kD,kx,ky)); axpy4(u0D,u0x,u0y,kD,kx,ky,0.5*dt,out=(s1D,s1x,s1y))
    bc(s1D,s1x,s1y,t+0.5*dt,   out=(s1D,s1x,s1y))
    rhs(s1D,s1x,s1y, t+0.5*dt, out=(kD,kx,ky))   # ...stages 2-4 analogous
# Different dimensions (D, Rx, Ry) cannot share one array; tuple fields keep them in one
# program call. Aliasing out= with an input is only safe for pointwise ops.
```

## Performance assessment

Nothing here was measured. All figures are estimates from the survey, and the per-call overhead and C++ ratio numbers are unverified recollections.

Small 2D (2.6k to 5k points, bar of roughly 30 to 40 microseconds per RHS): likely 1x to 5x slower than the C++/Kokkos graph path unless the full RK4 step is a single program. No published per-call overhead was found. The survey recollects tens of microseconds per program call (argument processing, offset-provider handling, shared-library call), low confidence. Only an amortized single-program step can compete.

Large 3D (64^3 to 256^3): plausibly at parity or better than the Kokkos block kernel, since GridTools/gtfn and DaCe generate fused, tiled, memory-bound stencils. Supporting evidence is the GMD 2026 ICON paper (Dipankar et al., GMD 19, 713-729): about 10% faster than Fortran+OpenACC near the 40-GPU R2B8 configuration, with the advantage shrinking at higher GPU counts, poor strong scaling, and a hand-written CUDA reference suggesting 2x or more is still available. Whether that dycore uses gt4py.next specifically was not checked, and the result is GPU-only, so it is indirect evidence for CPU parity here. Cut-cell R work (about 0.5% of D) is gather-based and small; the risk is indirection cost for closure rows and temporaries generated for neighbor_sum.

Dispatch overhead: undocumented. No CUDA-graph or command-buffer capture is exposed, so fusion has to replace graphs. Per-operator launches at small sizes would be unsuitable; per-step programs are fine.

Fusion: expressions inside one field_operator inline into one stencil. A program with multiple operators emits sequential kernels (gtfn) or an SDFG that may fuse maps (DaCe). Fields on different Dimensions (D versus R) are separate domains, so expect roughly 3 to 6 kernels per RHS rather than one. JIT warm-up is seconds to minutes (C++ generation, cmake, compiler), cached on disk. Connectivity values changed at runtime need no recompile, only shape or type changes do (survey claim, unverified).

## Suitability as a DSL substrate

As a code-generation target for a "tables plus stencil plus ELL gather" IR, gt4py.next offers fused stencils, neighbor_sum over padded tables, multiple dimensions, domain-restricted writes, and ahead-of-time compile with runtime-swappable table values. Downsides:

1. Gather-only model with rectangular domains (unverified but believed): no scatter or ragged segments, so the DSL must lower cut-cell fragments to masks and closure tables itself. GT4Py does not carry the hard part.
2. Experimental API, repeated breaking changes, alpha DaCe pin.
3. No Apple GPU, so the nice-to-have is unmet and GT4Py cannot be the sole substrate.
4. Heavy toolchain (C++, cmake, boost, DaCe) and slow JIT.
5. Dimensions-as-types are rigid for four buffers of unrelated length, though separate Dimensions work.
6. Reductions and dt control live outside the compiled programs.

Realistic role: one backend among several (GT4Py for CPU and CUDA, a torch/MLX or custom Metal backend for Apple), fed by a shared table-driven IR. On M-series, CPU gtfn or DaCe with fp64 is the only supported route.

## Risks

- No Metal/MPS backend and no sign of one; the Apple-GPU goal is unreachable through GT4Py itself.
- gt4py.next is experimental; 1.2.2 and 1.2.3 removed features, and DaCe is on an alpha series with a documented a8 exclusion.
- Python per-call overhead is unquantified and may miss the small-2D bar.
- Gather-only and rectangular-domain constraints push all cut-cell complexity into the DSL.
- No ragged/CSR; padding waste for r != dir Bf rows (12 to 16 nnz); extra kernels for D versus R.
- Heavy toolchain and long JIT on macOS; macOS is not in release notes or classifiers (POSIX only).
- No built-in global reductions; stats and dt need a device sync.
- Single-institution maintenance (CSCS, MeteoSwiss, ETH) driven by weather and climate priorities.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| gt4py 1.2.3 released 5 Oct 2026, Beta, Python >=3.12,<3.15, POSIX and NVIDIA CUDA classifiers, listed extras | confirmed | PyPI page. Minor omissions: version exclusions and jax-cuda extras (now included above). |
| No Metal/MPS/macOS GPU support in 1.2.0 to 1.2.3 notes, README or extras | confirmed (absence of evidence) | Releases page, README, PyPI extras. GitHub issues not searched. |
| Cartesian backends numpy, debug, gt:cpu_ifirst/kfirst, gt:gpu, dace:cpu, dace:cpu_kfirst, dace:gpu; next experimental and incompatible | confirmed | README lists exactly these eight and says next is experimental and not compatible. README lists no next backends, so those names are unverified. |
| 1.2.x churn: field[...] removal (1.2.2), imperative gtfn removal (1.2.3), DaCe a5 to a10, a8 excluded | confirmed | Release notes. Reason for the a8 exclusion not stated. "Every release breaks" is an extrapolation. |
| GMD 2026 ICON GT4Py dycore matches or exceeds OpenACC on GPUs | confirmed with qualification | About 10% faster near 40 GPUs, advantage shrinks at scale, poor strong scaling, hand-written CUDA 2x or more faster. Use of gt4py.next specifically not checked. |
| Embedded jax support (pytree 1.2.1, Tracer dispatch 1.2.3); only path to Apple GPU via third-party plugins, fp32 only | uncertain | Release-note facts confirmed. Plugin route, fp32-only and "only path" not checked. |
| gt4py.next is gather-based with neighbor_sum over local dims, no scatter | uncertain | Reference docs not read; searches returned only release lists and design notes. |
| Cartesian is in maintenance-like state while next is the future | refuted/unsupported | Release notes show active Cartesian changes (staggered fields, lazy_function, IntEnum). Dropped. |
| Per-call overhead "tens of microseconds"; "0.8-1.5x C++" and similar ratios | unverified | Author recollection; no measurement or source. |
| API spellings and code sketches (NeighborTableOffsetProvider, DimensionKind.LOCAL, sparse dimensions of size 5) | unverified | Survey admits they are from memory; recent versions may use Connectivity objects. Sketches are schematic. |
| Apple GPUs lack fp64; jax-metal is fp32 only; macOS arm64 builds fine in principle | unverified | Plausible, not checked here. |

## Sources

- https://pypi.org/project/gt4py/
- https://github.com/GridTools/gt4py
- https://github.com/GridTools/gt4py/releases
- https://gridtools.github.io/gt4py/latest/ (not read in full)
- https://gmd.copernicus.org/articles/19/713/2026/
- https://egusphere.copernicus.org/preprints/2025/egusphere-2025-4808/
- https://developer.apple.com/metal/jax/ (not verified)
- https://pypi.org/project/jax-mps/ (not verified)
- https://pypi.org/project/metaljax/ (not verified)
- https://hackmd.io/@gridtools/r1dkZt1f1x
- https://libraries.io/pypi/gt4py
