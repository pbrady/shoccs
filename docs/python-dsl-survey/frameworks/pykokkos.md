# PyKokkos (kokkos/pykokkos + pykokkos-base) as a SHOCCS substrate

## Verdict

| Item | Assessment |
|---|---|
| Fit for SHOCCS | Poor as a DSL substrate. Usable only as a CPU/CUDA prototyping layer. |
| CPU parity | Large 3D: expected near parity (same generated Kokkos C++). Small 2D: likely well below parity because of per-launch Python dispatch (unmeasured). |
| Apple GPU (MPS/Metal) | None. No Metal path documented in PyKokkos; Kokkos Metal absence is unverified but nothing was found. |
| fp64 | Yes on CPU, CUDA and HIP (`pk.double`). Not applicable to Metal. |
| Risk | High: no tagged releases, open Kokkos 5.0 migration, open macOS build bugs, cannot reuse existing C++ kernels. |

Verifier-adjusted overall score: 3 / 10.

## Status as of 2026-10-12

- Version: there is no tagged GitHub release and no current PyPI release of PyKokkos. The `pykokkos-base` PyPI package is stale (last release 0.0.7, 2022; not re-verified by the verifier). Normal installation is from source (README: conda, Python 3.13, `pip install -e .`, with cupy and patchelf for CUDA use).
- Activity (confirmed): commits continue through 2026-10-07 ("Including LICENSE"), 2026-07-06 (#431 multi-level scratch), 2026-06-29 (#426 dispatch argument parsing).
- Health (confirmed): about 127 stars, 64 open issues, 24 open PRs. Issue #369 "Kokkos 5.0 migration + 0.1 version release" is open (reopened). macOS issues #390 (OpenMP build failure) and #413 (rsqrt failure) are open. SHOCCS pins Kokkos 5.1.1, so version coupling is a live concern.
- Character: research-grade academic project (UT Austin, hosted under the Kokkos org). Not production infrastructure.
- Mechanism: type-annotated Python `@pk.workunit` functions are translated to Kokkos C++ at JIT time, compiled with g++, and cached on disk.

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| OpenMP (CPU) | Supported, the default space | Yes | Real Kokkos C++ is generated, so inner-loop quality should match hand-written Kokkos. macOS OpenMP build fails (#390). |
| Threads / Serial | Unverified | Yes if present | The survey claimed support. The verifier found only `pk.OpenMP`, `pk.Cuda`, `pk.HIP` in the docs and the README install example sets Threads OFF. Treat as unsupported until checked. |
| CUDA | Supported | Yes | Fusion and CuPy interop are CUDA oriented. |
| HIP (AMD) | Supported per policies docs | Yes | README shows no HIP build flag. |
| SYCL | Not listed in fetched docs | n/a | Treat as unsupported. |
| Metal / MPS | None | n/a | No Metal in README, policies docs or the issue list pages read. Apple GPUs have no fp64, so even a hypothetical backend would be fp32 only. |

View rank is a build-time limit (README flags `-DENABLE_VIEW_RANKS=3`, `-DENABLE_LAYOUTS=ON`). Execution spaces are chosen at pykokkos-base build time.

## Idiom mapping

Support values: native, workaround, unsupported. "Setup" means the idiom runs once at mesh/operator construction and is not on the hot path. Performance entries for kernels are expectations, not measurements.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | `pk.View([n], pk.double)` per buffer in a dict keyed by (slot, buffer), or one packed view with static offsets | Same as C++ (setup) |
| I02 | workaround | Host NumPy ray casting (vectorised quadratics, lexsort), copy to Views | Setup; NumPy likely faster than C++ virtual shape loop |
| I03 | workaround | Host NumPy merge-join producing SoA line tables, uploaded as int32 Views | Setup |
| I04 | workaround | Host NumPy masks/index arrays uploaded once as int32 Views | Setup; hoisting fixes I40 |
| I05 | workaround | Host Python/SymPy constants passed as float kwargs or small Views | Setup |
| I06 | workaround | Evaluate closures on host (lambdify/CSE over a psi array), upload table | Setup |
| I07 | unsupported | Data-dependent dispatch and ragged footprints stay on host; emit padded ELL tables | Setup |
| I08 | workaround | Host NumPy assembly grouped by closure kind | Setup |
| I09 | workaround | Host NumPy, emit ELL tables | Setup |
| I10 | workaround | `np.lexsort` / scipy COO to CSR on host, or compact COO | Setup |
| I11 | workaround | `np.concatenate`/`cumsum` then upload; dedupe closures | Setup |
| I12 | workaround | Plain Python config instead of Lua | Setup |
| I13 | native | Flat 1D Views with manual index arithmetic `i*ny*nz + j*nz + k`, or 3D View with LayoutRight | Same as C++ |
| I14 | native | Array convention only; build tables with consistent lexsort | n/a |
| I15 | native | Views passed as workunit keyword arguments; scratch buffers are Views | Same as C++ inside a kernel |
| I16 | workaround | (a) `TeamPolicy` with `TeamThreadRange` reading a 13-int meta View, near port of the C++ kernel (compile status unverified); (b) preferred: `RangePolicy` over the fast index with per-direction stride, separate interior and closure kernels | (a) about C++ on host plus glue; (b) possibly better on the slow axis; needs measurement |
| I17 | native | `@pk.workunit` with `RangePolicy`, inner loop over taps with stride. No documented `MDRangePolicy`, so 2D/3D decoding is manual | Equal to hand-written Kokkos; memory bound |
| I18 | native | Small `RangePolicy` workunit over wall-plane rows with dense R x T coefficient View | Equal to C++; launch latency bound when small |
| I19 | native | ELL workunit: per row sum over k of `C[r,k]*x[idx[r,k]]`, write `b[out[r]]` | Equal to C++ |
| I20 | native | `RangePolicy` over rows with row-pointer View | Equal to C++ |
| I21 | native | Compact COO workunit `b[row[e]] += w[e]*xR[col[e]]` (unique rows, no atomics) | Better than C++ (no full-height row pointers) |
| I22 | native | Face-slab axpy workunit | Equal or better |
| I23 | native | Fixed-width ELL gather-dot workunit | Equal to C++ |
| I24 | native | Same ELL workunit or merged into I23 | Equal |
| I25 | native | `pk.parallel_for(n, fill_zero, v=v)` or write-not-accumulate in operator kernel | Equal per launch plus Python launch cost |
| I26 | native | Scale workunit, or fold the diffusivity into stencil taps | Equal; fold gains |
| I27 | native | Sequential `parallel_for` calls, or one hand-fused kernel summing three directions | Equal; fused better but hand-written |
| I28 | native | One workunit writing three output Views, or three launches | Equal |
| I29 | native | Pointwise workunit `gx*dx + gy*dy + gz*dz`, or fused into derivative kernels | Equal |
| I30 | native | Masked workunit with uint8 mask View, or gather over int32 index View | Equal to C++ |
| I31 | native | Face fill workunit via index arithmetic or flattened index list | Equal |
| I32 | unsupported | No Kokkos Graph exposure found. Substitutes: hand-fuse the RHS, or PK_FUSION lazy fusion. No whole-RHS compile | Per-launch Python overhead; likely worse than C++ graph at small 2D |
| I33 | native | Plain Python function calling `parallel_for` sequentially; `pk.fence()` | Python launch cost per call |
| I34 | workaround | Workunit using `math.exp`/`cos` on coordinate Views. MMS must be in the restricted subset; SymPy output needs a printer for pk-compatible Python | Faster than C++ host virtual calls |
| I35 | unsupported | Arbitrary Python callbacks inside workunits are not possible (open request #394). Use a precomputed table or array-expression MMS | n/a |
| I36 | workaround | Source workunit taking `t` as scalar kwarg, launched before the RHS | Equal to C++ kernel |
| I37 | native | One axpby workunit per buffer or per packed buffer, Python loop over stages | Launch overhead dominates at about 2.6k points |
| I38 | native | `pk.deep_copy(dst, src)` | Equal |
| I39 | native | Boundary-only workunit evaluating exact solution on face/object index sets | Better than C++ (no full-field eval) |
| I40 | native | Hoist to setup | Better than C++ |
| I41 | native | `pk.parallel_reduce` with `pk.Acc`; MaxLoc likely needs two passes | Equal to C++; readback returns a Python float |
| I42 | native | Error field workunit; dump via NumPy `tofile` | IO bound |
| I43 | native | Plain Python loop | Negligible |
| I44 | workaround | Export tables to scipy.sparse and `numpy.linalg.eigvals` on host | Host fp64 |
| I45 | native | NumPy oracle; not PyKokkos relevant | n/a |
| I46 | native | Hand-written pointwise workunits, or `PK_FUSION=naive` for lazily traced sequences | Equal per kernel |
| I47 | workaround | Pointwise physics workunits plus batched derivative workunits with a component index; fixed-arity signatures make many-component systems verbose; no automatic cross-component fusion | Equal for hand-fused kernels |
| I48 | native | Stay in NumPy/SciPy | n/a |
| I49 | native | Existing SymPy layer plus a printer emitting pk-subset Python or NumPy lambdify | Setup |
| I50 | native | Host scipy | n/a |
| I51 | native | In-process Python driver replaces the subprocess bridge, if PyKokkos is importable. Install is a real source build of pykokkos-base and Kokkos | n/a |

Idioms left out entirely: I07 (ragged output), I32 (graph replay), I35 (Python callbacks in kernels). Everything else is native or needs a host-side workaround.

## Code sketches

These sketches are from the survey and are unverified: none was compiled. In particular the nested closure and 2D int Views inside `TeamPolicy` are only shown in the docs for simple nested `parallel_for`.

Per-line block matvec with strides and per-line dense closures (I16, port (a)):

```python
@pk.workunit
def block_matvec(team: pk.TeamMember, meta: pk.View2D[pk.int32], co: pk.View1D[pk.double],
                 x: pk.View1D[pk.double], b: pk.View1D[pk.double], accum: int):
    ln: int = team.league_rank()
    ro: int = meta[ln][0]; c0: int = meta[ln][1]; s: int = meta[ln][2]
    lr: int = meta[ln][3]; lc: int = meta[ln][4]; lco: int = meta[ln][5]
    ir: int = meta[ln][6]; ico: int = meta[ln][7]; w: int = meta[ln][8]
    rr: int = meta[ln][9]; rc: int = meta[ln][10]; rco: int = meta[ln][11]; rcol: int = meta[ln][12]
    def rows(r: int):
        acc: float = 0.0
        if r < lr:
            for j in range(lc): acc += co[lco + r*lc + j] * x[c0 + j*s]
        elif r < lr + ir:
            for j in range(w): acc += co[ico + j] * x[ro + r*s + (j - w//2)*s]
        else:
            q: int = r - lr - ir
            for j in range(rc): acc += co[rco + q*rc + j] * x[rcol + j*s]
        if accum == 1: b[ro + r*s] += acc
        else: b[ro + r*s] = acc
    pk.parallel_for(pk.TeamThreadRange(team, lr + ir + rr), rows)
# launch: pk.parallel_for(pk.TeamPolicy(n_lines, pk.AUTO), block_matvec,
#                         meta=meta, co=co, x=x, b=b, accum=1)
```

Cut-cell coupling to R points (I21, I23, I24):

```python
@pk.workunit
def coo_scatter(e: int, row: pk.View1D[pk.int32], col: pk.View1D[pk.int32],
                w: pk.View1D[pk.double], src: pk.View1D[pk.double], dst: pk.View1D[pk.double]):
    dst[row[e]] += w[e] * src[col[e]]          # rows unique per entry

@pk.workunit
def ell_gather(r: int, idx: pk.View2D[pk.int32], C: pk.View2D[pk.double], out: pk.View1D[pk.int32],
               srcD: pk.View1D[pk.double], srcR: pk.View1D[pk.double], nD: int, dst: pk.View1D[pk.double]):
    acc: float = 0.0
    for k in range(idx.extent(1)):             # padded to Tmax; padded C == 0
        i: int = idx[r][k]
        acc += C[r][k] * (srcD[i] if i < nD else srcR[i - nD])
    dst[out[r]] += acc

pk.parallel_for(nB, coo_scatter, row=Brow, col=Bcol, w=Bw, src=uR, dst=duD)
pk.parallel_for(nR, ell_gather, idx=Bf_idx, C=Bf_C, out=Rrows, srcD=uD, srcR=uR, nD=nD, dst=duR)
```

Gather/scatter boundary application (I30, I31):

```python
@pk.workunit
def masked_add(i: int, mask: pk.View1D[pk.uint8], src: pk.View1D[pk.double], du: pk.View1D[pk.double]):
    if mask[i] != 0:
        du[i] += src[i]

@pk.workunit
def fill_at(k: int, idx: pk.View1D[pk.int32], val: pk.View1D[pk.double], u: pk.View1D[pk.double]):
    u[idx[k]] = val[k]                          # unique indices, no atomics

pk.parallel_for(nD, masked_add, mask=fluid, src=src, du=du)
pk.parallel_for(n_dir, fill_at, idx=dir_idx, val=g_t, u=u)
```

RK4 slot updates over four buffers (I37, I38):

```python
@pk.workunit
def lc(i: int, d: pk.View1D[pk.double], a: pk.View1D[pk.double], c: float, r: pk.View1D[pk.double]):
    d[i] = a[i] + c * r[i]

@pk.workunit
def acc(i: int, d: pk.View1D[pk.double], c: float, r: pk.View1D[pk.double]):
    d[i] += c * r[i]

def rk4_step(u0, u1, rk, k, sizes, rhs, ub, t, dt):    # each slot = tuple of 4 Views (D,Rx,Ry,Rz)
    for b in range(4): pk.deep_copy(u1[b], u0[b]); rk[b].fill(0)
    cs = [(0.0, dt/6), (dt/2, dt/3), (dt/2, dt/3), (dt, dt/6)]
    for s, (ca, cb) in enumerate(cs):
        if s > 0:
            for b in range(4): pk.parallel_for(sizes[b], lc, d=u1[b], a=u0[b], c=ca, r=k[b])
            ub(u1, t + ca)
        rhs(u1, k, t + ca)
        for b in range(4): pk.parallel_for(sizes[b], acc, d=rk[b], c=cb, r=k[b])
    for b in range(4): pk.parallel_for(sizes[b], lc, d=u1[b], a=u0[b], c=1.0, r=rk[b])
# about 36 launches per step; pack D|Rx|Ry|Rz into one View to cut 4x.
```

## Performance assessment

All figures below are estimates. No authoritative PyKokkos launch-overhead benchmark was found, and the verifier could not confirm one.

- Kernel bodies: compile to ordinary Kokkos C++, so per-kernel time should match hand-written Kokkos on the same space.
- Small 2D (about 2.6k points): launch cost dominates. The survey estimated tens of microseconds per Python-dispatched call (unmeasured) and 160 to 300 launches per heat step, giving roughly 6 to 15 ms against an estimated 0.3 to 1 ms for the C++ graph path, a 3 to 10x slowdown. Treat this as a hypothesis to measure, not a result. The only related datum, the fusion page's 27.2 s for 100k batches on GPU with CuPy, implies about 272 us per batch, but that is a different regime and says little about host launch cost. Open issue #402 (slow program exit after `parallel_` calls) is a further hint of overhead.
- Large 3D (for example 256^3): launch overhead amortises over millisecond kernels, so parity with C++ is expected, provided the kernel is written as a `RangePolicy` over the fast index. A literal port of the team-per-line kernel inherits its strided-axis weakness. Only `RangePolicy` and `TeamPolicy` are documented, so 3D tiling must be hand written.
- Dispatch: Python argument parsing was refactored in #426. First call of each workunit triggers C++ generation and a g++ compile (seconds to tens of seconds), cached on disk. Issue #437 suggests view-type precompilation is already a startup cost.
- Fusion: PyFuser is enabled by `PK_FUSION` (docs example `naive`, off by default). It lazily traces consecutive `parallel_for` calls and fuses them when a result is requested. Docs benchmark (confirmed): 100,000 batches of a `parallel_for` over 10 work units on CuPy arrays with the Cuda space, 27.213 s to 14.840 s, about 1.8x, on one machine; the page says it is most effective on GPU spaces. The paper's "up to 3.8x" is unverified. No evidence it fuses TeamPolicy kernels, reductions, or gather/scatter over different index spaces. It is not whole-step graph replay, and there is no jax.jit equivalent.

## Suitability as a DSL substrate

Weak to moderate, leaning weak.

- For: it is a Python-to-Kokkos generator, so a DSL could emit workunit source and inherit Kokkos-grade fp64 kernels on CPU/CUDA/HIP. A precomputed-table design (ELL/COO/closure tables) maps cleanly onto `RangePolicy` workunits and handles the cut-cell unstructuredness well.
- Against: no whole-RHS compile or graph, so the small-2D bar is hard to meet; restricted Python subset with fixed-arity typed signatures; only `RangePolicy` and `TeamPolicy` documented; g++ JIT with cold-compile cost (compiler fixed to g++, #372); no Metal; thin maintenance.
- Cannot reuse the existing C++ Kokkos kernels (they would be rewritten as Python workunits, or wrapped by a separate pybind11 module outside PyKokkos). Embedding in a C++ driver is unsupported, and calling Python from C++ is only a feature request (#394).
- It adds little over the C++ Kokkos code already in the repo except Python authoring. A DSL wanting backend control is better off emitting Kokkos C++ directly for CPU/CUDA and using a separate Metal/MLX/torch path. PyKokkos is at best a prototyping layer.

## Risks

1. No Metal/MPS: fails the Apple GPU nice-to-have, and fixing it needs a new Kokkos execution space (multi-person-year work; Kokkos Metal absence itself is unverified).
2. Python dispatch cost per kernel threatens the small-2D performance bar, with no graph replay (unmeasured, must benchmark).
3. Project health: no releases, stale pykokkos-base on PyPI, 64 open issues, Kokkos 5.0 migration (#369) in flux against SHOCCS's Kokkos 5.1.1, macOS failures (#390, #413). The developer machine is macOS, so CPU use there is itself at risk.
4. Documentation gaps: nested reduce, `ThreadVectorRange`, `MDRangePolicy` not documented (not proven absent). The TeamPolicy block-matvec port is unverified.
5. JIT and g++ dependency, conda/cupy/patchelf install path, build-time View rank and layout flags complicate CI and the Spack toolchain.
6. Existing C++ kernels are not reusable; work is duplicated.
7. Restricted Python subset blocks callbacks (#394) and complex kernel control flow.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| Supports CPU (OpenMP), CUDA, HIP; no Metal/MPS | Confirmed | README lists CPUs, NVIDIA and AMD GPUs; policies page lists `pk.OpenMP`, `pk.Cuda`, `pk.HIP`; no Metal mentioned. Only the first 100k characters of the issue list were searched. |
| "Threads/Serial" CPU backends | Weakly supported (corrected to unverified) | Not in the docs; README install example sets Threads OFF; Serial undocumented. |
| Kokkos itself has no Metal backend | Uncertain | Web search found nothing, which is absence of evidence. The Kokkos core wiki backend list was not checked directly. |
| Policies are RangePolicy and TeamPolicy only; nested reduce, ThreadVectorRange, MDRange undocumented | Confirmed (as "not documented") | Policies page documents only those two, with `TeamThreadRange` in the example. Not documented is not unimplemented; #426 mentions "team policy inner loop arguments". |
| PyFuser lazy, `PK_FUSION`, 27.2 s to 14.8 s on 100k calls | Confirmed | Fusion page: `PK_FUSION="naive"`, off by default, 27.213 s to 14.840 s, about 1.8x, CuPy with Cuda, effective mainly on GPU spaces. Paper's 3.8x not verified. |
| Active, no releases, commits through 2026-10-07, #369, #390, #413 open | Confirmed | GitHub page and API: 127 stars, 64 open issues, 24 PRs, commits dated 2026-10-07, 2026-07-06, 2026-06-29. pykokkos-base 0.0.7 staleness not re-checked. |
| Python callbacks in kernels unsupported (#394); C++ kernels not reusable | Confirmed | #394 open, created 2026-03-10. Reuse limitation follows from the Python-to-C++ translation architecture; not separately verified. |
| Per-launch dispatch is tens of microseconds and exceeds C++ at small 2D | Uncertain | No benchmark found. Fusion-page numbers are a different regime. #402 open, #426 refactor. Carried as a risk to measure. |
| Score and conclusion (not a suitable substrate) | Holds | Core facts confirmed; the two uncertain claims do not change the conclusion. Adjusted overall stays 3. |

Other notes from the verifier: the survey's I03 entry had a malformed `unused` key (formatting artifact, ignored here); the View-rank build limit and the conda/cupy/patchelf install dependencies were missing from the survey and are added above; Python subset details and View dimension limits in the survey were from memory and remain low confidence.

## Sources

- https://github.com/kokkos/pykokkos
- https://kokkos.org/pykokkos/
- https://kokkos.org/pykokkos/policies.html
- https://kokkos.org/pykokkos/fusion.html
- https://users.ece.utexas.edu/~gligoric/papers/AlAwarETAL25PyFuser.pdf
- https://conf.researchr.org/details/issta-2025/issta-2025-papers/81/Dynamically-Fusing-Python-HPC-Kernels
- https://api.github.com/repos/kokkos/pykokkos/commits?per_page=5
- https://api.github.com/repos/kokkos/pykokkos/issues?state=open&per_page=100
- https://github.com/kokkos/pykokkos/issues/394
- https://pypi.org/project/pykokkos-base/0.0.7rc5
- https://www.piwheels.org/project/pykokkos-base/
- https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/HierarchicalParallelism
