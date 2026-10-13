# Halide (Python bindings) as a SHOCCS execution layer

| Verdict box | |
|---|---|
| Fit for SHOCCS | Good CPU/CUDA execution substrate under a custom DSL; geometry must be flattened to masks and ELL tables on the host. Overall 6.3 / 10 (verifier-adjusted). |
| CPU parity | Likely parity or better (8/10), via whole-step fusion. Unmeasured: all numbers are reasoned estimates. |
| Apple GPU | Weak (4/10). `host-metal` exists but is effectively fp32 only; it uses Halide's own Metal runtime, not MPS/MPSGraph. |
| fp64 | Yes on CPU and CUDA. No on Metal (64-bit float literals are a hard codegen error; Apple GPUs lack fp64). |
| Risk | Medium. Mature project, but manual scheduling, hand-built sparse forms, JIT latency, and unproven fp32 closures. |

## Status as of 2026-10-12

- Latest release found: v21.0.0, published 2025-09-16 (confirmed via GitHub API and PyPI). It requires LLVM 21.1.1 or newer (21.1.0 has an NVPTX bug) and drops Python 3.8. PyPI lists `requires_python >=3.9` and 27 wheels.
- Pip wheels with the Python bindings have existed since 19.0.0 (2024-12-17).
- The v21 notes say the Python bindings were substantially improved with many missing bindings filled in. v17 replaced ParamMap with Callable. v18 added proper bfloat and int64 support in Python.
- GPU autoscheduling: Anderson2021 has existed since v16; v21 added experimental GPU scheduling to Mullapudi2016.
- No newer release than v21.0.0 was found. Package-index tags such as `halide-llvm@22.1.7` hint at an LLVM 22 development line, but no Halide 22 release was seen.
- Halide targets image and array pipelines. It has no sparse or ragged concept and no time-loop construct.

## Backends and fp64

| Backend | Status | fp64 |
|---|---|---|
| CPU (x86, ARM64 via LLVM) | Mature, vectorized, built-in thread pool | Yes |
| CUDA (NVPTX) | Mature; needs LLVM 21.1.1+ in v21 | Yes |
| Metal (`host-metal`) | Emits MSL through Halide's own runtime, not MPS. Release-note activity is minor (dot() in v16, faster codegen in v17, highlighting in v21) | No. `CodeGen_Metal_Dev.cpp` errors on 64-bit float literals ("Metal does not support 64-bit floating point literals"); Apple GPUs have no fp64 |
| OpenCL | Available | Device dependent |
| Vulkan | Release notes say it "matured substantially" in v19; the earlier "beta" label may be stale | Device dependent |
| D3D12, WebGPU, Hexagon | Available | Not assessed |

Apple-GPU port options: fp32 only, or hand-written double-float emulation under `strict_float`, which is slow and fragile (the claim that MLX maintainers consider emulated fp64 too slow is unverified). Practical plan: fp64 reference on CPU/CUDA, Metal as an experimental fp32 target.

## Idiom mapping

Support: native, workaround (expressible via host preprocessing or gather forms), none. Performance is a reasoned estimate unless stated.

| ID | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | Flat packed state [D, Rx, Ry, Rz] as one 1-D `hl.Buffer(float64)` or four Buffers; static offsets are pipeline constants | Equal to C++ |
| I02 | workaround | Ray casting in NumPy/Numba on host; pass SoA tables as input Buffers. Halide is poor at sorting and compaction | Setup only |
| I03 | workaround | Host NumPy (searchsorted/lexsort) builds fragment tables, flattened to per-point masks and row-id maps | Setup only |
| I04 | native | Masks (uint8) and int32 index tables built on host; faces via `select` on coordinate Vars | Equal; masks suit GPU |
| I05 | native | Python floats baked as Expr constants, or ImageParam taps; compile-time taps allow unrolling | Equal or better |
| I06 | workaround | Closed forms evaluated host-side with `sympy.lambdify` in fp64, or as Halide Exprs over R points (poles need clamps) | Setup only |
| I07 | workaround | Host Numba/NumPy per-pencil searchsorted; emit padded ELL (idx, coef) tables | Setup only; faster than C++ O(R) scan |
| I08 | workaround | Host builds closure table, int32 row-id map (-1 if none), interior mask | Setup only |
| I09 | workaround | Host composes derivative x interpolation coefficients into ELL tables over the concatenated state | Setup only |
| I10 | native | scipy / `np.lexsort`; keep compact ELL or COO, avoid full-height row pointers | Setup only |
| I11 | native | `np.concatenate`/`cumsum`/`unique`; no per-line allocations | Setup only |
| I12 | native | Pure Python config replaces Lua; not Halide's job | N/A |
| I13 | native | `hl.Buffer` from a NumPy array; Halide dim 0 is fastest, so reverse axes or declare Vars as (z,y,x). Use Int(64) Exprs beyond 2^31 points | Equal |
| I14 | native | R ordering is the Buffer index; contract lives on the Python side | Equal |
| I15 | native | Func inputs/outputs are Buffers; scratch is inlined or a `compute_root` intermediate | Equal or better |
| I16 | workaround | Per-line block matvec restructured to per-point gather: `select(rid>=0, closure, select(interior, stencil, old))`. No scatter. See sketch 1 | Est. 1.0 to 2x faster than C++ on CPU; costs a D-sized int32 row-id map unless closures are confined to boundary slabs |
| I17 | native | Sum of taps over `BoundaryConditions.repeat_edge(u)`, `vectorize` inner, `parallel` outer, three axes fused by inlining | Est. 0.9 to 1.3x vs best C++ |
| I18 | native | Constant R x T matrix as explicit Expr sums under `select(x < R, ...)` | Parity |
| I19 | workaround | Per-closure-row ELL (C, IDX) with an RDom over k, reached through the inverse row-id map; reads u via a flat alias ImageParam | Parity on CPU; indirect-gather divergence on GPU (about 2 percent of points) |
| I20 | workaround | No sparse type. ELL as `sum_k W(k,r)*x(COL(k,r))`; ragged CSR needs data-dependent RDom extents and parallelizes poorly | Parity for ELL; prefer ELL over CSR |
| I21 | workaround | Fold B into the closure ELL by indexing into the concatenated [D;R] state | Est. faster than C++ (no full-height row-pointer sweep) |
| I22 | native | Neumann term as `select(x<X, extra[x]*g(y,z), 0)`; omitted at construction if absent | Equal or better |
| I23 | workaround | Tiny pure Func over q: `sum_k W(k,q)*s(IDX(k,q))` with padded ELL; merge Bf and Br tables over the concatenated source | Parity on CPU; latency bound on Metal unless merged into one pipeline |
| I24 | workaround | Merge into the I23 table over [D;R]; no dependency chain in gather form | Parity or better |
| I25 | native | Zero-fill/accumulate chain disappears; output is one pure Func summing directional terms | Better (fewer passes) |
| I26 | native | Scale folded into the output Expr | Better (fused) |
| I27 | native | `du = Ox(u)+Oy(u)+Oz(u)` plus closure terms, `select`/mask for partial coverage. Corner points where closure rows overlap need one table per direction, summed | Est. better at large N |
| I28 | native | Three Funcs (or a component dim of 3) consumed by the dot product, fused | Better (no scratch fields) |
| I29 | native | `gGx*dux + gGy*duy + gGz*duz` inlined | Better |
| I30 | native | Dense masked form `du + select(fluid_mask, src, 0)`. Update-definition scatter is serial or needs atomics; avoid (not re-verified) | Parity or better |
| I31 | native | `select(x==0, g(y,z), v)` for faces; `select(dirichlet_mask(i), val(i), v)` for R points | Parity |
| I32 | native | One `hl.Pipeline` compiles the whole RHS or RK step into a single call; no graph, no pointer stability issue | Est. beats C++ at small sizes (about 40 nodes become one call) |
| I33 | native | A Python function builds the Funcs; no per-op fences | Equal or better |
| I34 | native | Closed-form Gaussian MMS as an Expr (`hl.exp`, `hl.cos`, `hl.pow`), t as a scalar param, fused into RHS | Much faster than C++ virtual host calls (est.) |
| I35 | native | Lua callbacks replaced by generated Exprs (SymPy to Halide) or tables | Faster than Lua |
| I36 | native | Source evaluated in-pipeline with t as a scalar param; no host sync | Better |
| I37 | native | Stage updates as fused Exprs over the packed state; whole RK4 step in one pipeline with `compute_root` at RHS boundaries (see sketch 4). Each stage RHS needs full-grid materialization | Est. better (about 36 launches become a handful of loop nests) |
| I38 | native | Swap input/output Buffers per call; no pointer baking | Better (no copies) |
| I39 | native | Exact solution evaluated only on boundary slices, fused into the stage Func | Better |
| I40 | native | Gather lists hoisted to setup as masks/Buffers | Better |
| I41 | workaround | RDom reductions (`hl.maximum`, `hl.minimum`, `hl.argmax`) into a tiny Buffer read back every K steps. CPU parallelism needs `rfactor` | Parity on CPU if scheduled; serial if naive; slower on Metal |
| I42 | native | Error field as a Func; read back as NumPy, `np.tofile`; dump gating in Python | Equal |
| I43 | native | Host Python loop calls the compiled Callable per step. Call overhead (tens of microseconds) is an unmeasured estimate | Fine versus a 0.3 to 1 ms step; possibly 5 to 15 percent at smallest sizes |
| I44 | native | Export tables to scipy.sparse and `numpy.linalg.eigvals`, or apply the pipeline to unit vectors | N/A |
| I45 | native | NumPy oracles | N/A |
| I46 | native | Func expressions act as expression templates with real fusion | Better |
| I47 | workaround | Func with component dim 5, pointwise nonlinear Exprs inlined, batched derivatives. Flux feeds a stencil, so use `compute_at` to avoid recompute vs temporaries. fp32 Euler on Metal is an open accuracy question | Parity if scheduled well; high scheduling effort |
| I48 | native | NumPy/SciPy as today | N/A |
| I49 | native | SymPy to Halide Expr printer (retarget CSE codegen) or `sympy.lambdify` for host batches | Setup only |
| I50 | native | Stays in SciPy on host in fp64 | N/A |
| I51 | native | In-process Python call to the pipeline replaces the subprocess boundary | Much faster per parameter point |

Not supported or only awkward: scatter updates (serial or atomic), ragged CSR with per-row extents, any sparse type, and fp64 on Metal.

## Code sketches

These are illustrative and untested. The survey's original first sketch contained `hl.sum_taps = sum(...)`, which is not a Halide API; it is replaced here by a plain Python `sum()` of Exprs. API spellings (`hl.Param`, `compile_to_callable`) are plausible but not checked against v21 bindings.

### 1. Per-line block matvec flattened to a row-id gather

```python
import halide as hl
x, y, z = hl.Var('x'), hl.Var('y'), hl.Var('z')
u    = hl.ImageParam(hl.Float(64), 3, 'u')     # D as (nz,ny,nx) after axis reversal
uf   = hl.ImageParam(hl.Float(64), 1, 'uf')    # same memory, flat [D|Rx|Ry|Rz]
rid  = hl.ImageParam(hl.Int(32), 3, 'rid')     # closure row id per D point, -1 = none
mask = hl.ImageParam(hl.UInt(8), 3, 'imask')   # 1 = interior stencil row
C    = hl.ImageParam(hl.Float(64), 2, 'C')     # C[k,row] closure coeffs, padded TMAX
IDX  = hl.ImageParam(hl.Int(32), 2, 'IDX')     # flat column index per (k,row)
k = hl.RDom([hl.Range(0, TMAX)])
ub = hl.BoundaryConditions.repeat_edge(u)
interior = sum(c[j] * ub(x + (j - P), y, z) for j in range(2 * P + 1))  # axis shown for x
r = hl.max(rid(x, y, z), 0)
closure = hl.sum(C(k, r) * uf(IDX(k, r)))
old = u(x, y, z)
du = hl.Func('du')
du[x, y, z] = hl.select(rid(x, y, z) >= 0, closure,
                        hl.select(mask(x, y, z) != 0, interior, old))
du.parallel(y).vectorize(x, 8)   # one such Func per axis, summed for the laplacian
```

### 2. CSR cut-cell coupling as an ELL gather over the concatenated state

```python
s   = hl.ImageParam(hl.Float(64), 1, 's')      # flat [D|Rx|Ry|Rz]
W   = hl.ImageParam(hl.Float(64), 2, 'W')      # W[k,q] padded ELL, k < KMAX (12-25)
COL = hl.ImageParam(hl.Int(32), 2, 'COL')      # column in concatenated state; pad has W=0
q = hl.Var('q'); k = hl.RDom([hl.Range(0, KMAX)])
dR = hl.Func('dR')
dR[q] = hl.sum(W(k, q) * s(COL(k, q)))
dR.compute_root().parallel(q, 16)
# B (R->D) is folded into the closure ELL of sketch 1: those rows index the R
# portion of s via COL, so no full-height CSR sweep exists. s holds the old
# state, so R-to-R dependencies stay correct.
```

### 3. Boundary application as selects instead of scatter

```python
t    = hl.Param(hl.Float(64), 't')
dmsk = hl.ImageParam(hl.UInt(8), 1, 'dmsk')   # object-Dirichlet mask over R
i = hl.Var('i')
sol = lambda xx, yy, zz: gauss_sol(xx, yy, zz, t)   # Python builds an hl.Expr
v = hl.Func('v')
v[x, y, z] = hl.select((x == 0) | (x == NX-1) | (y == 0) | (y == NY-1),
                       sol(X[x], Y[y], Z[z]), u_new(x, y, z))
vR_out = hl.Func('vR_out')
vR_out[i] = hl.select(dmsk(i) != 0, solR(i, t), uR_new(i))
# A scatter update definition f[IDX(r)] = val(r) exists but is serial on CPU
# and needs atomic() on GPU; the mask gather avoids it.
```

### 4. RK4 on the packed state, one pipeline per step

```python
S  = hl.ImageParam(hl.Float(64), 1, 'S')       # packed [D|Rx|Ry|Rz]
dt = hl.Param(hl.Float(64), 'dt'); t = hl.Param(hl.Float(64), 't')
i = hl.Var('i')
k1 = build_rhs(S, t);                         k1.compute_root()
u2 = hl.Func('u2'); u2[i] = bc(S[i] + 0.5*dt*k1[i], t + 0.5*dt)
k2 = build_rhs(u2, t + 0.5*dt);               k2.compute_root()
u3 = hl.Func('u3'); u3[i] = bc(S[i] + 0.5*dt*k2[i], t + 0.5*dt)
k3 = build_rhs(u3, t + 0.5*dt);               k3.compute_root()
u4 = hl.Func('u4'); u4[i] = bc(S[i] + dt*k3[i], t + dt)
k4 = build_rhs(u4, t + dt);                   k4.compute_root()
out = hl.Func('out')
out[i] = bc(S[i] + dt/6*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]), t + dt)
call = hl.Pipeline(out).compile_to_callable([S, dt, t, *tables])
```

## Performance assessment

All figures below are estimates; the survey found no measured Halide cut-cell or PDE benchmark.

- Small 2D: likely parity or faster if the whole step is one Pipeline. The C++ bar is about 30 to 40 us per RHS and 0.3 to 1 ms per step, dominated by about 280 launches and host MMS evaluations, which fusion removes. Tiny R tables are cheap on CPU; on Metal many tiny kernels are launch-latency bound, so keep Metal off at this size.
- Large 3D: memory-bound stencils are Halide's strength. Expect parity to 2x over the C++ one-team-per-line kernel, which loses locality on the slow axis. Risk: D-sized row-id and mask maps add traffic (10 to 30 percent, unless closures are confined to boundary slabs), and flat int32 indices cap at 2^31 points. The 256^3 parity claim is an extrapolation.
- Dispatch overhead: one Pipeline per step replaces about 280 launches. Python Callable overhead (tens of microseconds) and JIT compile time (seconds to tens of seconds for a large fused RK4 step) are unmeasured. No persistent JIT cache was confirmed, so keep Callables in process or compile AOT via Generators. Halide's own thread pool can oversubscribe with Kokkos/OpenMP if mixed.
- Fusion: whole-RHS or whole-RK4 fusion is Halide's core feature, controlled by `compute_at`, `compute_root`, `store_at`. Stage boundaries (stencil of stencil) force materialization or recompute, a manual schedule decision. The step loop stays in Python or C.

## Suitability as a DSL substrate

Strong candidate as the CPU/CUDA back end under a custom Python DSL. The algorithm/schedule split fits a DSL: the front end emits Funcs from stencil and closure definitions, and a separate schedule picks fusion and parallelism. Cut-cell unstructuredness is absorbed by a host-side precompute (masks, row-id map, ELL tables) that Halide consumes as ordinary Buffers, so pipelines are geometry independent and need no recompile per geometry. Costs: all irregular indexing is hand-built gather form; schedules are manual and target specific; JIT latency; no fp64 on Metal; and the Python tracing front end and geometry precompute are the real work. Interop: `hl.Buffer` supports the buffer protocol for NumPy (documented at halide-lang.org/docs/Python.html, not re-fetched during verification). AOT output can wrap Kokkos host pointers via `Halide::Runtime::Buffer` for C++ use. Versus JAX, torch.compile or Taichi: best control and likely best CPU stencil speed, but least ecosystem, no practical autodiff, and a smaller contributor pool.

## Risks

- No fp64 on Metal; fp32 numerics for high-order closures with small psi are unproven.
- Gather/ELL formulation adds index-map bandwidth at 3D.
- Poor schedules lose to C++; GPU schedules differ from CPU ones; GPU autoschedulers are experimental for this workload.
- LLVM compile latency for a large fused RK4 pipeline.
- Reductions and atomics are weaker on Metal; a per-step device-to-host sync is needed for stats.
- Thread-pool conflicts with Kokkos/OpenMP.
- Niche community for PDE use and API churn across releases (v17 ParamMap removal, v21 dropping Python 3.8).
- Euler (multi-component, nonlinear flux) raises scheduling complexity.
- No benchmarks were run; all performance claims are unmeasured.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| v21.0.0 latest, LLVM 21.1.1+, drops Py3.8, published 2025-09-16, wheels since 19.0.0 | Confirmed | GitHub API (published_at 2025-09-16), release notes, PyPI JSON (27 wheels), 19.0.0 uploaded 2024-12-17 |
| Late-May-2026 mirror lists v21.0.0 as latest | Unverified | Not checked; an LLVM 22 tag hint exists but no Halide 22 release found |
| Metal/Apple GPUs have no fp64; Halide cannot run Float(64) on host-metal | Confirmed, with correction | Apple forum; `CodeGen_Metal_Dev.cpp` errors on 64-bit float literals. The type printer still maps to `double`, so "codegen rejects Float(64)" is imprecise |
| v21 requires Python 3.9+/3.10+ | Corrected | PyPI says `>=3.9`; "3.10+" unsupported |
| Metal work ongoing; OpenGLCompute removed in v18; Vulkan beta | Unverified | Metal changes confirmed but minor; v18 removal not seen in fetched notes; v19 says Vulkan matured, so "beta" may be stale |
| Buffer protocol zero-copy NumPy, Callable, Generators; v21 bindings improved | Confirmed (release notes); buffer protocol not re-fetched | Release notes v16 to v21; docs page |
| Experimental GPU Mullapudi2016 in v21 | Confirmed, with correction | Notes say so; Anderson2021 GPU autoscheduler existed since v16 |
| Callable overhead tens of us; JIT seconds to tens of seconds; no persistent JIT cache | Unverified | No source either way; survey labels them estimates |
| MLX maintainers say emulated fp64 too slow | Unverified | Issue not fetched; consistent with conclusion |
| No sparse primitive; scatter serial or atomic on GPU | Unverified | Consistent with Halide's model, not sourced this session |
| Sketch uses `hl.sum_taps` | Refuted | Not a Halide API; replaced with `sum()` |
| `hl.Param`, `compile_to_callable` spellings | Unverified | Plausible, not checked |
| Performance ratios (0.9 to 2x, 5 to 15 percent) | Unverified | Reasoned estimates, no measurements |

## Sources

- https://github.com/halide/Halide/releases
- https://api.github.com/repos/halide/Halide/releases/latest
- https://pypi.org/pypi/halide/json
- https://github.com/halide/pypi
- https://halide-lang.org/docs/Python.html
- https://halide-lang.org/docs/md_doc_2_python.html
- https://halide-lang.org/docs/Vulkan.html
- https://github.com/halide/Halide/blob/main/src/CodeGen_Metal_Dev.cpp
- https://developer.apple.com/forums/thread/709663
- https://upd.dev/ml-explore/mlx/issues/1905
- https://skia.googlesource.com/skia/+/71a35d49b73448e646c82538ecb009fc68471580
- https://halide-lang.org/tutorials/
