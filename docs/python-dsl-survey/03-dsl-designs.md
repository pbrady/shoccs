# Part III: A purpose-built Python DSL for SHOCCS

Status as of 2026-10-12. This is the third part of the survey on re-expressing the SHOCCS C++ solver in Python. Part I is the idiom catalog (the 51 idioms I01 to I51, the seven-primitive set P0 to P6 and the performance bar). Part II is the framework survey (`docs/python-dsl-survey/frameworks/*.md`, one file per package) together with `docs/python-dsl-survey/00-apple-gpu-facts.md`. This part asks a different question: if no package fits, what would we build?

How to read this document:

- Sections 1 and 2 set out the design space and the five angles that were explored.
- Sections 3 to 7 give one full section per design, written to the same template so they can be compared column by column.
- Section 8 is the judging: the rubric, three judges' scores, the combined ranking, the grafts (ideas worth moving between designs) and the blind spots all five share.
- Section 9 is the recommended hybrid, a single concrete proposal with a phased roadmap. Its first phase is the smallest experiment that de-risks the performance claim.
- Section 10 lists the evidence that would change the recommendation.

Three warnings apply to everything below.

1. Every performance number attributed to the current C++ comes from `benchmarks/baselines/baseline-20260320-024030.json`: Linux aarch64 binaries run in a 4-vCPU container, internally noisy by about 2x, with no cut-cell, 2D or end-to-end-step measurement. No native macOS build existed in this worktree when the designs were written (`build/` was absent). Every speedup quoted in a design is therefore a projection against an estimated bar, and is labelled as such. The regime-C step bar in each design's performance table (240 ms to 1 s) is that design's own extrapolation from Part I's RHS range of 30 to 110 ms plus 30 to 60 ms of slot-op traffic per step; Part I gives no step figure at 256^3. Since the designs were written, the C++ was measured again inside the project devcontainer (`05-measurements.md`, Part IV, 2026-10-12: Docker Desktop on the owner's Mac, 8 vCPU, Kokkos OpenMP, Release, with cut-cell 2D and end-to-end steps included). Those figures supersede the March baseline wherever a matching case exists. In Sections 3 to 7, every C++ number comes from the March baseline the designers worked from; Sections 8 to 10 and Part IV use the fresh measurements. There was no native macOS build until 2026-10-13; since then Sections 8.5, 9.1, 9.2 and 10 quote the native figures of `05-measurements.md` Section E with the container figures beside them.
2. Every design abandons bitwise parity with the C++ at uncovered points (solid points, Dirichlet rows, Dirichlet-skipped lines), where the C++ leaves stale values (`src/matrices/block.hpp:173-175`, the `eq_t` partial-coverage semantics). Parity is tolerance-based on fluid D rows and R rows.
3. Claims about third-party frameworks are sourced to the Part II file for that framework, which records the verification status of each claim. Where the Part II file marks a claim unverified, so does this document.

Naming note: two of the five designs were independently named "kerf" (a kerf is the cut a blade leaves). They are different designs. This document labels them D1 kerf-forge (the SymPy-to-C++/MSL code generator) and D4 Kerf-lang (the Python-subset kernel language). The judges' score sheets in Section 8 used "Kerf-A"/"kerf" for D1 and "Kerf-B"/"Kerf" for D4.

Primitive labels: Part I's seven kernel shapes keep their labels P0 to P6. The backend primitives of D5 (Section 7) and of the hybrid (Section 9) are labelled B1 to B11 (the same order and meanings as the P1 to P11 of earlier drafts), so that P3 always means Part I's ELL gather-dot and B3 always means the backend axis stencil. D3's own pass label "P4 stage fusion" in Section 5.7 is unrelated to both and is left as is.

## 1. The design space

Part I established that the runtime of SHOCCS has exactly seven kernel shapes (P0 field container, P1 fused elementwise map, P2 constant-coefficient axis stencil, P3 fixed-width ELL gather-dot, P4 face plane assign, P5 masked reduction, P6 whole-step compilation) and that cut-cell irregularity enters in exactly six places: fragmented lines (`src/mesh/mesh.cpp:30-87`, `src/operators/derivative.cpp:305-449`), irregular-length R buffers (`src/mesh/object_geometry.cpp:37-114`), psi-dependent closure blocks (`src/operators/derivative.cpp:336-413`), the CSR couplings B/N/Bf/Br (O is the block operator) (`src/operators/derivative.cpp:71-257, 440-448`; `src/matrices/csr.hpp:16-24`), gather lists instead of masks (`src/fields/selection_desc.hpp`; `src/mesh/mesh.hpp:95-118`), and partial-coverage write semantics (`src/matrices/block.hpp:173-175`).

Every design in this part therefore faces the same four decisions, and the designs differ mostly in how they answer them.

| Decision | Options | Why it matters |
|---|---|---|
| Where irregularity lives | Setup-time tables only (all five designs) versus runtime data structures (the C++ today: ragged inner_block leagues, full-height CSRs) | Tables make the runtime a fixed, small kernel set and let geometry, psi, alpha and BC values change without recompiling |
| How the structured part and the irregular part are split | By a hand-derived rule (distance from fragment ends) or by comparing rows against the exact assembled matrix | The exact-matrix rule is correct by construction and gives the scipy export for free; the hand rule is cheaper to build but has corner cases (short fragments, overlapping bodies) |
| How kernels come into existence | Hand-written templated C++/MSL (D3, D5), generated from a kernel IR (D1, D4), or lowered to an existing array framework (D2) | Determines who owns a compiler, the compile latency at define time, and how much cross-kernel fusion is possible |
| What Python is allowed to do at run time | Nothing per stage (D1, D3, D4, D5: one native call per K steps) or a traced program executed by a framework (D2) | At 2.6k to 5k points, dispatch and Python overhead decide parity; at 16.8M points, bytes per point per stage decide it |

The Apple GPU question is answered the same way by all five: no fp64 exists on any Apple GPU (`00-apple-gpu-facts.md`, verdict box), so any Metal path is fp32 or hand-written double-float (df64) and is an opt-in accelerator for large 3D on Pro/Max chips, not a replacement for the fp64 CPU path. The differences are in how much Metal code each design owns and whether df64 covers transcendentals.

## 2. The five angles

| Label | Angle | One-line thesis | Runtime language | Compiler owned? |
|---|---|---|---|---|
| D1 kerf-forge | Code-generating DSL (Devito/pystencils style, specialised to line-block cut-cell operators) | SymPy front end lowers at define time to a loop-level kernel IR and emits one fused Kokkos C++ (or MSL) kernel per RK stage; Python orchestrates only | Generated C++20 on Kokkos; generated MSL | Yes (IR, four emitters) |
| D2 ccsx | Array-framework lowering | Exact assembly in NumPy, structural split by row comparison, K-step `jax.jit` on CPU/CUDA fp64, MLX plus generated MSL on Apple | JAX/XLA, MLX; no C++ at run time | Small (tracer, 8-primitive kernel IR, printers) |
| D3 Pencil | Linear-algebra-first | Every operator is data in a block-sparse "line format" (BLX = banded lines plus exception rows); four hand-written kernels per backend execute it | Hand-written C++20 header templates, MSL, CUDA | No (closed IR, SymPy printer only) |
| D4 Kerf-lang | Kernel DSL (Taichi/Warp style) with a typed cut-mesh data model | Python-subset kernels parsed from the AST into a Point IR, fused by a scheduler into S+E sweeps, emitted through Jinja2 templates | Generated C++ via ctypes, MSL, CUDA | Yes (AST front end, Point IR, scheduler, three printers) |
| D5 thin waist | The skeptic: keep the C++ core | nanobind exposes mesh, stencils and a refactored assembly that exports compact tables; a ten-primitive backend executes a recorded schedule; Python replaces Lua | Existing C++ plus Kokkos; hand-written MSL later | No |

All five converge on the same data-layout decisions, which is itself a finding: one packed vector per field with static offsets [D | Rx | Ry | Rz] (replacing the four Views per scalar of `src/fields/field_registry.hpp`), int32 indices by default, per-direction constant taps stored once (not per line as in `coeffs_d`, `src/matrices/block.hpp:24-28, 65-93`), one byte of role or class bits per D point instead of the sorted fluid gather list, wall closures deduplicated, object closures as per-row data with the B column folded in, and Bf+Br merged into one row per R point so the full-height row-pointer arrays of B and N (`src/operators/derivative.cpp:441-442`, |D|+1 int64 each) disappear.

## 3. D1 kerf-forge: a code-generating DSL

### 3.1 Pitch

Users write the PDE in SymPy terms: fields, `kf.D`/`kf.Laplacian` operators, BCs on grid faces and bodies, a Butcher tableau and a manufactured solution. At define time kerf-forge (1) assembles every geometry- and psi-dependent coefficient once on the host in fp64, (2) builds a kernel IR for each RK stage in which cut-cell unstructuredness has been reduced to three things (a per-tile, per-axis class, one ELL gather-dot table per output space over a packed source buffer, and a bitmask per point), and (3) emits one fused kernel per RK stage. That kernel computes the stage RHS, folds B, N and Br into the closure ELL rows, and in its epilogue does the RK accumulate, the next-stage state and the Dirichlet fill. An RK4 step is four dispatches. The same IR emits MSL for Apple GPUs in f32 or df64 and encodes K steps per command buffer.

The argument for performance is structural, not measured: the generated code has the shape a careful engineer would write by hand (a 2.5D-blocked, z-vectorised constant-tap stencil in pure tiles, a lookup path only in the small fraction of tiles touching a body). Whether the generator actually reaches that shape is the thing the prototype must show.

### 3.2 Architecture

Layers, top to bottom:

- L0 Front end (pure Python). `Grid`, shapes `Sphere`/`YZRect` (semantics of `src/mesh/shapes.hpp`), `Domain`. Field types `Scalar`, `System(ncomp)`, `Param`. `Scheme` wraps a stencil_gen `CutCellResult`/`InterpResult` (`scripts/stencil_gen/stencil_gen/temo.py`, `interp.py`) or a compiled C++ stencil through an `any_stencil` binding (`src/stencils/stencil.hpp:95-120`) for hand-written schemes such as E4_2. Operators `D`, `Grad`, `Laplacian`, `Div` are lazy SymPy `Function` subclasses. BCs: grid {Dirichlet, Neumann, Floating}; objects {Dirichlet, Neumann, Floating, Slip}. `Equation`, Butcher-tableau integrators, `Program = kf.compile(...)`.
- L1 Geometry and assembly (host fp64). Produces `OperatorTables`: an R SoA per axis (psi, ray_outside, solid_ic, shape_id, position, normal) in the C++ (slow, fast, t) order (I14, asserted by a lexsort test); a fragment table per axis; a point bitmask (u8 per D point: fluid, grid-Dirichlet, interior-along-axis-a, written-by-any-operator), which replaces the 63 MB fluid gather list at 256^3 with 16.8 MB; a tile decomposition (tiles of 1 x 8 x 64 or 128) with a class per axis, PURE / WALL_L / WALL_R / MIXED, tile lists sorted by class; a closure ELL per (operator, axis) with the B column as an ordinary entry into the R segment and Neumann wall data as an entry into a face-data segment G; a MIXED-tile row map stored only for mixed tiles; an R-row ELL per R space with Bf and Br merged; R-space classes (EVOLVED, ALGEBRAIC_DIRICHLET, ALGEBRAIC_NEUMANN, SLIP); and MMS hoisting, in which SymPy separates each source and boundary function into a sum of products S_g(x) c_g(t), the spatial factors being evaluated once in fp64.
- L2 Kernel IR (plain dataclasses holding SymPy expressions). `Kernel{domain, body, stores, epilogue, variant}`; domains are `TileDomain`, `RowDomain` or `SpaceFusedDomain` (one dispatch whose global id range is partitioned between tiles and rows); the body is an SSA list produced by `sympy.cse`; loads are `Affine`, `Indirect`, `Const` or `TileTemp`.
- L3 Emitters. `emit_cpp` (Kokkos RangePolicy over tile indices or a Serial instance for small problems, inner `#pragma omp simd` loop along z), `emit_msl`, `emit_numpy` (a reference interpreter used as the test oracle), `emit_cuda` (the same C++ through Kokkos CUDA with `TeamPolicy(tile)`).
- L4 Runtime (`libkerf_rt`, C++20 plus nanobind). An arena with one allocation per slot (ncomp x packed_len), a table store, a kernel registry (generated code compiled into a nanobind extension module per program hash), a stats ring buffer, `run(nsteps)` executed on the C++ side, metal-cpp device with shared-storage buffers, matrix export in pure Python (scipy.sparse from the same tables).

How the six irregularities are represented: fragmented lines disappear at run time (fragment ends become per-point interior bits plus closure rows); irregular R lengths are segments of the packed buffer; psi closures are per-row ELL coefficients evaluated at setup; CSR couplings are all folded into ELL rows over the packed source; gather lists become bitmasks and class-sorted tile lists; partial-coverage writes become a total definition (every output point written exactly once per stage).

### 3.3 User-facing example

```python
import sympy as sp
import kerf as kf
from kerf.sym import x, y, t, nx, ny           # SymPy symbols; nx, ny = outward normal at R points

# ---- geometry: 2D, 51x51 on [0,2]^2 with one circular body ------------------
grid = kf.Grid(extents=(51, 51), bounds=((0, 2), (0, 2)))
body = kf.Sphere(center=(sp.Rational(16, 17), sp.Rational(25, 22)), radius=sp.sqrt(3) / 10)
dom  = kf.Domain(grid, shapes=[body])        # ray casting -> dom.R[0], dom.R[1] (SoA: psi, normal, position ...)
print(dom.summary())                          # |D|=2601, |Rx|=16, |Ry|=16, lines/dir, fluid count

# ---- scheme: E4 cut-cell family (2nd derivative E4_2: R=3,T=5,X=3, interp {5,5}) ----
E4 = kf.schemes.E4(derivative=2)              # resolves to E4_2 (C++ stencil via binding until regenerable)
E4g = kf.schemes.E4(derivative=1)             # used by the Neumann-object projection (n . grad u at R)

# ---- manufactured solution and forcing (pure SymPy, no Lua callbacks) --------
kappa = sp.Rational(1, 30)
u_ex = 2 * sp.exp(-((x - 1)**2 + (y - 1)**2)) * sp.cos(sp.Rational(1, 2) * t)
src  = sp.diff(u_ex, t) - kappa * (sp.diff(u_ex, x, 2) + sp.diff(u_ex, y, 2))
g_n  = nx * sp.diff(u_ex, x) + ny * sp.diff(u_ex, y)     # Neumann data on the body

# ---- field, operator, equation ----------------------------------------------
u = kf.Scalar("u", dom)                       # spaces: u.D (51,51), u.R[0] (16,), u.R[1] (16,)
lap = kf.Laplacian(E4)                        # lazy: sum_a D2_a, tables built at compile time

bcs = kf.BCs(
    grid={f: kf.Dirichlet(u_ex) for f in ("xmin", "xmax", "ymin", "ymax")},
    objects={body: kf.Neumann(g_n, gradient_scheme=E4g)},  # R points become algebraic (projected each stage)
)
eq = kf.Equation(kf.dt(u), kappa * lap(u) + src, bcs=bcs)

prog = kf.compile(eq, integrator=kf.RK4(), backend="cpu", dtype="f64")   # or backend="metal", dtype="df64"
print(prog.report())       # kernels, tile classes, ELL rows, flops/bytes per stage, cache key, compile time

# ---- initial condition, diagnostics, time loop -------------------------------
state = prog.init(u=u_ex.subs(t, 0))          # D fluid + R points; solid D zero (as heat.cpp IC)
dt = sp.Rational(1, 2) * grid.hmin**2 / (4 * kappa)
stats = kf.Stats(linf=kf.Linf(u - u_ex, where="fluid+evolved_R"), minmax=kf.MinMax(u))
out = kf.XdmfWriter("out/heat", every_time=0.01, fields={"u": u, "err": abs(u - u_ex)})

for chunk in prog.run(state, t0=0.0, t_end=0.3, dt=float(dt), stats=stats, steps_per_call="auto"):
    print(f"step {chunk.step:5d}  t={chunk.time:.4f}  linf={chunk.stats.linf:.3e}")
    if not chunk.valid:                       # NaN/blow-up check on device-buffered stats
        break
    out.maybe_write(chunk.time, state)        # zero-copy numpy views: state.u.D, state.u.R[0]

# ---- inspection / analysis using the same tables ------------------------------
A = prog.linear_operator(u)                   # scipy.sparse over packed [D|Rx|Ry] DOFs (I44/I50)
print("max Re(lambda) =", kf.analysis.rightmost_eig(A))
k1 = prog.rhs(state, t=0.0)                   # single RHS evaluation, for parity tests vs C++
```

Note that the example ships an object Neumann BC. The C++ never assembles object Neumann data (`nu.R` is never read; `src/operators/derivative.cpp:148` region), so this is new numerics bundled into the DSL pitch. The judges flagged it; see Section 8.

### 3.4 IR and lowering

The pipeline is SymPy front end, space equations, assembly, stage kernel IR, schedule passes, emitter, compile and cache, nanobind module.

1. Front-end tree. `eq.rhs` is a SymPy expression whose leaves are `FieldAccess`, `Coord`, `Time`, `Param`, `Normal` and whose operator nodes are `D(expr, axis, order, scheme)`.
2. Space equations. For each output component and each space in {D, Rx, Ry, Rz}, du[c,s] = F_s(...). Each `D` node becomes `OpRef(op_id, axis, inner_expr)`. BC lowering: grid-Dirichlet points get du = 0 and the epilogue writes u_next = g(t_{i+1}); grid-Neumann adds a G segment consumed by the X extra entries; floating objects evolve R rows; Dirichlet objects set u_R = g(t) in the epilogue; Neumann objects are a new capability whose R values are algebraic, u_R = P_D u_D + P_R u_R(other) + c g, solved by one ELL projection when P_R is empty, by a `scipy.sparse.linalg.splu` factorisation stored as explicit rows when the inverse stays sparse, and otherwise by k Jacobi sweeps in the epilogue with the residual reported; Slip objects project R momentum.
3. Assembly. Phase 1 binds `ccs::mesh` and `ccs::derivative` through nanobind and converts `block::metadata_view`/`coefficients_view` (`src/matrices/block.hpp:112-113`) plus the CSR `w/v/u` arrays (`src/matrices/csr.hpp:19-21`) into kerf tables (`tables_from_cpp`). Phase 2 is a NumPy reimplementation of ray casting, `init_line` and the nbs/interp evaluation, validated against phase 1 to 1e-14.
4. Stage kernel IR. One `Kernel` per RK stage with domain `SpaceFused(TileDomain(class-sorted tiles), RowDomain(R evolved rows), RowDomain(R algebraic rows))`. Body per D point: for each axis, PURE tiles use literal taps; WALL tiles select a literal closure row or the interior taps (the branch is uniform per (y,z) plane); MIXED tiles look up `slot_a[p]` (-1 interior, -2 no-write, else ELL row). Then r = F_D(d_0, d_1, d_2, ...), r = select(bit_dirichlet, 0, r), and the epilogue does acc += b_i dt r, u_next = u0 + a_{i+1} dt r, the Dirichlet select, and on the last stage the stats reductions.
5. Schedule passes, deterministic, no autotuner: stage fusion (the RK update becomes the epilogue, slot copies become pointer rotation, which is legal because the graph-pointer constraint of `src/simulation/simulation_cycle.cpp:123-125` no longer exists); space fusion; variant specialisation per tile class; temp placement for nonlinear inner expressions (GLOBAL, TILE or RECOMPUTE); CSE; precision legalisation (f64 passes through, f32 casts tables on upload, df64 rewrites + and * to error-free transforms and literals to (hi, lo) pairs); blocking (2.5D marching along x with a rolling window of planes, vectorisation along z, roles transposed in 2D).
6. Emit, compile, cache. `emit_cpp` produces one translation unit per program with `extern "C"` stage functions plus an `NB_MODULE`. Compiled with the project's flags against a precompiled Kokkos header. Cache key = sha256(IR text, backend, dtype, compiler id, flags). Geometry, grid size, psi, BC values and MMS parameters are runtime data, so a geometry sweep never recompiles. Alpha is compile-time by default, with `runtime_params=["alpha"]` for BO loops.

### 3.5 Backends, including the Metal path

| Backend | Status in the plan | Kernel shape | Notes |
|---|---|---|---|
| CPU fp64 | Primary | `Kokkos::parallel_for(RangePolicy)` over class-sorted tiles, or a Serial instance / direct call for |D| below about 40k (threshold to be measured in phase 0); plain C++ loop inside each tile with `#pragma omp simd` along z; R rows as a second range in the same region; per-thread reduction partials | No Kokkos Graph is used because a step is 4 calls. An optional Kokkos-free OpenMP prelude exists for pip-style deployment. Apple clang ships no OpenMP (verified on this machine: no libomp), so the OpenMP assumption needs a toolchain step |
| Metal | Phase 3, f32 or df64 only | A tile maps to a threadgroup (for example 8 x 32 threads), x-march in registers, MIXED tiles branch on `slot_a`; R rows are extra threadgroups of the same dispatch; one command buffer encodes K steps x 4 stage dispatches plus a reduction into a stats ring | metal-cpp runtime bound through nanobind; shared-storage buffers; df64 header (two_sum, fma two_prod, Newton division); transcendentals are avoided by construction because separable sources are hoisted; non-separable in-kernel exp/cos raises a compile error in df64 mode. The 120 us round trip the design assumed is an earlier unverified forum figure (`00-apple-gpu-facts.md`); the measured sync floor on the owner's M4 is 167 to 181 us (`05-measurements.md` Section D) |
| CUDA/HIP | Phase 5 | The same generated TU under Kokkos CUDA with `TeamPolicy(n_tiles, Ty)` and `ThreadVectorRange` over z; K-step batching through a Kokkos graph | Raw CUDA C++ through NVRTC is the fallback if Kokkos CUDA compile times are painful |

### 3.6 Performance argument

What the generated step does: an RK4 step is 4 stage kernels. Each stage reads u_stage (stencil footprint served from cache by 2.5D blocking), u0 and acc, and writes acc and u_next, about 5 doubles per D point per stage, plus one bitmask byte, plus tables only in MIXED tiles. Flops per point per stage for 3D E2 heat are about 30 (E4 about 42, E8 about 66). The C++ per RK4 step does about 280 launches, 26 fences, about 27 full-field slot touches, 6 blocking reductions and about 14 full-field host MMS evaluations (Part I hot loop).

| Regime | C++ bar (Part I, projected) | kerf-forge projection | Basis |
|---|---|---|---|
| A: heat.lua 51x51 (|D|=2601, |R|=16+16) | RHS about 30-40 us; step 0.3-1 ms (estimate) | Stage about 10-15 us; step 30-60 us (Serial path, no fork/join, no allocations) | 2601 x about 25 flops plus about 120 closure/R rows; L1/L2-resident. The win is removing about 270 of 280 dispatches and replacing about 37k virtual MMS calls per step with 2 FMA per point. Unmeasured |
| B: 64^3 E2, no shapes | RHS 1646 us (measured, noisy); plus about 180 us slot ops per stage | 0.1-0.2 ms per stage | 10.5 MB per stage, SLC-resident; the slow-axis team-per-line matvec (13.7 ns/pt at 64^3, `bench_block`) is replaced by the rolling plane window. Unmeasured |
| C: 256^3 plus sphere | RHS 30-110 ms (central 60, extrapolated); step 270-500 ms | About 35-40 ms per step (7 ms per stage at 96 GB/s plus 1-2 ms of gathered closure loads plus under 0.5 ms of R rows) | Assumes the 96 GB/s single-thread axpy rate of `00-apple-gpu-facts.md` is a DRAM ceiling, which is unverified (see blind spots). Metal fp32 on M4 Max about 3.4 ms per step, df64 about 7 ms, a 5-10x gain over CPU on Max and about 1x on base M4, all projected |

The small-size Python risk is structurally absent because Python is never on the per-stage path: setup is seconds, compile is 5-30 s cached, and there is one call per K steps.

### 3.7 Idiom coverage

| Idiom | Mechanism in kerf-forge | Risk |
|---|---|---|
| I01 | Runtime arena, one allocation per slot, ncomp x packed_len with static offsets; slots rotated by pointer swap inside `run()` | low |
| I02 | Phase 1: C++ `object_geometry` via nanobind (bit-exact); phase 2: vectorised NumPy ray casting reproducing snap/clamp/nextafter, checked against the C++ | medium |
| I03 | Phase 1: `mesh::lines` via binding; phase 2: NumPy merge-join; consumed only to derive interior bits and closure rows | low |
| I04 | Bitmask byte per D point plus an R class array per R space, built once | low |
| I05 | Interior taps and psi=1 wall closures become constexpr literals; RBF solves stay host-side NumPy; alpha compile-time or runtime param | low |
| I06 | Batched host evaluation over psi arrays by lambdify or a NumPy-retargeted `StencilCodePrinter`, or the vectorised C++ nbs wrapper | medium |
| I07 | Host assembly: `interp()` dispatch ported with searchsorted fragment lookup (phase 2) or the C++ path (phase 1) | high |
| I08 | Converter from inner_block lines to interior bits plus closure ELL rows; short fragments detected and merged or rejected | medium |
| I09 | Host assembly of R-row ELL (Bf+Br merged), reused for Neumann-object projection rows | medium |
| I10 | Not needed at run time; duplicates kept as separate ELL entries | low |
| I11 | Deduplicated closure table plus per-row object coefficients; a handful of uploads | low |
| I12 | Python front end plus `kf.from_lua()` translator | low |
| I13 | C order z-fastest kept; constexpr strides per program | low |
| I14 | R order kept; assembly asserts the lexsort; packed offsets static | medium |
| I15 | Disappears: work buffers become temps or arena slots; numpy zero-copy views | low |
| I16 | Tile-parallel stage kernel with PURE/WALL/MIXED variants; all axes summed in registers | medium |
| I17 | PURE-tile constexpr-tap loop, z-vectorised, x via rolling plane window | low |
| I18 | WALL tile variants with literal closure rows; Dirichlet wall row is a Dirichlet-bit point | low |
| I19 | MIXED-tile slot lookup into per-row object closure ELL | medium |
| I20 | No generic CSR kernel; every coupling is an ELL row inside the stage kernel | low |
| I21 | B folded into the closure ELL row of the same output | low |
| I22 | Neumann wall data in face segment G; zero cost when no Neumann | low |
| I23 | R-row ELL over packed source as a RowDomain in the same dispatch | low |
| I24 | Merged into I23 rows | low |
| I25 | Eliminated: total definition | low |
| I26 | Folded into the pointwise body | low |
| I27 | Fixed documented summation order; parity tolerance about 1e-14 relative | low |
| I28 | Grad yields 3 register values per point | low |
| I29 | Pointwise body with gG as Param fields | low |
| I30 | Source added in-kernel from separable spatial factors; masks via bitmask select | low |
| I31 | Dirichlet bit: du=0 in body, u_next=g in epilogue | low |
| I32 | Stage fusion: 4 dispatches per RK4 step, K steps per call | low |
| I33 | `prog.rhs()` calls an emitted standalone RHS kernel | low |
| I34 | SymPy MMS; separable terms hoisted; non-separable emitted in-kernel | low |
| I35 | Not supported by design; `kf.from_lua` translates gaussian types, raises on Lua callbacks | medium |
| I36 | Disappears | low |
| I37 | RK update is the stage epilogue | low |
| I38 | Pointer rotation in the arena | low |
| I39 | Boundary values only at boundary points in the epilogue | low |
| I40 | Hoisted to setup | low |
| I41 | Last-stage epilogue reduction into a stats ring, host reads every K steps | low |
| I42 | Error field only when a dump is due; same binary and XDMF layout | low |
| I43 | Python loop over `prog.run()` chunks | low |
| I44 | `prog.linear_operator()` from the same tables, cross-checked by probing `prog.rhs` with unit vectors | low |
| I45 | `emit_numpy` reference interpreter | low |
| I46 | Subsumed by SymPy bodies | low |
| I47 | `System(ncomp)`, symbolic fluxes, Div, Slip BCs, adaptive dt | high |
| I48 | Superseded by `linear_operator()` | low |
| I49 | `CutCellResult`/`InterpResult` reused; `StencilCodePrinter` (`scripts/stencil_gen/stencil_gen/printer.py:11`) subclassed for NumPy, MSL and C++ | low |
| I50 | Fed by `linear_operator()`; host fp64 scipy | low |
| I51 | In-process programs; `runtime_params=['alpha']` avoids recompiles | medium |

### 3.8 Awkward or unsupported

- Per-point host callbacks (Lua or Python) in the hot path (I35). Only SymPy-expressible sources and BCs; tabulated data as a Param field.
- Bitwise parity with the C++ is impossible: axes are summed per point in registers, B is folded into closure dots, uncovered points get zero. Parity is tolerance-based.
- Moving geometry: tables are setup-time; each change re-runs assembly and re-uploads without recompiling. Device-side `interp()` branching is not ported.
- Object Neumann BCs are new numerics; the projection formulation and the non-diagonal R-R case are unverified.
- Schemes without `interp_info` (E4_1, E4u_1, E6u_1, E8u_1) need interpolation rows borrowed from another scheme for floating or Neumann objects in 2D/3D; numerical properties unverified.
- Hand-written stencils (E4_2, splines, E4_1 guards) come through the C++ binding until regenerable, so the C++ stencils library stays a dependency in phases 1 and 2.
- Fully solid pencils, concave and overlapping bodies, planar objects with cross-direction interpolation: same restrictions as the C++.
- Metal: no fp64; non-separable transcendental sources refused in df64 mode; small 2D on Metal is never faster than the CPU.
- A C++ toolchain and Kokkos headers are required at define time; first compile 5-30 s. A wheel-only deployment needs AOT-prebuilt variants.
- Debugging generated code; the team owns a compiler.
- Implicit time stepping and global solves are out of scope.
- Nonlinear-in-derivative consumers force GLOBAL temps until a TILE strategy with closure-aware halos exists.

### 3.9 Reuse of existing code

Reused as is: the stencil_gen symbolic layer (`temo.py`, `interp.py`, `conservation.py`, `phs.py`), `codegen.apply_cse` (`scripts/stencil_gen/stencil_gen/codegen.py:15`), `StencilCodePrinter` subclassed three ways, the analysis cascade (`non_normality.py`, `gks_kreiss.py`, `group_velocity.py`) fed by `prog.linear_operator()`, the BO driver with `cpp_bridge.py` replaced by in-process calls. Reused through nanobind in phases 1 and 2, removable later: `ccs::cartesian`, `object_geometry`, `mesh`, the derivative/laplacian/gradient assembly exported through `block::metadata_view`/`coefficients_view`, `any_stencil`, and the C++ heat/scalar_wave RHS as the parity oracle. Reused as infrastructure: Kokkos 5.1.1, the CMake flags, Catch2 tests as pytest parity suites, the benchmarks, the Lua configs via `kf.from_lua`, the IO format. Retired once kerf-forge reaches parity: `src/matrices`, the `src/fields` expression templates and selection kernels, `src/temporal/slot_ops`, the Kokkos Graph glue, `simulation_cycle`.

### 3.10 Euler extension

State is `kf.System('U', dom, components=('rho','mx','my','E'))`, component-major SoA (ncomp x packed_len), which removes the `n_vectors == 0` and `scalar_handle{0}` limitations of `src/temporal/slot_ops.hpp:13` and `src/systems/system.cpp`. The user writes fluxes symbolically and the RHS is `-kf.Div(F, scheme=E4_1) + kf.Filter(U, ...)`. Lowering applies the temp-placement pass: v1 materialises fluxes globally (about 15 temps in 3D, roughly 2 GB extra traffic per stage at 256^3, about 25 ms per stage on CPU); v2 computes fluxes into tile buffers with halo P in PURE tiles while MIXED tiles read GLOBAL temps materialised only around them. R-point physics: slip walls project R momentum in the R epilogue; characteristic face BCs are face-segment kernels; Dirichlet object values come from the vortex exact solution (`src/systems/inviscid_vortex.cpp:13-116`) translated to SymPy. Adaptive dt is a last-stage reduction into a device scalar with no host sync on Metal or CUDA. Filtering is another axis-stencil operator through the same tile machinery. Precision: fp64 CPU/CUDA is the reference; df64 is required on Metal and its division costs 2-3x in flops, still near bandwidth-bound by the design's estimate (unmeasured).

### 3.11 Migration phases

| Phase | Duration | Content | Gate |
|---|---|---|---|
| 0 | 2-3 weeks | Native rebuild with per-region timers; a C++ test executable that dumps block meta/coeffs, CSR w/v/u, R records, lines and one RHS to .npz; minimal IR plus `emit_cpp` for one stage kernel (heat, E2_2) | The prototype experiment of 3.14 |
| 1 | 6-8 weeks | nanobind `shoccs_core`; `tables_from_cpp`; full stage-kernel generation; runtime arena; `kf.from_lua` for heat | heat.lua, brady_livescu_*.lua and 3D bench configs match the C++ RHS to 1e-13 relative; step at least as fast as the C++ in all three regimes |
| 2 | 6-8 weeks | Python-native assembly; scalar_wave; Neumann grid BCs; Neumann-object projection; `linear_operator()` wired into stencil_gen; cpp_bridge replaced; XDMF writer | Both assembly paths identical; all shipped Lua configs pass; t-laplacian's R-point discrepancy diagnosed against an independent assembly |
| 3 | 6-10 weeks | `emit_msl`, df64 header, metal-cpp runtime, K-step batching, on-device reductions, precision study | df64 Linf within 2x of f64 at all tested resolutions; f32 accuracy documented; large 3D faster than CPU on Pro/Max |
| 4 | 8-12 weeks | Euler: System fields, flux temps, Slip/characteristic BCs, adaptive dt, filter | Isentropic vortex convergence in 2D with and without a body |
| 5 | 3-4 weeks | CUDA/HIP via Kokkos device backends | Parity with CPU fp64; at least 60% bandwidth efficiency at 256^3 |
| 6 | ongoing | Retire `src/matrices`, `src/temporal`, graph glue, `simulation_cycle` | C++ dependency reduces to generated code plus `libkerf_rt` |

### 3.12 Effort

About 8-12 person-months for one experienced HPC/compiler-minded developer to reach phase 4. Code: front end, IR and passes about 5-7k lines of Python; emitters 2-3k; `libkerf_rt` 2-3k lines of C++ and metal-cpp; tests 3-4k. Ongoing maintenance of a small compiler, 10-20% of one FTE. The judges considered this optimistic (Section 8).

### 3.13 Risks

- The bar is unverified; a fixed-up C++ (hoisted object lists, compact B/N, boundary-only MMS) could narrow the claimed margins.
- Owning a code generator: complexity creep across Temp placement, tile variants, df64 legalisation and four emitters; bus factor.
- R-point correctness: t-laplacian already fails on the Bf/Br path (`docs/CLEANUP_PLAN.md` §0a), so there is no trusted reference for R values.
- fp32 and df64 accuracy of small-psi high-order closures on Metal is unknown.
- Compile latency and toolchain friction at define time.
- Object Neumann projection and composed-scheme interpolation are new numerics.
- Euler GLOBAL flux temps may cost 2-3x the ideal bandwidth until TILE placement works.
- Short fragments and multi-body geometries may produce overlapping closure rows.
- Metal small-size behaviour is unmeasured; K-step batching conflicts with adaptive dt and dump schedules.
- A NumPy geometry port can silently diverge in R ordering.

### 3.14 Killer prototype experiment

About 2-3 weeks, no nanobind. (a) Dump tables from a throwaway C++ executable for heat.lua (51x51, E2_2, floating circle) and bench_rhs 64^3 (E2, no shapes): block meta/coeffs, CSR w/v/u, R records, masks, one state u and the C++ graph-path RHS du. (b) A 300-line Python converter into kerf tables. (c) A 500-line generator emitting one C++ stage function (RHS plus RK4 epilogue, PURE/WALL/MIXED variants, R rows) plus a ctypes shim, compiled with clang -O3. (d) Time RHS-only and the full step against the natively rebuilt C++. (e) Optionally emit the same stage as MSL f32.

Pass (all must hold): RHS parity to 1e-13 relative on fluid D and evolved R points for both configs; heat.lua kerf RHS at or below the C++ graph RHS and kerf step at or below 0.5x the C++ step (median of 1000 steps, native); 64^3 stage at or below 0.5x the C++ RHS; uncached generation plus compile under 30 s, cached under 1 s. Kill: parity fails by more than 1e-12 for structural reasons (closure/B folding cannot reproduce the operator, e.g. overlapping rows); the 64^3 stage is not faster than the C++ RHS after reasonable tuning (the tile and 2.5D thesis is wrong); the 2D step cannot beat the C++ step (fixed per-call overhead dominates).

## 4. D2 ccsx: array-framework lowering

### 4.1 Pitch

The user writes the PDE as Python on `Field` objects (`lap(u)`, `grad(u)`, pointwise physics) with a SymPy manufactured solution. Setup runs in fp64 NumPy/SciPy and reproduces the C++ mesh and operator assembly exactly, over a two-buffer layout (D as an (nx,ny,nz) array plus one global R vector). It then splits every operator into a STRUCTURED part (per-axis constant-tap stencils whose wall closures are fixed small dense blocks on the first and last planes, covering about 97-99.5% of rows) and an IRREGULAR part (one compact ELL row table plus an int32 row map holding every row the structured part gets wrong). Rows are classified by row-wise comparison against the exact assembled matrix, so the split is correct by construction and the exact matrix doubles as the stability export. At run time only u and t change: a K-step chunk is traced into one `jax.jit` executable with `lax.fori_loop`, buffer donation and an `optimization_barrier` per stage. Every cut-cell term is a gather, never a scatter. On Apple GPUs the same kernel IR lowers to MLX, with the fused stencil-combine-epilogue emitted as one generated `mx.fast.metal_kernel` in fp32 or df64. No C++ runs at run time.

### 4.2 Architecture

- L0 Front end: `Grid`, shapes, `Domain`, `scheme(name, alpha)`, `BoundaryConditions`, `Field(ncomp, dtype)`, `LinearOperator` objects, `@system` functions, integrators from Butcher tables, `Simulation`, io. MMS and sources are SymPy expressions lambdified to the backend's array module.
- L1 Geometry and operator compiler (NumPy/SciPy fp64, once per geometry/BCs/scheme/alpha): G1 `raycast.py` (vectorised intersection, `np.nextafter` stepping, the 1e-12 snap and psi clamp, bit-faithful to `src/mesh/object_geometry.cpp:78-112`); G2 `fragments.py` (a vectorised `init_line`, a per-pencil fragment index replacing the O(|R|) `interp_line` scan of `src/mesh/mesh.cpp:170`); G3 `coeffs.py` (batched `nbs(psi[n])` from lambdified stencil_gen output or transliterated C++ bodies); G4 `assemble.py` (a Python port of `domain_discretization` and `cut_discretization` emitting an exact `scipy.sparse` COO matrix over the packed index space); G5 `split.py` (the key step: rasterise the structured operator, classify each D row as regular iff it equals the exact row to 4 ulp, merge irregular rows across directions, pad to Kd and Kr); G6 `sets.py` (masks and static face slices). Output is an `OperatorPlan` whose tables are padded to shape buckets so psi/alpha sweeps recompile nothing: they are runtime arguments of the executable.
- L2 Semantic IR: a typed DAG traced from the `@system` function with Field proxies (`Input`, `Time`, `Pointwise`, `Apply`, `SourceAt`, `Restrict`, `BCApply`, `Reduce`), wrapped by an integrator template into `Step` and by `Chunk(K)`.
- L3 Kernel IR, 8 primitives: K0 Layout, K1 Map, K2 AxisStencil (shifted slices plus einsum closures on the first and last planes), K3 RowGather (the ELL dot over D and R columns), K4 Combine (rowmap select: regular, zero, or table value plus the structured directions flagged in `dirbits`), K5 SliceSet/MaskSet, K6 Reduce, K7 Loop with Barrier markers. Fusion rule: one stage = tiny K3, then one D loop fusion (K2 for all axes, K4, the system epilogue, the RK epilogue), then the R epilogue, then BC slices, then a Barrier. The RHS k is never materialised.
- L4 Backends: jax-cpu (f64), jax-cuda (f64), mlx-metal (f32 or df64 via generated MSL), numpy-ref (the oracle), optional torch.

Sizes at 256^3 with a sphere: rowmap 67 MB int32 (shared between gradient components), irregular D rows about 230k x Kd about 10 (about 30 MB), R rows about 76k x Kd about 40 (about 37 MB). There are no full-height row pointers.

### 4.3 User-facing example

```python
import sympy as sp
import ccsx as cx

# ---- manufactured solution: symbolic; ccsx derives source and BC data ----
x, y, t = sp.symbols("x y t", real=True)
kappa = 0.1
S = sp.cos(3 * t) * sp.exp(-((x - 0.25) ** 2 + (y + 0.1) ** 2) / 0.2 ** 2)
f = sp.diff(S, t) - kappa * (sp.diff(S, x, 2) + sp.diff(S, y, 2))

# ---- geometry: 2D box with a circular obstacle (a "sphere" in 2D) ----
grid = cx.Grid(lo=(-1.0, -1.0), hi=(1.0, 1.0), n=(101, 101))
body = cx.Sphere(center=(0.05, 0.02), radius=0.31)
dom = cx.Domain(grid, shapes=[body])      # ray casting -> R points, fragments, masks
print(dom.summary())                      # |D|=10201, |Rx|, |Ry|, lines/dir, fluid count

# ---- discretisation: E4 second-derivative family (has Neumann closures) ----
E4 = cx.scheme("E4_2", alpha=cx.known_values("E4_2"))

bcs = cx.BoundaryConditions(
    grid=cx.Dirichlet(S),                                 # u = S on all four walls
    objects={body: cx.Neumann(dudx=[sp.diff(S, x),        # d_dir u at each R(dir)
                                    sp.diff(S, y)])},     # point, from the MMS
)
lap = cx.Laplacian(dom, E4, bcs)  # setup: exact assembly -> structured + row table
# lap.as_scipy() -> exact packed [D|R] matrix (eigen/stability export)

# ---- the PDE: u_t = kappa * lap(u) + f ----
@cx.system(state={"u": 1}, bcs=bcs)
def heat(s, t):
    u = s.u                    # Field proxy: u.D is (nx, ny); u.R[0], u.R[1] are cut points
    rhs = kappa * lap(u)       # traced as Apply(lap, u) -> K2/K3/K4
    rhs += cx.source(f, t, on=dom.fluid | dom.R.nondirichlet)   # fused, at time t
    return {"u": rhs}          # Dirichlet rows zeroed and BC values applied by bcs

# ---- initial condition, integrator, driver ----
u0 = cx.Field.from_expr(dom, S.subs(t, 0))     # solid D points -> 0, R at intersections
dt = 0.2 * grid.hmin ** 2 / (4 * kappa)
sim = cx.Simulation(
    heat, cx.RK4(), state={"u": u0},
    backend="jax-cpu", dtype="f64",           # or "mlx-metal"/"f32", "jax-cuda"/"f64"
    steps_per_call=32,                        # K steps per compiled call (one readback)
    stats=[cx.stats.minmax("u", on=dom.fluid),
           cx.stats.linf_error("u", exact=S, on=dom.fluid | dom.R.nondirichlet)],
)

for chunk in sim.run(t_end=0.5, dt=dt, dump_every=0.05):   # chunks end on dump times
    print(chunk.step, chunk.t, chunk.stats["linf_error"][-1])
    if not chunk.valid:                       # first invalid step index is in the chunk
        raise RuntimeError(f"blew up at step {chunk.first_invalid}")
    if chunk.dump_due:
        cx.io.write_xdmf("out/heat", chunk.state, exact=S)   # same D|Rx|Ry|Rz layout
```

The object Neumann BC here is the "directional data" form only (d_dir u at each R(dir) point, which is what the nbs Neumann closures consume). A physical n.grad u = g condition is explicitly not provided.

### 4.4 IR and lowering

1. Trace. Field proxies record SIR nodes; Python control flow on static values is allowed, data-dependent control flow is rejected (the `interp()` branches live in setup).
2. Operator resolution. Each `Apply` gets an `OperatorPlan`; a Gradient yields 3 plans sharing one rowmap; a Laplacian sums its directions into one plan. Each plan validates itself: structured part plus table part equals A @ u to 1e-13 relative on random u.
3. KIR lowering of `Apply`: S_d = AxisStencil per active axis; vals = RowGather; y.D = Combine; y.R = vals[rows_R]. BC application is K5 slices with values evaluated only on face coordinates, plus `where(dirR, gR, uR)`.
4. Schedule. One D fusion group per stage; K3 before, R epilogue after, BCs after both, a Barrier (`jax.lax.optimization_barrier`) closes the stage. Without the barrier XLA:CPU was observed (locally, unverified, `frameworks/jax.md:221`) to inline stencil producers into consumers and recompute them for a 3-5x slowdown. Nonlinear inputs to stencils (Euler fluxes) follow a materialise-vs-recompute rule.
5. Emit, jax: AxisStencil as `jnp.pad` plus `lax.slice_in_dim` sums with closures via `jnp.einsum` and `lax.dynamic_update_slice_in_dim`; RowGather as `jnp.take` with a zero ghost appended to the gather source (the only concatenate); Combine as nested `jnp.where` with the small-array gather fused into the elementwise loop. No scatter anywhere, which avoids the D-sized copy that `.at[].set` forces (`frameworks/jax.md:221`, local observation). Chunk is `jax.jit(chunk, donate_argnums=(state,))` with `lax.fori_loop`; tables are jit arguments so alpha changes cost zero recompiles; `jax_enable_x64` is set by ccsx.
6. Emit, mlx-metal: the D fusion group becomes one generated `mx.fast.metal_kernel` (about 600 lines of printer), one thread per D point, outputs written as new arrays because MLX custom-kernel inputs are read-only (`frameworks/mlx.md`); K steps evaluated with one `mx.eval`; df64 variant with a `struct df {float hi, lo;}` and Dekker/Knuth arithmetic; df64 transcendentals are a later item.
7. Emit, numpy-ref: the same KIR eagerly, plus a `scipy` mode y = A @ u for oracle checks.

Semantic decisions made explicit: total output (solid and Dirichlet-line points are 0), Dirichlet R and face rows of the RHS are 0 with values imposed after each stage, duplicate CSR entries summed at assembly, solid D points not integrated.

### 4.5 Backends, including the Metal path

| Backend | Status | Mechanism | Caveats (from Part II) |
|---|---|---|---|
| jax-cpu fp64 | Primary and parity target | One jit executable per K-step chunk; about 4-6 fused thunks per RK stage, each a parallel loop fusion tiled along the slowest axis across XLA's thread pool | Performance depends on XLA:CPU fusion heuristics; jax is pinned (0.11.x, `frameworks/jax.md:3`) and an HLO-inspection test asserts the number of fusions per stage. Escape hatch: a Numba `@njit(parallel=True)` chunk from the same KIR |
| mlx-metal | Opt-in, f32 or df64 | State as `mx.array` in unified memory; the fused D stage as a generated custom kernel; K steps built lazily and committed with one `mx.eval`; `mx.compile` keeps Python graph construction under about 20 us per step | No fp64 (MLX #1905 closed as not planned, `00-apple-gpu-facts.md`); custom kernels have read-only inputs, an evolving API and no replayable graph; MLX is pre-1.0. torch-MPS via `torch.mps.compile_shader` with the same MSL printer is the documented fallback. The survey's own jax-mps measurements are 3-5x slower than CPU at 2D sizes (`frameworks/jax.md:225`, local, outside the sandbox) |
| jax-cuda fp64 | Phase 6 | The same lowering; `lax.fori_loop` runs as a device while loop | No CUDA hardware locally; a `jax.experimental.pallas` kernel from the same tile printer if XLA's fusion is below 50% of bandwidth |

### 4.6 Performance argument

Counting per RK4 stage: the D fusion group reads u_s, u0, rk, the int32 rowmap and the gathered compact vals, and writes u_next and rk_next, about 44 B per point for fp64; per step 176 B per point against about 27 full-field touches in today's slot ops plus about 14 doubles per point per RHS.

| Regime | C++ bar | ccsx projection | Basis |
|---|---|---|---|
| A: 51x51 | RHS 30-40 us; step 0.3-1 ms (estimate) | About 8-15 us per stage, 35-60 us per step, plus about 15 us per chunk call amortised over K=32 | Consistent with the local unverified measurement of a JAX 51x51 toy: RHS 14 us, step 66-74 us (`frameworks/jax.md:201-203`). About 2x under the per-RHS bar; the step comparison is against an estimate |
| B: 64^3 | RHS 1.65 ms (measured); step about 7 ms with slot ops | About 0.3 ms per stage allowing 3x for imperfect fusion, 1.2 ms per step | Local measurement: RHS with ELL tables 424 us, RK4 step with barriers 1.7 ms (`frameworks/jax.md:216-217`, unverified, synthetic tables) |
| C: 256^3 plus sphere | RHS 30-110 ms; step 240-440 ms plus slot ops | About 50 ms per step with a 1.5x XLA inefficiency factor; Apple GPU fp32 about 9 ms per step on M4 Pro, df64 about 1.5-2.5x over CPU fp64 on Pro/Max | Local measurement: RK4 step 86 ms at 256^3 (`frameworks/jax.md:217`). Memory under 0.8 GB |

Dispatch: one Python-to-XLA call per K steps (10-20 us), one readback per chunk. Compile cost 1-5 s per (grid shape, table bucket, K, dtype), with the persistent compilation cache for BO sweeps.

### 4.7 Idiom coverage

| Idiom | Mechanism in ccsx | Risk |
|---|---|---|
| I01 | State pytree {comp: (D array, R vector with zero ghost)}; donation replaces slots | low |
| I02 | Vectorised NumPy ray casting; snap/clamp verbatim; `np.lexsort`; parity against the C++ geometry.csv | medium |
| I03 | Fragments from consecutive sorted R hits per pencil; vectorised `dirichlet_line` | low |
| I04 | Bool masks and static face slices built once, passed as executable arguments | low |
| I05 | Taps and uniform closures from stencil_gen or transliterated; RBF solve in NumPy; alpha length validated | low |
| I06 | Batched `nbs(psi[n])` via `sympy.lambdify(cse=True)` or transliterated bodies | medium |
| I07 | Host port of `stencil::interp` and `interp_deriv_coefficients` (`derivative.cpp:22`), grouped by branch, searchsorted fragment lookup | high |
| I08 | Port of `domain_discretization` emitting exact COO; short fragments raise | medium |
| I09 | Port of `cut_discretization` producing R rows in the same COO | medium |
| I10 | scipy COO with duplicates summed; no row pointers at run time | low |
| I11 | Replaced by the split: AxisSpec plus compact ELL | low |
| I12 | Python front end; `ccsx.from_lua` via lupa | low |
| I13 | D as (nx,ny,nz) C-order arrays; shifted slices; int32 rowmap caps D at 2^31 | low |
| I14 | R order by lexsort; offsets in one place; permutation test against the C++ dump | medium |
| I15 | Disappears: traced temporaries | low |
| I16 | K2 per axis plus K3/K4 for rows classified against the exact matrix | medium |
| I17 | K2 interior shifted-slice sum fused into the stage loop | low |
| I18 | K2 wall closures as einsum on the first/last planes; Dirichlet wall rows ZERO in the rowmap | low |
| I19 | Irregular rows in the ELL table | low |
| I20 | No generic CSR kernel | low |
| I21 | B entries are the R columns of the irregular D rows | low |
| I22 | K2 X-extras times a face vector evaluated on face coordinates only | low |
| I23 | R rows of the ELL table, merged over directions | low |
| I24 | Same rows, R columns | low |
| I25 | Disappears: Combine is total | low |
| I26 | Pointwise epilogue in the fusion group | low |
| I27 | Functional sum inside Combine | low |
| I28 | Gradient = 3 plans sharing one rowmap; stacked value consumed in the same fusion | low |
| I29 | Pointwise gG dot grad fused with Combine | low |
| I30 | Masked add with the fluid and nondirR masks | low |
| I31 | ZERO rows in the rowmap or face slices; `dynamic_update_slice` of boundary-only values | low |
| I32 | `jax.jit` of a K-step chunk with `fori_loop` and donation; MLX lazy eval; CUDA device loop | medium |
| I33 | numpy-ref backend | low |
| I34 | SymPy MMS lambdified to jnp/mx over broadcast coordinates | low |
| I35 | Dropped by design | medium |
| I36 | Source is a traced function of t | low |
| I37 | Butcher-table template; stage update as the fusion epilogue | low |
| I38 | Free with donation | low |
| I39 | BCApply inside the compiled step on face slices and R masks | low |
| I40 | Setup masks | low |
| I41 | K6 masked reductions into a device stats buffer f[K, nstat]; one readback per chunk; first-invalid index and rollback | medium |
| I42 | Error field in dump-due chunks; XDMF writer in Python | low |
| I43 | Host loop over chunks ending on dump times; adaptive dt carried in loop state | low |
| I44 | `op.as_scipy()` is the exact packed matrix in any dimension | low |
| I45 | numpy-ref and scipy mode | low |
| I46 | Pointwise SIR nodes and XLA fusion | low |
| I47 | Multi-component state, flux functions, batched Derivative plans, slip projection with `normal[nR,3]` | medium |
| I48 | Replaced; cross-check tests added | low |
| I49 | stencil_gen symbolic matrices lambdified; printer gains a NumPy target; hand-written schemes transliterated (`tools/eval_poly.py` style) | medium |
| I50 | Cascade consumes `op.as_scipy()` | low |
| I51 | In-process runs; alpha changes only coefficient arrays, zero recompiles | low |

### 4.8 Awkward or unsupported

- fp64 on Apple GPU is impossible; the Metal path is fp32 (pending the accuracy study) or hand-written df64; df64 transcendentals need custom polynomials or a precomputed separable time factor.
- Small 2D on Metal: no bandwidth edge on 85-160 KB states; ccsx warns and routes to jax-cpu.
- Moving bodies mean rerunning the NumPy compiler per step.
- Data-dependent runtime control flow (adaptive limiters) does not fit static-shape tracing; it must be `where`-selects between precomputed alternatives.
- Object Neumann only in the directional-data form.
- Schemes with empty `interp_info` raise at setup for floating or Neumann objects in 2D/3D.
- Parity is tables equal to 4 ulp and end-to-end Linf about 1e-12 relative; not bitwise.
- XLA:CPU performance can regress across releases; pinning and a perf regression suite are required.
- Lua callbacks dropped.
- Fully solid pencils, planar objects and concave bodies inherit the C++ restrictions as explicit errors.
- Non-Hermitian sparse eigenproblems stay host SciPy.

### 4.9 Reuse of existing code

Python reused directly: `temo.py` `CutCellResult`/`InterpResult` and `interp.py` as the symbolic source, with a new `stencil_gen.numeric` module wrapping `sympy.lambdify(..., cse=True, modules='numpy')` (new code: stencil_gen uses a `C99CodePrinter` subclass today, `printer.py:8-11`, and no lambdify); `codegen.py`/`printer.py` gain NumPy and MSL targets; `tools/eval_poly.py` and `eval_e2_1.py` generalised into a transliteration tool for `src/stencils/E4_2.cpp`, the E4_1 guards and the splines; `known_values.json`; `conservation.py`; the analysis cascade and `bo.py` consume `op.as_scipy()` and in-process runs. C++ reused as specification and offline oracle only: `object_geometry.cpp`, `mesh.cpp`, `derivative.cpp` ported line-for-line in semantics; a new dev-only `tools/dump_operator.cpp` writes R records, the line table, inner_block meta and coeffs and every CSR as COO .npz, and is the parity oracle in CI; `shapes.hpp`, `boundaries.cpp`, the XDMF layout and the Lua schema reproduced for compatibility.

### 4.10 Euler extension

`Field(ncomp=4|5)` laid out (ncomp, nx, ny, nz) and (ncomp, nR+1). Dx and Dy are first-derivative plans applied with a leading component batch dimension: K2 taps shared across components, K3 gathers idx/C once and applies them to all components. A divergence of directional fluxes merges into one plan. Flux evaluation follows the materialise-vs-recompute rule (recompute inside the stencil loop for w at most 5 on CPU fp64). Slip wall is a `BCApply(obj, Slip)` on R values; characteristic faces are pointwise eigen-decompositions on face slices; adaptive dt is a K6 reduction carried in the `fori_loop` state; the filter is another AxisSpec through the same path. Diagnostics: per-component Linf against the isentropic vortex, conservation integrals, and `jax.jacfwd` of the RHS giving the linearised operator for I50 at no extra cost. Precision: fp64 on CPU/CUDA; df64 recommended on Metal.

### 4.11 Migration phases

| Phase | Duration | Content | Gate |
|---|---|---|---|
| 0 | 2 weeks | Native build with timers; `tools/dump_operator.cpp`; fp32/df64 accuracy study with a NumPy eager reference on dumped operators | Decides whether Metal is df64-only |
| 1 | 5-6 weeks | raycast, fragments, coeffs, assemble, split, sets; parity against the dump for every shipped config and scheme | `op.as_scipy()` matches C++ tables to 4 ulp; split identity holds on random vectors |
| 2 | 4 weeks | SIR tracing, KIR, scheduler, JAX emitter, integrators, boundary-only MMS, stats, chunked driver, numpy-ref | heat.lua end-to-end parity; the two-gate spike of 4.14 |
| 3 | 5 weeks | Gradient plans, scalar_wave, eigenvalue system, 3D, wall and object Neumann, XDMF, from_lua, BO switched to in-process, perf regression suite in CI | |
| 4 | 6 weeks | Euler | Isentropic vortex convergence in fp64, 2D with a cylinder, then 3D |
| 5 | 6-8 weeks | MLX backend, MSL printer (f32), df64 arithmetic, df64 transcendentals | Ship only if the phase-0 fp32 study passes or df64 parity is shown |
| 6 | 2 weeks | jax-cuda validation on a rented GPU | |
| 7 | ongoing | ccsx becomes the default driver; C++ frozen as oracle for one cycle | |

### 4.12 Effort

Phases 0-2 about 11-12 weeks; phase 3 about 5 weeks; Euler 6 weeks; Apple GPU 6-8 weeks; CUDA 2 weeks. About 7-8 months in total, or about 4.5 months for CPU-only parity plus Euler. Code: setup compiler about 3.5k lines (assemble and interp about 1.5k, the riskiest), SIR/KIR/scheduler 1.5k, JAX emitter 0.8k, MSL printer plus df64 1.5k, driver/io/front end 1.5k, tests 4k, plus 300 lines of C++ for `dump_operator`.

### 4.13 Risks

- The C++ bar is unmeasured; if native C++ is 60-100 us per step for heat.lua the regime-A margin shrinks to about 1x.
- XLA:CPU fusion behaviour (recomputation without barriers, multi-axis shifted-slice fusion, thread partitioning at 64^3) could cost 2-4x.
- Assembly-port correctness on R rows, where the C++ oracle itself may be wrong (t-laplacian).
- fp32 accuracy of small-psi closures may force df64-only Metal.
- MLX custom-kernel limits (read-only inputs, evolving API, no replayable graph).
- The int32 rowmap read (about 9% of per-stage traffic) becomes an extra pass if a backend cannot fuse the gather.
- Compile latency and shape buckets.
- Bus factor and ecosystem churn: MLX is pre-1.0 and Apple-specific.

### 4.14 Killer prototype experiment

The two-gate spike, 3-4 weeks. Implement the split plus a JAX chunk for one operator family only (E2_2 Laplacian, Dirichlet walls, a floating circle), fed from `dump_operator.npz` instead of the Python geometry port. Cases: heat.lua 51x51; bench_rhs heat 64^3 without shapes; a new 128^3 heat config with a sphere of r=0.25L. Gate 1 (correctness): for 100 random u, ||(S+T)u - Au|| / ||Au|| at most 1e-13 on D and R; end-to-end heat.lua Linf and min/max stats match `logs/system.csv` to 1e-10 relative, with any R-point disagreement traced to a documented C++ issue. Gate 2 (performance, native C++ with all cores versus jax-cpu fp64, median of 20 runs, K=32): time per RK4 step at most 1.0x C++ for all three cases, and at least 40 GB/s effective per stage at 128^3. Kill: after one week of tuning JAX is slower than C++ by more than 1.5x in any case and the Numba variant of the same KIR is also above 1.0x, in which case the array-lowering angle is rejected in favour of a code-generating DSL; if gate 1 fails on D rows, the exact-assembly-plus-split premise is wrong. Side experiment (2 days): case 1 and an E4_2 sphere case in the NumPy reference with fp32 operator and state; pass if fp32 Linf is at most 2x the fp64 discretisation error and the rightmost eigenvalue keeps its sign; otherwise Metal is scoped to df64 only.

## 5. D3 Pencil: linear-algebra-first with the BLX line format

### 5.1 Pitch

Algebraically, a SHOCCS cut-cell operator is A = sum_d (M_d T_d + S_d) + E: a constant-tap Toeplitz band T_d per axis, a 0/1 row mask M_d saying which rows the band owns (derived from the fragment table), a handful of plane-constant wall-closure slabs S_d, and a small sparse exception matrix E over the packed vector V = [D | Rx | Ry | Rz] holding everything irregular (psi-dependent object closures, the B column into R, the Bf/Br rows, Neumann extras). Pencil assembles A once in NumPy from the SymPy stencil pipeline (vectorised `lambdify` over the psi array), exports it as scipy CSR for eigen and stability work, and executes it with four hand-written structural kernels per backend: STRUCT (fused 3-axis masked band plus slabs with prologue and epilogue maps), EXC (CSR rows over V), MAP and REDUCE. Because the operator is data, a backend is about 600 lines of kernels plus a SymPy printer. One RK4 stage of the heat equation lowers to two kernels, STRUCT then EXC, with the stage update and Dirichlet fill fused into their epilogues: about 12 launches per step instead of about 280. Nonlinear Euler fits as a pointwise flux map followed by the same linear operators.

### 5.2 Architecture

Layers (Python unless stated):

1. `pencil.geom`: `Mesh` (same `ic(i,j,k)` as `src/mesh/mesh.hpp`), `Shape`, `Domain`. `Domain.build()` produces per direction the R-record SoA in the C++ (slow, fast, t) order, the fragment table `Frags`, and bitsets (fluid, Dirichlet faces, src = fluid and not Dirichlet, per-direction `band_bits` for rows the band owns, `wall_pencil_bits`). Bitsets are uint64 words along k (0.125 B per point each).
2. `pencil.schemes`: `Scheme` objects backed by stencil_gen `CutCellResult`/`InterpResult`. `closure(bc, psi[n], right)` evaluates one `sympy.lambdify` of the CSE'd block over the whole psi array, with the psi clamp and alpha guards of the shipped `src/stencils` files re-applied. For schemes without a regenerable SymPy spec (E4_2, the spline/RBF E4u variants, the hand-guarded E4_1) `pencil.schemes.cxx` wraps the C++ stencil structs through a 150-line nanobind module.
3. `pencil.assemble`: replicates `derivative.cpp:305-449` and `159-257` but emits taps, `Slabs` (psi=1 wall closures, one coefficient row per (side, r), masked by `wall_pencil_bits`; Dirichlet walls drop row 0 but keep the wall column exactly like the C++), and COO triples over V for everything else. `laplacian` sums three derivatives, `gradient` is three LinOps, `filter` reuses the machinery. Every LinOp has `.to_scipy()`, which is the eigenvalue/GKS/pseudospectra path (replacing `unit_stride_visitor`/`coefficient_visitor` and the dense reimplementations of I48).
4. `pencil.ir`: index spaces, vectors, LinOp algebra, Map expressions (SymPy trees), Reduce, Program.
5. `pencil.lower`: the passes of 5.4.
6. `pencil.rt` (C++20 plus nanobind): `Buffer` (page-aligned, zero-copy with NumPy; shared-storage MTLBuffer on Apple), `Plan` (array of kernel records), `run(plan, nsteps)`. Header templates `struct_kernel<W, DT, Prologue, Epilogue>`, `exc_kernel`, `map_kernel`, `reduce_kernel`. Precompiled instantiations for W in {3,5,7,9,11} with table-driven affine epilogues cover the no-compiler fallback; a per-system generated TU is compiled at setup and cached by hash.
7. `pencil.sim`: `System`, integrators, stats, io, a Lua importer via `lupa`.

The line format BLX, one per (LinOp, direction): `taps f64[w]` with scale folded in; `band_bits`; `slabs` (at most 2R entries per direction, 16-88 doubles in total, against the per-line copies of 11-163 doubles per line in `coeffs_d`); optional `rowscale f64[|D|]` (a left Diag for variable-coefficient products such as scalar_wave's gG dot grad, I29); `frags` as metadata. `Exc`, one per LinOp over V: `rows i32[nE]` (unique, sorted), `indptr`, `cols`, `vals`. nE is about 3(R-1)|R| + 3|R|, about 0.6% of |V| at 256^3 with a sphere; wall closures are not in Exc, which is what keeps it small. A `Field` is one packed `f64[|V|]`; a `State` (Euler) is `f64[ncomp, |V|]`. Plan buffers for RK4 as designed: u0, u1, rk (the stage RHS is never materialised). The judging found a read-after-write hazard in this three-buffer plan; see 5.13 and Section 8.

Mapping of the six irregularities: fragmented lines become `band_bits` plus `Frags` metadata; irregular R lengths become V offsets; psi closures become rows of E evaluated for all psi at once; the CSR couplings become a single E (B merged into the closure row, Br merged into the Bf row, N as a slab broadcast or E rows from a data vector); gather lists become bitsets evaluated in kernel epilogues; partial-coverage writes become a total output (STRUCT writes every V row, 0 where no band or slab row covers it; EXC accumulates its unique rows).

### 5.3 User-facing example

```python
import numpy as np
import sympy as sp
import pencil as pc
from pencil.schemes import E4          # SymPy-backed (stencil_gen) scheme family

# ---- geometry: 2D grid, one embedded sphere --------------------------------
mesh = pc.Mesh(extents=(51, 51), bounds=((0.0, 0.0), (2.0, 2.0)))   # nz = 1, z fastest, same ic() as C++
ball = pc.Sphere(center=(16/17, 25/22), radius=np.sqrt(3)/10)
dom  = pc.Domain(mesh, shapes=[ball])         # ray casting, R records, fragments, bitsets (setup, NumPy)
print(dom.R[0].psi[:4], dom.R[0].position[:4], len(dom.R[1]))   # R points are first-class, per axis

# ---- manufactured solution and boundary conditions (SymPy, no Lua) ---------
x, y, t = pc.coords("x y t")
S  = 2*sp.exp(-((x-1)**2 + (y-1)**2)) * sp.cos(0.5*t)             # exact solution
k  = sp.Rational(1, 30)
bc = pc.BCs(
    grid={"xmin": pc.Dirichlet(S), "xmax": pc.Dirichlet(S),
          "ymin": pc.Dirichlet(S), "ymax": pc.Dirichlet(S)},
    objects={ball: pc.Neumann(pc.normal_derivative(S))},         # object Neumann: g_R vector from n . grad S at R points
)

# ---- operators: assembled once into explicit matrices over V = D|Rx|Ry|Rz ----
scheme = E4(order=2)                               # second-derivative E4 cut-cell closures (R=3, T=5, X=3)
lap    = pc.laplacian(dom, scheme, bc)             # LinOp: sum_d (M_d T_d + S_d) + E   (BLX + Exc)
A      = (k*lap).to_scipy()                        # scipy CSR for eigs / stability tooling; k folded into values
print(A.shape, A.nnz, lap.exc.nrows, lap.blx[0].taps)

# ---- system: rhs is a traced Program over V --------------------------------
src = sp.diff(S, t) - k*(sp.diff(S, x, 2) + sp.diff(S, y, 2))     # MMS source, separable in t -> tables * cos(wt)

def rhs(u, t):                                     # u: pc.Field over V; traced once, lowered to kernels
    return k*lap(u) + pc.where(dom.src_mask, src, 0.0)             # src_mask = fluid & ~dirichlet (D) | non-Dirichlet R

sys = pc.System(dom, state=pc.Field("u"), rhs=rhs, boundary=bc, exact=S)
u0  = sys.project(S, t=0.0)                        # S at fluid D and all R points, 0 at solid points
print(u0.D.reshape(mesh.shape)[25, 25], u0.Rx[:3])

# ---- compile to a plan (CPU f64 now; backend="metal", dtype="f32"/"df64" later) ----
run = pc.compile(sys, integrator=pc.RK4(), backend="cpu", dtype="f64")
dt  = 0.5 * mesh.hmin**2 / (4*float(k))             # same parabolic CFL as heat::timestep_size
res = run(u0, t0=0.0, t_end=0.3, dt=dt,
          stats=["minmax", "linf_error"], stats_every=1,          # masked reductions over fluid D and non-Dirichlet R
          write=pc.XDMF("out", every_time=0.01))
print(res.steps, res.linf_error[-1])

# ---- the time loop is also available explicitly ------------------------------
u, tt = u0, 0.0
while tt < 0.3:
    u = run.step(u, tt, dt)                         # one C call: ~12 kernels, no Python per kernel
    tt += dt
err = pc.linf(u - sys.project(S, t=tt), mask=dom.fluid_mask)
```

### 5.4 IR and lowering

The IR is small, closed and typed by index space:

```text
Space  := D(nx,ny,nz) | R(d) | V | Face(d, side) | Pencil(d)
Vec    := (name, Space, dtype)
LinOp  := Band(d, taps) | Masked(bits, LinOp) | Slab(d, side, r, coef[T], pencil_bits)
        | Sparse(rows: Space, cols: Space, CSR) | Diag(Vec) | Scale(c) | Sum[...]
        | Embed(Space -> V) | Restrict(V -> Space)
Map    := (SymPy expr over Vecs, coords, t; Space; optional mask bits)
Reduce := (min | max | argmax | sum, mask, Map)
Node   := Apply(LinOp, in, out, accumulate) | Map | Reduce
Program:= DAG of Nodes; integrator stages are Programs parametrised by (a_i, b_i, t_i)
```

Lowering passes: P1 fold (Scale into taps, slab coefficients and CSR values; Sum of Sparse into one CSR, duplicates summed where the C++ keeps them separately, `src/matrices/csr.cpp:10-35`; Diag on the left of Band becomes `rowscale`, Diag on the right is rejected to CSR because it breaks constant taps). P2 ownership (STRUCT writes every row of its output, 0 if unowned; EXC accumulates; rows in both are correct because EXC runs after STRUCT). P3 map fusion (a Map feeding only an Apply input becomes a prologue if its cost is at most about 2x the band cost, else it is materialised; a Map consuming an Apply output is fused as an epilogue only if affine in the output; constant-in-time parts become tables; sources separable in t, detected by `expr.as_independent(t)`, become a table times a plan scalar). P4 stage fusion (matching `src/temporal/rk4.cpp:16-47`: STRUCT's epilogue writes u1' = u0 + a dt f_struct with the Dirichlet fill, and rk += b dt f_struct, for all rows; EXC's epilogue adds its contribution at its rows; the last stage writes u_new = u0 + rk directly). P5 buffers (V-sized u0, u1, rk; slot roles are plan parameters so the per-step swap of `simulation_cycle.cpp:123-125` is a permutation). P6 emit (one translation unit per backend printed by a target printer derived from `StencilCodePrinter`; compile with clang/gcc, `MTLDevice.newLibraryWithSource` or NVRTC; cache by hash; build the Plan).

Worked example: the heat.lua RHS plus RK4 stage lowers to K1 STRUCT (dirs {x,y}, band_bits, four wall slabs, epilogue with the masked source table times cos(wt) and the Dirichlet fill) and K2 EXC (2 x 16 object closure rows plus 16+16 R rows, epilogue adding a dt e and b dt e). Per step: 4 x (K1, K2) plus one REDUCE, 9 kernel records, one C call.

### 5.5 Backends, including the Metal path

| Backend | Status | Mechanism | Caveats |
|---|---|---|---|
| CPU fp64 | Now | `struct_kernel`: OpenMP over i-blocks, j tiled by 8-16, k innermost and SIMD; the i and j ranges are split into [0,R), [R,n-R), [n-R,n) so the interior region has no slab branches; band_bits read as one uint64 word per 64 consecutive k and tested per lane. `exc_kernel`: OpenMP over unique rows, no atomics. Precompiled instantiations for W in {3,5,7,9,11} ship in the wheel | Assumes "OpenMP (libomp on macOS)"; libomp is not installed on this machine, so a toolchain step is unbudgeted. Single-thread already saturates about 96 GB/s on this host according to `00-apple-gpu-facts.md` (unverified), so threads matter in regimes A and B rather than C |
| Metal | Next | The same four kernels as MSL templates with `T = float` or a df64 struct (about 150 extra lines for df64 add/mul/fma), compiled at setup from source; `struct_kernel` uses a threadgroup of 256 threads along k with a threadgroup-memory tile of 256+2P values; K RK steps encoded into one command buffer; stats read back after `waitUntilCompleted` once per K steps; shared-storage buffers wrapping page-aligned NumPy arrays | fp64 CPU is the reference; 'f32' mode runs everything in fp32 (viability unprejudged); 'df64' covers the structural kernels and tables while transcendental maps are CPU-evaluated into tables or reduced to separable time factors. The 120 us round trip the design assumed is an earlier unverified forum figure; the measured sync floor on the owner's M4 is 167 to 181 us (`05-measurements.md` Section D). Expected win only in regime C on Pro/Max parts |
| CUDA | Later | The same headers compiled as CUDA C++ through NVRTC via `cupy.RawModule`, fp64 native; stream capture of one K-step sequence into a CUDA graph | Drops Kokkos entirely; the lab has no cupy/NVRTC glue today |

### 5.6 Performance argument

Traffic model per RK4 stage (fused, fp64): reads u1 (about 1 pass with blocking), u0, rk, source table (1 pass when separable), masks (0.05 pass); writes u1', rk. About 6 passes = 48 B per point per stage plus EXC traffic under 1% of a pass. Per step about 25 passes, about 200 B per point.

| Regime | C++ bar | Pencil projection | Basis |
|---|---|---|---|
| A: heat.lua, |V| = 2633 | RHS 30-40 us; step 0.3-1 ms (estimate) | Per stage about 4 us (STRUCT 2-3 us, EXC under 1 us, epilogue folded); per step 20-25 us with one masked reduction and one nanobind call | Even at 3x pessimism the step stays under 80 us against a C++ step that is launch-bound at about 300 launches x 0.7-0.8 us. Unmeasured |
| B: 32^3 / 64^3 | RHS 157 / 1646 us (measured); slot ops about 0.7 ms per step at 64^3 | RHS-equivalent 16-50 us at 32^3, 130-400 us at 64^3 | Assumes a fused 3-axis masked stencil at 0.5-1.5 ns per point on 4 cores; the baseline's own expression kernels do 0.25-0.4 ns per element per pass. Kill criterion: above 2 ns per point at 64^3 |
| C: 256^3 plus sphere | RHS 30-110 ms; step plausibly 0.4-1 s | 3.4 GB per step: 35 ms at 96 GB/s, 28 ms at 120 GB/s; on an M4 Pro/Max GPU in df64 6-12 ms | Exc rows are about 7 MB per RHS, 0.05 passes; the band mask adds 6 MB per direction. Projected |

Where Pencil can lose: mask-bit decode and blend in the SIMD inner loop (expected under 10%, unmeasured); x-direction neighbours stride ny*nz apart, so j-blocking must keep 2P+1 planes of a j-tile in L2 (164 KB at nz = 256, fits); one call per step or per K steps; compile at setup 1-3 s, cached; Metal readback latency amortised by K-step command buffers.

### 5.7 Idiom coverage

| Idiom | Mechanism in Pencil | Risk |
|---|---|---|
| I01 | One packed `f64[|V|]` per field; `State = f64[ncomp,|V|]`; plan owns u0, u1, rk | low |
| I02 | Vectorised NumPy ray casting, identical snap/clamp, lexsort; validated against geometry.csv | medium |
| I03 | Fragments per pencil via searchsorted on pencil id; `dirichlet_line` pencils dropped; `Frags` SoA | low |
| I04 | Setup bitsets over D or R consumed by epilogues | low |
| I05 | `taps(h)` and psi=1 closures from SymPy blocks or the C++ oracle; alpha length validated | low |
| I06 | One `lambdify` per (scheme, bc, side) over the whole psi array; clamps and guards re-added | medium |
| I07 | Host port of `stencil::interp` (`src/stencils/stencil.hpp:213`) and `interp_deriv_coefficients` with per-pencil fragment lookup; convexity assumptions asserted | high |
| I08 | Grouped by (kind, bc): wall ends to slabs, object ends to E rows | low |
| I09 | R rows appended to the same COO | medium |
| I10 | scipy COO to CSR, compacted to unique nonempty rows | low |
| I11 | No per-line copies | low |
| I12 | Python front end; `lupa` importer for parity runs | low |
| I13 | Same C-order layout and `ic()`; int32 V indices | low |
| I14 | R ordering is explicit data; matrix-equality test makes drift loud | medium |
| I15 | Plan-owned V arrays | low |
| I16 | `Masked(Band)` plus `Slab`: a coalesced 3-axis masked stencil over the whole grid | medium |
| I17 | taps per direction, fused across directions in one pass | low |
| I18 | `Slab` rows in the edge regions of STRUCT | low |
| I19 | Rows of E | low |
| I20 | EXC: compact CSR over V, row-parallel, unique rows | low |
| I21 | B is the R column inside each object closure row | low |
| I22 | Wall Neumann as slab broadcast from a face vector; object Neumann as E rows from g_R (new capability) | low |
| I23 | R rows of E | low |
| I24 | Merged into I23 | low |
| I25 | Not needed: STRUCT defines every row | low |
| I26 | Folded into matrix values | low |
| I27 | Algebraic Sum; STRUCT before EXC | low |
| I28 | Three LinOps batched in one STRUCT launch or fused into the consumer | low |
| I29 | `rowscale_d` per direction | low |
| I30 | `src_bits` in the STRUCT epilogue; R source in EXC rows | low |
| I31 | `where(dir_bits, S_face_table * time_factor, u1')` in the epilogue | low |
| I32 | Plan = ordered kernel records; one call per step or K steps; one command buffer per K steps on Metal | low |
| I33 | `LinOp.__call__` eagerly; `to_scipy() @ v` as oracle | low |
| I34 | SymPy printed into the generated TU; separable sources become tables times a scalar | low |
| I35 | Not supported in the hot loop; NumPy table Maps as a slow path | low |
| I36 | Gone | low |
| I37 | Stage update fused into epilogues (P4) | low |
| I38 | Slot permutation | low |
| I39 | Boundary-only evaluation inside the epilogue | low |
| I40 | Setup bitsets | low |
| I41 | REDUCE kernel in one pass; read back every K steps | low |
| I42 | MAP only when a dump is due; NumPy writes the same binary layout | low |
| I43 | Python loop over `run.step` | low |
| I44 | `to_scipy()` in any dimension | low |
| I45 | scipy one-liners | low |
| I46 | Map nodes | low |
| I47 | `State` batching; flux Maps as prologues; see 5.10 | medium |
| I48 | Superseded by `to_scipy()` | low |
| I49 | `CutCellResult`/`InterpResult`, `apply_cse`, `StencilCodePrinter` subclassed; conservation weights as quadrature | low |
| I50 | scipy LinearOperator from `to_scipy()` | low |
| I51 | In-process | low |

### 5.8 Awkward or unsupported

- Per-point callbacks dropped.
- Solution-dependent stencils (WENO, limiters, adaptive footprints) are not expressible as LinOps; there is no stencil-map primitive, so shock-capturing Euler needs a new kernel kind. Pencil is the only design that says this plainly.
- Moving geometry only by host re-assembly per step.
- Column-scaled or spatially varying taps (div(kappa(x) grad u) in non-conservative form) fall back to CSR rows and lose the Toeplitz advantage (3-5x more bytes for those rows).
- fp64 on Apple GPUs: only structural kernels and tables get df64; transcendental maps are fp32 on the GPU or CPU tables.
- Compile-at-setup toolchain dependency; the AOT fallback evaluates maps in NumPy per stage.
- Short fragments, fully solid pencils, concave bodies and planar objects inherit the C++ restrictions; assembly raises.
- Outputs are total at uncovered points; parity only on fluid rows.
- Implicit integration is available through scipy on the exported matrix but is not designed for large 3D.
- No MPI; one device per plan.
- int32 V indices cap at 2^31 points.

### 5.9 Reuse of existing code

Reused as is or with small changes: the stencil_gen symbolic layer (`temo.CutCellResult`, `interp.InterpResult`, `codegen.apply_cse`, `printer.StencilCodePrinter`, `conservation.py`, the RBF solves in `phs.py`); the stability cascade unchanged, now consuming `LinOp.to_scipy()`; the mesh algorithms as a specification (`object_geometry.cpp`, `mesh.cpp:30-137, 170`, `stencil.hpp:213`, `derivative.cpp:22-70, 159-449`) ported line by line and verified against C++ outputs; the C++ stencil structs through a 150-line nanobind module as the coefficient oracle for E4_2, the E4u families and E4_1; a parity harness applying `ccs::derivative`/`laplacian` to unit vectors (2633 applications for heat.lua) and dumping geometry.csv; the IO layout; Lua configs via lupa. Not reused: the Kokkos kernels (`block.hpp`, `csr.cpp`), `field_registry`, `expr.hpp`, the graph code, `systems/`, `temporal/`, `simulation/`.

### 5.10 Euler extension

State `U = f64[5, |V|]`, component-major. Program: dU/dt = -sum_d Dx_d(F_d(U)) plus wall and face terms, where Dx_d is the first-derivative LinOp from the same assembler and F_d a pointwise SymPy Map. The flux Map is a STRUCT prologue: the kernel evaluates primitives once per footprint point per tile; on CPU recompute is cheap, on GPU v1 materialises F_d per direction (3 x 5 V buffers, 2 GB at 256^3 fp64) and v2 fuses. Traffic per stage with a fused prologue is 25 passes against 35 for a materialised design. Boundary physics: slip at R points as a BlockDiag(3x3 from `normal[nR,3]`) LinOp in the stage epilogue; characteristic inflow/outflow as SymPy-generated eigen-decomposition Maps on Face spaces; Dirichlet from the vortex solution. dt as a REDUCE Map producing a device scalar (Metal either reads back per step or runs a frozen dt for K steps with a safety factor). A filter LinOp with its own closures in the same BLX. Diagnostics: per-component REDUCE, multi-variable XDMF, a conservation monitor with the SymPy weights, and `to_scipy()` of the operator linearised about a state, J = -sum_d Dx_d Diag(dF_d/dU). Shock capturing is outside the angle.

### 5.11 Migration phases

| Phase | Duration | Content | Gate |
|---|---|---|---|
| 0 | 2-3 weeks | NumPy assembly of the heat.lua laplacian and a 64^3 grid as BLX plus Exc plus `to_scipy()`; parity harness against the C++ operator applied to unit vectors; precompiled STRUCT/EXC kernels for w in {3,5}; timing against a native rebuild | The prototype of 5.14 |
| 1 | 6-8 weeks | Heat at full parity on CPU fp64: front end, RK4/Euler stage fusion, SymPy map codegen with cache, separable-source tables, masked stats, XDMF, Lua importer; schemes E2_1, E2_2, E4_2, E4u_1; CI job comparing C++ and Pencil fields to 1e-12 | |
| 2 | 3-4 weeks | scalar_wave, hyperbolic_eigenvalues via `to_scipy()`, stencil_gen's BO and stability loops wired in-process, object Neumann and 3D tests; Pencil adopted as the reference for R-point values via the monomial oracle | |
| 3 | 4-6 weeks | Metal: metal-cpp/nanobind runtime, MSL kernels (f32 and df64), K-step command buffers, zero-copy buffers; fp32 versus fp64 convergence study | Regime-C numbers on a Pro/Max part |
| 4 | 8-12 weeks | Euler: State batching, flux prologues, wall projection and characteristic BCs, adaptive dt, filter, isentropic vortex, conservation monitor; CUDA runtime (1-2 weeks of that budget) | |
| 5 | ongoing | Freeze the C++ runtime as a parity reference; keep `src/stencils` as oracle until every scheme regenerates | |

### 5.12 Effort

About 6-8 calendar months to Euler-capable with CPU and Metal; heat/scalar_wave parity on CPU in about 3 months. Code: about 5-7k lines of Python (geom 800, schemes 600, assemble 1200, ir and lower 1500, sim/io 1000, tests 2000), about 1.2k lines of C++ (kernels 600, runtime 500, stencil oracle 150), about 800 lines of MSL plus 400 of metal-cpp glue, about 400 lines of CUDA later. Setup per run: about 50 ms at heat.lua scale, 5-20 s at 256^3 (dominated by the I07 rows in pure Python).

### 5.13 Risks

- R-point parity has no trusted reference; mitigation is the reference-free monomial oracle (apply the assembled matrix to monomials and check exactness to scheme order at every fluid and R row). This is the one proposal in the set that escapes the broken t-laplacian reference.
- Bit-exact geometry (snap, clamp, nextafter, lexsort order): a one-ulp difference silently reorders R.
- Performance claims rest on an unmeasured C++ step and on expected stencil throughput.
- fp32 viability on Apple GPUs; df64 only for structural kernels.
- Compile-at-setup toolchain fragility; the precompiled instantiations are the fallback.
- Three kernel dialects to keep in sync.
- The I07 port is the most intricate path.
- SymPy pipeline gaps keep the C++ stencil oracle alive longer than planned.
- Euler fusion complexity; v1 accepts 2x traffic.
- Scope creep in the front end; the IR must stay closed.
- Found in judging, not in the design: with three buffers (u0, u1, rk) and STRUCT writing u1' in place, EXC runs afterwards and needs the old u1 at its gathered columns, which has been overwritten. The fix is a fourth buffer (ping-pong), which changes the traffic count slightly and is adopted in the hybrid.

### 5.14 Killer prototype experiment

Build the BLX plus Exc decomposition of the heat.lua laplacian (51x51, E2_2, four Dirichlet walls, one floating circle) in NumPy with the lambdified E2_2 closures, cross-checked against the C++ stencil oracle, and the same for a 64^3 grid with a sphere of radius 0.25L. (1) Parity: assemble A_py over V (|V| = 2633); obtain A_cxx by applying the C++ laplacian plus Bf/Br to the 2633 unit vectors through a 60-line harness; pass if max |A_py - A_cxx| is at most 1e-12 times max |A_cxx| on all fluid and R rows, and if A_py applied to monomials x^m y^n (m+n at most 3) reproduces the Laplacian exactly on every fluid and R row. (2) Performance: the precompiled STRUCT (w=3, 2 directions) plus EXC kernels for one RHS and one fused RK4 step, 1000 repetitions, on the development Mac with 4 threads, against a native rebuild of bench_rhs and a timed heat.lua run; pass if the RHS-equivalent is at or below the C++ RHS at 51x51 (bar about 32 us) and at 64^3 (bar 1.6 ms, target 0.5 ms), and the fused step is at or below the measured C++ step at 51x51. Kill: parity fails on R rows after the C++ oracle is matched (the decomposition is wrong), or STRUCT exceeds 2 ns per point at 64^3 (the masked-Toeplitz format does not recover stencil bandwidth). About 2-3 weeks.

## 6. D4 Kerf-lang: a kernel DSL with a typed cut-mesh data model

### 6.1 Pitch

Every spatial operator becomes L u = S u + E u. S is one constant-coefficient, masked, multi-axis stencil sweep over the full box (about 91% of D rows, about 97% of fluid rows). E is one sorted, unique-output ELL table holding every non-uniform row: wall and object closures, B, N, Bf, Br, plus Neumann-object constraint rows eliminated at setup. Both are built once on the host in fp64 from the existing C++ mesh and stencil code. Users write pointwise physics, MMS functions, BCs and reductions as typed Python-subset kernels (`@kf.func`, `@kf.pointwise`, `@kf.reduce`). The compiler traces the RHS and RK step, fuses pointwise producers and consumers into the sweep prologue and epilogue, merges E rows into the sweep with a sorted merge-join, and emits one native function that runs K whole RK4 steps per Python call. The decision is a small custom compiler (Python AST to a typed Point IR, a fusing scheduler, Jinja2-templated C++/MSL/CUDA), not Taichi, Warp, Numba or Mojo, because none offers f64 CPU plus df64 Metal plus cross-kernel fusion: Taichi has no Metal f64, no fusion and no release since 2025-07 (`frameworks/taichi.md`); Warp runs CPU launches serially and has no Metal (`frameworks/warp.md`); Numba has no Metal (`frameworks/numba.md`); Mojo is not Python (`frameworks/mojo.md`).

### 6.2 Architecture

- L0 Front end: `Grid`, `Shape`, `CutMesh`, `BCs`, `schemes.*`, `ops.*` (LinearOp algebra), `Field`, the kernel decorators, `kf.RK4`/`Euler` steppers, `kf.compile(..., backend, dtype)`.
- L1 Setup engine (host fp64, NumPy plus a nanobind module `kerf._cpp`): builds the typed cut-mesh data model and the operator tables. Phase 0 delegates to the existing C++ (`object_geometry`, `mesh::init_line`, `derivative::domain_discretization` and `cut_discretization`, the `src/stencils` nbs/interp code) and converts their outputs; later phases port pieces to NumPy and check them bit-for-bit.
- L2 Kerf IR: a Point IR (typed SSA per point) plus a Program graph (`Pointwise`, `LinearApply`, `Enforce`, `Constrain`, `Reduce`, `StageUpdate`).
- L3 Scheduler: fusion into sweeps, scratch and liveness, ping-pong buffers, hoist-versus-recompute for time-separable terms.
- L4 Codegen: Jinja2 templates per (sweep kind x backend). A C++20 TU compiled by clang++ into a `.so` with a C ABI and loaded via ctypes; MSL compiled at run time; CUDA through NVRTC. A NumPy reference interpreter of the Program graph serves as oracle.
- L5 Runtime (`kerf-rt`, header-only C++, about 2k lines): a persistent spin-barrier thread pool (correctly noting that Apple clang ships no OpenMP), a 64 B-aligned field allocator, a stats ring, a df64 library with C++ and MSL versions.

Typed data model (SoA, immutable after setup): `Grid`; `Space` D | Rx | Ry | Rz with static offsets; `Field[ncomp, dtype]` as one `(ncomp, nU)` array; `RPoints[d]` in the C++ order with an added `fluid_ic`; `Fragments[d]` (`src/mesh/mesh_types.hpp` flattened) exposed as `mesh.lines[d]`; `PointSet` as a tagged union of face slices, D bitmasks and R index lists with set algebra evaluated at setup; `PointClass` bits u8 per D point (bit d = interior in direction d, bit 3 fluid, bit 4 Dirichlet face, bit 5 has E row), 16.8 MB at 256^3.

Operator tables per LinearOp after composition: S as constexpr per-direction taps baked into generated code; E sorted by output in packed U space, unique per output, with `out`, `nnz`, `cols`, `coef_id` into a deduplicated coefficient table (all wall rows of one kind share an id), optional `g_col` for boundary data, and a `tile_ptr` (tile = 8x8x8) so GPU sweep threads find their E rows without a D-sized map. heat.lua: about 700 rows. 256^3 sphere: about 1.0-1.1M unique outputs plus 76k R rows, roughly 40-70 MB. A constraint matrix A for Neumann/Robin objects is assembled by `scipy.sparse.linalg.splu` of the n.grad rows and eliminated into E (E := E + E_R A), so per-stage work has no extra phase.

Kernel shapes emitted: `K_sweep` (box sweep with a 2P+1 plane ring buffer on CPU, register queues and a threadgroup tile on Metal; prologue, core, E patch by merge-join cursor or `tile_ptr` lookup, epilogue), `K_rows` (E-row gather-dot for R outputs and the GPU `corr[]`), `K_rlist` (R-space epilogue), `K_reduce`. One RK4 step = 4 x (K_rows, K_sweep, K_rlist) plus fused reductions; one function call per K steps on CPU, one command buffer per K steps on Metal.

### 6.3 User-facing example

```python
import kerf as kf
from kerf import math as km

# ---- geometry and cut mesh (setup on host in fp64; ray casting reuses the C++ object_geometry) ----
grid   = kf.Grid(shape=(101, 101), lo=(-1.0, -1.0), hi=(1.0, 1.0))
sphere = kf.Sphere(center=(0.1, -0.05), radius=0.3, name="body")
mesh   = kf.CutMesh(grid, shapes=[sphere])   # mesh.D, mesh.R[0], mesh.R[1], mesh.lines[d], mesh.fluid

K, OMEGA, SIG, C = 0.1, 2.0 * km.pi, 0.25, (0.4, -0.3)    # captured as compile-time constants

# ---- manufactured solution: Python-subset kernels, compiled for D points and R points ----
@kf.func
def ue(x: kf.real, y: kf.real, t: kf.real) -> kf.real:
    return km.exp(-((x - C[0])**2 + (y - C[1])**2) / SIG**2) * km.cos(OMEGA * t)

@kf.func
def grad_ue(x: kf.real, y: kf.real, t: kf.real) -> kf.vec2:
    g = -2.0 / SIG**2 * ue(x, y, t)
    return kf.vec2((x - C[0]) * g, (y - C[1]) * g)

@kf.pointwise                                   # defined on any PointSet; p carries coordinates
def source(p: kf.Point, t: kf.real) -> kf.real:
    r2 = (p.x - C[0])**2 + (p.y - C[1])**2
    dudt = -OMEGA * km.exp(-r2 / SIG**2) * km.sin(OMEGA * t)
    lap  = (4.0 * r2 / SIG**4 - 4.0 / SIG**2) * ue(p.x, p.y, t)
    return dudt - K * lap

@kf.pointwise(over=mesh.R_of("body"))           # R points only: p.normal and p.psi exist here
def flux_n(p: kf.RPoint, t: kf.real) -> kf.real:
    return kf.dot(p.normal.xy, grad_ue(p.x, p.y, t))

# ---- boundary conditions: Dirichlet on all four walls, Neumann on the obstacle ----
bcs = kf.BCs(mesh,
             walls=kf.Dirichlet(ue),                  # ue(p.x, p.y, t) at face points only
             objects={"body": kf.Neumann(flux_n)})   # body R values become constrained DOFs

scheme = kf.schemes.E4_2(alpha=kf.known_values("E4_2"))  # psi closures via src/stencils / stencil_gen
lap    = kf.ops.laplacian(mesh, scheme, bcs)             # LinearOp on Field(D|Rx|Ry): S + E tables

@kf.rhs(bcs)                                    # traced once at compile time, never run eagerly
def heat(u: kf.Field, t: kf.real) -> kf.Field:
    return K * lap(u) + source(t)               # one sweep: stencil + E rows + epilogue

@kf.reduce(over=mesh.fluid | mesh.R, op="max")
def linf(u: kf.Field, p: kf.Point, t: kf.real) -> kf.real:
    return km.abs(u[p] - ue(p.x, p.y, t))

step = kf.RK4(heat)                             # stage axpy + BC enforcement + RHS in one native fn
prog = kf.compile(step, reductions=[linf], backend="cpu", dtype=kf.f64)
# prog = kf.compile(step, reductions=[linf], backend="metal", dtype=kf.df64)  # same source, Apple GPU

u  = prog.field(init=lambda p: ue(p.x, p.y, 0.0))        # fills fluid D, wall faces and Rx/Ry
dt = 0.2 * grid.h.min()**2 / (4.0 * K)
t, t_end = 0.0, 1.0
while t < t_end - 1e-12:
    n = min(200, round((t_end - t) / dt))
    t, stats = prog.advance(u, t, dt, nsteps=n)          # n RK4 steps per native call
    print(f"t={t:.4f}  linf={stats['linf'][-1]:.3e}")    # per-step values buffered on device
    if not stats.ok:                                     # NaN/limit check runs every step natively
        break
kf.io.write_xdmf("out/heat", u, mesh, t)                 # D then Rx, Ry, Rz, same layout as today
print(lap.to_scipy().shape)                              # exact cut-cell operator for eig/GKS work
```

Note that the MMS, its gradient and its Laplacian are hand-coded here rather than derived by `sympy.diff`. The numerics judge counted this against the design: it discards the SymPy derivation that D1 and D2 exploit and invites MMS mistakes that look like scheme bugs.

### 6.4 IR and lowering

Front end: `@kf.func`, `@kf.pointwise` and `@kf.reduce` bodies are parsed with `ast.parse(inspect.getsource(f))` and a visitor builds Point IR. Allowed: typed params, local assignment, arithmetic with integer exponents, comparisons, if/else lowered to select or structured branches, `for` over a constant range (unrolled), calls to other `@kf.func` (inlined) and `km.*` intrinsics; closure-captured numbers become constants; field access is only `u[p]` at the same point, neighbour access being reserved for library kernels with a declared halo. Rejected: Python objects, recursion, dynamic containers, exceptions, unbounded loops. Programs are traced with proxies into a DAG over Field values.

Point IR: SSA values typed {f64, f32, df64, i32, i64, bool, vecN<T>}, ops {add, sub, mul, div, fma, neg, cmp, select, intrinsic, load_field, load_attr, load_param, const}; kernels are polymorphic in a `real` type variable bound at compile.

Setup-time lowering of LinearOps: assembly via `kerf._cpp` (phase 0) yields inner_blocks and the CSRs; decomposition into S+E sets PointClass bit d for rows whose coefficients equal the direction's interior taps and makes everything else an E row with B, Bf and Br appended; constraint elimination for Neumann objects; dedup of the coefficient table; a verification hook assembles S+E as scipy CSR and compares with the C++ operator on random vectors to 1e-14 relative, which also provides `to_scipy()`.

Compile-time lowering: fusion rules (Pointwise into Pointwise always; Pointwise into a LinearApply input is `compute_at` the sweep window; LinearApply into Pointwise or StageUpdate fuses into the epilogue; LinearApply into LinearApply is a materialisation barrier; Reduce fuses into its producer's epilogue; `Enforce` at t_{s+1} fuses into stage s; `Constrain` materialisation only at chunk end). A time-separation pass detects sum_m f_m(x) h_m(t) by dependence analysis and either hoists f_m into aux fields or recomputes in-kernel, using a cost model (hoist when flops per point exceed 20 and the field is cache-resident, recompute when bandwidth-bound). Boundary-only evaluation of Dirichlet values (204 points for heat.lua). A buffer plan with ping-pong stage inputs. A precision pass: f64 native on CPU and CUDA; `df64` rewrites every f64 op to df64 intrinsics with range-reduced polynomial transcendentals; constants and table coefficients split on the host into (hi, lo) float pairs from the fp64 values; `f32` casts tables on upload and keeps accumulation in df64 by default. Codegen through one Jinja2 template per sweep kind per backend, specialised by constexpr extents, taps, ncomp and P; geometry tables are runtime arguments so a new psi, alpha or shape at the same grid size does not recompile. ABI: `extern "C" int kerf_advance(Ctx*, int nsteps, double t0, double dt, double* stats_ring, int* status)`.

### 6.5 Backends, including the Metal path

| Backend | Status | Mechanism | Caveats |
|---|---|---|---|
| CPU fp64 | Primary | Generated C++20 compiled by the system clang++ (-O3 -mcpu=native -ffp-contract=fast, no -ffast-math), loaded via ctypes, with the persistent thread pool; single-threaded when nD is under about 60k; vectorisation along the fast axis with `#pragma clang loop vectorize(enable)`; the E patch is a scalar fix-up after the vector body for a tile; a 2P+1 plane ring buffer for x taps in 3D | The merge-join patch sits inside the vectorised loop; if clang does not vectorise it well, large 3D falls back to unfused sweeps |
| Metal | Secondary, df64 default, fp32 opt-in | MSL generated from the same Point IR, compiled at run time, cached; a 1.5k-line metal-cpp extension owns shared-storage buffers and encodes K steps x 4 stages x {K_rows, K_sweep, K_rlist} into one command buffer; reductions via threadgroup reduction plus atomic max on the uint bit pattern; df64 storage as float2 | df64 library about 600 lines of MSL with a C++ twin for unit tests. fp32 mode quantified as risky: E4_1 clamps imply coefficients up to about 1e4/h, so fp32 row cancellation can lose about 3 decimal digits; df64 keeps about 1e-10. Recommended only for 3D at 128^3 or more on Pro/Max |
| CUDA | Tertiary | The same backend-neutral sweep bodies wrapped by a `__global__` kernel (2.5D blocking); NVRTC through cuda-python; the K-step loop captured into a CUDA graph | No Kokkos dependency |

### 6.6 Performance argument

Where the C++ loses time today: about 280 launches at about 0.8 us, about 26 fences, about 14 full-field host MMS sweeps, 24 heap allocations, about 18 index-View uploads, full-height B/N row-pointer sweeps, and one-team-per-strided-line block matvecs. Kerf-lang removes all of these structurally.

| Regime | C++ bar | Kerf-lang projection | Basis |
|---|---|---|---|
| A: heat.lua | RHS 30-40 us; step 0.3-1 ms (estimate) | Per RHS about 3-5 us; per step about 15-30 us single-threaded | 2601 x 2 axes x 5 taps at 0.5-1 ns per point; about 700 E rows; hoisted separable source; Dirichlet face values hoisted. Metal at this size roughly ties the CPU. Unmeasured |
| B: 32^3 / 64^3 | RHS 157 / 1646 us | About 0.15-0.25 ms per stage at 64^3; 25-40 us at 32^3 | 5 doubles per point per stage, SLC-resident, on 4 threads. Unmeasured |
| C: 256^3 plus sphere | RHS 30-110 ms; step 270-300 ms | About 8 ms per stage, about 32 ms per step; if RK fusion fails about 10 ms per stage | 670 MB per stage at the 96 GB/s figure; E rows 0.5-1 ms; PointClass bits 0.2 ms; the slow-axis taps cost no DRAM traffic because of the 9-plane ring buffer. Metal df64 on M4 Max about 1.7 ms per stage, M4 Pro about 3 ms, base M4 about 1x. All projected |

Compile cost: clang -O3 of one generated TU (3-8k lines) 1-3 s, cached; Metal library compile 0.2-1 s. Setup at 256^3 requires replacing the O(|R|) `interp_line` scan with searchsorted first.

### 6.7 Idiom coverage

| Idiom | Mechanism in Kerf-lang | Risk |
|---|---|---|
| I01 | One packed `(ncomp, nU)` array per field; program-owned u0, ping-pong u_a/u_b, acc, aux | low |
| I02 | Phase 0: `kerf._cpp` wraps `object_geometry` (bit-exact); later a NumPy port accepted only if bit-identical | low |
| I03 | `init_line` and `dirichlet_line` output converted to `Fragments[d]`, exposed as `mesh.lines[d]` | low |
| I04 | `PointSet` union built once; fluid as a PointClass bit | low |
| I05 | Taps baked as constexpr; uniform closures in the deduplicated E coefficient table | low |
| I06 | Batched setup evaluation through nanobind bindings of the `src/stencils` nbs (keeps hand-written guards); a later printer retarget cross-checks | low |
| I07 | Phase 0 via the binding; phase 2 NumPy port with searchsorted | medium |
| I08 | Table builder groups fragments by (kinds, BC); rows dropped exactly as `submatrix_size` (`derivative.cpp:259-303`) | medium |
| I09 | Bf/Br rows merged per R output into E rows | low |
| I10 | No runtime CSR; `np.lexsort` into sorted unique-output ELL | low |
| I11 | Dedup coefficient table plus `coef_id` | low |
| I12 | Python front end; optional `kf.from_lua` via lupa | low |
| I13 | Constexpr strides; i64 indices emitted when nD at or above 2^31 | low |
| I14 | R order kept exactly; a checksum of (psi, solid_ic) order stored with tables and verified on load | medium |
| I15 | Spans and work buffers vanish | low |
| I16 | `K_sweep` masked constant stencil plus E merge-join patch | medium |
| I17 | S core with SIMD along the fast axis and ring-buffered planes | low |
| I18 | E rows with shared `coef_id`; a fast path emits dedicated plane loops when a whole face is a wall of one kind | low |
| I19 | E rows with unique coefficient rows | low |
| I20 | One sorted ELL gather-dot with unique outputs, no atomics | low |
| I21 | B entries as extra columns on the E row of their D output | low |
| I22 | Neumann wall extras as E-row columns pointing at a boundary-data slot | low |
| I23 | E rows with outputs in R positions | low |
| I24 | Merged into the same E row as Bf | low |
| I25 | No zero fill: the epilogue defines du totally | low |
| I26 | Folded into S taps and E coefficients | low |
| I27 | Deterministic per-point sum | low |
| I28 | Multi-output sweep emitting du_x, du_y, du_z in registers | low |
| I29 | Epilogue Pointwise over the in-register gradient | low |
| I30 | Masked add via PointClass bits; `K_rlist` over R lists | low |
| I31 | `Enforce` node: du=0 at Dirichlet rows, u=g(t_next) written by the previous epilogue | low |
| I32 | Whole K-step native function; one command buffer per K steps; CUDA graph | low |
| I33 | `backend='numpy'` interprets the Program DAG | low |
| I34 | `@kf.func` compiled pointwise; time-separation pass; boundary-only evaluation | low |
| I35 | Unsupported in the hot path | low |
| I36 | t is a kernel parameter | low |
| I37 | RK4 stage arithmetic fused into the sweep and R epilogues | low |
| I38 | Ping-pong pointer swap inside the native loop | low |
| I39 | `Enforce` and `Constrain` nodes; Neumann objects eliminated into E at setup | medium |
| I40 | PointSets built once | low |
| I41 | `@kf.reduce` fused into the final stage epilogue into a per-step stats ring; NaN check sets a status flag | low |
| I42 | Host writes from numpy views at chunk boundaries | low |
| I43 | Python loop over `prog.advance` chunks; device-side dt for Euler | low |
| I44 | `LinearOp.to_scipy()` over packed U | low |
| I45 | NumPy reference backend | low |
| I46 | Point IR pointwise fusion | low |
| I47 | `Field(ncomp)`, flux kernels in the sweep prologue, batched S/E, slip constraints via `p.normal`, device dt | high |
| I48 | Replaced by `to_scipy()` | low |
| I49 | Reused for derivation; new printer targets (NumPy batch evaluator, `@kf.func` source) | low |
| I50 | Consumes `to_scipy()`; Jacobians via a 'dual' dtype lowering or finite differences | medium |
| I51 | In-process; alpha, psi and geometry are runtime table data | low |

### 6.8 Awkward or unsupported

- Arbitrary Python inside kernels is rejected at parse time.
- User kernels cannot do neighbour access across fragment ends or into R points; spatial coupling must go through a LinearOp. The maintainer judge drew the conclusion: the kernel DSL is in practice a pointwise DSL, and a pointwise DSL is a SymPy expression printer, which the repo already has.
- Lua callbacks unsupported.
- Moving geometry needs setup re-run; device-side `interp()` is not ported.
- Schemes without `interp_info` cannot drive floating or Neumann objects in 2D/3D; the Neumann example needs new cross-direction interpolation rows from `interp.py`.
- Very short fragments are refused until a merged-closure rule is chosen.
- Fully solid pencils, concave bodies, planar objects inherit the C++ restrictions.
- Metal: no fp64; df64 has the fp32 exponent range, flushes denormals and costs 10-20x per transcendental; fp32 unvalidated; small 2D not faster than CPU.
- A LinearOp applied to a LinearOp output forces materialisation.
- Bitwise parity not a goal at solid points or in summation order.
- Native stack traces do not map back to Python lines.
- JIT latency of 1-3 s per (grid shape, scheme, dtype).

### 6.9 Reuse of existing code

C++ via `kerf._cpp` built from the existing CMake targets: `object_geometry.cpp`, `sphere.cpp`, `rect.hpp`, `mesh.cpp` exported as SoA NumPy (the bit-exact source of truth for R ordering); `src/stencils/*.cpp` via `any_stencil` (keeps the hand-edited E4_1 clamp and the non-regenerable E4_2 and spline families); `derivative.cpp` assembly in phase 0, exporting `meta_d`/`coeffs_d` and the CSR arrays; the Catch2 tests and Lua configs as parity oracles; the binary and XDMF layout. Python: `CutCellResult`/`InterpResult` and the CSE codegen with new printer subclasses for a NumPy batch target and a `@kf.func` target; `interp.py` to generate the missing E4 first-derivative cross-direction rows; `known_values.json`; the analysis cascade fed by `to_scipy()`; `bo.py` driving programs in process. Not reused: the Kokkos runtime kernels, graphs, slot_ops, field_registry, expression templates, systems RHS code.

### 6.10 Euler extension

`Field(mesh, ncomp=4|5)` component-major; each component carries D, Rx, Ry, Rz. User code: `@kf.func prim(q)`, `@kf.pointwise flux_x(q)`, `flux_y`, `D1 = kf.ops.gradient(mesh, E4_1(...), bcs)`, `@kf.rhs euler(q, t): return -(D1.x(flux_x(q)) + D1.y(flux_y(q))) + filt(q)`, `@kf.reduce wave_speed`, `step = kf.RK4(euler, dt=kf.cfl(0.5, wave_speed))`. Lowering: flux prologue fusion (`compute_at` the sweep window, so the 15 flux fields are never materialised); batched S/E over components (one coefficient row serves all components); slip wall as a `Constrain` over `mesh.R_of(body)` using `p.normal`; characteristic faces as `@kf.pointwise` over faces with a small dense eigen-decomposition in registers; adaptive dt as a fused reduction on device; a filter LinearOp summed with the derivative at setup when both act on q. Cost at 256^3: 25 doubles per point per stage, 3.4 GB, about 35 ms on CPU at 96 GB/s; Metal df64 on M4 Max about 8.5 ms per stage if df64 division and sqrt stay under the bandwidth limit (the risk to measure). Gaps: E4_1/E4u_1 lack `interp_info`; shocks need a nonlinear filter or limiter outside the linear S+E model unless expressed as a pointwise-weighted LinearOp.

### 6.11 Migration phases

| Phase | Duration | Content |
|---|---|---|
| 0 | 3 weeks | Native build with per-region timers; `kerf._cpp` exporting mesh SoA, batch nbs/interp, block meta/coeffs and CSR arrays; a NumPy reference that reproduces the C++ RHS to 1e-14 relative |
| 1 | 5 weeks | S+E table builder, dedup, sorted ELL, PointClass bits, `to_scipy()`; hand-instantiated C++ templates for the heat RHS plus fused RK4 stage; ctypes loader; K-step advance. This is the validating prototype gate |
| 2 | 7 weeks | The language: AST front end, Point IR, tracing, scheduler, time-separation pass, Enforce/Mask/Reduce nodes, NumPy interpreter, compile cache; heat and scalar_wave ported; cpp_bridge replaced |
| 3 | 4 weeks | 3D performance: thread pool, plane ring buffer, multi-output sweeps, NumPy ports of ray casting and interp with bit-identity tests |
| 4 | 7 weeks | Metal: MSL printer, df64 library with ulp tests, metal-cpp runner, GPU `K_rows`/`corr`/`tile_ptr` path, reductions; fp32 versus df64 versus f64 study on psi sweeps |
| 5 | 8-10 weeks | Euler |
| 6 | 3-4 weeks | CUDA |
| 7 | 2-3 weeks | Lua importer, docs, legacy freeze |

### 6.12 Effort

About 10-13 engineer-months to heat, scalar_wave and Euler on CPU fp64 plus Metal df64; about one more month for CUDA. Size: about 6-8k lines of Python front end/IR/scheduler, 4-6k lines of setup/table builder, 2k lines of `kerf-rt`, 3k lines of templates, 1k lines of df64, 1.5k lines of metal-cpp runner, 1k lines of bindings, tests about equal to code. The validating prototype is 6-8 weeks. Ongoing maintenance 15-25% of an FTE. This is the most expensive of the five.

### 6.13 Risks

- The baseline is unmeasured.
- S+E decomposition fidelity: reproducing every row and column drop of `derivative.cpp` (ldd/rdd, Dirichlet row removal, object boundary row to Bf/Br, Neumann extras, skipped Dirichlet lines) is subtle.
- Codegen quality: the fused sweep must reach about 80% of streaming bandwidth with the merge-join patch and ring buffer.
- df64 on Metal: accuracy and speed of div/sqrt/exp unproven; fp32 may be unusable for small-psi closures.
- New numerics hidden in the example: Neumann object BCs and E4 cross-direction interpolation rows.
- Building and maintaining a custom compiler.
- Toolchain coupling (clang++, Metal compiler, nanobind/metal-cpp packaging); the sandbox hides the Metal device.
- Silent R-ordering corruption from a ported ray caster.
- Scope creep toward a general-purpose DSL.

### 6.14 Killer prototype experiment

Goal: validate S+E fidelity, whole-step CPU speed and df64 Metal viability in 6-8 weeks without the AST front end. (1) Fidelity: `kerf._cpp` exports the C++ laplacian for heat.lua and the scalar_wave.lua gradient (71x71, polyE2_1); Python builds S+E tables and applies them via NumPy and a hand-instantiated generated sweep; compare against the C++-assembled operator on 20 random packed vectors. (2) Speed: a hand-written template instance of the fused RK4 step for heat.lua runs 1000 steps via one ctypes call per 200 steps; compare against the phase-0 native timing; also a 256^3 no-shape E4u_1 heat stage and a 128^3 sphere E2_2 stage. (3) Metal: the same 256^3 sweep plus E rows in MSL, fp32 and df64. Pass if: (a) S+E matches the C++ operator to 1e-13 on covered D rows and R rows with identical R ordering; (b) the heat.lua 1000-step Linf history matches `system.csv` to 1e-11 and the step time is at most 0.5x the C++ step (target at most 60 us); (c) the 256^3 fused stage on CPU is at most 1.0x the C++ RHS (target at most 12 ms, at least 60 GB/s effective); (d) Metal df64 agrees with CPU f64 to 1e-12 and reaches at least 50% of the fp32 kernel's throughput. Kill or redesign: (a) fails structurally, fall back to a general CSR correction; (b) or (c) slower than C++ after one optimisation iteration, abandon the custom codegen and emit Kokkos C++ from the same tables; (d) df64 under 30% of fp32 or error above 1e-10, drop Metal.

## 7. D5 pyshoccs thin waist: nanobind, compact tables, a ten-primitive schedule, no compiler

### 7.1 Pitch

Everything irregular about cut cells is setup-time data that already lives in C++ (`src/operators/derivative.cpp:159-449`, `src/mesh/mesh.cpp`, `src/stencils/*`). The runtime has exactly six kernel shapes (`src/matrices/block.hpp:117-193`, `src/matrices/csr.hpp`, `src/systems/heat.cpp:119-294`). So the right cut is a thin waist: (1) C++ keeps mesh, stencils, BCs and the assembly logic, but assembly now exports compact, deduplicated tables over one packed index space per field; (2) a tiny `ccs::backend` interface with ten primitives consumes those tables through a recorded, replayable `schedule` whose only runtime inputs are buffer ids, t and dt; (3) Python describes the problem, composes operators into schedules, runs the outer loop and exports operators to scipy. The "DSL" is a Python API plus a schedule data structure, not a compiler.

The verdict on whether a real DSL is needed, as this design states it: no, for heat, scalar wave and Euler as scoped, because the kernel vocabulary is closed under "linear operators from tables plus pointwise maps plus reductions" and the slow parts of today's C++ (launch count, full-height CSR sweeps, per-line coefficient copies, host MMS evaluation) are all removed by a fixed primitive set. What is lost: arbitrary cross-kernel fusion (an estimated 1.3-2x left on the table in regimes B and C), user-defined pointwise physics without a C++ compile (mitigated by a SymPy-to-C++ functor plugin compiled at setup), and dtype genericity for free.

### 7.2 Architecture

- L5 `shoccs` Python package: `Grid`, `Shape`, `Mesh` (wrapping the C++ mesh), `schemes.*`, `BCs`, `mms.Symbolic`/`mms.Gaussian`, `Field`/`FieldSpace`, `Derivative`/`Gradient`/`Laplacian`/`Filter`, integrators, `Simulation`, `Operator.to_scipy()`. Replaces `src/lib/run_from_sol.cpp`, `src/app/shoccs.cpp` and every `from_lua`.
- L4 `_shoccs` nanobind extension (scikit-build-core, nanobind at or above 2.4): exposes `ccs::mesh` (lines and R records as SoA arrays), `ccs::stencil`, `ccs::bcs`, a new `ccs::operator_tables`, the backend API and DLPack views. `Kokkos::initialize` runs at import, `Kokkos::finalize` in `atexit` after the arena singleton is destroyed; `schedule.run()` releases the GIL.
- L3 `ccs::backend` (C++20): opaque `buffer` (id, dtype in {f64, f32, df64, i32, u8}, length), an `arena`, ten primitive entry points, and `schedule = std::vector<node>` where a node is {primitive id, buffer ids, table ids, immediates, parameter refs}. Backends: `kokkos_backend` (host today, Cuda later, identical source because every kernel captures Views rather than host `std::vector` pointers as `src/matrices/csr.cpp:26-47` does) and `metal_backend` (metal-cpp, offline-compiled `.metallib`). A schedule is validated once and compiled once.
- L2 Operator tables (the data IR), built once per (geometry, BCs, scheme, alpha).
- L1 Existing C++ kept as is, with the assembly loops of `derivative.cpp` refactored to emit tables instead of (block, csr); `src/matrices/*` retained as the parity oracle.

Packed field layout: one buffer of length Npk = |D| + |Rx| + |Ry| + |Rz| (padded to 8) with static offsets, replacing the four Views per scalar of `src/fields/field_registry.hpp` and the four-span bridge of `src/fields/scalar.hpp`; multi-component state is `[ncomp, Npk]`; every index table is in packed space, int32 by default. Consequences: every slot or elementwise op is one launch instead of four; B, Bf and Br become ordinary rows of the same CSR type; swap instead of deep_copy is legal because schedules bind buffer ids and the arena resolves pointers at `run()`.

Per-direction table set for one derivative: `taps[w]` once; role bits (uint8 per D point, bit d set iff the point is an interior row of some fragment in direction d); face closures `C_L[R_L x T_L]`, `C_R[R_R x T_R]` plus pencil masks; a compact Drow CSR (object-end closure rows with their B column folded in as a column into the R segment, wall-closure rows of short fragments, N entries as columns into a separate `nu` buffer, absent when no Neumann BC exists); an Rrow CSR per r with Bf_r and Br_r merged, duplicates summed. A Laplacian merges three directions: one Drow CSR, one Rrow CSR. Sizes at 256^3 with a sphere (E4u_1): role bits 16.8 MB, Drow about 12 MB per direction, Rrow about 24 MB, about 80 MB per Laplacian against today's about 22 MB `coeffs_d` per direction plus 2 x 134 MB int64 row pointers per direction.

The ten primitives, labelled B1 to B10 to keep them apart from Part I's kernel shapes P0 to P6 (kernel shape in brackets): B1 `lincomb` [1-D range over Npk x ncomp]; B2 `map` with a functor from a fixed menu or a setup-compiled plugin [1-D range]; B3 `axis_stencil` with up to 3 directions, taps, role bits, op, scale and nout 1 or 3 [3-D range, threads along z, tiled so a (2P+1)-plane working set stays cache-resident, total definition]; B4 `face_closure` [2-D range over pencils of the face plane, a batched small GEMM]; B5 `rowset_matvec` [1-D over rows]; B6 `gather`; B7 `scatter` (unique indices by contract); B8 `fill_at`; B9 `reduce` (min, max, maxloc, sum, max_abs_plus) with controlled readback; B10 `point_eval` with a compiled closed form (the gaussian sums of `src/mms/gauss2d.cpp` and `gauss3d.cpp` ported to a device functor) or a SymPy-generated plugin.

### 7.3 User-facing example

```python
import numpy as np, sympy as sp
import shoccs as sh
from shoccs import bc, mms

# ---- 1. Grid and geometry (host, fp64; C++ mesh under the hood: ray casting, fragments, R tables)
grid = sh.Grid(extents=(51, 51), bounds=((0.0, 2.0), (0.0, 2.0)))        # 2D: nz = 1, z-lines absent
ball = sh.Sphere(center=(16/17, 25/22), radius=np.sqrt(3)/10)
mesh = sh.Mesh(grid, shapes=[ball])
print(mesh.n_D, mesh.n_R)             # 2601, (16, 16, 0): R points per axis, generation order kept

# ---- 2. Manufactured solution as SymPy; compiled once into a point_eval functor (value, ddt, grad, lap)
x, y, t = sp.symbols("x y t")
S = 2 * sp.exp(-((x - 1)**2 + (y - 1)**2)) * sp.cos(0.5 * t)
sol = mms.Symbolic(S, coords=(x, y), time=t)   # sol(t), sol.ddt(t), sol.lap(t), sol.normal_derivative(t)

# ---- 3. Boundary conditions: Dirichlet on the 4 grid faces, Neumann on the object
bcs = sh.BCs(grid={"xmin": bc.Dirichlet(sol), "xmax": bc.Dirichlet(sol),
                   "ymin": bc.Dirichlet(sol), "ymax": bc.Dirichlet(sol)},
             objects={ball: bc.Neumann(sol.normal_derivative)})   # g = n . grad S at the R points

# ---- 4. Scheme and operators: E4 second-derivative family (E4_2: P=2, R=3, T=5, X=3), tables built once
scheme = sh.schemes.E4(order=2)                 # alpha defaults from stencil_gen/sweeps/known_values.json
lap = sh.Laplacian(mesh, scheme, bcs)           # taps[2][5], role bits, 4 face blocks, Drow CSR, Rrow CSR
# lap.to_scipy() -> scipy.sparse CSR over the packed D|Rx|Ry|Rz vector, for the stability toolkit

# ---- 5. Backend and fields: one packed buffer per field, fixed offsets
be = sh.backend("kokkos", dtype="f64")          # alternatives: sh.backend("metal", dtype="f32"|"df64")
space = sh.FieldSpace(mesh, be)                  # Npk = 2601 + 16 + 16 + 0 (padded to 2640)
k = 1.0 / 30.0

# ---- 6. The PDE: rhs/apply_bc are RECORDED into a schedule, never executed eagerly
def rhs(u, du, t):                              # u, du: space.Field; t: schedule parameter
    du[...] = k * lap(u)                        # B3 (2 dirs fused, eq, scale=k) + 2x B4 + B5 Drow + B5 Rrow
    src = sol.ddt(t) - k * sol.lap(t)           # symbolic; becomes ONE point_eval functor
    du.on(mesh.fluid | mesh.R.non_dirichlet) += src          # B10 fused with masked add
    du.on(bcs.dirichlet_set) = 0.0              # B8 over the flattened face list (204 pts) + Dirichlet R list (0 here)

def apply_bc(u, t):
    u.on(bcs.dirichlet_set) = sol(t)            # B10 only on 204 face points, not the whole field
    bcs.neumann_data[...] = sol.normal_derivative(t)         # B10 on the 16+16 object R points -> nu buffer

step = sh.RK4(space, rhs, apply_bc)             # records 1 schedule: 4 x (lincomb, bc, rhs, accumulate) + final
sim = sh.Simulation(step,
                    dt=sh.cfl.parabolic(0.5, mesh, k),       # 0.5 * h^2 / (4k), constant
                    t_end=0.3,
                    stats=sh.stats.linf_vs(sol, over=mesh.fluid | mesh.R.non_dirichlet),   # B9 maxloc, read back per step
                    io=sh.io.xdmf("out/heat", every_time=0.01, error_vs=sol))

u0 = space.field("u")
u0.on(mesh.fluid).set(sol(0.0)); u0.R.set(sol(0.0))          # initial condition on fluid D and all R points
sim.run(u0)                                      # Python loop: step.run(t, dt) -> C++ schedule replay, 1 readback/step
print(sim.stats[-1].linf, sim.stats[-1].argmax)  # R points are first-class: argmax may land in the Rx segment
```

### 7.4 IR and lowering

Two IRs, both plain data, no tracing.

IR-A, operator tables, produced at setup. The phase-1 producer is C++: a new `ccs::assemble::derivative_tables(dir, mesh, stencil, grid_bcs, obj_bcs)` factored out of `derivative.cpp:305-449` and `159-257`. The loop bodies stay (they are the hard, psi- and interp-dependent part, I06 to I09); only the sinks change: instead of `O_builder.add_inner_block(...)` (`derivative.cpp:431`) it records {start_ic, len, stride, left_kind, right_kind, closure rows}; instead of `B_builder.add_point(row, object_coordinate, v)` (`derivative.cpp:364, 410`) it records (row, off_R(dir) + object_coordinate, v); `OB_builder::add_cut_row`/`add_interp_row` (`derivative.cpp:72-135`) emit rows in the R(r) segment. A NumPy post-pass derives role bits from fragment tables, chooses the face kernel versus the CSR fallback per line (fallback iff len is under R_L + R_R or the line end is an object), lexsorts and merges duplicate (row, col) pairs, merges directions, and uploads. The legacy `derivative` object is kept so that phase-1 parity tests can compare table matvec against block-plus-csr matvec on random vectors.

IR-B, the schedule, produced when the user calls `sh.RK4(...)`/`Simulation`: a flat node list over arena buffer ids plus a parameter vector. Lowering rules: `du[...] = k * lap(u)` becomes B3 (eq, scale k), per-face B4 (plus_eq), B5 Drow, B5 Nrow if Neumann, B5 Rrow; the scale is pushed into every node's epilogue (I26 disappears) and zero-fill (I25) disappears because B3 with op=eq defines every packed point. `du.on(set) += expr(t)` becomes B10 with a SymPy expression simplified, CSE'd (`codegen.py:15`) and either matched to the built-in gaussian-sum functor or printed as a C++ device functor compiled into a plugin `.so` cached by hash. `grad(u)` becomes B3 with nout=3; the `eq_t` partial-coverage semantics are replaced by the total definition (a documented bitwise change). RK4 (`rk4.cpp:16-47`) becomes lincomb, apply_bc, rhs and accumulate nodes per stage, with u0/u1 swapped by id at the end of the step. Validation at record time checks dtype, length, CSR column bounds and aliasing (out must not alias an input of B3 or B5). Compile: Kokkos host stores the node list (about 0.7 us dispatch per node, the measured floor), Cuda builds a Kokkos Graph, Metal pre-creates pipeline states and encodes all nodes into one command buffer per `run()`. `Operator.to_scipy()` assembles the same tables into scipy CSR, with no second code path for coefficients.

### 7.5 Backends, including the Metal path

| Backend | Status | Mechanism | Caveats |
|---|---|---|---|
| kokkos_backend, fp64 | Primary | All ten primitives as Kokkos `parallel_for`/`parallel_reduce` over arena Views; B3 as `MDRangePolicy<Rank<3>>` with tiles (4, 8, nz), taps captured by value, three directional sums accumulated in registers; B5 one thread per row; B9 Kokkos reducers. Build with scikit-build-core plus nanobind linking the existing `shoccs-*` libraries | The dispatch floor is the same Kokkos host dispatch the C++ uses; a heat RHS is 6-8 nodes. Kokkos-in-Python lifecycle rules (single initialize, no fork after import, OpenMP threads versus the GIL) are a known nuisance |
| metal_backend | Phase 4, macOS only | metal-cpp vendored; MSL kernels for B1 to B10 compiled offline by `xcrun metal` into `shoccs.metallib`, templated by function constants on dtype in {float, df64}; shared-storage buffers; `schedule.run()` encodes all nodes into one command buffer and waits only on readback; K steps batched per command buffer | Recent Xcode ships the Metal compiler as a separate download (offline `.metallib` builds need it; runtime `newLibraryWithSource` does not; stated in the design, unverified against current Xcode docs). B10 is fp32-only on Metal; df64 schedules evaluate MMS terms on the CPU with a sync per stage. No fp64. Metal CI needs a self-hosted macOS runner |
| Kokkos::Cuda | Phase 7 | The same source with the execution space as a template parameter; a Kokkos Graph of the whole step; async B9 readback every K steps; plugin functors through nvcc | Validation requires Linux hardware the owner's Mac cannot provide |

### 7.6 Performance argument

| Regime | C++ bar | Thin-waist projection | Basis |
|---|---|---|---|
| A: heat.lua | RHS 30-40 us (about 40 nodes); step 0.3-1 ms (estimate) | RHS without source about 10-12 us (6-8 nodes); with the B10 source about 45-50 us on 4 threads; per step about 215-230 us | The source term, one exp plus cos plus sin plus about 30 flops per point, is about 120 us single-threaded at 2601 points; the C++ pays more for the same evaluation (two virtual `any_sol` calls per point, `heat.cpp:111-117`) outside its measured 32 us. So the claim holds at step level, not per RHS, and depends on what is counted. Separable-source hoisting (graft from D1) removes this term |
| B: 64^3 | RHS 1.65 ms (measured) | RHS about 0.4-0.6 ms (3x under the bar); 32^3 about 60-90 us against 157 | Fused B3 over a 2 MB cache-resident buffer, compute-bound at about 1-2 ns per point on 4 cores; the one-team-per-strided-line matvec at 13.7 ns per point is exactly what B3's plane tiling fixes |
| C: 256^3 plus sphere | RHS 30-110 ms; step 300-400 ms (estimate) | RHS without source about 5-6 ms; per step about 45 ms (4 RHS plus about 21 ms of lincomb traffic plus stats) | About 2 GB of lincomb traffic per step is the price of no fusion; a fused design moves about 35 ms worth. With an MMS source, B10 at about 25 ms per RHS dominates everything, a property of the test problem |
| Metal fp32/df64 | | fp32 at 256^3 on M4 Max about 1-1.5 ms per RHS, about 8 ms per step with K steps per command buffer; df64 about 2-3 ms per RHS, about 2x over CPU fp64 on Max, about 1x on base chips | Regime A on Metal loses unless 10-50 steps are batched with no readback; the design's honest expectation is that the CPU wins all shipped cases |

### 7.7 Idiom coverage

| Idiom | Mechanism in the thin waist | Risk |
|---|---|---|
| I01 | One packed arena buffer per field; `[ncomp, Npk]` for multi-component | low |
| I02 | C++ `object_geometry` unchanged through nanobind; SoA views | low |
| I03 | `mesh::lines(dir)` and `dirichlet_line` reused; NumPy post-pass derives role bits | low |
| I04 | Masks and lists built once in Python from exported tables | low |
| I05 | C++ stencil classes through nanobind; alpha from `known_values.json`; length check replaces silent truncation | low |
| I06 | Setup-only, evaluated by the existing C++ nbs inside the assembly loop | low |
| I07 | Kept entirely in C++ inside the table exporter; the O(|R|^2) scan at 256^3 is a known weakness | medium |
| I08 | `domain_discretization` loop body reused; sinks emit records | medium |
| I09 | `cut_discretization` loop body reused; Bf and Br merged into one Rrow CSR | medium |
| I10 | COO to compact CSR in NumPy (lexsort, `np.add.reduceat`) | low |
| I11 | Table export; no per-line allocations | low |
| I12 | Python front end replaces `from_lua`; Lua optional for the legacy binary | low |
| I13 | Same flat layout; B3 threads along the fast index; int64 variant when Npk exceeds 2^31 | low |
| I14 | Per-direction R order preserved inside the packed buffer; parity tests on random vectors catch permutation slips | low |
| I15 | Arena buffers replace spans and system-owned vectors | low |
| I16 | B3 (interior, role bits) plus B4 (wall blocks) plus B5 (object rows, short-fragment fallback) | medium |
| I17 | B3 with up to 3 directions fused | low |
| I18 | B4 `face_closure` | low |
| I19 | B5 rows of the Drow CSR | low |
| I20 | B5 compact CSR kernel | low |
| I21 | B disappears as a matrix: extra columns of the object closure rows | low |
| I22 | N as a small CSR keyed to `nu`; object Neumann data needs the X extra coefficients for object ends, a new numerical capability | medium |
| I23 | Bf_r rows in the Rrow CSR | low |
| I24 | Br_r merged into the same rows | low |
| I25 | Eliminated: B3 with op=eq is total | low |
| I26 | Scale folded into node epilogues | low |
| I27 | Schedule order is the accumulation order | low |
| I28 | B3 nout=3 | low |
| I29 | B2 `dot3` over the packed buffer | low |
| I30 | B10/B2 with the fluid mask; B7 scatter with unique indices for R lists | low |
| I31 | B8 `fill_at` with the reused plane descriptors (`selection_desc.hpp:3-15`) | low |
| I32 | Schedule record/compile/run; swap by id; Kokkos Graph on Cuda; one command buffer on Metal | low |
| I33 | A one-operator schedule | low |
| I34 | B10 with a device gaussian-sum functor or a SymPy-printed plugin, restricted to index sets | low |
| I35 | Dropped by design | high |
| I36 | B10 inside the recorded RHS with t as a parameter | low |
| I37 | B1 over the whole packed buffer | low |
| I38 | Not needed: swap by id | low |
| I39 | B10 on Dirichlet face sets and R lists only | low |
| I40 | Lists built once | low |
| I41 | B9 maxloc of |u - sol(t)| in one pass | low |
| I42 | B2 `abs_diff` gated on dump-due; existing C++ `field_io` through nanobind | low |
| I43 | Python while loop | low |
| I44 | `Operator.to_scipy()` from the same tables | low |
| I45 | NumPy oracles plus the retained legacy classes | low |
| I46 | B2 fixed functor menu plus plugins; the one place the lack of a DSL shows | medium |
| I47 | `[ncomp, Npk]` state, batched B3/B4/B5, flux and primitive functors, normal projection, B9 dt | medium |
| I48 | Superseded by `to_scipy()` | low |
| I49 | `printer.py` gains a device-functor target | low |
| I50 | Unchanged host scipy consuming `to_scipy()` | low |
| I51 | In-process; BO drives `sh.Simulation` directly | low |

### 7.8 Awkward or unsupported

- I35 unsupported in the hot loop; pointwise physics must be SymPy expressions or precomputed tables; a C++ compiler is required at setup for plugin functors.
- No general fusion: lincomb cannot fuse with the RHS, flux cannot fuse into the stencil read; an estimated 1.3-2x is left on the table in regimes B and C.
- The functor menu is a slippery slope toward an accidental DSL. Guardrail: every functor is a named, tested template; shape-dependent control flow goes through tables.
- `eq_t` semantics change at uncovered points.
- Object Neumann data is not assembled in today's C++; `bc.Neumann` on objects needs the X extra coefficients for object ends.
- Short fragments with n_interior at most 0 must be detected and routed to CSR rows or rejected.
- Moving geometry means re-running the C++ assembly per step.
- df64 on Metal covers B1/B3/B4/B5/B9 only.
- Kokkos inside a Python process: single initialize, finalize ordering, threads versus the GIL, no fork.
- Three languages; Metal CI needs a self-hosted runner; CUDA needs Linux hardware.
- Setup at 256^3 bounded by the C++ `interp_line` scans.
- t-laplacian fails on R-point values, so there is no trusted oracle for the Rrow path.

### 7.9 Reuse of existing code

Reused unchanged behind nanobind: `src/mesh/*`, `src/stencils/*`, `src/operators/boundaries.hpp`, `src/indexing.hpp`, `index_extents.hpp`, `selection_desc.hpp` descriptors, `src/io/field_io` and the XDMF writer, the gauss2d/gauss3d formulas as a device functor, `benchmarks/`. Refactored and reused: the `derivative.cpp` assembly loops become `assemble::derivative_tables` with changed sinks; the `csr.hpp` builder pattern as the COO collector. Retained as oracle/legacy: `block.hpp`, `inner_block`, `dense`, `circulant`, the csr matvec, the operator apply and graph paths, `src/systems/*`, `src/temporal/*`, `src/simulation/*`, `src/lib`, `src/app` (the Lua binary keeps building until phase-3 parity is signed off). Python assets: `printer.py`/`codegen.py` (new device-functor target), `known_values.json`, the analysis cascade consuming `to_scipy()`, `cpp_bridge.py` retired. Dropped: sol2/Lua in the Python path, lapackpp, the `simulation_builder` stub.

### 7.10 Euler extension

State `U = space.field("U", ncomp=4|5)` as one `[ncomp, Npk]` buffer; every primitive takes a batch count so tables are shared across components. RHS per direction d: F = B2 `euler_flux_dir(U, d, gamma)` into a reused flux buffer; dU via `Derivative_d(F)` (batched B3, B4, B5 Drow, B5 Rrow with scale -1). Memory at 256^3 fp64: U, F (one, reused), dU, rk, u1 at 670 MB each. 3D RHS traffic about 8 GB, about 80 ms at 96 GB/s. Slip at R points as B2 `project_normal`; characteristic faces as B2 over face descriptors with the exact state from B10; dt as B9 `max_abs_plus`; filter as another table operator; conservation as B9 weighted sums. What changes in the primitives: a `batch` field on the nodes, Euler functors in the B2 menu, `max_abs_plus` in B9. No new primitive is required, which this design presents as the strongest evidence that the ten-primitive vocabulary is closed for this PDE family.

### 7.11 Migration phases

| Phase | Duration | Content | Gate |
|---|---|---|---|
| 0 | 1 week | Native macOS build; per-region timers; heat.lua, scalar_wave.lua, bench_rhs 32^3/64^3; counts from geometry.csv | A measured bar |
| 1 | 2-3 weeks | `assemble::derivative_tables`; nanobind `_shoccs`; Python skeleton; `Operator.to_scipy()` | Table-assembled scipy matvec equals legacy block-plus-csr matvec on 100 random vectors to 1e-14 on fluid D and all R rows |
| 2 | 4-6 weeks | Arena, buffers, B1 to B10 on Kokkos host, schedule record/validate/run, GIL release; heat and scalar_wave RHS as schedules | RHS matches the C++ graph path to 1e-13; RHS at most 1.0x C++ at 51x51 and at most 0.5x at 64^3 |
| 3 | 2-3 weeks | RK4/Euler schedules, boundary-only apply_bc, B9 stats, Simulation loop, XDMF, SymPy MMS functors | Linf histories match the Lua binary to 1e-12; step at most 1.0x; the Lua binary becomes legacy |
| 4 | 6-8 weeks | Metal backend fp32, df64 for B1/B3/B4/B5/B9, K-step batching | Heat 256^3 plus sphere runs; precision study; df64 RHS timed against the CPU |
| 5 | 3-4 weeks, optional | Vectorised NumPy/Numba assembly with searchsorted lookup | Only if 256^3 setup time or BO geometry iteration becomes the bottleneck |
| 6 | 6-10 weeks | Euler | Isentropic vortex in 2D then 3D |
| 7 | 2-3 weeks | CUDA as a Kokkos build-matrix entry | |

### 7.12 Effort

CPU-only parity (phases 0-3) about 10-13 weeks for one developer who knows the codebase; Metal 6-8 weeks; Python assembly 3-4 weeks optional; Euler 6-10 weeks; CUDA 2-3 weeks plus hardware. About 7-9 developer-months to a complete system with Metal and Euler; phases 4-7 are parallelisable once the primitive API freezes. Ongoing cost: three languages, scikit-build-core, and a curated functor menu.

### 7.13 Risks

- The bar is unmeasured on the target machine.
- Kokkos-in-Python lifecycle crashes at interpreter exit or in pytest (pytest-xdist is already used by the Python CI job).
- Semantic drift (total outputs, summed duplicates, boundary-only MMS, object Neumann) without the phase-1 and phase-2 gates.
- The B2 functor menu becomes an unplanned DSL.
- Metal fp32 may be unstable for small-psi closures; df64 is hand-written numerics with a bounded advantage.
- Setup scalability at 256^3.
- Maintenance surface across C++, MSL, nanobind, Python and printer targets; Metal and CUDA cannot both be tested on one machine.
- Geometry assumptions (`derivative.cpp:46-47, 89-90`; `object_geometry.cpp`) inherited; the exporter must fail loudly.
- The thin waist forbids whole-step fusion; if lincomb plus RHS traffic at 256^3 becomes the bottleneck, the only recourse is a hand-added fused `rk_stage` primitive.

### 7.14 Killer prototype experiment

About 1.5-2 weeks, C++ only, no Python. (1) Implement `assemble::derivative_tables` by copying the two assembly loops with new sinks, plus a C++ post-pass (sort and merge rows, role bits, face-versus-CSR decision). (2) Implement a minimal `kokkos_backend` with B1, B3 (fused dirs, role bits), B4, B5 and B8, and a plain schedule runner; feed the source term from the existing vectors. (3) Hand-build the heat RHS schedule for heat.lua's geometry and for a 64^3 E2 no-shape case; run it against `heat::submit_rhs_graph` (`heat.cpp:290`) on identical inputs in one process. Pass (all three, on the native build, 4 and 10 threads): (a) max |du_schedule - du_cpp| / max |du_cpp| at most 1e-13 over fluid D points and all non-Dirichlet R points, every remaining difference confined to solid or Dirichlet points; (b) RHS wall time at most 1.0x the C++ graph RHS at 51x51 (expected 10-12 us against about 30 us excluding source) and at most 0.5x at 64^3 (expected at most 0.8 ms against about 1.6 ms); (c) table memory for a 64^3 sphere case below the legacy footprint. Fail: the ten-primitive vocabulary or the compact tables do not pay, the alternative is a tracing DSL for CPU, and the Metal plan is moot. A second, independent gate for the Apple GPU: a standalone metal-cpp program running B3 (7-point, 3 directions, role bits) at 256^3 in fp32 and df64, timed against the Kokkos B3 fp64 number; proceed with Metal only if df64 is at most 0.5x the CPU time.

## 8. Judging

### 8.1 Rubric

Three judges scored each design on seven criteria from 1 (poor) to 5 (strong). The weights sum to 15, so the maximum weighted total is 75.

| Criterion | Weight | What 5 means | What 1 means |
|---|---|---|---|
| Performance parity | 3 | Credible structural argument that the fp64 CPU path matches or beats the C++ in all three regimes (dispatch-bound 2.6k-5k points, cache-resident 32^3-64^3, bandwidth-bound 256^3), with a measurable gate | Parity depends on unverified third-party heuristics or has no clear gate |
| Idiom coverage | 3 | All 51 catalog idioms have a named mechanism; the six irregularities are represented without special cases | Several idioms left as workarounds or hand-waved |
| Apple GPU reach | 2 | A concrete fp32 and df64 Metal path with K-step batching, honest about where it wins | No Metal path or one that depends on an unmaintained framework |
| Migration cost | 2 | Incremental, with the Lua binary working throughout; months, not a year | A long rewrite before anything runs |
| Maintainability | 2 | No compiler owned; small, closed kernel set; clear oracles | A custom compiler with several emitters and 15-25% FTE upkeep |
| Reuse | 1 | Existing C++ geometry, stencils, assembly and Kokkos kept; stencil_gen reused | Everything reimplemented |
| Euler extensibility | 2 | Multi-component state, flux fusion, R-point wall physics and adaptive dt designed in | Euler needs new kernel kinds or materialises all fluxes |

The judges were three personas with different lenses. Judge 1 is a performance engineer (does it match C++/Kokkos at 5k points and at 16M points, on fp64 CPU and on an Apple GPU without fp64). Judge 2 is a numerical-methods researcher (does it keep cut-cell fidelity; can a researcher add E6/E8/RBF schemes or Euler without touching runtime C++; does it feed the SymPy derivation and stability cascade with the exact operator the solver runs). Judge 3 is the maintainer of a small research code (one or two developers, Spack devcontainer plus GHCR CI, Kokkos 5.1.1 host-only), who checked facts in the worktree before scoring: no native macOS build existed at the time (one has since 2026-10-13, Part IV E); Apple clang 21 with no Homebrew libomp, so "OpenMP on the Mac" was an unbudgeted toolchain step (since taken: Spack llvm-openmp); stencil_gen is about 49.5k lines built on a `C99CodePrinter` subclass (`printer.py:8-11`) and uses no lambdify today; t-laplacian fails only on the Bf/Br R-row path (`docs/CLEANUP_PLAN.md` §0a); `frameworks/jax.md` holds the only local timings in the survey.

### 8.2 Scores per judge

Judge 1, performance engineer:

| Design | Perf | Idiom | Apple | Migr | Maint | Reuse | Euler | Weighted |
|---|---|---|---|---|---|---|---|---|
| D5 thin waist | 4 | 4 | 3 | 5 | 3 | 5 | 3 | 57 |
| D1 kerf-forge | 5 | 4 | 3 | 2 | 2 | 4 | 4 | 53 |
| D4 Kerf-lang | 5 | 5 | 3 | 1 | 1 | 4 | 4 | 52 |
| D3 Pencil | 4 | 4 | 3 | 3 | 3 | 4 | 3 | 52 |
| D2 ccsx | 3 | 4 | 3 | 3 | 3 | 3 | 4 | 50 |

Judge 1's summary: every design compares against the same unmeasured, about 2x noisy baseline, so every 5-30x claim is a projection; the sound claims are structural (about 280 launches, 26 fences and about 14 host MMS sweeps per step removed), the weak ones are bandwidth claims assuming the 96 GB/s single-thread figure is a DRAM ceiling at 256^3. The thin waist is the only design that makes the bar cheap and falsifiable in 1.5-2 weeks against the real Kokkos RHS, at the price of 1.3-2x fusion left on the table. kerf-forge has the best structural story but "as fast as hand-written by construction" is an assertion. Kerf-lang has the most complete mechanics and the highest cost, with a scalar merge-join patch inside a vector body as its central risk. Pencil's closed IR is the most disciplined; its mask-decode cost and the EXC second touch of cold lines are unmeasured. ccsx has the cleverest correctness idea and the weakest performance control (XLA heuristics, a 67 MB rowmap read, a gate of only 1.0x).

Judge 2, numerical-methods researcher:

| Design | Perf | Idiom | Apple | Migr | Maint | Reuse | Euler | Weighted |
|---|---|---|---|---|---|---|---|---|
| D5 thin waist | 4 | 4 | 3 | 5 | 3 | 5 | 3 | 57 |
| D1 kerf-forge | 5 | 4 | 4 | 2 | 2 | 5 | 4 | 56 |
| D3 Pencil | 4 | 4 | 4 | 3 | 4 | 4 | 3 | 56 |
| D2 ccsx | 3 | 5 | 3 | 3 | 4 | 3 | 4 | 55 |
| D4 Kerf-lang | 5 | 4 | 4 | 1 | 2 | 4 | 4 | 53 |

Judge 2's summary: the thin waist has the highest fidelity on day one because it keeps the psi closures, interp rows, E4_1 guards and R ordering in the proven C++, and its `to_scipy()` is built from the tables the kernels consume; it also inherits the C++ bugs and makes every new physics a hand-written functor. kerf-forge fits the SymPy pipeline best (schemes wrap `CutCellResult`, the MMS is derived by `sympy.diff`, separable terms are hoisted), but its Neumann-object Jacobi fallback enforces the BC inexactly and alters the semi-discrete operator, and alpha is compile-time by default. Pencil's `A = sum_d (M_d T_d + S_d) + E` is the cleanest statement and its monomial oracle is the only proposal that escapes the broken t-laplacian reference; the three-buffer RK plan has a read-after-write hazard. ccsx is the strongest fidelity architecture (exact matrix first, split verified against it, tables as jit arguments, `jax.jacfwd` for the linearised Euler operator) and the most honest about object Neumann. Kerf-lang's exact constraint elimination is better numerics than D1's projection and its fp32 cancellation estimate is quantified, but hand-coding the MMS discards the SymPy derivation and the researcher would maintain a language rather than a solver.

Judge 3, maintainer (half points allowed):

| Design | Perf | Idiom | Apple | Migr | Maint | Reuse | Euler | Weighted |
|---|---|---|---|---|---|---|---|---|
| D5 thin waist | 4 | 4 | 3 | 4 | 4 | 5 | 3 | 57 |
| D3 Pencil | 4 | 4 | 4 | 3 | 4 | 3 | 3 | 55 |
| D1 kerf-forge | 4.5 | 4.5 | 3 | 2 | 2.5 | 3.5 | 4.5 | 54.5 |
| D2 ccsx | 3.5 | 4.5 | 3 | 2.5 | 3.5 | 2 | 4.5 | 53 |
| D4 Kerf-lang | 4 | 4.5 | 3.5 | 1.5 | 2 | 3.5 | 4.5 | 52 |

Judge 3's summary: the thin waist clears the bar with the least speculative argument because it keeps the mechanism the C++ uses (Kokkos host dispatch at the measured floor) and removes what is demonstrably wasted; its own regime-A estimate exposes that the transcendental source dominates the step without D1's hoisting. Pencil offers the smallest Metal surface, a no-compiler fallback through precompiled instantiations, and the best answer to t-laplacian, but assumes libomp and ports geometry and assembly to NumPy from day one. kerf-forge has the highest ceiling and the best-argued one, owns four emitters, and needs a compiler and Kokkos headers at define time. ccsx is the only design backed by any local measurement and has the best onboarding story (no C++ at run time), but its performance is hostage to a dependency's heuristics and it drops Kokkos. Kerf-lang's merge-join sits inside the vector loop, its wall closures bloat E to about 1M rows at 256^3 where others keep slabs, and most of its language surface buys little over a SymPy front end.

### 8.3 Combined ranking

The combined ranking supplied by the workflow summed weighted totals keyed on the design names as each judge wrote them. Because the judges spelled every name differently, the raw sum pairs only two of the fifteen rows and is reproduced here only for the record:

```text
Pencil (BLX = banded lines + exception rows, linear-algebra-first)              107
Thin waist (nanobind plus compact tables plus a ten-primitive schedule)          57
pyshoccs thin waist (nanobind + 10 primitives)                                   57
pyshoccs thin waist (nanobind + tables + schedule, no compiler)                  57
kerf (cut-cell kernel forge, SymPy front end -> C++/Kokkos + MSL codegen)        56
Pencil (BLX banded lines + exception rows, LA-first)                             56
ccsx (array lowering: exact assembly + structural split, JAX f64 / MLX Metal)    55
kerf (cut-cell kernel forge: SymPy front end -> kernel IR -> Kokkos C++ / MSL)   54.5
Kerf-A (kerf cut-cell kernel forge, SymPy-to-C++/MSL code generator)             53
Kerf (kernel DSL: Python-AST -> Point IR -> C++/MSL/CUDA, S+E runtime)           53
ccsx (array-framework lowering: exact assembly, structural split, K-step jax.jit) 53
Kerf-B (typed cut-mesh kernel DSL with S+E sweep and a custom compiler)          52
Kerf (kernel DSL: Python-subset kernels -> Point IR -> Jinja2 C++/MSL/CUDA)      52
ccsx (JAX/XLA fp64 plus MLX array-lowering DSL)                                  50
```

Normalising the names gives the ranking the judges actually produced. Each design's three totals are summed (maximum 225).

| Rank | Design | Judge 1 | Judge 2 | Judge 3 | Sum | Rank by each judge |
|---|---|---|---|---|---|---|
| 1 | D5 thin waist | 57 | 57 | 57 | 171 | 1, 1, 1 |
| 2 | D1 kerf-forge | 53 | 56 | 54.5 | 163.5 | 2, 2, 3 |
| 3 | D3 Pencil | 52 | 56 | 55 | 163 | 4, 3, 2 |
| 4 | D2 ccsx | 50 | 55 | 53 | 158 | 5, 4, 4 |
| 5 | D4 Kerf-lang | 52 | 53 | 52 | 157 | 3, 5, 5 |

Three observations about this table matter more than the order. First, the thin waist is ranked first by all three judges, and for the same reason each time: it is the only design whose central claim can be tested in about two weeks of C++ against the live Kokkos RHS, and it is the only one that keeps the proven geometry, stencils and assembly loops. Second, the spread between second and fifth is 6.5 points out of 225, which is inside the noise of a 1-to-5 rubric; the ordering of D1, D3, D2 and D4 should not be read as a finding. Third, the thin waist wins on migration cost, maintainability and reuse and loses on performance ceiling and Euler to the compiler designs, which is exactly the trade the hybrid in Section 9 tries to settle by adding one fused primitive instead of a compiler.

### 8.4 Grafts

The judges proposed ideas worth moving from one design into others. Duplicates across judges are merged.

| # | Idea | From | Into | Why |
|---|---|---|---|---|
| G1 | Classify each row against the exact assembled matrix: a row is structured iff it equals the stencil row in support and value to 4 ulp; everything else, including every R row, goes into the table. The exact matrix is also the scipy export and the reference oracle | D2 ccsx | All table builders (D1, D3, D4, D5) | Correct by construction; handles short fragments and multi-body overlap without special rules; one object feeds `to_scipy`, GKS and pseudospectra |
| G2 | Reference-free oracle: apply the assembled operator to monomials up to the scheme's order and require exactness at every fluid and R row | D3 Pencil | All five | t-laplacian already fails on the Bf/Br path, so parity with the C++ proves nothing on R rows; this is the only proposed test that can adjudicate which side is wrong |
| G3 | Phase-1 C++ table exporter: keep the `derivative.cpp` assembly loop bodies, change only the sinks, keep the legacy block-plus-CSR operator as a permanent parity oracle; defer the NumPy port of ray casting and interpolation (I07, the highest-risk port in every Python-assembly design) to after the perf gate | D5 thin waist | D1, D2, D3, D4 | Removes the largest early-phase risk and keeps bit-exact R ordering |
| G4 | PURE/WALL/MIXED tile classes so that the branchy closure lookup is paid only in the 2-10% of tiles touching a body | D1 kerf-forge | D5 (B3), D3 (STRUCT) | Keeps the pure-tile loop free of table loads |
| G5 | Fuse the RK stage epilogue (acc += b dt du; u_next = u0 + a dt du; Dirichlet value write; stats partials) into the stencil sweep and the table-row kernel, as one fused `rk_stage` primitive added immediately | D1, D3, D4 | D5 | Recovers most of the 1.3-2x the thin waist concedes; keeps the schedule as data |
| G6 | SymPy-derived MMS and BC data (`sympy.diff`), with time-separable terms split into spatial factors precomputed once in fp64 and scalars c_g(t) per stage; boundary-only evaluation of BC values; a hoist-versus-recompute cost model (hoist at 5k points, recompute at 256^3 where aux loads cost bandwidth) | D1 (hoisting), D4 (cost model) | D5 (B10), D3, D2 | Removes transcendentals from the hot loop (the thin waist's own estimate has the source at about 120 us single-threaded against a 20 us step) and makes df64 Metal trivially correct for the shipped gaussian times cos(wt) MMS |
| G7 | Exact elimination of object Neumann/Robin constraints into the exception rows at setup (u_R = A u_D + b g, then E := E + E_R A) | D4 Kerf-lang | D1, D3, D5 | Replaces D1's epilogue Jacobi sweeps, which enforce the BC inexactly and alter the spectrum |
| G8 | Make psi, alpha and the wall-closure coefficients runtime table arguments by default, never compile-time literals | D2 ccsx, D5 | D1 (alpha is compile-time by default), D4 (constexpr taps), D3 (compile per system and scheme) | BO loops and alpha/epsilon sweeps must never recompile |
| G9 | Chunked driver with a device stats buffer f[K, nstat], a first-invalid-step index and rollback to the chunk start; fixed table shape buckets | D2 ccsx | D1, D3, D4, D5 | Makes K-step batching compatible with NaN detection and dump schedules |
| G10 | Wall closures as plane-constant slabs or a `face_closure` primitive, so the exception table holds only psi-dependent object rows and R rows | D3 Pencil, D5 | D4 (which puts about 1M unique wall rows into E at 256^3) | 40-70 MB and a divergent merge-join avoided |
| G11 | `K_rows corr[]` plus `tile_ptr` for the GPU path: precompute the row correction, then a tile-local lookup avoids a D-sized map | D4 Kerf-lang | D1, D3 Metal backends; D2 (67 MB int32 rowmap) | Removes a full-size index read per stage on the GPU |
| G12 | A persistent spin-barrier thread pool, since Apple clang ships no OpenMP (confirmed: no libomp on this machine) | D4 Kerf-lang | D3 (assumes libomp), D1 (`#pragma omp simd`, Kokkos OpenMP; alternatively Kokkos Threads) | Small-size parallelism on macOS is otherwise an unbudgeted toolchain step |
| G13 | Batch the components of the packed [ncomp, Npk] state so one index/coefficient row serves all Euler components; flux-prologue ring buffer so fluxes are never materialised on CPU | D3, D4 | D5, D1 (Euler) | Amortises table traffic 4-5x |
| G14 | A NumPy reference interpreter of the same IR or schedule | D1, D4 | D5 (nearly free: a schedule is a node list), D3 | An in-process oracle and debugger that needs no Kokkos |
| G15 | `tools/dump_operator.cpp`: a 300-line dev tool that writes block meta/coeffs, every CSR, the R records and one RHS result as .npz from a Lua config | D2 ccsx | D3, D1, D5 | Decouples all Python work from a nanobind build and lets the first gate run in the existing C++ CI job |
| G16 | A launch or fusion-count regression test (assert the node count per recorded schedule, or the kernel count per generated step) in CI | D2 ccsx (HLO inspection) | D5, D3, D1, D4 | The dispatch-bound 2D regime cannot silently regress |
| G17 | Linearised operator export for Euler: `jax.jacfwd` in D2, or the explicit J = -sum_d Dx_d Diag(dF_d/dU) assembled in scipy from the tables and a SymPy flux Jacobian | D2, D3 | D5, D1, D4 (which proposes a dual-number dtype, far more work) | Feeds the GKS/eigen cascade for Euler about a base state |
| G18 | Precompiled kernel instantiations for W in {3,5,7,9,11} with table-driven epilogues as a complete no-toolchain fallback | D3 Pencil | D1, D4, D5 | E6/E8 experiments and CI run without clang or Kokkos headers at define time |
| G19 | Coefficient-level precision gating: split fp64 table coefficients into (hi, lo) df64 pairs on the host and estimate cancellation loss per closure row from max|coef| x h | D4 Kerf-lang | D1, D2, D3, D5 | A gate based on the coefficients rather than only on end-to-end Linf |
| G20 | Explicit kill criteria and a hand-instantiated prototype with no front end, gated on the native macOS C++ step time | D4, D5 | D2, D3 prototype plans | Keeps the first experiment small and decisive |

### 8.5 Blind spots shared by all five designs

Grouped from the three judges' lists.

Measurement and the bar.

- The C++ bar was unmeasured on the target machine when the five designs were written, and none says what happens if the fixed-up C++ step for heat.lua is already near the discretization cost, which would collapse several headline margins to about 1x. It was first measured on 2026-10-12 inside the devcontainer (`05-measurements.md` Sections A to C: Docker Desktop on the owner's Mac, 8 vCPU, Kokkos OpenMP, Release), when no native macOS build existed because the C++ dependency stack lived only in the container's Spack view; since 2026-10-13 a native build exists (merged upstream as "Support a native macOS build", `docs/reference/macos-host-build.md`: Apple clang 21, Kokkos 5.1.1 OpenMP via llvm-openmp, Release) and Section E measures the same cases on it, on the same M4 as the Python runs. The native shipped RK4 step is 0.856 ms for heat.lua at 51x51 at 1 thread (noisy: minimum 0.648, p90 1.35 ms; 0.266 ms with four spin-waiting threads, 2.26 ms with four threads at libomp defaults), 0.278 ms for scalar_wave.lua at 71x71 (1 thread; 0.198 ms at 4 threads spin), 2.17 ms at 201x201, 8.18 ms at 401x401 and 17.49 ms at 65^3 with a sphere (4 threads spin; 19.96 ms at 4 threads default). The native RHS graph submit alone is 21.0 us at 51x51 (1 thread; 54.1 us at 4 threads spin, 309 us at 4 threads default), 33.3 us at 71x71, 217 / 228 us at 201x201 (1 thread / 4 threads spin), 827 / 575 us at 401x401 (1 thread / 4 threads default) and 2065 / 961 us at 65^3 (1 thread / 4 threads default; 1810 us spinning); scalar_wave at 71x71 is 25.2 us (1 thread). The container figures were 0.90 ms, 0.43 ms, 4.6 ms, 13.7 ms and 37.9 ms for the steps and 22.7 (87.8 at 8 threads), 34.5, 198 / 152, 772 / 422 and 1867 / 798 us for the submits, scalar_wave 30.7 us; the one-thread submit is within 10% of the container at every size. The launch floor is 0.28 us per tiny kernel at 1 thread natively (0.55 us in the container). Four threads with libomp defaults are slower than one at every case below 201x201 because idle libomp threads sleep between parallel regions; spin-waiting removes that cost for small cases and slows the large kernels, so the bar per case is the better of the two settings. The toolchain decision the earlier draft asked for is made: Spack llvm-openmp with Apple clang 21. The Python side was measured natively on 2026-10-12 (Section D, synthetic SHOCCS-shaped step): the best fused fp64 CPU paths from Python cost 0.085 to 0.15 ms at 51x51, 1.0x to 1.8x the four native one-thread C++ RHS submits (84 us) and 0.4x to 0.7x of the four-thread-spin submits (216 us) (container: 0.9x to 1.6x of 91 us), and 4.2 to 4.5 ms at 65^3, 1.1x to 1.2x the native four-thread submits (3.8 ms) and about half the one-thread submits (8.3 ms) (container: within 1.4x of the eight-thread submits, 3.2 ms). None of the five designs anticipated that an array framework on the CPU, with no compiler of our own, might already sit at the discretization bar; the hybrid's phase 1 therefore carries an array arm (Section 9.1).
- The honest comparison is against a cheaply fixed C++, frozen as exactly four changes: (1) hoist the object-BC gather lists out of the per-call path; (2) compact B and N tables with no full-height row pointers; (3) manufactured-solution evaluation once per step and only where needed, with time factors hoisted (fill_source and update_boundary); (4) packed [D | Rx | Ry | Rz] buffers. That is roughly two weeks of work in the existing code, and it would erase most of the claimed 5-20x per-step margins. The native decomposition (`05-measurements.md` Section E; the container one is in Section C) shows how much of the shipped step is not discretization at all: at 51x51 on 1 thread, host fill_source is 59.8 us per stage, 28 to 37% of the 0.65 to 0.86 ms step (container: 132 us, 59% of 0.90 ms); update_boundary is 24.4 us per stage (11 to 15%; container 34 us, 15%); the four RHS graph submits, 21.0 us each, are 10 to 13% (container 10%); stats 26.9 us (3 to 4%; container 32 us, 4%); the stage work sums to 448 us and the remainder of 0.2 to 0.4 ms is noisy (container: 786 us of 0.90 ms, remainder 0.11 ms). At 65^3 fill_source is 49% of the 19.96 ms four-thread step (4 x 2429 us; container: 72% at 8 threads). Every regime-A step comparison in Sections 3 to 7 was therefore made against a bar inflated by host manufactured-solution evaluation. The fixed-up C++ step, modelled from change (3) alone with the RHS submit and stats kept and fill_source plus update_boundary replaced by a residual of 5 us per stage at 51x51 and 0.3 ms per stage at 65^3, is estimated natively at 131 us at 51x51 at 1 thread (4 x (21.0 + 5) + 26.9 us; at most about 0.33 to 0.53 ms if the noisy 0.2 to 0.4 ms one-thread remainder stays), 6.0 ms at 65^3 on 4 threads with libomp defaults (4 x (0.961 + 0.3) + 0.955 ms) and 12.4 ms at 65^3 on 1 thread (4 x (2.065 + 0.3) + 2.96 ms). At 4 threads spin the stage micro-benchmarks overstate the in-step cost (Section E: 570 us of stage work against a 0.266 ms step), so there is no decomposed estimate there; the fixed-up 4-thread-spin heat.lua step lies between the one-thread estimate of 131 us and the measured 0.266 ms. The container estimates, built the same way, were about 0.15 ms (at most about 0.26 ms with the remainder) and about 5.5 ms on 8 threads (4 x (23 + 5) + 32 us and 4 x (0.8 + 0.3) + 1.1 ms). Every "no regression" claim is a claim against a strawman until that fixed-up C++ is measured, which phase 1(b) does on the native build.
- The 96 GB/s single-thread streaming rate used as the 256^3 DRAM ceiling by D1, D2, D3 and D4 is a small-array axpy and is probably cache-assisted (`00-apple-gpu-facts.md` marks it as needing a recheck). Per-core bandwidth, SLC effects and what 8-10 threads reach on 134 MB buffers are unmeasured, so the "5 doubles per point per stage" claims have no verified roofline. The 1M-element axpby is 289 us at 1 thread natively (108 GiB/s; `05-measurements.md` Section E) and was 463 us at 1 thread (0.44 ns per element, 67 GiB/s) and 385 us at 8 threads in the container (Section B); the 1M case moves about 25 MB and may itself be cache-assisted, so 134 MB buffers stay unmeasured until the phase-1 STREAM triad.
- No design measures the cost of mixed-tile divergence, merge-join patching and gather-dot rows at realistic psi distributions. The 2-10% mixed-tile fraction at 256^3 with a sphere is asserted, not computed, and could be higher on thin geometries.
- Small-size parallelism is under-specified: whether 5k points should run on one core or several, and thread wake-up cost on macOS without OpenMP. Only D4 proposes its own pool. The wake-up cost is now measured with libomp (`05-measurements.md` Section E): the 51x51 RHS submit is 309 us on four threads at libomp defaults against 21.0 us on one, 54.1 us with spin-waiting.
- Time-separable source hoisting trades flops for bandwidth; at 256^3 each hoisted aux field adds a 134 MB read per stage. Non-separable fp64 transcendental throughput (Apple libm versus SLEEF versus vectorised) is not benchmarked.
- Python-side overhead at 5k points is asserted away by "one call per K steps", but dump schedules, adaptive dt and per-step stats force chunk boundaries; nobody quantifies the resulting K for realistic runs.

Correctness and numerics.

- The correctness reference for R points is missing. t-laplacian fails on exactly the Bf/Br path, yet every design uses the C++ operator as its oracle. Only D3 offers a reference-free check; none proposes fixing t-laplacian first, which is the cheapest way to obtain an oracle for exactly the rows the cut-cell machinery is about.
- No design specifies a parity protocol: which rows are compared, at what tolerance per quantity (operator table, single RHS, Linf history after 2650 steps), and how it survives the untrusted R rows.
- Setup-time fp64 accuracy at small psi is not addressed. Lambdified, CSE'd closures and the hand-written C++ forms evaluate rational functions with heavy cancellation near the 1e-12 and 1e-4 clamps; parity targets of 1e-13 or 4 ulp on psi-dependent rows will fail for reasons unrelated to bugs. Those rows need a psi-aware tolerance or an exact-rational (sympy/mpmath) reference.
- None certifies stability at setup: no design runs a rightmost-eigenvalue check on the assembled cut-cell operator per (geometry, scheme, alpha) and refuses to run or sets dt from it, even though small-psi closures make the stable dt geometry-dependent and every design keeps a geometry-blind dt formula.
- Floating R points as evolved ODE unknowns: no design examines whether evolving u_R through the Bf/Br rows is stable or accurate in its own right; the t-laplacian failure is treated as a porting nuisance rather than a possible formulation issue.
- Object Neumann is treated in three incompatible ways (projection with Jacobi fallback, directional data, exact elimination), all new numerics never assembled by the C++. None proposes deriving the object-Neumann closures in stencil_gen and running them through the GKS cascade before writing kernel code. Three front-end examples ship this BC; they should use Dirichlet or floating objects, which exist today.
- Explicit RK stability margins with merged Br coupling were not analysed. Order-of-accuracy verification (MMS refinement including worst-case small psi) is not a CI gate in any migration plan.
- Conservation for Euler on cut cells: every design writes Euler as a collocated -D(F(U)) with ordinary closures and only monitors conservation. None uses the conservation-by-construction closures the repo recently derived (polyE2_1 conservation, commit 4895b5a) or builds SBP or split-form closures. Shock capturing, limiters and nonlinear filters are out of scope everywhere; only D3 says so.
- Composing schemes (E4_1 derivative rows with interpolation borrowed from E4_2 or polyE2, because E4_1/E4u/E6u/E8u lack `interp_info`) is mentioned only as a limitation; nobody plans the stencil_gen derivation of matching cross-direction interpolants for E6/E8.
- Stiffness from small psi (implicit or IMEX treatment) and variable or conservative coefficients (div(k(x) grad u)) are out of scope; only D3 notes that variable coefficients break the constant-tap assumption.

Apple GPU.

- fp32 and df64 viability of small-psi high-order closures is untested everywhere; "df64 is bandwidth-bound at 4-11 flop/B" is an estimate with no Metal measurement; df64 division, sqrt and exp costs, flush-to-zero and the fp32 exponent range are hand-waved; the command-buffer cost rests on one forum report. Only D2's 2-day NumPy fp32 study on dumped operators precedes any MSL, and it should in every plan. No design states the acceptance criterion in terms that matter (does the convergence order survive, does the rightmost eigenvalue keep its sign).
- The Apple GPU as designed delivers nothing for the problems the group actually runs. All five conclude Metal pays only at 128^3-256^3 on Pro/Max chips and never in regime A, yet the shipped configs, the BO loop and the stability sweeps are all small 2D. None proposes ensemble batching of many independent small 2D cases (geometries, alphas, schemes) into one dispatch, which turns the launch-bound regime into a throughput problem the GPU can win. Also, parabolic dt scales as h^2 (`src/systems/heat.cpp:356-361`: dt = cfl h_min^2 / (4k)), so the step count for a fixed end time grows as (N-1)^2. Estimate: heat.lua takes 50 steps to t = 0.3 at 51x51 (`05-measurements.md` Section A), so a 256^3 run on the same domain needs about 1.3k steps for the same end time and reaches 1e5 steps only for an end time about 80 times longer; an earlier draft said "over 1e5 steps" without this derivation, which does not support it for the shipped end time. Either way the large-3D regime they optimise for is closer to an Euler regime that none has yet. The host runs of `05-measurements.md` Section D now bound the ensemble idea: on the M4 any GPU op plus synchronisation costs 167 to 181 us (the measured replacement for the earlier unverified forum figure of about 120 us), a whole-step-compiled fp32 synthetic step is 0.35 to 0.5 ms from 51x51 to 201x201 against 0.085 ms for the best CPU fp64 path at 51x51, and the GPU is 4 to 6x ahead of the CPU only at 129^3 and 257^3; the fp32 error on that synthetic E2 problem is 1e-7 to 2.4e-7 relative after 10 steps and says nothing about small-psi closures. Every other Apple GPU performance figure in this document is a projection.

Engineering and project.

- Build and CI integration is unplanned in all five: nanobind/scikit-build-core, metal-cpp, a runtime clang or Metal compiler, JAX or MLX pins and a self-hosted macOS runner are never costed against the Spack devcontainer plus GHCR image plus uv Python job that exist today. Specific traps: the Metal compiler as a separate Xcode download (stated by D5, unverified against current Xcode documentation), the Seatbelt sandbox hiding the Metal device (observed in the survey), and Kokkos-in-Python lifecycle rules colliding with pytest-xdist.
- Three of five designs (D2, D3, D4) silently drop Kokkos, the portability layer the upstream code adopted after the range-v3 migration; CUDA/HIP then becomes XLA:GPU/Pallas, NVRTC plus cupy, or hand-written CUDA templates, and the lab's Kokkos expertise stops being reusable.
- Setup-time scalability is hand-waved. Every design keeps the O(|R|) `interp_line` scan (`mesh.cpp:170`) or replaces it "later"; none estimates end-to-end setup at 256^3 with a sphere in pure Python; D5 admits "minutes". For the BO loop this is the dominant cost, not the step.
- JIT or compile latency (1-30 s) in BO loops and CI, and the need for clang or Metal toolchains at run time, are treated as mitigated by caching with no wheel-distribution plan.
- No CUDA machine, a hidden Metal device in the sandbox, no self-hosted macOS CI. Multi-GPU/MPI and memory capacity at 256^3 with Euler (3-4 GB per field set) are ignored.
- Data-dependent or moving geometry, solution-dependent stencils and implicit time stepping are excluded by all five in one line each. That means "the variety of use cases" is answered as "explicit RK on fixed geometry with linear spatial operators plus pointwise physics"; none says what the first out-of-scope request (likely a filter or limiter for Euler shocks) would cost in each architecture.
- The R-ordering contract is not treated as versioned data with a stored checksum in tables and output files (only D4 mentions a load-time checksum); a one-ulp snap difference silently permutes R and corrupts every table.

## 9. Recommended hybrid

All three judges converged on the same shape independently: build the thin waist, store the operator in Pencil's line format, verify the split with ccsx's exact-matrix rule and Pencil's monomial oracle, take kerf-forge's separable-source hoisting, and add exactly one fused primitive instead of a compiler. What follows is that proposal made concrete. It is not a new compiler and it does not need an AST front end or an IR with emitters.

### 9.1 What is built

Working name: `pyshoccs`. Four pieces.

1. A table exporter in C++, `ccs::assemble::derivative_tables`, obtained by keeping the loop bodies of `domain_discretization` (`src/operators/derivative.cpp:305-449`) and `cut_discretization` (`derivative.cpp:159-257`) verbatim and changing only their sinks. It emits, per operator, over one packed vector V = [D | Rx | Ry | Rz] with static offsets:
   - the exact operator A as COO (every O, B, N, Bf and Br entry, duplicates summed), which is the source of truth, the scipy export and the oracle;
   - per-direction interior taps, stored once;
   - one role-bit byte per D point (bit d = this row is owned by the direction-d band; bit 3 fluid; bit 4 Dirichlet face);
   - wall slabs: the psi=1 closure block per (axis, side, BC kind), stored once, with a pencil mask (Pencil's `Slab`, the thin waist's B4 data);
   - one exception table `Exc`: a sorted, unique-output compact CSR over V holding object closure rows with their B column folded in, wall rows of short fragments, every Bf+Br row merged per R point, and Neumann wall extras as columns into a small face-data buffer that is absent when no Neumann BC exists;
   - an R-ordering checksum over (psi, solid_ic) per direction, stored with the tables and later with output files.
   The split is decided by ccsx's rule (G1): a D row is structured iff the structured operator's row equals A's row in support and value to a psi-aware tolerance (4 ulp away from the clamps, looser near them); all other rows go to `Exc`. A row-wise identity test (structured part plus `Exc` equals A on random vectors) runs at build time.
2. A nanobind module `_shoccs` (scikit-build-core) exposing `ccs::mesh` (lines and R records as SoA NumPy views), `ccs::stencil`, `ccs::bcs`, the tables, the legacy `derivative` apply (as a frozen oracle), the backend API and DLPack views. `Kokkos::initialize` at import, `finalize` in `atexit` after the arena, GIL released around `run()`.
3. A `ccs::backend` with ten primitives on Kokkos host: D5's B1 to B10 with B4 retired and B11 added, with two edits taken from Pencil and the compiler designs:
   - B3 `axis_stencil` is the fused 3-axis masked band plus slabs kernel with an affine epilogue (PURE/WALL/MIXED tile variants, G4, so pure tiles load no tables), plane-tiled so the (2P+1)-plane working set stays in cache and the slow axis is never a team-per-line. D5's separate B4 `face_closure` (a batched small GEMM over the pencils of a face plane) is absorbed into this slab form: the wall slabs with their pencil masks are B3 data, not a separate kernel, so the label B4 is retired and not reused;
   - one extra primitive, B11 `rk_stage`: B3 whose epilogue performs the RK axpy pair (acc += b_s dt du; u_next = u0 + a_s dt du), the masked source add from hoisted spatial factors times c_g(t), the Dirichlet fill at t_next on face points, and per-thread stats partials, writing into a ping-pong buffer (fixing Pencil's three-buffer hazard); B5 `rowset_matvec` then accumulates the `Exc` rows into the same outputs. One RK4 step is 4 x (B11, B5, R-row B5) plus one reduction, about 12-16 nodes, K steps per native call.
   - the remaining primitives as in D5: B1 `lincomb`, B2 `map` (a small fixed functor menu plus SymPy-printed plugin functors compiled at setup and cached by hash, with Pencil's precompiled instantiations for W in {3,5,7,9,11} as the no-toolchain fallback, G18), B6 `gather`, B7 `scatter`, B8 `fill_at`, B9 `reduce` (min, max, maxloc, sum, max_abs_plus), B10 `point_eval`. The hybrid therefore has ten primitives in all: B1, B2, B3, B5, B6, B7, B8, B9, B10 and B11 (earlier drafts said "ten plus one" without noting that B4 had been absorbed).
   - a schedule as a flat node list over buffer ids plus a parameter vector (t, dt, stage coefficients, the c_g(t) scalars), validated once; a NumPy interpreter of the same node list as oracle (G14); a node-count assertion per step in CI (G16).
4. A Python package `shoccs` (NumPy plus SymPy): `Grid`, `Shape`, `Mesh`, `schemes`, `BCs`, `mms.Symbolic` (sources and BC data by `sympy.diff`; time-separable terms split into spatial factors evaluated once in fp64 at D, R and face points, G6; boundary-only BC evaluation), `Operator` with `.to_scipy()` returning A, integrators recording schedules, `Simulation` with a chunked driver, a device stats ring, a first-invalid-step index (G9), and io through the existing C++ writer. psi, alpha and every coefficient are runtime arrays (G8), so BO and alpha sweeps never recompile, and `cpp_bridge.py` is retired in favour of in-process runs.

What is deliberately not built: a tracing front end, a kernel IR with emitters, a Python-AST language, a tile-class compiler. The one place a DSL is genuinely needed is pointwise physics (D5's functor menu), and the repository already owns the right tool for it: `StencilCodePrinter` (`scripts/stencil_gen/stencil_gen/printer.py:11`, a `C99CodePrinter` subclass) prints SymPy expressions as C++ functors today and can print MSL later. Promote to a kerf-forge style generated stage kernel only if the measured profile shows the hand kernels cannot reach it, and then use the hand kernels as the oracle.

Adjustment after the host measurement (`05-measurements.md` Section D), stated as an adjustment to the plan above and not as a new design: phase 1 gains a parallel array arm. The real exported tables from piece 1 run through `experiments/host_primitives_bench.py` with NumPy, compiled MLX CPU and torch.compile CPU, alongside the C++ fused-stage arm, and gate (2) of Section 9.2 applies to both arms. The motivation is measured but synthetic: a fused fp64 step from Python costs 0.085 ms (compiled MLX CPU), 0.103 ms (eager NumPy) and 0.149 ms (torch eager) at 51x51 against 84 us for four native one-thread C++ RHS submits (216 us with four spin-waiting threads), and 4.2 to 4.5 ms at 65^3 against 3.8 ms (4 threads) and 8.3 ms (1 thread) of native submits (`05-measurements.md` Section E; container bar 91 us, and 3.2 ms at 8 threads, 7.5 ms at 1), with random closure coefficients, no host MMS and no update_boundary. If the array arm meets the discretization bar with real tables, piece 3 (the Kokkos CPU backend) becomes optional and the thin waist shrinks to C++ assembly plus nanobind export plus a Python runtime; pieces 1, 2 and 4 and every verification gate stay as written.

Verification gates, all in CI from the start: (a) structured plus `Exc` equals A on random vectors; (b) Pencil's monomial exactness on every fluid and R row (G2), with residual <= 16 eps x sum |coef x x^k| per row, run first on the dumped E2 Laplacian to settle t-laplacian; (c) for clamp-adjacent rows, defined as closure rows whose psi lies below 1e-4 or above 1 - 1e-4 (the E4_1 clamp bounds, the tightest in the code; the geometry clamp is [1e-12, 1 - 1e-12] at `src/mesh/object_geometry.cpp:87-95`), either a psi-aware tolerance tol = 64 eps x max|coef| x h, scaled by the row's cancellation estimate (G19), or an exact-rational reference (sympy or mpmath evaluation of the closure at that psi) to 4 ulp; (d) an optional setup-time rightmost-eigenvalue check of A on small problems that refuses unstable (geometry, scheme, alpha) combinations and informs dt; (e) one parity protocol used by every phase: fluid D rows away from the psi clamps are compared with the legacy C++ at 1e-13 relative for a single RHS (max |du_new - du_cpp| / max |du_cpp|) and at 4 ulp for operator tables; clamp-adjacent rows fall under (c); R rows are judged against the monomial oracle (b) and never against the legacy C++, because t-laplacian shows the legacy Bf/Br path is itself untrusted; the heat.lua Linf history must match `logs/system.csv` to 1e-11 with every R-point disagreement traced through (b); solid, Dirichlet and Dirichlet-skipped points are excluded (warning 2).

Idiom coverage of the hybrid. D1 to D5 each map all 51 idioms, but the hybrid changes D5 (B4 absorbed, B11 added, G1, G6, G7 and G9 grafted), so its own coverage is stated here. Rows that differ from D5's table in Section 7.7 are marked with an asterisk.

| Idiom | Mechanism in the hybrid | Risk |
|---|---|---|
| I01 | One packed arena buffer per field; `[ncomp, Npk]` for multi-component | low |
| I02 | C++ `object_geometry` unchanged through nanobind; SoA views | low |
| I03 | `mesh::lines(dir)` and `dirichlet_line` reused; role bits derived in the C++ post-pass | low |
| I04 | Masks and lists built once in Python from exported tables | low |
| I05 | C++ stencil classes through nanobind; alpha from `known_values.json` as a runtime table argument (G8); length check replaces silent truncation | low |
| I06 | Setup-only, evaluated by the existing C++ nbs inside the assembly loop | low |
| I07* | Kept in C++ inside the table exporter (G3); the NumPy port with searchsorted lookup (D5's optional phase 5) is pulled forward only if setup dominates the BO loop (Section 10) | medium |
| I08 | `domain_discretization` loop body reused; sinks emit records | medium |
| I09 | `cut_discretization` loop body reused; Bf and Br merged into one `Exc` row per R point | medium |
| I10* | COO collected in C++, lexsorted and merged in the post-pass; the same COO is A, the exact operator (G1) | low |
| I11 | Table export; no per-line allocations | low |
| I12 | Python front end replaces `from_lua`; Lua optional for the legacy binary | low |
| I13 | Same flat layout; B3 threads along the fast index; int64 variant when Npk exceeds 2^31 | low |
| I14* | Per-direction R order preserved inside the packed buffer; an R-ordering checksum stored with tables and output files | low |
| I15 | Arena buffers replace spans and system-owned vectors | low |
| I16* | B3 (interior rows by role bits, wall slabs by pencil mask, PURE/WALL/MIXED tiles, G4) plus B5 over `Exc` (object rows, short-fragment rows); the split decided against A (G1) | medium |
| I17 | B3 with up to 3 directions fused | low |
| I18* | Wall slabs inside B3 (D5's B4 absorbed; G10) | low |
| I19 | B5 rows of `Exc` | low |
| I20* | B5 over the single compact `Exc` CSR | low |
| I21 | B disappears as a matrix: extra columns of the object closure rows | low |
| I22* | Object Neumann and Robin constraints eliminated exactly into `Exc` rows at setup (G7); Neumann wall extras as columns into a small face-data buffer; still new numerics never assembled by the C++ | medium |
| I23 | Bf_r rows in `Exc` | low |
| I24 | Br_r merged into the same rows | low |
| I25 | Eliminated: B3 and B11 with op=eq are total | low |
| I26 | Scale folded into node epilogues | low |
| I27 | Schedule order is the accumulation order | low |
| I28 | B3 nout=3 | low |
| I29 | B2 `dot3` over the packed buffer | low |
| I30* | Hoisted spatial factors times c_g(t) added inside the B11 epilogue (G6); B7 scatter with unique indices for R lists | low |
| I31 | B8 `fill_at` with the reused plane descriptors (`selection_desc.hpp:3-15`); the Dirichlet fill also lives in the B11 epilogue | low |
| I32* | Schedule record/compile/run; swap by id; chunked driver with device stats ring and first-invalid-step index (G9); Kokkos Graph on Cuda; one command buffer on Metal; node-count assertion (G16) | low |
| I33 | A one-operator schedule | low |
| I34* | SymPy MMS by `sympy.diff`; time-separable terms evaluated once in fp64 at D, R and face points (G6); B10 only for non-separable terms or at setup | low |
| I35 | Dropped by design | high |
| I36* | c_g(t) scalars in the parameter vector; no per-stage field evaluation | low |
| I37* | B11 `rk_stage` epilogue (axpy pair into a ping-pong buffer, G5); B1 `lincomb` only where fusion is impossible | low |
| I38 | Not needed: swap by id | low |
| I39* | Dirichlet face values from hoisted factors inside B11; B10 on R lists only when non-separable | low |
| I40 | Lists built once | low |
| I41* | Per-thread stats partials in the B11 epilogue plus a B9 reduction into the device stats ring, read back per chunk | low |
| I42 | B2 `abs_diff` gated on dump-due; existing C++ `field_io` through nanobind | low |
| I43 | Python while loop over K-step chunks | low |
| I44* | `Operator.to_scipy()` returns A, the exact COO the tables were split from | low |
| I45* | A as the oracle, the monomial oracle (G2), the NumPy schedule interpreter (G14) and the retained legacy classes | low |
| I46* | B2 fixed functor menu plus printer-generated plugins, with precompiled instantiations as the no-toolchain fallback (G18); the one place the lack of a DSL shows | medium |
| I47* | `[ncomp, Npk]` state, batched B3/B5/B11 (G13), flux and primitive functors, normal projection, B9 dt, linearised operator from the tables (G17) | medium |
| I48 | Superseded by `to_scipy()` | low |
| I49 | `printer.py` gains a device-functor target | low |
| I50* | Unchanged host scipy consuming `to_scipy()`; the Euler linearisation through G17 | low |
| I51 | In-process; BO drives `sh.Simulation` directly with psi, alpha and coefficients as runtime arrays (G8) | low |

### 9.2 Roadmap

Phase 1 is the smallest experiment that de-risks the performance claim. It is C++ only, needs no Python, no nanobind and no new build system, and it answers the question every other phase depends on: do compact tables plus a fused stage kernel match or beat the live Kokkos RHS and step on this machine?

| Phase | Duration | Content | Gate |
|---|---|---|---|
| 1. Bar and kernel experiment | 3 to 5 weeks | (a) Native macOS build: done. The C++ dependency stack (Kokkos 5.1.1, sol2, spdlog, lapackpp and the rest of the Spack view) existed only in the devcontainer until 2026-10-13, and every measurement before then (`05-measurements.md` Sections A to C) came from the container (Docker Desktop on the owner's Mac, 8 vCPU, Kokkos OpenMP, Release). The native build is merged upstream ("Support a native macOS build", `docs/reference/macos-host-build.md`): a user-managed spack environment supplying llvm-openmp, Apple clang 21, Kokkos 5.1.1 OpenMP, Release. The thread count and toolchain for every later comparison are fixed by Section E: Apple clang 21 with Kokkos OpenMP (llvm-openmp); 1 thread for the shipped 2D cases, where four threads with libomp defaults are slower than one (the 51x51 RHS submit is 309 us against 21.0 us) and spin-waiting threads win only end to end; 4 threads from 201x201 up, spin-waiting (`KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active`) below 65^3 and libomp defaults at 65^3 for the RHS submit (961 against 1810 us), the better of the two settings per case (the shipped 65^3 step itself is still fastest spinning, 17.49 against 19.96 ms). The per-region timers of `benchmarks/bench_rhs_cutcell.cpp` (fill_source, graph submit, update_boundary, stats) have been run natively for heat.lua, scalar_wave.lua, 201x201, 401x401 and 65^3 beside the container columns (Section E); the 128^3 sphere case is still to be added. (b) One to two further weeks for the fixed-up C++ on a tagged branch, frozen as exactly four changes: (1) hoist the object-BC gather lists out of the per-call path; (2) compact B and N tables with no full-height row pointers; (3) manufactured-solution evaluation once per step and only where needed, with time factors hoisted (fill_source and update_boundary); (4) packed [D | Rx | Ry | Rz] buffers. The native measurement predicts what this buys: the shipped step is 0.856 ms at 51x51 at 1 thread (28 to 37% of it host fill_source) and 0.266 ms at 4 threads spin, and 19.96 ms at 65^3 at 4 threads (17.49 ms spinning); the fixed-up step is estimated at 131 us at 1 thread and 6.0 ms at 65^3 at 4 threads (12.4 ms at 1 thread), from change (3) alone (container: 0.90 ms with 59% fill_source and 37.9 ms; estimates about 0.15 ms and about 5.5 ms). (c) `derivative_tables` with changed sinks plus a C++ post-pass (role bits, slabs, sorted `Exc`); a `dump_operator` tool writing the same as .npz (G15). (d) A minimal Kokkos backend with B1, B3 (fused directions, role bits, tile classes, slabs), B5, B8 and the fused B11, and a plain schedule runner. (e) Hand-built schedules for the heat.lua RHS and step and for 65^3 with the sphere and the 128^3 sphere, run against `heat::submit_rhs_graph` (`src/systems/heat.cpp:290`) and a timed `simulation_cycle::run` on identical inputs in one process; plus a STREAM triad over three 134 MB arrays with the same compiler and thread count on the same machine. (f) On the side, the monomial oracle on the dumped E2 Laplacian (settles t-laplacian) and D2's 2-day NumPy fp32/df64 emulation study on the dumped operators (decides whether Metal is df64-only before any MSL exists). (g) The array arm, added after the host measurement of `05-measurements.md` Section D: the real exported tables from (c) run through `experiments/host_primitives_bench.py` with NumPy, compiled MLX CPU and torch.compile CPU, in the same process and at the same thread count as the C++ fused-stage arm of (e). The synthetic result that motivates it (0.085 to 0.15 ms per fused fp64 step at 51x51 against 84 us of native 1-thread RHS submits and 216 us at 4 threads spin; 4.2 to 4.5 ms at 65^3 against 3.8 ms at 4 threads and 8.3 ms at 1 thread; container bar 91 us and 3.2 ms at 8 threads) is not a pass of the gate. Status on 2026-10-13: the exporter (`shoccs-export`), the array arm (`python/pyshoccs`) and the C++ fused-stage arm (`benchmarks/bench_tables.cpp`) are in progress on branch `pyshoccs-phase1` against the data contract in `docs/reference/pyshoccs-export.md` | Pass, all required, at the thread count and toolchain fixed in (a): (1) parity per the Part III 9.1 protocol: 1e-13 relative on fluid D rows away from the psi clamps; clamp-adjacent rows under tol = 64 eps x max|coef| x h or an exact-rational reference to 4 ulp; R rows on the monomial oracle with residual <= 16 eps x sum |coef x x^k|, not compared with the legacy C++. (2) RHS at most 1.0x the fixed-up C++ RHS graph at 51x51 (shipped, native: 21.0 us at 1 thread, 54 us at 4 threads spin; container 22.7 us at 1 thread) and at most 0.5x at 65^3 with the sphere (shipped, native: 2065 us at 1 thread, 961 us at 4 threads; container 1867 us at 1 thread, 798 us at 8); the gate applies to both arms, the C++ fused-stage arm and the array arm on the real tables. (3) The fused B11 step at most 0.5x the fixed-up C++ step on heat.lua (native fixed-up estimate 131 us at 1 thread, and between 131 us and the measured 0.266 ms at 4 threads spin; shipped native 0.856 ms at 1 thread, 0.266 ms at 4 threads spin; container estimate about 0.15 ms, shipped 0.90 ms). (4) The 128^3 sphere B11 stage at or above 70% of the STREAM-triad bandwidth measured in (e), with bytes per stage counted as: reads of u, u0, acc and one hoisted source factor (4 x 8 B per D point) and of the role bits (1 B), writes of u_next and acc (2 x 8 B), 49 B per D point in all, stencil halo re-reads not counted; plus, for the `Exc` rows, 12 B per nonzero (8 B coefficient, 4 B int32 column) and 8 B per gathered value. (5) Table memory for the 64^3 sphere case below the legacy `meta_d` plus `coeffs_d` plus CSR value and column footprint, not counting the legacy full-height row pointers (which would make the gate trivial). Tuning of B11 is time-boxed to 3 working days after its first complete run; the gate is judged on the best result inside the box. Fail: the unfused thin-waist primitives still win on launch count and remain the fallback; if even they miss at 65^3 the plane-tiled masked-stencil thesis is wrong and the recommendation changes (Part III Section 10). If the array arm alone passes gate (2) with real tables, the Kokkos CPU backend becomes optional (Part III 9.1) |
| 2. Bindings and front end | 3 weeks | nanobind `_shoccs`; Python `Grid`/`Shape`/`Mesh`/`schemes`/`BCs`/`Operator`; `to_scipy()` wired into the stencil_gen stability cascade and the BO loop; R-ordering checksum in tables; psi-aware closure tolerance test | `to_scipy()` matvec equals the legacy block-plus-CSR matvec to 4 ulp (elementwise relative) on 100 random vectors over fluid D rows away from the clamps, clamp-adjacent rows under tol = 64 eps x max|coef| x h, R rows on the monomial oracle with residual <= 16 eps x sum |coef x x^k|; stencil_gen's 1D dense matrices equal `to_scipy()` to 4 ulp on unfragmented lines; `cpp_bridge` retired for new sweeps |
| 3. Whole-step schedules | 4 to 6 weeks | The remaining primitives; RK4/Euler schedules with B11; SymPy MMS with separable hoisting and boundary-only BC evaluation; plugin functors through the printer; stats ring with first-invalid index; XDMF through the C++ writer; NumPy schedule interpreter; node-count assertion in CI; heat and scalar_wave end to end | Linf histories match the Lua binary to 1e-11 with R-point differences explained by the monomial oracle; step at most 1.0x the fixed-up C++ step everywhere (native estimates: 131 us at 51x51 at 1 thread, 6.0 ms at 65^3 at 4 threads and 12.4 ms at 1 thread; container estimates about 0.15 ms and about 5.5 ms) and at most 0.5x at every size where phase-1 gate (2) passed at 0.5x, at the thread count fixed in phase 1; the Lua binary becomes legacy. About 3 months in from the start, Lua is gone and nothing has regressed |
| 4. Metal, gated | 6 to 8 weeks | Only if the phase-1 precision study passed: five MSL kernel templates behind the same primitive API (lincomb, fused band plus slab stage, `Exc` rows, map, reduce), runtime-compiled with `newLibraryWithSource`, df64 only for the structural kernels with sources hoisted so no transcendental runs on the GPU, fp64 tables split into hi/lo pairs (G19), K-step command buffers with one readback, `corr[]` plus `tile_ptr` for the row path (G11). Design the batch dimension in from the start so many independent 2D cases (BO sweeps) can share one dispatch; that, not 256^3 heat, is the GPU use case this group has. The host runs of `05-measurements.md` Section D give the first measured Apple GPU figures: a whole-step-compiled fp32 synthetic step is 0.35 to 0.5 ms from 51x51 to 201x201, held by a 167 to 181 us sync floor, and 4 to 6x faster than the best CPU fp64 path at 129^3 and 257^3; every other Apple GPU figure in this document is still a projection | (1) Standalone gate: B3 at 256^3 in df64 at most 0.5x the Kokkos fp64 time at 8 threads on the same M4, before the rest is written. (2) Ensemble gate: N independent heat.lua 51x51 cases per dispatch, N swept up to the memory limit; batched throughput (cases x steps per second) at least 1.0x the same N cases run on all CPU cores with the phase-3 backend, in the precision the study admitted. The floor to amortise is measured: 167 to 181 us per GPU dispatch plus synchronisation on the M4 (Part IV D), against 0.085 to 0.15 ms for a whole fused CPU fp64 step at 51x51, so batching N cases per dispatch must beat N CPU runs with that floor paid once per dispatch. (3) df64 Linf within 2x of f64 on heat.lua and scalar_wave.lua; convergence order preserved (observed order within 0.3 of the design order on the MMS refinement series). Failing both (1) and (2) deletes the phase |
| 5. Euler | 6 to 10 weeks | Batched primitives over [ncomp, Npk]; flux as a B2 functor materialised per direction in v1; slip projection and characteristic faces as functors; adaptive dt as a B9 reduction; filter as another table operator through the same exact-matrix path; conservation-by-construction closures (polyE2_1 style) rather than monitor-only; linearised operator for I50 from the tables and a SymPy flux Jacobian (G17); upgrade B11 with a flux-prologue ring buffer (G13) only after measuring the materialisation cost | Isentropic vortex in 2D with a cylinder, then 3D: observed convergence order within 0.3 of the design order of the scheme used; L2 error at 101x101 below a target to be fixed in phase 5 planning (none of the five designs gives one); discrete conservation error consistent with the conservation-by-construction closures |
| 6. CUDA | 2 to 3 weeks plus hardware | A Kokkos build-matrix entry (every kernel captures Views), a Kokkos Graph of the step schedule, async stats readback | Parity with the CPU results to 1e-12 relative (single RHS on the rows of the phase-1 protocol, and the heat.lua Linf history) on the same tables |
| 7. Codegen, only when it pays | on demand | Promote the printer-based functor path to a generated fused stage kernel (kerf-forge style, same tables, same gates) when the functor menu passes about ten entries or Euler flux fusion is measured to matter | The generated kernel passes the phase-1 parity protocol with the hand kernels as the oracle, keeps or lowers the node count per step, and at 256^3 Euler brings the step to at most 0.5x the hand-kernel step (the condition under which Part III Section 10 says flux fusion matters); otherwise it is not merged. The tables and gates do not change, so later codegen cannot change the numerics |

Effort: about 3 months to CPU parity with Lua retired; 6 to 9 months to Metal plus Euler; two people can work in parallel after phase 3 freezes the primitive API. Ongoing cost is curating a functor menu and a schedule validator, not maintaining a compiler.

### 9.3 Why this and not the runner-up

The compiler designs (D1, D4) have a higher ceiling: four dispatches per step, tile variants, Euler fluxes computed in the sweep window. The hybrid gets most of that ceiling from one hand-written fused primitive and gives up the rest in exchange for not owning an IR, three or four emitters and a df64 legaliser at 10-25% of an FTE. If phase 1 shows the fused B11 reaches the 128^3 bandwidth gate (70% of the STREAM triad), the remaining gap to a generator is the Euler flux prologue, which is deferred to measurement. The array design (D2) contributes its two best ideas (exact-matrix classification, the fp32 study before Metal) and is kept as a reference executor idea only, because its runtime performance depends on XLA heuristics and its Apple path is still a hand-written MSL printer behind an MLX custom kernel with read-only inputs. Pencil (D3) contributes the storage format, the monomial oracle and the precompiled fallback; the hybrid differs from it in keeping the C++ assembly loops and Kokkos rather than porting geometry to NumPy and dropping the portability layer.

## 10. What would change the recommendation

The recommendation is conditional on evidence that does not yet exist. The following findings, each testable in phase 1 or shortly after, would change it.

| Evidence | Effect on the recommendation |
|---|---|
| The stop rule: the native fixed-up heat.lua step is within 1.5x of four native RHS graph submits plus stats (no non-discretization overhead left to remove), or gate (3) fails; the numeric threshold is set after phase 1(b). Restated from `05-measurements.md` Section E: at 51x51 and 1 thread four submits plus stats are 4 x 21.0 + 26.9 = 111 us, so the threshold is 166 us and the modelled fixed-up step is 131 us (1.18x); at 65^3 and 4 threads with libomp defaults they are 4 x 961 + 955 = 4.8 ms, the threshold 7.2 ms and the estimate 6.0 ms (1.25x); the rule fires in both cases if the estimate holds. The shipped native step is 0.856 ms at 1 thread (28 to 37% of it host fill_source) and 0.266 ms at 4 threads spin; the container step was 0.90 ms, 59% fill_source, against 4 x 22.7 + 32 us. That is, the launch count and host MMS were the whole problem and the four frozen C++ fixes removed them. The rule is judged after phase 1(b) on the measured fixed-up step, not on this estimate | The Python layer is justified only as a Lua replacement and a `to_scipy()` provider. Stop at phase 2, skip the backend rewrite, and keep the C++ runtime. On the native estimate this is the expected outcome: once change (3) is made the C++ step is almost entirely its discretization, so phases 2 and later cannot be justified by removing host overhead; they rest on gate (2), beating the native RHS graph itself (21.0 us at 1 thread and 54 us at 4 threads spin at 51x51; 2065 us at 1 thread and 961 us at 4 threads at 65^3), on the Python front end and `to_scipy()`, and on the ensemble GPU question of phase 4 |
| The phase-1 array arm (Section 9.1) meets gate (2) with the real exported tables: NumPy, compiled MLX CPU or torch.compile CPU at most 1.0x the fixed-up C++ RHS graph at 51x51 and 0.5x at 65^3 (the synthetic host result of `05-measurements.md` Section D against the native bar of Section E is 1.0x to 1.8x of four 1-thread submits at 51x51 (84 us) and 0.4x to 0.7x of the 4-thread-spin submits (216 us), and 1.1x to 1.2x of the 4-thread submits at 65^3 (3.8 ms); against the container bar it was 0.9x to 1.6x and within 1.4x; neither is a pass) | The Kokkos CPU backend (piece 3) becomes optional; the thin waist shrinks to C++ assembly plus nanobind export plus a Python runtime, with the tables, the oracle and every gate unchanged |
| The fused B11 cannot reach 70% of the STREAM-triad bandwidth at 128^3 to 256^3 within the 3-working-day tuning box, and the unfused B3 also misses the 65^3 gate | The plane-tiled masked-stencil thesis is wrong for this hardware. Re-evaluate D1's tile-class generator (whose 2.5D rolling window is a different memory pattern) with the same tables and gates before concluding that no Python route exists |
| The monomial oracle shows the Bf/Br R-row formulation itself is inexact or unstable (the t-laplacian failure is a formulation issue, not a bug) | Stop all porting work on R rows; the numerics must be fixed in stencil_gen and the C++ first, since every design would otherwise port a wrong operator faithfully |
| The NumPy fp32 study shows fp32 keeps the convergence order and the rightmost-eigenvalue sign for E2_2, polyE2_1 and E4_2 at shipped psi distributions (the only datapoint so far, `05-measurements.md` Section D, is 1e-7 to 2.4e-7 relative after 10 steps on synthetic E2 closures and does not cover small psi) | Metal becomes fp32-first with df64 as an option, the GPU window widens by about 2x, and MLX or torch-MPS custom kernels (`frameworks/mlx.md`, `frameworks/pytorch.md`) become acceptable hosts instead of metal-cpp; D2's Apple path regains value |
| The fp32 and df64 studies both fail (convergence order lost even with about 48-bit mantissa, or the fp32 exponent range or flushed denormals matter near the psi clamps) | Drop the Apple GPU goal for this solver; the recommendation becomes CPU and CUDA only and phase 4 is deleted |
| The group's dominant workload turns out to be ensembles of small 2D cases (BO sweeps, stability scans) rather than single large runs | Phase 4 is redesigned around a batch dimension over independent cases from the start, which is a regime where a GPU can win even without fp64 (unverified; nobody has measured batched small cut-cell cases on Metal). The measured floor it must amortise is 167 to 181 us per dispatch plus synchronisation, against 0.085 to 0.15 ms for a whole fused CPU fp64 step at 51x51 (`05-measurements.md` Section D), so N cases per dispatch must beat N CPU runs |
| Euler becomes the primary target before phase 3 is done, and v1 flux materialisation at 256^3 is measured to cost more than 2x the fused traffic | Bring phase 7 forward: generate the fused stage kernel from the tables (D1's temp-placement pass) rather than hand-writing per-system variants; the tables and gates stay |
| The lab decides to drop Kokkos (for example because upstream moves to another portability layer, or CUDA is never needed) | D3 Pencil becomes the better base: a closed IR with four kernels per backend and no Kokkos dependency, with the same tables |
| A measured, native JAX comparison reproduces the local jax.md numbers (step 66-74 us at 51x51, 1.7 ms at 64^3, 86 ms at 256^3; `frameworks/jax.md:201-217`, currently unverified) against the fixed-up C++, and XLA's fusion proves stable across two jax releases | D2 ccsx becomes a credible CPU/CUDA runtime with the best onboarding story; the hybrid's tables and verification gates transfer unchanged, so the switch is a backend change, not a redesign |
| Kokkos-in-Python lifecycle problems (initialize/finalize ordering, OpenMP threads versus the GIL, pytest-xdist forks) prove unfixable in practice | Move the K-step loop and the arena behind a C ABI loaded via ctypes (D4's runtime shape) with Kokkos initialised inside a worker process, or drop Kokkos for the CPU path (D3 or D4 runtime) while keeping the tables |
| Setup time at 256^3 (the O(|R|) `interp_line` scans, per-point virtual nbs calls) dominates the BO loop | Pull D5's optional phase 5 (vectorised NumPy assembly with searchsorted fragment lookup, batched lambdified closures) forward to before phase 4, with the C++ exporter kept as the oracle |
| A request for moving geometry, solution-dependent stencils or implicit time stepping arrives | None of the five designs covers it; the first would need device-side closure evaluation (the stencil_gen printer can emit it, the `interp()` dispatch cannot be ported as is), the second a nonlinear stencil-map kernel kind, the third preconditioned solves on the exported matrix. Each is a scope change to be costed separately, not an extension of this plan |

Phase 1(a) is done: the native build exists and `05-measurements.md` Section E measured the shipped C++ on it on 2026-10-13, so the single most valuable next action is phase 1(b), measuring the four frozen fixes natively, alongside the phase-1 arms in progress on branch `pyshoccs-phase1`. The container measurements have replaced the March baseline, Section D has measured the Python side on the host and Section E has put the C++ side on the same machine; the fixed-up C++ figure, which the stop rule and gate (3) are downstream of, is still an estimate (131 us at 51x51 and 1 thread; 6.0 ms at 65^3 and 4 threads; 12.4 ms at 65^3 and 1 thread).
