# Part VI: Where the DSL should go, given the research goal

Status: written 2026-10-14 on branch `pyshoccs-phase1` and revised the same day after an adversarial review. This part measures nothing new. It reframes Parts I to V around the owner's statement of the goal (Section 1). It synthesizes four inputs: a requirements analysis of the code, four candidate designs (optimization-first, exploration-first, correctness-first, incremental), and three independent judges (a researcher, a correctness auditor and a maintainer). Parts I to V optimized for runtime parity with the Kokkos solver. This part asks a different question: what makes the stability research loop faster and harder to get wrong. The earlier measurements stand, and their roadmap is re-sorted in Section 7.

Evidence tags: **[V]** means checked in code for this part (file:line given). **[R]** means the requirements analysis or a design verified it and this part did not re-check it. **[D]** means a doc or plan says it. **[E]** means an estimate that nobody has measured. `SG` is `scripts/stencil_gen/stencil_gen`.

## 0. Summary

**Build next.** Build a thin research harness in `python/pyshoccs` around one operator: the one the C++ solver assembles and `shoccs-export` writes. Every tier analyzes a C++ export. The tiers are a pole check, the monomial oracle, a 1D closure spectrum from the C++ eigenvalue path, the spectrum of the fully discrete RK4 map of the BC-reduced operator, an in-process linear RK4 on the same matrix, and an audit by the C++ binary. Heat and scalar_wave are linear and autonomous, so the spectrum decides stability; the time-domain tiers validate forcing, boundary data and the code path. Every tier returns a typed outcome with provenance into an append-only store. New cut-point ideas, new closure rows included, start as "surgery" on the exported operator, with no C++ until the idea wins. The first two weeks fix today's live silent-failure bugs (Section 8, step 0). Before any interpolation campaign, a go/no-go (step 4, about weeks 11 to 13) asks whether interpolation beats an algebraic R treatment at all. If it does, the first campaign, on polyE2_1 with a floating object on a well-posed template, lands at about weeks 14 to 16.

**Stop doing.**
- Runtime-parity work: freeze the fused kernels; drop Metal, CUDA, kernel codegen and Lua retirement; defer nanobind.
- Optimizing on the uniform E4u cascade.
- Building any new Python assembler of the cut-cell operator.
- Using inf or 1e12 sentinels.
- Using `cpp_bridge` for cut-cell work.
- Running campaigns on templates that never exercise interpolation.
- Reading stability of a linear problem off a short simulation window.

**Where the evidence is thin.** Nobody has measured the cost of a first-derivative export, a dense eigensolve at N=31, or an in-process scalar_wave RK4. No proxy has yet been shown to discriminate stable from unstable on a cut geometry where interpolation is active. Nobody knows whether the interpolant degree that higher order demands (Section 4.7) leaves interpolation viable at E4.

## 1. The goal restated as testable requirements

The owner's goal has two parts:

- **G1.** Optimize the cut-cell stencils, interpolation included, for stability by combining cheap proxies with simulations in one coherent strategy.
- **G2.** Make the package easy to modify, so a new idea (possibly not interpolation) does not start from scratch and the software is hard to misuse.

The requirements analysis turned these into sixteen hard requirements. The table compresses them and states the recommended plan's coverage.

| # | Requirement | Acceptance test | Serves | Coverage in this plan |
|---|---|---|---|---|
| H1 | One scheme declaration: name, ν, supported BCs (object/grid), R-row policy, ordered named parameter schema with bounds and groups, pole surfaces. All consumers use it. | Registry and C++ factory list the same schemes. A wrong-length or wrong-name vector is rejected in both languages with an error naming the scheme. | G1, G2 | Partial: one Python declaration. The C++ factory stays hand-written and is checked against it in CI, not generated from it (step 7). |
| H2 | Every generated stencil regenerates from tracked code | CI regenerate check for polyE2_1. E4_1 has no generator in the tree [R] and is listed as legacy. | G2 | Numeric check, not byte diff (Section 5) |
| H3 | Symbolic closure and compiled C++ `nbs`/`interp_*` agree to 1e-12 relative at ≥100 feasible (ψ, params) points plus ψ endpoints | Fails today on E2_1 [R] | G1, G2 | Full, step 2 |
| H4 | Every proxy's operator is the solver's export, or a twin parity-checked against it, for ν=1 and ν=2 with R rows | Export supports gradient, scalar_wave, heat. The proxy operator equals the export to 1e-13. | G1 | Full by construction: every tier, T1.5 included, reads a C++ export |
| H5 | Gated polynomial-exactness oracle per row class and ν | Transverse ν=2 gate fails today. A ν=1 oracle exists. | G1, G2 | Full, step 1 |
| H6 | Unsupported (scheme, BC, object/grid, ν, geometry) combinations are hard errors in both languages | One negative test per known case | G2 | Full, step 0 |
| H7 | No silent defaults in experiment-defining configuration. Unknown keys are errors. | `max_tim`, a missing `interpolant_alpha` and a short alpha array all fail | G2 | Full, step 0 |
| H8 | Pole guard in both languages, with Infeasible as a typed outcome | E4_1 at α0=73/66 and E2_1 at ψ=1, α=[1,0,0,0] are refused before any NaN appears [R] | G1 | Python T0 and the codegen'd C++ throw, both in step 2 |
| H9 | Typed results (Ok, Infeasible, Unanalyzable, Error) with no sentinels. Objective paths are validated at construction. | A typo raises at construction. An injected exception appears in the record. | G1 | Full, steps 0 and 2 |
| H10 | Simulation as a calibrated oracle: binary path configurable, threads pinned, t_final verified, one pass definition, versioned and well-posed template that exercises interpolation | An early stop is reported as a non-success | G1 | Full, steps 0 and 3 |
| H11 | Proxy admission: discrimination pair under the simulation's BCs, agreement with dense eigenvalues of the H4 operator, measured cost | L3r and the E4_1 Kreiss sweep pass or are excluded | G1 | Full, step 5 |
| H12 | At least one proxy and one simulation exercise ν=2 cut R rows. t-laplacian gates once the interpolant is decided. | A CI job exists | G1 | Step 6, after object Neumann assembly (Section 4.4) |
| H13 | The cut-point treatment is a declared component. A new one plugs in without editing proxies or the optimizer. | A toy non-interpolating treatment is added by touching only its own files | G2 | Full for linear treatments that replace R rows or D closure rows (Sections 4.6, 4.7) on the existing V with 1D ψ geometry. Not covered: new unknowns from cell merging, multi-D geometry. |
| H14 | Conventions as typed fields: wall orientation, mirror sign by ν, h power per coefficient family, analyzed operator kind, ψ clamp | The mirror on a ν=2 closure is reverse-only, which fails today at `SG/reflection.py:941` [V] | G2 | Full for Python code that still mirrors. Surgery works on exported coordinates and never mirrors. |
| H15 | Provenance and an append-only store: SHA and dirty flag, binary hash, registry hash, full parameters, template text, seed, lock hash, tier versions | A rerun from a record reproduces it | G1 | Full, step 2 |
| H16 | The Python-C++ link (H3, H4, one simulation smoke test) gates every PR | A CI job exists and gates | G2 | Full, steps 1 to 3, at a CI cost (Section 10) |

Soft preferences, in order of weight: S1 author in Python; S2 evaluate in-process; S3 in parallel; S4 smooth margins; S5 minimal files per idea; S6 readable Lua; S7 C++ stays fast, but parity is not the objective; S8 prune degenerate machinery; S9 notebook friendly.

## 2. The research loop today

| Step | Code | Cost per evaluation | Duplication | Live hazards |
|---|---|---|---|---|
| Formulate | `SchemeParams` (`SG/temo.py:98-132`) [R]. The interp construction is p=1 only. | Human | Seven Python scheme-identity tables plus the C++ if-chain [R] | `zeros` read only as a truthy flag [R] |
| Derive | `derive_cut_cell_paper`, `derive_cut_cell_scheme`, `build_polyE2_1_spec` (`SG/interp.py:512-521` [V]) | 0.11 s (E2_1), 7.3 s (E4_1) cold [R] | Four cut-geometry (ψ) computations [R] | E2_1 Python ≠ C++ by O(1), cause unknown [R]. E4_2 symbolic derivation raises [R]. |
| Codegen | `StencilGenSpec` (`SG/codegen.py:233-267`) [R]. Only polyE2_1 has a production spec. Output is hand-copied into `src/stencils/`. | Seconds plus a rebuild | E2_1, E2_2, E4_2 and E4u are hand-written or Mathematica exports [R] | E4_1 cannot be regenerated from the tree [R] |
| Wire to runtime | `stencil::from_lua` (`src/stencils/stencil.cpp:20-101`) | none | Six Lua keys accepted; the bridge emits only `order = 1`, alpha, sigma, epsilon (`SG/cpp_bridge.py:79-85` [V]) | Zero padding (`src/stencils/stencil.hpp:40-45` [V]). The packed polyE2_1 path calls `subspan(6, 9)` and `subspan(9, 13)` (`stencil.cpp:87` [V]): undefined behaviour for any length, and an out-of-range read for vectors shorter than 13, since `copy_zero_padded` truncates. |
| Cheap proxies, uniform | `brady2d_stability_score` L1 to L7 | 9 ms (L1), 30 ms (L3), 10.25 s (to L7) [R] | Three first-derivative assemblers in Python plus C++, none linked by parity [R] | Analyzes E4u closures with no cut cells |
| Cheap proxies, cut | `cutcell_stability_report` (`SG/brady2d_stability.py:1534` [V]) | 14.7 s for E2_1 [R] | A fourth closure model (Python `CutCellEvaluator`), with no R rows | False positives for E4_1 at 6 of 20 ψ [R]. Not wired to any optimizer. |
| Simulate | `run_cpp_brady2d`, one subprocess per point | 1.6 to 1.7 s (N=31, t=10), 3.35 s (N=61) [R] | Three L8 pass definitions and three (N, t_final) windows [R] | Binary path fixed at `build/` (`SG/cpp_bridge.py:26` [V]). Verdict `final_linf < 10.0` (`:305-307` [V]) ignores a null return on "required timestep too small" (`src/simulation/simulation_cycle.cpp:80-84` [V]). The cut template's sphere is Dirichlet (`lua-configs/brady_livescu_4_3_cut.lua:32` [V]), so `cut_discretization` returns early (`src/operators/derivative.cpp:173-175` [V]) and interpolation is never exercised. |
| Optimize | scipy, SHGO, DE, NSGA-II, multi-fidelity BO | Serial; 50 to 1740 evaluations per run [R] | Bounds parsing in three copies; two sentinels (inf, 1e12) [R] | `except Exception: return float("inf")` (`SG/optimizer.py:319-320` [V]). Bounds exist only for E4u and RBF scalars (`:63-71` [V]). BO takes `posterior.mean.max()` (`SG/bo.py:964` [V]) but recommends by `argmin` (`:1321-1331` [V]). |
| Record | `known_values.json`, `bo_runs/`, `pareto_fronts/` | none | Encoders in two to four copies [R] | No provenance. The one BO run ended at the 1e12 sentinel. 10 of 19 Pareto members fail [R]. |

Key numbers:

| Item | Value |
|---|---|
| Files to try one non-interpolating idea today | About 5 C++, plus 4 shared C++ (`derivative.cpp`, `mesh.cpp`, `stencil.hpp`, export), plus 6 to 8 Python modules, plus Lua templates [R] |
| Independent first-derivative operator implementations | 4, with no parity link [R] |
| Exporter wall time, phase 1 | 0.039 s (heat 51x51) to 0.169 s (65^3) [D, Part V Section 1] |
| scipy CSR RK4 step on the real 51x51 operator | 0.099 ms [D, Part V Section 5] |
| Tests | 1267 in stencil_gen, 140 of them slow [R]; 39 in pyshoccs with an export (Part V Section 1). No CI job runs Python against C++ [R]. |

The structural fact behind every row: the loop that optimizes runs on uniform E4u closures, and the loop that knows about cut cells is never optimized. The stated target, polyE2_1's interpolation parameters `ia[4]` (`SG/interp.py:520` [V]), sits outside both, and no shipped template can exercise them.

## 3. The directions considered

**Optimization-first.** A `Study` pushes each point up a ladder of calibrated fidelities (poles, a 1D closure spectrum, the exported matrix's spectrum, an in-process linear simulation, a C++ audit), with a cache, a store, and a monitor that demotes a rung when a Wilson bound on its false-accept rate is exceeded. Its key observation: heat and scalar_wave are linear, so every tier below the audit can act on one matrix. Its weaknesses: the most infrastructure before any campaign (14 to 18 weeks), and a headline example on a floating sphere it admits is ill-posed on its inflow face.

**Exploration-first.** Python becomes the research reference: one idea file declares a closure, treatment or equation, and a new Python reference assembler builds the operator. It has the best idea-to-verdict loop and good small ideas: `along_ray` orientation, `Equation.classify` assigning a BC per record from the wave speed, the `known_inexact` and `python-only` tags, and eigenvector localization on R points. Its weakness is decisive: a fourth Python implementation of the logic in `derivative.cpp:22-70` and `:208-251` and `stencil.hpp:213-270`, guarded by 1e-13 parity on shipped cases. Two drifting implementations caused the project's costliest incident: E4_1 was mis-ported twice and the error took about 16 weeks to find [R].

**Correctness-first.** One registry, a generated C++ registry header, strict `from_lua`, a single Python assembler that raises `OracleFailure` at construction, convention-derived mirrors, and a negative control for every test. Its step 0, one week of known-defect fixes each with a test that fails on the parent commit, is the best first move any design proposed. Its weaknesses: about seven weeks of gates before the first campaign, the same second-assembler risk, and a promised zero-diff regeneration of E4_1, which has no generator in the tree [R].

**Incremental.** The solver's export is the single numerical truth, and `pyshoccs` grows a thin harness around it: registry, typed results, an eigenvalue proxy on the true BC-reduced operator, a checked simulation runner, a study store. New ideas are row surgery on the exported operator, and a fourth Python assembler is forbidden. Its premise checks out: `extract_coo` takes any `derivative` (`src/operators/tables.hpp:69-70` [V]), and only the exporter's driver is tied to heat and the Laplacian (`src/app/export_tables.cpp:11,336,413-415,660` [V]). It alone both spots the ill-posed floating sphere and prices a fix, and its five steps each stand alone (10 to 12 weeks). Its weaknesses: scheme declarations stay in two languages, checked rather than generated; every point pays an export subprocess; its example B extrapolates D rows, which still read cross-line data; its provenance and calibration discipline is thin.

Judges' scores (0 to 10 per criterion; criteria in the order optimize, explore, misuse resistance, cost and risk; totals out of 40):

| Design | Researcher | Correctness | Maintainer | Mean per criterion (opt / expl / misuse / cost) | Mean total |
|---|---|---|---|---|---|
| Optimization-first | 9 / 7 / 8 / 5 = 29 | 8 / 6 / 8 / 6 = 28 | 8 / 6 / 7 / 5 = 26 | 8.3 / 6.3 / 7.7 / 5.3 | 27.7 |
| Exploration-first | 8 / 9 / 7 / 4 = 28 | 7 / 9 / 6 / 5 = 27 | 6 / 8 / 6 / 4 = 24 | 7.0 / 8.7 / 6.3 / 4.3 | 26.3 |
| Correctness-first | 7 / 7 / 10 / 5 = 29 | 7 / 6 / 8 / 5 = 26 | 7 / 6 / 9 / 5 = 27 | 7.0 / 6.3 / 9.0 / 5.0 | 27.3 |
| Incremental | 8 / 7 / 8 / 8 = 31 | 8 / 7 / 8 / 8 = 31 | 7 / 6 / 7 / 9 = 29 | 7.7 / 6.7 / 7.7 / 8.3 | **30.3** |

**Why the recommendation wins.** The totals are close (26 to 31 of 40), and the margin comes almost entirely from cost and risk. What matters more is that all three judges independently reached the same structure: incremental as the spine, with grafts. That ties to the owner's own words: the software has been easy to misuse, and the worst misuse was two sources of truth for one operator. Exporting the solver's operator is the only route to a single source of truth without first writing and validating a new assembler. The grafts:

- **From correctness-first:** step 0, with fail-on-parent tests, and the negative-control rule.
- **From optimization-first:** the typed outcome, provenance and store; calibration with demotion; the in-process RK4 on the exported matrix, here as a validation tier; the `Unanalyzable` refusal for unported ideas.
- **From exploration-first:** the treatment declaration (role, exact degree, `python-only`, `known_inexact`) and the typed conventions, which make the surgery lane safe and cheap.

Two gaps no design closed are closed here: a well-posed template that needs no per-record BC (Section 4.4), and a surgery role for D closure rows (Section 4.7), without which a new closure family still takes the 15-file path of Section 2.

**Re-score of the grafted plan.** This part's own estimate, not a judge's [E]: optimize 8, explore 8 (the closure-row role and the step-4 go/no-go answer the exploration gap; incremental scored 6.7), misuse 8, cost and risk 7 (the plan grows from 10 to 12 weeks to about 17 to 20). About 31 of 40: the grafts move points from cost to exploration rather than raising the total.

## 4. The recommended package

### 4.1 Layers

| Layer | Location | Content | Status |
|---|---|---|---|
| Derivation | `scripts/stencil_gen` | sympy closures, denominators, `StencilGenSpec` codegen | Reused unchanged as a library. pyshoccs imports it; the reverse never happens. |
| Solver | `src/` | Stencils, assembly, mesh, systems | Reused. Changes are strictness, capability errors, codegen'd pole checks, one template feature (step 3) and object Neumann assembly (step 6). |
| Export (the waist) | `src/app/export_tables.cpp`, `src/operators/tables.*` | Gains the first derivative and gradient, scalar_wave with its coefficient fields, alpha, dirty flag and binary hash in `meta.json`, an R-ordering checksum, closure dumps for H3, a 1D export over a ψ grid for T1.5, a mode without R rows, and per-entry slot metadata (Section 4.7) | Extended |
| `pyshoccs.schemes` | Python | `SchemeDecl` registry, `ParamSpace`, `Conventions`, poles from the sympy spec, split-array Lua emitter | New |
| `pyshoccs.cases` | Python | Versioned templates with posedness tags, geometry ensembles, the export cache | New |
| `pyshoccs.operator` | Python | `Operator` over an export: `semi_discrete()` with BC reduction and the forcing split off; treatment application | New, on `tables.py` |
| `pyshoccs.oracle`, `parity` | Python | ν-general monomial oracle with gated row classes; H3 and H4 harnesses | Extended |
| `pyshoccs.tiers`, `sim` | Python | T0 to T3 in-process; T4 C++ runner | New. `heat.py:58` (`rk4_step_generic`) is generalized. |
| `pyshoccs.surgery` | Python | `Treatment` with roles `algebraic` and `closure`, exact degree and promotion status; `line_nodes` and orientation helpers from exported records | New |
| `pyshoccs.study`, `store` | Python | Driver over typed outcomes, calibration, disagreement monitor, append-only JSONL store with content-hash keys | New |

The JSONL store is the lean form the maintainer judge asked for. Move to SQLite or Parquet only if querying hurts.

### 4.2 Core abstractions

```python
@dataclass(frozen=True)
class Param: name: str; group: str; lo: float; hi: float; active_for: frozenset[str]

@dataclass(frozen=True)
class Conventions:
    nu: Literal[1, 2]                       # mirror sign derived: odd -> negate+reverse, even -> reverse
    h_power: Mapping[str, int]              # {"nbs": -nu, "interp": 0}
    psi_clamp: tuple[float, float]          # must equal export meta; C++ clamps to [1e-12, 1-1e-12] (object_geometry.cpp:95)

@dataclass(frozen=True)
class SchemeDecl:
    name: str; lua_type: str; conv: Conventions
    params: tuple[Param, ...]               # ordered; groups map to split Lua arrays
    supports: frozenset[tuple[str, str]]    # (object|grid, dirichlet|floating|neumann)
    treatment: "Treatment"                  # Interpolate(...) for shipped schemes
    status: Literal["generated", "legacy"]  # legacy is refused by Study unless H3 passes
    def space(self, bcs, fixed=None) -> "ParamSpace"   # groups live under these BCs; fixed ones recorded
    def poles(self) -> "PoleSet"            # from SG/cutcell_eval.py:81 _collect_denominators
    def lua(self, p: "Point") -> str        # split arrays, explicit order, every wall named, never packed
    def with_treatment(self, t: "Treatment") -> "SchemeDecl"

class Operator:                              # only ever built from an export
    tables: Tables; prov: Provenance
    def semi_discrete(self) -> tuple[sp.csr_matrix, "Forcing"]   # unknowns only; Dirichlet columns split off
    def apply(self, t: "Treatment", p: "Point") -> "Operator"     # surgery; result keeps a lineage tag

Outcome = Ok | Infeasible | Unanalyzable | Error   # Ok(values: dict, margins: dict, diag: dict)

class Tier(Protocol):
    name: str; version: str; metrics: frozenset[str]
    admission: "Admission | None"            # None: diagnostic only, may not gate
    def __call__(self, decl, p, case) -> Outcome

@dataclass(frozen=True)
class Provenance:                            # attached by the store; a missing field raises
    git_sha: str; dirty: bool; binary_sha256: str | None; registry_hash: str
    params: dict[str, float]; template_text: str; seed: int; lock_hash: str
    tier_versions: dict[str, str]; cpp_validated: bool

class Study:
    def __init__(self, name, decl, space, cases, ladder, objective, constraints, search, seed, store): ...
    def run(self, budget) -> "StudyRecord"   # report() refuses to name a best point while a disagreement is unacknowledged
    def calibrate(self, panel) -> "AdmissionReport"
    def validate(self, point, holdout) -> "StudyRecord"
```

### 4.3 Reuse, wrap, retire, by file

| File | Fate | Reason |
|---|---|---|
| `SG/temo.py`, `SG/interp.py`, `SG/codegen.py`, `SG/printer.py`, `SG/taylor_system.py`, `SG/conservation.py` | Reuse | The derivation layer is sound where it is generated. Poles come from `SG/cutcell_eval.py:81` (`_collect_denominators`). |
| `python/pyshoccs/pyshoccs/tables.py` (`to_scipy` at `:325`, `locations` at `:294`, R records with `ray_outside` and `solid_coord` at `:169-170` [V]), `oracle.py`, `parity.py`, `structure.py` | Reuse and extend | The phase-1 independent arm. The oracle is ν=2 only today (`oracle.py:3-9` [V]). |
| `python/pyshoccs/pyshoccs/heat.py` | Generalize | `rk4_step_generic` (`:58` [V]) becomes T3 for any `LinearSystem` |
| `src/app/export_tables.cpp`, `src/operators/tables.*` | Extend | Heat and Laplacian only today (`:11,336,413-415,660` [V]). Meta has `git_commit` but no alpha (`:608-609` [V]). |
| `src/systems/hyperbolic_eigenvalues.cpp`, `src/operators/eigenvalue_visitor.*` | Reuse as the T1.5 source | The 1D C++ gradient's eigenvalues (`hyperbolic_eigenvalues.cpp:84-88`, 1D by assertion at `eigenvalue_visitor.hpp:20-25` [V]). Export that operator over a ψ grid instead of reducing it to one number. |
| `src/stencils/stencil.cpp`, `stencil.hpp` | Fix | Packed path (`stencil.cpp:87`), `copy_zero_padded` (`stencil.hpp:40-45`) |
| `SG/cpp_bridge.py` | Retire for cut work; keep for legacy uniform sweeps after step 0 hardens it | `pyshoccs.sim.Cpp` replaces it |
| `SG/optimizer.py` | Wrap its scipy drivers behind typed outcomes; retire `make_objective` | `:210-224`, `:319-320` |
| `SG/bo.py`, `SG/pareto.py` | Freeze as legacy | The 1e12 sentinel and the unresolved max/argmin question. A driver only re-enters behind the ask/tell interface. |
| `SG/brady2d_stability.py` L1 to L7, `cutcell_stability_report`, the Kreiss sweep (`SG/cutcell_normal_modes.py:338` [R]) | Legacy. The Kreiss sweep is diagnostic only unless its closure rows are parity-checked against the 1D export (H4). | They analyze an operator the solver does not run |
| `SG/interval_operator.py:106`, `SG/phs.py:622`, `SG/brady2d_stability.py:497` assemblers | No new callers | The waist replaces them for cut work |
| `SG/reflection.py:941` | Fix with a ν guard in step 0 | The unconditional `-weights` [V] |
| `sweeps/optimize.py`, `sweeps/bo.py`, `sweeps/pareto.py` | Frozen | Three copies of bounds parsing [R] |
| `src/operators/tables_kernels.hpp`, `benchmarks/bench_tables.cpp` | Freeze: keep building and tested, invest nothing | Runtime does not limit the loop |

### 4.4 A well-posed template without a per-record BC

Three facts close off the obvious templates:

- The scalar wave advects with velocity +r̂ away from the system centre. The RHS is `gG · ∇u` with `gG = -(x - c)/|x - c|` (`src/systems/scalar_wave.cpp:27-33,75-77,208` [V]).
- Object BCs are per shape (`derivative.cpp:192,212` [V]).
- Any closed body outside the source therefore has an inflow face and an outflow face. A Dirichlet sphere never exercises interpolation. A floating sphere is ill-posed on its inflow face.

Three designs answered this with a per-record "characteristic" object BC. That touches assembly at four call sites (`derivative.cpp:192,212,339,388` [V]) and the systems' Dirichlet fill [R], and it is the riskiest C++ item in any plan.

A cheaper template exists. Reverse the field to converge on c (velocity -r̂) and put c inside the sphere. Then:

- every point of the sphere is outflow, so the sphere is floating, interpolation is exercised everywhere on it, and it is well-posed;
- every domain wall is inflow, so all four walls are named Dirichlet. This replaces the default-floating far walls (`src/operators/boundaries.cpp:44` [V]) that dominate today's growth rate (E2_1 at α=0 grows at +7.9 on the uniform template, `SG/brady2d_stability.py:1169-1171` [V]).

The Dirichlet walls still carry closure parameters. A Dirichlet grid wall takes its closure from `st.nbs(h, grid_bcs[dir].left, 1.0, ...)` (`derivative.cpp:369-374` [V]), and polyE2_1's `nbs_dirichlet` reads `da` (`src/stencils/polyE2_1.cpp:208-215` [V]) with denominator 1 + 2ψ + 3da2 + 2ψ·da2 (`:221-222` [V]), which is 3 + 5da2 at ψ=1. So `da` is live on the walls, with a pole at da2 = -0.6, and must be fixed at certified values or optimized (Section 4.5).

The C++ change is small but more than one key [V]. `scalar_wave` has no source term (`rhs` at `scalar_wave.cpp:195-239`), so one Lua key must flip the sign in both `neg_G_at` and `solution_at` (`:28-40`), giving u = sin(2π(r - R + t)). `from_lua` already takes c and R from the first sphere when `system.center` and `radius` are omitted (`:149-161`), which puts c inside the sphere. `neg_G_at` divides by |x - c| at every location (`:75-77`), so the template loader refuses a c that lands on a grid node, which would give NaN. Dirichlet cut closures at general ψ need the diverging case with c inside the sphere, which is all inflow and needs no interpolation. The per-record BC stays the general answer for bodies with both inflow and outflow faces (Q9). The plan tries the converging template first, with a timebox (step 3).

**The parabolic case differs.** `heat.lua:19` uses a floating sphere. A parabolic problem needs a physical condition on every boundary, and "floating" supplies none, so that template analyzes a non-physical BC. A Dirichlet object exercises no R rows (`derivative.cpp:173-175`). The physical non-Dirichlet case is a Neumann object: the parser accepts it (`boundaries.cpp:30-31,81` [V]) and E2_2 declares it (`E2_2.cpp:24-25` [V]), but the assembly handles Neumann only on grid walls (`derivative.cpp:375,422` [V]). Posedness tags therefore cover parabolic systems too, and object Neumann assembly is a prerequisite of the ν=2 study (step 6).

### 4.5 Worked example A: optimize polyE2_1's interpolation with proxies and simulations

This runs in step 5, and only if step 4's go/no-go keeps interpolation.

```python
from pyshoccs import schemes, cases, tiers as T, Study, objectives as O, search

decl  = schemes.get("E2-poly")                  # polyE2_1: fa[6] da[3] ia[4], nu=1, status="generated"
train = cases.ensemble("research/sink_sphere_v1.lua", n=31, count=8, seed=7,
                       radius=(0.15, 0.30), center_offset=(0.0, 0.08))   # realised psi spread over (0, 1)
hold  = cases.ensemble("research/sink_sphere_v1.lua", n=61, count=4, seed=99,
                       radius=(0.10, 0.35), center_offset=(0.0, 0.10))
space = decl.space(train.bcs, fixed={"da": "scalar_wave.lua"})  # fa + ia free (10); da live on the walls,
                                                                # fixed after a T1.5 check at psi=1
ladder = T.Ladder(
    gates=[T.Poles(psi=decl.conv.psi_clamp),                       # T0, da included
           T.Oracle(classes=("closure", "R_axial", "R_transverse")),  # T1, consistency
           T.Observable(group="ia", min_rows=8)],                  # refuses ensembles where ia barely acts
    decide=T.Spectrum(map="rk4", metrics=("cfl_margin", "abscissa", "transient")),  # T2
    validate=[T.LinearRK4(t_final=10.0),                           # T3, forcing and data
              T.CppRun(threads=1, incumbents=5, random_fraction=0.02)],  # T4, code path
    refine=T.Refine(n=(31, 61, 121), fraction=0.1),                # N trend on a subset
)
study = Study("polyE2-sink-v1", decl, space, train, ladder,
              objective=O.maximize("T2.cfl_margin", reduce="min_over_cases"),  # path checked now
              constraints=[O.ge("T2.cfl_margin", 1.0), O.le("T2.transient", 10.0)],
              search=search.Staged(sobol=512, local="cmaes"), seed=7, store="studies/")
rec = study.run(budget=dict(evals=600, wall="4h", workers=4))
rec.report()                                     # best, per-tier confusion vs T2, disagreements, provenance
study.validate(rec.best(), hold)                 # T2 at N=61 and 121, T3 and T4 on held-out geometry
```

What happens underneath:

1. **Construction checks**, all before any evaluation:
   - every case's BCs are in `decl.supports`, and every wall is named;
   - the posedness tags show no Dirichlet on an outflow surface and no floating object under a parabolic system;
   - the template exercises the treatment (it refuses an all-Dirichlet object because of `derivative.cpp:173-175`);
   - `ia` is observable. It appears only in `interp_wall` (`polyE2_1.cpp:59-135` [V]); the interior interpolant is parameter-free (`:44-56` [V]). Only rows whose interpolation line meets a neighbouring R point or a wall depend on it (for E2_2 at heat51, 12 of 32 transverse rows [D, Part V Section 3]). The check counts rows with a nonzero ∂L/∂ia per case, reports the count, and refuses an ensemble below `min_rows`;
   - the fixed `da` passed T1.5 at ψ=1 and T0;
   - every metric path resolves against a tier's declared `metrics`;
   - H3 has a passing record for this registry hash.
2. **T0.** For each point, T0 evaluates the declared denominators on a ψ grid plus endpoints. polyE2_1's Dirichlet pole (da2 in (-0.6, -1/3) [R]) is checked even with `da` fixed, because the walls make it live.
3. **Export.** The Lua is emitted with split `floating_alpha`, `dirichlet_alpha` and `interpolant_alpha`. `shoccs-export --operator gradient` writes A_x and A_y and the coefficient fields `gG`, so Python never recomputes physics. The export is keyed by a content hash; with the slot export (Section 4.7) it runs once per geometry.
4. **Reduction.** `semi_discrete()` forms L = diag(gG_x) A_x + diag(gG_y) A_y on the unknowns: fluid D points and floating R points. Dirichlet wall columns are split off as forcing.
5. **T1** runs the oracle on the exported rows.
6. **T2** takes dense eigenvalues of L (about 1,000 unknowns at N=31 [E]) and reports `cfl_margin`, the largest multiple of the template Δt at which every Δtλ lies in the RK4 stability region (≥ 1 means stable at the template CFL); the spectral abscissa; and a transient bound, max over n ≤ n_T of ‖G^n‖₂ for G = R4(ΔtL), by power iteration. An eigenvalue with Re λ = 1e-3 needs t of about 1,000 to show in a simulation, so no window over t in [5, 10] can overrule T2 (Section 6).
7. **Refinement.** Every incumbent and 10 % of points also get T2 at N=61 and 121 (sparse shift-invert near the RK4 boundary at 121 [E]). A `cfl_margin` that falls with N flags a GKS-type instability and rejects the point.
8. **T3** integrates the same L with its forcing and checks t_reached = t_final. It validates the forcing split and the Dirichlet data, not stability.
9. **T4** runs the binary on incumbents and a random 2 % slice. A T3 and T4 Linf mismatch beyond 1e-10 relative halts the study as a defect.
10. **Store.** Every outcome is appended with provenance.

Cost [E]: export 0.1 to 0.2 s (once per geometry with the slot export) plus eigensolve 0.3 s plus T3 tens of ms, so at most about 0.5 s per case. Eight cases and 600 evaluations come to about 40 minutes serially and under 15 minutes on four performance cores, plus the N=121 subset. Step 1 measures the real figures.

**Files the researcher touches: 1** (the study script or notebook), plus the template once.

### 4.6 Worked example B: algebraic outflow extrapolation, no interpolation

The idea: at a floating (outflow) R point the value is not evolved. It is set algebraically from fluid values on its own grid line, u_R = Σ w_j(β) u_j. The weights blend the degree-2 Lagrange weights on 3 nodes with the degree-3 weights on 4 nodes through one parameter β. No derivative is formed at an R point, so no transverse row and no cross-line interpolant exist. The D-row closures (`fa`) still read u_R. The analyzed operator is M_eff = M_EE + M_ER C on the evolved set E. It is the cleanest test of H13 because it removes interpolation rather than rewriting it, and it is step 4's comparison arm.

```python
# file 1: python/pyshoccs/ideas/ray_extrap.py
import numpy as np
from pyshoccs.surgery import Treatment, Param, line_nodes, lagrange_at

class RayExtrap(Treatment):
    name, role = "ray-extrap", "algebraic"   # algebraic rows are eliminated: M_EE + M_ER C
    applies_to = {"floating"}                # Dirichlet records keep their data
    nu = {1}
    replaces = "interp"                      # removes group ia from the schema
    params = [Param("beta", "xa", -0.5, 0.5, frozenset({"floating"}))]
    exact_degree = {"R_value": 2}            # oracle gate: exact on x^0..x^2 along the line
    promotion = "python-only"                # T4 returns Unanalyzable, never runs the interpolating C++

    def constraint(self, tab, rec, beta):
        cols, x = line_nodes(tab, rec, k=4)  # raises ShortLine (record -> Unanalyzable) if < 4 fluid nodes
        x0 = tab.locations[rec.row, rec.dir]
        w3, w4 = lagrange_at(x[:3], x0), lagrange_at(x, x0)
        return cols, (1 - beta) * np.r_[w3, 0.0] + beta * w4
```

```python
# file 2: python/pyshoccs/ideas/test_ray_extrap.py
from pyshoccs import schemes, cases, oracle, tiers as T
from pyshoccs.ideas.ray_extrap import RayExtrap

DECL = schemes.get("E2-poly").with_treatment(RayExtrap())
CASE = cases.load("research/sink_sphere_v1.lua", n=31)

def test_exactness():                       # degree 2 passes; an orientation slip fails at x^1
    assert oracle.constraint(DECL, CASE, beta=0.3).max_degree_passed >= 2

def test_negative_control():                # mutation fixture: mirrored nodes must fail the oracle
    assert not oracle.constraint(DECL.mutate("reverse_nodes"), CASE, beta=0.3).passed

def test_short_line_refused():              # a line with 3 fluid nodes is Unanalyzable, not extrapolated
    assert T.Spectrum()(DECL, {"beta": 0.0}, cases.load("research/short_line_v1.lua", n=31)).unanalyzable

def test_discrimination():                  # T2 separates a stable and a T3-only unstable anchor
    T.Spectrum().assert_discriminates(DECL, CASE, stable={"beta": 0.0}, unstable="anchor:far-extrap")
```

**Bounded growth.** The weights are not small: at distance ψh the 3-node weights reach 3 at ψ=1 and the 4-node weights reach 4. Large weights are not a defect in themselves. After elimination, each derivative's D rows on a grid line read u_R only through values on that same line, so the treatment reduces exactly to a 1D boundary closure per line, and its stability is a GKS question in ψ. T1.5 answers it by applying the same surgery to the 1D C++ export over a continuous ψ grid; T2 checks the 2D coupling. (Section 3's objection to incremental's example B is its cross-line reads, not its weight size.)

The study is one call, `study.derive("ray-extrap-v1", decl=DECL, free=("fa", "xa"))`. The package supplies the schema with `ia` removed and `xa` added; poles (Lagrange weights at distinct nodes have none on (0, 1]); the oracle gate; the capability check; conformance tests; and results labelled `cpp_validated=False`, which `Study` refuses to call final. `line_nodes` resolves orientation from `ray_outside` and `solid_coord` on exported positions, never from ψ, so hazard B's frame and sign class cannot arise. The "far-extrap" anchor (high-degree extrapolation from distant nodes) is python-only, so only T2 and T3 label it.

**Files: 2 per idea, 0 C++, 0 Lua**, once step 4 has built what they assume: `line_nodes`, the elimination, `oracle.constraint`, the `mutate` fixture, `assert_discriminates`, and an export mode that omits R rows. Today the same idea costs about 15 files (Section 2).

Promotion if it wins [E]: once, 1 to 2 weeks, a C++ "algebraic R" feature (constraint records skip R rows and `update_boundary` fills them); then, per idea, a generated weights file, a registry entry, and an M_eff parity test at 1e-13. An idea that needs new unknowns (cell merging), nonlinear redistribution or geometry beyond the 1D ψ record (`src/mesh/mesh_types.hpp:31-38` [V]) is outside the lane and needs C++ geometry work in any design.

### 4.7 Closure rows, and whether interpolation is viable

**Closure-row role.** The owner's core research object is the closure itself. A new closure family, an E4 interpolating scheme or an SBP-SAT closure changes D rows. With only the algebraic role, that still needs derivation, codegen, a hand copy into `src/stencils`, the factory, `from_lua` and a rebuild. `Treatment` therefore gets a `closure` role: it replaces the D closure rows (and axial R rows) of declared records with rows evaluated from a lambdified sympy closure at each exported row's ψ, side and BC kind. That needs the export to record, per entry, which closure call and row filled it and, for an interpolated entry, which interpolant call multiplied it (the inputs of `add_interp_row`, `derivative.cpp:245` [V]). This is the "parameter-slot export" that the earlier draft ranked second because it is a value-level second implementation. Two checks make it a checked one: H3 (lambdified and compiled closures agree at 100 points), and a per-geometry fill check that Python-filled values at the shipped parameters equal the export to 1e-13 before any surgery. It also removes the per-point export subprocess, because a parameter change only refills values. A new family declares its rows, ν, exact degree and poles; T0 to T2 run on it unchanged; T4 returns `Unanalyzable` until promotion.

**Is interpolation viable?** Phase 1's finding is structural. A transverse row of a ν-th derivative scales interpolated cross-line values by h^{-ν}, so an interpolant of degree q leaves an O(h^{q+1-ν}) error in that row (Part V Section 3 shows the q=1, ν=2 case [D]). Order p in the row needs q ≥ p + ν - 1. Every shipped interpolant is linear: E2_2's (`E2_2.cpp:31-43` [V]), E2_1's (a line-for-line copy, `E2_1.cpp:38-90` against `E2_2.cpp:31-83` [V]) and polyE2_1's interior one (`polyE2_1.cpp:44-56` [V]); E4_1 has none (`E4_1.cpp:38` [V]). An E4 closure needs q of 3 to 5, so 4 to 6 cross-line points per row, and the wider footprint meets more neighbouring R points and walls, where the ψ-dependent wall interpolant applies. E2 results may therefore not transfer to E4. This is the key viability question, and step 4 asks it cheaply before any campaign: compare polyE2_1 with interpolation against the algebraic treatment by T2 margin, and run E4_1 with algebraic R, the one route that gets the paper's E4 closure onto a floating body without an interpolant.

## 5. Misuse resistance

| Hazard class (historical incident) | Mechanism | Test that proves it | Step |
|---|---|---|---|
| A. Two sources of truth (E4_1 mis-ported twice, about 16 weeks; E2_1 Python ≠ C++ now [R]; E2_1's interpolant copied from E2_2, `E2_1.cpp:38-90` [V]) | One operator, the export; no new assembler. H3 closure parity from C++ closure dumps against the lambdified sympy, and the fill check for the slot export. `status="legacy"` is refused by `Study`. A copied interpolant is tagged `copied` in the registry. | H3 at 100 points plus endpoints for polyE2_1 and E4_1. E2_1 is an xfail with a reason until Q1. The ν=1 transverse oracle on an E2_1 export records degree 1 against its `copied` tag. | 2 |
| A'. Regeneration drift | Numeric regeneration check (lambdified spec against compiled closure), not a byte diff. The incremental design reports that regenerated polyE2_1 text differs from the committed file [R]. | CI job compares numerically | 2 |
| B. Convention mismatch (wrong-sign metric in phases 29 to 31, r flip, h=1 weights, revneg mirror [R]) | Surgery and tiers use exported coordinates and the exported operator only. `Conventions` derives the mirror from ν wherever Python still mirrors. `semi_discrete()` is the only place a sign is applied. At promotion, the C++ orientation (the `right` flag and y shifts, `stencil.hpp:250,264` [V]) is covered by the M_eff parity test. | ν=2 mirror is reverse-only (fails today at `SG/reflection.py:941`). The uniform E4u export equals stencil_gen's dense uniform matrix entrywise to 1e-13. A mutation flipping `right` in the promoted C++ fails parity. | 0, 1, 7 |
| C. Silent config defaults (`copy_zero_padded`, missing `interpolant_alpha`, `order=2` dropping alpha, unknown keys, `max_tim`, unnamed walls defaulting to floating, `boundaries.cpp:44` [V], on which the shipped cut template relies, `brady_livescu_4_3_cut.lua:23-26`) | Strict `from_lua`: exact counts per scheme, unknown keys rejected, packed path deleted. A defaulted wall is logged at warn in C++ and refused by the pyshoccs loader. Python emits Lua only from `SchemeDecl.lua`. | One negative test each: short alpha, missing `interpolant_alpha`, `order=2` with alpha, `max_tim`, packed 13-vector, template without `xmax` | 0, 3 |
| D. Unsupported combination (E4u with shapes at +6.63 [R]; empty interp with a floating object; unsupported BC; object Neumann; exporter on non-heat) | C++ capability errors in `cut_discretization`, `bcs::from_lua` and the exporter. `decl.supports` is checked at `Study` construction. | One negative test per case | 0, 2 |
| E. Poles (E4_1 at α0=73/66; polyE2 da2, live on Dirichlet walls; the 197/288 "constraint" that was a mis-port pole) | T0 returns `Infeasible` from declared denominators before export. Codegen'd C++ checks throw at construction below 1e-12 (hand-written for E4_1). | Both named points return `Infeasible` with no NaN | 2 |
| F. Proxy measures something else (defective κ, L4/L5 parameter-independent, L3 under an L7 label, E4_1 Kreiss false positives, L3r over-rejection) | Tiers consume only exports. Admission (Section 6) needs a discrimination pair on the sink template, zero false accepts, and a measured cost. A tier without admission is diagnostic only. | Admission report per tier. L3r and the Kreiss sweep are admitted or excluded on the record. | 5 |
| G. Failure looks like a number (`return inf`, objective typo, 1e12 BO run, `gate_layer` forwarded as 3, early stop read as stable, `max_error` early exit, missing binary) | `Outcome` union. Objective paths validated at construction. Unclassified exceptions become `Error`, and the study aborts above an error-rate threshold. "Timestep too small" and the `max_error` exit (`simulation_cycle.cpp:79,132-137` [V], a null Linf at the time reached) exit nonzero. The runner checks `t_reached == t_final` and that the binary exists. The BO regression test (`tests/test_phs.py:2360-2400` [V]), which passes on the sentinel run, labels sentinel runs as failed. | Typo raises. Injected exception, injected early stop and a tiny `max_error` appear as `Error`. A sentinel BO record fails its regression test. | 0, 2 |
| H. Wrong problem (Dirichlet at an outflow cut end; default-floating far walls; floating object under heat) | Posedness tags on versioned templates, for hyperbolic and parabolic systems. The converging template is well-posed by construction. Ill-posed setups are refused unless `allow_ill_posed=True` is recorded. | The shipped cut template and the heat floating-sphere template are refused for a study | 3 |
| I. Latent numerics (h-independent ψ(1-ψ) error in ν=2 transverse R rows, Part V Section 3; E4_2 R rows never examined) | Oracle gated per row class and ν. A treatment whose declared exactness is below what its ν needs requires `accept_inexact=True`, recorded in provenance (`known_inexact` for E2_2 today). | Transverse ν=2 gate fails on E2_2 as a listed expected failure. Its removal is the step-6 success criterion. | 1, 6 |
| J. Tests that pin artifacts (24 tautological follow-ups; phase 29 assertions encoding the bug [R]) | Every fix lands with a test that fails on the parent commit. Every tier and treatment ships a negative control through a mutation fixture. Tests check invariants, not stored numbers. | Review rule plus the fixture. CI runs mutations on conformance tests. | 0 onward |
| K. No provenance (no SHA, binary hash or template; seedless Pareto names; unseeded ARPACK) | `Provenance` attached by the store, with a writer that raises on a missing field. Seeds go in keys. Append-only. A stored optimum is re-checked against the current registry hash on load. | Rerun from a record reproduces T0 to T3 bit for bit | 2 |

## 6. The multi-fidelity strategy

| Tier | What it measures | Role | Cost |
|---|---|---|---|
| T0 Poles | Denominators over the ψ clamp | Hard gate (feasibility) | ms [E] |
| T1 Oracle | Polynomial exactness per row class at declared degree | Hard gate (a failure is a bug, not a verdict) | ms [E] |
| T1.5 Closure spectrum (candidate) | Eigenvalues of the 1D C++ gradient exported over a continuous ψ grid (the `hyperbolic_eigenvalues` path), with surgery applied | Covers every ψ, not just the realised ones; locates failures. Admitted only if it agrees with T2. | 10 to 100 ms [E] |
| T2 Spectrum | `cfl_margin`, spectral abscissa and transient bound of the RK4 map of the BC-reduced exported operator; N=31, plus 61 and 121 on a subset | **Decision tier**, and the smooth margin for the driver | 0.3 to 0.5 s per case at N=31 [E]; minutes at N=121 [E] |
| T3 LinearRK4 | Forced integration of the same matrix to t_final | Validates the forcing split and Dirichlet data | tens of ms per case [E], scaled from 0.099 ms/step [D] |
| T4 C++ audit | The binary on the same template | Validates the code path | 1.6 to 3.4 s [R] |

**What each tier is for.** For a linear autonomous problem on fixed geometry, T2's spectrum of the fully discrete map decides asymptotic stability exactly, and its transient bound covers non-normal growth over the horizon of interest. A simulation window cannot overrule it. T3 and T4 earn their place by checking what the spectrum does not see: the forcing, the boundary data and the C++ code path. Cheaper tiers earn theirs by locating failures (which ψ, which R records) and by covering ψ continuously, which 2D geometries cannot. A fixed-N spectrum of a bounded non-normal hyperbolic operator can miss instabilities that only grow with N (the GKS type), so T2 runs an N sequence on a subset and verdicts use the trend.

**Validation of proxies against simulation (admission).** The calibration panel is built on the sink template with polyE2_1, so every anchor runs under the simulation's BCs (H11):

- **Stable anchor:** the shipped polyE2_1 values (`scalar_wave.lua:26-38` [V]), labelled by T2 at N=31, 61 and 121 and confirmed by a long T4 run. If they turn out unstable here, they become an unstable anchor.
- **Unstable anchors:** polyE2_1 points from a brute-force T4 scan (about 200 Sobol points at N=31, t=10) whose growth exceeds the existing 0.05 tolerance (`SG/brady2d_stability.py:1160-1166` [V]), confirmed by T2.
- **T3-only anchor:** the far-extrapolation treatment, which is python-only.
- **Excluded:** uniform schemes (refused with shapes after step 0), Dirichlet at outflow (refused by the posedness tags), and pole-adjacent points (a weak test, as the correctness judge said).
- **Random points:** at least 50 random feasible points.

A cheap tier (T1.5, the Kreiss sweep) is admitted as a gate when it shows zero false accepts against the T2 and T4 labels, an over-reject rate below a declared bound, and a measured cost. A tier admitted as a margin must also show a Spearman rank correlation with T2's `cfl_margin` above a declared value. Labels come from T2 and simulation together, never from two cheap proxies agreeing; exploration-first's panel had that circularity. T2 itself is checked on the anchors: a T2 verdict that T4 contradicts at a horizon long enough to show the mode is an export or reduction defect.

**Surfacing disagreement during a run.**

1. **Forced audit.** Every incumbent and a random 2 % slice go up the whole ladder whatever their rank, and 10 % of points rejected by a cheap tier get T2 anyway to estimate false rejects.
2. **Running confusion.** Each cheap tier keeps a confusion matrix against T2, with Wilson upper bounds on false-accept and over-reject rates. A false accept on an incumbent, or a bound crossing its threshold, demotes the tier to diagnostic for the rest of the study, as a record.
3. **Diagnosis by disagreement type:**

   | Disagreement | Meaning | What happens |
   |---|---|---|
   | T2 stable, T3 grows | Transient growth, or a forcing error | Compare with T2's transient bound. Growth beyond it is a forcing-split defect and halts the study. |
   | T2 unstable, T3 flat | A slow mode the window cannot show | T2 wins; the point is rejected. The horizon is not extended. |
   | `cfl_margin` falls with N | GKS-type instability | Point rejected and flagged |
   | T1.5 passes, T2 fails | A genuinely 2D mechanism | The eigenvector location is recorded |

4. **Parity break.** T3 against T4 beyond 1e-10 relative halts the study. That is an H4 break, not physics.
5. **Report gate.** `report()` refuses to name a best point while any disagreement is unacknowledged.

**Objective shape (default, pending Q3 and Q5).** Maximize the worst-case `cfl_margin` over the training ensemble, subject to `cfl_margin` ≥ 1, a transient bound, the T0 and T1 gates, and a non-falling N trend. Validate on a held-out ensemble with T2 at N=61 and 121, T3 and T4.

## 7. The survey roadmap and the phase-1 artifacts

| Item (`docs/PYTHON_DSL_SURVEY.md:132-146`) | Verdict | Reason |
|---|---|---|
| Phase-1 exporter, `tables.*`, `npy.hpp` | **Keep, generalize first** | The waist of this plan |
| pyshoccs `tables`, `oracle`, `parity`, `structure`, `mms`, `heat` | **Keep, extend** | The oracle found the ψ(1-ψ) defect; `heat.py`'s RK4 becomes T3 |
| Fused Kokkos kernels, `bench_tables` | **Freeze** | Research speed is bound by evaluation count and false proxies, not RHS microseconds |
| Owed phase-1 items: fixed-up C++ (b), 128^3, fp32/df64 | **Drop for this goal** | They serve runtime gates the goal does not have |
| Owed: transverse R numerics | **Promote** to the ν=2 study (step 6) | H12, and a direct test of interpolation for the second derivative |
| 2. Bindings | **Defer nanobind; the slot export comes first** | Export costs 0.04 to 0.17 s per case [D] |
| 3. Whole-step schedules, Lua retired | **Drop "Lua gone"** | Python generates Lua and C++ reads it strictly |
| 4. Metal, 6. CUDA, 7. Kernel codegen | **Drop** | No research need; small 2D ensembles parallelize across CPU processes |
| 5. Euler | **Defer** until the scalar_wave and heat ladder is calibrated | Its useful piece is a linearized-operator export for T2 |

## 8. Phased plan

| Step | Duration | Deliverables | Acceptance |
|---|---|---|---|
| **0. Stop silent failure** | 2 weeks | Delete the packed polyE2_1 path (`stencil.cpp:87`) and require split arrays. Strict `from_lua` (counts, unknown keys, `order=2` with alpha). Remove `copy_zero_padded` from parameter reads. C++ capability errors (empty interp with a non-Dirichlet object, unsupported BC, object Neumann, uniform scheme with shapes, exporter on non-heat). A warning for a defaulted wall. "Required timestep too small" and the `max_error` exit return nonzero. `cpp_bridge` gets a configurable binary path, pinned threads and a t_final check. `make_objective` stops swallowing exceptions. ν guard at `SG/reflection.py:941`. The BO max/argmin question at `SG/bo.py:964` is resolved and documented. | About 14 new tests, each failing on the parent commit and passing after. Updated in the same commits, as deliverables: `src/systems/hyperbolic_eigenvalues.t.cpp:47-49`, `src/operators/eigenvalue_visitor.t.cpp:101-103` and `eigenvalues.lua:38-40` gain `interpolant_alpha`; `tests/test_optimizer.py:242-256` expect typed errors instead of `inf`; `tests/test_phs.py:2360-2400` fails sentinel runs. ctest otherwise stays at 47/48 (`t-laplacian`). |
| **1. Truth layer and CI link** | 2 to 3 weeks | `shoccs-export --operator {gradient, laplacian}` chosen by system type, with scalar_wave coefficient fields. 1D export over a ψ grid (T1.5). `meta.json` gains alpha, dirty flag, binary hash and R checksum. `Operator.semi_discrete`. ν=1 oracle. Gated transverse oracle with an expected-failure list. CI job: build the exporter, export three cases, run pyshoccs. **Measure** first-derivative export time, dense eigensolve at N=31, 61 and 121, and scipy RK4 for scalar_wave at N=31. | The exported scalar_wave operator times u equals the C++ RHS to 1e-13 on fluid rows. The uniform E4u export equals stencil_gen's dense uniform matrix entrywise to 1e-13 (eigenvalues of non-normal operators are too ill-conditioned to compare). The CI job gates a PR. The costs are written into this part. |
| *Gate after step 1* | | If the export dominates per-point cost (above about 0.5 s), build the slot export (Section 4.7) in step 2, since H3 and the fill check make it a checked implementation. nanobind over assembly and export is the second choice. | Decision recorded |
| **2. Registry, outcomes, provenance, H3** | 2 to 3 weeks | `pyshoccs.schemes` for E2-poly and E4_1 (E2_1 per Q1). `Outcome`, `Provenance`, JSONL store. T0, and codegen'd C++ denominator checks that throw at construction (hand-written for E4_1). Closure dumps and the H3 harness. Registry against C++ factory set-equality test. | H3 passes at 100 points plus endpoints for polyE2_1 and E4_1. E2_1 is an xfail or fixed. The two pole points return `Infeasible` in Python and throw in C++. A typo raises. A rerun reproduces a record bit for bit. |
| **3. Well-posed templates and simulation tiers** | 3 weeks, timeboxed | Converging-field option in `scalar_wave::from_lua`, with the sign in `neg_G_at` and `solution_at` and a grid-node guard on c. `research/sink_sphere_v1.lua` with four named Dirichlet walls and a floating sphere. Posedness tags for hyperbolic and parabolic systems. The `ia` observability check. T3 `LinearRK4`. T4 `sim.Cpp`. Fallback if the timebox fails: per-record characteristic BC as its own project (Q9). | T3 and T4 Linf histories agree to 1e-10 relative on polyE2_1 on the sink template at N=31. An injected early stop is `Error`. The shipped cut template and the heat floating-sphere template are refused. The `ia` row count is reported per case. |
| **4. Viability go/no-go** | 2 weeks | Minimal surgery lane: the algebraic role, `line_nodes` with the `ShortLine` refusal, the elimination, `oracle.constraint`, `mutate`, `assert_discriminates`, and an export mode without R rows. On the sink ensemble at N=31 and 61, compare best T2 `cfl_margin` for polyE2_1 (256 Sobol points over fa and ia), polyE2_1 `fa` with `RayExtrap`, and E4_1 with `RayExtrap`. | A recorded decision: campaign A on interpolation, on algebraic R, or on both, with the degree-scaling question (Section 4.7) answered for E4 or listed as open. |
| **5. Calibration, Study, campaign A** | 3 weeks | Admission harness and the sink-template anchor panel. `Study` with forced audit, confusion tracking, demotion and report gate. Process pool pinned to one thread per worker. Campaign A (Section 4.5) on the step-4 winner. | Each cheap tier is admitted or excluded with a written confusion matrix and cost. The campaign finishes in budget with full provenance. The winner passes T2 at N=61 and 121, T3 and T4 on 4 held-out geometries. |
| **6. Closure-row role and ν=2 study** | 3 to 4 weeks | Object Neumann assembly for E2_2 (about 1 week [E]). The `closure` role and slot export if not built in step 2. A ν=2 study on a Neumann-sphere heat template comparing today's linear transverse interpolant, a quadratic one, and an algebraic transverse row. A decision on making t-laplacian gating. | A ν=2 treatment passes the transverse oracle where E2_2 fails. A closure-role idea runs T0 to T2 having touched only its own files. |
| **7. Promotion machinery (gated)** | 3 to 5 weeks, only when an idea wins | Cut-policy hook replacing the inline branch in `cut_discretization` (`derivative.cpp:159-257`). Algebraic-R support. Generated C++ count table from `SchemeDecl`. Revisit a Python reference assembler only if the surgery lane failed on two real ideas and Q6 says Python is the authority. | The promoted idea's C++ export equals the Python M_eff to 1e-13 in CI, and the `right`-flip mutation fails it. Full ctest and export parity on every shipped case pass. |

Steps 0 to 3 take about 9 to 11 weeks. The go/no-go lands at about weeks 11 to 13, campaign A at about weeks 14 to 16, and steps 0 to 6 take about 17 to 20 weeks. Step 0 alone also helps the existing stencil_gen loop.

**Startable today.** The first commit of step 0 needs no owner decision. Write a ctest that builds `E2-poly` from a packed `alpha` and asserts a hard error naming the scheme. Today the call is undefined behaviour for any length (`subspan(6, 9)` and `subspan(9, 13)` exceed a 13-element span) but reads out of range only for vectors shorter than 13, because `copy_zero_padded` truncates (`stencil.hpp:40-45`); a sanitizer catches the short case and only hardened span checks catch the 13-element one. Then delete the packed path in `stencil.cpp:85-87` in favour of that error. The rest of step 0 follows the same pattern.

## 9. Decisions needed from the owner

| # | Question | Default the plan assumes | Needed by |
|---|---|---|---|
| Q1 | E2_1: is the Python `derive_cut_cell_scheme(E2_1)` closure or `E2_1.cpp` the intended one? They differ by up to 7.0 at ψ=1 at the same α [R]. Is E2_1 still a target, or superseded by polyE2_1? | polyE2_1 is the target. E2_1 is legacy, carried as an H3 xfail. | Step 2 |
| Q2 | Is the immediate target first-derivative interpolation (polyE2_1's 13 parameters), with the ν=2 interpolant a consistency repair? Should E4-order interpolating schemes exist? | Campaign A is ν=1, on whichever treatment wins step 4. The ν=2 transverse rows are repaired for consistency in step 6, not optimized. No E4 interpolating scheme until the degree question (Section 4.7) is answered. | Step 4 |
| Q3 | What defines "stable": PDE, integrator, semi-discrete or fully discrete spectrum, horizon and N? | scalar_wave with RK4: the fully discrete spectrum at the template CFL, worst-case `cfl_margin` over the ensemble at N=31 with a non-falling trend over 61 and 121, and a transient bound. T3 and T4 validate. | Step 3 |
| Q4 | Which reference template? May the converging-field sink sphere with Dirichlet inflow walls replace the default-floating far walls? | Yes, with an 8-geometry training ensemble and a 4-geometry holdout | Step 3 |
| Q5 | Worst case over ψ in [1e-12, 1 - 1e-12] or [1e-4, 1 - 1e-4]? Is accuracy a constraint or a co-objective? | The C++ clamp at 1e-12. Accuracy is a constraint through the T1 oracle. | Step 2 |
| Q6 | Must C++ stay the authority, or may Python become the research reference with C++ generated? | C++ stays the authority. No Python reference assembler before the step-7 gate. | Step 7 |
| Q7 | Which non-interpolation families are in scope: extrapolation, SBP-SAT penalties, direct one-sided R rows, cell merging, state redistribution? | Linear treatments on the existing V and 1D ψ (the first three). Cell merging and redistribution are out of scope. | Step 4 |
| Q8 | Plan 49: do the L1 to L7 cascade, RBF families, NSGA-II and multi-fidelity BO stay, move to legacy, or get deleted? | Frozen as legacy, not deleted. Revisit after step 5's calibration. | Step 5 |
| Q9 | Are Neumann objects, or bodies with mixed inflow and outflow faces, needed? | Neumann objects: yes, for the ν=2 study. Mixed bodies: not for campaign A; the per-record BC becomes its own project only if needed. | Step 6 |
| Q10 | What wall-clock budget per campaign? Is GPU or multi-machine parallelism in scope? | Overnight on the M4's four performance cores, CPU only | Step 5 |
| Q11 | May `python/pyshoccs` depend on `scripts/stencil_gen` as a library? | Yes, one-way | Step 2 |

## 10. Risks and what would change the recommendation

| Risk | Signal | Response |
|---|---|---|
| Transient growth matters at practical horizons although the spectrum is stable | T2-stable points grow in T3 within T2's transient bound | Tighten the transient constraint, or make a pseudospectral margin the objective |
| Export latency or volume dominates | Step 1 above about 0.5 s per case; 600 evaluations over 8 cases is 4,800 exports of about 99 files and 0.69 MB each at heat51 [D, Part V Section 1], about 3.3 GB | Slot export (one export per geometry), then nanobind; the cache evicts exports after reduction |
| The converging template is harder than planned, or unrepresentative (no body with both inflow and outflow faces) | The step 3 timebox expires, or the owner rejects it (Q4) | Build the per-record characteristic BC as its own project. Campaign A slips by about 3 weeks [E]. |
| H3 reveals that polyE2_1 also mismatches | A step 2 failure | Stop campaign A until fixed. That is why H3 runs first. |
| Parity is self-referential | The export is faithful to C++ that is wrong in places (`t-laplacian` fails today) | Only the oracle and the template's exact solution are independent of the C++. Gate on them, not on parity alone. |
| Step 7 edits `cut_discretization`, which every scheme shares | Any promotion | Full ctest plus export parity on every shipped case before merge; budget a re-validation week [E] |
| H16's CI cost | Python in the `cpp` job needs `.github/workflows/` and devcontainer image changes, and a cold image build takes hours [D, CLAUDE.md] | Batch those protected-file changes once, in step 1; run Python against an exporter artifact from the `cpp` job |
| Owner time | A Q blocks its step (Section 9, last column) | Decide Q1, Q5 and Q11 before step 2, Q3 and Q4 before step 3, Q2 and Q7 before step 4 |
| The surgery lane cannot express the ideas that matter (new unknowns, multi-D geometry, nonlinear redistribution) | Q7, or a failed step-6 idea | C++ geometry work is needed in any design. Re-open the Python reference assembler question then, with Q6 answered. |
| Linear premise fails (Euler, nonlinear instability) | The owner moves to Euler | Linearized-operator export for T2. The binary becomes the decision tier. |
| Over-building before the science question is settled | Steps 0 to 2 slip past 8 weeks | Cut step 5's monitor to a forced audit plus a confusion table. Keep the gates and the step-4 go/no-go. |
| The bottleneck is derivation or ideas, not evaluation | The owner's view, or step 6 shows idea formulation dominates | Shift effort to the derivation layer: E4_2 symbolic derivation, a generator for E4_1, and the 330 s evaluator [D] |

**What would change the recommendation outright:**

- **Q6 answered "Python is the reference" and Q7 naming ideas that need multi-D geometry.** The exploration-first architecture becomes the target, still starting from steps 0 to 2 here.
- **Step 4 says interpolation loses.** Campaign A runs on algebraic R, H3 for polyE2_1 shrinks to its D closures, and the closure-role work of step 6 moves ahead of campaign A.
- **Cheap tiers shown to agree with T2 almost always.** The admission and disagreement machinery shrinks to a filter.

The evidence is thinnest at three points, and steps 1 and 4 are the earliest that can settle them:

- every cost figure marked [E];
- whether interpolation beats an algebraic R treatment on a cut geometry, at E2 and at E4 (L3 agreed with L8 on 6 of 6 cases, but on uniform closures only [R]);
- the well-posedness and sign details of the converging template.
