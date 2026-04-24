# Framework Architecture

Current state of `scripts/stencil_gen/` after plans 40–44. Organized by module; includes exact file paths and the most important public functions/classes.

## Dependency Flow

```
phs.py ─────────────────────────────────────────────────────────┐
temo.py ──────────────────────────────────────────────────────── │
codegen.py (depends on temo, phs) ──────────────────────────── │
group_velocity.py (depends on phs) ──────────────────────────── ▼
gks_kreiss.py ──────────────────────────────────────────────── brady2d_stability.py
non_normality.py ────────────────────────────────────────────── │
cpp_bridge.py ──────────────────────────────────────────────── │
                                                               ▼
                                                       optimizer.py
                                                               │
                                        ┌──────────────────────┤
                                        ▼                      ▼
                                sweeps/brady2d_sweep.py    sweeps/optimize.py
                                        │                      │
                                        └──────────┬───────────┘
                                                   ▼
                                           sweeps/__main__.py  (CLI dispatch)
                                           brady2d_cli.py  (standalone entrypoint)
```

---

## `stencil_gen/` Package

### `brady2d_stability.py` — the layered stability scorer

**Key public API:**

- `brady2d_stability_score(scheme, kernel, params, *, max_layer=7, short_circuit=True, layer8_N=31, layer8_t_final=10.0) -> StabilityReport`
- `StabilityReport` dataclass: `layer1..layer8: dict|None`, `layer_bl42: dict|None`, `non_normality: NonNormalityReport|None`, `kreiss: KreissResult|None`, `overall_verdict`, `failed_layer`, `failed_reason`, `compute_time`.

**Layer functions:**

| Function | Layer | What it tests |
|---|---|---|
| `layer1_interior_boundary_gv(scheme, kernel, params)` | L1 | Interior + boundary GV error vs `L1_TOL` |
| `layer2_kreiss_gks(scheme, kernel, params)` | L2 | Rigorous GKS Kreiss determinant — returns `KreissResult` |
| `layer3_1d_eigenvalue(scheme, kernel, params)` | L3 | 1D semi-discrete advection max-real-eigenvalue stability |
| `layer_bl42_reflecting_hyperbolic(scheme, kernel, params, n_values=(21,41,81))` | L3r | BL §4.2 coupled hyperbolic eigenvalue check (runs at max_layer≥3) |
| `layer4_local_gv_2d(scheme, kernel, params)` | L4 | 2D local GV with varying coefficients |
| `layer5_anisotropy(scheme, kernel, params)` | L5 | 2D anisotropy profile |
| `layer6_non_normality(scheme, kernel, params)` | L6 | Non-normality diagnostics on 1D operator |
| `layer7_sparse_2d_eigenvalue(scheme, kernel, params)` | L7 | Full sparse 2D operator eigenvalue (most expensive Python layer) |
| `layer7_with_non_normality(...)` | L7+nn | L7 extended with non-normality on 2D operator |
| `layer8_cpp_simulation(scheme, kernel, params, *, N, t_final)` | L8 | C++ simulation via `cpp_bridge.run_cpp_brady2d` |
| `build_bl42_operator(D) -> scipy.sparse.csr_matrix` | helper | BL §4.2 block operator from 1D D |
| `build_sparse_2d_operator(scheme, kernel, params) -> csr_matrix` | helper | 2D operator for L7 |

**Key constants:** `L1_TOL`, `L4_TOL`, `L5_TOL`, `L7_TOL = 0.1`, `BL42_TOL = 1e-10`, `L8_FINAL_LINF_TOL = 1.0`.

**Imports:** `gks_kreiss`, `group_velocity`, `non_normality`, `phs`, `cpp_bridge`, `benchmarks.brady_livescu_2d`, `benchmarks.brady_livescu_4_2`.

---

### `gks_kreiss.py` — rigorous Kreiss determinant

- `KreissResult` dataclass: `is_stable`, `witness_s`, `witness_sigma_min`, `imaginary_axis_perturbation_verdict`.
- `DefectiveKappaError(RuntimeError)`.
- `kreiss_stability_check(interior_row, boundary_rows, *, grid, refine) -> KreissResult`.
- Helpers: `kappa_roots`, `kreiss_matrix`, `min_singular_value`, `make_s_grid`, `_sweep_grid`, `_refine_witness`, `_classify_imag_axis`.

---

### `non_normality.py` — transient-growth diagnostics

- `NonNormalityReport` dataclass: `spectral_abscissa`, `numerical_abscissa`, `henrici_departure`, `eigenvector_condition`, `pseudospectral_abscissa`, `kreiss_constant`, `transient_growth_bound`.
- `compute_non_normality(L, s_grid) -> NonNormalityReport` (orchestrator).
- Standalone functions: `spectral_abscissa_sparse(L, k=20, shift_invert=True)`, `numerical_abscissa_sparse(L)`, `henrici_departure(L)`, `eigenvector_condition(L, small_dense_threshold=900)`, `pseudospectral_abscissa_estimate(L, s_grid, epsilon)`, `kreiss_constant_estimate(L, s_grid)`.

---

### `optimizer.py` — optimization framework

- `OptimizeResult` frozen dataclass: `best_params`, `best_x`, `best_objective`, `best_report`, `method`, `converged`, `n_evals`, `compute_time`, `history`, `extras`.
- `DEFAULT_BOUNDS: dict[tuple[str,str], list[tuple[float,float]]]`:
  - `("E2"/"E4", "tension")`: `[(0.5, 20.0)]`
  - `("E2"/"E4", "gaussian"/"multiquadric")`: `[(0.1, 5.0)]`
  - `("E4", "classical")`: `[(-2.0, 2.0), (0.05, 2.0)]`
- `params_from_vector(kernel, x) -> dict`, `vector_from_params(kernel, params) -> np.ndarray`.
- `extract_field(report, dotted_path) -> float` — e.g. `"layer3.max_stab_eig"`, `"layer_bl42.max_spectral_abscissa"`, `"layer6.transient_growth_bound"`.
- `make_objective(scheme, kernel, report_field, *, gate_layer, max_layer) -> Callable[[np.ndarray], float]` — returns `+inf` on gate failure.
- `run_scipy_local(f, x0, bounds, method, ...) -> OptimizeResult` — Nelder-Mead / COBYQA.
- `multi_start_optimize(scheme, kernel, report_field, bounds, *, n_restarts, seed, ...) -> OptimizeResult`.
- `run_scipy_shgo(f, bounds, ...)`, `run_scipy_de(f, bounds, ...)`.
- `run_staged_optimize(scheme, kernel, report_field, bounds, *, inner_gate=3, inner_max_layer=3, validator_max_layer=6, top_k=5, method, n_restarts, seed, max_evals) -> OptimizeResult` — cheap inner + top-k validation.

**`_FIELD_LAYER_ALIAS`**: maps non-numeric layer prefixes (e.g., `"layer_bl42"`, `"kreiss"`) to their numeric tier for `make_objective` to infer `max_layer`.

---

### `cpp_bridge.py` — Python→C++ subprocess bridge

- `SHOCCS_BINARY: Path = repo_root / "build/src/app/shoccs"`.
- `BRADY_LIVESCU_TEMPLATE: Path = repo_root / "lua-configs/brady_livescu_4_3.lua"`.
- `BridgeResult` dataclass: `final_linf`, `linf_trace`, `t_trace`, `stable`, `wall_time_s`, `exit_code`, `stderr`.
- `make_brady2d_lua(scheme_type, params, *, N, t_final, template) -> str` — substitutes `--{{N}}--`, `--{{T_FINAL}}--`, `--{{SCHEME_TABLE}}--` markers.
- `run_cpp_brady2d(scheme_type, params, *, N=31, t_final=10.0, timeout=300.0, ...) -> BridgeResult` — writes temp Lua, runs `shoccs` in isolated tempdir, parses `logs/system.csv`.
- `_scheme_table_for(scheme_type, params) -> str` — maps Python scheme/params to Lua table fragment.

---

### `brady2d_cli.py` — standalone CLI

`python -m stencil_gen.brady2d_cli` with flags:
- `--scheme {E2,E4}`
- `--kernel {classical,tension,gaussian,multiquadric,phs}`
- `--sigma`, `--epsilon`, `--alpha`
- `--max-layer`, `--short-circuit/--no-short-circuit`, `--json-output`
- `--run-calibration [--update-known-values]` → writes to `known_values.json["brady2d_calibration"]`
- `--alpha-basin-survey [--n-seeds N]` → multi-seed E4 classical-α basin study

---

### Benchmarks (`stencil_gen/benchmarks/`)

- `brady_livescu_2d.py` — BL §4.3 data: `psi`, `c_x/y`, `exact_solution`, `make_coefficient_field(N)`, `L_DOMAIN = sqrt(2)`.
- `brady_livescu_4_2.py` — BL §4.2 data: `initial_u`, `initial_v`, `exact_solution(x, t)`, `continuous_eigenvalues(k_max)` returning `±i(2k-1)π/2`.
- `brady2d_calibration.py` — `FAMILIES`, `run_calibration(max_layer)`, `format_calibration_table`.
- `alpha_basin_survey.py` — `run_survey`, `format_survey_table` for multi-seed E4 classical-α basin study.

---

### Pre-existing modules (one-line purpose)

| Module | Purpose |
|---|---|
| `phs.py` | RBF/PHS stencil weights (`phs_stencil_weights`, `build_diff_matrix_rbf`, cut_cell_weights); supports kernels: `tension`, `gaussian`, `multiquadric`, `phs` |
| `temo.py` | SymPy-based cut-cell stencil derivation (`derive_cut_cell_mathematica`, `Dimensions`, `SchemeParams`) |
| `group_velocity.py` | Modified wavenumber, GV, anisotropy, GKS boundary GV, cut-cell GV, ray tracing |
| `codegen.py` | Emits C++ stencil `.cpp` from SymPy (`generate_stencil_cpp`, `StencilGenSpec`) |
| `printer.py` | `build_symbol_map(scalar_params=)` — SymPy → C++ name mapping |

---

## `sweeps/` Package — CLI dispatch via `python -m sweeps <subcmd>`

### `__main__.py` dispatch table

| Subcommand | Module | Purpose |
|---|---|---|
| `epsilon` | `epsilon_sweep.py` | Gaussian/MQ ε sweep vs. stability |
| `tension` | `tension_sweep.py` | Tension σ sweep |
| `tension-penalty` | `tension_penalty_sweep.py` | (σ, γ) joint sweep |
| `footprint` | `footprint_sweep.py` | Stencil footprint (nextra) sweep |
| `comparison` | `comparison.py` | Multi-method comparison table |
| `alpha` | `alpha_extraction.py` | Extract boundary α from optimal ε |
| `mixed-epsilon` | `mixed_epsilon_sweep.py` | Per-row ε coordinate descent |
| `gv-stability-pareto` | `gv_stability_pareto.py` | Pareto front: GV error vs. stability eigenvalue |
| `brady2d` | `brady2d_sweep.py` | BL2D layered sweep with optional `--validate-with-cpp` |
| `optimize` | `optimize.py` | Optimization CLI (Nelder-Mead/COBYQA/SHGO/DE/staged) |
| `all` | `__main__._run_all` | Run all sweeps sequentially; `--quick` reduces resolution |

### `brady2d_sweep.py`
- `SweepPoint` dataclass: `kernel`, `param_value`, `report`, `passed`.
- `run_brady2d_sweep(scheme, kernel, *, param_range, max_layer, ...) -> list[SweepPoint]`.
- `rank_for_l8(points, *, max_layer) -> list[SweepPoint]`.
- `--validate-with-cpp`: re-runs survivors at `max_layer=8`.
- `--persist`: writes `known_values.json["brady2d_sweep"][scheme][kernel]` (not yet populated).

### `optimize.py`
- `main(argv)` parses `--scheme`, `--kernel`, `--method`, `--bounds`, `--gate-layer`, `--max-layer`, `--validator-max-layer`, `--top-k`, `--n-restarts`, `--validate-with-cpp`, `--update-known-values`, etc.
- Delegates to `optimizer.run_staged_optimize`, `multi_start_optimize`, `run_scipy_shgo`, `run_scipy_de`.
- Persists to `known_values.json["brady2d_optima"][scheme][kernel]` when `--update-known-values`.

### Other sweep modules
- `gv_objectives.py` — `interior_gv_error_max`, `boundary_gv_error_max`, `cutcell_gv_min_C`, `gv_score_from_matrix`, `print_gks_advisory`.
- `_common.py` — `load_known_values()`, `save_known_values()`, `SCHEME_PARAMS`, `print_table`.

---

## `docs/` Reference Documents

| File | Covers |
|---|---|
| `brady2d_stability_reference.md` | Full L1–L8 layer specification: problem, thresholds, `StabilityReport`, `BridgeResult`, cost knobs |
| `bl42_reference.md` | BL §4.2 L3r layer: operator construction, API, cascade position |
| `optimization_reference.md` | Optimizer architecture, bounds, `OptimizeResult`, drivers, staged pipeline, CLI |
| `group_velocity_reference.md` | `group_velocity.py` API: modified wavenumber, GKS, cut-cell GV, anisotropy, ray tracing |
| `pipeline_reference.md` | End-to-end stencil derivation pipeline: SymPy → `temo` → `codegen` → C++ |
| `sweeps_reference.md` | Sweeps package overview, subcommands, `known_values.json` schema |
| `testing_reference.md` | Test organization, fixtures, `test_phs.py` regression tests |

`/workspace/docs/brady2d_cpp_bridge_reference.md` documents the bridge specifically (outside the stencil_gen/docs subdir).

---

## C++ Side (`/workspace/src/stencils/`)

### Runtime-parameterized stencil structs

| Struct | Parameters | File |
|---|---|---|
| `E4_1` | `alpha[2]` (α[0] free, α[1] ≥ 197/288) | `E4_1.cpp` |
| `E6u_1` | `alpha[5]` | `E6u_1.cpp` |
| `tension_E4u_1` | `sigma: real` | `tension_E4u_1.cpp` |
| `gaussian_E4u_1` | `epsilon: real` | `gaussian_E4u_1.cpp` |
| `multiquadric_E4u_1` | `epsilon: real` | `multiquadric_E4u_1.cpp` |
| `polyE2_1` | `floating_alpha[6]`, `dirichlet_alpha[9]`, `interpolant_alpha[13]` | `polyE2_1.cpp` |

Fixed (non-parameterized): `E2_1`, `E2_2`, `E4_2`, `E4u_1`, `E8u_1`.

### `stencil::from_lua` dispatch (`stencil.cpp`)

Reads `scheme.order` and `scheme.type`:

```
order=2, type="E2"               → make_E2_1(alpha)
order=4, type="E2"               → make_E2_2(alpha)
order=4, type="E4"               → make_E4_1(alpha)
order=4, type="E4u"              → make_E4u_1(alpha)
order=6, type="E6u"              → make_E6u_1(alpha)
order=8, type="E8u"              → make_E8u_1(alpha)
any,     type="tension_E4u"      → make_tension_E4u_1(sigma)       (default 3.0)
any,     type="gaussian_E4u"     → make_gaussian_E4u_1(epsilon)    (default 0.9)
any,     type="multiquadric_E4u" → make_multiquadric_E4u_1(epsilon) (default 1.0)
any,     type="E2-poly"          → make_polyE2_1(floating, dirichlet, interpolant)
```

Unknown type: logs `"scheme.order/type not recognized"` and returns `std::nullopt`.

---

## Lua Configs (`/workspace/lua-configs/`)

| File | Purpose |
|---|---|
| `brady_livescu_4_3.lua` | **Template** with `--{{N}}--`, `--{{T_FINAL}}--`, `--{{SCHEME_TABLE}}--` markers; NOT directly runnable |
| `brady_livescu_4_3_n61.lua` | Standalone BL §4.3 run at N=61 |
| `brady_livescu_4_3_long.lua` | Standalone BL §4.3 long run (higher `max_time`) |

---

## `known_values.json` Schema

**Path:** `scripts/stencil_gen/sweeps/known_values.json`.

Top-level keys:

| Key | Content |
|---|---|
| `E2_1`, `E4_1` | Optimal kernel params — `params` (p,q,nextra,nu), `tension`, `gaussian`, `multiquadric`, `phs_k2`. `E4_1` also has `known_unstable` |
| `footprint` | `E4_nextra{0,1,2}_{phs,tension_3}` entries |
| `brady2d_calibration` | Full `StabilityReport` serializations for 9 reference (scheme, kernel) pairs — see `scientific_findings.md` for current pass/fail status |
| `brady2d_sweep`, `brady2d_optima` | Written by `--persist`/`--update-known-values`; not yet populated |

---

## Python environment — freshly rebuilt

The devcontainer was just rebuilt with two new Python deps:

- `pymoo>=0.6` — required by plan 45 (NSGA-II multi-objective).
- `nlopt>=2.7` — optional; provides BOBYQA as alternative to scipy COBYQA. Needs `swig` as system dep (now in Dockerfile apt-install).

Both are listed in `scripts/stencil_gen/pyproject.toml`. Run `cd scripts/stencil_gen && uv sync` if the project venv is stale.
