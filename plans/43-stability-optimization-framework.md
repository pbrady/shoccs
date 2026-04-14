# Phase 43: Stability Optimization Framework

**Goal:** Turn the existing sweep framework from brute-force grid search into actual optimization. Wrap `brady2d_stability_score` as an objective function for off-the-shelf optimizers (scipy.optimize), support single-objective + feasibility-cliff constraints, random-restart for multi-basin landscapes, SHGO and DE for global coverage, and staged cheap-layer-inner + expensive-layer-validator pipelines. Persist discovered optima to `known_values.json["brady2d_optima"]` and regression-test them. Final validation runs winners through the C++ solver via the plan-42 L8 bridge.

**Depends on:** Phase 41 (analytical stack; `brady2d_stability_score`, `StabilityReport`) and Phase 42 (C++ bridge for L8 validation).

**What this plan does NOT do** (explicit, to keep scope tight):

- **Multi-objective Pareto optimization** (pymoo NSGA-II). Single-objective plus feasibility cliff covers the working-out-which-parameter-is-best question cleanly; multi-objective is a separate infrastructure and output schema. Defer to a follow-up plan (44). The architecture here supports weighted scalarization as a bridge path.
- **Multi-fidelity Bayesian optimization** (BoTorch / Emukit). We use the built-in layered short-circuit as a manual cascade (staged cheap inner → expensive validator). Defer BO to a follow-up plan (45).
- **Brady-Livescu 1D Euler reproduction**. Their 2019 paper's objective requires a full nonlinear 1D Euler RK4 solver in Python (not present in this repo) and returns a two-phase blow-up-or-monotonicity score. Reproducing exactly is explicitly *not a target* — the paper notes their published α's are "simply the first entries in the databases" and the procedure is known to be multi-modal (Table 4 reports 101 found E4 schemes from random restarts). Defer to plan 46.
- **Classical-α E2_1 (4D)**. The user noted second-order stability is inconsequential. Skip in favor of E4_1 (2D, single hard inequality `α₁ ≥ 197/288`).
- **E6 / E8 classical schemes**. No Python derivation pipeline exists for them; out of scope.
- **NLopt dependency**. `pip install nlopt` fails in this container; the container would need a spack rebuild. Skip — use `scipy.optimize.minimize(method="COBYQA")` which matches BOBYQA's derivative-free trust-region design and is built-in (scipy ≥ 1.14; this repo has 1.17).
- **Tension-penalty and mixed-epsilon kernels through the layered cascade** (resolved at 43.1d, option b). `brady2d_stability_score` and every layer helper dispatch only `kernel ∈ {"classical", "tension", "gaussian", "multiquadric"}`; routing `"tension-penalty"`/`"mixed-epsilon"` through the 2D eigenvalue / non-normality / sparse paths would require a substantial extension to the Brady-Livescu pipeline for marginal optimizer reach — those families already have standalone exploratory sweeps (`sweeps/tension_penalty_sweep.py`, `sweeps/mixed_epsilon_sweep.py`). Defer the extension to a follow-up plan. `DEFAULT_BOUNDS` and `params_from_vector`/`vector_from_params` are pruned to the four supported kernels.

**Approach — the layered pipeline as a manual cascade:**

```
            fast (sub-ms)             slower (seconds)          minutes
candidate → L1 L2 L3 feasibility → L4 L5 L6 L7 metrics → top-k → L8 C++ sim
           short-circuit on fail     re-evaluate survivors     validate winner
           cheap inner loop          staged outer loop          single final check
```

- **Inner objective**: `f(x) = brady2d_stability_score(..., gate_layer=3, max_layer=3).extract(field)`. Returns `+inf` if L1/L2/L3 fails; otherwise a finite scalar. Each call is ~tens of ms. An inner optimizer can do thousands of evaluations comfortably.
- **Outer validation**: take the top-k survivors, re-evaluate at `max_layer=6` or `max_layer=7` (seconds each). Pick the best.
- **Final validation (optional)**: run L8 (~minutes) on the final winner to confirm the C++ simulation agrees.

**Parameter spaces in scope:**

| Family | Scheme | Dim | Bounds | Constraints |
|---|---|---|---|---|
| Tension | E2_1, E4_1 | 1 | σ ∈ [0.5, 20] | none beyond stability |
| Gaussian | E2_1, E4_1 | 1 | ε ∈ [0.1, 5] (log) | none beyond stability |
| Multiquadric | E2_1, E4_1 | 1 | ε ∈ [0.1, 5] (log) | none beyond stability |
| Classical-α | E4_1 | 2 | α₀ ∈ [-2, 2], α₁ ∈ [197/288, 2] | α₁ ≥ 197/288 hard |

(Tension-penalty and mixed-ε are out of scope — see "What this plan does NOT do".)

**Algorithm stack** (all in scipy; no new dependencies):

| Method | Use | scipy call |
|---|---|---|
| Nelder-Mead | 1-2D local refine from a good starting point | `minimize(method="Nelder-Mead")` |
| COBYQA | derivative-free trust region, 1-6D; better near cliffs than NM | `minimize(method="COBYQA")` |
| SHGO | global, 1-4D, deterministic simplicial homology | `shgo(f, bounds)` |
| differential_evolution | global, 4-6D, population-based | `differential_evolution(f, bounds)` |
| multi-start wrapper | any of the above × Sobol-sequence seeds | plan-local utility |

**Read first:**

- `plans/41-brady-livescu-2d-analytical-stability.md` (layer definitions, `StabilityReport` schema)
- `plans/42-cpp-bridge-runtime-parameterized-stencils.md` (L8 bridge)
- `scripts/stencil_gen/stencil_gen/brady2d_stability.py` (`brady2d_stability_score`, `StabilityReport`, `_SCHEME_PARAMS`, layer functions)
- `scripts/stencil_gen/sweeps/brady2d_sweep.py` (`_params_for`, `_report_to_dict`, `rank_for_l8` — all candidates for reuse)
- `scripts/stencil_gen/sweeps/__main__.py` (CLI dispatch pattern)
- `scripts/stencil_gen/sweeps/_common.py` (`load_known_values`, `save_known_values`)
- `src/stencils/E4_1.cpp` lines 35–41 (the α₁ ≥ 197/288 hard constraint — confirm this is in the Python `_build_classical_diff_matrix` too, or is enforced externally by the optimizer)

**Test commands:**

```bash
# Fast: the optimizer module in isolation
cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q

# CLI smoke: optimize tension E4 GV error at a tiny budget
cd scripts/stencil_gen && uv run python -m sweeps optimize \
    --scheme E4 --kernel tension \
    --objective layer1.boundary_gv_err \
    --gate-layer 3 --bounds 0.5 20 \
    --method Nelder-Mead --max-evals 40

# Regression suite still green
cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegression"
```

---

## Items

### 43.1 — Optimizer module skeleton and primitives

- [x] **43.1a** Create `scripts/stencil_gen/stencil_gen/optimizer.py` with:
  - Module docstring citing the layered cascade approach and linking to plan 43.
  - `@dataclass(frozen=True) class OptimizeResult` with fields: `best_params: dict`, `best_x: np.ndarray`, `best_objective: float`, `best_report: dict` (serialized `StabilityReport`), `method: str`, `converged: bool`, `n_evals: int`, `compute_time: float`, `history: list[tuple[np.ndarray, float]]`.
  - Constants: `DEFAULT_BOUNDS` dict mapping `(scheme, kernel) -> list[tuple[float, float]]` with the bounds table above. E4 classical α has `[(-2.0, 2.0), (197.0/288.0, 2.0)]`.
  - All function stubs raise `NotImplementedError`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.optimizer import OptimizeResult, DEFAULT_BOUNDS; print(list(DEFAULT_BOUNDS.keys())[:3])"`

- [x] **43.1b** Implement `params_from_vector(kernel: str, x: np.ndarray) -> dict` and `vector_from_params(kernel: str, params: dict) -> np.ndarray`:
  - `kernel="tension"` / `"gaussian"` / `"multiquadric"`: 1D, `x=[σ]` or `x=[ε]` → `{"sigma": x[0]}` or `{"epsilon": x[0]}`.
  - `kernel="tension-penalty"`: 2D `x=[σ, γ]` → `{"sigma": x[0], "gamma": x[1]}`.
  - `kernel="mixed-epsilon"`: variable-dim `x=[ε₀, ε₁, ...]` → `{"epsilons": list(x)}`.
  - `kernel="classical"`: 2D `x=[α₀, α₁]` → `{"alpha": [float(x[0]), float(x[1])]}`.
  - Inverse functions mirror exactly. Add a round-trip test: `params_from_vector(k, vector_from_params(k, p)) == p` for each kernel.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestParamsVector"`

- [x] **43.1c** Implement `extract_field(report: StabilityReport, dotted_path: str) -> float`:
  - Supports `layer1.boundary_gv_err`, `layer3.max_stab_eig`, `layer6.spectral_abscissa`, `layer6.kreiss_constant`, `layer6.transient_growth_bound`, `layer7.max_spectral_abscissa`, `kreiss.witness_sigma_min`, etc.
  - Uses `operator.attrgetter` for the first segment and dict `[key]` for the remainder.
  - Returns `float("inf")` if any segment is missing (e.g., layer not run).
  - Tests: for a populated StabilityReport, `extract_field(r, "layer1.boundary_gv_err")` returns the expected number; for `"layer99.foo"` returns +inf.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestExtractField"`

### 43.1d — Prerequisite: reconcile kernel scope with `brady2d_stability_score`

- [x] **43.1d** Resolved as **option (b)**: narrow the 43.2–43.10 scope to the four kernels that `brady2d_stability_score` already routes (`classical`, `tension`, `gaussian`, `multiquadric`). `tension-penalty` and `mixed-epsilon` are handled by standalone sweeps that bypass the layered cascade; extending every layer helper to a fifth/sixth kernel is disproportionate to the optimizer's reach. Pruned `("E4", "tension-penalty")` and `("E4", "mixed-epsilon")` from `DEFAULT_BOUNDS`; removed the corresponding branches in `params_from_vector` / `vector_from_params`; added a `test_pruned_kernels_rejected` parametrization that confirms those names now raise `ValueError("unknown kernel")`; noted the deferral in the plan's "What this plan does NOT do" and removed the two rows from the parameter-spaces table.
  - Files: `scripts/stencil_gen/stencil_gen/optimizer.py`, `scripts/stencil_gen/tests/test_optimizer.py`, `plans/43-stability-optimization-framework.md`.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q` — green, 32 tests.

### 43.2 — Objective factory with feasibility cliff

- [ ] **43.2a** Implement `make_objective(scheme, kernel, report_field, *, gate_layer=3, max_layer=None) -> Callable[[np.ndarray], float]`:
  - `max_layer` defaults to the layer implied by `report_field` (e.g., `layer6.*` → `max_layer=6`). If less than `gate_layer`, raise `ValueError`.
  - Returned function: converts `x` via `params_from_vector`, calls `brady2d_stability_score(scheme, kernel, params, max_layer=max_layer, short_circuit=True)`, checks `report.failed_layer is None or report.failed_layer > gate_layer`. If gate failed, return `+inf`. Otherwise return `extract_field(report, report_field)`.
  - Wraps `brady2d_stability_score` in `try/except Exception` returning `+inf` (ill-conditioned RBF systems at extreme parameters can raise).
  - Memoizes nothing — each call is fresh.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestMakeObjective"`

- [ ] **43.2b** Tests for `make_objective`:
  - `test_objective_returns_finite_on_feasible` — E4 tension σ=3.0 passes L1-L3, objective returns finite `layer1.boundary_gv_err`.
  - `test_objective_returns_inf_on_gate_failure` — deliberately-bad parameters (e.g., E4 Gaussian ε=0.01) fail L3, objective returns `+inf`.
  - `test_objective_catches_exception` — monkey-patch to raise, verify `+inf` returned.
  - `test_objective_raises_on_bad_field` — `"layer99.foo"` at `gate_layer=3, max_layer=3` returns +inf without error.
  - File: `scripts/stencil_gen/tests/test_optimizer.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestMakeObjective"`

### 43.3 — Baseline local optimizer: Nelder-Mead and COBYQA

- [ ] **43.3a** Implement `run_scipy_local(f, x0, bounds, *, method="Nelder-Mead", max_evals=200, tol=1e-6) -> OptimizeResult`:
  - Wraps `scipy.optimize.minimize`.
  - For `method="Nelder-Mead"`: translates bounds via `options={"xatol": tol, "fatol": tol, "maxfev": max_evals, "bounds": bounds}`.
  - For `method="COBYQA"`: uses `constraints` or `bounds` per the method's API. COBYQA takes `bounds` directly in scipy ≥ 1.14.
  - Records `history` via a callback that appends `(x.copy(), fval)` to a list.
  - Checks `result.success`; sets `converged = result.success and np.isfinite(result.fun)`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestRunScipyLocal and Nelder"`

- [ ] **43.3b** Add COBYQA version gate and tests:
  - At import time, set `_COBYQA_AVAILABLE = "COBYQA" in` the result of a quick probe (`scipy.optimize.minimize` with `method="COBYQA"` on a 1-var identity; if it raises `ValueError`, not available).
  - If `method="COBYQA"` is requested but unavailable, raise `RuntimeError("COBYQA requires scipy >= 1.14; got {version}")`.
  - Test: COBYQA on tension E4 converges to within 5% of the known σ=3.0 optimum.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`, `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestRunScipyLocal and COBYQA"`

### 43.4 — Multi-start wrapper

- [ ] **43.4a** Implement `multi_start_optimize(f, bounds, n_restarts=10, *, method="Nelder-Mead", seed=0) -> OptimizeResult`:
  - Generates `n_restarts` starting points via `scipy.stats.qmc.Sobol(d=len(bounds), seed=seed)` scaled to the bounds.
  - For each starting point, runs `run_scipy_local(f, x0, bounds, method=method)`.
  - Aggregates: returns the `OptimizeResult` with the smallest finite `best_objective` across restarts, with `history` concatenated from all runs and `n_evals` summed.
  - If all restarts return `+inf` (fully infeasible region), returns the last result with `converged=False`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestMultiStart"`

- [ ] **43.4b** Tests `TestMultiStart`:
  - `test_multi_start_finds_known_optimum` — tension E4 with `n_restarts=5`, expects `best_x[0]` within 1.0 of σ=3.0.
  - `test_multi_start_deterministic` — same seed produces same result across two calls.
  - `test_multi_start_handles_fully_infeasible` — set bounds entirely in known-unstable region, verify `converged=False` returned gracefully.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestMultiStart"`

### 43.5 — Global optimizers: SHGO and DE

- [ ] **43.5a** Implement `run_scipy_shgo(f, bounds, *, n=100, iters=3) -> OptimizeResult`:
  - Wraps `scipy.optimize.shgo(f, bounds, n=n, iters=iters, minimizer_kwargs={"method": "Nelder-Mead"})`.
  - Post-processes `result.xl`/`result.funl` (all local minima found) — picks the global minimum plus records the count of distinct local minima.
  - Adds to `OptimizeResult`: extra field `n_local_minima` (via `history` aux or a new field in an `extras: dict` field).
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestSHGO"`

- [ ] **43.5b** Implement `run_scipy_de(f, bounds, *, popsize=15, maxiter=100, seed=0, strategy="best1bin") -> OptimizeResult`:
  - Wraps `scipy.optimize.differential_evolution(f, bounds, popsize=popsize, maxiter=maxiter, seed=seed, strategy=strategy, tol=1e-7, init="sobol", polish=True)`.
  - Records `result.nfev` as `n_evals`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestDE"`

- [ ] **43.5c** Integration tests `TestGlobalOptimizers`:
  - `test_shgo_finds_tension_optimum` — 1D tension E4, bounds [0.5, 20], SHGO converges within 5% of σ=3.0 and reports `n_local_minima >= 1`.
  - `test_de_finds_tension_optimum` — same, DE with popsize=10, maxiter=20 (kept small for test speed); within 10%.
  - `test_shgo_2d_classical_alpha` — E4_1 classical-α over `[(-2, 2), (197/288, 2)]`, SHGO finds at least 1 feasible local min; compare against Brady-Livescu stored value with loose (within-basin) tolerance.
  - Mark the classical-α test `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestGlobalOptimizers and not slow"`

### 43.6 — Staged optimization: cheap inner + expensive validator

- [ ] **43.6a** Implement `run_staged_optimize(scheme, kernel, report_field, bounds, *, inner_gate=3, inner_max_layer=3, validator_max_layer=6, top_k=5, method="Nelder-Mead", n_restarts=20, seed=0) -> OptimizeResult`:
  - Stage 1 — inner: constructs `f_inner = make_objective(scheme, kernel, report_field, gate_layer=inner_gate, max_layer=inner_max_layer)`. Runs `multi_start_optimize(f_inner, bounds, n_restarts)`.
  - Stage 2 — validation: takes the `top_k` distinct candidate points from the multi-start `history` (deduplicated by rounding to 6 decimals). For each, re-runs `brady2d_stability_score` at `max_layer=validator_max_layer` and re-ranks by the same `report_field`.
  - Returns the `OptimizeResult` whose parameters give the best `validator` objective. Adds a `stage: Literal["inner", "validated"]` to indicate whether the outer validator actually re-ranked (i.e., validator picked a different candidate than inner).
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestStaged"`

- [ ] **43.6b** Tests `TestStaged`:
  - `test_staged_tension_e4_convergence` — staged run at n_restarts=3, inner_gate=3, validator=6 finds σ within 10% of 3.0.
  - `test_staged_validator_reorders` — synthetic test: inner and validator metrics disagree; validator output differs from inner output. Use `report_field="layer6.transient_growth_bound"` which isn't available at L3; force the validator stage to re-order.
  - Mark `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestStaged"`

### 43.7 — CLI: `sweeps/optimize.py`

- [ ] **43.7a** Create `scripts/stencil_gen/sweeps/optimize.py::main(argv) -> int`:
  - Args:
    - `--scheme {E2,E4}` (required)
    - `--kernel {tension,gaussian,multiquadric,tension-penalty,mixed-epsilon,classical}` (required)
    - `--objective FIELD` (required; dotted path like `layer6.spectral_abscissa`)
    - `--gate-layer INT` (default 3)
    - `--max-layer INT` (default: layer implied by objective)
    - `--bounds LO1 HI1 [LO2 HI2 ...]` (if absent, falls back to `DEFAULT_BOUNDS`)
    - `--method {Nelder-Mead,COBYQA,SHGO,DE,staged}` (default `staged`)
    - `--n-restarts INT` (default 10; used by Nelder-Mead, COBYQA, staged)
    - `--max-evals INT` (default 200; used by local methods)
    - `--seed INT` (default 0)
    - `--validate-with-cpp` (runs L8 on final winner)
    - `--update-known-values`
    - `--json-output PATH`
  - Dispatches to the right `run_*` function, prints a summary, optionally writes to `known_values.json`.
  - File: `scripts/stencil_gen/sweeps/optimize.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps optimize --scheme E4 --kernel tension --objective layer1.boundary_gv_err --bounds 0.5 20 --method Nelder-Mead --max-evals 40`

- [ ] **43.7b** Register `optimize` subcommand in `scripts/stencil_gen/sweeps/__main__.py`:
  - Add `sub_opt = subparsers.add_parser("optimize", help="Optimize boundary-closure parameters against a stability objective")` with all args.
  - Add dispatch block with lazy import.
  - Do NOT add to `_run_all` in quick mode — optimization runs are not smoke tests.
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps optimize --help`

- [ ] **43.7c** CLI smoke tests `TestOptimizeCLI`:
  - `test_cli_tension_nelder_mead` — subprocess invocation with a tiny budget completes and prints a summary containing "best".
  - `test_cli_rejects_bad_objective` — objective `layer99.foo` exits non-zero.
  - `test_cli_rejects_missing_bounds_and_kernel` — dimensions mismatch returns a clear error.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestOptimizeCLI"`

### 43.8 — Persistence to `known_values.json`

- [ ] **43.8a** In `sweeps/optimize.py`, when `--update-known-values` is set:
  - Load with `load_known_values()`.
  - Deep-set `kv["brady2d_optima"][scheme][kernel][objective_field]` = serialized `OptimizeResult` dict (best_x, best_params, best_objective, method, bounds, n_evals, compute_time, converged). Omit `history` from the persisted form to keep JSON small.
  - Save with `save_known_values(kv)`.
  - Additive: verify existing keys (including `brady2d_calibration`, `brady2d_sweep`, per-scheme entries) are not modified.
  - File: `scripts/stencil_gen/sweeps/optimize.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps optimize --scheme E4 --kernel tension --objective layer1.boundary_gv_err --bounds 0.5 20 --method Nelder-Mead --max-evals 40 --update-known-values && uv run python -c "import json; d=json.load(open('sweeps/known_values.json')); assert 'brady2d_optima' in d; print(d['brady2d_optima'])"`

- [ ] **43.8b** Add `TestRegressionBrady2DOptima` in `test_phs.py`:
  - Loads `brady2d_optima` from `known_values.json`. Graceful skip if absent.
  - For each stored entry, rebuilds `f = make_objective(...)`, evaluates at `best_x`, asserts the result matches the stored `best_objective` within 1% relative tolerance.
  - Also verifies `converged is True` at the stored result.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionBrady2DOptima"`

### 43.9 — Classical-α E4_1 2D optimization

- [ ] **43.9a** Extend `params_from_vector`/`vector_from_params` for `kernel="classical"` and add the 197/288 constraint:
  - Confirm the Python `_build_classical_diff_matrix` accepts `alpha` and does NOT impose the 197/288 bound (the C++ imposes it at construction; the Python path should either mirror it or rely on the optimizer to respect bounds).
  - If the Python path accepts `alpha[1] < 197/288` without error but produces a singular D, the objective returns `+inf` via the `try/except` — acceptable.
  - `DEFAULT_BOUNDS[("E4", "classical")]` already has `α₁ ≥ 197/288`; confirm and add a test that setting `α₁ < 197/288` raises in optimizer setup (bound violation).
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestClassicalAlphaBounds"`

- [ ] **43.9b** Single-seed optimization run on E4_1 classical-α:
  - `run_staged_optimize(scheme="E4", kernel="classical", report_field="layer6.transient_growth_bound", bounds=DEFAULT_BOUNDS[("E4", "classical")], inner_gate=3, inner_max_layer=3, validator_max_layer=6, top_k=5, method="Nelder-Mead", n_restarts=20, seed=0)`.
  - Assert converged result has `best_x[1] >= 197/288 - 1e-9` (hard bound respected).
  - Record the result.
  - Mark `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestStagedClassicalAlpha" -m slow`

- [ ] **43.9c** Multi-seed diversity study (analog of Brady-Livescu Table 4):
  - Create `scripts/stencil_gen/benchmarks/alpha_basin_survey.py` with a `run_survey(n_seeds=20, bounds=..., ...)` function.
  - For each seed, run `run_staged_optimize` and record the best `(α₀, α₁)`. Cluster results by rounding to 2 decimals; report the count of distinct basins found.
  - CLI entry: `python -m stencil_gen.brady2d_cli --alpha-basin-survey --n-seeds 20`. Prints a table of `(α₀, α₁, objective, n_seeds_in_basin)`.
  - Not a test, but keep it under 200 lines. Mark the test `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/stencil_gen/benchmarks/alpha_basin_survey.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.benchmarks.alpha_basin_survey import run_survey; r = run_survey(n_seeds=3, bounds=[(-2,2),(197/288,2)]); print(len(r['basins']))"`

- [ ] **43.9d** Compare the survey's top basin against Brady-Livescu's published E4 α:
  - Read the published values from `stencil_gen/alpha_extraction.py` (they're stored there).
  - Assertion test: at least one basin in the survey output has `(α₀, α₁)` within a 0.5 L∞ ball of the published value.
  - The paper doesn't claim uniqueness, so this is a containment check, not an identity check.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestAlphaSurveyVsPublished" -m slow`

### 43.10 — L8 validation of optimizer winners

- [ ] **43.10a** Wire `--validate-with-cpp` in `sweeps/optimize.py`:
  - After the main optimization completes, if `--validate-with-cpp` set and kernel is one of the C++-supported families (`classical`, `tension`, `gaussian`, `multiquadric`), call `brady2d_stability_score(..., max_layer=8, layer8_N=31, layer8_t_final=5.0)` at `best_params`.
  - Append the L8 report to the persisted result under `cpp_validation: {stable, final_linf, wall_time_s}`.
  - If L8 fails (simulation blows up), log a warning but do not alter `best_objective` — the analytical verdict stands for now; L8 disagreement is diagnostic.
  - File: `scripts/stencil_gen/sweeps/optimize.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps optimize --scheme E4 --kernel tension --objective layer1.boundary_gv_err --bounds 0.5 20 --method Nelder-Mead --max-evals 40 --validate-with-cpp`

- [ ] **43.10b** Test `TestOptimizeCppValidation`:
  - Mock or skip if `build/shoccs` not present.
  - `test_validate_classical_e4_published_alpha` — optimizer winner passes L8 (final_linf < 1.0 at t=5, N=21).
  - Mark `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestOptimizeCppValidation"`

### 43.11 — Documentation and skill updates

- [ ] **43.11a** Create `scripts/stencil_gen/docs/optimization_reference.md`:
  - Architecture diagram (cheap inner → top-k → expensive validator → L8).
  - API reference for `make_objective`, `params_from_vector`, `extract_field`, `run_staged_optimize`, `multi_start_optimize`, `run_scipy_local`, `run_scipy_shgo`, `run_scipy_de`.
  - Recipe: "How to optimize a new family" — bounds declaration + kernel routing.
  - Recipe: "How to add a new objective field" — just pick a dotted path; no code changes needed.
  - Known limitations (multi-objective, multi-fidelity BO, Brady-Livescu 1D Euler reproduction — all deferred).
  - File: `scripts/stencil_gen/docs/optimization_reference.md` (new)
  - Test: (no test)

- [ ] **43.11b** Update `scripts/stencil_gen/docs/brady2d_stability_reference.md` with a new "Optimization" section cross-linking to the new doc.
  - File: `scripts/stencil_gen/docs/brady2d_stability_reference.md`
  - Test: (no test)

- [ ] **43.11c** Update `.claude/skills/stencil-sweeps/SKILL.md`:
  - Add `optimize` subcommand under CLI quick reference.
  - Add a one-line bullet under "When to Use" for optimization.
  - File: `.claude/skills/stencil-sweeps/SKILL.md`
  - Test: (no test)

- [ ] **43.11d** Update `.claude/skills/group-velocity-analysis/SKILL.md`:
  - Add a bullet pointing to the new optimization layer (the scoring pipeline now feeds a concrete optimizer).
  - File: `.claude/skills/group-velocity-analysis/SKILL.md`
  - Test: (no test)

---

## Ordering

```
43.1a → 43.1b → 43.1c → 43.1d         # skeleton + primitives + scope reconcile
  ↓
43.2a → 43.2b                          # objective factory
  ↓
43.3a → 43.3b                          # baseline local (Nelder-Mead + COBYQA)
  ↓
43.4a → 43.4b                          # multi-start wrapper
  ↓
43.5a → 43.5b → 43.5c                  # SHGO + DE
  ↓
43.6a → 43.6b                          # staged pipeline
  ↓
43.7a → 43.7b → 43.7c                  # CLI
  ↓
43.8a → 43.8b                          # persistence + regression
  ↓
43.9a → 43.9b → 43.9c → 43.9d          # classical-α (depends on all prior)
  ↓
43.10a → 43.10b                        # L8 validation
  ↓
43.11a → 43.11b → 43.11c → 43.11d      # docs
```

Parallelizable after 43.4 completes:
- 43.5 (SHGO/DE) and 43.6 (staged) are independent.
- 43.9 is the most downstream and depends on 43.6, 43.7, 43.8 all landing.

---

## Completion Criteria

- `stencil_gen/optimizer.py` exports `OptimizeResult`, `params_from_vector`, `vector_from_params`, `extract_field`, `make_objective`, `run_scipy_local`, `run_scipy_shgo`, `run_scipy_de`, `multi_start_optimize`, `run_staged_optimize`, `DEFAULT_BOUNDS`.
- No new external dependencies — only `scipy.optimize` and `scipy.stats.qmc` (both already present).
- `python -m sweeps optimize --scheme E4 --kernel tension --objective layer1.boundary_gv_err --bounds 0.5 20 --method Nelder-Mead --max-evals 40` runs end-to-end and prints `best_params` converging within 10% of the sweep-derived σ=3.0.
- `python -m sweeps optimize --scheme E4 --kernel classical --objective layer6.transient_growth_bound --method staged --n-restarts 20 --update-known-values` runs, respects the α₁ ≥ 197/288 bound, finds at least one feasible local minimum, and persists to `known_values.json["brady2d_optima"]["E4"]["classical"]`.
- `scripts/stencil_gen/benchmarks/alpha_basin_survey.py` with `n_seeds=20` reports at least 3 distinct basins for E4 classical-α (cross-checks Brady-Livescu's multi-modality finding of 101 E4 schemes at their full budget).
- The survey's top basin contains a point within 0.5 L∞ of Brady-Livescu's published E4 α (stored in `alpha_extraction.py`).
- `TestRegressionBrady2DOptima` passes: re-runs stored optima and verifies each matches within 1% of the recorded objective.
- `cd scripts/stencil_gen && uv run pytest tests/ -x -q` continues to pass in under 60 seconds (new slow tests marked).
- Plan 44 (multi-objective Pareto via pymoo) can now start cleanly — the `make_objective` factory extends to weighted scalarization without refactoring, and a future NSGA-II caller can reuse `params_from_vector` unchanged.
