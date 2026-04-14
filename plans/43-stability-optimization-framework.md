# Phase 43: Stability Optimization Framework

**Goal:** Turn the existing sweep framework from brute-force grid search into actual optimization. Wrap `brady2d_stability_score` as an objective function for off-the-shelf optimizers (scipy.optimize), support single-objective + feasibility-cliff constraints, random-restart for multi-basin landscapes, SHGO and DE for global coverage, and staged cheap-layer-inner + expensive-layer-validator pipelines. Persist discovered optima to `known_values.json["brady2d_optima"]` and regression-test them. Final validation runs winners through the C++ solver via the plan-42 L8 bridge.

**Depends on:** Phase 41 (analytical stack; `brady2d_stability_score`, `StabilityReport`) and Phase 42 (C++ bridge for L8 validation).

**What this plan does NOT do** (explicit, to keep scope tight):

- **Multi-objective Pareto optimization** (pymoo NSGA-II). Single-objective plus feasibility cliff covers the working-out-which-parameter-is-best question cleanly; multi-objective is a separate infrastructure and output schema. Defer to a follow-up plan (44). The architecture here supports weighted scalarization as a bridge path.
- **Multi-fidelity Bayesian optimization** (BoTorch / Emukit). We use the built-in layered short-circuit as a manual cascade (staged cheap inner → expensive validator). Defer BO to a follow-up plan (45).
- **Brady-Livescu 1D Euler reproduction**. Their 2019 paper's objective requires a full nonlinear 1D Euler RK4 solver in Python (not present in this repo) and returns a two-phase blow-up-or-monotonicity score. Reproducing exactly is explicitly *not a target* — the paper notes their published α's are "simply the first entries in the databases" and the procedure is known to be multi-modal (Table 4 reports 101 found E4 schemes from random restarts). Defer to plan 46.
- **Classical-α E2_1 (4D)**. The user noted second-order stability is inconsequential. Skip in favor of E4_1 (2D, `α₁` lower bound is analytical-stability-driven, not the C++ cut-cell 197/288 floor — see 43.9a).
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
| Classical-α | E4_1 | 2 | α₀ ∈ [-2, 2], α₁ ∈ [0.05, 2] | analytical-stability only; C++ cut-cell 197/288 floor enforced at L8 diagnosis — see 43.9a |

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
- `src/stencils/E4_1.cpp` lines 35–41 (the α₁ ≥ 197/288 constraint is cut-cell-specific — it exists to keep the psi denominator non-zero — and is **not** enforced by the Python analytical pipeline L1–L7; it fires only as an L8 diagnostic. See 43.9a.)

**Test commands:**

```bash
# Fast: the optimizer module in isolation
cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q

# CLI smoke: optimize tension E4 stability margin at a tiny budget
# (uses layer3.max_stab_eig — a stability-margin field with an interior
# minimum — instead of layer1.boundary_gv_err, which is monotone over the
# feasible region and drives σ to the lower bound; see 43.3b resolution.)
cd scripts/stencil_gen && uv run python -m sweeps optimize \
    --scheme E4 --kernel tension \
    --objective layer3.max_stab_eig \
    --gate-layer 3 --max-layer 3 --bounds 0.5 20 \
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
  - Constants: `DEFAULT_BOUNDS` dict mapping `(scheme, kernel) -> list[tuple[float, float]]` with the bounds table above. E4 classical α has `[(-2.0, 2.0), (0.05, 2.0)]` — see 43.9a for the analytical-vs-cut-cell bound reconciliation.
  - All function stubs raise `NotImplementedError`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.optimizer import OptimizeResult, DEFAULT_BOUNDS; print(list(DEFAULT_BOUNDS.keys())[:3])"`

- [x] **43.1b** Implement `params_from_vector(kernel: str, x: np.ndarray) -> dict` and `vector_from_params(kernel: str, params: dict) -> np.ndarray`:
  - `kernel="tension"` / `"gaussian"` / `"multiquadric"`: 1D, `x=[σ]` or `x=[ε]` → `{"sigma": x[0]}` or `{"epsilon": x[0]}`.
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

- [x] **43.1e** Follow-up cleanup missed by 43.1d: two downstream task specs still reference the pruned `tension-penalty` / `mixed-epsilon` kernels and must be reconciled with option (b) before the next work pass proceeds.
  - In **43.1b** (lines 93–96) delete the two bullets for `kernel="tension-penalty"` and `kernel="mixed-epsilon"`; the spec should match the actual implementation in `optimizer.py` (tension / gaussian / multiquadric / classical only). Leave the round-trip test bullet intact.
  - In **43.7a** (the `--kernel` argparse choices bullet) change `{tension,gaussian,multiquadric,tension-penalty,mixed-epsilon,classical}` to `{tension,gaussian,multiquadric,classical}` so a literal reading of the CLI spec cannot reintroduce the pruned kernels when 43.7a is implemented.
  - File: `plans/43-stability-optimization-framework.md` only (no code changes).
  - Test: `grep -n "tension-penalty\|mixed-epsilon" plans/43-stability-optimization-framework.md` should return only the two expected locations — the "What this plan does NOT do" bullet and the 43.1d resolution bullet.

### 43.2 — Objective factory with feasibility cliff

- [x] **43.2a** Implement `make_objective(scheme, kernel, report_field, *, gate_layer=3, max_layer=None) -> Callable[[np.ndarray], float]`:
  - `max_layer` defaults to the layer implied by `report_field` (e.g., `layer6.*` → `max_layer=6`). If less than `gate_layer`, raise `ValueError`.
  - Returned function: converts `x` via `params_from_vector`, calls `brady2d_stability_score(scheme, kernel, params, max_layer=max_layer, short_circuit=True)`, checks `report.failed_layer is None or report.failed_layer > gate_layer`. If gate failed, return `+inf`. Otherwise return `extract_field(report, report_field)`.
  - Wraps `brady2d_stability_score` in `try/except Exception` returning `+inf` (ill-conditioned RBF systems at extreme parameters can raise).
  - Memoizes nothing — each call is fresh.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestMakeObjective"`

- [x] **43.2b** Tests for `make_objective`:
  - `test_objective_returns_finite_on_feasible` — E4 tension at a sweep-known-feasible σ (e.g. σ=3) passes L1-L3, objective returns a finite `layer1.boundary_gv_err`. (Feasibility at that σ; not an optimum claim — see 43.3b.)
  - `test_objective_returns_inf_on_gate_failure` — deliberately-bad parameters (e.g., E4 Gaussian ε=0.01) fail L3, objective returns `+inf`.
  - `test_objective_catches_exception` — monkey-patch to raise, verify `+inf` returned.
  - `test_objective_raises_on_bad_field` — `"layer99.foo"` at `gate_layer=3, max_layer=3` returns +inf without error.
  - File: `scripts/stencil_gen/tests/test_optimizer.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestMakeObjective"`

### 43.3 — Baseline local optimizer: Nelder-Mead and COBYQA

- [x] **43.3a** Implement `run_scipy_local(f, x0, bounds, *, method="Nelder-Mead", max_evals=200, tol=1e-6) -> OptimizeResult`:
  - Wraps `scipy.optimize.minimize` with `bounds` passed as the top-level keyword (scipy rejects `options={"bounds": ...}` for Nelder-Mead — it collides with the internal forwarding; minor correction to the original spec).
  - `history` is captured by wrapping the objective in a recorder (every evaluation → `(x.copy(), fval)`), not via scipy's per-iteration callback which only samples once per simplex step and would miss most feasibility-cliff evaluations.
  - Nelder-Mead options: `{"xatol": tol, "fatol": tol, "maxfev": max_evals, "adaptive": True}`.
  - COBYQA options: `{"maxfev": max_evals, "feasibility_tol": tol}`. COBYQA accepts `bounds` directly in scipy ≥ 1.14.
  - `converged = result.success and np.isfinite(result.fun)`; `best_params` left as `{}` (the driver is kernel-agnostic; higher-level wrappers own the kernel and will fill this via `dataclasses.replace`).
  - Rejects unknown methods and bounds-length mismatches with `ValueError`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestRunScipyLocal and not COBYQA"` — 6 tests green.

- [x] **43.3b** Add COBYQA version gate and tests:
  - `_probe_cobyqa_available()` runs a 2-evaluation minimize on an identity quadratic at import time and stores the result in `_COBYQA_AVAILABLE`; any exception marks the method unavailable.
  - `run_scipy_local(..., method="COBYQA")` raises `RuntimeError("COBYQA requires scipy >= 1.14; got {version}")` when the probe failed.
  - Integration test: COBYQA on tension-E4 `layer1.boundary_gv_err` from a feasible `x0=2.0` converges to a finite objective no worse than `f(x0)` within bounds. The plan's original claim that σ=3.0 is this metric's global minimum was incorrect — σ=3.0 is the sweep-derived *stability* optimum across the weighted landscape, but `layer1.boundary_gv_err` alone is monotone over the feasible region and minimized near the lower bound (σ≈0.5). Convergence-to-a-specific-σ is therefore the wrong invariant for this single-metric objective; feasibility + improvement is the right one. Follow-up: if a "σ=3.0 is the optimum" test is still wanted, the CLI smoke example in the plan header should be rephrased around an objective whose minimum *is* σ=3.0 (a weighted/stability-margin field), not `layer1.boundary_gv_err`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`, `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "COBYQA"` — 3 tests green (2 COBYQA-available + 1 unavailable-gate).

- [x] **43.3c** Plan-file cleanup missed by 43.3a/43.3b: the "σ=3.0 is the tension-E4 optimum" assumption that 43.3b refuted for `layer1.boundary_gv_err` still survives in several downstream task specs and the completion criteria, and those references will silently reintroduce a broken test invariant when 43.4/43.5/43.6 are implemented. Reconcile them now so the next work pass starts from a consistent spec. No code changes.
  - In the plan-header **Test commands** block (lines 67–72), change the CLI smoke example's objective from `layer1.boundary_gv_err` to one whose minimum is actually interior and stability-driven (e.g. `--objective layer3.max_stab_eig` with `--max-layer 3`); the current invocation would drive σ to the lower bound and defeats the "smoke" purpose.
  - In **43.4b** (`test_multi_start_finds_known_optimum`, line 167) replace "expects `best_x[0]` within 1.0 of σ=3.0" with "expects a finite feasible result, bound-respecting, and no worse than the best random restart's starting objective." If a specific-σ convergence test is still desired, specify the objective explicitly as a stability-margin field (e.g. `layer3.max_stab_eig` or a weighted composite) and cite the sweep-known optimum for *that* field, not the generic σ=3.0.
  - In **43.5c** (lines 189–191): `test_shgo_finds_tension_optimum` and `test_de_finds_tension_optimum` similarly drop the "within 5% / 10% of σ=3.0" acceptance — either pin an explicit objective whose minimum is σ=3.0, or assert only "finds a finite feasible global minimum, bound-respecting."
  - In **43.6b** (line 206) `test_staged_tension_e4_convergence` drop "finds σ within 10% of 3.0" and replace with "validator stage returns a finite feasible best that improves on or ties the inner stage's best at the same point."
  - In the **Completion Criteria** (line 376): either (a) rewrite the tension-E4 CLI smoke-convergence bullet to match whatever objective 43.3c's edits pick for the header (with a correct expected optimum), or (b) relax it to "runs end-to-end, prints a feasible `best_params`, and respects the stated bounds" — dropping the σ=3.0 figure entirely.
  - File: `plans/43-stability-optimization-framework.md` only.
  - Test: `grep -n "σ=3.0\|σ within\|of σ=3\|of 3.0\|within 5% of 3\|within 10% of 3" plans/43-stability-optimization-framework.md` — every surviving hit must be inside the 43.3b resolution narrative or the 43.3c item itself (i.e. archival text), not in an active task spec or completion criterion.

### 43.4 — Multi-start wrapper

- [x] **43.4a** Implemented `multi_start_optimize(f, bounds, n_restarts=10, *, method="Nelder-Mead", seed=0, max_evals=200, tol=1e-6)`:
  - Sobol-seeded starting points via `scipy.stats.qmc.Sobol(d=len(bounds), seed=seed)` scaled to bounds with `qmc.scale`.
  - Delegates each restart to `run_scipy_local`, aggregates `history` (concatenated), `n_evals` (summed), and `compute_time` (summed).
  - Returns the restart with the smallest finite `best_objective`; if every restart is infeasible, returns the last restart's record with `converged=False`.
  - Added `max_evals` and `tol` passthrough (not in the original spec but needed so CLI/test callers can keep runs tight).
  - `extras = {inner_method, n_restarts, seed, n_feasible_restarts}` for diagnostics.
  - Added `ValueError` gates on `n_restarts < 1` and empty `bounds`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestMultiStart"` — 6 tests green.

- [x] **43.4b** Tests `TestMultiStart`:
  - `test_multi_start_converges_on_quadratic` — analytic quadratic `(x-3)²` on `[0, 10]`, `n_restarts=4`: converges, `n_evals` matches `len(history)`, extras populated.
  - `test_multi_start_deterministic` — same seed produces identical `best_x`, `best_objective`, and `n_evals` across two calls.
  - `test_multi_start_handles_fully_infeasible` — objective returns `+inf` everywhere: result has non-finite `best_objective`, `converged=False`, `n_feasible_restarts == 0`.
  - `test_multi_start_rejects_zero_restarts` / `test_multi_start_rejects_empty_bounds` — input validation.
  - `test_multi_start_finds_feasible_optimum` — tension-E4 against `layer3.max_stab_eig`, `bounds=[(0.5, 20)]`, `n_restarts=4`: finite, bound-respecting, no worse than the best Sobol-sampled starting objective. (No specific-σ claim — see 43.3b.)
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestMultiStart"` — 6 tests green.

### 43.5 — Global optimizers: SHGO and DE

- [x] **43.5a** Implemented `run_scipy_shgo(f, bounds, *, n=100, iters=3) -> OptimizeResult`:
  - Wraps `scipy.optimize.shgo(f, bounds, n=n, iters=iters, minimizer_kwargs={"method": "Nelder-Mead"})`.
  - Wraps the objective in a history recorder (same pattern as `run_scipy_local`) so every feasibility-cliff evaluation shows up in `history`, not just iteration endpoints.
  - Post-processes `result.xl`/`result.funl` into `extras["local_minima"] = [(x, f)]` and `extras["n_local_minima"]`; scipy's simplicial-homology pass already yields one entry per distinct basin.
  - Fully-infeasible domain handling: scipy returns `result.x=None` / `result.fun=None` in that case and the `xl`/`funl` attributes may be missing. We detect the condition, return `best_objective=+inf`, `converged=False`, and fall back to the bound midpoint for `best_x` so callers don't have to special-case `AttributeError`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestSHGO"` — 4 tests green.

- [x] **43.5b** Implemented `run_scipy_de(f, bounds, *, popsize=15, maxiter=100, seed=0, strategy="best1bin") -> OptimizeResult`:
  - Wraps `scipy.optimize.differential_evolution` with `tol=1e-7`, `init="sobol"`, `polish=True`; records each evaluation via the same history-recorder pattern as `run_scipy_local`/`run_scipy_shgo`.
  - Uses `result.nfev` as `n_evals`; `converged = result.success and finite(best_objective)`.
  - Input validation: `ValueError` on empty `bounds`, `popsize < 1`, `maxiter < 1`.
  - `extras = {popsize, maxiter, seed, strategy, scipy_message}`.
  - Tests note: scipy DE's population-convergence tolerance can leave `result.success=False` even after the polish pass has pinned the minimum, so `TestDE::test_de_converges_on_quadratic` asserts finite convergence-to-a-known-optimum rather than `r.converged`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestDE"` — 6 tests green.

- [x] **43.5b-r1** Fixed the `run_scipy_de` docstring: replaced "a final Nelder-Mead polish (``polish=True``)" with the scipy-documented behavior (L-BFGS-B for bounded/unconstrained, `trust-constr` fallback when constraints are supplied — we do not pass constraints today). Docstring-only change, no API impact.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py` (docstring only).
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestDE"` — 7 green.

- [x] **43.5b-r2** Added `TestDE::test_de_handles_fully_infeasible`: calls `run_scipy_de(lambda x: float("inf"), bounds=[(0.0, 1.0)], popsize=4, maxiter=3, seed=0)` and asserts `not np.isfinite(r.best_objective)`, `r.converged is False`, `r.best_x.shape == (1,)`, and `len(r.history) > 0` (recorder captures rejected evaluations). Parallels `test_shgo_handles_fully_infeasible`.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestDE"` — 7 green.

- [x] **43.5c** Integration tests `TestGlobalOptimizers` landed — three tests added in `scripts/stencil_gen/tests/test_optimizer.py`:
  - `test_shgo_finds_tension_optimum` — 1D tension E4 against `layer3.max_stab_eig`, bounds `[(0.5, 20)]`, SHGO with `n=8, iters=1`. Asserts finite feasible result, bound-respecting, `n_local_minima >= 1`. ~27 s.
  - `test_de_finds_tension_optimum` — same objective, DE with `popsize=6, maxiter=8` (tighter than the 10/20 proposed to keep runtime < 30 s; DE already lands on a finite feasible minimum at this budget). ~22 s.
  - `test_shgo_2d_classical_alpha` (`@pytest.mark.slow`) — E4 classical-α, SHGO with `n=6, iters=1`. Asserts finite feasible, bound-respecting, at least one local minimum, and best_x within 0.5 L∞ of the Brady-Livescu stored α ≈ `[-0.7733, 0.1624]`. ~19 s.
  - **Important finding — plan text updated**: the plan's `DEFAULT_BOUNDS[("E4", "classical")] = [(-2, 2), (197/288, 2)]` encodes the C++ hard constraint `α₁ ≥ 197/288`, but the Brady-Livescu published feasible point sits at `α₁ ≈ 0.162` — **below** that lower bound. Probing DEFAULT_BOUNDS on an 11×8 grid returned zero L3-feasible points, so the 2D SHGO test uses relaxed bounds `[(-1.2, -0.3), (0.05, 0.4)]` that admit the published region. Resolving the DEFAULT_BOUNDS/Brady-Livescu mismatch — either widening the bounds or documenting a Python-vs-C++ divergence — is deferred to 43.9a (where the 197/288 constraint is explicitly re-examined).
  - Files: `scripts/stencil_gen/tests/test_optimizer.py`.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestGlobalOptimizers and not slow"` — 2 green; `--run-slow -k test_shgo_2d_classical_alpha` — 1 green.

### 43.6 — Staged optimization: cheap inner + expensive validator

- [x] **43.6a** Implemented `run_staged_optimize(scheme, kernel, report_field, bounds, *, inner_gate=3, inner_max_layer=3, validator_max_layer=6, top_k=5, method="Nelder-Mead", n_restarts=20, seed=0, max_evals=200, tol=1e-6) -> OptimizeResult`:
  - Stage 1 — inner: builds `f_inner = make_objective(...)` and runs `multi_start_optimize`. When `report_field` implies a layer deeper than `inner_max_layer` (e.g. `layer6.transient_growth_bound` with an L3 inner), the inner field falls back to `layer3.max_stab_eig` — the canonical Brady-Livescu short-circuit metric — so the inner stage still drives on a valid margin; the caller's original `report_field` is carried through to the validator stage. Recorded in `extras["inner_field"]` for transparency.
  - Stage 2 — validator: takes `top_k` feasible candidates from the inner `history`, deduplicated by rounding `x` to 6 decimals (``_top_k_candidates``), re-runs `brady2d_stability_score` at `max_layer=validator_max_layer` with `try/except` returning `+inf` on failure, and re-ranks by `report_field`.
  - Returns the `OptimizeResult` whose parameters give the best validator objective. `extras["stage"]` is `"validated"` when the validator's top pick differs from the inner's top candidate (6-decimal dedup key), else `"inner"`. `extras["validator_ranking"] = [(x, f), ...]`, plus `inner_*` diagnostics (inner_method, inner_n_restarts, inner_seed, inner_n_feasible_restarts, inner_best_objective, inner_best_x, inner_max_layer, validator_max_layer).
  - Added local `_report_to_dict` (mirror of `sweeps/brady2d_sweep._report_to_dict`, duplicated so `stencil_gen` doesn't depend on `sweeps`) to populate `best_report`.
  - Added `max_evals` and `tol` passthrough plus input validation (`inner_max_layer < inner_gate`, `validator_max_layer < inner_max_layer`, `top_k < 1` all raise `ValueError`). Fallback when every validator candidate blows up: returns the inner result wrapped as `method="staged"` with `stage="inner"` so callers always see the pipeline marker.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q -k "TestStaged" --run-slow` — 5 green (3 fast + 2 slow), total 16 s.

- [x] **43.6b** Added `TestStaged` — 5 tests in `scripts/stencil_gen/tests/test_optimizer.py`:
  - `test_staged_rejects_shallow_validator` / `test_staged_rejects_inner_shallower_than_gate` / `test_staged_rejects_zero_top_k` — input-validation guards; fast.
  - `test_staged_tension_e4_convergence` (`@pytest.mark.slow`) — tension E4, n_restarts=3, inner_max_layer=3, validator_max_layer=3 against `layer3.max_stab_eig`. Asserts `method=="staged"`, converged, finite/bound-respecting, validator winner ≤ inner best (since fields coincide here), validator_ranking sorted ascending, and `layer3` in `best_report`. ~2 s.
  - `test_staged_validator_reorders` (`@pytest.mark.slow`) — tension E4, `report_field="layer6.transient_growth_bound"`, validator_max_layer=6. Asserts the inner fallback field is `layer3.max_stab_eig`, `validator_max_layer=6` is recorded, `extras["stage"]` is populated, and `best_report` carries the L6 payload. ~14 s.
  - Full slow run: `SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q --run-slow` — 76 passed.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q -k "TestStaged" --run-slow`

- [x] **43.6c** Fixed `run_staged_optimize` fallback-path bugs introduced in 43.6a:
  1. The fallback `extras` dict (optimizer.py, validator-all-blowup branch) now mirrors the success-path keys — added `inner_best_objective` and `inner_best_x`. Downstream callers and tests that read those keys no longer `KeyError` on the fallback branch.
  2. The fallback now forces `converged=False` in the `dataclasses.replace(inner_result, method="staged", ...)` call — the staged pipeline did not converge at the validator depth (by definition of hitting the fallback), so inheriting `inner_result.converged=True` was wrong.
  - Added `test_staged_validator_all_blowups` in `TestStaged` that monkey-patches `brady2d_stability_score` so max_layer=3 (inner) returns a feasible L3 report with a σ-dependent `max_stab_eig` (the inner multi-start has a real objective to descend on), and max_layer=6 (validator) raises every time. Asserts `r.method == "staged"`, `r.extras["stage"] == "inner"`, `r.converged is False`, both `inner_best_objective` / `inner_best_x` present, and every `validator_ranking` entry infeasible. Fast (<1 s), no real L6 runs.
  - Files: `scripts/stencil_gen/stencil_gen/optimizer.py`, `scripts/stencil_gen/tests/test_optimizer.py`.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestStaged" --run-slow` — 6 green (3 guards + 1 new fallback + 2 slow integration), 16 s.

- [x] **43.6d** Replaced `test_staged_validator_reorders` with a deterministic synthetic version that monkey-patches both `multi_start_optimize` (canned inner history) and `brady2d_stability_score` (validator-depth ranking) so the test actually enforces the validator-re-ranking invariant the plan's 43.6b called for.
  - Canned inner history ranks A=2.0 best on `layer3.max_stab_eig`, then B=8.0, then C=5.0. Validator ranks by `layer6.transient_growth_bound` = (σ-8)², so B=8.0 wins. Asserts `r.extras["stage"] == "validated"`, `np.allclose(r.best_x, B)`, `r.best_x != inner_result.best_x`, `r.best_objective == 0.0`, validator ranking sorted ascending with B first, inner diagnostics preserved in extras, and L6 payload in `best_report`. Added `OptimizeResult` to the test-file imports.
  - Dropped the `@pytest.mark.slow` mark since the test no longer runs live L6 analysis; the prior tension-E4 L6 integration test was not retained as a slow smoke — its only assertion beyond what `test_staged_tension_e4_convergence` already covers was the now-deterministic re-order, and the synthetic test subsumes that coverage. `test_staged_tension_e4_convergence` remains as the live-pipeline smoke.
  - Files: `scripts/stencil_gen/tests/test_optimizer.py`.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "test_staged_validator_reorders"` — green in 1.7 s. Full `TestStaged --run-slow`: 6 passed in 8 s (was ~16 s).

- [x] **43.6e** Replaced the tautological `r.extras["inner_best_objective"] == pytest.approx(r.extras["inner_best_objective"])` assertion in `test_staged_validator_all_blowups` with assertions that tie the fallback extras to the result's public fields: `r.extras["inner_best_objective"] == pytest.approx(r.best_objective)` and `np.allclose(r.extras["inner_best_x"], r.best_x)` plus a shape-parity check. Kept the original finite/shape guards. These now actually verify the 43.6c extras-parity fix — since the fallback is built via `replace(inner_result, method="staged", converged=False, ...)` without touching `best_objective`/`best_x`, tying the extras to those public fields catches any regression that mutated them out of sync.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "test_staged_validator_all_blowups"` — green in 1.65 s. Full `TestStaged --run-slow`: 6 passed in 8.0 s.

### 43.7 — CLI: `sweeps/optimize.py`

- [x] **43.7a** Created `scripts/stencil_gen/sweeps/optimize.py::main(argv) -> int`:
  - Argparse surface covers `--scheme {E2,E4}`, `--kernel {tension,gaussian,multiquadric,classical}`, `--objective FIELD`, `--gate-layer`, `--max-layer`, `--bounds` (flat list parsed into pairs, with an odd-length guard), `--method {Nelder-Mead,COBYQA,SHGO,DE,staged}`, `--n-restarts`, `--max-evals`, `--seed`, `--validate-with-cpp` (stubbed, full wiring in 43.10a), `--update-known-values`, and `--json-output`.
  - Method-specific extras added beyond the original spec so SHGO/DE/staged can be exercised from the CLI without editing the module: `--validator-max-layer`, `--top-k`, `--inner-method` (staged); `--shgo-n`, `--shgo-iters` (SHGO); `--de-popsize`, `--de-maxiter` (DE).
  - Dispatch: Nelder-Mead / COBYQA via `multi_start_optimize`; SHGO via `run_scipy_shgo`; DE via `run_scipy_de`; `staged` via `run_staged_optimize`. For kernel-agnostic drivers (multi-start / SHGO / DE), `best_params` is backfilled via `params_from_vector` on a successful (finite) run so the persisted JSON carries the kernel-native dict.
  - Bounds fall back to `DEFAULT_BOUNDS[(scheme, kernel)]` when `--bounds` is absent; unknown combinations error out with a clear argparse message.
  - `_print_summary` renders scheme / kernel / method / objective / bounds and the result fields (`best_x`, `best_params`, `best_objective`, `converged`, `n_evals`, `compute_time`) plus each extras entry (length-summarised for `validator_ranking` / `local_minima`).
  - Persistence (under `--update-known-values`) writes to `kv["brady2d_optima"][scheme][kernel][objective]` with the 43.8a schema (best_x, best_params, best_objective, method, bounds, n_evals, compute_time, converged, best_report) and omits `history`. `--json-output PATH` mirrors the same dict to disk.
  - File: `scripts/stencil_gen/sweeps/optimize.py` (new)
  - Test: run directly via `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python -m sweeps.optimize --scheme E4 --kernel tension --objective layer3.max_stab_eig --gate-layer 3 --max-layer 3 --bounds 0.5 20 --method Nelder-Mead --max-evals 40 --n-restarts 3` — succeeds, prints `best_objective = -1.220708e-04` at σ ≈ 1.644 in ~6 s. Top-level `python -m sweeps optimize ...` dispatch is wired in 43.7b.

- [x] **43.7b** Registered the `optimize` subcommand in `scripts/stencil_gen/sweeps/__main__.py`:
  - Added `sub_opt = subparsers.add_parser("optimize", ...)` mirroring every flag in `sweeps/optimize.py::main`, including the staged knobs (`--validator-max-layer`, `--top-k`, `--inner-method`), SHGO/DE knobs (`--shgo-n`, `--shgo-iters`, `--de-popsize`, `--de-maxiter`), and post-run flags (`--validate-with-cpp`, `--update-known-values`, `--json-output`).
  - Added a dispatch block with a lazy `from .optimize import main as optimize_main`. Forwards all scalar args unconditionally and appends `--max-layer`, `--bounds`, `--json-output`, and the two boolean flags only when set — matching the argparse defaults in `optimize.py` so the forwarded argv never stomps on `None` defaults.
  - Not added to `_run_all` (optimization runs are not smoke tests; plan spec).
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps optimize --help` — prints the full argparse surface. End-to-end smoke: `SYMPY_CACHE_SIZE=50000 uv run python -m sweeps optimize --scheme E4 --kernel tension --objective layer3.max_stab_eig --gate-layer 3 --max-layer 3 --bounds 0.5 20 --method Nelder-Mead --max-evals 40 --n-restarts 2` → `best_objective = -1.220708e-04` at σ ≈ 1.644 in ~7 s (converged, 36 evals, 2 feasible restarts).

- [x] **43.7c** Added `TestOptimizeCLI` — 3 tests in `scripts/stencil_gen/tests/test_optimizer.py`:
  - `test_cli_tension_nelder_mead` (`@pytest.mark.slow`) — subprocess `python -m sweeps.optimize` with `--scheme E4 --kernel tension --objective layer3.max_stab_eig --bounds 0.5 20 --method Nelder-Mead --n-restarts 1 --max-evals 10`. Asserts `returncode == 0` and the summary contains `best_objective` / `best_params`. Marked slow because the subprocess pays the full SymPy cold-start tax (~5-7 min in a fresh env); timeout set to 900 s. Verified green via `--run-slow` on first manual invocation (393 s).
  - `test_cli_rejects_bad_objective` — in-process `main(["--objective", "bogus.field", ...])`. The plan's original `layer99.foo` is *not* a bad-field example: `_LAYER_PREFIX_RE` matches any `layer\d+\.` so `max_layer` would infer to 99 and the CLI would run silently (verified: took 6 min wall to produce `best_objective = inf`). Switched to `bogus.field`, which has no layer prefix and no alias entry, so `make_objective` raises `ValueError` up front → `parser.error` → `SystemExit(2)`. Fast.
  - `test_cli_rejects_kernel_bounds_dim_mismatch` — in-process `main(["--kernel", "classical", "--bounds", "0.5", "20", ...])`. Without the new `_validate_kernel_bounds_dim` guard, this would silently Sobol-sample a 1D start into a 2D kernel, every evaluation would throw-and-swallow, and the CLI would exit 0 with `best_objective = inf`. The guard now raises `ValueError("kernel='classical' expects 2 bound pair(s); got 1")` → `SystemExit`. Fast.
  - New CLI validation: `_KERNEL_DIM` map + `_validate_kernel_bounds_dim(kernel, bounds)` called after `_resolve_bounds` in `sweeps/optimize.py::main`. Catches user error at parse time instead of hiding it behind a silent infeasible run.
  - Files: `scripts/stencil_gen/tests/test_optimizer.py`, `scripts/stencil_gen/sweeps/optimize.py`.
  - Test: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q -k "TestOptimizeCLI"` — 2 passed, 1 slow skipped, ~4 s. Full non-slow suite still green: 77 passed, 3 skipped, 95 s.

### 43.8 — Persistence to `known_values.json`

- [x] **43.8a** Persistence path already landed with 43.7a (`sweeps/optimize.py::_result_to_persist_dict` + the `--update-known-values` block in `main`). This item closes out the contract test: a `TestOptimizeCLI::test_cli_update_known_values_additive_and_drops_history` case in `scripts/stencil_gen/tests/test_optimizer.py` monkey-patches `load_known_values`/`save_known_values` and `_run_method` (so the test never enters the real SymPy pipeline) and pins three invariants:
  1. The result is deep-set under `kv["brady2d_optima"][scheme][kernel][objective]` with the serialised `OptimizeResult` fields (`best_x`, `best_params`, `best_objective`, `method`, `bounds`, `n_evals`, `compute_time`, `converged`, `best_report`).
  2. `history` is **never** persisted (asserted via `"history" not in opt`).
  3. Existing top-level keys (`brady2d_calibration`, `brady2d_sweep`) are untouched after the write, and a second call with a *different* objective coexists with the first under the same `[scheme][kernel]` bucket (additive within `brady2d_optima`, not clobbering).
  - File: `scripts/stencil_gen/tests/test_optimizer.py` (test), `scripts/stencil_gen/sweeps/optimize.py` (unchanged — contract already met).
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "test_cli_update_known_values"` — 1 green, 2 s. Full `TestOptimizeCLI`: 3 passed, 1 skipped (slow subprocess), 1.5 s. Full `test_optimizer.py` suite: 78 passed, 3 skipped, 77 s.

- [x] **43.8b** Added `TestRegressionBrady2DOptima` in `scripts/stencil_gen/tests/test_phs.py`, alongside the existing `TestRegressionBrady2D{Calibration,Sweep}` classes:
  - Loads `brady2d_optima` from `known_values.json`; uses the same `_KNOWN` helper and `_skip_if_absent` fixture pattern as the sibling regression classes — graceful skip when the key is absent (the current state; first optimizer persistence hasn't landed yet).
  - For each stored `[scheme][kernel][objective]` entry: asserts `converged is True`, rebuilds `f = make_objective(scheme=..., kernel=..., report_field=objective)` (uses default `gate_layer=3` and infers `max_layer` from the dotted path — matches how the CLI built the objective when it persisted), evaluates at `np.asarray(entry["best_x"])`, asserts the recomputed value is finite and within 1% relative tolerance of the stored `best_objective` (denominator floored at 1e-12 to handle near-zero optima like `layer3.max_stab_eig = -1.22e-4`).
  - A secondary `pytest.skip` fires if the `brady2d_optima` subtree is present but empty, mirroring the `checked == 0` guard in `TestRegressionBrady2DSweep`.
  - Sanity-check: confirmed `make_objective` is bit-exact deterministic for tension-E4 `layer3.max_stab_eig` at σ=1.644 (two calls → 0.00e+00 relative difference), so the 1% tolerance has generous headroom against any sympy/numerics drift.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionBrady2DOptima"` — 1 skipped (expected; no stored optima yet). Full suite `uv run pytest tests/test_phs.py tests/test_optimizer.py` — 158 passed, 14 skipped in 78 s.

- [x] **43.8c** Closed the regression gate-config gap exposed by 43.8b. Persistence schema now records the objective-evaluation configuration so the regression test can rebuild `make_objective` deterministically instead of relying on defaults that may drift.
  - Added `_resolve_persisted_layers(args)` helper in `sweeps/optimize.py` that returns `(gate_layer, max_layer, validator_max_layer)`. For non-staged methods, `max_layer` is `args.max_layer` when explicit, else `_infer_max_layer(args.objective)` — and errors with `ValueError` when the objective has no layer prefix *and* no explicit `--max-layer` (can't happen via a working CLI run since `make_objective` would have errored first, but the guard prevents silent `None` persistence). For `method == "staged"`, persisted `max_layer` is the *inner* depth (`args.max_layer` or the 3 fallback that `_run_method` uses), and `validator_max_layer` is `args.validator_max_layer`. Non-staged methods never record `validator_max_layer`.
  - Extended `_result_to_persist_dict(...)` with `gate_layer: int`, `max_layer: int`, and optional `validator_max_layer: int | None` parameters. `gate_layer`/`max_layer` are always recorded; `validator_max_layer` is only written when non-`None` (i.e. staged path). Downstream wrapper `main()` calls `_resolve_persisted_layers(args)` once and forwards the triple — CLI parse errors on a missing `--max-layer` for a non-layer-prefixed objective are surfaced via `parser.error`.
  - Updated `TestRegressionBrady2DOptima::test_all_optima_objective_matches` in `tests/test_phs.py` to read `entry["gate_layer"]` / `entry["max_layer"]` and thread them into `make_objective`. For staged entries (`entry["method"] == "staged"`), `validator_max_layer` is the correct evaluation depth (that's where `best_objective` was computed), so the test uses it instead of the inner `max_layer`. Missing `gate_layer`/`max_layer` triggers `pytest.skip` with a "re-run optimizer to refresh" hint so pre-43.8c JSON won't hard-fail CI; a staged entry without `validator_max_layer` asserts loudly.
  - Extended `test_cli_update_known_values_additive_and_drops_history` in `tests/test_optimizer.py` with three new round-trip branches: (a) first CLI call (no `--max-layer`) asserts inferred `max_layer=3` from `layer3.max_stab_eig` and no `validator_max_layer`; (b) second CLI call (`--max-layer 6`) asserts explicit `max_layer=6` round-trips and still no `validator_max_layer`; (c) new third CLI call with `--method staged --validator-max-layer 7` asserts persisted `max_layer=3` (inner-depth default) and `validator_max_layer=7`.
  - Files: `scripts/stencil_gen/sweeps/optimize.py`, `scripts/stencil_gen/tests/test_phs.py`, `scripts/stencil_gen/tests/test_optimizer.py`.
  - Test: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py tests/test_phs.py -x -q -k "TestRegressionBrady2DOptima or test_cli_update_known_values"` — 1 passed (contract test), 1 skipped (regression, no stored optima yet), ~1.6 s. Full `tests/test_optimizer.py` suite: 78 passed, 3 skipped, ~76 s.

### 43.9 — Classical-α E4_1 2D optimization

- [x] **43.9a** Resolved the C++ 197/288 vs Brady-Livescu α-feasibility mismatch flagged by 43.5c. The C++ `E4_1` constraint `alpha[1] >= 197/288 ≈ 0.684` is *cut-cell-specific* — it exists to keep the psi denominator non-zero for `psi ∈ (0, 1)`. The analytical layers L1–L7 (and the Python `_build_classical_diff_matrix`) operate on uniform grids with no psi involvement, so that constraint does not apply to the optimizer's feasibility cliff. Verified empirically:
  - `_build_classical_diff_matrix(..., alpha_list=[-0.7733, 0.1624])` returns finite D without raising; published α is L3-feasible with `max_stab_eig ≈ -1.8e-4` (stable).
  - Grid-probe at `α₀=-0.77` across `α₁ ∈ [0.0, 0.5]`: L3-feasibility lives in `α₁ ∈ [~0.08, ~0.17]`; every point with `α₁ ≥ 0.2` fails L3 (unstable). The analytical and cut-cell feasible regions therefore do *not* overlap — choosing option (b) is the only way to give the optimizer a non-empty feasible interior.
  - Decision (option b): widened `DEFAULT_BOUNDS[("E4", "classical")]` from `[(-2, 2), (197/288, 2)]` to `[(-2.0, 2.0), (0.05, 2.0)]`. Added a docstring block above `DEFAULT_BOUNDS` explaining the Python-vs-C++ divergence and that L8 validation (plan 43.10) is where a 197/288 violation fires as a diagnostic.
  - `params_from_vector`/`vector_from_params` were already correctly implemented for `kernel="classical"` in 43.1b (verified via new round-trip test at the published point); no changes needed.
  - Updated the comment in `test_shgo_2d_classical_alpha` to drop the "deferred to 43.9a" note and explain that the narrower bounds there are a runtime-budget choice, not a feasibility workaround.
  - Added `TestClassicalAlphaBounds` (6 tests) in `scripts/stencil_gen/tests/test_optimizer.py` pinning:
    1. DEFAULT_BOUNDS admits the Brady-Livescu published α,
    2. DEFAULT_BOUNDS alpha[1] lower < 197/288 (confirms the bound shift vs prior spec),
    3. Round-trip `params_from_vector`/`vector_from_params` at the published α,
    4. Python `_build_classical_diff_matrix` accepts α below the C++ floor without raising,
    5. `make_objective` at the Brady-Livescu α returns a finite *negative* (stable) `layer3.max_stab_eig`,
    6. `make_objective` at `[α₀=-0.77, α₁=1.0]` (inside DEFAULT_BOUNDS but above the analytical feasible envelope) returns `+inf` — feasibility cliff handled gracefully.
  - Files: `scripts/stencil_gen/stencil_gen/optimizer.py`, `scripts/stencil_gen/tests/test_optimizer.py`.
  - Test: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q -k "TestClassicalAlphaBounds"` — 6 passed in 2.0 s. Full suite: 84 passed, 3 skipped, 76 s; slow `test_shgo_2d_classical_alpha` still green in 20 s.

- [x] **43.9a-r1** Plan-text cleanup missed by 43.9a: the decision to drop the 197/288 floor from `DEFAULT_BOUNDS[("E4", "classical")]` survives in several downstream task specs and the completion criteria as the old "α₁ ≥ 197/288 hard" invariant. These will silently encode the wrong bound / wrong assertion when 43.9b/c are implemented and will contradict the widened `DEFAULT_BOUNDS` that 43.9a actually shipped. Reconcile them now. No code changes. Resolution: updated the scope bullet (line 12), parameter-spaces table row (line 37), Read first bullet for `E4_1.cpp` (line 59), 43.1a inline spec (line 90), and the Completion Criteria line that previously said "respects the α₁ ≥ 197/288 bound". 43.9b's active bullet and 43.9c's test-command invocation were already in sync with the widened bounds (no edit needed). Verified via `grep -n "197/288\|197\.0/288" plans/43-stability-optimization-framework.md`: all surviving hits sit inside the 43.9a resolution narrative, the `TestClassicalAlphaBounds` bullet (archival), 43.9b's L8 diagnostic flag (which cites 197/288 as diagnostic-only), or this 43.9a-r1 item.
  - In the plan-header **scope** bullet (line 12), change "E4_1 (2D, single hard inequality `α₁ ≥ 197/288`)" to "E4_1 (2D, `α₁` lower bound is analytical-stability-driven, not the C++ cut-cell 197/288 floor — see 43.9a)".
  - In the plan-header **parameter-spaces table** (line 37), update the Classical-α row bounds from `α₁ ∈ [197/288, 2]` to `α₁ ∈ [0.05, 2]` and change the Constraints column from "α₁ ≥ 197/288 hard" to "analytical-stability only; C++ cut-cell 197/288 floor enforced at L8 diagnosis — see 43.9a".
  - In the **Read first** list (line 59), rephrase the `src/stencils/E4_1.cpp` bullet to note the 197/288 constraint is cut-cell-specific and is *not* enforced by the Python analytical pipeline (L1–L7) — it fires only as an L8 diagnostic.
  - In **43.1a** (line 90) change the inline spec `E4 classical α has `[(-2.0, 2.0), (197.0/288.0, 2.0)]`` to `E4 classical α has `[(-2.0, 2.0), (0.05, 2.0)]` — see 43.9a for the analytical-vs-cut-cell bound reconciliation`. This keeps the implemented constant matching the spec that describes it.
  - In **43.9b** (line 332) drop the bullet `Assert converged result has `best_x[1] >= 197/288 - 1e-9` (hard bound respected)` — it contradicts the widened `DEFAULT_BOUNDS` (which admits α₁ ≥ 0.05) and would fail on the Brady-Livescu feasible region (α₁ ≈ 0.16 < 197/288). Replace with: `Assert converged result is feasible (finite `best_objective`), bound-respecting (`best_x[i]` ∈ the widened `DEFAULT_BOUNDS` interval for each i), and records an L8 diagnostic flag `cpp_cutcell_violates_197_288 = best_x[1] < 197/288` in the result extras (purely informational — does not fail the test).`
  - In **43.9c** (line 344) change the test-command invocation `bounds=[(-2,2),(197/288,2)]` to `bounds=DEFAULT_BOUNDS[("E4","classical")]` (or the literal `[(-2,2),(0.05,2)]`) so the basin survey uses the same feasible envelope the rest of the pipeline uses.
  - In the **Completion Criteria** (line 434), change "respects the α₁ ≥ 197/288 bound" to "respects `DEFAULT_BOUNDS[("E4","classical")]` (widened in 43.9a to admit the Brady-Livescu analytical feasible region)".
  - File: `plans/43-stability-optimization-framework.md` only (no code changes).
  - Test: `grep -n "197/288\|197\.0/288" plans/43-stability-optimization-framework.md` — every surviving hit must be inside the 43.9a resolution narrative, the `TestClassicalAlphaBounds` description, the `Classical-α` parameter-spaces Constraints cell (which now cites 197/288 as an L8 diagnostic only), or this 43.9a-r1 item. No surviving hit may appear in an active (unchecked) task spec or in the `Completion Criteria` as a bound/assertion.

- [x] **43.9b** Single-seed optimization run on E4_1 classical-α — added `TestStagedClassicalAlpha::test_staged_classical_e4_single_seed` in `scripts/stencil_gen/tests/test_optimizer.py`:
  - Drives `run_staged_optimize(scheme="E4", kernel="classical", report_field="layer6.transient_growth_bound", bounds=DEFAULT_BOUNDS[("E4", "classical")], inner_gate=3, inner_max_layer=3, validator_max_layer=6, top_k=5, method="Nelder-Mead", n_restarts=20, seed=0, max_evals=60)`. `max_evals=60` (below the 200 default) keeps the slow run to ~2 min while still giving the inner multi-start enough budget to land on the Brady-Livescu basin at this seed.
  - Asserts `method=="staged"`, finite feasible `best_objective`, `best_x[i]` within each `DEFAULT_BOUNDS[("E4","classical")]` interval, and `best_params == {"alpha": [...]}` (2-element list).
  - Records the L8 cut-cell diagnostic `cpp_cutcell_violates_197_288 = bool(best_x[1] < 197/288)` into `r.extras` and asserts the flag is a `bool`. Purely informational — the Python analytical pipeline does not enforce the C++ cut-cell 197/288 floor (see 43.9a-r1); downstream plan 43.10 wiring can consume the flag.
  - Observed at seed=0: validator winner ≈ `[-0.81, 0.09]` (flag=True, as expected — α₁ < 197/288), `best_objective ≈ 4.83`, stage=`validated`, 1 feasible restart of 20 on the full envelope, ~135 s. Extra budget would improve basin hit rate but isn't needed for the test's invariants.
  - Marked `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q -k "TestStagedClassicalAlpha" --run-slow` — 1 passed in 138 s. Non-slow suite: `uv run pytest tests/test_optimizer.py -x -q` — 84 passed, 4 skipped, 77 s.

- [x] **43.9b-r1** Resolved via **option (a)** — persist the flag in the pipeline. Added module-level helper `_record_cpp_cutcell_diagnostic(extras, scheme, kernel, best_x)` in `stencil_gen/optimizer.py` that populates `extras["cpp_cutcell_violates_197_288"] = bool(best_x[1] < 197/288)` when `(scheme, kernel) == ("E4", "classical")` and `best_x` has at least two finite entries; no-op for other schemes/kernels or unusable vectors. `run_staged_optimize` now calls the helper on both the success and validator-all-blowup fallback paths (before building each `OptimizeResult`), so the flag shows up in every E4-classical staged result including those that never reach the validator.
  - Test updates in `scripts/stencil_gen/tests/test_optimizer.py`:
    - `TestStagedClassicalAlpha::test_staged_classical_e4_single_seed` now *observes* the flag: asserts `"cpp_cutcell_violates_197_288" in r.extras` and `r.extras[...] == (r.best_x[1] < 197/288)` — the tautological `isinstance(..., bool)` and the self-assignment are gone.
    - Added `TestStaged::test_staged_records_cpp_cutcell_flag_for_e4_classical` (fast, synthetic via `monkeypatch`) — exercises the success path (winner below floor → flag True), the fallback path (validator raises → flag still True from inner `best_x`), and the winner-above-floor case (flag False). ~1.6 s, no real L3/L6 runs.
    - Added `TestStaged::test_staged_omits_cpp_cutcell_flag_for_other_kernels` (fast, synthetic) — asserts the flag is *absent* for non-classical kernels so extras stays uncluttered.
  - Verified: `SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q -k "TestStaged and not slow"` — 7 passed, 1.65 s. `SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q -k "TestStagedClassicalAlpha" --run-slow` — 1 passed, 137 s. Full non-slow suite `SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q` — 86 passed, 4 skipped, 76 s.
  - Files: `scripts/stencil_gen/stencil_gen/optimizer.py`, `scripts/stencil_gen/tests/test_optimizer.py`.

- [x] **43.9b-r2** Resolved via **option (a)** + the test-name-drift narrative fix. Two issues closed in one pass.
  1. **Test-name drift fixed.** The 43.9b-r1 narrative at line 354 now reads `TestStaged::test_staged_omits_cpp_cutcell_flag_for_other_kernels`, matching the committed test (`tests/test_optimizer.py:1160`). Plan-file-only change; no tests moved.
  2. **Flag persisted via option (a).** `sweeps/optimize.py::_result_to_persist_dict` now copies `extras["cpp_cutcell_violates_197_288"]` (when present) into the persisted entry as a top-level `cpp_cutcell_violates_197_288: bool`. Keys absent from `extras` are not written — tension/gaussian/multiquadric entries stay uncluttered because `_record_cpp_cutcell_diagnostic` only populates the flag for E4 classical-alpha in the first place. A nested `diagnostics: {...}` dict was considered and rejected: the flag is currently the only diagnostic worth persisting and flattening keeps 43.10 consumers' lookups cheap. When a second diagnostic appears, the obvious refactor is to promote to an allow-list.
  - Extended `TestOptimizeCLI::test_cli_update_known_values_additive_and_drops_history` with a fourth CLI call that monkey-patches `_run_method` to return an E4-classical `OptimizeResult` whose extras carry `cpp_cutcell_violates_197_288=True`. Asserts the persisted `brady2d_optima["E4"]["classical"]["layer3.max_stab_eig"]` entry carries `cpp_cutcell_violates_197_288 is True`, and the three previously-persisted tension entries (which had no flag in extras) do *not* gain the key.
  - Files: `scripts/stencil_gen/sweeps/optimize.py`, `scripts/stencil_gen/tests/test_optimizer.py`, `plans/43-stability-optimization-framework.md`.
  - Test: `grep -n "test_staged_omits_cpp_cutcell_flag_for_e4_tension" plans/43-stability-optimization-framework.md` — 0 hits. `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q -k "test_cli_update_known_values"` — 1 passed, 1.6 s. Full non-slow suite: 86 passed, 4 skipped, 75 s.

- [x] **43.9c** Multi-seed diversity study (analog of Brady-Livescu Table 4) — added `scripts/stencil_gen/stencil_gen/benchmarks/alpha_basin_survey.py` (186 lines, under the 200-line cap):
  - `run_survey(n_seeds=20, bounds=None, *, scheme="E4", kernel="classical", report_field="layer6.transient_growth_bound", inner_gate=3, inner_max_layer=3, validator_max_layer=6, top_k=5, method="Nelder-Mead", n_restarts=20, max_evals=60, base_seed=0, cluster_decimals=2)` drives `run_staged_optimize` across seeds `range(base_seed, base_seed+n_seeds)`, records the validator winner, and clusters winners by rounding `best_x` to `cluster_decimals` decimals (plan specifies 2). `bounds=None` defaults to `DEFAULT_BOUNDS[(scheme, kernel)]` (the widened envelope from 43.9a — the analytical feasible region, which extends below the cut-cell 197/288 floor).
  - Returns `{"seed_results": [...], "basins": [...], "n_distinct_basins": int, "n_feasible_seeds": int, "compute_time": float, "config": {...}}`. Each `basins[i]` holds `{"alpha": [α₀, α₁], "best_objective": float, "n_seeds_in_basin": int, "seeds": [seed, ...], "cpp_cutcell_violates_197_288": bool|None}` and basins are sorted ascending by `best_objective`. The basin keeps the *best* alpha/objective across seeds that land in the same rounded cluster.
  - Propagates the `cpp_cutcell_violates_197_288` L8 diagnostic (plan 43.9b-r1) through both per-seed entries and per-basin summaries — purely informational, consumed by plan 43.10.
  - `format_survey_table(survey)` renders a markdown table of the basins plus a header line with scheme/kernel/n_seeds/feasible/n_distinct_basins/wall-clock.
  - CLI: new `--alpha-basin-survey` mode in `scripts/stencil_gen/stencil_gen/brady2d_cli.py` with `--n-seeds`, `--base-seed`, `--n-restarts`, and `--survey-max-evals` knobs. Prints the `format_survey_table` output and honours `--json-output`. Returns exit 0 iff at least one seed was feasible.
  - Input validation: `ValueError` on `n_seeds < 1` and `cluster_decimals < 0`.
  - Files: `scripts/stencil_gen/stencil_gen/benchmarks/alpha_basin_survey.py` (new), `scripts/stencil_gen/stencil_gen/brady2d_cli.py` (wire-up).
  - Verified:
    - `uv run python -m stencil_gen.brady2d_cli --help` prints the four new flags.
    - Tiny smoke `run_survey(n_seeds=2, bounds=[(-2,2),(0.05,2)], n_restarts=2, max_evals=10)` returns `{n_distinct_basins:0, n_feasible_seeds:0, len(basins):0, len(seed_results):2, config keys populated}` in ~3.5 s — function runs clean even when no seed finds a feasible point (fallback path exercised).
    - Synthetic multi-basin test monkey-patching `run_staged_optimize` with 4 hand-built `OptimizeResult`s (3 feasible landing in 2 basins + 1 infeasible) confirms basin clustering, ascending-objective sort, per-basin seed tracking, and best-alpha-per-basin all work.
    - Existing optimizer suite still green: `SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q` — 86 passed, 4 skipped, 75 s.
  - The plan-specified heavyweight smoke `run_survey(n_seeds=3, bounds=[(-2,2),(0.05,2)])` (production defaults, ~7 min wall) is deferred until 43.9d needs the actual basin data — the synthetic test above gives full logic coverage without the budget.
  - Test: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python -c "from stencil_gen.benchmarks.alpha_basin_survey import run_survey; r = run_survey(n_seeds=2, bounds=[(-2,2),(0.05,2)], n_restarts=2, max_evals=10); print(len(r['basins']))"` — green in ~4 s.

- [x] **43.9c-r1** Added `TestAlphaBasinSurvey` — 5 synthetic tests in `scripts/stencil_gen/tests/test_optimizer.py`, all fast (no real L3/L6 runs) via `monkeypatch.setattr(alpha_basin_survey, "run_staged_optimize", ...)` returning canned `OptimizeResult`s keyed by `seed`. A private `_canned(...)` helper on the class builds a 2D-alpha or 1D-sigma canned result with configurable `best_objective`, `stage`, and `cpp_cutcell_violates_197_288`. Coverage matches the spec exactly:
  1. `test_run_survey_clusters_multiple_basins` — 4 seeds (3 feasible landing in 2 rounded clusters, 1 infeasible). Asserts `n_distinct_basins==2`, `n_feasible_seeds==3`, ascending-by-objective sort (seed-2 basin first at obj=1.0), the two-seed basin retains seed-2's lower-objective alpha (`[-0.797, 0.099]`) — the winner-replaces-basin branch — and the `config` dict round-trips every key including `cluster_decimals=2`, `n_seeds=4`, bounds, inner/validator layers.
  2. `test_run_survey_propagates_cpp_cutcell_flag` — three seeds with flag ∈ {True, False, None}; asserts both per-seed entries and per-basin summaries carry the flag through identity-equal.
  3. `test_run_survey_all_infeasible_returns_empty_basins` — every canned result has `+inf`; asserts `basins==[]`, `n_distinct_basins==0`, `n_feasible_seeds==0`, `len(seed_results)==n_seeds`, `all(not feasible)`, and `format_survey_table` still renders (header contains `distinct basins=0` and the column schema line).
  4. `test_run_survey_rejects_bad_inputs` — `run_survey(n_seeds=0)` and `run_survey(..., cluster_decimals=-1)` each raise `ValueError` with the documented message stems.
  5. `test_format_survey_table_renders_header_and_rows` — minimal fake survey with three hand-built basins (flag ∈ {True, False, None}); asserts scheme/kernel/report_field appear in the top header line, `n_seeds=10`/`feasible=6`/`distinct basins=3`/`wall=12.3s` appear in the stats line, the 5-column schema `| α₀ | α₁ | best_objective | n_seeds | 197/288 viol |` is present, and the `viol` cell renders `"yes"`/`"no"`/`"-"` for True/False/None respectively.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`.
  - Test: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q -k "TestAlphaBasinSurvey"` — 5 passed in 1.7 s. Full non-slow suite: `uv run pytest tests/test_optimizer.py -x -q` — 91 passed, 4 skipped, 74 s.

- [x] **43.9d** Added `TestAlphaSurveyVsPublished::test_top_basin_within_published_l_infinity_ball` in `scripts/stencil_gen/tests/test_optimizer.py` — a slow-marked live-pipeline regression that drives `run_survey(n_seeds=1, base_seed=0, bounds=DEFAULT_BOUNDS[("E4","classical")])` with the same staged knobs as 43.9b (`inner_gate=3`, `inner_max_layer=3`, `validator_max_layer=6`, `top_k=5`, `n_restarts=20`, `max_evals=60`) and asserts containment (not identity) against Brady-Livescu's published α.
  - **Published-value source corrected vs plan text.** The plan said to read from `stencil_gen/alpha_extraction.py`, but no such module exists — the repo's `sweeps/alpha_extraction.py` only defines E2 production α's (`PRODUCTION_ALPHAS = {"E2": [...]}`). The canonical Brady-Livescu E4 constant lives at `sweeps.brady2d_sweep.CLASSICAL_E4_ALPHA = [-0.7733323791884821, 0.1623961700641681]` (identical copy in private `stencil_gen.benchmarks.brady2d_calibration._E4_CLASSICAL_ALPHA`). Test imports the public `sweeps.brady2d_sweep` constant and documents the divergence in the class docstring; no new published-value module was created.
  - **Why n_seeds=1 instead of the 20-seed production default.** 43.9b proved seed=0 lands feasibly in the Brady-Livescu basin at α ≈ [-0.81, 0.09] with `best_objective ≈ 4.83` in ~138 s. A single feasible seed is sufficient to populate one basin and exercise the `run_survey` aggregation path; the containment assertion does not require multi-basin diversity. Multi-seed diversity (`n_distinct_basins ≥ 3`) is a Completion-Criteria concern for a production run, not a CI regression gate — keeping this test at n_seeds=1 holds the slow-suite cost to ~2 min per run. The synthetic `TestAlphaBasinSurvey` class already covers multi-basin clustering / propagation / rendering deterministically.
  - Assertions: `n_feasible_seeds >= 1`, `len(basins) >= 1`, and `any(max(|basin.alpha - published|) <= 0.5 for basin in basins)`. The 0.5 L∞ tolerance matches the plan spec and the 43.5c `test_shgo_2d_classical_alpha` precedent. Observed: seed=0 basin at [-0.81, 0.09], L∞ distance 0.072 — well inside the 0.5 ball.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q -k "TestAlphaSurveyVsPublished" --run-slow` — 1 passed in 138 s. Non-slow suite: `uv run pytest tests/test_optimizer.py -x -q` — 91 passed, 5 skipped, 77 s.

- [x] **43.9d-r1** Reconciled the Completion Criteria with the published-value source correction made in 43.9d. Updated the bullet to name `sweeps/brady2d_sweep.py::CLASSICAL_E4_ALPHA` as the canonical source and cross-references 43.9d's rationale that `alpha_extraction.py` only holds E2 production α's. Plan-text-only change; no code edits.
  - File: `plans/43-stability-optimization-framework.md` (Completion Criteria section only — no code change).
  - Test: `grep -n "alpha_extraction" plans/43-stability-optimization-framework.md` — surviving hits sit in 43.9d's narrative and this 43.9d-r1 item, not in an active Completion Criteria assertion.

### 43.10 — L8 validation of optimizer winners

- [x] **43.10a** Wired `--validate-with-cpp` in `scripts/stencil_gen/sweeps/optimize.py`:
  - Added module-level helper `_run_cpp_validation(scheme, kernel, best_params, best_objective, *, N=31, t_final=5.0)` that re-runs the analytical winner at `max_layer=8` via `brady2d_stability_score(..., short_circuit=False)`. Using `short_circuit=False` guarantees L8 executes regardless of intermediate-layer diagnostics at the winner (the analytical cascade already produced a feasible verdict; L8 is an independent sanity check, not a gate).
  - Skip paths (return `None`, print a one-line reason, never mutate the result): empty `best_params`, non-finite `best_objective`, unsupported scheme/kernel pair (`_CPP_SUPPORTED_SCHEMES = ("E4",)`, `_CPP_SUPPORTED_KERNELS = ("classical", "tension", "gaussian", "multiquadric")`, matching the L8 dispatch table in `brady2d_stability._L8_SCHEME_TYPE`), or missing `SHOCCS_BINARY`. Each skip emits a user-visible reason so the CLI never silently swallows the `--validate-with-cpp` flag.
  - Failure path (L8 runs but `stable=False` or an exception is raised): logs a `WARNING:` line, returns the `{stable, final_linf, wall_time_s}` dict (with NaN on exception), and the caller's `best_objective` is **not** altered — per plan directive, the analytical verdict stands and L8 disagreement is diagnostic only.
  - Extended `_result_to_persist_dict` with an optional `cpp_validation: dict | None = None` parameter; when non-`None`, the persisted entry grows a top-level `cpp_validation: {stable, final_linf, wall_time_s}` sub-dict (the `BridgeResult` from `report.layer8` is intentionally dropped — it contains numpy arrays and subprocess metadata not worth round-tripping through JSON).
  - Updated `main()` to call `_run_cpp_validation` just before `_result_to_persist_dict`, threading the result through so it lands in both the `--update-known-values` and `--json-output` payloads.
  - Live smoke: `SYMPY_CACHE_SIZE=50000 uv run python -m sweeps optimize --scheme E4 --kernel tension --objective layer3.max_stab_eig --gate-layer 3 --max-layer 3 --bounds 0.5 20 --method Nelder-Mead --max-evals 40 --n-restarts 2 --validate-with-cpp --json-output /tmp/opt_cpp.json` → optimizer converges at σ≈1.6445 (best_objective=-1.22e-4), then L8 PASS with final_linf=6.99e-03 at N=31 t_final=5.0 (wall=0.12s); persisted JSON carries `cpp_validation: {stable: true, final_linf: 6.99e-3, wall_time_s: 0.12}`.
  - Skip-path smoke: empty `best_params`, non-finite `best_objective`, E2/tension, and a hypothetical `tension-penalty` kernel each print their skip reason and return `None` without touching the persisted dict.
  - Full optimizer suite `SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_optimizer.py -x -q` — 91 passed, 5 skipped in 76 s. Test class for this path (`TestOptimizeCppValidation`) lands in 43.10b.
  - File: `scripts/stencil_gen/sweeps/optimize.py`

- [ ] **43.10a-r1** Fix `_run_cpp_validation` PASS/FAIL criterion to match the cascade's L8 verdict. Currently the helper prints "L8 PASS" / "L8 FAIL" based only on `l8["stable"]`, but the cascade in `brady2d_stability.brady2d_stability_score` (and the CLI summary formatter at `brady2d_stability.py:200`) treats PASS as `stable AND final_linf <= L8_FINAL_LINF_TOL` (=1.0). A winner that runs without blow-up but overshoots the accuracy ceiling is the exact analytical-vs-C++ disagreement `--validate-with-cpp` is meant to surface, and it is currently reported as PASS with no warning. Required changes:
  - Import `L8_FINAL_LINF_TOL` from `stencil_gen.brady2d_stability` (or read it off the module) in `scripts/stencil_gen/sweeps/optimize.py` and reuse it in `_run_cpp_validation` — do not hard-code `1.0`.
  - Redefine the banner criterion as `passed = stable and final_linf <= L8_FINAL_LINF_TOL`. When `stable=True` but `final_linf > L8_FINAL_LINF_TOL`, emit the same `WARNING:` line the hard-blow-up branch uses today, with the reason worded explicitly as `L8 soft-failure: final_linf=... > L8_FINAL_LINF_TOL=...` so the diagnostic reads as a cascade disagreement, not a pass.
  - Leave the persisted schema alone (`cpp_validation: {stable, final_linf, wall_time_s}`) — consumers can re-derive the verdict, and the field name `stable` must keep its bridge-level meaning to match `brady2d_stability.BridgeResult`.
  - File: `scripts/stencil_gen/sweeps/optimize.py`
  - Test: extend `TestOptimizeCppValidation` (43.10b) with a case that monkeypatches `brady2d_stability_score` to return `layer8 = {"stable": True, "final_linf": 2.0, "wall_time_s": 0.1}` and asserts the CLI prints the soft-failure warning while leaving `best_objective` untouched.

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
43.1a → 43.1b → 43.1c → 43.1d → 43.1e  # skeleton + primitives + scope reconcile + plan cleanup
  ↓
43.2a → 43.2b                          # objective factory
  ↓
43.3a → 43.3b                          # baseline local (Nelder-Mead + COBYQA)
  ↓
43.4a → 43.4b                          # multi-start wrapper
  ↓
43.5a → 43.5b → 43.5c                  # SHGO + DE
  ↓
43.6a → 43.6b → 43.6c → 43.6d → 43.6e  # staged pipeline + fallback-extras/converged fix + deterministic re-order test + tautology-assertion fix
  ↓
43.7a → 43.7b → 43.7c                  # CLI
  ↓
43.8a → 43.8b → 43.8c                  # persistence + regression + persist gate/max_layer
  ↓
43.9a → 43.9a-r1 → 43.9b → 43.9b-r1 → 43.9b-r2 → 43.9c → 43.9c-r1 → 43.9d → 43.9d-r1  # classical-α (depends on all prior)  [43.9d done; 43.9d-r1 plan-text reconcile]
  ↓
43.10a → 43.10a-r1 → 43.10b            # L8 validation (r1 tightens PASS criterion)
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
- `python -m sweeps optimize --scheme E4 --kernel tension --objective layer3.max_stab_eig --gate-layer 3 --max-layer 3 --bounds 0.5 20 --method Nelder-Mead --max-evals 40` runs end-to-end, prints a feasible `best_params`, and respects the stated bounds. (The earlier specific-σ acceptance figure was dropped — see 43.3b.)
- `python -m sweeps optimize --scheme E4 --kernel classical --objective layer6.transient_growth_bound --method staged --n-restarts 20 --update-known-values` runs, respects `DEFAULT_BOUNDS[("E4","classical")]` (widened in 43.9a to admit the Brady-Livescu analytical feasible region), finds at least one feasible local minimum, and persists to `known_values.json["brady2d_optima"]["E4"]["classical"]`.
- `scripts/stencil_gen/benchmarks/alpha_basin_survey.py` with `n_seeds=20` reports at least 3 distinct basins for E4 classical-α (cross-checks Brady-Livescu's multi-modality finding of 101 E4 schemes at their full budget).
- The survey's top basin contains a point within 0.5 L∞ of Brady-Livescu's published E4 α (stored in `sweeps/brady2d_sweep.py` as `CLASSICAL_E4_ALPHA`; see 43.9d for why this is not `alpha_extraction.py`, which only holds E2 production α's).
- `TestRegressionBrady2DOptima` passes: re-runs stored optima and verifies each matches within 1% of the recorded objective.
- `cd scripts/stencil_gen && uv run pytest tests/ -x -q` continues to pass in under 60 seconds (new slow tests marked).
- Plan 44 (multi-objective Pareto via pymoo) can now start cleanly — the `make_objective` factory extends to weighted scalarization without refactoring, and a future NSGA-II caller can reuse `params_from_vector` unchanged.
