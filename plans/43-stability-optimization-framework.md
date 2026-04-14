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
  - Constants: `DEFAULT_BOUNDS` dict mapping `(scheme, kernel) -> list[tuple[float, float]]` with the bounds table above. E4 classical α has `[(-2.0, 2.0), (197.0/288.0, 2.0)]`.
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

- [ ] **43.8c** Close the regression gate-config gap exposed by 43.8b: the persistence schema (`sweeps/optimize.py::_result_to_persist_dict`) currently omits `gate_layer`, `max_layer`, and — for the staged method — `validator_max_layer`, while `TestRegressionBrady2DOptima` rebuilds the objective via `make_objective(scheme, kernel, report_field=objective)` with only the defaults. If the CLI was ever invoked with `--gate-layer != 3`, an explicit `--max-layer` that diverges from `_infer_max_layer(report_field)`, or (staged) `--validator-max-layer != 6`, the regression test would silently rebuild against a different feasibility gate / layer depth — either asserting mismatch on a legitimate winner, or passing spuriously at a close-but-wrong objective value.
  - Extend `_result_to_persist_dict` to record `gate_layer: int`, `max_layer: int` (the *effective* value — i.e. the inferred layer when `--max-layer` is None, so the persisted dict is self-describing and never records `None`), and, when `method == "staged"`, `validator_max_layer: int`. Use the resolved values from `args` (with inference via `_infer_max_layer` when `args.max_layer is None`), not the raw argparse strings.
  - Update `TestRegressionBrady2DOptima` in `tests/test_phs.py` to read `entry["gate_layer"]` and `entry["max_layer"]` from each persisted entry and thread them into `make_objective(..., gate_layer=..., max_layer=...)`. Keep a graceful skip (with a clear message) when an older entry predates these fields so pre-43.8c JSON doesn't break CI, but fail loudly if the fields are present and inconsistent with what `make_objective` would produce.
  - Extend the 43.8a contract test (`test_cli_update_known_values_additive_and_drops_history` in `tests/test_optimizer.py`) to assert the three new keys round-trip under both a non-staged and a staged path, including the inferred-vs-explicit `max_layer` branch.
  - Files: `scripts/stencil_gen/sweeps/optimize.py`, `scripts/stencil_gen/tests/test_phs.py`, `scripts/stencil_gen/tests/test_optimizer.py`.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py tests/test_phs.py -x -q -k "TestRegressionBrady2DOptima or test_cli_update_known_values"`.

### 43.9 — Classical-α E4_1 2D optimization

- [ ] **43.9a** Extend `params_from_vector`/`vector_from_params` for `kernel="classical"` and reconcile the 197/288 constraint with the Brady-Livescu published α:
  - **43.5c finding (must resolve here)**: the Brady-Livescu stored feasible point α ≈ `[-0.7733, 0.1624]` has `α₁ ≈ 0.162`, i.e. *below* 197/288. An 11×8 grid-probe of `DEFAULT_BOUNDS[("E4", "classical")] = [(-2, 2), (197/288, 2)]` finds zero L3-feasible points. So either the Python `_build_classical_diff_matrix` does not match the C++ 197/288 constraint, or the C++ constraint is misstated. Decide and document.
  - Confirm the Python `_build_classical_diff_matrix` accepts `alpha` and does NOT impose the 197/288 bound (the C++ imposes it at construction; the Python path should either mirror it or rely on the optimizer to respect bounds).
  - If the Python path accepts `alpha[1] < 197/288` without error but produces a singular D, the objective returns `+inf` via the `try/except` — acceptable.
  - `DEFAULT_BOUNDS[("E4", "classical")]` currently has `α₁ ≥ 197/288`; based on the reconciliation above, either (a) keep the bound and document a Python-vs-C++ divergence, or (b) widen the bound to include the published feasible region (e.g. `α₁ ≥ 0.05`) and explain why the C++ 197/288 constraint doesn't apply here. Add a test covering whichever decision is made.
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
- `python -m sweeps optimize --scheme E4 --kernel tension --objective layer3.max_stab_eig --gate-layer 3 --max-layer 3 --bounds 0.5 20 --method Nelder-Mead --max-evals 40` runs end-to-end, prints a feasible `best_params`, and respects the stated bounds. (The earlier specific-σ acceptance figure was dropped — see 43.3b.)
- `python -m sweeps optimize --scheme E4 --kernel classical --objective layer6.transient_growth_bound --method staged --n-restarts 20 --update-known-values` runs, respects the α₁ ≥ 197/288 bound, finds at least one feasible local minimum, and persists to `known_values.json["brady2d_optima"]["E4"]["classical"]`.
- `scripts/stencil_gen/benchmarks/alpha_basin_survey.py` with `n_seeds=20` reports at least 3 distinct basins for E4 classical-α (cross-checks Brady-Livescu's multi-modality finding of 101 E4 schemes at their full budget).
- The survey's top basin contains a point within 0.5 L∞ of Brady-Livescu's published E4 α (stored in `alpha_extraction.py`).
- `TestRegressionBrady2DOptima` passes: re-runs stored optima and verifies each matches within 1% of the recorded objective.
- `cd scripts/stencil_gen && uv run pytest tests/ -x -q` continues to pass in under 60 seconds (new slow tests marked).
- Plan 44 (multi-objective Pareto via pymoo) can now start cleanly — the `make_objective` factory extends to weighted scalarization without refactoring, and a future NSGA-II caller can reuse `params_from_vector` unchanged.
