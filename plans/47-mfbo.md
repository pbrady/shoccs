# Phase 47: Multi-Fidelity Bayesian Optimization (MF-BO)

**Goal:** Replace the hand-coded `run_staged_optimize` cheap-inner + expensive-validator heuristic with a principled multi-fidelity Bayesian optimizer that uses a Gaussian-process surrogate over the cascade's discrete fidelity levels and a cost-aware acquisition function. The MF-BO chooses `(x, m)` jointly to maximize expected information gain at the high-fidelity target per second of wall-time. Target benefit: 2× speedup vs `run_staged_optimize` on E4 classical-α with `layer7.max_spectral_abscissa` as the HF objective, measured by simple-regret-vs-wall-time over 5 seeds.

**Depends on:** Plan 41 (cascade), Plan 43 (`make_objective` infrastructure), Plan 45 (`make_multi_objective` + per-run JSON pattern), Plan 46 (cleanup of CLI surface and schema completeness; especially 46.0 + 46.2a that fixed `_report_to_dict` for full serialization).

**Background — why MF-BO and not just a better single-fidelity BO:**

Our cascade has a 5+ orders of magnitude cost spread (Plan 46 measurements):

| Layer | Wall time | Cost ratio (L1=1) | What it tests |
|---|---|---|---|
| L1 | 76 ms | 1.0 | GV dispersion (interior + boundary) |
| L3 | 38 ms | 0.5 | 1D advection eigenvalue |
| L3r | 486 ms | 6.4 | BL §4.2 reflecting-hyperbolic spectrum |
| L6 | 846 ms | 11 | Non-normality on 1D operator |
| L7 | 1434 ms | 19 | Full 2D varying-coefficient spectral abscissa |
| L7+nn | 17.5 s | 230 | L7 + non-normality (transient growth) |
| L8 | ~3 s | ~40 | Compiled C++ simulation (validator) |

Single-fidelity BO would query L7 at every iteration. Multi-fidelity BO learns from cheap layers and only spends L7 budget where the cheap layers say "promising." `run_staged_optimize` does this by hand with a fixed top-K threshold; MF-BO does it via a principled Pareto-optimal cost/benefit tradeoff inside the GP posterior.

**Critical observation from `docs/handoff/scientific_findings.md` finding #1:** L3 → L3r is **not** a refinement chain — they test different physics (1D periodic advection vs. reflecting BCs). So our model cannot use a Kennedy-O'Hagan autoregressive ladder; it needs a coregionalization (ICM) kernel that lets the data report the actual layer-pair correlations. Tension closures pass L3 universally but fail L3r at `max_spectral_abscissa ≈ 0.95` — the GP must learn this independence rather than have it baked in.

**Why BoTorch:**

- **Verified clean aarch64 install** (no source-build hack like nlopt). `botorch==0.17.2` (released 2026-03-04), `torch==2.11.0` (manylinux_2_28_aarch64 wheel on PyPI), `gpytorch==1.15.2` — all wheels, all resolves cleanly via `uv sync`.
- **First-class MF API:** `qMultiFidelityKnowledgeGradient`, `qMultiFidelityMaxValueEntropy`, `SingleTaskMultiFidelityGP`, `IndexKernel` (ICM), `AffineFidelityCostModel`, `InverseCostWeightedUtility` — all stable in 0.17.x, all documented in the `discrete_multi_fidelity_bo` tutorial.
- **PyTorch backend** keeps NumPy 2 compatibility (Emukit forces NumPy 1.26 via GPy; Trieste forces TF 2.16 with NumPy 1.26).
- **Active maintenance:** weekly commits, frequent point releases, integrates with Ax 1.0.

**What this plan does NOT do:**

- **Multi-objective MF-BO** (Pareto fronts at multiple fidelities). Plan 45's `pareto.py` covers single-fidelity Pareto; multi-objective MF-BO is a future extension.
- **Continuous fidelity in N** (BoTorch's `LinearTruncatedFidelityKernel` with `s ∈ [0, 1]`). The cascade's discrete layer indices are the natural fidelity axis; varying N within L7 is a separate dimension worth exploring later.
- **Autoregressive Kennedy-O'Hagan model** (`f_m(x) = ρ_{m-1} f_{m-1}(x) + δ_m(x)`). The L3 ↔ L3r independence rules out a single global AR chain. The ICM model lets the coregionalization matrix `B` be learned end-to-end. AR-style links between specific pairs (L7 → L8) are a future extension.
- **Learned cost model.** Plan 46 measured costs to within ±10% per layer (except L7 which is kernel-dependent — 4.7× spread for tension); we use a constant `cost(m)` table. Per-layer GP cost models defer to a follow-up.
- **L8 in the GP.** L8 (the C++ simulation) is the validator at the end, not a fidelity inside the GP. Adding L8 as a 6th fidelity needs ≥10 L8 evaluations to anchor a `B_{·,8}` row meaningfully — defer until we have those data.
- **MES acquisition (`qMultiFidelityMaxValueEntropy`).** Add as a fallback / extension if KG diagnostics show pathology (Gumbel sampling collapse, multi-modal posterior trapping). MVP ships KG only; the swap is a one-line change once the rest of the infrastructure exists.
- **GPU.** Our problem is 1–2D; a GP fit takes microseconds on CPU. We use `torch.device("cpu")` everywhere and pull CPU-only PyTorch wheels via `https://download.pytorch.org/whl/cpu`.
- **Plan 48 — Brady-Livescu 1D Euler.** Renumbered from 47 → 48 by plan 46.7a.

**Read first:**

- `docs/handoff/scientific_findings.md` finding #1 (tension fails L3r), #2 (multi-modal classical-α), #9 (multi-objective trade-offs) — shape the validation strategy.
- `docs/handoff/framework_architecture.md` — cascade overview.
- `scripts/stencil_gen/stencil_gen/optimizer.py` lines 75–83 (`DEFAULT_BOUNDS`), 88–131 (`OptimizeResult`), 140–189 (`params_from_vector` / `vector_from_params`), 194–238 (`extract_field`), 241–263 (`_LAYER_PREFIX_RE`, `_FIELD_LAYER_ALIAS`, `_infer_max_layer`), 266–333 (`make_objective`), 707–763 (`_report_to_dict`, after plan 46.2a). Reuse all these.
- `scripts/stencil_gen/stencil_gen/pareto.py` (entire file) — the pattern for "new optimization driver in its own module": `ParetoPoint`/`ParetoResult` dataclasses, `make_multi_objective` factory, `_PARETO_SENTINEL = 1e12`, `run_nsga2`, `_HVCallback`. The MF-BO module mirrors this structure.
- `scripts/stencil_gen/sweeps/pareto.py` (entire file) — CLI surface to mirror.
- `scripts/stencil_gen/sweeps/_pareto_io.py` (entire file) — per-run JSON persistence pattern. The MF-BO module uses the same `<scheme>_<kernel>_<mangled>_<seed>.json` filename scheme.
- `scripts/stencil_gen/sweeps/__main__.py` — subcommand dispatch pattern.
- `scripts/stencil_gen/stencil_gen/brady2d_stability.py` lines 80–110 (`StabilityReport` dataclass), 1140+ (`brady2d_stability_score`).
- BoTorch tutorial: https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/ — direct template for our problem shape.
- BoTorch cost-aware tutorial: https://botorch.org/docs/tutorials/cost_aware_bayesian_optimization/.
- Wu et al. 2020 (https://arxiv.org/abs/1903.04703) — practical MF-BO for hyperparameter tuning, the cost-weighted KG formulation.

**Test commands:**

```bash
# Fast: MF-BO unit tests (mock GP, mock objective)
cd scripts/stencil_gen && uv run pytest tests/test_bo.py tests/test_sweep_bo.py -x -q -k "TestBOResult or TestMakeMultiFidelityObjective or TestCostModel or TestDOE or TestAcquisition or TestBOCLI or TestBOIO"

# Slow integration: AugmentedBranin synthetic + failure-mode regressions
cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestBranin or TestBiasMisspec or TestCostMisspec or TestMultiModal" --run-slow

# CLI smoke: tiny MF-BO run on E4 tension (fast convergence, low cost)
cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python -m sweeps bo \
    --scheme E4 --kernel tension \
    --objective layer3.max_stab_eig \
    --cheap-fidelities 1 \
    --bounds 0.5 20 --budget-evals 15 --seed 1 --persist
```

---

## Items

### 47.0 — Setup: dependencies + skeleton module

- [x] **47.0a** Add BoTorch + dependencies to `pyproject.toml` and verify clean aarch64 install:
  - `dependencies` += `"torch>=2.2,<3"`, `"gpytorch>=1.15,<2"`, `"botorch>=0.17,<0.18"`. Pin botorch at `<0.18` because the 0.17.x → 0.18 transition typically removes deprecated APIs (the 0.17.0 transition removed `get_fitted_map_saas_ensemble`, `qMultiObjectiveMaxValueEntropy`, `FullyBayesianPosterior`).
  - Add `[tool.uv]` section: `extra-index-url = ["https://download.pytorch.org/whl/cpu"]` to skip the ~3 GB of NVIDIA CUDA wheels the default index pulls. Our GP fits run in microseconds — no GPU benefit.
  - The existing `[tool.uv.sources]` nlopt git override stays as-is from plan 46.0a.
  - Verify: `cd scripts/stencil_gen && uv sync` succeeds, `uv run python -c "import botorch; from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient; from botorch.models import SingleTaskMultiFidelityGP; from botorch.models.cost import AffineFidelityCostModel; print('botorch', botorch.__version__)"` returns the version cleanly.
  - Create empty skeleton module `scripts/stencil_gen/stencil_gen/bo.py` with module docstring citing Wu et al. 2020 + the BoTorch discrete-fidelity tutorial. Module imports: `numpy as np`, `torch`, `botorch`, plus the reused symbols from `optimizer.py` (`params_from_vector`, `vector_from_params`, `extract_field`, `_FIELD_LAYER_ALIAS`, `_infer_max_layer`, `_report_to_dict`, `DEFAULT_BOUNDS`).
  - File: `scripts/stencil_gen/pyproject.toml`, `scripts/stencil_gen/stencil_gen/bo.py` (new)
  - Test: `cd scripts/stencil_gen && uv sync && uv run python -c "import stencil_gen.bo; import botorch, torch, gpytorch; print('botorch', botorch.__version__, 'torch', torch.__version__, 'gpytorch', gpytorch.__version__)"`
  - **Done 2026-04-27.** Resolved versions: `botorch==0.17.2`, `torch==2.11.0+cpu`, `gpytorch==1.15.2` — all wheels, ~141 MB CPU torch (no NVIDIA stack pulled).
  - **Resolution note:** added `[tool.uv]` `index-strategy = "unsafe-best-match"` alongside the CPU torch `extra-index-url` because the CPU index ships an older `numpy` package and uv's default first-index rule otherwise refuses to resolve `numpy>=2.4.4` from PyPI when both indexes carry it.
  - **Import-path correction for downstream items (47.3a):** the plan cites `botorch.acquisition.multi_fidelity.qMultiFidelityKnowledgeGradient`, but in `botorch==0.17.2` that path does not exist — `qMultiFidelityKnowledgeGradient` lives at `botorch.acquisition.knowledge_gradient`. Use `from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient` in 47.3a. Same caveat for MES if/when adopted: it is at `botorch.acquisition.max_value_entropy_search`, not `multi_fidelity`.
  - Skeleton `bo.py` imports `numpy`, `torch`, `botorch` plus the seven reused symbols from `optimizer.py`; `__all__ = []` until 47.1a populates it.
  - Existing fast suite (865 passed, 137 skipped, 1 xfailed) green after install — no regression from the new dependency stack.

- [x] **47.0b** Fix stale `botorch.acquisition.multi_fidelity` import-path references in this plan file. The 47.0a Resolution note documents the correction (`qMultiFidelityKnowledgeGradient` lives at `botorch.acquisition.knowledge_gradient` in `botorch==0.17.x`), but two surrounding text references still cite the non-existent path:
  - 47.0a "Verify" bullet (currently line ~84): replace `from botorch.acquisition.multi_fidelity import qMultiFidelityKnowledgeGradient` with `from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient` so the documented verification command actually runs to completion.
  - Completion-criteria second bullet (currently line ~421): same replacement so the criterion is achievable.
  - Leave the 47.0a Resolution note unchanged — it explains the historical confusion.
  - File: `plans/47-mfbo.md`
  - Test: `grep -n "acquisition\.multi_fidelity" plans/47-mfbo.md` returns only the Resolution-note line(s) that explain the correction (not the verify command or completion criterion).
  - **Done 2026-04-27.** Replaced the two stale `botorch.acquisition.multi_fidelity` references (47.0a Verify bullet + completion criterion) with `botorch.acquisition.knowledge_gradient`. Remaining grep matches (lines 90, 94, 95) are all explanatory text describing the correction itself.

### 47.1 — Core dataclasses + multi-fidelity objective factory

- [x] **47.1a** Add `BOResult` and `BOPoint` frozen dataclasses + `_BO_SENTINEL` to `bo.py`, modeled on `pareto.py`'s `ParetoPoint`/`ParetoResult`:
  ```python
  _BO_SENTINEL: float = 1e12  # finite sentinel; KG/MES break on inf

  @dataclass(frozen=True)
  class BOEval:
      x: np.ndarray              # design vector
      params: dict               # params_from_vector(kernel, x)
      fidelity: int              # cascade layer index
      value: float               # extracted field value at this fidelity
      wall_time: float           # measured per-eval (for empirical cost calibration)
      report: dict               # _report_to_dict serialization

  @dataclass(frozen=True)
  class BOResult:
      best_x: np.ndarray         # incumbent at HF
      best_params: dict
      best_objective: float      # posterior mean at incumbent, target fidelity
      best_report: dict          # full HF report
      method: str                # "BoTorch-qMFKG" (or fallback name)
      scheme: str
      kernel: str
      bounds: tuple[tuple[float, float], ...]
      fidelity_levels: tuple[int, ...]   # sorted
      hf_level: int                       # max(fidelity_levels)
      report_fields_by_layer: dict[int, str]
      cost_model: dict[int, float]        # actual cost table used
      n_evals_per_fidelity: dict[int, int]
      wall_time_per_fidelity: dict[int, float]
      total_compute_time: float
      eval_history: tuple[BOEval, ...]    # full per-eval log
      hf_eval_history: tuple[BOEval, ...] # filtered to HF only (convergence trace)
      gp_hyperparameters: dict            # final lengthscale, outputscale, noise, B matrix
      seed: int
      converged: bool
      stop_reason: str                    # "budget" | "variance" | "stagnation" | "error"
      extras: dict                        # e.g., baseline OptimizeResult, cpp_validation
  ```
  Frozen with tuples (not lists) for immutability — same convention as `ParetoResult`.
  - File: `scripts/stencil_gen/stencil_gen/bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestBOResult"`
  - **Done 2026-04-27.** Added `_BO_SENTINEL = 1e12`, `BOEval` (6 fields: x, params, fidelity, value, wall_time, report), `BOResult` (22 fields per the plan spec). `__all__` updated to export the three names. The plan named the per-eval class "BOPoint" in the heading but the schema in the body says `BOEval`; the body is authoritative — followed it. Verified by smoke check: instantiate both, assignment raises `FrozenInstanceError`, tuple-typed fields hold tuples, `dataclasses.asdict(BOResult)` round-trips. `tests/test_bo.py` will be created in 47.1c — the plan's `Test:` line for 47.1a is forward-looking. Adjacent `tests/test_pareto.py` still green (22 passed, 1 skipped).

- [x] **47.1b** Add `make_multi_fidelity_objective(scheme, kernel, report_fields_by_layer, *, gate_layer=None) -> Callable[[np.ndarray, int], tuple[float, float, dict]]`:
  - `report_fields_by_layer`: dict mapping layer index → dotted path, e.g. `{1: "layer1.boundary_gv_err", 3: "layer3.max_stab_eig", 7: "layer7.max_spectral_abscissa"}`. The HF layer is `max(report_fields_by_layer)`; the HF field is the optimization target.
  - Auto-infer `gate_layer = min(report_fields_by_layer) - 1` if not specified, max(0, ...). The cheapest fidelity's own layer is then a usable result, and only failures *below* the cheapest fidelity gate to sentinel.
  - Returns closure `f(x, m) -> (value, wall_time, report_dict)` (3-tuple, NOT just the value, so the BO loop can record per-eval wall time without a side channel — cleaner than the closure-with-defaultdict pattern Agent 4 sketched).
  - On any of: gate trip, shape-mismatch in `params_from_vector`, exception from `brady2d_stability_score`, `m not in report_fields_by_layer` — return `(_BO_SENTINEL, measured_wall_time, {"error": str(exc)})`.
  - Internally: time the `brady2d_stability_score(scheme, kernel, params, max_layer=m, short_circuit=True)` call with `time.perf_counter()`. Extract field via `extract_field`. Serialize via `_report_to_dict`.
  - Validation: raise `ValueError` at factory time (not at call time) if any field's `_infer_max_layer` is greater than the layer it's keyed under (you can't extract `layer7.*` at `m=3`).
  - File: `scripts/stencil_gen/stencil_gen/bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestMakeMultiFidelityObjective"`
  - **Done 2026-04-27.** Factory added with all four sentinel paths + factory-time field-vs-layer validation + empty-mapping guard (an extra `ValueError` not enumerated in the plan body — natural extension since the auto `min(...)` would otherwise raise an opaque `IndexError`). Also returns sentinel when `extract_field` returns non-finite (e.g. layer present but field path missing) — preserves the BO-must-have-finite-training-Y invariant. Smoke check at BL's published α `[-0.7733, 0.1624]` returns `L1 boundary_gv_err = 2.1055e-02` (finite) and `L3 max_stab_eig = -1.8083e-04` (finite) with positive wall-times. Unknown-fidelity, shape-mismatch, and L7-keyed-under-L3 all rejected as expected. `tests/test_bo.py` is created in 47.1c — the plan's `Test:` line is forward-looking. Adjacent regression suite (pareto + optimizer = 154 tests) green.

- [x] **47.1c** Tests in `tests/test_bo.py` (new file) — `TestBOResult` (3) + `TestMakeMultiFidelityObjective` (8):
  - `TestBOResult::test_frozen_dataclasses` — assignment raises `FrozenInstanceError`.
  - `TestBOResult::test_eval_history_is_tuple_not_list` — schema enforcement.
  - `TestBOResult::test_serializable_via_json_dumps` — round-trip works through a numpy + dataclass-aware encoder (build the encoder in 47.4c; for now just assert `dataclasses.asdict()` succeeds).
  - `TestMakeMultiFidelityObjective::test_returns_3tuple` — factory returns a closure that yields `(value, wall_time, report)` for valid `(x, m)`.
  - `test_sentinel_on_gate_trip` — passing infeasible `x` returns sentinel value, finite wall_time.
  - `test_sentinel_on_shape_mismatch` — wrong-shape `x` returns sentinel without raising.
  - `test_unknown_fidelity_returns_sentinel` — `f(x, 99)` (not in `report_fields_by_layer`) returns sentinel.
  - `test_field_layer_validation_at_factory_time` — `make_multi_fidelity_objective(..., {3: "layer7.max_spectral_abscissa"})` raises at factory time (you can't extract layer7 at max_layer=3).
  - `test_finite_at_known_feasible_point` — `α = [-0.7733, 0.1624]` (BL published optimum) returns finite values for L1, L3, and L3r.
  - `test_gate_layer_default` — auto-inferred to `min(layers) - 1`.
  - `test_gate_layer_explicit_override` — explicit kwarg preserved.
  - `test_wall_time_recorded` — measured wall_time is positive and roughly matches `time.perf_counter` deltas.
  - File: `scripts/stencil_gen/tests/test_bo.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestBOResult or TestMakeMultiFidelityObjective"`
  - **Done 2026-04-27.** Created `tests/test_bo.py` with 14 tests (3 `TestBOResult` + 11 `TestMakeMultiFidelityObjective`), all green in 3.7 s. Three deviations from the plan worth flagging for downstream items:
    - `test_finite_at_known_feasible_point` exercises `m ∈ {1, 3}` only (the plan listed `{1, 3, 3r}` aka `layer_bl42`). L3r is excluded to keep the test in the fast suite — adding `layer_bl42` triggers the `_RESOLVED_FRAC` 1D BL §4.2 eigenvalue problem that costs ~0.5 s, pushing this single test past most fast-suite-budget guidance. Promote to a `@pytest.mark.slow` companion in 47.6 if 3-fidelity coverage of the BL anchor is wanted.
    - Added two extra tests beyond the plan's 8: `test_rejects_empty_mapping` (factory guard added in 47.1b) and `test_sentinel_when_field_path_missing` (covers the non-finite-extract sentinel path also added in 47.1b — both behaviours were specced in 47.1b's body but not enumerated under 47.1c).
    - Test for "explicit gate_layer override" used `gate_layer=5` with a non-existent failed_layer=1 setup so the override is observable (passes when override active, would fail with default `gate_layer=0` since `1 > 0`). Matches the plan's spirit: prove the explicit kwarg is honoured.
    - Adjacent regression suite (`tests/test_pareto.py tests/test_optimizer.py tests/test_brady2d_stability.py tests/test_bo.py`) green: 261 passed, 23 skipped in 270 s. No regression from the new dependency stack or factory.

### 47.2 — GP + cost model + DOE

- [ ] **47.2a** Add `build_mf_gp(train_X, train_Y, fidelity_dim, num_fidelities, *, rank=2) -> SingleTaskMultiFidelityGP` to `bo.py`:
  - `train_X` shape `(N, d+1)` with the last column = fidelity index (integer 0..num_fidelities-1, NOT the literal layer index — internal indexing).
  - `train_Y` shape `(N, 1)` — scalar objective values (not the sentinel rows; those are filtered before GP fit).
  - Kernel: outer product of `MaternKernel(nu=2.5, ard_num_dims=d)` on the design columns and `IndexKernel(num_tasks=num_fidelities, rank=rank)` on the fidelity column. `IndexKernel` is BoTorch/GPyTorch's ICM implementation: parameterizes `B = W Wᵀ + diag(κ)` with `W` rank-`r` and learned end-to-end.
  - Likelihood: `GaussianLikelihood` with `noise_constraint=GreaterThan(1e-9)` to prevent Cholesky failures when the cascade is essentially noise-free.
  - Wrap with `SingleTaskMultiFidelityGP` from `botorch.models` (which expects fidelity-aware kernels and provides MF-aware posterior projection).
  - Fit hyperparameters via `fit_gpytorch_mll(ExactMarginalLogLikelihood)` from `botorch.fit`.
  - Return the fitted model.
  - File: `scripts/stencil_gen/stencil_gen/bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestMFGP"`

- [ ] **47.2b** Add cost model construction `build_cost_model(cost_table: dict[int, float], fidelity_dim: int) -> InverseCostWeightedUtility` to `bo.py`:
  - Default `cost_table` from plan 46 measurements: `{1: 0.076, 3: 0.038, 6: 0.846, 7: 1.434}` for {L1, L3, L6, L7}; for {L1, L3, L3r, L6, L7}: `{1: 0.076, 3: 0.038, 5: 0.486, 6: 0.846, 7: 1.434}` (using sequential internal indices 0..4 mapping to layers; or use a `_LAYER_TO_INDEX` dict to keep external API stable).
  - Implementation: use `AffineFidelityCostModel(fidelity_weights={fidelity_dim: <ratio>}, fixed_cost=<floor>)` parameterized to produce the table values at integer fidelity indices. `AffineFidelityCostModel` natively supports continuous fidelity in `[0, 1]`; for discrete, use `GenericDeterministicModel` with a step function returning `cost_table[m]`.
  - Apply cost floor: `c'(m) = max(c(m), 0.05 * c(hf))` to prevent acquisition over-exploitation of cheapest layer (Agent 2 mitigation).
  - Wrap in `InverseCostWeightedUtility(cost_model=..., use_mean=True)`.
  - Expose `DEFAULT_COST_TABLE` as module-level constant for reuse + persistence.
  - File: `scripts/stencil_gen/stencil_gen/bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestCostModel"`

- [ ] **47.2c** Add `build_initial_design(bounds, fidelity_levels, *, n_init=None, hf_anchors=3, mid_anchors=2, seed=0) -> tuple[np.ndarray, np.ndarray]`:
  - `n_init` default = `5*d + 3` (Loeppky et al. 2009 rule).
  - Sobol' sequence in `x` via `torch.quasirandom.SobolEngine(d, scramble=True, seed=seed)`. Scale to `bounds`.
  - **Stratified fidelity allocation**: 70% cheapest fidelity, 20% mid (median fidelity_levels by cost), 10% HF — so for `n_init=13` and 5 fidelities: 9 × cheap, 3 × mid, 2 × HF. Pair the HF anchor points at the same `x` as 3 of the cheap points (paired evals essential for `B` matrix identification — Agent 2 pitfall #1).
  - Returns `(X_init, fid_indices)` — `X_init` is `(n_init_total, d)`, `fid_indices` is `(n_init_total,)` (integer fidelity indices 0..K-1, NOT layer numbers — the BO module is the only place that does this internal indexing).
  - File: `scripts/stencil_gen/stencil_gen/bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestDOE"`

- [ ] **47.2d** Tests in `tests/test_bo.py` — `TestMFGP` (4) + `TestCostModel` (4) + `TestDOE` (5):
  - `TestMFGP::test_gp_fits_on_synthetic_data` — 10-point synthetic data on a quadratic; assert posterior mean at training points within 1e-3 of training Y (after fit).
  - `test_index_kernel_correlation_matrix_psd` — extract the learned `B` matrix; verify positive semi-definite (eigenvalues ≥ 0).
  - `test_seed_determinism` — same seed produces identical hyperparameters across runs.
  - `test_noise_floor_respected` — likelihood noise stays ≥ 1e-9 even on noise-free data.
  - `TestCostModel::test_default_table_matches_plan_46_measurements` — assert `DEFAULT_COST_TABLE` keys + values match the documented table.
  - `test_inverse_cost_weighted_utility_construction` — instantiate without errors.
  - `test_cost_floor_applied` — set `c(L1) = 0.001` and `c(L7) = 1.0`; assert effective cost(L1) ≥ 0.05 (floor active).
  - `test_cost_table_persisted_in_BOResult` — `BOResult.cost_model` reflects the actual table used (not `None`).
  - `TestDOE::test_n_init_default` — for `d=2` returns 13 points.
  - `test_fidelity_stratification` — 70/20/10 allocation matches.
  - `test_hf_anchor_paired_with_cheap` — at least 3 (x, fid=hf) points share x with (x, fid=cheap) points.
  - `test_seed_determinism` — same seed, same `(X, fid)`.
  - `test_bounds_respected` — every point in `X_init` is within `bounds`.
  - File: `scripts/stencil_gen/tests/test_bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestMFGP or TestCostModel or TestDOE"`

### 47.3 — Acquisition + BO loop

- [ ] **47.3a** Add `build_acquisition(model, cost_utility, target_fidelity_index, *, num_fantasies=64, candidate_set_size=512) -> qMultiFidelityKnowledgeGradient` to `bo.py`:
  - Construct `qMultiFidelityKnowledgeGradient(model=model, cost_aware_utility=cost_utility, num_fantasies=num_fantasies, project=project_to_target_fidelity)`.
  - `project_to_target_fidelity`: closure that snaps the fidelity column to `target_fidelity_index` for the inner argmax of the posterior mean. BoTorch tutorial provides the recipe.
  - For optimization: use `optimize_acqf_mixed` (continuous over `x`, discrete over `m`) with `q=1` (sequential, not batch — our HF cost is too high to amortize batches).
  - Return both the acquisition function AND a callable `optimize(bounds, fidelity_choices) -> tuple[np.ndarray, int, float]` that returns `(x_next, fidelity_next, acq_value)`.
  - Document a clear comment: "If KG diagnostics show degeneracy (Gumbel sampling collapse, all fantasies within 1e-6), swap to qMultiFidelityMaxValueEntropy at line N — single-line change."
  - File: `scripts/stencil_gen/stencil_gen/bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestAcquisition"`

- [ ] **47.3b** Add `run_mfbo(scheme, kernel, report_fields_by_layer, bounds, *, budget_evals=None, budget_seconds=None, cost_table=None, seed=0, n_init=None, num_fantasies=64, verbose=False) -> BOResult`:
  - Build the multi-fidelity objective via 47.1b's factory.
  - Build initial design via 47.2c, evaluate every `(x_i, m_i)` pair at the real objective, assemble training tensors.
  - Filter sentinel rows from training data (don't fit GP on `_BO_SENTINEL` values).
  - Loop:
    1. Fit GP via 47.2a's `build_mf_gp`.
    2. Build acquisition + cost utility via 47.3a's `build_acquisition`.
    3. `optimize_acqf_mixed` to get `(x_next, fid_next)`.
    4. Evaluate the objective; record into `eval_history`.
    5. Check stopping criteria (budget / variance / stagnation).
    6. Append to training data.
  - Stopping:
    - **Primary:** `budget_evals` total evaluations OR `budget_seconds` wall-time (mutually exclusive; one must be set).
    - **Variance guard:** posterior variance at incumbent at HF below `1e-6 * (f_max - f_min)^2` ⇒ early-exit, `stop_reason="variance"`.
    - **Stagnation guard:** no improvement in `min(hf_eval_history)` over last `K=10` HF evals ⇒ exit, `stop_reason="stagnation"`.
  - Recommendation rule: `x_inc = argmin_x μ_n(x, m=hf)` evaluated on a 1024-point Sobol' grid (use posterior MEAN, not best-observed — standard for noisy / multi-fidelity GPs).
  - Final HF evaluation at the recommended `x_inc` to populate `best_objective` + `best_report` from real data, not GP posterior.
  - Build `BOResult` with full eval history, per-fidelity counts and wall times, final GP hyperparameters (extract from `model.covar_module.state_dict()`).
  - Determinism: seed `torch.manual_seed(seed)` + `np.random.seed(seed)` + Sobol' engine `seed=seed` + `optimize_acqf` `options={"seed": seed}`.
  - File: `scripts/stencil_gen/stencil_gen/bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestRunMFBO"`

- [ ] **47.3c** Tests in `tests/test_bo.py` — `TestAcquisition` (3) + `TestRunMFBO` (5):
  - `TestAcquisition::test_qmfkg_constructor` — instantiate without errors on a fitted GP.
  - `test_optimize_acqf_mixed_returns_valid_point` — returned `x_next` within bounds, `fid_next` in fidelity choices.
  - `test_acquisition_value_finite` — for a non-degenerate GP, returned acq value is finite + non-zero.
  - `TestRunMFBO::test_seed_determinism` — same seed, same `best_x` to within 1e-6.
  - `test_budget_evals_respected` — `budget_evals=20` ⇒ `sum(n_evals_per_fidelity.values()) == 20`.
  - `test_stop_reason_recorded` — synthetic objective that converges fast triggers `stop_reason="variance"`.
  - `test_sentinel_rows_filtered_from_gp` — objective returns sentinel for half the initial design; verify GP only fits on finite-value rows; `BOResult.extras["n_sentinel_filtered"]` ≥ 1.
  - `@pytest.mark.slow def test_synthetic_quadratic_2d` — 2D quadratic `f(x, m) = (x-x*)^T(x-x*) + bias(m)`; assert MF-BO converges to `x*` within 1e-2 in ≤20 evals; verify ≥ 30% of evals at cheap fidelity (cost-aware working).
  - File: `scripts/stencil_gen/tests/test_bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestAcquisition or TestRunMFBO"`

### 47.4 — CLI + persistence

- [ ] **47.4a** Create `scripts/stencil_gen/sweeps/bo.py` CLI module mirroring `sweeps/pareto.py`:
  - `--scheme {E2,E4}`, `--kernel {classical,tension,gaussian,multiquadric}`, `--bounds LO HI [LO HI ...]` (with auto-default from `DEFAULT_BOUNDS`).
  - `--objective FIELD` — single dotted path; the HF objective. `_infer_max_layer(field)` gives the HF layer.
  - `--cheap-fidelities N [N ...]` — list of layer indices, e.g. `1 3` or `1 3 6`.
  - `--fidelity-fields LAYER=FIELD [...]` — explicit per-layer field overrides; otherwise default canonical fields (`1=layer1.boundary_gv_err`, `3=layer3.max_stab_eig`, `5=layer_bl42.max_spectral_abscissa` (we use index 5 for L3r since it's between L3 and L6 in cost), `6=layer6.transient_growth_bound`, `7=layer7.max_spectral_abscissa`).
  - `--budget-evals N` (default 60) OR `--budget-seconds S` — one or the other, validated at parse time.
  - `--n-init N`, `--num-fantasies N`, `--seed N` (default 1).
  - `--cost-model {constant,empirical}` (default `constant`; `empirical` for future learned cost — flag accepted but raises `NotImplementedError` for now).
  - `--persist`, `--validate-with-cpp`, `--verbose`.
  - `--baseline {staged,none}` (default `none`) — when `staged`, runs `run_staged_optimize` alongside MF-BO with same `(scheme, kernel, bounds)` and same eval-budget for fair comparison; serialized into `BOResult.extras["baseline"]`.
  - `_resolve_bounds`, `_validate_kernel_bounds_dim` — copy from `sweeps/pareto.py`. (Or factor into `sweeps/_bounds.py` as a separate cleanup item — leave for plan 48 cleanup.)
  - `main(argv)` orchestrates: parse → run_mfbo → optionally run_staged_optimize → optionally validate-with-cpp → optionally persist → print summary table (best_x, best_objective, evals/cost per fidelity, baseline comparison if present).
  - File: `scripts/stencil_gen/sweeps/bo.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_bo.py -x -q -k "TestBOCLI"`

- [ ] **47.4b** Register `bo` subcommand in `scripts/stencil_gen/sweeps/__main__.py`:
  - Add `from .bo import main as bo_main` at top.
  - Add `sub_bo = subparsers.add_parser("bo", help="Multi-fidelity Bayesian optimization (BoTorch qMFKG)")` with the full argparse surface from 47.4a (mirror the pareto sub-parser pattern). **Use `default=None` for `--gate-layer`-style arguments** to avoid the plan-46.0 dispatch bug.
  - Hook execution in the `if args.command == "bo": ...` branch with conditional forwarding (omit args if `None`, exactly like pareto's pattern).
  - Do NOT add `bo` to `_run_all()` — same exclusion as `optimize` and `pareto` (too expensive to run blindly).
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps bo --help; uv run python -m sweeps --help | grep -E '^  (bo|pareto|optimize)'`

- [ ] **47.4c** Create `scripts/stencil_gen/sweeps/_bo_io.py` with persistence helpers, modeled on `_pareto_io.py`:
  - `BO_RUNS_DIR: Path = Path(__file__).parent / "bo_runs"`.
  - `save_bo_run(result: BOResult, directory: Path = BO_RUNS_DIR) -> Path`:
    - Filename: `{scheme}_{kernel}_{mangled_objective}_{seed}.json`. Including seed avoids clobbering across replicates.
    - Schema: `OrderedDict` with explicit key order matching the dataclass field order, plus an `extras` tail.
    - `_BOEncoder(json.JSONEncoder)`: handles `np.ndarray`, `np.generic`, dataclasses (via `asdict`), `Path`, `complex` (the `complex` handler is critical because `KreissResult.witness_s` may flow through if L2 is in fidelity layers — see plan 46.2b).
  - `load_bo_run(path: Path) -> dict` — raw dict for regression tests to rebuild from.
  - `iter_bo_runs(directory: Path = BO_RUNS_DIR) -> Iterator[Path]` — sorted glob `*.json`.
  - Create `sweeps/bo_runs/.gitkeep` so the empty directory tracks (matches `pareto_fronts/` convention).
  - File: `scripts/stencil_gen/sweeps/_bo_io.py` (new), `scripts/stencil_gen/sweeps/bo_runs/.gitkeep` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_bo.py -x -q -k "TestBOIO"`

- [ ] **47.4d** Tests in `tests/test_sweep_bo.py` (new) — `TestBOCLI` (6) + `TestBOIO` (5):
  - `TestBOCLI::test_argparse_minimal_invocation` — mock `run_mfbo`, verify CLI dispatch.
  - `test_argparse_rejects_no_budget` — passing neither `--budget-evals` nor `--budget-seconds` raises.
  - `test_argparse_rejects_both_budgets` — passing both raises.
  - `test_argparse_rejects_bad_field_layer` — `--objective layer7.foo --cheap-fidelities 8` raises (cheap > HF).
  - `test_dispatch_via_main` — `python -m sweeps bo --help` exits 0.
  - `test_baseline_staged_invokes_run_staged_optimize` — mock both; verify both called.
  - `TestBOIO::test_save_bo_run_creates_file` — synthetic `BOResult`, `tmp_path`, verify file exists.
  - `test_roundtrip_preserves_eval_history` — save + load, assert all `BOEval` fields round-trip.
  - `test_serializer_handles_complex` — inject a complex value into `extras`, verify no `TypeError`.
  - `test_filename_includes_seed` — same scheme/kernel/objective with different seeds → different files.
  - `test_iter_bo_runs_sorted` — multiple files in `tmp_path`, verify sorted iteration.
  - File: `scripts/stencil_gen/tests/test_sweep_bo.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_bo.py -x -q -k "TestBOCLI or TestBOIO"`

### 47.5 — Integration features

- [ ] **47.5a** Wire `--validate-with-cpp` into `sweeps/bo.py::main`. After `run_mfbo` returns, when the flag is set: call `brady2d_stability_score(scheme, kernel, result.best_params, max_layer=8, layer8_N=31, layer8_t_final=5.0)`. Capture `{l8_final_linf, l8_stable, cpp_cutcell_violates_197_288, wall_time_s}` into `result.extras["cpp_validation"]`. Skip cleanly with a logged "skipped" message if the kernel is not in `_CPP_SUPPORTED_KERNELS` (reuse from `sweeps/optimize.py:44`) or if `SHOCCS_BINARY` doesn't exist. **Validation must run BEFORE persistence** so the persisted JSON includes the cpp_validation payload (lesson from plan 45.5a.1). Add explanatory comment.
  - File: `scripts/stencil_gen/sweeps/bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_bo.py -x -q -k "TestValidateWithCpp"`

- [ ] **47.5b** Wire `--baseline staged` into `sweeps/bo.py::main`. When the flag is set: after `run_mfbo`, run `run_staged_optimize(scheme, kernel, result.report_fields_by_layer[result.hf_level], bounds, n_restarts=10, seed=args.seed)`. Wall-clock time both runs; serialize the baseline `OptimizeResult` into `result.extras["baseline"]` via `_result_to_persist_dict(...)` from `sweeps/optimize.py`. Print a side-by-side comparison table: `(method, best_objective, total_compute_time, n_evals_at_HF)` for both. Both runs use the same `args.seed` for fairness.
  - File: `scripts/stencil_gen/sweeps/bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_bo.py -x -q -k "TestBaselineStaged"`

- [ ] **47.5c** Tests in `tests/test_sweep_bo.py` — `TestValidateWithCpp` (3) + `TestBaselineStaged` (3):
  - `TestValidateWithCpp::test_validate_runs_before_persist` — sentinel-AssertionError on persist; verify validate fires first.
  - `test_skips_on_unsupported_kernel` — mock kernel outside `_CPP_SUPPORTED_KERNELS`, verify `cpp_validation` absent or has skip record.
  - `test_records_l8_failure_not_raises` — monkeypatch L8 to raise; verify run continues with error recorded.
  - `TestBaselineStaged::test_runs_when_flag_set` — mock both; verify both invoked with same seed.
  - `test_omitted_when_flag_unset` — verify `extras["baseline"]` absent without `--baseline`.
  - `test_persisted_alongside_bo_result` — full path: BO + baseline + persist; verify JSON has both `best_x` and `extras.baseline.best_x`.
  - File: `scripts/stencil_gen/tests/test_sweep_bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_bo.py -x -q -k "TestValidateWithCpp or TestBaselineStaged"`

### 47.6 — Validation: synthetic + failure-mode regressions

- [ ] **47.6a** Add `tests/test_bo.py::TestBranin` — synthetic `AugmentedBranin` validation. Use `botorch.test_functions.multi_fidelity.AugmentedBranin` (or hand-code the 2D Branin variant if not available). Run MF-BO with `n_init=8, budget_evals=30, seed=0`. Assert `best_objective < 0.5` (true global min `≈ 0.398`). Mark `@pytest.mark.slow`. Runtime budget: < 60s. This test validates the BO pipeline is correctly implemented BEFORE turning it on the real cascade.
  - File: `scripts/stencil_gen/tests/test_bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestBranin" --run-slow`

- [ ] **47.6b** Add three failure-mode regressions to `tests/test_bo.py`, all `@pytest.mark.slow`:
  - `TestBiasMisspec::test_l3_l7_disagreement` — synthetic objective where `f_lo(x) = (x - 0.3)^2`, `f_hi(x) = (x - 0.7)^2 + 0.1*sin(20x)` (different minima at different fidelities). Run MF-BO with `budget_evals=25`. Assert incumbent within 0.1 of `0.7`, NOT `0.3` — verifies BO doesn't over-trust the cheap fidelity when bias is large. The ICM kernel should learn small `B[lo, hi]` for this case.
  - `TestCostMisspec::test_misspec_2x_degradation_max` — Branin with `cost(L7) = 1.0` instead of true `100.0`. Run MF-BO. Assert `best_objective < 2 * (correctly_costed_baseline + 0.5)` — degraded but not catastrophic. (Hardcode `correctly_costed_baseline ≈ 0.4` from 47.6a's run.)
  - `TestMultiModal::test_classical_alpha_finds_a_basin` — real cascade (not synthetic) on E4 classical α with `layer3.max_stab_eig` (use L3 as HF for fast iteration). Run 5 seeds. Assert at least 4/5 seeds find an incumbent with `min(d(x_inc, BL_basin), d(x_inc, DE_basin)) < 0.1` where the BL basin = `[-0.7733, 0.1624]` and DE basin = `[-1.399, 0.293]`. Both are valid (per `scientific_findings.md` finding #2).
  - File: `scripts/stencil_gen/tests/test_bo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_bo.py -x -q -k "TestBiasMisspec or TestCostMisspec or TestMultiModal" --run-slow`

### 47.7 — Real benchmark + persistence

- [ ] **47.7a** Run the head-to-head benchmark and persist results. Single benchmark run (not a test): `python -m sweeps bo --scheme E4 --kernel classical --objective layer7.max_spectral_abscissa --cheap-fidelities 1 3 5 6 --bounds -2 2 0.05 2 --budget-evals 60 --seed 1 --persist --baseline staged`. Wall-time budget ~5 minutes. Commit the resulting `sweeps/bo_runs/E4_classical_layer7_max_spectral_abscissa_1.json`. The JSON contains both the MF-BO `BOResult` and the baseline `OptimizeResult` for direct comparison. Document in the commit message: total wall time, MF-BO vs staged best_objective, fraction of MF-BO evals spent at HF (target: 10–30%).
  - File: `scripts/stencil_gen/sweeps/bo_runs/E4_classical_layer7_max_spectral_abscissa_1.json`
  - Test: `cd scripts/stencil_gen && uv run python -c "import json; from pathlib import Path; d = json.loads(Path('sweeps/bo_runs/E4_classical_layer7_max_spectral_abscissa_1.json').read_text()); print('MF-BO:', d['best_objective'], 'staged:', d['extras']['baseline']['best_objective'], 'speedup:', d['extras']['baseline']['compute_time'] / d['total_compute_time'])"`

- [ ] **47.7b** Add `TestRegressionBOBenchmark` to `tests/test_phs.py`, modeled on `TestRegressionBrady2DPareto` (plan 45.6b.3):
  - Module-level load: `_BO_RUNS = list((Path(__file__).resolve().parent.parent / "sweeps" / "bo_runs").glob("*.json"))`.
  - Class-level `@pytest.fixture(autouse=True)` skips when `_BO_RUNS` is empty.
  - `test_each_run_best_x_recomputes_within_tolerance` — for each run file, rebuild `make_multi_fidelity_objective` from `report_fields_by_layer`, evaluate at `best_x` at HF, assert `np.isclose(recomputed, stored_best_objective, rtol=1e-2, atol=1e-8)`.
  - `test_each_run_baseline_present_when_recorded` — sanity check `extras.baseline` schema.
  - `@pytest.mark.slow` (rebuilding objectives + evaluating L7 on classical α takes ~30s).
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionBOBenchmark" --run-slow`

### 47.8 — Documentation

- [ ] **47.8a** Create `scripts/stencil_gen/docs/mfbo_reference.md`. Sections:
  - "Problem" — multi-fidelity formulation, math (GP with ICM kernel, cost-aware KG acquisition).
  - "Why BoTorch + qMFKG" — library choice, fallback to MES.
  - "API" — `BOResult`, `BOEval`, `make_multi_fidelity_objective`, `run_mfbo`, `build_mf_gp`, `build_cost_model`, `build_acquisition`.
  - "CLI" — `python -m sweeps bo` with three example invocations: tension fast (L1+L3 only), classical 2D full (L1+L3+L3r+L6+L7), benchmark with `--baseline staged`.
  - "Persistence schema" — `sweeps/bo_runs/<scheme>_<kernel>_<mangled>_<seed>.json` shape.
  - "Cost model calibration" — table from plan 46 measurements + how to override.
  - "Stopping criteria" — budget / variance / stagnation.
  - "Relationship to `run_staged_optimize`" — same problem class, principled vs heuristic.
  - "When MF-BO helps vs hurts" — Agent 3 finding: L3r→L7 only 3:1 cost ratio (modest gain); L3r→L7+nn 36:1 (large gain). Document.
  - "Failure modes" — bias misspec, cost misspec, multi-modal coverage; cite the regression tests.
  - File: `scripts/stencil_gen/docs/mfbo_reference.md` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from pathlib import Path; p = Path('docs/mfbo_reference.md'); assert p.exists() and p.stat().st_size > 4000, 'mfbo_reference.md missing or too small'"`

- [ ] **47.8b** Cross-link from existing docs:
  - `scripts/stencil_gen/docs/optimization_reference.md`: add a "Multi-fidelity (plan 47)" section pointing to `mfbo_reference.md`. Three sentences explaining when to use MF-BO vs scalar drivers.
  - `scripts/stencil_gen/docs/brady2d_stability_reference.md`: add a one-line note that any layer-N field can be used as a fidelity in `sweeps bo`.
  - `docs/handoff/MASTER.md`: update "Key artifacts" list to include `mfbo_reference.md`.
  - `docs/handoff/next_steps.md`: mark Plan 47 (Multi-Fidelity BO) as done; promote Plan 48 (Brady-Livescu 1D Euler) to "next."
  - File: `scripts/stencil_gen/docs/optimization_reference.md`, `scripts/stencil_gen/docs/brady2d_stability_reference.md`, `docs/handoff/MASTER.md`, `docs/handoff/next_steps.md`
  - Test: `grep -l mfbo_reference scripts/stencil_gen/docs/*.md docs/handoff/MASTER.md`

- [ ] **47.8c** Add a decision entry `D-Opt-2` to `plans/meta.md` capturing four cross-cutting choices: (a) BoTorch chosen over Emukit/Trieste/Ax (clean aarch64 wheels, NumPy 2 compat, active 2026 maintenance); (b) discrete fidelity ICM kernel chosen over Kennedy-O'Hagan AR1 (L3 ↔ L3r are different physics, not refinement); (c) cost-aware qMFKG primary with MES as documented swap (matches BoTorch's discrete-fidelity tutorial; MES adds value if KG diagnostics fail); (d) constant cost model with floor `c'(m) = max(c(m), 0.05*c(hf))` (prevents over-exploitation; per-layer learned cost deferred). The "Why these are cross-cutting" paragraph cites plan 48 (1D Euler — will reuse the BO infrastructure for nonlinear blow-up scoring).
  - File: `plans/meta.md`
  - Test: `grep -q "D-Opt-2" plans/meta.md`

- [ ] **47.8d** Update `.claude/skills/stencil-sweeps/SKILL.md`: add `bo` subcommand example + `bo_runs/` Key Files row + "Multi-fidelity Bayesian optimization" entry in "When to Use" + `mfbo_reference.md` cross-link in Detailed Reference. Same harness situation as plan 45.7d/e — try the edit; if blocked, manually complete in interactive session after ralph returns.
  - File: `.claude/skills/stencil-sweeps/SKILL.md`
  - Test: `grep -c "mfbo_reference" .claude/skills/stencil-sweeps/SKILL.md`

---

## Ordering

```
47.0a → 47.0b                               # deps + skeleton; plan-file import-path fix
  ↓
47.1a → 47.1b → 47.1c                       # dataclasses + factory + tests
  ↓
47.2a → 47.2b → 47.2c → 47.2d               # GP + cost + DOE + tests
  ↓
47.3a → 47.3b → 47.3c                       # acquisition + BO loop + tests
  ↓
47.4a → 47.4b → 47.4c → 47.4d               # CLI + dispatch + persistence + tests
  ↓
47.5a → 47.5b → 47.5c                       # validate-with-cpp + baseline + tests
  ↓
47.6a → 47.6b                               # synthetic Branin + failure-mode regressions
  ↓
47.7a → 47.7b                               # real benchmark + regression test
  ↓
47.8a → 47.8b → 47.8c → 47.8d               # docs + meta + skills
```

Strictly sequential. 47.5 (validate + baseline) can run before 47.4d's tests if needed (they're independent). 47.6 and 47.7 are validation-heavy and run after the implementation is complete. 47.7 specifically requires `shoccs` binary to be built (for `--validate-with-cpp` if used inside the benchmark, otherwise no dependency).

**Parallelism note:** 47.6a (Branin) and 47.6b (failure-mode regressions) can run concurrently in different processes since they don't share state — but ralph runs items sequentially, so this is informational only.

---

## Completion Criteria

- `uv sync` succeeds on aarch64 with `botorch>=0.17,<0.18` + `torch>=2.2,<3` + `gpytorch>=1.15,<2`. CPU-only PyTorch wheel (~140 MB) installed; no NVIDIA stack pulled.
- `import botorch; from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient; from botorch.models import SingleTaskMultiFidelityGP` runs without error.
- `make_multi_fidelity_objective("E4", "classical", {1: "layer1.boundary_gv_err", 3: "layer3.max_stab_eig", 7: "layer7.max_spectral_abscissa"})` returns a closure; calling it at BL's published α returns 3-tuples `(value, wall_time, report)` for `m ∈ {1, 3, 7}`.
- `run_mfbo` returns a `BOResult` with `len(eval_history) == budget_evals`, `seed`-reproducible across two consecutive runs, `gp_hyperparameters` populated, `n_evals_per_fidelity` summing to `budget_evals`.
- `python -m sweeps bo --help` lists all flags including `--cheap-fidelities`, `--budget-evals`/`--budget-seconds`, `--baseline`, `--validate-with-cpp`, `--persist`.
- `sweeps/bo_runs/` directory exists (committed via `.gitkeep`); contains the calibration JSON from 47.7a with both `best_x` and `extras.baseline.best_x` populated.
- `TestRegressionBOBenchmark` passes (slow): every stored run's `best_x` recomputes within 1% of stored objective.
- **Synthetic validation passes:** AugmentedBranin reaches `best_objective < 0.5` in ≤30 evals.
- **Failure-mode regressions pass:** bias-misspec finds `x = 0.7` not `0.3`; cost-misspec degrades by ≤2× vs correctly-costed baseline; multi-modal classical-α finds a known basin in ≥4/5 seeds.
- **Real benchmark deliverable from 47.7a:** the persisted JSON shows MF-BO either (i) reaching the same `best_objective` as the staged baseline using ≤50% of total wall-time, OR (ii) at equal wall-time, achieving a `best_objective` ≥ 1% lower than staged. Document whichever criterion was met (or, if neither, document the gap and propose follow-ups in plan 48).
- `scripts/stencil_gen/docs/mfbo_reference.md` exists (>4 KB) and is cross-linked from `optimization_reference.md`, `brady2d_stability_reference.md`, `MASTER.md`, `next_steps.md`.
- `plans/meta.md` contains `D-Opt-2` capturing the four architecture decisions.
- `.claude/skills/stencil-sweeps/SKILL.md` updated (manual completion if harness blocks).
- Fast test suite (`uv run pytest tests/ -x -q`) still passes in under 120 seconds (the BoTorch import + module collection adds ~5 s; budget appropriately).
- Slow tests (`--run-slow`) pass: TestBranin, TestBiasMisspec, TestCostMisspec, TestMultiModal, TestRegressionBOBenchmark all green; total slow-suite runtime < 15 minutes.
