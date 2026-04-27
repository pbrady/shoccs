# Phase 46: Hardening — Address Review-Pass Patterns from Plan 45

**Goal:** Plan 45's review pass surfaced several recurring categories of bugs (CLI surfaces vs. library defaults, schema completeness, sibling non-determinism, vacuous tests, silent fallbacks). Most were fixed inline as in-plan follow-ups. This plan addresses the **siblings** — the same patterns elsewhere in the codebase that the review pass didn't get to. It also activates dormant `TestRegression*` infrastructure that has been sitting idle since plans 40–43 because the populating sweeps were never run. The result is a tighter regression net before plan 47 (Multi-Fidelity BO) starts piling on optimizer machinery.

**Depends on:** Plan 45 (the review-pass findings that motivate every item here).

**Background — what the review pass found in plan 45:**

| Category | Plan-45 fix | Sibling found in this audit |
|---|---|---|
| **Non-determinism** | `1c75893` 45.6b.1: seed `eigs` in `spectral_abscissa_sparse` | `numerical_abscissa_sparse` calls `eigsh` without `rng` (latent — only triggers above 900-DOF) |
| **CLI vs library defaults** | `5d6e2ab` 45.0d: `sweeps/optimize.py` `--gate-layer` → `None` | `sweeps/__main__.py:212` optimize sub-parser **still hardcodes `default=3`** and forwards unconditionally — nullifies 45.0d for `python -m sweeps optimize` (the documented entry per `CLAUDE.md`) |
| **Schema completeness** | `8587963` 45.6a.1.1: `_report_to_dict` add `layer_bl42` | `compute_time` missing in 2 of 3 copies; `non_normality` and `kreiss` (top-level fields) missing in **all 3 copies**; `layer8` missing in `brady2d_calibration._report_to_dict` |
| **Vacuous tests** | `1985fd1`/`f83bcbc` 44.4n: harden conditional assertions | `test_sentinel_rows_excluded` (`>= 0` is trivially true), `test_default_gate_for_bl42_objective` (mocked `failed_layer=2` doesn't distinguish new from old), `test_result_metadata_populated` (no front-size guard before loop) |
| **Silent fallbacks** | (none in plan 45) | `rank_for_l8` silently sets `key = lambda p: 0.0` when `max_layer < 3` — produces arbitrary ordering with no diagnostic |
| **Dormant regression tests** | (none in plan 45) | `TestRegressionGV` (6 dormant sub-tests), `TestRegressionBrady2DSweep` (2 sub-tests), `TestRegressionBrady2DOptima` (1 sub-test) |

**The motivating fact:** the `python -m sweeps optimize` CLI is *still broken* for the `layer_bl42` objective even after plan 45.0d. Users hitting the documented entry point still trigger the 2634-eval `+inf` DE trap unless they manually pass `--gate-layer 2`. That's a regression the review pass missed because plan 45 only audited the standalone CLI module, not the umbrella dispatcher.

**Why this comes before plan 47 (BO):**

- `_report_to_dict` gaps would silently corrupt BO surrogate training data.
- The `__main__.py` gate-layer bug would affect any future BO CLI built on the same dispatch pattern.
- Activating dormant regression tests catches future regressions earlier, before they surface as plan-N+1 review-pass items.

**What this plan does NOT do:**

- **Plan 47 — Multi-Fidelity BO.** Renumbered from 46 → 47.
- **Plan 48 — BL 1D Euler reproduction.** Renumbered from 47 → 48.
- **Refactor `_report_to_dict` into a single canonical implementation.** Three copies (in `optimizer.py`, `brady2d_sweep.py`, `brady2d_calibration.py`) is a code smell, but the calibration data depends on the current schema, and consolidating risks invalidating `known_values.json["brady2d_calibration"]`. This plan patches each copy in parallel; consolidation deferred.
- **Comprehensive determinism re-audit.** This plan addresses the one sibling found by Agent 1; broader audit would need ARPACK/scipy.sparse expertise outside ralph's scope.
- **Activate `TestRegressionBrady2DOptima`.** Requires 5–30 min of optimizer wall-time per entry plus `shoccs` binary. Defer to when an optimization run lands as part of plan 47 or hand-driven research.
- **Test runtime audit / `@pytest.mark.slow` tagging review.** Not flagged as a problem by the audit; defer.

**Read first:**

- `git show 5d6e2ab` — the 45.0d fix that *should* have wired auto-infer through, but only touched the standalone CLI.
- `scripts/stencil_gen/sweeps/__main__.py` lines 200–410 (optimize sub-parser registration + forwarding logic — this is where the bug lives).
- `scripts/stencil_gen/stencil_gen/non_normality.py` lines 150–200 (`numerical_abscissa_sparse` and the `eigsh` call site).
- `git show 1c75893` — pattern for the rng-seed fix (apply same pattern to `eigsh`).
- `scripts/stencil_gen/stencil_gen/optimizer.py:706` (`_report_to_dict` copy 1).
- `scripts/stencil_gen/sweeps/brady2d_sweep.py:195` (`_report_to_dict` copy 2).
- `scripts/stencil_gen/stencil_gen/benchmarks/brady2d_calibration.py:66` (`_report_to_dict` copy 3 — diverges from the others).
- `scripts/stencil_gen/stencil_gen/brady2d_stability.py` lines 80–110 (`StabilityReport` dataclass — the source of truth for what fields exist).
- `scripts/stencil_gen/sweeps/_pareto_io.py` `_ParetoEncoder` (no `complex` handler — needed for `KreissResult.witness_s`).
- `scripts/stencil_gen/tests/test_phs.py` — the dormant `TestRegression*` block; understand the skip-when-absent pattern.
- `scripts/stencil_gen/sweeps/brady2d_sweep.py` lines 180–200 (`rank_for_l8` silent fallback).

**Test commands:**

```bash
# Fast: all hardening test changes
cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py tests/test_pareto.py tests/test_sweep_pareto.py tests/test_non_normality.py -x -q -k "TestGateLayerInfer or numerical_abscissa or report_to_dict or sentinel_rows or result_metadata"

# Regression: dormant tests now active
cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionGV or TestRegressionBrady2DSweep or TestRegressionBrady2DPareto"

# CLI smoke: gate_layer auto-infer through __main__ dispatch
cd scripts/stencil_gen && uv run python -m sweeps optimize --scheme E4 --kernel tension --objective layer_bl42.max_spectral_abscissa --bounds 0.5 20 --method Nelder-Mead --max-evals 20
# (must print a finite best_objective, not +inf)
```

---

## Items

### 46.0 — Critical: fix `__main__.py` optimize gate_layer dispatch bug

- [x] **46.0a** Fix `scripts/stencil_gen/sweeps/__main__.py` lines 212 and 393 to mirror the 45.0d `sweeps/optimize.py` standalone fix:
  - Line 212: change `sub_opt.add_argument("--gate-layer", type=int, default=3)` to `sub_opt.add_argument("--gate-layer", type=int, default=None, help="Layer N where failure short-circuits to +inf. Default: max(--max-layer - 1, 0) auto-inferred from objective.")`.
  - Line ~393: change unconditional `forwarded.extend(["--gate-layer", str(args.gate_layer)])` to `if args.gate_layer is not None: forwarded.extend(["--gate-layer", str(args.gate_layer)])` — mirror the pareto sub-parser pattern at lines 356–357.
  - Verify by running `python -m sweeps optimize --objective layer_bl42.max_spectral_abscissa --kernel tension --scheme E4 --bounds 0.5 20 --method Nelder-Mead --max-evals 5` and confirming the resolved `gate_layer=2` (not 3) appears in the run summary.
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps optimize --scheme E4 --kernel tension --objective layer_bl42.max_spectral_abscissa --bounds 0.5 20 --method Nelder-Mead --max-evals 5 2>&1 | grep -E '(gate_layer|best_objective)'`

- [x] **46.0b** Add `TestSweepsMainDispatchGateLayerInfer` to `scripts/stencil_gen/tests/test_optimizer.py` with two tests:
  - `test_dispatch_omitting_gate_layer_uses_auto_infer`: subprocess-runs `python -m sweeps optimize --scheme E4 --kernel tension --objective layer_bl42.max_spectral_abscissa --bounds 0.5 20 --method Nelder-Mead --max-evals 5 --json-output -` (or whatever flag emits the persisted record); asserts the dispatched `gate_layer` is `2`, not `3`.
  - `test_dispatch_explicit_gate_layer_preserved`: same call with `--gate-layer 4`; asserts dispatched `gate_layer == 4`.
  - These distinguish the new auto-infer from the old hardcoded default at the **dispatch level** (not just the standalone CLI level), which is what 45.0d's regression test missed.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestSweepsMainDispatchGateLayerInfer"`

### 46.1 — Sibling non-determinism: `numerical_abscissa_sparse`

- [x] **46.1a** Patch `numerical_abscissa_sparse` in `scripts/stencil_gen/stencil_gen/non_normality.py` line ~150 to mirror the `1c75893` 45.6b.1 fix on `spectral_abscissa_sparse`:
  - Add `rng_seed: int = 0` to the function signature.
  - Pass `rng=rng_seed` to the `eigsh(H, k=k_use, which="LA", return_eigenvectors=False)` call at line ~187.
  - Update docstring with the same note about cross-process determinism.
  - Update the one caller (`compute_non_normality` at line ~447 — currently does NOT pass `rng_seed`; default of 0 is fine, but explicitly thread it through if `compute_non_normality` itself accepts a seed parameter).
  - File: `scripts/stencil_gen/stencil_gen/non_normality.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_non_normality.py -x -q -k "numerical_abscissa"`

- [ ] **46.1b** Add `test_numerical_abscissa_sparse_deterministic_across_calls` to `tests/test_non_normality.py`:
  - Construct a synthetic 1000×1000 non-symmetric sparse matrix (forces the `eigsh` path: `n > 900`).
  - **Gotcha — do not copy the skew-symmetric construction from `TestSpectralAbscissaDeterminism._bl42_like_matrix`.** That makes `H = (L+L^T)/2 = 0`, and `eigsh` on the zero matrix fails to converge, raising `RuntimeError: ... N > 900 prevents dense fallback` before the determinism path is exercised. Verified empirically against the current `numerical_abscissa_sparse` (`rng_seed=0`, n=1000 skew-symmetric → RuntimeError). Use a **non-symmetric** matrix whose Hermitian part has non-trivial spectrum, e.g. `A = rng.standard_normal((n, n)) - 5.0 * np.eye(n); L = sp.csr_matrix(A)` — this produces a well-conditioned `H = (L+L^T)/2` and gives finite results across both default and `rng_seed=42` calls.
  - Compute `numerical_abscissa_sparse(L)` twice; assert exact equality (not allclose) — `rng=0` should be byte-identical.
  - Compute with `rng_seed=42` and verify it differs from the default (otherwise the seed is not actually being threaded through ARPACK). At n=1000 with the construction above, the difference is at the ULP level (~1e-14 relative), so use `!=`, not `np.isclose`.
  - Mark `@pytest.mark.slow` if wall time exceeds 1 s.
  - File: `scripts/stencil_gen/tests/test_non_normality.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_non_normality.py -x -q -k "test_numerical_abscissa_sparse_deterministic"`

### 46.2 — Schema completeness: `_report_to_dict` × 3 + `_ParetoEncoder`

- [ ] **46.2a** Add `compute_time` serialization to two of the three `_report_to_dict` copies:
  - `scripts/stencil_gen/stencil_gen/optimizer.py` line ~706: append `"compute_time": float(report.compute_time)` to the returned dict.
  - `scripts/stencil_gen/sweeps/brady2d_sweep.py` line ~195: same addition.
  - The third copy (`brady2d_calibration.py:72`) already has it — leave alone.
  - This was the silent data loss flagged by the audit: every Pareto front member's report currently lacks `compute_time`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`, `scripts/stencil_gen/sweeps/brady2d_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py tests/test_pareto.py -x -q -k "report_to_dict"`

- [ ] **46.2b** Add `non_normality` and top-level `kreiss` field handlers to all three `_report_to_dict` copies, plus a `complex` handler in `_ParetoEncoder`:
  - In each `_report_to_dict` copy: `if report.non_normality is not None: out["non_normality"] = dataclasses.asdict(report.non_normality)`. Same for `report.kreiss` (a `KreissResult` with a `complex` `witness_s` field).
  - In `scripts/stencil_gen/sweeps/_pareto_io.py` `_ParetoEncoder.default`: add `if isinstance(obj, complex): return [obj.real, obj.imag]` (matches the convention in `brady2d_cli.py:28`).
  - Without this, any future code path that populates `report.kreiss` (currently latent — `layer2_kreiss_gks` populates `report.kreiss`, not `report.layer2`, in plan 41's design) will crash at `json.dumps` time inside `save_pareto_front`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`, `scripts/stencil_gen/sweeps/brady2d_sweep.py`, `scripts/stencil_gen/stencil_gen/benchmarks/brady2d_calibration.py`, `scripts/stencil_gen/sweeps/_pareto_io.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_pareto.py tests/test_sweep_pareto.py -x -q -k "kreiss or non_normality or complex"`

- [ ] **46.2c** Add `layer8` serialization to `brady2d_calibration._report_to_dict`:
  - File: `scripts/stencil_gen/stencil_gen/benchmarks/brady2d_calibration.py:66`. The other two copies already handle `layer8`; this one diverges. Calibration is currently always `max_layer ≤ 7`, so the gap is benign — but plan 47 (Multi-Fidelity BO) might want to calibrate at L8, and a silent drop would be a debugging nightmare.
  - File: `scripts/stencil_gen/stencil_gen/benchmarks/brady2d_calibration.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_benchmarks.py -x -q -k "calibration and layer8"` (test added in 46.2d)

- [ ] **46.2d** Add `TestReportToDictSchemaParity` to `tests/test_optimizer.py`:
  - Build a fully-populated `StabilityReport` with all 15 fields set (use synthetic data; no need for real eigenvalues).
  - Run each of the three `_report_to_dict` copies on it.
  - Assert each output has all expected keys for the layers run, plus `compute_time`, `failed_layer`, `failed_reason`, `overall_verdict`, `non_normality`, `kreiss`.
  - Run the result through `json.dumps` with `_ParetoEncoder` to confirm no crash.
  - Single test, parametrized over the three copies: `@pytest.mark.parametrize("serializer", [opt._report_to_dict, sweep._report_to_dict, calib._report_to_dict])`.
  - Future-proofs the schema: any new field added to `StabilityReport` will fail this test until added to all serializers.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestReportToDictSchemaParity"`

### 46.3 — Activate `TestRegressionGV` (cheap, ~5–10 min total)

- [ ] **46.3a** Run E2 + E4 tension `--include-gv` sweeps and persist:
  - `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python -m sweeps tension --scheme E2 --include-gv --update-known-values`
  - Same for `--scheme E4`.
  - Commit the resulting `known_values.json` deltas (should add `E2_1.tension_gv`, `E2_1.tension.gv_error`, and the same for E4).
  - Verify activation: `uv run pytest tests/test_phs.py -x -q -k "TestRegressionGV"` — `test_scheme_gv_entries_match_stored_error` and `test_scheme_primary_gv_error_match` should no longer skip for tension.
  - File: `scripts/stencil_gen/sweeps/known_values.json`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionGV and tension"`

- [ ] **46.3b** Run E2 + E4 gaussian `--include-gv` sweeps and persist:
  - `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python -m sweeps epsilon --scheme E2 --kernel gaussian --include-gv --update-known-values`
  - Same for `--scheme E4`.
  - Commit the resulting `known_values.json` deltas.
  - File: `scripts/stencil_gen/sweeps/known_values.json`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionGV and gaussian"`

- [ ] **46.3c** Run E2 + E4 multiquadric `--include-gv` sweeps and persist:
  - `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python -m sweeps epsilon --scheme E2 --kernel multiquadric --include-gv --update-known-values`
  - Same for `--scheme E4`.
  - Commit the resulting `known_values.json` deltas.
  - Verify the broader `TestRegressionGV` block now activates fully (the per-scheme GV entries check exits successfully without skips for any of the three kernels).
  - File: `scripts/stencil_gen/sweeps/known_values.json`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionGV"`

### 46.4 — Activate `TestRegressionBrady2DSweep` (cheap classical seed)

- [ ] **46.4a** Run a single `brady2d` sweep entry to populate the dormant key:
  - `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python -m sweeps brady2d --scheme E4 --kernel classical --max-layer 3 --update-known-values`
  - Commit the resulting `known_values.json["brady2d_sweep"]` entry.
  - Verify activation: `uv run pytest tests/test_phs.py -x -q -k "TestRegressionBrady2DSweep"` — both sub-tests should run and pass.
  - File: `scripts/stencil_gen/sweeps/known_values.json`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionBrady2DSweep"`

### 46.5 — Test hardening: vacuous assertions

- [ ] **46.5a** Strengthen `test_sentinel_rows_excluded` in `tests/test_pareto.py:410`:
  - Replace `assert res.extras["n_sentinel_filtered"] >= 0` (trivially true — counts can't be negative) with `assert res.extras["n_sentinel_filtered"] >= 1, "expected at least one infeasible eval to have been filtered"`.
  - If the assertion is unreliable at the test's `pop_size=20, n_gen=4` budget, also assert `len(res.front) < 20` to prove the filter ran (a 20-member final population would mean nothing was filtered).
  - File: `scripts/stencil_gen/tests/test_pareto.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_pareto.py -x -q -k "test_sentinel_rows_excluded"`

- [ ] **46.5b** Strengthen `test_default_gate_for_bl42_objective` in `tests/test_optimizer.py:459`:
  - Current test mocks `failed_layer=2`, which gates under both old (`gate_layer=3`) and new (`gate_layer=2` auto-inferred) defaults — vacuous as a regression for plan 45.0b.
  - Add a sub-case (or new test method) with `r.failed_layer = 3` and `r.layer_bl42 = {"max_spectral_abscissa": 5.0}`. Assert the closure returns `5.0` (finite), not `+inf`. **This is the case that would have failed under the old hardcoded `gate_layer=3`** and is the true regression test for auto-infer.
  - Cross-reference: `test_bl42_l3r_failure_returns_finite` already exists per 45.0e — verify the two tests don't duplicate, and consolidate or cross-link in docstrings if so.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "test_default_gate_for_bl42_objective or test_bl42_l3r_failure_returns_finite"`

- [ ] **46.5c** Add a non-empty front guard to `test_result_metadata_populated` in `tests/test_pareto.py:480`:
  - Current loop `for p in res.front: assert ...` is vacuous if `res.front` is empty (could happen with bad seeding on the synthetic ZDT1-like objective).
  - Add `assert len(res.front) >= 1, "front must be non-empty for ZDT1-like; check seeding"` before the loop.
  - File: `scripts/stencil_gen/tests/test_pareto.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_pareto.py -x -q -k "test_result_metadata_populated"`

### 46.6 — Diagnostic on silent fallback in `rank_for_l8`

- [ ] **46.6a** Replace the silent `key = lambda p: 0.0` fallback in `scripts/stencil_gen/sweeps/brady2d_sweep.py:189`:
  - Current `else: key = lambda p: 0.0` produces arbitrary equal-rank ordering when `max_layer < 3` or layer3 reports are missing — silently masks an unusable state.
  - Replace with an explicit `warnings.warn(f"rank_for_l8: max_layer={max_layer} too shallow for meaningful ranking; using insertion order", UserWarning)` and `key = lambda p: 0.0` (preserve the same behavior, but make the silent case visible). User can grep `UserWarning` in CI logs.
  - **Don't** convert to `assert False` — the function is called from sweep paths that legitimately run at `max_layer=1` or `max_layer=2` for fast diagnostics.
  - File: `scripts/stencil_gen/sweeps/brady2d_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_brady2d.py -x -q -k "rank_for_l8 and shallow_max_layer"` (test added below if absent)

### 46.7 — Documentation: plan-number renumbering + dormant-test cleanup

- [ ] **46.7a** Renumber the queued plans in `docs/handoff/next_steps.md`: rename "Plan 46 — Multi-Fidelity Bayesian Optimization" to "Plan 47" and "Plan 47 — Brady-Livescu 1D Euler Reproduction" to "Plan 48". Add a new "Plan 46 — Hardening (this plan)" entry above them with a one-paragraph summary citing the review-pass-driven motivation.
  - File: `docs/handoff/next_steps.md`
  - Test: `grep -E '^## Plan [0-9]+' docs/handoff/next_steps.md | head -5`

- [ ] **46.7b** Update `plans/meta.md` D-Opt-1 entry:
  - The "Why these are cross-cutting" paragraph mentions "plans 46/47" referring to BO and 1D Euler. Update to "plans 47/48" to match 46.7a's renumbering.
  - The `## Implementing plan items` footer references plan 45 items only; no change there.
  - File: `plans/meta.md`
  - Test: `grep -A 1 "cross-cutting" plans/meta.md | grep -E "(46|47|48)"`

- [ ] **46.7c** Update `docs/handoff/known_limitations.md`:
  - Remove the "TestRegressionGV is dormant" entry (now active per 46.3).
  - Update the "brady2d_sweep and brady2d_optima keys not populated" entry: `brady2d_sweep` is now populated for `E4_classical` (per 46.4); leave `brady2d_optima` deferred with a one-line note.
  - Add a new "Three copies of `_report_to_dict`" code-smell entry under "Miscellaneous sharp edges": call out that consolidation was deferred at plan 46 to avoid invalidating `known_values.json["brady2d_calibration"]`; the hardening plan patched all three in parallel; future plan should consolidate when calibration data is regenerable.
  - File: `docs/handoff/known_limitations.md`
  - Test: `grep -c "TestRegressionGV is dormant" docs/handoff/known_limitations.md` (expect 0); `grep -c "Three copies of _report_to_dict" docs/handoff/known_limitations.md` (expect 1)

---

## Ordering

```
46.0a → 46.0b                            # critical CLI dispatch fix + regression test
  ↓
46.1a → 46.1b                            # sibling non-determinism + test
  ↓
46.2a → 46.2b → 46.2c → 46.2d            # schema completeness across 3 _report_to_dict copies
  ↓
46.3a → 46.3b → 46.3c                    # activate TestRegressionGV (tension, gaussian, multiquadric)
  ↓
46.4a                                    # activate TestRegressionBrady2DSweep
  ↓
46.5a → 46.5b → 46.5c                    # test hardening
  ↓
46.6a                                    # rank_for_l8 diagnostic
  ↓
46.7a → 46.7b → 46.7c                    # documentation
```

Strictly sequential. 46.0 first because it's the user-visible bug. 46.1 and 46.2 next because they fix latent data-corruption issues that 46.3/46.4 might exercise. 46.5/46.6 are independent test cleanups that can be reordered if needed. 46.7 last (docs always last).

**Parallelism note:** 46.3a/b/c can technically run in parallel (different sweep commands on the same `known_values.json`), but ralph's workflow assumes sequential commits — leave them sequential to avoid merge conflicts.

---

## Completion Criteria

- `python -m sweeps optimize --objective layer_bl42.max_spectral_abscissa --kernel tension --scheme E4 --bounds 0.5 20 --method Nelder-Mead --max-evals 5` returns a finite `best_objective` (the documented entry point now respects auto-infer).
- `numerical_abscissa_sparse` accepts a `rng_seed` parameter; two consecutive calls on a 1000-DOF non-symmetric matrix produce byte-identical results at the default seed.
- All three `_report_to_dict` copies serialize `compute_time`, `non_normality`, `kreiss`, and `layer8` (when populated). `TestReportToDictSchemaParity` passes for all three.
- `_ParetoEncoder` handles `complex` values (no crash if `KreissResult.witness_s` is ever populated and reaches the encoder).
- All E2/E4 × {tension, gaussian, multiquadric} `*_gv` entries are present in `known_values.json`. `TestRegressionGV` runs with zero skips for the per-scheme block.
- `known_values.json["brady2d_sweep"]["E4"]["classical"]` is populated. `TestRegressionBrady2DSweep` runs with zero skips for the classical seed.
- `test_sentinel_rows_excluded`, `test_default_gate_for_bl42_objective`, and `test_result_metadata_populated` all distinguish their target behavior from the trivial / pre-fix case.
- `rank_for_l8` issues a `UserWarning` (not silent fallback) when invoked with `max_layer < 3` or missing layer3 reports.
- `docs/handoff/next_steps.md` lists "Plan 46 — Hardening (this plan)" above renumbered plans 47 (BO) and 48 (1D Euler). `plans/meta.md` D-Opt-1 references match the new numbering.
- `cd scripts/stencil_gen && uv run pytest tests/ -x -q` (fast suite) still passes in under 90 seconds.
- The Pareto front JSON files written before this plan still load cleanly under the updated `_ParetoEncoder` and `_report_to_dict` schemas (no breaking format change — only additive fields).
