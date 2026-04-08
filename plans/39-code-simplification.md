# Phase 39: Code Simplification — Group Velocity, Sweeps, and Test Cleanup

**Goal:** Simplify the code produced in Phases 34-38 by merging near-duplicate functions, removing redundant tests, DRYing up constants, and fixing missed slow markers. No new functionality — only cleanup.

**Depends on:** Phase 38 (complete — sweep extraction done)

**Read first:**
- `scripts/stencil_gen/stencil_gen/group_velocity.py` (1,132 lines — merge duplicates)
- `scripts/stencil_gen/tests/test_group_velocity.py` (~2,011 lines — remove research artifacts)
- `scripts/stencil_gen/sweeps/_common.py` (shared helpers — add constants)
- `scripts/stencil_gen/sweeps/epsilon_sweep.py` (duplicated logic with tension_sweep.py)
- `scripts/stencil_gen/sweeps/tension_sweep.py` (duplicated logic with epsilon_sweep.py)
- `scripts/stencil_gen/sweeps/__main__.py` (annotation fix, dispatch cleanup)
- `scripts/stencil_gen/tests/test_phs.py` (missed slow markers, dead imports)

**Test commands:**
```bash
# Fast suite (should drop below 15s after fixing missed slow markers)
cd scripts/stencil_gen && uv run pytest tests/ -x -q

# Group velocity tests only
cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q

# PHS regression tests only
cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegression"
```

---

## Items

### 39.1 — Fix missed `@pytest.mark.slow` in `test_phs.py` (highest impact)

- [x] **39.1a** Mark `TestModifiedWavenumber` sigma-sweep tests as `@pytest.mark.slow`:
  - `test_e2_boundary_at_optimal_sigma` — calls `_find_best_sigma` (101 coarse + 200 fine = 301 eigendecomps)
  - `test_e4_boundary_at_optimal_sigma` — same pattern, 301 eigendecomps
  - `test_dispersion_comparison` — calls `_find_best_sigma` twice (E2 + E4 = 602 eigendecomps)
  - These three tests are the main reason the default suite is ~27s instead of <15s.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q` (should be much faster)

### 39.2 — Clean up `test_phs.py` imports and artifacts

- [x] **39.2a** Remove unused `Symbol` and `symbols` imports:
  - Line 4: `from sympy import Rational, S, Symbol, cancel, symbols` → remove `Symbol` and `symbols`.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestPHSCore"`

- [x] **39.2b** Remove 8 redundant `import numpy as np` inside `TestTensionSpline` methods:
  - `np` is already imported at module level. The local re-imports are copy-paste artifacts.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestTensionSpline"`

- [x] **39.2c** Move deferred `import json as _json` and `from pathlib import Path as _Path` to top of file:
  - Drop the private aliases — `json` and `Path` don't conflict with anything.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegression"`

- [x] **39.2d** Add graceful fallback for `_load_known_values()`:
  - Currently a missing `known_values.json` kills all test collection. Wrap with try/except and skip regression tests if file is absent.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q`

### 39.3 — Merge near-duplicate functions in `group_velocity.py`

- [x] **39.3a** Make `modified_wavenumber` delegate to `modified_wavenumber_nonuniform`:
  - The uniform version just computes `offsets = node_indices - i_eval` then does the same `exp(1j * outer) @ w`. Make it a one-line delegate.
  - Same for `group_velocity_exact` → delegate to `group_velocity_exact_nonuniform`.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestCore"`

- [x] **39.3b** Merge `_build_profile` and `_build_profile_nonuniform` into one function:
  - Both have identical cutoff scan logic (7 lines copy-pasted). The only difference is how kstar and C are computed.
  - New signature: `_build_profile(weights, offsets, xi_array, order)` using the nonuniform path.
  - Update callers to pass `np.asarray(nodes) - i_eval` as offsets.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q`

- [x] **39.3c** Remove redundant `speed_ratio` field from `AnisotropyResult`:
  - Always set to `speed` (exact speed = 1.0). The field is definitionally identical to `speed`.
  - Remove from dataclass, update the one test that asserts on it.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`, `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "Test2D"`

- [x] **39.3d** Rename `group_velocity()` to `group_velocity_numerical()`:
  - The numerical-diff version is strictly inferior to `group_velocity_exact()`. Rename to make the relationship explicit.
  - Update the one test that calls it (`test_numerical_vs_analytical_gradient`).
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`, `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestCore"`

- [x] **39.3e** Extract shared `AnisotropyResult` construction into `_make_anisotropy_result(theta, C_x, C_y)`:
  - Duplicated in `anisotropy_profile()` and `boundary_group_velocity_2d()`.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "Test2D"`

### 39.4 — Remove redundant and diagnostic tests from `test_group_velocity.py`

- [x] **39.4a** Remove `test_e2_group_velocity_is_cos_xi` — strict subset of `test_interior_group_velocity_e2`:
  - The latter already checks cos(xi) agreement AND cutoff_xi AND return type.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestInterior"`

- [x] **39.4b** Remove `test_group_velocity_comparison_table` — zero-assertion diagnostic that always passes:
  - Prints a formatted table, asserts nothing. Belongs in sweeps/ if anywhere.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestInterior"`

- [x] **39.4c** Remove `test_ray_trace` — fully covered by `test_ray_trace_uniform`:
  - The E2 single-wavenumber case is a subset of the E4 multi-wavenumber test.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestVarying"`

- [x] **39.4d** Remove timing/scaling tests — these are benchmarks, not regressions:
  - Removed `TestScalingComparison` (3 methods), `test_gv_cost_vs_eigenvalue_cost`, and unused `import time`.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q` — 51 passed, 1 skipped

### 39.5 — DRY up sweeps package constants

- [x] **39.5a** Move `STABILITY_TOL` and `SCHEME_PARAMS` to `_common.py`:
  - Moved both constants to `_common.py`. Removed local definitions from epsilon_sweep.py, tension_sweep.py, comparison.py, tension_penalty_sweep.py, footprint_sweep.py, mixed_epsilon_sweep.py, alpha_extraction.py. All import from `_common` now.
  - File: `scripts/stencil_gen/sweeps/_common.py`, and all 6 sweep scripts
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps epsilon --scheme E2 --n-eps 5` — verified OK

- [ ] **39.5b** Extract shared `print_sweep_table` and `report_stable_ranges` to `_common.py`:
  - Nearly identical between `epsilon_sweep.py` and `tension_sweep.py`. Parameterize with `param_label` argument.
  - Remove or wire in the unused `dense_sweep_min` helper already in `_common.py`.
  - File: `scripts/stencil_gen/sweeps/_common.py`, `sweeps/epsilon_sweep.py`, `sweeps/tension_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps epsilon --scheme E2 --n-eps 5`

- [ ] **39.5c** Fix `callable` annotation in `__main__.py`:
  - Line 169: `callable` (lowercase builtin) → `Callable` from `collections.abc`.
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps --help`

### 39.6 — Final Timing Verification

- [ ] **39.6a** Mark slow tests in `test_e4_cut_cell.py` with `@pytest.mark.slow`:
  - `test_e4_cut_cell.py` has 0 slow markers but dominates the default suite at ~26s (the rest runs in ~6s).
  - All `TestDeriveCutCellScheme`, `TestE4CutCellSchemeWithZeros`, `TestE4CutCellConservationSolution`, `TestFractionFreeConservation`, and `TestE4TEMOConstruction` tests involve symbolic derivation and should be marked slow.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/ -x -q` (should drop to ~6s)

- [ ] **39.6b** Verify default test suite is under 15 seconds:
  - After 39.1a and 39.6a, both eigendecomp and symbolic-derivation tests will be skipped by default.
  - Run: `cd scripts/stencil_gen && uv run pytest tests/ --durations=10 -q`
  - File: N/A (verification only)

---

## Ordering

```
39.1a (slow markers) — do first, highest impact on test speed
39.2a-d (test_phs cleanup) — independent of 39.1
39.3a-e (group_velocity merges) — independent of 39.1-39.2
39.4a-d (test_group_velocity cleanup) — independent of 39.3
39.5a-c (sweeps DRY) — independent of 39.1-39.4
39.6a (slow-mark test_e4_cut_cell) — before timing verification
39.6b (timing verification) — do last
```

---

## Completion Criteria

- Default `uv run pytest tests/ -x -q` completes in under 15 seconds.
- `group_velocity.py` has no duplicate functions — uniform variants delegate to nonuniform.
- `_build_profile` exists once, not twice.
- `test_group_velocity.py` has no zero-assertion diagnostic tests or timing benchmarks.
- `sweeps/` has `STABILITY_TOL` and `SCHEME_PARAMS` defined once in `_common.py`.
- `test_phs.py` has no dead imports, no redundant local imports, no deferred stdlib imports.
- All existing tests still pass.
