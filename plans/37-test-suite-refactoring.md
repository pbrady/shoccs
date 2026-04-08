# Phase 37: Test Suite Refactoring — Separate Research Sweeps from Regression Tests

**Goal:** Reduce the stencil_gen test suite from ~6 minutes to <15 seconds for default runs by separating research/exploration sweeps (parameter space searches, 2D optimization grids, Groebner basis proofs) from regression tests (verify known-good values with spot checks).

**Depends on:** None (standalone refactoring, does not change production code)

**Read first:**
- `scripts/stencil_gen/tests/test_phs.py` (5,878 lines — the main offender, ~30,000 eigenvalue decomps)
- `scripts/stencil_gen/tests/test_e4_cut_cell.py` (the other offender — 300s dominated by one 197s test)
- `scripts/stencil_gen/tests/conftest.py` (existing shared fixtures)
- `scripts/stencil_gen/stencil_gen/phs.py` (stability_eigenvalue, build_diff_matrix_rbf — no caching)

**Test commands:**
```bash
# Default run (should be <15s after refactoring)
cd scripts/stencil_gen && uv run pytest tests/ -x -q

# Run including slow research tests
cd scripts/stencil_gen && uv run pytest tests/ -x -q --run-slow

# Run only the fast regression tests
cd scripts/stencil_gen && uv run pytest tests/ -x -q -m "not slow"

# Timing check
cd scripts/stencil_gen && uv run pytest tests/ --durations=10 -q
```

**Current timing (measured):**

| File | Time | Tests | Problem |
|------|------|-------|---------|
| `test_e4_cut_cell.py` | ~300s | 131 | 197s Groebner proof + 18 duplicate pipeline calls |
| `test_phs.py` | ~27s | 131 | ~30,000 eigenvalue decomps in parameter sweeps |
| Everything else | ~7s | 265 | Fine |

---

## Items

### 37.1 — Add `@pytest.mark.slow` infrastructure

- [x] **37.1a** Register a `slow` marker in `pyproject.toml` and add a `conftest.py` hook to skip slow tests by default:
  - Add to `pyproject.toml`: `markers = ["slow: marks tests as slow (deselected by default, use --run-slow to include)"]`
  - Add to `conftest.py`: `pytest_addoption` hook for `--run-slow` flag, and `pytest_collection_modifyitems` to skip `slow`-marked tests unless `--run-slow` is passed.
  - File: `scripts/stencil_gen/pyproject.toml`, `scripts/stencil_gen/tests/conftest.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_printer.py -x -q` (verify no breakage)

### 37.2 — Mark research sweeps as slow in `test_phs.py`

The following test classes are parameter space explorations, not regression tests. They re-discover optimal parameters from scratch via dense sweeps rather than verifying known values.

- [x] **37.2a** Mark Phase 29/30/31 sweep classes as `@pytest.mark.slow` — these are fully superseded by Phase 32 corrected equivalents:
  - `TestEpsilonSweepE2` (984 eigendecomps, duplicated by `TestCorrectedSweepE2`)
  - `TestEpsilonSweepE4` (984 eigendecomps, duplicated by `TestCorrectedSweepE4`)
  - `TestMixedEpsilon` (2,675 eigendecomps)
  - `TestStableEpsilonAlphas` (811 eigendecomps)
  - `TestComparisonTable` (2,460 eigendecomps, duplicated by `TestCorrectedComparison`)
  - `TestTensionSweepE2` (832 eigendecomps, duplicated by `TestCorrectedTensionE2`)
  - `TestTensionSweepE4` (1,511 eigendecomps, duplicated by `TestCorrectedTensionE4`)
  - `TestTensionOptimalSigma` (1,615 eigendecomps)
  - `TestTensionConservationE2` (2,895 eigendecomps)
  - `TestTensionConservationE4` (2,319 eigendecomps, duplicated by `TestCorrectedTensionPenaltyE4`)
  - `TestTensionComparison` (3,640 eigendecomps, duplicated by `TestCorrectedComparison`)
  - `TestFootprintE4Quick` (162 eigendecomps, duplicated by `TestCorrectedFootprint`)
  - `TestFootprintSweep` (404 eigendecomps, duplicated by `TestCorrectedFootprint`)
  - `TestFootprintPenalty` (4,000 eigendecomps)
  - `TestCrossValidationE2` (303 eigendecomps)
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q` (only non-slow tests run)

- [x] **37.2b** Mark Phase 32 dense sweep classes as `@pytest.mark.slow` — these also do production-scale sweeps:
  - `TestCorrectedSweepE2` (627 eigendecomps — 60-point sweeps)
  - `TestCorrectedSweepE4` (627 eigendecomps)
  - `TestCorrectedTensionE2` (649 eigendecomps — 61+200 point sweeps)
  - `TestCorrectedTensionE4` (649 eigendecomps)
  - `TestCorrectedTensionPenaltyE4` (1,304 eigendecomps — 25x25 2D sweep)
  - `TestCorrectedFootprint` (570 eigendecomps — 81+101 point sweeps)
  - `TestCorrectedComparison` (2,536 eigendecomps)
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q`

### 37.3 — Add fast regression replacements for swept tests in `test_phs.py`

Each slow sweep class should have a fast regression equivalent that verifies the known-good result with 1-3 spot checks instead of a 60-200 point sweep.

- [x] **37.3a** Add `TestRegressionE2Stability` (fast) to replace swept E2 stability verification:
  - Test `test_e2_tension_optimal_sigma` — verify `stability_eigenvalue(40, 1, 1, sigma_known, "tension", 1, 1) <= 0` for the known optimal sigma. One eigendecomp instead of 300.
  - Test `test_e2_gaussian_optimal_epsilon` — same for known Gaussian epsilon. One eigendecomp.
  - Test `test_e2_stable_at_multiple_grid_sizes` — verify stability at n=20,40,80 with known-good params. Three eigendecomps instead of 180.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionE2"`

- [x] **37.3b** Add `TestRegressionE4Stability` (fast) — same pattern for E4:
  - Verify known-good tension sigma, Gaussian epsilon, and multiquadric epsilon each produce stability eigenvalue <= 0 at 2-3 grid sizes.
  - Total: ~10 eigendecomps instead of ~5,000.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionE4"`

- [x] **37.3c** Add `TestRegressionFootprint` (fast) — verify known-good nextra/sigma/gamma values:
  - One stability check per known-good (nextra, sigma, gamma) triple. ~5 eigendecomps instead of 4,000.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionFootprint"`

- [x] **37.3d** Add `TestRegressionComparison` (fast) — verify comparison table values:
  - Hard-code the expected comparison table values and verify they still match. No sweeps needed.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionComparison"`

### 37.4 — Mark slow tests in `test_e4_cut_cell.py`

- [x] **37.4a** Mark `test_e4_1_conservation_constant_weights_infeasible_r5` as `@pytest.mark.slow`:
  - This 197-second Groebner basis proof proved infeasibility once. It doesn't need to re-prove it every run. The result is documented and the infeasibility is structural.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -x -q -k "not slow"`

- [x] **37.4b** Mark `TestMathematicaWorkflow` as `@pytest.mark.slow`:
  - Already skipped in CLAUDE.md's recommended command, but formalize with the marker. The 337s fixture setup (`derive_cut_cell_mathematica(E4_1, psi)`) is a full pipeline run.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -x -q`

- [x] **37.4c** Mark `TestPolynomialFullStencil` and `TestE4CodeGeneration` as `@pytest.mark.slow`:
  - Already skipped in CLAUDE.md, formalize with marker.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 37.5 — Cache expensive derivations in `test_e4_cut_cell.py`

- [ ] **37.5a** Add module-scoped fixtures for repeated `derive_uniform_boundary_for_temo(E4_1)` calls:
  - Currently called 8 times (without zeros) and 10 times (with zeros={3,4}) across class fixtures and standalone functions.
  - Add to `conftest.py` or top of `test_e4_cut_cell.py`:
    ```python
    @pytest.fixture(scope="module")
    def e4_1_uniform():
        return derive_uniform_boundary_for_temo(E4_1)

    @pytest.fixture(scope="module")
    def e4_1_uniform_zeros():
        return derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})
    ```
  - Update all class fixtures and standalone functions to use these module fixtures.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`, `scripts/stencil_gen/tests/conftest.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -x -q -k "not slow"`

- [ ] **37.5b** Add module-scoped fixture for `derive_cut_cell_scheme(E4_1, psi)`:
  - Currently called 7 times independently. Each takes ~0.7s.
  - Add fixture, update `TestDeriveCutCellScheme` and standalone functions to use it.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -x -q -k "not slow"`

### 37.6 — Update CLAUDE.md test commands

- [ ] **37.6a** Update CLAUDE.md test commands to use the new default (slow tests skipped):
  - Replace the long `-k "not TestMathematicaWorkflow and ..."` exclusion with the simple default behavior.
  - Add `--run-slow` documentation for when full sweeps are needed.
  - File: `CLAUDE.md`
  - Test: N/A (documentation)

### 37.7 — Verify final timing

- [ ] **37.7a** Run `uv run pytest tests/ --durations=10 -q` and verify total time is under 15 seconds:
  - Expected: fast unit tests (~7s) + cached E4 cut-cell tests (~5s) + fast PHS regression (~2s) + group velocity tests (~1s) = ~15s.
  - If still >15s, identify remaining bottlenecks and add more caching or slow markers.
  - File: N/A (verification only)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/ --durations=10 -q`

---

## Ordering

```
37.1a (slow marker infra) — do first, everything depends on it
37.2a-b (mark sweep classes) — independent of 37.3, can do in parallel
37.3a-d (fast replacements) — can start after 37.1a
37.4a-c (mark e4 slow tests) — independent of 37.2-37.3
37.5a-b (cache derivations) — independent of 37.2-37.4
37.6a (update docs) — do after 37.2-37.5 are done
37.7a (verify timing) — do last
```

---

## Completion Criteria

- Default `uv run pytest tests/ -x -q` completes in under 15 seconds.
- `uv run pytest tests/ -x -q --run-slow` still runs all research sweeps (no tests deleted).
- Fast regression tests verify all known-good values that the slow sweeps originally discovered.
- No production/research sweep runs during default CI testing.
- All existing tests still pass when `--run-slow` is used.
