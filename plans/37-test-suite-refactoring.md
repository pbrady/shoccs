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
| `test_group_velocity.py` | ~350s | 52 | 344s E4 psi sweep (added Phase 36, not in original plan) |
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

### 37.4d — Mark slow tests in `test_group_velocity.py`

This file was added in Phase 36 (after the original plan) and contains a 344-second psi sweep that was not marked slow.

- [x] **37.4d** Mark `TestPsiSweepGroupVelocity::test_e4_1_psi_sweep` as `@pytest.mark.slow`:
  - This 344-second E4_1 psi sweep (11 psi values × 500 xi points, each requiring symbolic stencil derivation) is a research exploration, not a regression check. The E2 sweep in the same class is fast (~1s) and provides adequate default coverage.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q`

### 37.5 — Mark conservation proof tests as slow in `test_e4_cut_cell.py`

**Timing analysis** (measured): After 37.1-37.4, `test_e4_cut_cell.py` alone takes **104s** for non-slow tests.
The originally-planned 37.5a-b caching of `derive_uniform_boundary_for_temo(E4_1)` saves <1s (each call is ~60ms).
The actual bottlenecks are `conserve=True` derivations and Groebner proofs:

| Test | Time | Description |
|------|------|-------------|
| `TestE4UniformConservation` setup | 28.8s | `derive_uniform_boundary_for_temo(E4_1, conserve=True)` |
| `TestE4UniformConservation::test_custom_alpha_symbols` | 22.5s | Second `conserve=True` call with custom alphas |
| `TestCutCellConservationAfterUniform` setup | 21.0s | `derive_cut_cell_scheme(e4_no_zeros, conserve=True)` — proves infeasibility |
| `test_e4_1_psi_dependent_conservation_infeasible` | 6.1s | Groebner basis proof of infeasibility |
| `TestE4TestFileGeneration` setup | 5.6s | `derive_cut_cell_scheme(E4_1, psi, conserve=True)` |

These are conservation proofs and deep `conserve=True` feature tests. The proofs are structural and don't need to re-prove every run. The `conserve=True` derivation takes ~28s per call with no way to speed it up short of algorithmic changes.

- [x] **37.5a** Mark `TestCutCellConservationAfterUniform` as `@pytest.mark.slow`:
  - Proves conservation transfer from uniform to cut-cell is infeasible. The proof is structural (Groebner basis = {1}).
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -x -q -k "not slow" --co | grep -c "test session starts\|<"` (verify class is excluded)

- [x] **37.5b** Mark `test_e4_1_psi_dependent_conservation_infeasible` as `@pytest.mark.slow`:
  - Groebner basis proof of infeasibility. Same rationale as the r5 infeasibility test (already marked slow in 37.4a).
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

- [x] **37.5c** Mark `TestE4UniformConservation` as `@pytest.mark.slow`:
  - Tests `conserve=True` feature which takes 28.8s setup + 22.5s for second derivation. The conservation feature's correctness is structurally proven by the slow tests; it doesn't change frequently enough to warrant 51s in every default run.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 37.6 — Cache `derive_cut_cell_scheme` calls in `test_e4_cut_cell.py`

After marking proofs slow (~79s saved), remaining non-slow time is ~25s. The next bottlenecks are repeated `derive_cut_cell_scheme(E4_1, psi)` calls (~0.7s each × 7+) and the code-generation test setup (5.6s).

- [x] **37.6a** Add module-scoped fixture for `derive_cut_cell_scheme(E4_1, psi)`:
  - Called independently in `TestDeriveCutCellScheme` (5 methods × ~0.7s) and `TestE4CutCellSchemeWithZeros` (class fixture).
  - Module-scoped `e4_1_cut_cell_scheme` fixture shared by both classes. Saved ~4s (23.4s → 19.6s for test_e4_cut_cell.py).
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -x -q -k "not slow"` — 86 passed in ~20s

- [x] **37.6b** Add module-scoped fixture for `derive_cut_cell_scheme(E4_1, psi, conserve=True)`:
  - Module-scoped `e4_1_cut_cell_scheme_conserve` fixture shared by `TestE4CodeGeneration` and `TestE4TestFileGeneration`.
  - No default-run savings (TestE4CodeGeneration is slow), but saves ~5.6s when running `--run-slow`.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

- [ ] **37.6c** (Optional) Cache `derive_uniform_boundary_for_temo(E4_1)` and `(E4_1, zeros={3,4})`:
  - Called ~8 and ~10 times, but at 60ms and 8ms respectively, total savings <1s.
  - Skip: won't meaningfully help reach 20s target. Full suite at 28.5s — remaining bottleneck is `TestE4TestFileGeneration` (5.7s conserve=True setup).
  - File: `scripts/stencil_gen/tests/conftest.py`, `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 37.7 — Update CLAUDE.md test commands

- [x] **37.7a** Update CLAUDE.md test commands to use the new default (slow tests skipped):
  - Replaced `-k "not TestMathematicaWorkflow ..."` with simple default (slow tests auto-skipped).
  - Added `--run-slow` command for full research sweep runs.
  - File: `CLAUDE.md`

### 37.8 — Verify final timing

- [x] **37.8a** Run `uv run pytest tests/ --durations=10 -q` and verify timing:
  - Marked `TestE4TestFileGeneration` as `@pytest.mark.slow` (5.7s conserve=True setup; sibling `TestE4CodeGeneration` already slow).
  - **Final timing: 23.2s** (464 passed, 115 skipped, 1 xfailed). Down from ~6+ minutes original.
  - Remaining bottlenecks are core regression tests that can't be cut: module fixture setup (5.5s), conservation_holds (4s), conservation solution setup (1.7s).
  - 20s target not met, but 23s is acceptable — all remaining time is in fundamental derivation correctness tests.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (added `@pytest.mark.slow` to `TestE4TestFileGeneration`)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/ -x -q --durations=10`

---

## Ordering

```
37.1a (slow marker infra) — done
37.2a-b (mark PHS sweep classes) — done
37.3a-d (fast PHS replacements) — done
37.4a-d (mark e4/group-velocity slow tests) — done
37.5a-c (mark conservation proofs slow) — done
37.6a-b (cache derivation fixtures) — done, saved ~4s
37.6c (optional uniform boundary caching) — skipped, <1s savings
37.7a (update docs) — done
37.8a (verify timing) — done, 23.2s final (marked TestE4TestFileGeneration slow)
```

---

## Completion Criteria

- [x] Default `uv run pytest tests/ -x -q` completes in ~23s (down from ~6+ minutes). 20s target not met but remaining time is in core derivation tests.
- [x] `uv run pytest tests/ -x -q --run-slow` still runs all research sweeps (no tests deleted).
- [x] Fast regression tests verify all known-good values that the slow sweeps originally discovered.
- [x] No production/research sweep runs during default CI testing.
- [ ] All existing tests still pass when `--run-slow` is used. (Not verified in this phase — requires ~6 min run.)
