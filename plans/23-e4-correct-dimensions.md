# Phase 23: E4 Cut-Cell with Correct Dimensions and Conservation

**Goal:** Re-derive the E4_1 cut-cell stencil with the correct dimensions (R=5, T=7 per Eq. 11b: r = q+1+nextra = 4) and enforce discrete conservation. The previous derivation used wrong dimensions (R=4) which made conservation structurally infeasible. With the correct R=5, conservation should be achievable.

**Depends on:** Phase 20, 21, 22 (dimension fix committed in `27621e3`)

**Read first:**
- `scripts/stencil_gen/stencil_gen/temo.py` (just-fixed `compute_dimensions`, `construct_cut_cell_stencil`, `derive_uniform_boundary_for_temo`)
- `scripts/stencil_gen/tests/test_e4_cut_cell.py` (currently 15 failures + 38 errors due to dimension change)
- `scripts/stencil_gen/tests/test_temo.py` (E2_1 and E2_2 tests — should all still pass)
- `plans/stencil-derivation-math-reference.md` (Section 4: TEMO, especially Eq. 11)
- `src/stencils/E2_1.cpp` (reference conservative cut-cell stencil: R=4, T=5)

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/ -v
cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v
cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v
```

---

## Current State

- `compute_dimensions` fixed: E4_1 now gives r=4, t=6, R=5, T=7 (was r=3, R=4)
- E2_1 and E2_2 tests all pass (253 tests) — dimensions unchanged since p=q=1
- E4_1 tests: 15 failures + 38 errors — all hardcoded R=4 assumptions
- The "Approach A/B/C/D" infeasibility tests from Phase 22 are now invalid (they proved infeasibility at the WRONG dimensions)
- The E4_1 C++ stencil in `src/stencils/E4_1.cpp` was generated with wrong dimensions and must be regenerated
- `derive_uniform_boundary_for_temo` needs to handle r=4 (was only tested with r=3)

**Key dimension change:**
- E4_1 uniform boundary: 4×6 (was 3×6) — one more boundary row
- E4_1 cut-cell: 5×7 (was 4×7) — one more row for conservation DOF
- Conservation: 7 equations, 5 weight unknowns → 2 excess constraints (feasible)
- Expected free params after conservation: 4 (matching Table 1)

---

## Items

### 23.1 — Delete invalid infeasibility tests

- [ ] **23.1a** Remove the Phase 22 infeasibility test classes that were based on wrong dimensions:
  - Delete `TestApproachAMinorConditions` (proved infeasibility at R=4, now R=5)
  - Delete `TestApproachBParametricWeights` (same)
  - Delete `TestApproachCEntryLevelUnknowns` (same)
  - Delete `TestApproachDIncreasedDimensions` (same)
  - Keep the xfail conservation test `test_e4_1_conservation_columns_xfail` — it should be fixed to pass later
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: errors should drop from 38 to near 0

### 23.2 — Fix E4_1 uniform boundary bridge for r=4

- [ ] **23.2a** Update `derive_uniform_boundary_for_temo` to handle the new E4_1 dimensions:
  - With r=4, the uniform boundary has 4 rows × 6 columns (was 3×6)
  - Row 0..2: each has `t - (q+1) = 6 - 4 = 2` free columns
  - Row 3: the conservation-constrained row (analogous to the last row in E2_1)
  - The conservation constraint (column sums = 0) determines row 3's entries
  - Free params: from Table 1, 4 survive conservation (α_{04}, α_{14}, α_{24}, α_{25}), 2 are zeroed (α_{05}, α_{15})
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: verify B_u shape is (4, 6)

- [ ] **23.2b** Fix all E4_1 dimension assertions in tests:
  - Update shape checks from R=4 to R=5, uniform rows from 3 to 4
  - Update coefficient count checks (floating: 5×7=35, Dirichlet: 4×7=28)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v`

### 23.3 — Enforce conservation in the cut-cell stencil

- [ ] **23.3a** Add conservation enforcement to the E4_1 TEMO pipeline:
  - After constructing the 5×7 cut-cell stencil B_l(ψ), apply conservation
  - Conservation: Σ_i w_i · B[i,j] = 0 for interior columns, with w_0 = ψ
  - With R=5 and 5 weight unknowns, the 7-equation system has 2 excess constraints
  - These 2 constraints consume 2 of the 10 free stencil entries
  - Plus the wall-column constraint consumes 1 more → 7 free entries
  - With 2 zeroed entries and 1 more from alpha distribution → 4 free alphas (matches Table 1)
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **23.3b** Verify conservation holds symbolically:
  - Check `Σ_i w_i · B[i,j] = 0` as a polynomial identity in ψ and α for ALL interior columns
  - Remove the xfail marker from the conservation test
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k conservation`

### 23.4 — Re-generate E4_1 C++ code

- [ ] **23.4a** Re-generate E4_1.cpp with correct dimensions and conservation:
  - P=2, R=5, T=7 (was R=4)
  - 5×7=35 floating coefficients (was 4×7=28)
  - 4×7=28 Dirichlet coefficients (was 3×7=21)
  - Update `src/stencils/E4_1.cpp` and `src/stencils/E4_1.t.cpp`
  - Rebuild and test: `cmake --build build --target t-E4_1 && ctest --test-dir build -R t-E4_1`
  - File: `src/stencils/E4_1.cpp`, `src/stencils/E4_1.t.cpp`

### 23.5 — Update plans and decision records

- [ ] **23.5a** Update meta.md: revert D-R25, document the correct formula:
  - D-R25 was wrong — Eq. 11b gives `r = q + 1 + nextra`, not `r = p + 1 + nextra`
  - The formulas were only for 1st derivative operators; 2nd derivatives use different sizing
  - File: `plans/meta.md`

- [ ] **23.5b** Update Phase 22 plan to document the resolution:
  - The infeasibility was caused by wrong dimensions, not a fundamental mathematical limitation
  - With correct R=5, conservation is feasible
  - File: `plans/22-cut-cell-conservation.md`

### 23.6 — Regression: verify E2_1 and E2_2 unchanged

- [ ] **23.6a** Verify all E2 tests still pass:
  - Run full test suite and confirm E2_1 and E2_2 tests are unaffected
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v`

---

## Key Insight

The paper's Eq. 11b (`r = q + 1 + nextra`) was correct all along. The D-R25 "correction" to `r = p + 1 + nextra` was introduced because E4_2 (2nd derivative) dimensions didn't match, but the Eq. 11 sizing formula applies only to 1st derivative operators. For E4_1 (1st derivative, q=3, nextra=0), the correct uniform boundary has r=4 rows (not 3), giving the cut-cell stencil R=5 rows — enough for conservation.
