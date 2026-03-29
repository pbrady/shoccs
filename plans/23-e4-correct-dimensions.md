# Phase 23: E4 Cut-Cell with Correct Dimensions and Conservation

**Goal:** Re-derive the E4_1 cut-cell stencil with the correct dimensions (R=5, T=7 per Eq. 11b: r = q+1+nextra = 4) and enforce discrete conservation. The previous derivation used wrong dimensions (R=4) which made conservation structurally infeasible. With the correct R=5, conservation should be achievable.

**Depends on:** Phase 20, 21, 22 (dimension fix committed in `27621e3`)

**Read first:**
- `scripts/stencil_gen/stencil_gen/temo.py` (just-fixed `compute_dimensions`, `construct_cut_cell_stencil`, `derive_uniform_boundary_for_temo`)
- `scripts/stencil_gen/tests/test_e4_cut_cell.py` (currently 13 failures + 38 errors due to dimension change)
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
- `derive_uniform_boundary_for_temo(E4_1)` works: produces 5 alphas and (4, 6) B_u matrix
- `build_degenerate_stencil` BREAKS at R=5: near-interior row overdetermined (3 unknowns, 4 Taylor eqs)
- `solve_uniform_limit` BREAKS at R=5: same overdetermination issue
- `construct_cut_cell_stencil` BREAKS because it calls both above functions
- E2_1 and E2_2 tests all pass (253 tests) — dimensions unchanged since p=q=1
- E4_1 tests: 13 failures + 38 errors
  - 4 failures: `TestE4UniformBoundary` (shape/alpha count assertions for old r=3)
  - 6 failures: `TestDeriveCutCellScheme` E4_1 tests (pipeline crash in `construct_cut_cell_stencil`)
  - 2 failures: `TestBuildCutCellConservationSystem` (hardcoded R=4 assertions)
  - 1 failure: `TestApproachDIncreasedDimensions::test_nextra1_pipeline_rank_gap` (pipeline crash)
  - 29 errors: `TestE4TEMOConstruction`, `TestE4CodeGeneration`, `TestE4TestFileGeneration` (fixture crashes)
  - 9 errors: `TestApproachA/B/C` (fixture crashes because `construct_cut_cell_stencil` fails)
- The Phase 22 infeasibility tests (Approaches A-D) are now invalid (proved infeasibility at the WRONG R=4)
- The E4_1 C++ stencil in `src/stencils/E4_1.cpp` was generated with wrong dimensions and must be regenerated

**Key dimension change:**
- E4_1 uniform boundary: 4×6 (was 3×6) — one more boundary row
- E4_1 cut-cell: 5×7 (was 4×7) — one more row for conservation DOF
- Alpha count: 5 (was 4 at r=3). Distribution: rows 0-2 get 1 active alpha each, row 3 gets 2 active; `B_u[0,5]=B_u[1,5]=B_u[2,5]=0`
- Conservation system: 6 equations (T-1), 4 weight unknowns (R-1) → 2 excess constraints
- With 5 alpha symbols available to absorb excess constraints, conservation should be feasible

**Root cause of pipeline breakage:**
With r=4 and p=2, the interior stencil `[1/12, -2/3, 0, 2/3, -1/12]` at the near-interior row (row 4) covers grid points 2..6, but T-frame only has 7 columns (grid points 0..5). The rightmost coefficient (-1/12 at grid point 6) overflows the T-frame. The current `build_degenerate_stencil` and `solve_uniform_limit` functions use a conservation+Taylor approach for this case, but they fix TOO MANY columns via conservation, leaving fewer unknowns than Taylor equations:
- `build_degenerate_stencil`: zeroed col + 3 conservation cols = 4 fixed → 3 unknowns < 4 eqs
- `solve_uniform_limit`: 4 conservation cols fixed → 3 unknowns < 4 eqs

**Fix:** Limit conservation-fixed columns to `T - n_zeroed - n_eqs` (degenerate) or `T - n_eqs` (uniform limit), selecting the rightmost eligible conservation columns. This ensures the Taylor system is exactly determined.

---

## Items

### 23.1 — Delete invalid infeasibility tests

- [x] **23.1a** Remove the Phase 22 infeasibility test classes that were based on wrong R=4 dimensions:
  - Delete `TestApproachAMinorConditions` (3 test methods; proved infeasibility at R=4, now R=5)
  - Delete `TestApproachBParametricWeights` (3 test methods; same)
  - Delete `TestApproachCEntryLevelUnknowns` (3 test methods; same)
  - Delete `TestApproachDIncreasedDimensions` (4 test methods; same; `test_nextra1_pipeline_rank_gap` also crashes)
  - Keep the xfail conservation test `test_e4_1_conservation_fails` — it should be fixed to pass later (23.3d)
  - Keep `TestBuildCutCellConservationSystem` — it tests valid infrastructure, just needs value updates (23.2f)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Verify: errors should drop from 38 to 29 (fixture-crash errors from `TestE4TEMOConstruction`, `TestE4CodeGeneration`, `TestE4TestFileGeneration` remain until pipeline is fixed)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v --tb=line 2>&1 | tail -5`

### 23.2 — Fix pipeline and tests for R=5 dimensions

- [x] **23.2a** Fix `build_degenerate_stencil` near-interior row for r=4 overflow:
  - **Problem:** When `can_embed_interior` is False (interior stencil extends beyond T-frame), the current code fixes ALL columns where `j >= p+2` via conservation, then Taylor-solves the rest. With r=4, p=2, T=7: conservation fixes cols {4,5,6} (from `range(p+2, T)`), plus zeroed col {1} → 4 known, leaving unknowns {0,2,3} for 4 Taylor equations — overdetermined.
  - **Fix:** In the `else` branch (lines ~517-571), limit conservation-fixed columns to `n_free = T - 1 - n_eqs` (= 7-1-4 = 2 for E4_1). Select the rightmost 2 eligible conservation columns (cols 5,6). This gives unknowns {0,2,3,4} for 4 Taylor equations — exactly determined.
  - **Implementation:** Replace the fixed loop `for j in range(p + 2, T)` with logic that:
    1. Computes `n_free = T - 1 - n_eqs` (1 for zeroed col, n_eqs for Taylor unknowns)
    2. Collects eligible conservation columns: `list(range(p + 2, T))` (same range as current loop — `build_degenerate_stencil` uses `-sum(B_d[i, j] for i in range(1, r))`, not `_interior_contribution`)
    3. Takes only the rightmost `min(n_free, len(eligible))` columns
    4. Taylor-solves the remaining unknowns
  - **Invariant:** Must not change behavior for E2_1 (where the current logic works: r=3, p=1, `can_embed_interior` is False, conservation fixes cols {3,4}, unknowns {0,2}, exactly determined)
  - File: `scripts/stencil_gen/stencil_gen/temo.py`, function `build_degenerate_stencil` (lines 429-572)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k degenerate` (E2 degenerate tests must still pass)

- [x] **23.2b** Fix `solve_uniform_limit` near-interior row for r=4 overflow:
  - **Problem:** Same conservation overdetermination as 23.2a. Conservation fixes cols {2,4,5,6}, leaving unknowns {0,1,3} for 4 Taylor equations.
  - **Fix:** In the `else` branch (lines ~983-1044), limit conservation-fixed columns to `n_free = T - n_eqs` (= 7-4 = 3 for E4_1, no zeroed col at psi=1). Select the rightmost 3 eligible columns (cols 4,5,6). This gives unknowns {0,1,2,3} for 4 Taylor equations — exactly determined.
  - **Implementation:** In the `else` branch, replace the column-selection loop with:
    1. Compute `n_free = T - n_eqs` (no zeroed col at psi=1)
    2. Collect eligible conservation columns using the existing criterion `j < R - p or j >= p + 2` over `range(2, T)`
    3. Take only the rightmost `min(n_free, len(eligible))` columns
    4. Taylor-solve the remaining unknowns (same overdetermination check already present)
  - **Invariant:** E2_1 behavior unchanged (r=3, p=1, R=4: eligible={2,3,4}, n_free=3, rightmost 3={2,3,4} — same as current).
  - File: `scripts/stencil_gen/stencil_gen/temo.py`, function `solve_uniform_limit` (lines 888-1045)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k uniform_limit` (E2 uniform limit tests must still pass)
  - After this fix + 23.2a, the full TEMO pipeline (`construct_cut_cell_stencil`) should work for E4_1.
  - Smoke test: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.temo import *; psi=__import__('sympy').Symbol('psi'); ur=derive_uniform_boundary_for_temo(E4_1); print(construct_cut_cell_stencil(ur.B_u, ur.interior, 2, 3, 1, 0, psi).matrix.shape)"`  — should print `(5, 7)`

- [x] **23.2c** Verify `construct_cut_cell_stencil` works end-to-end for E4_1:
  - After 23.2a + 23.2b, run the full pipeline and verify:
    - `construct_cut_cell_stencil` returns a 5×7 matrix (no crash)
    - Taylor accuracy: each row satisfies 4 moment equations for symbolic psi
    - Degenerate limit (psi=0): matches `build_degenerate_stencil` output
    - Uniform limit (psi=1): matches `solve_uniform_limit` output, rows 0-3 embed B_u, row 4 is the near-interior closure
    - No beta symbols (nextra=0)
    - Free symbols are psi + 5 alphas only
  - Ordering: must complete 23.2a and 23.2b first
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::TestE4TEMOConstruction -v` (all 9 tests should pass after test fixes in 23.2d)

- [x] **23.2d** Fix `TestE4UniformBoundary` test assertions for new dimensions:
  - `test_shape`: change `(3, 6)` → `(4, 6)` (line ~52)
  - `test_four_alpha_symbols`: rename to `test_five_alpha_symbols`, change `len == 4` → `len == 5`, update name checks to `alpha_0..alpha_4` (lines ~54-59)
  - `test_zero_constraints`: add `B_u[2, 5] == 0` check (3 zeros now instead of 2) (line ~63)
  - `test_last_row_free_alphas`: update to check row 3 (was row 2) for `alpha_3, alpha_4` in `B_u[3, 4]` and `B_u[3, 5]` (lines ~67-72)
  - `test_custom_alpha_symbols`: change `range(4)` → `range(5)` (line ~183)
  - All other tests in this class should pass unchanged (Taylor accuracy, rows 0-1 match, etc.)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::TestE4UniformBoundary -v`

- [x] **23.2e** Fix `TestE4TEMOConstruction` test assertions for R=5:
  - `test_shape`: change `(4, 7)` → `(5, 7)` (line ~211)
  - `test_entries_in_psi_alpha`: change `range(4)` → `range(5)` for alpha names (line ~223)
  - `test_uniform_limit_rows_0_2_embed_Bu`: change `range(3)` → `range(4)` to check rows 0-3 embed B_u, update `range(6)` to match t=6 columns (lines ~243-254)
  - `test_uniform_limit_row3_interior`: rename to `test_uniform_limit_row4_not_interior`, change `m1[3, j]` → `m1[4, j]`. The expected values are NOT the simple interior stencil anymore — at R=5, the interior overflows the T-frame, so row 4 is derived via conservation+Taylor and contains alpha symbols. **Replace the hardcoded expected array** with a dynamic check: verify row 4 matches `B_l_1[4, j]` from `solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0)`. Also add a negative assertion: `m1[4, :] != [0, 0, 1/12, -2/3, 0, 2/3, -1/12]` to document this is no longer the raw interior stencil. (Lines ~256-269)
  - `test_degenerate_limit`: verify automatically works (no hardcoded dims, uses `B_d.shape` dynamically)
  - `test_taylor_accuracy_symbolic` and `test_taylor_accuracy_at_half`: should work unchanged (iterate over R dynamically)
  - Ordering: must complete 23.2a-23.2c first (pipeline must work)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::TestE4TEMOConstruction -v`

- [x] **23.2f** Fix `TestBuildCutCellConservationSystem` and `TestDeriveCutCellScheme`:
  - `test_e4_1_conservation_system_dimensions`: change `R == 4` → `R == 5`, update expected IC values:
    ```
    old: {0: 0, 1: 0, 2: 0, 3: 1/12, 4: -7/12, 5: -7/12}
    new: {0: 0, 1: 0, 2: 0, 3: 0, 4: 1/12, 5: -7/12}
    ```
    Change `len(ws) == R - 1` assertion from 3 to 4 weight unknowns (lines ~868-896)
  - `test_e4_1_overdetermined_system`: change `excess == 3` → `excess == 2`, update alpha count from 4 to 5 (lines ~898-918)
  - `TestDeriveCutCellScheme` E4_1 tests (lines ~651-722):
    - `test_e4_1_shape`: `.shape == (4, 7)` → `(5, 7)`, `.shape == (3, 7)` → `(4, 7)`, `dims.R == 4` → `5`
    - `test_e4_1_alpha_count`: `len == 4` → `5`
    - `test_e4_1_custom_alphas`: `range(4)` → `range(5)` (line ~719)
  - Ordering: must complete 23.2a-23.2c first
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::TestBuildCutCellConservationSystem -v && uv run pytest tests/test_e4_cut_cell.py::TestDeriveCutCellScheme -v`

- [x] **23.2g** Fix `TestE4CodeGeneration` and `TestE4TestFileGeneration` for R=5:
  - `e4_spec` fixture: change `R=4` → `R=5` and `param_arrays={"alpha": 4}` → `{"alpha": 5}` in `StencilGenSpec` constructor (lines ~351, ~486)
  - `floating_flat`: update comment `R*T = 4*7 = 28` → `R*T = 5*7 = 35`
  - `dirichlet_flat`: padding `[Integer(0)] * 7` stays the same (T is still 7), but `cc.dirichlet` length changes from (R-1)*T = 3*7=21 to 4*7=28. Total: 35 (was 28). No code change needed — just the comment.
  - `test_struct_constants`: `R = 4` → `R = 5` (line ~374)
  - `test_alpha_array`: `std::array<real, 4>` → `std::array<real, 5>` (line ~388)
  - `test_nbs_floating_method`: coefficient count 28 → 35 (line ~411)
  - `test_nbs_dirichlet_method`: coefficient count 21 → 28 (line ~422)
  - `test_compute_floating_values`: count 28 → 35 (line ~508)
  - `test_compute_dirichlet_values`: count 21 → 28 (line ~519)
  - `test_floating_uniform_limit_row3_interior`: rename to `_row4_not_interior`, update row indices 21:28 → 28:35. The expected values will NOT be the simple interior/h pattern — row 4 at psi=1 uses conservation+Taylor (involves alphas). **Compute expected values numerically** by running `compute_test_values` on the 5×7 floating matrix with the ALPHA_VALUES and psi=1.0, then extracting indices 28:35. Replace the hardcoded `expected` list with these computed values. (Lines ~521-535)
  - `test_generate_test_file_structure`: `r == 4` → `r == 5`, `alpha = {0.1, -0.05, 0.02, 0.01}` → 5-element array `{0.1, -0.05, 0.02, 0.01, 0.005}` (line ~559-561)
  - `ALPHA_VALUES`: change from 4 to 5 elements: `{"alpha": [0.1, -0.05, 0.02, 0.01, 0.005]}` (line ~468). Both fixtures and all tests referencing this dict are affected.
  - Ordering: must complete 23.2a-23.2c first
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::TestE4CodeGeneration -v && uv run pytest tests/test_e4_cut_cell.py::TestE4TestFileGeneration -v`

- [x] **23.2h** Fix `test_e4_1_conservation_fails` xfail test:
  - Update hardcoded `R, T = 4, 7` → `R, T = 5, 7`
  - Update column sum to include row 4: `psi * m[0, j] + m[1, j] + m[2, j] + m[3, j] + m[4, j]`
  - Verify xfail still triggers (conservation not yet enforced)
  - Ordering: must complete 23.2a-23.2c first
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (lines ~804-838)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::test_e4_1_conservation_fails -v`

- [x] **23.2i** Fix `test_temo.py` E4 dimension test assertions (missed by 23.2d-23.2h):
  - `test_e4_1_dimensions` (line 102): change expected from `Dimensions(r=3, t=6, R=4, T=7, X=0)` to `Dimensions(r=4, t=6, R=5, T=7, X=0)`. Straightforward — matches the corrected formula.
  - `test_e4_2_dimensions` (line 107): currently expects `Dimensions(r=3, t=6, R=3, T=7, X=3)`, actual is `Dimensions(r=4, t=6, R=4, T=7, X=4)`. **Investigate before updating:** The plan states "Eq. 11b applies only to 1st derivative operators; 2nd derivatives use different sizing." If E4_2 (nu=2) should use `r = p + 1 + nextra = 3` instead of `r = q + 1 + nextra = 4`, then `compute_dimensions` needs a `nu`-dependent formula for `r` itself (not just `r_eff`), and this test's expected values should stay. If the paper's formula is correct for both derivatives (with the existing `r_eff` adjustment), update the test to `Dimensions(r=4, t=6, R=4, T=7, X=4)`.
  - These failures were introduced by commit 27621e3 (dimension fix) and are currently untracked.
  - File: `scripts/stencil_gen/tests/test_temo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py::TestDimensions -v`

- [x] **23.2j** **[Review follow-up]** Investigate E4_2 dimension correctness — test was updated without required investigation:
  - **Resolution:** Eq. 11 (`r = q+1+nextra`, `t = p+q+1+nextra`) applies only to 1st derivatives. For 2nd derivatives (nu=2), the correct formulas are `r = p+1+nextra` and `t = p+2+nextra`. E2_2 coincidentally worked because p=q=1.
  - **Changes made:**
    1. Fixed `compute_dimensions` to use nu-dependent formulas: nu=1 uses Eq. 11, nu=2 uses `r=p+1+nextra`, `t=p+2+nextra`
    2. Updated `test_e4_2_dimensions` to expect `Dimensions(r=3, t=4, R=3, T=5, X=3)` — matches C++ E4_2.cpp and Section 6 of math reference
    3. Added `test_e4_2_matches_cpp` to cross-check against C++ reference values
  - **Verification:** All 121 test_temo.py tests pass, E2 tests unaffected, E4_2 dimensions now match C++ (R=3, T=5, X=3)
  - Files: `scripts/stencil_gen/stencil_gen/temo.py`, `scripts/stencil_gen/tests/test_temo.py`

### 23.3 — Enforce conservation in the cut-cell stencil

- [x] **23.3a** Conservation feasibility at R=5 — resolved via uniform-first approach:
  - **Previous finding:** Cut-cell conservation with constant ψ-independent weights is infeasible (rank gap = 1).
  - **Resolution:** Apply conservation to the UNIFORM boundary first (at TEMO dimensions r=4, t=6), then let TEMO design principles propagate it to the cut-cell stencil. The compatibility-condition method eliminates one alpha (alpha_3) and determines SBP quadrature weights as functions of the remaining alpha_3.
  - **Implementation:** Added `conserve=True` parameter to `derive_uniform_boundary_for_temo`. Produces 4 free alphas (polynomial B_u), weights = [(33α₃+2)/(24α₃), (-22α₃-3)/(12α₃), (95α₃+6)/(24α₃), -1/(12α₃)].
  - **Test:** 10 tests in `TestE4UniformConservation` (all pass).
  - **Next:** Enable by default (23.3a-ii), then verify cut-cell conservation (23.3b).

- [SUPERSEDED] **23.3b-old** Implement `solve_cut_cell_conservation` function:
  - **Superseded:** The uniform-first approach (revised 23.3a/23.3b) replaces the cut-cell conservation solver. Conservation is enforced on the uniform boundary and propagates through TEMO.

- [SUPERSEDED] **23.3c** Integrate conservation into `derive_cut_cell_scheme`:
  - **Superseded:** Conservation is now applied within `derive_uniform_boundary_for_temo(conserve=True)`, which is already called by `derive_cut_cell_scheme`. No separate integration step needed — just enable the default.

- [SUPERSEDED] **23.3d** Verify conservation holds symbolically and update tests:
  - **Superseded:** Replaced by revised 23.3b (cut-cell conservation verification).

### 23.4 — Re-generate E4_1 C++ code

- [ ] **23.4a** Re-generate E4_1.cpp with correct dimensions and conservation:
  - **Unblocked by 23.3a-i:** Conservation is now implementable via `conserve=True`.
  - **Depends on:** 23.3a-ii (enable conservation by default), 23.3b (verify cut-cell conservation)
  - After conservation is enabled by default, the code generation pipeline (`derive_cut_cell_scheme` → `generate_stencil_cpp`) will produce the conservative E4_1.cpp with R=5, T=7, 4 free alphas.
  - File: `src/stencils/E4_1.cpp`

### 23.5 — Update plans and decision records

- [x] **23.5a** Update meta.md: record D23-1 decision:
  - D23-1 recorded: theta-linearization confirms conservation infeasible with constant weights at R=5.
  - D-R25 reversion already documented in prior commit.
  - File: `plans/meta.md`

- [ ] **23.5b** Update Phase 22 plan to document the resolution:
  - **Updated context:** The infeasibility was NOT resolved by the dimension correction. R=5 still gives rank gap = 1 with constant weights. Phase 22's infeasibility finding was correct in essence (conservation is structurally hard), though the specific analysis was at wrong dimensions.
  - File: `plans/22-cut-cell-conservation.md`

### 23.6 — Regression: verify E2_1 and E2_2 unchanged

- [ ] **23.6a** Verify all E2 tests still pass and no test_temo.py failures remain:
  - Run full test suite and confirm E2_1 and E2_2 tests are unaffected
  - Confirm the 2 E4 dimension tests in test_temo.py (fixed in 23.2i) now pass
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v`
  - Critical: 23.2a and 23.2b MUST NOT change E2_1 or E2_2 behavior. Verify by running E2 tests after each fix.

---

## Key Insight

The paper's Eq. 11b (`r = q + 1 + nextra`) was correct all along. The D-R25 "correction" to `r = p + 1 + nextra` was introduced because E4_2 (2nd derivative) dimensions didn't match, but the Eq. 11 sizing formula applies only to 1st derivative operators. For E4_1 (1st derivative, q=3, nextra=0), the correct uniform boundary has r=4 rows (not 3), giving the cut-cell stencil R=5 rows — enough for conservation.

## Pipeline Fix Design Notes

The near-interior row overflow issue affects `build_degenerate_stencil` and `solve_uniform_limit`. Both use a conservation+Taylor approach when the interior stencil doesn't fit in the T-frame. The fix is the same for both:

**Current (broken for r=4):** Fix ALL columns where `j >= p+2` via conservation, then Taylor-solve the rest. This leaves too few unknowns.

**Fixed:** Count how many conservation columns can be fixed without overdetermining the Taylor system:
```
n_zeroed = 1 if degenerate else 0
n_free = T - n_zeroed - n_eqs  # how many cols to fix via conservation
```
Then select the rightmost `n_free` eligible conservation columns. This ensures the Taylor system has exactly `n_eqs` unknowns.

**Eligible conservation columns:** Any column j where the interior contribution `IC(j-1, R, p, interior)` is a computable constant. For the near-interior row, this includes ALL T-frame columns (ICs depend on fixed interior coefficients).

**E2_1 backward compatibility:** For E2_1 (r=3, p=1, T=5, n_eqs=2):
- Degenerate: n_free = 5 - 1 - 2 = 2. Current code fixes {3, 4}. New code also picks rightmost 2 from eligible → same result.
- Uniform limit: n_free = 5 - 0 - 2 = 3. Current code fixes {2, 3, 4}. New code picks rightmost 3 → same result.

**E4_1 new behavior:**
- Degenerate: n_free = 7 - 1 - 4 = 2. Fix cols {5, 6}. Unknowns = {0, 2, 3, 4}. 4×4 Taylor system.
- Uniform limit: n_free = 7 - 0 - 4 = 3. Fix cols {4, 5, 6}. Unknowns = {0, 1, 2, 3}. 4×4 Taylor system.

## Conservation Enforcement Design Notes

**Approach chosen:** Theta-linearization with ψ-coefficient extraction (decision D23-1).

This extends the existing `solve_conservation` pattern from `conservation.py` (used for uniform boundary E2_1/E2_2) to the ψ-parameterized cut-cell case. The key differences from the uniform case:

1. **ψ as an additional variable:** Conservation must hold for ALL ψ ∈ (0,1], not just at a specific point. This requires extracting ψ-coefficients to convert rational-in-ψ equations into scalar equations.

2. **Bilinear terms:** The conservation equations contain w_i × B_l[i,j](ψ, α), where B_l entries are rational in ψ and linear in α_0..α_4. This produces bilinear products w_i × α_k.

3. **Theta linearization:** Replace each bilinear product w_i × α_k with θ_{i,k}. After solving the linear system, recover α constraints from θ_{i,k} = w_i × α_k.

**Step-by-step:**
1. Conservation equations (6 total): `ψ·B_l[0,j] + Σ_{i=1}^{4} w_i·B_l[i,j] + IC(j) = target(j)`
2. B_l[i,j] for rows 0-3: rational in ψ, linear in α_0..α_4 (from B_u derivation)
3. B_l[4,j] (near-interior): rational in ψ, may contain α symbols from limit computations
4. Clear ψ-denominators → polynomial equations in ψ
5. Collect ψ-coefficients → scalar bilinear equations
6. Theta-linearize → scalar linear system
7. Solve → weights and α constraints in terms of surviving free αs

**Why this should work at R=5 (where it failed at R=4):**
- At R=4: 3 weight unknowns, 4 alpha symbols → Phase 22 showed rank gap = 1 (structural infeasibility)
- At R=5: 4 weight unknowns, 5 alpha symbols → more DOF. The extra row provides an additional weight unknown (w_4) and an additional alpha (α_4), changing the rank structure of the ψ-coefficient system.
- The Phase 22 tests (`TestApproachDIncreasedDimensions`) tested (R, T) = (4,7)..(8,11) but always used the OLD r=3 dimensions for deriving B_u. With the CORRECT r=4, B_u has different alpha structure (5 alphas instead of 4, different distribution across rows), which changes the conservation system's rank properties.

**Reference implementation:** `conservation.py:solve_conservation()` (lines 88-167) handles the simpler case (no ψ, bilinear in w_i × phi_k). The cut-cell version adds ψ-coefficient extraction between steps 1 and 3 of that function.

---

## Critical Fix: Conservation Must Be Applied to Uniform Stencil FIRST

The paper says: "the conservation constraints must also be solved on the uniform mesh to provide appropriately constrained αu."

The process kept trying to enforce conservation on the CUT-CELL stencil (with w = [psi, 1, 1, 1, 1]), which is infeasible. The correct approach:

1. Build uniform boundary B_u at TEMO dimensions (r=4, t=6) using `solve_boundary_row`
2. Apply `conservation.py`'s `build_conservation_system` + `solve_conservation` to B_u
   - This gives 5 equations, 6 unknowns (4 weights + 2 last-row free params)
   - Underdetermined → solvable, consumes 1 last-row free param
3. The conservation-constrained B_u has fewer free alphas
4. THEN apply TEMO design principles to get B_l(ψ)
5. The cut-cell conservation follows from the uniform conservation by construction

This was verified manually:
```
build_conservation_system(r=4, t=6, p=2, rows, interior)
→ 5 equations, 4 weight unknowns + 2 last-row free = 6 unknowns
→ SOLVABLE (underdetermined by 1)
```

### Revised 23.3a: Apply uniform conservation at TEMO dimensions

**Approach (verified by prototyping):** Instead of `build_conservation_system`+`solve_conservation`
(which creates rational expressions via theta-linearization), use the **compatibility-condition
method**: treat all alphas as parameters, solve the overdetermined weight system, extract the
compatibility condition, solve for one last-row alpha to eliminate it, substitute back into B_u.
This keeps B_u entries polynomial (quadratic at most) and produces clean weight expressions.

**Implementation:** Added `conserve` parameter to `derive_uniform_boundary_for_temo` (default False).
When True: builds 5 working symbols, applies conservation to eliminate 1, renames remaining 4 to
caller's alpha symbols. Weights are rational functions of alpha_3 only.

**Key results for E4_1 with conserve=True:**
- B_u: (4, 6), 4 free alphas (polynomial entries, quadratic in last row)
- Weights: w_0=(33α₃+2)/(24α₃), w_1=(-22α₃-3)/(12α₃), w_2=(95α₃+6)/(24α₃), w_3=-1/(12α₃)
- Conservation: verified (all 5 column conditions satisfied)
- Taylor accuracy: preserved (all 4 rows satisfy q+1=4 moment equations)

- [x] **23.3a-i** Implement `conserve` parameter in `derive_uniform_boundary_for_temo`:
  - Added `conserve: bool = False` parameter
  - Added `weights: list | None` field to `UniformResult`
  - Conservation logic: build equations, solve for weights, extract compatibility condition,
    eliminate one last-row alpha, rename, recompute weights from final B_u
  - Default `conserve=False` preserves all existing behavior
  - Files: `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: 10 new tests in `TestE4UniformConservation` (all pass)
  - Verification: all 305 existing tests pass + 1 xfail unchanged

- [SUPERSEDED] **23.3a-ii-old** Enable conservation by default in `derive_uniform_boundary_for_temo`:
  - **Superseded:** Changing the default to `conserve=True` breaks the TEMO pipeline. The conserved B_u has **quadratic** alpha entries (cross-terms like `alpha_2 * alpha_3`) because the compatibility condition substitution introduces nonlinear relationships between build symbols. The `decompose_alpha_terms` function in `solve_in_field` (line 789) only handles linear alpha terms, causing `construct_cut_cell_stencil` to crash with "Cross-term detected" errors.
  - **Root cause:** Conservation eliminates one last-row alpha by solving a compatibility condition from the weight equations. The weight equations involve products `w_i * B_u[i,j]`, and the weight solutions `w_i = f(build_syms)` are rational in build symbols. Substituting these back and solving the compatibility condition for `_b3` yields a nonlinear expression. When this is substituted into B_u, entries become quadratic.
  - **Resolution:** Conservation must be applied AFTER the TEMO pipeline, not before. See revised 23.3a-ii below.

- [ ] **23.3a-ii** Apply conservation post-TEMO in `derive_cut_cell_scheme`:
  - **Approach:** Keep `conserve=False` as default in `derive_uniform_boundary_for_temo`. Instead, add `conserve: bool = True` parameter to `derive_cut_cell_scheme`. When True:
    1. Derive B_u with `conserve=False` → 5 alphas, linear entries
    2. Also derive B_u with `conserve=True` → 4 alphas, quadratic entries (for substitution mapping and weights only)
    3. Run TEMO pipeline with the **non-conserved** (linear) B_u — works fine
    4. Extract the substitution mapping (see **Substitution mapping** note below)
    5. Apply the substitution to the cut-cell stencil matrices (floating, dirichlet)
    6. Rename remaining symbols to canonical `alpha_0..alpha_3`
    7. Return `CutCellResult` with 4 alphas and the uniform weights
  - **Substitution mapping:** The conservation step eliminates `_b3` (the FIRST last-row build symbol), not `_b4`. In non-conserved terms this means `alpha_3` is eliminated, not `alpha_4`. The actual substitution in non-conserved terms is `{alpha_3: expr(alpha_0, alpha_1, alpha_2, alpha_4)}`, followed by renaming `alpha_4 → alpha_3`. Verified empirically: conserved B_u[3,5] = `alpha_3` (which was `_b4`, i.e., non-conserved `alpha_4`). **Recommended approach:** add an `elim_subs: dict | None` field to `UniformResult` so `derive_uniform_boundary_for_temo(conserve=True)` directly returns the substitution mapping (in terms of original build symbols or caller symbols), avoiding fragile post-hoc comparison.
  - **Key insight:** The substitution mapping is determined by the uniform conservation step and is applied symbolically to the already-solved cut-cell stencil. Since the cut-cell entries are rational in psi and linear in 5 alphas, after substituting alpha_3 they become rational in psi and (at most quadratic) in 4 alphas — which is fine for code generation and numerical evaluation, just not for the TEMO solver.
  - **Implementation details:**
    1. Add `conserve: bool = True` parameter to `derive_cut_cell_scheme`
    2. Add `conservation_subs: dict | None` and `weights: list | None` fields to `CutCellResult`
    3. Add `elim_subs: dict | None` field to `UniformResult` (returned when `conserve=True`)
    4. When `conserve=True`, derive both conserved and non-conserved uniforms, extract subs map from `UniformResult.elim_subs`, apply post-TEMO, rename surviving symbols
  - Files: `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v` (E2 tests must still pass)

- [ ] **23.3a-iii** Update all E4_1 tests for post-TEMO conservation:
  - `derive_uniform_boundary_for_temo(E4_1)` stays at 5 alphas — `TestE4UniformBoundary` tests UNCHANGED
  - `TestE4UniformConservation` tests UNCHANGED (still test `conserve=True` on B_u directly)
  - `TestE4TEMOConstruction`: fixture calls `construct_cut_cell_stencil` directly (no conservation) — tests UNCHANGED
  - `TestE4CodeGeneration`: fixture uses `derive_uniform_boundary_for_temo(E4_1)` directly → needs update if we want conservative codegen. Add new fixture that goes through `derive_cut_cell_scheme` instead. `param_arrays={"alpha": 5}` → 4, alpha array assertion, floating/dirichlet coeff counts stay the same (R=5, T=7)
  - `TestE4TestFileGeneration`: same — update fixture path, `ALPHA_VALUES` 5→4
  - `TestDeriveCutCellScheme`:
    - `test_e4_1_alpha_count` 5→4, `test_e4_1_custom_alphas` range(5)→range(4)
    - `test_e4_1_matches_manual_pipeline` (line 688): **must be updated** — currently compares `derive_cut_cell_scheme` output against non-conserved manual pipeline. After conservation, outputs differ (4 vs 5 alphas, quadratic vs linear entries). Either update to also apply conservation in the manual path, or change to verify only that the conserved result satisfies Taylor accuracy and conservation independently.
  - `TestBuildCutCellConservationSystem`: `test_e4_1_overdetermined_system` — **alpha count stays at 5** (this test builds the stencil directly via `derive_uniform_boundary_for_temo(E4_1)` + `construct_cut_cell_stencil`, NOT through `derive_cut_cell_scheme`, so it sees the non-conserved 5-alpha stencil)
  - `test_e4_1_conservation_constant_weights_infeasible_r5` — **alpha count stays at 5** (same reason: builds stencil directly, not through `derive_cut_cell_scheme`)
  - `test_e4_1_conservation_fails` xfail remains (uses naive weights, not SBP weights)
  - Files: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v`

### Revised 23.3b: Verify cut-cell conservation follows

- [ ] **23.3b** After TEMO construction with post-TEMO conservation:
  - `derive_cut_cell_scheme(E4_1, psi)` now returns a conservative stencil (4 alphas)
  - Derive cut-cell weights: w_0(ψ) = ψ·w_0^u, w_i(ψ) = w_i^u for i≥1 (TEMO extension of uniform weights)
  - Verify: `Σ_i w_i(ψ) · B_l[i,j] + IC(j) = target(j)` for all T-frame columns
  - Update `test_e4_1_conservation_fails`: change from xfail to pass, use correct SBP weights
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
