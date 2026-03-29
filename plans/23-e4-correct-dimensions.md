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

- [ ] **23.2a** Fix `build_degenerate_stencil` near-interior row for r=4 overflow:
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

- [ ] **23.2b** Fix `solve_uniform_limit` near-interior row for r=4 overflow:
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

- [ ] **23.2c** Verify `construct_cut_cell_stencil` works end-to-end for E4_1:
  - After 23.2a + 23.2b, run the full pipeline and verify:
    - `construct_cut_cell_stencil` returns a 5×7 matrix (no crash)
    - Taylor accuracy: each row satisfies 4 moment equations for symbolic psi
    - Degenerate limit (psi=0): matches `build_degenerate_stencil` output
    - Uniform limit (psi=1): matches `solve_uniform_limit` output, rows 0-3 embed B_u, row 4 is the near-interior closure
    - No beta symbols (nextra=0)
    - Free symbols are psi + 5 alphas only
  - Ordering: must complete 23.2a and 23.2b first
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::TestE4TEMOConstruction -v` (all 9 tests should pass after test fixes in 23.2d)

- [ ] **23.2d** Fix `TestE4UniformBoundary` test assertions for new dimensions:
  - `test_shape`: change `(3, 6)` → `(4, 6)` (line ~52)
  - `test_four_alpha_symbols`: rename to `test_five_alpha_symbols`, change `len == 4` → `len == 5`, update name checks to `alpha_0..alpha_4` (lines ~54-59)
  - `test_zero_constraints`: add `B_u[2, 5] == 0` check (3 zeros now instead of 2) (line ~63)
  - `test_last_row_free_alphas`: update to check row 3 (was row 2) for `alpha_3, alpha_4` in `B_u[3, 4]` and `B_u[3, 5]` (lines ~67-72)
  - `test_custom_alpha_symbols`: change `range(4)` → `range(5)` (line ~183)
  - All other tests in this class should pass unchanged (Taylor accuracy, rows 0-1 match, etc.)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::TestE4UniformBoundary -v`

- [ ] **23.2e** Fix `TestE4TEMOConstruction` test assertions for R=5:
  - `test_shape`: change `(4, 7)` → `(5, 7)` (line ~211)
  - `test_entries_in_psi_alpha`: change `range(4)` → `range(5)` for alpha names (line ~223)
  - `test_uniform_limit_rows_0_2_embed_Bu`: change `range(3)` → `range(4)` to check rows 0-3 embed B_u, update `range(6)` to match t=6 columns (lines ~243-254)
  - `test_uniform_limit_row3_interior`: rename to `test_uniform_limit_row4_not_interior`, change `m1[3, j]` → `m1[4, j]`. The expected values are NOT the simple interior stencil anymore — at R=5, the interior overflows the T-frame, so row 4 is derived via conservation+Taylor and contains alpha symbols. **Replace the hardcoded expected array** with a dynamic check: verify row 4 matches `B_l_1[4, j]` from `solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0)`. Also add a negative assertion: `m1[4, :] != [0, 0, 1/12, -2/3, 0, 2/3, -1/12]` to document this is no longer the raw interior stencil. (Lines ~256-269)
  - `test_degenerate_limit`: verify automatically works (no hardcoded dims, uses `B_d.shape` dynamically)
  - `test_taylor_accuracy_symbolic` and `test_taylor_accuracy_at_half`: should work unchanged (iterate over R dynamically)
  - Ordering: must complete 23.2a-23.2c first (pipeline must work)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::TestE4TEMOConstruction -v`

- [ ] **23.2f** Fix `TestBuildCutCellConservationSystem` and `TestDeriveCutCellScheme`:
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

- [ ] **23.2g** Fix `TestE4CodeGeneration` and `TestE4TestFileGeneration` for R=5:
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

- [ ] **23.2h** Fix `test_e4_1_conservation_fails` xfail test:
  - Update hardcoded `R, T = 4, 7` → `R, T = 5, 7`
  - Update column sum to include row 4: `psi * m[0, j] + m[1, j] + m[2, j] + m[3, j] + m[4, j]`
  - Verify xfail still triggers (conservation not yet enforced)
  - Ordering: must complete 23.2a-23.2c first
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (lines ~804-838)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::test_e4_1_conservation_fails -v`

### 23.3 — Enforce conservation in the cut-cell stencil

- [ ] **23.3a** Verify conservation feasibility at R=5 via theta-linearization:
  - **Goal:** Confirm that the conservation rank gap is 0 at R=5 (was 1 at R=4).
  - After 23.2a-23.2c produce a working 5×7 stencil, replicate the Phase 22 Approach C analysis at R=5:
    1. Build the 5×7 stencil via `construct_cut_cell_stencil`
    2. Build conservation equations via `build_cut_cell_conservation_system` → 6 equations in w_1..w_4
    3. Each equation is rational in ψ with bilinear terms w_i × α_k (since B_l[i,j] contains α_0..α_4 in rows 0-3)
    4. Clear the common ψ-denominator from each equation → polynomial in ψ
    5. Collect ψ-coefficients → scalar equations that are bilinear in (w_i, α_k)
    6. Theta-linearize: for row 0 (weight=ψ, fixed), keep as-is; for rows 1-4, replace w_i × α_k → θ_{i,k}
    7. Linear unknowns: {w_1..w_4, θ_{i,k} for each bilinear pair} (exact count depends on which α_k appear in which rows)
    8. Check rank(M) == rank([M|b]) — if so, conservation is feasible
  - **Expected:** rank gap = 0 (the extra row and weight at R=5 resolve the R=4 infeasibility)
  - **If rank gap ≠ 0:** BLOCKED — conservation infeasibility is structural even at R=5
  - **Implementation:** Add test `test_e4_1_conservation_feasible_r5` in `test_e4_cut_cell.py`, following the pattern from `TestApproachCEntryLevelUnknowns.test_rank_gap_constant_weights_8_betas` (lines 1331-1416) but using the standard alpha-parameterized pipeline at R=5 instead of the beta-parameterized one at R=4.
  - Ordering: must complete 23.2a-23.2c first
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::test_e4_1_conservation_feasible_r5 -v`
  - Also record decision **D23-1** in `plans/meta.md`: theta-linearization approach (same as `solve_conservation` in `conservation.py`, extended for the ψ-parameterized cut-cell case)

- [ ] **23.3b** Implement `solve_cut_cell_conservation` function:
  - **Function signature:** `solve_cut_cell_conservation(B_l: Matrix, R: int, T: int, p: int, nu: int, interior_coeffs: list, psi: Symbol, alpha_syms: list[Symbol]) -> dict[Symbol, Expr]`
  - **Algorithm** (mirrors `solve_conservation` from `conservation.py:88-167`, adapted for the ψ variable):
    1. Call `build_cut_cell_conservation_system(B_l, R, T, p, nu, interior_coeffs, psi)` → 6 equations, w_syms=[w_1..w_4]
    2. Identify bilinear products: for each equation, find all terms of the form `w_i * expr_involving_alpha_k` (i=1..4)
    3. Create θ_{i,k} symbols for each unique bilinear pair w_i × α_k
    4. Build substitution dict: `{w_i * α_k: θ_{i,k}}` and apply `expand().subs()` to each equation
    5. Clear ψ-denominators: for each linearized equation, `cancel()` → `fraction()` → take numerator
    6. Collect ψ-coefficients: `Poly(num, psi).all_coeffs()` → scalar equations (no ψ)
    7. Solve the scalar linear system: `linear_eq_to_matrix(scalar_eqs, [w_1..w_4] + [θ_...])` → `linsolve()`
    8. Recover α constraints: for each θ_{i,k}, α_k = θ_{i,k} / w_i; collect all α constraints and simplify
    9. Return solution dict mapping each w_i and constrained α_k to expressions in surviving free alphas
  - **Note on surviving alphas:** The exact number of free alphas after conservation depends on the rank of the scalar system. The plan estimates 3 surviving (5 - 2 excess), but this may vary. The function should detect the free parameters from the linsolve result.
  - **Invariant:** The function is only called for E4_1 (R=5); E2_1 and E2_2 conservation is handled by the existing uniform-boundary `solve_conservation` path.
  - Ordering: must complete 23.2a-23.2c first (needs working pipeline); should complete 23.3a first (feasibility confirmation — if rank gap ≠ 0, this function cannot produce a solution)
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (add after `build_cut_cell_conservation_system`, around line 1360)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "conservation and not fails"`

- [ ] **23.3c** Integrate conservation into `derive_cut_cell_scheme`:
  - In `derive_cut_cell_scheme` (temo.py line 2028), after `construct_cut_cell_stencil` produces `floating_result`:
    1. Call `solve_cut_cell_conservation(floating_result.matrix, ...)` → solution dict
    2. Substitute constrained alpha values into `floating_result.matrix`
    3. Update `alpha_symbols` list to contain only the surviving free alphas
    4. Pass the updated matrix and alpha list to `assemble_cut_cell_result`
  - Add `weights: dict[Symbol, Expr] | None` field to `CutCellResult` to carry the norm weight solution (needed for 23.4 C++ generation)
  - **Guard:** Only apply conservation when the scheme has a conservation solver (E4_1). For E2_1/E2_2, skip (conservation is handled in their uniform boundary derivation).
  - Ordering: must complete 23.3a and 23.3b first
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py::TestDeriveCutCellScheme -v`

- [ ] **23.3d** Verify conservation holds symbolically and update tests:
  - Update `test_e4_1_conservation_fails` (lines 804-838):
    - Remove the `@pytest.mark.xfail` marker
    - Use the conservative stencil from `derive_cut_cell_scheme(E4_1, psi)` instead of the raw `construct_cut_cell_stencil` output
    - R and column sum already updated to R=5 with row 4 in 23.2h; no further dimension changes needed
    - Use the solved weights (w_0=ψ, w_1..w_4 from the conservation solution) instead of assuming w_i=1
    - Verify `Σ_i w_i · B[i,j] + IC(j) = target(j)` as a polynomial identity in ψ and surviving alphas for ALL j=0..T-2
  - Update `TestBuildCutCellConservationSystem` tests (if needed) to reflect the conservative stencil
  - Ordering: must complete 23.3b and 23.3c first
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k conservation`

### 23.4 — Re-generate E4_1 C++ code

- [ ] **23.4a** Re-generate E4_1.cpp with correct dimensions and conservation:
  - P=2, R=5, T=7 (was R=4)
  - 5×7=35 floating coefficients (was 4×7=28)
  - 4×7=28 Dirichlet coefficients (was 3×7=21)
  - Alpha array size: number of surviving free alphas after conservation (estimated 3, verify from 23.3b result). If conservation leaves N free alphas → `std::array<real, N> alpha`.
  - Dirichlet info: {P, R-1=4, T, 0} (was R-1=3)
  - **Process:**
    1. Run the test suite which generates C++ to `scripts/stencil_gen/output/E4_1.cpp` and `E4_1.t.cpp` (via `TestE4CodeGeneration.test_write_output` and `TestE4TestFileGeneration.test_write_test_output`)
    2. Copy generated files: `cp scripts/stencil_gen/output/E4_1.cpp src/stencils/E4_1.cpp && cp scripts/stencil_gen/output/E4_1.t.cpp src/stencils/E4_1.t.cpp`
    3. Rebuild and test: `cmake --build build --target t-E4_1 && ctest --test-dir build -R t-E4_1`
  - **Note:** The `TestE4CodeGeneration` and `TestE4TestFileGeneration` tests (fixed in 23.2g) must produce correct C++ before this step.
  - Ordering: must complete 23.2g and 23.3c first (need both correct dimensions AND conservation)
  - File: `src/stencils/E4_1.cpp`, `src/stencils/E4_1.t.cpp`

### 23.5 — Update plans and decision records

- [ ] **23.5a** Update meta.md: revert D-R25, document the correct formula:
  - D-R25 was wrong — Eq. 11b gives `r = q + 1 + nextra`, not `r = p + 1 + nextra`
  - The formulas were only for 1st derivative operators; 2nd derivatives use different sizing
  - Add new decision D23-1 (conservation approach): theta-linearization with ψ-coefficient extraction (done as part of 23.3a, but ensure it's in meta.md)
  - File: `plans/meta.md`

- [ ] **23.5b** Update Phase 22 plan to document the resolution:
  - The infeasibility was caused by wrong dimensions, not a fundamental mathematical limitation
  - With correct R=5, conservation is feasible (verified in 23.3a)
  - File: `plans/22-cut-cell-conservation.md`

### 23.6 — Regression: verify E2_1 and E2_2 unchanged

- [ ] **23.6a** Verify all E2 tests still pass:
  - Run full test suite and confirm E2_1 and E2_2 tests are unaffected
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
