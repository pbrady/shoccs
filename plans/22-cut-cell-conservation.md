# Phase 22: Cut-Cell Discrete Conservation

**Goal:** Fix the TEMO pipeline to enforce discrete conservation (SBP property) for cut-cell stencils. Currently E4_1 violates conservation on all columns. The fix requires coupling the Taylor accuracy solve with the conservation column-sum constraints, creating a larger system that must be solved simultaneously in the QQ(ψ) fraction field.

**Depends on:** Phase 20 (pipeline), Phase 21 (E4_1 generation)

**Read first:**
- `plans/stencil-derivation-math-reference.md` (Section 4.4: Cut-Cell Conservation)
- `scripts/stencil_gen/stencil_gen/temo.py` (current TEMO pipeline — especially `construct_cut_cell_stencil`, `solve_temo_row`, `identify_prescribed_entries`)
- `scripts/stencil_gen/stencil_gen/conservation.py` (uniform conservation solver — reference for column-sum approach)
- `scripts/stencil_gen/tests/test_temo.py` (E2_1 conservation tests — `test_conservation_symbolic`, `test_conservation_numeric`)
- `src/stencils/E2_1.cpp` (known-correct conservative cut-cell stencil)

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/ -v
cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k conservation
cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k conservation
```

---

## Problem Analysis

### The bug
The current TEMO pipeline solves each row of the cut-cell stencil B_l(ψ) independently for Taylor accuracy, then assembles the result. It does NOT enforce the discrete conservation constraint `Σ_i w_i · B[i,j] = 0` for interior columns. This is fine for E2_1 (where the excess conservation residuals are trivially zero — the TEMO construction inherits conservation from the uniform boundary), but fails for E4_1 (where 3 excess constraints must be absorbed by stencil entries).

### Why E2_1 works without explicit conservation enforcement
- E2_1: R=4, T=5, p=1, nextra=1
- Interior contribution IC uses `r = R = 4` (first interior grid point after the R-row boundary block). All IC values within the conservation range (j=0..T−2=3) are zero — no interior row (starting at grid point 4) reaches these T-frame columns with p=1.
- Conservation equations: T−1 = 4 (all columns j=0..T−2 per `conservation.py`)
- Weight unknowns: 3 (w_1, w_2, w_3). The phi placeholders from `nextra=1` are resolved during `derive_uniform_boundary_for_temo` (line 388–395), NOT during cut-cell conservation. By the time the conservation system is built, B_l has no phi symbols — only (ψ, alpha_0..alpha_3).
- System is **overdetermined by 1** (4 equations, 3 weight unknowns, excess = q = 1). **CORRECTION**: Investigation in 22.3a showed that the excess residual is NOT identically zero. E2_1's conservation system (all T-1=4 column equations) is inconsistent with w_i=1 for columns 0,1,2. The existing passing tests only check columns 3,4 where IC=0 and column sums are trivially zero. Full conservation for E2_1 also requires alpha constraints and non-trivial ψ-dependent weights. Additionally, rows 1 and 2 of B_l are identical when α=0, making the system rank-deficient at that point.

### Why E4_1 fails
- E4_1: R=4, T=7, p=2, nextra=0
- Interior contribution IC uses `r = R = 4`. Nonzero IC at T-frame columns 3, 4, 5 (grid-frame 2, 3, 4) — 3 columns. Computed values: IC(3)=1/12, IC(4)=−7/12, IC(5)=−7/12. Columns 0, 1, 2 have IC=0 (interior rows at grid point ≥4 don't reach these columns with p=2).
- Conservation equations: T−1 = 6 (all columns j=0..T−2 per `conservation.py`)
- Weight unknowns (w_1, w_2, w_3): 3, nextra=0 → no phi placeholders
- **3 excess constraints** → must be satisfied by stencil entries (alpha parameters)
- Currently ignored → conservation violated on ALL columns with nonzero IC

### Degrees of freedom budget
Each row has 7 columns, 1 prescribed (Category A zeroed column), 4 Taylor equations → **2 free entries per row**, 8 total across 4 rows.

After conservation (6 equations, 3 weight unknowns + entries from 4 alpha symbols):
- 3 weight unknowns absorb 3 of 6 conservation equations
- 3 excess equations constrain 3 of the alpha parameters
- Expected: 4 − 3 = **1 remaining free alpha** for optimization (but actual count depends on rank; see DD22-3)

### The fix: coupled Taylor + conservation solve
Instead of solving each row independently, solve a single coupled system:
- **Taylor accuracy**: 4 rows × 4 equations = 16 equations (but some entries are prescribed, reducing unknowns)
- **Conservation**: 6 equations (5 column sums + 1 wall constraint)
- **Unknowns**: 8 free stencil entries + 3 weights = 11 unknowns
- **System**: 22 equations, 11 unknowns → overdetermined by 11, but 16 Taylor equations determine 8 stencil entries in terms of 8 free params, leaving 8 free params + 3 weights = 11 unknowns for the 6 conservation equations → still need 5 to remain free (the α^u optimization params)

The practical approach: solve Taylor per-row first (as now) to get stencil entries as functions of the free parameters, then substitute into the conservation equations and solve for the weights and constrained free parameters.

---

## Items

### 22.1 — Add conservation verification test that exposes the bug

- [x] **22.1a** Add `test_e4_1_conservation_fails` to `test_e4_cut_cell.py`:
  - Construct the E4_1 cut-cell stencil using the current pipeline
  - For each T-frame column j in 0..T−2 (j=0..5), compute the full conservation sum: `w_0 * B[0,j] + w_1 * B[1,j] + w_2 * B[2,j] + w_3 * B[3,j] + IC(j)` where w_0=ψ, w_i=1 for i≥1, and `IC(j) = _interior_contribution(j-1, R=4, p=2, interior)`. For j=0: add +1 (wall target = −1). Assert each column sum simplifies to 0.
  - Import `_interior_contribution` from `stencil_gen.conservation`
  - This test should FAIL for at least columns j=3,4,5 (where IC is nonzero and the boundary block doesn't compensate)
  - Mark with `@pytest.mark.xfail(reason="conservation not yet enforced for E4_1")`
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k conservation`

### 22.2 — Build the coupled conservation system

- [x] **22.2a** Implement `build_cut_cell_conservation_system()` in `temo.py`:
  - **Signature:** `build_cut_cell_conservation_system(B_l: Matrix, R: int, T: int, p: int, nu: int, interior_coeffs: list, psi: Symbol) -> tuple[list[Expr], list[Symbol]]`
    - `B_l` is the R×T cut-cell stencil matrix (entries are rational in ψ and α symbols)
    - `nu` is the derivative order (1 or 2) — determines wall column target (DD22-4)
    - Returns `(equations, w_symbols)` where equations are expressions that must equal zero
  - **Interior column identification:** T-frame column j (0-indexed, col 0 = wall) corresponds to grid point j−1. The boundary block has R rows covering grid points 0..R−1. Interior rows (using the unmodified interior stencil with unit weight) start at grid point R. Interior row at grid point R+m (m ≥ 0) covers grid-frame columns (R+m−p)..(R+m+p), i.e., T-frame columns (R+m−p+1)..(R+m+p+1). For E4_1 (R=4, p=2): nonzero IC at T-frame columns 3, 4, 5; IC(3)=1/12, IC(4)=−7/12, IC(5)=−7/12. T-frame columns 0, 1, 2 have IC=0.
  - **Interior contribution IC(j):** Import `_interior_contribution` from `conservation.py`. Compute `IC(j) = _interior_contribution(j-1, R, p, interior_coeffs)` where `j-1` converts T-frame to grid-frame, and `R` is the first interior grid point. **Critical:** use `r = R` (NOT `r = R-1`). The boundary block's R rows cover grid points 0..R−1; the near-interior row (R−1) IS part of the boundary block, so it must NOT be counted again in IC. This differs from the uniform case where the boundary has r_eff rows and interior starts at grid point r_eff. In the cut-cell case, R = r_eff + 1, so interior starts at R = r_eff + 1.
  - **Conservation equations:** For each T-frame column j = 0..T−2 (T−1 total equations):
    `w_0 * B_l[0,j] + Σ_{i=1}^{R-1} w_i * B_l[i,j] + IC(j) = target(j)`
    where `w_0 = psi` (fixed), `w_1..w_{R-1}` are symbol unknowns.
    Target: `target(0) = -1` for nu=1 (SBP boundary term), `target(j) = 0` for j ≥ 1.
    For nu=2: `target(j) = 0` for ALL j (constant annihilation: HD₂·1=0). See DD22-4.
    Equation form: `col_sum - target(j) = 0`, i.e., for j=0 with nu=1: `col_sum + 1 = 0`.
  - **Wall column (j=0) for nu=1:** `col_sum + 1 = 0` (column 0 sums to −1, matching `conservation.py` line 80).
  - **Wall column (j=0) for nu=2:** `col_sum = 0` (all columns sum to 0, per DD22-4).
  - **Column 1 (x_0):** For nu=1, column 1 is the zeroed Category-A column. Its equation is generated like any other column; it may be trivially satisfied or impose a constraint on weights.
  - **Imports needed:** `from stencil_gen.conservation import _interior_contribution` (reuse existing function, do NOT reimplement).
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (add after `construct_cut_cell_stencil`, around line 1282)
  - Verify: new function is importable and produces the correct number of equations (T−1 = 4 for E2_1, 6 for E4_1)

- [x] **22.2b** Test conservation system dimensions and IC values:
  - For E2_1: call with E2_1's cut-cell stencil → expect T−1 = 4 equations (j = 0..3), 3 weight unknowns `w_1, w_2, w_3` (w_0=ψ is fixed). System is overdetermined by 1 (excess = q = 1). All IC values should be 0 (no interior row at grid point ≥4 reaches T-frame columns 0..3 with p=1). Note: E2_1 has 4 alpha parameters but the 1 excess residual should be identically zero — verify this in 22.3a tests.
  - For E4_1: call with E4_1's cut-cell stencil → expect T−1 = 6 equations (j = 0..5), 3 weight unknowns `w_1, w_2, w_3` (w_0=ψ is fixed), nextra=0 → 3 excess constraints. Verify IC values: IC(0)=IC(1)=IC(2)=0, IC(3)=1/12, IC(4)=−7/12, IC(5)=−7/12.
  - The E4_1 system has 6 equations and only 3 weight unknowns → 3 excess constraints that must be absorbed by the 4 alpha parameters, confirming the problem
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k conservation_system`

### 22.3 — Integrate conservation into the TEMO solve

#### Investigation Findings (from 22.3a attempt)

The two-phase approach described below was attempted and found to have a **fundamental mathematical flaw**: the conservation system is bilinear in (weights × alpha), and the two-phase solve (weights first, alpha constraints from residuals) encounters a singularity at the constrained alpha values. Specifically:

1. **Step 3 weight solve** produces w_i as rational functions of (ψ, α) with denominators that depend on α.
2. **Steps 4-5** correctly derive α constraints by setting excess residual numerators to zero (for E4_1: α₀=11/6, α₁=1/3, α₂=−4α₃−1/6).
3. **Step 6 fails**: at the constrained α values, the weight denominators are identically zero (0/0). Re-solving the constrained system gives rank([A|b]) > rank(A) — the system is **inconsistent**.
4. **Root cause**: The 4×4 minor determinants of the augmented matrix [A|b] (the true consistency conditions) give **nonlinear** equations in α (containing α₁·α₃ cross-terms). These 67 psi-coefficient equations from 15 minors have no common solution satisfying all of them — the solution from `solve` satisfies only 46/67 equations.
5. **E2_1 also affected**: Even E2_1's conservation system (T−1=4 equations, 3 weight unknowns) is inconsistent with w_i=1 for columns 0,1,2. The existing passing tests only check columns 3,4 (where IC=0 and the column sums are trivially zero). Full conservation requires non-trivial ψ-dependent weights AND alpha constraints simultaneously.
6. **Key structural issue**: For E2_1, rows 1 and 2 of B_l are identical when all α=0, making the conservation matrix rank-deficient. Non-zero α values break this degeneracy.

**The problem requires a different approach.** The conservation system couples weights and alpha through bilinear terms (w_i × B_l[i,j](α)). Solving must handle both simultaneously, not sequentially. Possible approaches:
- **(A) Augmented consistency**: formulate conservation as requiring all 4×4 minors of [A(α)|b(α)] to vanish for all ψ; solve the resulting nonlinear α-system, then recover weights.
- **(B) Direct parametric solve**: parameterize w_i as rational functions of ψ with unknown coefficients, substitute into conservation, match ψ-coefficients, solve the resulting polynomial system in (weight coefficients, α).
- **(C) Entry-level unknowns**: instead of working with α, treat the 8 free stencil entries directly as unknowns alongside weights, giving a system that may be solvable despite higher dimensionality.

**Next step**: Implement sub-items 22.3a-i through 22.3a-iii below as a prerequisite investigation to select and validate the correct approach before full implementation.

- [ ] **22.3a-i** Investigate approach (A) — augmented matrix minor conditions:
  - Compute all C(6,4)=15 minor determinants of [A(ψ,α)|b(ψ,α)] for E4_1 symbolically (all 4 alpha free, not fixing alpha_3=0).
  - Extract ψ-coefficient equations from each minor.
  - Determine if the resulting α-system has any solution (check consistency).
  - If consistent, solve for α and verify the conservation system becomes consistent with those α values.
  - **Key question to answer**: does a valid (α₀,α₁,α₂,α₃) exist such that the conservation system A(ψ,α)w=b(ψ,α) is consistent for all ψ?
  - File: exploratory script or test

- [ ] **22.3a-ii** Investigate approach (B) — parametric weight functions:
  - Parameterize weights as w_i = p_i(ψ)/q(ψ) where q is the common TEMO denominator (ψ+1)(ψ+2)(ψ+3) and p_i are polynomials in ψ of degree ≤ 3 with unknown rational coefficients.
  - Substitute into all 6 conservation equations, clear denominators.
  - Collect all ψ-coefficient equations.
  - Determine if the resulting system (in weight coefficients and alpha) is solvable.
  - File: exploratory script or test

- [ ] **22.3a-iii** Choose approach and implement `enforce_cut_cell_conservation()`:
  - Based on findings from 22.3a-i and 22.3a-ii, implement the working approach.
  - Must pass verification: for the conserved stencil, all T−1 conservation column sums must be zero (symbolically, for all ψ and remaining free α).
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **22.3b** Integrate into `construct_cut_cell_stencil()` and propagate through pipeline:
  - **`StencilResult` dataclass (line 843):** Add field `weight_solutions: dict | None = None` (maps `w_i → expr(psi, alpha)`) and `alpha_symbols: list | None = None` (the remaining free alphas after conservation). The dataclass is not frozen, so new Optional fields with defaults can be appended without breaking existing callers.
  - **`construct_cut_cell_stencil` (lines 1206-1281):** Add parameter `enforce_conservation: bool = True` (note: `nu` is already a parameter). After assembling `matrix = Matrix(rows)` at line 1278 and before the return at line 1279:
    - Call `enforce_cut_cell_conservation(matrix, R, T, p, nu, interior, psi, alpha_syms)`
    - Replace `matrix` with the conserved version, store weight solutions and remaining alphas in the returned `StencilResult`
    - When `enforce_conservation=False`, skip enforcement and return as before
  - **`derive_cut_cell_scheme` (line 1950):** `scheme.nu` is already passed to `construct_cut_cell_stencil` at line 1981. Propagate `floating_result.weight_solutions` and `floating_result.alpha_symbols` to `assemble_cut_cell_result`. Currently calls `assemble_cut_cell_result(floating_result.matrix, ..., dims, uniform.alpha_symbols)` at line 1995 — change `uniform.alpha_symbols` to `floating_result.alpha_symbols if floating_result.alpha_symbols is not None else uniform.alpha_symbols`. **Do NOT use `or`** — an empty list is a valid conservation result (all alphas constrained) and `[] or [alpha_0, ...]` would incorrectly fall through to the original list. Also pass `floating_result.weight_solutions` to `assemble_cut_cell_result`.
  - **`assemble_cut_cell_result` (line 1905):** Add `weight_solutions: dict | None = None` parameter. Pass through to `CutCellResult`.
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (4 modification points: `StencilResult` at line 843, `construct_cut_cell_stencil` at line 1278, `assemble_cut_cell_result` at line 1905, `derive_cut_cell_scheme` at line 1995)
  - **Co-implement with 22.3c** — the `assemble_cut_cell_result` signature change (adding `weight_solutions`) requires the `CutCellResult` field from 22.3c to exist. Do both items in the same work pass.
  - **Temporarily broken tests:** After this item, `construct_cut_cell_stencil` enforces conservation by default, which changes the E4_1 stencil. The following tests in `test_e4_cut_cell.py` will fail until fixed by 22.4b/22.4c/22.4d: `test_e4_1_alpha_count` (line 659), `test_e4_1_matches_manual_pipeline` (line 665), `test_e4_1_custom_alphas` (line 707), `TestE4CodeGeneration.e4_spec` fixture (line 323), `TestE4TestFileGeneration.e4_spec` fixture (line 461). E2_1 and E2_2 tests are unaffected (their conservation residuals are trivially zero, stencils unchanged).

- [ ] **22.3c** Handle the quadrature weight output:
  - The conservation solve produces ψ-dependent quadrature weights w_0=ψ, w_1(ψ,α), ..., w_{R-1}(ψ,α)
  - **`CutCellResult` dataclass (line 1290):** Add `weight_solutions: dict | None = None` field (maps `w_i → expr(psi, alpha_remaining)`).
  - **`assemble_cut_cell_result()` (line 1905):** Accept `weight_solutions` parameter (added in 22.3b) and store in `CutCellResult.weight_solutions`.
  - **Test:** For E2_1, verify weight_solutions gives w_0=ψ, w_1=w_2=w_3=1 (the exactly-determined case produces trivial weights). For E4_1, verify weights are rational functions of (ψ, α_remaining).
  - **Note on C++ weights:** The current C++ stencil struct (`struct info` in `stencil.hpp`) has no weight storage — it only stores `{p, r, t, nextra}`. The C++ solver currently hardcodes `w_0=psi, w_i=1` for the norm. If conservation produces non-trivial weights (w_i ≠ 1), the C++ side will need a `norm_weights(psi)` method. But this is a **separate Phase 23+ concern** — for now, record the weights in the Python pipeline and verify them in tests.
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

### 22.4 — Validate conservation for E4_1

- [ ] **22.4a** Remove the `xfail` marker from `test_e4_1_conservation_fails` and adapt for solved weights:
  - The test should now PASS with the conservation-enforced pipeline
  - **If weights are trivial** (w_i = 1 for all i ≥ 1 — check `CutCellResult.weight_solutions`): remove the xfail marker, update the test to use `derive_cut_cell_scheme(E4_1, psi)` instead of the manual pipeline, and the naive weights (psi, 1, 1, 1) still work.
  - **If weights are non-trivial** (some w_i ≠ 1): remove the xfail marker AND rewrite the column-sum check to use the conservation-solved weights from `result.weight_solutions`. The conservation sum becomes: `w_0 · B[0,j] + Σ_{i≥1} w_i(ψ,α) · B[i,j] + IC(j) = target(j)` where each w_i is retrieved from `weight_solutions`. Use `cancel()` to verify each column sum equals target symbolically.
  - **In both cases:** verify conservation symbolically (for all ψ and remaining α) by checking `cancel(col_sum - target) == 0`. Also verify numerically at 2-3 specific (ψ, α) values as a cross-check.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

- [ ] **22.4b** Verify E4_1 free parameter count after conservation:
  - Count the alpha symbols remaining in B_l after conservation enforcement
  - Based on the degrees-of-freedom analysis: expect either 4 alphas with 3 constrained (→ 1 free) OR the problem analysis's "4 α^u" count if the wall constraint doesn't consume an alpha. The actual count must be determined during 22.3a implementation.
  - Document the actual free parameter count in this plan item once determined
  - **Update existing test:** `TestDeriveCutCellScheme::test_e4_1_alpha_count` (line 659 of `test_e4_cut_cell.py`) currently asserts `len(result.alpha_symbols) == 4`. After conservation enforcement reduces the alpha count, this test will fail. Update it to assert the correct post-conservation count (1 if 3 constrained, or the actual count from 22.3a).
  - Add a new `test_e4_1_free_param_count` test that explicitly verifies: (a) the number of free alpha symbols in `result.alpha_symbols`, (b) that `result.floating.free_symbols` contains exactly `{psi} | set(result.alpha_symbols)`, and (c) the constrained alphas no longer appear.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k free_param`

- [ ] **22.4c** Re-generate E4_1 C++ code with conservation-enforced stencil:
  - The generated code will produce different (larger) expressions than Phase 21's output
  - **Update two fixtures:** Both `TestE4CodeGeneration.e4_spec` (line 323) and `TestE4TestFileGeneration.e4_spec` (line 461) currently call `construct_cut_cell_stencil` + `assemble_cut_cell_result` manually with `ur.alpha_symbols`. Replace both with `result = derive_cut_cell_scheme(E4_1, psi)`, which includes conservation enforcement. Extract `floating_flat = list(result.floating)`, `dirichlet_flat = [Integer(0)] * result.dims.T + list(result.dirichlet)`, and use `result.alpha_symbols` for the `StencilGenSpec`.
  - **Update `param_arrays`:** Change `param_arrays={"alpha": 4}` to `param_arrays={"alpha": N}` where N = `len(result.alpha_symbols)` (the post-conservation free alpha count from 22.4b).
  - **Update assertions that depend on alpha count:**
    - `test_alpha_array` (line 377): currently asserts `std::array<real, 4> alpha;` — update `4` to N.
    - `TestE4TestFileGeneration.ALPHA_VALUES` (line 459): currently `{"alpha": [0.1, -0.05, 0.02, 0.01]}` (4 values) — reduce to N values.
  - **Regenerate output:** The `test_write_output` test (line 446) writes to `scripts/stencil_gen/output/E4_1.cpp`. The new file will have different coefficient expressions with fewer alpha parameters.
  - Verify: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "TestE4CodeGeneration or TestE4TestFileGeneration"`
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Depends on: 22.3b (conservation enforcement in pipeline), 22.4b (need to know the free parameter count)

- [ ] **22.4d** Update `test_e4_1_matches_manual_pipeline` and `test_e4_1_custom_alphas` for conservation:
  - **`test_e4_1_matches_manual_pipeline` (line 665):** This test compares `derive_cut_cell_scheme(E4_1, psi)` against the manual pipeline (`construct_cut_cell_stencil` + `assemble_cut_cell_result`). After 22.3b, both paths enforce conservation by default, so the comparison should still hold. However, the manual path at line 675 passes `ur.alpha_symbols` (the full 4 alphas) to `assemble_cut_cell_result`, while the auto path propagates the conservation-reduced alpha list. Fix: change the manual path to use `stencil.alpha_symbols` (the conservation-reduced list from `StencilResult`) instead of `ur.alpha_symbols`. Also add an assertion that `auto.alpha_symbols == manual.alpha_symbols` (both should have the reduced count).
  - **`test_e4_1_custom_alphas` (line 707):** Currently creates 4 custom alpha symbols and asserts `result.alpha_symbols == syms` and `result.floating.free_symbols <= {psi} | set(syms)`. After conservation: `result.alpha_symbols` is a subset of `syms` (the remaining free alphas). Fix: change the assertion to `set(result.alpha_symbols) <= set(syms)` and `len(result.alpha_symbols) == N` (post-conservation count from 22.4b). Also update `result.floating.free_symbols <= {psi} | set(result.alpha_symbols)` (use the result's own alpha list).
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Depends on: 22.3b, 22.4b (need to know the free parameter count)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "matches_manual or custom_alphas"`

### 22.5 — Regression test: E2_1 conservation still holds

- [ ] **22.5a** Verify E2_1 is unchanged by the conservation enforcement:
  - Since E2_1's excess conservation residuals are trivially zero (no alpha constraints needed), the stencil entries should be identical before and after the fix.
  - **Existing tests that must still pass (no modifications needed):**
    - `test_temo.py::TestE2_1Integration` — all tests including `test_conservation_symbolic` (line 1596) and `test_conservation_numeric` (line 1535). These use `construct_cut_cell_stencil` which now enforces conservation by default, but E2_1's stencil is unchanged by conservation.
    - `test_e4_cut_cell.py::TestDeriveCutCellScheme::test_e2_1_reproduces_existing` (line 715) — compares `derive_cut_cell_scheme(E2_1, psi)` against manual pipeline. Both paths now enforce conservation; E2_1 is unchanged by conservation so the comparison holds. The manual path at line 721 calls `construct_cut_cell_stencil` (which now enforces conservation by default) and passes `ur.alpha_symbols` to `assemble_cut_cell_result`. Since E2_1's `StencilResult.alpha_symbols` equals `ur.alpha_symbols` (no alphas constrained), the manual path remains consistent.
  - **New test to add:** `test_e2_1_weight_solutions` in `test_temo.py::TestE2_1Integration`:
    - Call `construct_cut_cell_stencil(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1, psi)` and verify `result.weight_solutions` is populated.
    - Verify weights: `w_0 = psi` (fixed, not in weight_solutions), `w_1 = 1, w_2 = 1, w_3 = 1` — all non-trivial weights are 1 (the conservation solve for E2_1 produces unit weights because the boundary already satisfies conservation).
    - Verify `result.alpha_symbols` contains all 4 original alpha symbols (none constrained).
    - File: `scripts/stencil_gen/tests/test_temo.py`
  - Test commands:
    - `cd scripts/stencil_gen && uv run pytest tests/test_temo.py::TestE2_1Integration -v`
    - `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k e2_1`

### 22.6 — Validate conservation for E2_2

- [ ] **22.6a** Add conservation verification for E2_2 (2nd derivative):
  - **Dimensions:** E2_2 has p=1, q=1, nextra=0, nu=2. Computed: r=2, t=3, r_eff=1, R=2, T=4.
  - **Conservation system:** T−1 = 3 equations (j=0..2 in T-frame). Weight unknowns: w₁ only (w₀=ψ fixed, R−1=1 unknown). System is overdetermined: 3 equations, 1 weight unknown → 2 excess constraints on alphas. But E2_2 has **0 alpha symbols** — the existing E2_2 stencil is fully determined by the Taylor solve. The 2 excess constraints must be implicitly satisfied.
  - **IC values:** Using `_interior_contribution(j-1, R, p, interior)` where R=2 (first interior grid point after the R-row boundary block). For E2_2 (R=2, p=1, interior=[1, -2, 1]): IC(0) = _ic(-1, 2, 1, ...) = 0; IC(1) = _ic(0, 2, 1, ...) = 0 (m_hi = 0-2+1 = -1, empty); IC(2) = _ic(1, 2, 1, ...) = interior[1-2+1] = interior[0] = 1. So IC(0)=0, IC(1)=0, IC(2)=1.
  - **Wall column convention for nu=2:** Per DD22-4, for the 2nd derivative ALL column targets are 0 (including column 0). This differs from nu=1 where column 0 targets −1. The `build_cut_cell_conservation_system` function (22.2a) already parameterizes this by `nu`. Verify that the E2_2 conservation equations use `target(j) = 0` for all j.
  - **Expected behavior with `enforce_cut_cell_conservation`:** E2_2 exercises the "no-alpha overdetermined" edge case from 22.3a. The weight solve (Step 3) determines w₁ as a rational function of ψ. The 2 excess residuals (Step 4) must be identically zero, confirming that the Taylor solve implicitly satisfies conservation for E2_2. The stencil matrix should be UNCHANGED from the pre-conservation version. Test this by comparing entry-by-entry: `derive_cut_cell_scheme(E2_2, psi)` vs. the manual pipeline (identical to existing `test_e2_2_reproduces_existing` at line 745 of `test_e4_cut_cell.py`). Verify `result.weight_solutions` maps w₁ to its solved value (a function of ψ only, no alphas).
  - **Test structure:** Build the E2_2 cut-cell conservation system using `build_cut_cell_conservation_system()` from 22.2a. Verify equation count (3) and that the system is solvable. Verify that the 2 excess residuals are zero. Verify that the conservation-enforced stencil is identical to the non-enforced version.
  - **Polynomial exactness cross-check:** Verify that the conservation-enforced E2_2 stencil still satisfies `f(x)=x² → f''=2` (matching the existing `test_conservation_polynomial_exactness` test at line 1772 of `test_temo.py`).
  - File: `scripts/stencil_gen/tests/test_temo.py`
  - Depends on: 22.2a, 22.3a-22.3b (uses `enforce_cut_cell_conservation` via `derive_cut_cell_scheme`)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k e2_2_conservation`

### 22.7 — Update codegen for quadrature weights

- [ ] **22.7a** Determine if C++ weight output is needed in this phase:
  - The current C++ stencils (E2_1.cpp) have **no** weight storage — the `struct info` in `stencil.hpp` only stores `{p, r, t, nextra}` and there is no `w` array, weight method, or quadrature concept anywhere in `src/stencils/`
  - The C++ solver implicitly assumes `w_0=psi, w_i=1` for the SBP norm
  - If E4_1 conservation produces weights where `w_i = 1` for all i≥1 (which is possible if the conservation system only constrains alpha parameters, not weights), then NO C++ changes are needed
  - If weights are non-trivial (w_i ≠ 1), defer the C++ weight infrastructure to a separate phase. The Python pipeline records the weights (22.3c) and tests verify them, but C++ codegen for weights is out of scope for Phase 22.
  - **Decision needed:** After 22.3a implementation, check whether E4_1 weights are trivial. Record the decision in this item.
  - File: `scripts/stencil_gen/stencil_gen/codegen.py` (only if non-trivial weights found)

---

## Design Decisions (to be recorded in plans/meta.md if cross-cutting)

### DD22-1: Conservation enforcement approach
**ORIGINALLY CHOSEN: Two-phase** (Taylor first, then conservation substitution). **STATUS: FAILED** — the two-phase approach encounters a bilinear singularity where the weight denominators vanish at the constrained alpha values, making the system inconsistent after alpha substitution. See "Investigation Findings" in §22.3 for details. A new approach must be selected from the alternatives described in 22.3a-i/ii/iii.

### DD22-2: Bilinear term handling
**ORIGINALLY CHOSEN: (a) Treat alphas as parameters, solve for w's.** **STATUS: INSUFFICIENT** — while the conservation equations are linear in w_i when α symbols are parameters, the resulting parametric weight solution has denominators that depend on α. At the α values required by conservation, these denominators vanish (the coefficient matrix A(α) loses rank). The bilinear coupling MUST be handled simultaneously. The augmented matrix minor approach (all 4×4 minors of [A|b] must vanish) gives the true consistency conditions but produces nonlinear equations in α.

### DD22-3: Alpha parameter reduction
**TBD (narrowed):** Conservation enforcement will reduce E4_1's free alpha count from 4 to a smaller number. Based on the DOF analysis:
- 6 conservation equations in 3 weight unknowns → 3 excess equations constraining alphas
- 4 alpha symbols − 3 constraints → **expected: 1 free alpha**
- But: the 3 residual equations from Step 4 of 22.3a may not all be independent (some may be redundant), so the actual free count could be 1–4. Must be determined empirically during 22.3a and recorded in 22.4b.
- This affects the C++ constructor signature (`std::span<const real>` size), the `param_arrays={"alpha": N}` in codegen, and the `alpha_symbols` list in `CutCellResult`.

### DD22-4: Wall column conservation target by derivative order
**RESOLVED:** The SBP conservation target for column 0 (the wall/boundary column) depends on derivative order:
- **nu=1 (1st derivative):** Target = −1. From the SBP property Q + Qᵀ = B where B = diag(−1, 0, …, 0, 1). The norm-weighted column 0 sum must equal −1. Conservation equation: `col_sum + 1 = 0`.
- **nu=2 (2nd derivative):** Target = 0. From constant annihilation: HD₂·𝟏 = 0, meaning ALL norm-weighted column sums equal zero, including column 0. Conservation equation: `col_sum = 0`.
- The existing `conservation.py` only handles nu=1 cases (E2u_1, E4u_1) and hardcodes `col_sum + 1 = 0` for j=0. The new `build_cut_cell_conservation_system` (22.2a) parameterizes by `nu`.
- **Note:** `conservation.py`'s `build_conservation_system` should eventually be updated to accept `nu` as well, but that is out of scope for Phase 22 (uniform conservation is only used for nu=1 currently).

---

## Performance Considerations

The conservation solve is small but mathematically nontrivial:
- E4_1: 6 conservation equations, bilinear in 3 weight unknowns × 4 alpha symbols
- The augmented matrix approach produces 15 minor determinants, each polynomial in (ψ, α). After extracting ψ-coefficients: 67 nonlinear equations in α (containing cross-terms like α₁·α₃)
- The parametric weight approach may require parameterizing w_i as rational functions of ψ with ~12 unknown coefficients
- SymPy's `solve()` returns 0 solutions for the full bilinear system; `nsolve()` reports singular Jacobian
- Target: TBD after selecting approach in 22.3a-i/ii/iii

## Key Implementation Insight

The original two-phase approach (Taylor → conservation) fails due to bilinear singularities. The correct approach must handle the coupling between weights and alpha simultaneously. The conservation system A(ψ,α)·w = b(ψ,α) requires the augmented matrix [A|b] to have rank ≤ rank(A) for ALL ψ, which gives polynomial conditions on α (from minor determinants). These conditions are nonlinear because the A matrix entries involve α linearly, making the 4×4 minors quadratic or higher in α.
