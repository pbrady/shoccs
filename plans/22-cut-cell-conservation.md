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
The current TEMO pipeline solves each row of the cut-cell stencil B_l(ψ) independently for Taylor accuracy, then assembles the result. It does NOT enforce the discrete conservation constraint `Σ_i w_i · B[i,j] = 0` for interior columns. This is fine for E2_1 (where the conservation system is exactly determined by weights alone), but fails for E4_1 (where 3 excess constraints must be absorbed by stencil entries).

### Why E2_1 works without explicit conservation enforcement
- E2_1: R=4, T=5, p=1, nextra=1
- Columns with nonzero IC (grid-frame): 3 (1 column; T-frame: 4)
- Conservation equations: T−1 = 4 (all columns j=0..T−2 per `conservation.py`)
- Unknowns: 3 weights (w_1, w_2, w_3) + 1 phi placeholder (nextra=1) = 4
- **Exactly determined** → weights + phi absorb all constraints, no alpha constraints needed

### Why E4_1 fails
- E4_1: R=4, T=7, p=2, nextra=0
- Columns with nonzero IC (grid-frame): 2, 3, 4, 5 (4 columns; T-frame: 3, 4, 5, 6)
- Conservation equations: T−1 = 6 (all columns j=0..T−2 per `conservation.py`)
- Weight unknowns (w_1, w_2, w_3): 3, nextra=0 → no phi placeholders
- **3 excess constraints** → must be satisfied by stencil entries (alpha parameters)
- Currently ignored → conservation violated on ALL columns

### Degrees of freedom budget
Each row has 7 columns, 1 prescribed (Category A zeroed column), 4 Taylor equations → **2 free entries per row**, 8 total across 4 rows.

After conservation:
- 3 weight unknowns absorb 3 of 6 conservation equations
- 3 excess equations constrain 3 of the 8 free entries
- 5 remaining free entries → these are the optimization parameters (map to 4 α^u plus the wall constraint absorbs 1)

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

- [ ] **22.1a** Add `test_e4_1_conservation_fails` to `test_e4_cut_cell.py`:
  - Construct the E4_1 cut-cell stencil using the current pipeline
  - Check `Σ_i w_i · B[i,j] = 0` for j = 2..6 (interior columns) with w_0=ψ, w_i=1
  - This test should FAIL (proving the bug exists)
  - Mark with `@pytest.mark.xfail(reason="conservation not yet enforced for E4_1")`
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k conservation`

### 22.2 — Build the coupled conservation system

- [ ] **22.2a** Implement `build_cut_cell_conservation_system()` in `temo.py`:
  - **Signature:** `build_cut_cell_conservation_system(B_l: Matrix, R: int, T: int, p: int, interior_coeffs: list, psi: Symbol) -> tuple[list[Expr], list[Symbol]]`
    - `B_l` is the R×T cut-cell stencil matrix (entries are rational in ψ and α symbols)
    - Returns `(equations, w_symbols)` where equations are expressions that must equal zero
  - **Interior column identification:** T-frame column j (0-indexed, col 0 = wall) corresponds to grid point j-1 (col 1 = x_0). Interior row R+m covers T-frame columns `(R+m-p+1)..(R+m+p+1)`. A column j has nonzero IC if it falls in range `[R-p+1, T-1]` (for E4_1: j = 3..6, 4 columns). Uses the same logic as `conservation.py:_interior_contribution()` adapted for T-frame indexing.
  - **Interior contribution IC(j):** Sum of interior stencil coefficients touching T-frame column j. Interior row R+m has coefficient `interior_coeffs[j-1 - (R+m) + p]` at column j. Equivalently: `_interior_contribution(j-1, R, p, interior_coeffs)` from `conservation.py` (adjusting for T-frame vs grid-frame). IC(j) = 0 for columns outside the overlap range.
  - **Conservation equations:** For each T-frame column j = 0..T−2 (matching `conservation.py` which iterates `range(t-1)`, giving T−1 total equations):
    `w_0 * B_l[0,j] + Σ_{i=1}^{R-1} w_i * B_l[i,j] + IC(j) = 0`
    where `w_0 = psi` (the cut-cell weight) and `w_1..w_{R-1}` are symbol unknowns.
  - **Wall column (j=0):** Use the standard `col_sum + 1 = 0` convention from `conservation.py` line 80 (column 0 sums to −1).
  - **Column 1 (x_0):** For nu=1, column 1 is the zeroed Category-A column. Its column sum involves `psi * B_l[0,1]` where `B_l[0,1]` is already prescribed. The equation for j=1 is generated like any other column; it may be trivially satisfied or impose a constraint on weights.
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (add after `construct_cut_cell_stencil`, around line 1282)
  - Verify: new function is importable and produces the correct number of equations

- [ ] **22.2b** Test conservation system dimensions:
  - For E2_1: call with E2_1's cut-cell stencil → expect T−1 = 4 equations (j = 0..3), 3 weight unknowns `w_1, w_2, w_3` (w_0=ψ is fixed) + 1 phi placeholder (nextra=1) = 4 unknowns → exactly determined
  - For E4_1: call with E4_1's cut-cell stencil → expect T−1 = 6 equations (j = 0..5), 3 weight unknowns `w_1, w_2, w_3` (w_0=ψ is fixed), nextra=0 → 3 excess constraints
  - The E4_1 system has 6 equations and only 3 weight unknowns → 3 excess constraints that must be absorbed by the 4 alpha parameters, confirming the problem
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k conservation_system`

### 22.3 — Integrate conservation into the TEMO solve

- [ ] **22.3a** Implement `enforce_cut_cell_conservation()` in `temo.py`:
  - **Signature:** `enforce_cut_cell_conservation(B_l: Matrix, R: int, T: int, p: int, interior_coeffs: list, psi: Symbol, alpha_symbols: list[Symbol]) -> tuple[Matrix, dict[Symbol, Expr], list[Symbol]]`
    - Returns `(B_l_conserved, weight_solutions, remaining_alphas)`
  - **Step 1 — Build conservation equations:** Call `build_cut_cell_conservation_system()` from 22.2a to get `(equations, w_symbols)`.
  - **Step 2 — Identify unknowns:** The equations are linear in `w_1..w_{R-1}` (weight unknowns) and the alpha symbols from B_u. Collect all unknowns. For E4_1: unknowns are `[w_1, w_2, w_3, alpha_0, alpha_1, alpha_2, alpha_3]` = 7 unknowns, 6 equations → 1 free parameter. But wait — alpha symbols appear in B_l entries which appear multiplied by w_i, creating bilinear terms `w_i * alpha_j`. This is the same issue `conservation.py:solve_conservation()` handles via the theta-linearization trick (lines 132-148). We may need to handle this similarly, or note that for nextra=0 cases (E4_1) where there are no phi placeholders, the bilinear terms are `w_i * alpha_j` products.
  - **Step 2a — Linearization strategy:** The conservation equations contain terms like `w_i * B_l[i,j]` where `B_l[i,j]` is linear in alpha symbols. This produces bilinear terms `w_i * alpha_k`. Approach: treat alpha symbols as **parameters** (not unknowns), solve the linear system in `w_1..w_{R-1}` only. If the system is overdetermined in the w's (E4_1 has 6 equations for 3 w's), use `linear_eq_to_matrix` + solve, which will yield constraints on the alpha symbols. Alternatively: substitute `theta_{i,k} = w_i * alpha_k` to linearize (following the theta trick in `conservation.py`).
  - **Step 2b — Practical approach for E4_1 (nextra=0):** Since the conservation equations are linear in `(w_1, w_2, w_3)` with coefficients that are rational in `(psi, alpha_0, ..., alpha_3)`, and we have 6 equations in 3 unknowns: pick 3 of the 6 equations to solve for `w_1, w_2, w_3` in terms of `(psi, alpha)`, then substitute into the remaining 3 equations. The remaining 3 equations become constraints on the alpha parameters that must hold as polynomial identities in psi. Solve these 3 equations for 3 of the 4 alphas, leaving 1 alpha free. This reduces E4_1 from 4 to 1 free alpha parameter. Alternatively, if the math in the problem analysis is correct (5 remaining free → 4 alpha^u), we need to re-examine which entries are truly free vs constrained.
  - **Key insight:** The conservation constraints reduce the alpha parameter count. After conservation, some alphas become functions of psi and the remaining free alphas. Substitute these back into B_l to get the conservation-enforced stencil.
  - **QQ(psi) arithmetic:** Since alpha symbols appear in the conservation equations, we cannot use `solve_in_field` (which operates purely in QQ(psi)). Instead, use SymPy's `linear_eq_to_matrix` + `linsolve` with the polynomial domain `ZZ(psi)` or just symbolic solve. Performance should still be acceptable since the system is small (≤6 equations).
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **22.3b** Integrate into `construct_cut_cell_stencil()`:
  - After assembling `matrix = Matrix(rows)` at line 1278, call `enforce_cut_cell_conservation()` if the scheme needs it
  - Add parameter `enforce_conservation: bool = True` to `construct_cut_cell_stencil()`
  - When `enforce_conservation=True` and the conservation system is overdetermined (more equations than weight unknowns), apply the enforcement
  - Update the returned `StencilResult` to reflect the reduced alpha count
  - Update `StencilResult` dataclass to include `weight_solutions: dict[Symbol, Expr] | None` (maps `w_i → expr(psi, alpha)`)
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (modify `construct_cut_cell_stencil` at lines 1206-1281)

- [ ] **22.3c** Handle the quadrature weight output:
  - The conservation solve produces ψ-dependent quadrature weights w_0=ψ, w_1(ψ,α), ..., w_{R-1}(ψ,α)
  - Store weights in `StencilResult.weight_solutions` (new field from 22.3b)
  - Add `weights: list | None` field to `CutCellResult` dataclass (line 1290)
  - In `assemble_cut_cell_result()` (line 1905), pass through weight solutions
  - For E2_1: verify weights match the known result (w_0=ψ, w_1=w_2=w_3=1 when the E2_1 conservation system is exactly determined)
  - **Note on C++ weights:** The current C++ stencil struct (`struct info` in `stencil.hpp`) has no weight storage — it only stores `{p, r, t, nextra}`. The C++ solver currently hardcodes `w_0=psi, w_i=1` for the norm. If conservation produces non-trivial weights (w_i ≠ 1), the C++ side will need a `norm_weights(psi)` method. But this is a **separate Phase 23+ concern** — for now, record the weights in the Python pipeline and verify them in tests.
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

### 22.4 — Validate conservation for E4_1

- [ ] **22.4a** Remove the `xfail` marker from `test_e4_1_conservation_fails`:
  - The test should now PASS with the conservation-enforced pipeline
  - Verify: `Σ_i w_i · B[i,j] = 0` as a polynomial identity in ψ and α for ALL interior columns
  - Verify symbolically (for all ψ and α) and numerically (at specific values)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

- [ ] **22.4b** Verify E4_1 free parameter count after conservation:
  - Count the alpha symbols remaining in B_l after conservation enforcement
  - Based on the degrees-of-freedom analysis: expect either 4 alphas with 3 constrained (→ 1 free) OR the problem analysis's "4 α^u" count if the wall constraint doesn't consume an alpha. The actual count must be determined during 22.3a implementation.
  - Document the actual free parameter count in this plan item once determined
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k free_param`

- [ ] **22.4c** Re-generate E4_1 C++ code with conservation-enforced stencil:
  - The generated code will produce different (larger) expressions than Phase 21's output
  - Update `StencilGenSpec` construction in `test_e4_cut_cell.py::TestE4CodeGeneration` to use the conservation-enforced pipeline (likely just changing to `derive_cut_cell_scheme`)
  - Run the codegen pipeline and write new E4_1.cpp
  - Verify it still compiles and passes structural checks
  - Update `param_arrays={"alpha": N}` where N is the actual free parameter count from 22.4b
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Depends on: 22.4b (need to know the free parameter count)

### 22.5 — Regression test: E2_1 conservation still holds

- [ ] **22.5a** Verify E2_1 is unchanged by the conservation enforcement:
  - Since E2_1's conservation is exactly determined by weights, the stencil entries should be identical before and after the fix
  - Run the pipeline for E2_1 and compare against existing test data
  - All existing E2_1 tests must still pass
  - File: `scripts/stencil_gen/tests/test_temo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v`

### 22.6 — Validate conservation for E2_2

- [ ] **22.6a** Add conservation verification for E2_2 (2nd derivative):
  - E2_2 uses different Design Principle variants (B^{d,1}/B^{d,2})
  - Check conservation column sums with appropriate norm weights
  - File: `scripts/stencil_gen/tests/test_temo.py`

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
**TBD:** Two-phase (Taylor first, then conservation substitution) vs. monolithic coupled system. The plan currently assumes two-phase based on the "Key Implementation Insight" section. To be confirmed during 22.3a implementation.

### DD22-2: Bilinear term handling
**TBD:** For E4_1 (nextra=0), the conservation equations contain bilinear terms `w_i * alpha_j`. Options:
- (a) Treat alphas as parameters, solve for w's → overdetermined, yields alpha constraints
- (b) Theta-linearization trick from `conservation.py`
- (c) Full symbolic solve with `linsolve` treating all as unknowns
To be decided during 22.3a based on which approach yields clean symbolic solutions.

### DD22-3: Alpha parameter reduction
**TBD:** Conservation enforcement will reduce E4_1's free alpha count from 4 to some smaller number. The exact count depends on how many conservation equations constrain alpha vs. weight parameters. This affects:
- The C++ constructor signature (`std::span<const real>` size)
- The `param_arrays={"alpha": N}` in codegen
- The `alpha_symbols` list in `UniformResult`
To be determined during 22.3a and recorded in 22.4b.

---

## Performance Considerations

The coupled system is larger but still tractable in QQ(ψ):
- E4_1: ~11 unknowns, ~22 equations → after row-by-row Taylor reduction, 6 conservation equations in ~6 unknowns (3 weights + 3 constrained free params)
- This is a linear system in QQ(ψ) with coefficients that depend on α
- The QQ(ψ) fraction field handles the ψ-rational arithmetic efficiently
- Target: full E4_1 derivation with conservation < 10 seconds

## Key Implementation Insight

The cleanest approach is probably:
1. **Keep the per-row Taylor solve** — each row's entries are functions of (ψ, row_free_params, α^u)
2. **Substitute into conservation equations** — conservation becomes a system in (weights, row_free_params)
3. **Solve the coupled conservation + weight system** — determine which row_free_params are constrained
4. **Substitute back** — replace constrained free params, leaving only α^u as optimization targets

This is a two-phase approach (Taylor first, then conservation) rather than solving one giant monolithic system. The Taylor phase produces the "shape" of each row; conservation then constrains the remaining degrees of freedom across rows.
