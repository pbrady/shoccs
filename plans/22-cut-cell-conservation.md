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
- Interior overlap columns: 3, 4 (2 columns)
- Conservation equations: 2 column sums + 1 wall constraint = 3
- Weight unknowns (w_1, w_2, w_3): 3
- **Exactly determined** → weights alone absorb all constraints, no stencil entry constraints needed

### Why E4_1 fails
- E4_1: R=4, T=7, p=2, nextra=0
- Interior overlap columns: 2, 3, 4, 5, 6 (5 columns)
- Conservation equations: 5 column sums + 1 wall constraint = 6
- Weight unknowns (w_1, w_2, w_3): 3
- **3 excess constraints** → must be satisfied by stencil entries
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

- [ ] **22.2a** Implement `build_cut_cell_conservation_system(stencil_matrix, interior_coeffs, p, R, T, psi)` in `temo.py`:
  - Identify interior columns: j where the interior stencil (half-width p) overlaps with the boundary block. For R rows and p half-width, these are columns j = R-p through T-1 in the cut-cell frame.
  - For each interior column j, write the constraint: `psi * B[0,j] + Σ_{i=1}^{R-1} w_i * B[i,j] + IC(j) = 0` where IC(j) is the contribution from the interior rows' stencil overlap at column j.
  - Compute IC(j) using the interior stencil coefficients: IC(j) = Σ_{k} γ_{j-R-k} for the appropriate interior rows that touch column j.
  - The wall-column constraint: `w_0 * B[0,0] = Σ_i w^u_i * B^u[i,0]` (Eq. from Section 3.3)
  - Return: list of constraint equations in (w_1, ..., w_{R-1}) and the stencil free parameters
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **22.2b** Test conservation system dimensions:
  - For E2_1: 3 equations, 3 weight unknowns → exactly determined
  - For E4_1: 6 equations, 3 weight unknowns + stencil free params → overdetermined in weights
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 22.3 — Integrate conservation into the TEMO solve

- [ ] **22.3a** Modify `construct_cut_cell_stencil` to accept conservation enforcement:
  - After the per-row Taylor solve (which gives entries as functions of free parameters), substitute into the conservation equations
  - The conservation equations become polynomial identities in (ψ, α^u, free_params)
  - Solve for as many free parameters as needed to satisfy conservation
  - Keep the remaining free parameters as the optimization α^u values
  - All arithmetic must use the QQ(ψ) fraction field for performance
  - The solve is: given conservation equations that are linear in (w_1..w_{R-1}, free_1..free_k), find the weights and determine which free_i are constrained
  - Key: the system is linear in the unknowns (weights and free params), so `linsolve` or `Matrix.solve` in QQ(ψ) works
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **22.3b** Handle the quadrature weight output:
  - The conservation solve produces ψ-dependent quadrature weights w_0=ψ, w_1(ψ,α), ..., w_{R-1}(ψ,α)
  - These weights are needed for the C++ stencil (they appear in `info.w` or similar)
  - Store weights in the `StencilResult` or `CutCellResult`
  - For E2_1: verify weights match the existing `w_0, w_1, w_2, w_3` values from E2_1.cpp
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

### 22.4 — Validate conservation for E4_1

- [ ] **22.4a** Remove the `xfail` marker from `test_e4_1_conservation_fails`:
  - The test should now PASS with the conservation-enforced pipeline
  - Verify: `Σ_i w_i · B[i,j] = 0` as a polynomial identity in ψ and α for ALL interior columns
  - Verify symbolically (for all ψ and α) and numerically (at specific values)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

- [ ] **22.4b** Verify E4_1 free parameter count:
  - After conservation, E4_1 should have exactly 4 free α parameters (matching Table 1)
  - These are the parameters that get passed as `std::span<const real>` in the C++ constructor
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

- [ ] **22.4c** Re-generate E4_1 C++ code with conservation-enforced stencil:
  - The generated code will produce different (larger) expressions than Phase 21's output
  - Run the codegen pipeline and write new E4_1.cpp
  - Verify it still compiles and passes structural checks
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

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

- [ ] **22.7a** Add quadrature weight output to generated C++ if needed:
  - The C++ stencil struct may need a method to provide quadrature weights for the SBP norm
  - Check existing stencils (E2_1.cpp) for how weights are stored/accessed
  - If weights are needed: add `w` array or method to the codegen struct template
  - File: `scripts/stencil_gen/stencil_gen/codegen.py`

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
