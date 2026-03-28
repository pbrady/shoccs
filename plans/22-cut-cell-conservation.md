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
- Columns with nonzero IC (grid-frame): 2, 3 (2 columns; T-frame: 3, 4)
- Conservation equations: T−1 = 4 (all columns j=0..T−2 per `conservation.py`)
- Unknowns: 3 weights (w_1, w_2, w_3) + 1 phi placeholder (nextra=1) = 4
- **Exactly determined** → weights + phi absorb all constraints, no alpha constraints needed

### Why E4_1 fails
- E4_1: R=4, T=7, p=2, nextra=0
- Columns with nonzero IC (grid-frame): 1, 2, 3, 4 (4 columns; T-frame: 2, 3, 4, 5)
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
  - **Interior column identification:** T-frame column j (0-indexed, col 0 = wall) corresponds to grid point j-1 (col 1 = x_0). Interior row at T-frame position R+m (grid point R-1+m, m >= 0) covers T-frame columns `(R+m-p)..(R+m+p)`. A column j has nonzero IC if `j >= R-p` and `j` is within reach of at least one interior row. For E4_1 (R=4, p=2): j = 2..6 in T-frame may have nonzero IC; computed values are IC(2)=1/12, IC(3)=-7/12, IC(4)=-7/12, IC(5)=1/12, IC(6)=0.
  - **Interior contribution IC(j):** Sum of interior stencil coefficients touching T-frame column j. Interior row at T-frame R+m (grid point R-1+m) has coefficient `interior_coeffs[(j-1) - (R-1+m) + p]` at column j. **Equivalently:** `_interior_contribution(j-1, r, p, interior_coeffs)` from `conservation.py` where `r = R-1` (the number of uniform boundary rows = first interior grid point). **Note:** The TEMO boundary block has R rows covering grid points 0..R-2; interior rows start at grid point R-1 = r. This is the SAME r used by `conservation.py` in the uniform case. IC(j) = 0 for T-frame columns 0 and 1 (wall and x_0 are too far left for any interior row to reach with E4_1 parameters).
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
  - **Chosen approach:** Two-phase solve — treat alpha as parameters, solve for weights first, then derive alpha constraints from overdetermined residuals (DD22-1 and DD22-2 resolved).
  - **Step 1 — Build conservation equations:** Call `build_cut_cell_conservation_system()` from 22.2a to get `(equations, w_symbols)`.
  - **Step 2 — Extract linear system in weights:** The equations have the form `Σ w_i · f_i(ψ, α) + g(ψ, α) = 0` where f_i and g are rational in (ψ, α). This is LINEAR in the w_i unknowns (α symbols are treated as parameters, NOT unknowns). Use `linear_eq_to_matrix(equations, w_symbols)` to extract coefficient matrix `A` (n_eq × n_w) and RHS vector `b` (n_eq × 1). For E4_1: A is 6×3, b is 6×1, with entries that are rational functions of (ψ, α₀..α₃). No bilinear term issue arises because α symbols are parameters, not unknowns in this solve.
  - **Step 3 — Solve for weights (overdetermined case):** When n_eq > n_w (E4_1: 6 > 3), select n_w pivot rows to form a square system. Try `A_pivot = A[:n_w, :]` first; if singular (det == 0), use `A.rref()` to identify n_w linearly independent rows. Solve: `w_sol = A_pivot.solve(b_pivot)` → 3×1 Matrix giving w₁, w₂, w₃ as rational functions of (ψ, α). Apply `cancel()` to each entry.
  - **Step 4 — Derive alpha constraints:** Substitute `w_sol` into the remaining n_eq − n_w equations: for each remaining row k, compute `residual_k = cancel((A[k,:] * w_sol)[0] - b[k])`. Each residual must be identically zero → polynomial identity in ψ. For E4_1: 3 residual equations. These residuals are rational in (ψ, α); clear denominators if needed by multiplying by `denom = fraction(residual_k)[1]`.
  - **Step 5 — Solve alpha constraints:** The 3 residual equations (after clearing denominators) are polynomial in (ψ, α). Since they must hold for ALL ψ, extract ψ-coefficient equations: for each residual, compute `Poly(numer, psi).all_coeffs()` → list of α-polynomial equations that must all equal zero. Collect all such equations. If they are linear in α (expected, since B_l entries are linear in α), use `linear_eq_to_matrix(all_alpha_eqs, alpha_symbols)` to solve. For E4_1 with 4 alphas and 3 constraints: expect 1 remaining free alpha. The solve produces `{alpha_k: expr(alpha_remaining)}` for the constrained alphas.
  - **Step 6 — Substitute back:** Replace the constrained alphas in B_l using `.xreplace(alpha_solutions)`, then `cancel()` each entry. Also substitute into `w_sol`. Return `(B_l_conserved, weight_solutions, [alpha_remaining])`.
  - **Edge case (exactly determined):** When n_eq == n_w (e.g., E2_1 with nextra=1 where phi symbols are already resolved), skip Steps 4-6. Just solve for weights and return B_l unchanged with all alphas free. For E2_1: 4 equations in 3+1=4 unknowns (3 weights + 1 phi from 22.2a) → exactly determined.
  - **Performance note:** The system is small (≤6 equations). SymPy's `linear_eq_to_matrix` + `Matrix.solve` handle rational function coefficients. No need for `solve_in_field` or QQ(ψ) domain arithmetic. Target: < 5 seconds for E4_1.
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (add after `build_cut_cell_conservation_system`, ~line 1300)

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
  - **Dimensions:** E2_2 has p=1, q=1, nextra=0, nu=2. Computed: r=2, t=3, r_eff=1, R=2, T=4.
  - **Conservation system:** T−1 = 3 equations (j=0..2 in T-frame). Weight unknowns: w₁ only (w₀=ψ fixed, R−1=1 unknown). System is overdetermined: 3 equations, 1 weight unknown → 2 excess constraints on alphas.
  - **IC values:** Using `_interior_contribution(j-1, r_eff, p, interior)` where r_eff=1 (first interior grid point for nu=2). Precompute IC for j=0..2.
  - **Wall column convention for nu=2:** For the 2nd derivative, the SBP conservation condition may use a different wall column target than the 1st derivative's `col_sum = -1`. Verify against the math reference (Section 4.4) and the uniform conservation in `conservation.py`. If conservation.py's `col_sum + 1 = 0` convention applies to nu=2 as well, use it; otherwise derive the correct target from the SBP property `H D₂ + D₂ᵀ H = ...`.
  - **Test structure:** Build the E2_2 cut-cell conservation system using `build_cut_cell_conservation_system()` from 22.2a. Verify equation count (3) and that the system is solvable. If the system yields alpha constraints, verify they are consistent with the existing E2_2 stencil (which was derived without explicit conservation).
  - **Polynomial exactness cross-check:** Verify that the conservation-enforced E2_2 stencil still satisfies `f(x)=x² → f''=2` (matching the existing `test_conservation_polynomial_exactness` test at line 1772 of `test_temo.py`).
  - File: `scripts/stencil_gen/tests/test_temo.py`
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
**CHOSEN: Two-phase** (Taylor first, then conservation substitution). The per-row Taylor solve (existing pipeline) produces B_l entries as functions of (ψ, α). Conservation equations are then formed from B_l and solved for weights + alpha constraints. This avoids building a single large coupled system and reuses the existing TEMO row-solve infrastructure.

### DD22-2: Bilinear term handling
**CHOSEN: (a) Treat alphas as parameters, solve for w's.** The conservation equations are linear in the weight unknowns w₁..w_{R-1} when alpha symbols are treated as parameters (not unknowns). The bilinear terms `w_i * alpha_j` are only problematic if we try to solve for BOTH w and alpha simultaneously. With the two-phase approach (solve w first, then derive alpha constraints from residuals), the system is always linear. No theta-linearization needed.

### DD22-3: Alpha parameter reduction
**TBD (narrowed):** Conservation enforcement will reduce E4_1's free alpha count from 4 to a smaller number. Based on the DOF analysis:
- 6 conservation equations in 3 weight unknowns → 3 excess equations constraining alphas
- 4 alpha symbols − 3 constraints → **expected: 1 free alpha**
- But: the 3 residual equations from Step 4 of 22.3a may not all be independent (some may be redundant), so the actual free count could be 1–4. Must be determined empirically during 22.3a and recorded in 22.4b.
- This affects the C++ constructor signature (`std::span<const real>` size), the `param_arrays={"alpha": N}` in codegen, and the `alpha_symbols` list in `CutCellResult`.

---

## Performance Considerations

The conservation solve is small and tractable:
- E4_1: 6 conservation equations, 3 weight unknowns → one 3×3 solve (Step 3) + 3 residual equations (Step 4) + one 3×4 alpha solve (Step 5)
- All solves use SymPy's `linear_eq_to_matrix` + `Matrix.solve` with rational function entries in (ψ, α)
- No need for QQ(ψ) fraction field arithmetic — standard symbolic solve suffices
- Target: full E4_1 derivation with conservation < 10 seconds (conservation solve itself < 5 seconds)

## Key Implementation Insight

The cleanest approach is probably:
1. **Keep the per-row Taylor solve** — each row's entries are functions of (ψ, row_free_params, α^u)
2. **Substitute into conservation equations** — conservation becomes a system in (weights, row_free_params)
3. **Solve the coupled conservation + weight system** — determine which row_free_params are constrained
4. **Substitute back** — replace constrained free params, leaving only α^u as optimization targets

This is a two-phase approach (Taylor first, then conservation) rather than solving one giant monolithic system. The Taylor phase produces the "shape" of each row; conservation then constrains the remaining degrees of freedom across rows.
