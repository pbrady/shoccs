# Phase 27: Polynomial TEMO Construction with Proper Conservation

**Goal:** Rewrite the E4_1 cut-cell stencil construction to produce entries that match the paper's structure: polynomial entries for boundary rows, rational entries with non-vanishing denominator for the near-interior row. The key constraint: all entries must be well-defined over ψ ∈ [0,1] with NO singularities (no ψ or (ψ-1) factors in denominators).

**Depends on:** Phase 26 (conservation solve works with zeros), plus the paper's Appendix A (Table A.4)

**Read first:**
- `scripts/stencil_gen/stencil_gen/temo.py` — current construction
- `plans/stencil-derivation-math-reference.md` Section 4.3 (B_l(ψ) construction)
- The paper's Appendix A (Table A.4) which shows the actual E4_1 coefficients:
  - Rows 0-3: polynomial entries in ψ (degree 1-4)
  - Row 4 (near-interior): rational entries with common denominator e0 (degree-8 polynomial in ψ, nonvanishing on [0,1])
  - Free parameters: alphau_0_4, alphau_1_4, alphau_2_4, alphau_2_5 (constant, NOT ψ-dependent)

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/ -v --timeout=300
cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v --timeout=300
```

---

## Key Insight from the Paper

The paper (Section 3.2) says: "Note that the αu terms are specified rather than solved for. This avoids any singularities in the coefficients in the range ψ ∈ [0,1] and satisfies the design principles by construction."

The Appendix A (Table A.4) confirms the structure:
- **Boundary rows (i=0..3):** Each α_{i,j} is a polynomial in ψ of degree ≤ 4, with coefficients that are LINEAR functions of the free α^u parameters
- **Near-interior row (i=4):** Each α_{4,j} is a ratio polynomial(ψ)/e0(ψ), where e0 is a degree-8 polynomial whose constant term ≈ -25.2 (nonzero at ψ=0) and which evaluates to ~348 at ψ=1 (nonzero at ψ=1)
- **No singularities** anywhere on [0,1]

## Current Problem

Our `solve_temo_row` uses the QQ(ψ) fraction field to solve the Taylor accuracy system. This produces rational functions with denominators like (ψ+1)(ψ+2)(ψ+3) that come from the Vandermonde matrix determinant. These denominators don't vanish on [0,1], BUT the conservation solve then produces expressions with ψ·(ψ-1) denominators.

The paper's approach is different: boundary row entries are constructed as polynomials in ψ (not by solving a rational system), and only the near-interior row has rational entries (from the overdetermined conservation+Taylor system).

## The Fix: Polynomial Ansatz for Boundary Rows

For each boundary row i (0..r-1), the paper's Eq. from Section 3.2 prescribes:
- The zeroed column: `α_{i,δ}(ψ) = ψ · α^u_{i,δ}` (polynomial, degree 1)
- Extra free parameters: `α_{i,j}(ψ) = ψ·α^u_{i,j} + (1-ψ)·α^u_{i-1,j-1}` (polynomial, degree 1)
- All other entries: must satisfy Taylor accuracy AND be polynomial in ψ

The key: if we ASSUME entries are polynomials of a specific degree d, Taylor accuracy gives algebraic constraints on the polynomial coefficients. Since the Vandermonde matrix has ψ-dependent entries only in the first column, and the prescribed entries are degree-1 polynomials in ψ, the solved entries should be polynomials of degree ≤ d (where d depends on the Taylor order and the ψ-degree of the prescribed entries).

From the paper's Table A.4: boundary row entries have degree ≤ 4 in ψ. Since q=3 (Taylor accuracy order) and the grid spacing introduces ψ-dependence through `Δ_{i,wall} = -(ψ+i)`, the maximum ψ-degree in the Vandermonde is q=3. Combined with a degree-1 prescribed entry, the solved entries are degree ≤ 4 in ψ.

## Procedure

1. For each boundary row i=0..r-1:
   a. Set up the Taylor Vandermonde system (n_eqs × T matrix) with ψ-dependent first column
   b. Prescribe Category A + extra entries (degree-1 polynomials in ψ with α coefficients)
   c. Multiply through by the common denominator of the Vandermonde (eliminates fractions)
   d. Solve the cleared system — result is POLYNOMIAL in ψ
   e. The entries are polynomials in (ψ, α^u), degree ≤ ~4 in ψ

2. For the near-interior row (i=r):
   a. Set up the Taylor system (same as above)
   b. Prescribe the zeroed column (degree-1 polynomial)
   c. Apply conservation constraints (from all rows)
   d. Solve the combined Taylor + conservation system
   e. The result has a common denominator e0(ψ) that is nonvanishing on [0,1]
   f. The entries are rational: polynomial/e0(ψ)

3. Conservation applies to the FULL system simultaneously with the near-interior row solve.

---

## Items

### 27.1 — Understand the polynomial structure

- [ ] **27.1a** For E4_1 boundary row 0, compute what `solve_temo_row` currently produces and compare with what the polynomial ansatz would produce:
  - Current: rational function with denominator from QQ(ψ) solve
  - Expected: polynomial of degree ≤ 4 in ψ
  - Check: clear the Vandermonde denominator BEFORE solving and verify the result is polynomial
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (diagnostic test)

- [ ] **27.1b** Verify the paper's degree bound:
  - The Vandermonde column 0 entries are `(-(ψ+i))^k / k!` — polynomials of degree k in ψ
  - When we clear the denominator (the determinant of the non-ψ part), the RHS becomes polynomial
  - Verify the maximum ψ-degree of the resulting polynomial entries matches the paper (≤ 4 for E4_1)

### 27.2 — Implement polynomial boundary row solve

- [ ] **27.2a** Add a `solve_temo_row_polynomial` function that:
  1. Takes the same inputs as `solve_temo_row`
  2. Instead of solving in QQ(ψ), clears the Vandermonde determinant from the ψ-dependent column
  3. Solves the cleared system over ZZ[ψ, α] (polynomial ring, no fraction field)
  4. Returns polynomial entries (degree ≤ 4 in ψ for E4_1)
  5. Verifies Taylor accuracy holds (polynomial identity in ψ)
  - The key difference from `solve_temo_row`: avoid the fraction field entirely
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **27.2b** Test polynomial boundary rows for E4_1:
  - All 4 boundary rows produce polynomial entries
  - Entries match the paper's degree bound (≤ 4)
  - Taylor accuracy verified symbolically
  - At ψ=1: entries match B_l(1) (uniform embedding)
  - At ψ=0: entries match B_d (degenerate)
  - No denominators that vanish on [0,1]
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 27.3 — Near-interior row with conservation

- [ ] **27.3a** For the near-interior row (row 4 of E4_1):
  - This row's Taylor system is underdetermined (more unknowns than equations after prescriptions)
  - The excess DOF are consumed by conservation constraints
  - Solve Taylor + conservation simultaneously for this row
  - The result is rational: polynomial numerators / common polynomial denominator
  - The denominator must be nonvanishing on [0,1]
  - Apply the paper's zero constraints (alpha_3=0, alpha_4=0) before solving
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **27.3b** Validate the near-interior row:
  - Denominator e0(ψ) is nonvanishing on [0,1] (check e0(0) ≠ 0, e0(1) ≠ 0, and no real roots in [0,1])
  - Taylor accuracy holds
  - Conservation column sums are zero for all ψ
  - At ψ=0 and ψ=1: correct limit behavior
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 27.4 — Full E4_1 construction

- [ ] **27.4a** Wire up the polynomial boundary rows + conservation-coupled near-interior row:
  - Update `construct_cut_cell_stencil` (or add a new function) to use the polynomial approach
  - For rows 0..r-1: use `solve_temo_row_polynomial`
  - For row r: solve Taylor + conservation simultaneously
  - Return the complete 5×7 stencil with polynomial/rational entries
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **27.4b** Validate the full E4_1 stencil:
  - All entries well-defined on ψ ∈ [0,1] (no singularities)
  - Taylor accuracy for all rows
  - Conservation column sums zero for all ψ
  - Correct limits at ψ=0 and ψ=1
  - Remaining free parameters: alphau_0_4, alphau_1_4, alphau_2_4, alphau_2_5 (constant)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 27.5 — Remove clamping and singularity guards

- [ ] **27.5a** Remove the ψ-clamping and alpha constraints from the generated E4_1.cpp:
  - Remove `psi = std::clamp(psi, 1e-4, 1-1e-4)`
  - Remove `alpha[1] >= 197/288` constraint
  - The stencil should work for all ψ ∈ [0,1] without guards
  - File: `src/stencils/E4_1.cpp`

### 27.6 — Regenerate E4_1 C++ and verify

- [ ] **27.6a** Regenerate E4_1.cpp with polynomial construction:
  - Build and test: `cmake --build build --target t-E4_1 && ctest --test-dir build -R t-E4_1`
  - Verify no division-by-zero at ψ=0, ψ=1, or any intermediate value
  - File: `src/stencils/E4_1.cpp`, `src/stencils/E4_1.t.cpp`

### 27.7 — E2_1 regression

- [ ] **27.7a** Verify E2_1 is unaffected (it already works correctly):
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "E2" --timeout=120`
