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

- [ ] **27.1a** Diagnostic: verify boundary rows are polynomial after QQ(ψ) solve
  - Write a diagnostic test in `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - For E4_1 with `zeros={3,4}`, call `construct_cut_cell_stencil` to get the
    raw TEMO output (before conservation)
  - For each boundary row i=0..3, check whether `cancel(entry)` is polynomial
    in ψ by extracting `fraction(entry)` and checking if the denominator is
    constant (no ψ)
  - **Expected result:** boundary row entries from `solve_temo_row` are
    rational in ψ with Vandermonde-type denominators (e.g. `(ψ+1)(ψ+2)(ψ+3)`)
    that factor out. After `cancel`, they should simplify to polynomials.
    If they DO simplify to polynomials, we may only need to change the
    conservation approach (not the boundary solve). If they DON'T, we need
    the polynomial ansatz.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (new test class
    `TestPolynomialStructure` near the end)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "TestPolynomialStructure" --timeout=300`

- [ ] **27.1b** Verify degree bound for polynomial entries
  - In the same `TestPolynomialStructure` class, for each boundary row entry
    that IS polynomial after `cancel`, compute its degree in ψ using
    `Poly(entry, psi).degree()`
  - Verify the maximum degree matches the paper's bound: ≤ q+1 = 4 for E4_1
    (q=3, plus 1 from the degree-1 prescribed entries interacting with the
    degree-3 Vandermonde column)
  - Also check: the entries are linear in the alpha symbols (degree ≤ 1 in
    each alpha_k) by using `Poly(entry, alpha_k).degree()` for each alpha
  - File: same test class as 27.1a

### 27.2 — Implement polynomial boundary row solve

- [ ] **27.2a** Add `solve_temo_row_polynomial` function
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (add after `solve_temo_row`
    at ~line 1350)
  - Signature: `solve_temo_row_polynomial(i, V, rhs, prescribed, psi, alpha_syms) -> RowSolveResult`
  - **Algorithm (polynomial ansatz approach):**
    1. Determine the maximum ψ-degree `d_max` for the solved entries. For E4_1:
       `d_max = q + 1 = 4` (from the q=3 Vandermonde interacting with degree-1
       prescribed entries). In general: `d_max = max(q, nu) + 1` for the
       first-derivative case.
    2. For each free column j (not prescribed), represent the unknown entry as
       `c_j(ψ) = c_{j,0} + c_{j,1}*ψ + ... + c_{j,d_max}*ψ^d_max` using
       fresh SymPy symbols `c_{j,k}`.
    3. Substitute these polynomial unknowns AND the prescribed entries into the
       Taylor accuracy equations `V * x = rhs`. Each equation becomes a
       polynomial identity in ψ.
    4. Expand and collect by powers of ψ: each coefficient of ψ^m must be zero
       (for matching equations) or match the RHS.
    5. This gives a purely rational (no-ψ) linear system in the c_{j,k}
       unknowns and the alpha symbols.
    6. Solve this system using `sympy.solve` or `linear_eq_to_matrix` +
       `linsolve`. The solution gives c_{j,k} as rational functions of the
       alpha symbols.
    7. Reconstruct each entry as `sum(c_{j,k} * ψ^k for k in range(d_max+1))`.
    8. Return a `RowSolveResult` with these polynomial entries.
  - **Key difference from `solve_temo_row`:** No QQ(ψ) fraction field. The ψ
    variable is expanded out, and we solve a larger but purely rational system.
  - **Size estimate:** ~80-100 lines of new code. No new files needed.
  - Must come before 27.2b.

- [ ] **27.2b** Test polynomial boundary rows for E4_1
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (new test class
    `TestPolynomialBoundaryRows`)
  - Tests to write:
    1. `test_all_entries_polynomial`: For rows 0-3, every entry from
       `solve_temo_row_polynomial` is a polynomial in ψ (no denominator after
       `fraction(cancel(entry))`).
    2. `test_degree_bound`: Max ψ-degree of each entry is ≤ 4 (for E4_1).
    3. `test_taylor_accuracy_symbolic`: For each row, verify
       `sum_j c_j * delta_j^m / m! = delta_{m,nu}` holds as a polynomial
       identity in ψ (substitute several ψ values, or check symbolically).
    4. `test_psi_1_matches_uniform`: At ψ=1, boundary row entries match
       `B_l(1)` (the uniform-limit embedding from `solve_uniform_limit`).
    5. `test_psi_0_matches_degenerate`: At ψ=0, boundary row entries match
       `B_d` (the degenerate stencil from `build_degenerate_stencil`).
    6. `test_linear_in_alphas`: Each entry is at most degree 1 in each
       alpha symbol.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "TestPolynomialBoundaryRows" --timeout=300`
  - Must come after 27.2a.

### 27.3 — Near-interior row with conservation

- [ ] **27.3a** Implement combined Taylor + conservation solve for the near-interior row
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (new function
    `solve_near_interior_with_conservation`, add after
    `solve_temo_row_polynomial`)
  - Signature: `solve_near_interior_with_conservation(boundary_rows: Matrix, B_u, interior, p, q, nu, psi, alpha_syms, zeros) -> tuple[list, list, Expr]`
    - Returns: `(row_coeffs, weights, denominator_e0)` where row_coeffs are
      rational in ψ with common denominator e0
  - **Algorithm:**
    1. Start with the polynomial boundary rows (from 27.2) as a partial R×T
       matrix (rows 0..r-1 filled, row r unknown).
    2. For row r (near-interior), prescribe the zeroed column:
       `α_{r,1}(ψ) = ψ * B_l_1[r,1]` (Category A, same as current code).
    3. Set up the Taylor system for row r: `V_r * x_r = rhs_r` where `V_r` is
       the cut-cell Vandermonde at row r (n_eqs × T with ψ-dependent col 0).
       Move prescribed column(s) to RHS.
    4. Set up conservation equations from the polynomial boundary rows:
       - Use `build_cut_cell_conservation_system` or equivalent with the
         polynomial boundary rows as input (w_0 = ψ is fixed).
       - The conservation equations involve: w_syms (weights w_1..w_{R-1}),
         the unknown row-r entries, and the alpha symbols.
    5. Combine the Taylor equations (from step 3) and conservation equations
       (from step 4) into a single linear system. The unknowns are: the free
       row-r entries + the weight symbols w_1..w_{R-1}.
    6. Solve the combined system. Because both Taylor and conservation equations
       are polynomial in ψ (the boundary rows are polynomial, w_0=ψ is
       polynomial), the solution is rational in ψ.
    7. Extract the common denominator e0(ψ) from the row-r entries.
    8. Verify e0(ψ) is nonvanishing on [0,1]: check `e0(0) ≠ 0`, `e0(1) ≠ 0`,
       and use Sturm's theorem or numerical evaluation to confirm no real
       roots in (0,1).
  - **Key difference from current approach:** Conservation and Taylor for
    row r are solved SIMULTANEOUSLY, not sequentially. This avoids the
    ψ*(ψ-1) denominator that arises from sequential conservation substitution.
  - **Size estimate:** ~100-130 lines. The current `build_cut_cell_conservation_system`
    and `solve_cut_cell_conservation` can be partially reused, but the
    simultaneous solve requires restructuring.
  - Must come after 27.2a (needs polynomial boundary rows as input).

- [ ] **27.3b** Validate the near-interior row
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (new test class
    `TestNearInteriorRow`)
  - Tests:
    1. `test_common_denominator_nonvanishing`: Extract e0(ψ) from the row-r
       entries. Verify `e0(0) ≠ 0` and `e0(1) ≠ 0`. Use
       `Poly(e0, psi).all_roots()` or numerical sampling to confirm no roots
       in [0,1].
    2. `test_taylor_accuracy`: Row r satisfies q+1=4 Taylor equations as a
       rational identity in ψ.
    3. `test_conservation_column_sums`: With the solved weights, weighted
       column sums equal zero (or -1 for col 0) as polynomial identities in ψ.
    4. `test_psi_0_limit`: At ψ=0, row-r entries match the degenerate stencil.
    5. `test_psi_1_limit`: At ψ=1, row-r entries match the uniform limit.
    6. `test_remaining_free_params`: The result has exactly 2 free parameters
       (for E4_1 with `zeros={3,4}`): one alpha and one weight.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "TestNearInteriorRow" --timeout=300`
  - Must come after 27.3a.

### 27.4 — Full E4_1 construction

- [ ] **27.4a** Wire up polynomial boundary rows + simultaneous near-interior solve
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - Add a new function `construct_cut_cell_stencil_polynomial(B_u, interior, p, q, nu, nextra, psi, zeros) -> StencilResult`
    that:
    1. Calls `solve_temo_row_polynomial` for rows 0..r-1 (boundary rows)
    2. Calls `solve_near_interior_with_conservation` for row r
    3. Returns the complete 5×7 stencil with polynomial/rational entries
    4. Returns the solved weights and remaining free parameters
  - Update `derive_cut_cell_scheme` to use `construct_cut_cell_stencil_polynomial`
    when `scheme.zeros` is set (the E4_1 path). The `zeros` path in
    `derive_cut_cell_scheme` (lines 2267-2312) currently does:
    ```
    uniform → construct_cut_cell_stencil → build_cut_cell_conservation_system
      → solve_cut_cell_conservation → xreplace
    ```
    Replace this with:
    ```
    uniform → construct_cut_cell_stencil_polynomial (does everything in one step)
    ```
  - The non-zeros paths (E2_1, E2_2, generic conservative) remain unchanged.
  - **Size estimate:** ~60-80 lines for the new function, ~20-30 lines to
    modify `derive_cut_cell_scheme`.
  - Must come after 27.3a.

- [ ] **27.4b** Validate the full E4_1 stencil
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (new test class
    `TestPolynomialFullStencil`)
  - Tests:
    1. `test_all_entries_well_defined`: For all 5 rows × 7 cols, no ψ or
       (ψ-1) factors in denominators. For rows 0-3, entries should be purely
       polynomial. For row 4, denominators should be e0(ψ) only.
    2. `test_taylor_accuracy_all_rows`: All 5 rows satisfy Taylor accuracy.
    3. `test_conservation_column_sums`: Conservation holds for all columns.
    4. `test_psi_0_limit`: Correct degenerate limit.
    5. `test_psi_1_limit`: Correct uniform limit.
    6. `test_free_parameter_count`: Exactly 2 free parameters after zeros +
       conservation.
    7. `test_matches_derive_cut_cell_scheme`: Calling `derive_cut_cell_scheme(E4_1, psi)`
       uses the polynomial path and produces the correct result.
  - Also update existing tests in `TestE4CutCellSchemeWithZeros` (lines 1022+)
    and `TestDeriveCutCellScheme` (lines 853+) to verify the new polynomial
    properties (entries well-defined at ψ=0 and ψ=1, no clamping needed).
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v --timeout=300`
  - Must come after 27.4a.

### 27.5 — Remove clamping and singularity guards

- [ ] **27.5a** Remove ψ-clamping from E4_1.cpp
  - File: `src/stencils/E4_1.cpp`
  - Remove the psi clamping on line 99:
    `psi = std::clamp(psi, psi_eps, 1.0 - psi_eps);`
    and the `constexpr real psi_eps = 1e-4;` on line 98.
  - Remove the alpha[1] lower-bound check in the constructor (lines 35-38):
    `if (alpha[1] < 197.0 / 288.0) throw ...`
  - Remove the comment block explaining the singularity constraints (lines 17-28).
  - **Note:** This item is a verification/cleanup step AFTER 27.6a regenerates
    E4_1.cpp. If the regenerated file is already clean (no clamping or alpha
    guards), this reduces to a verification check. If any guards remain in
    the generated file, remove them manually.
  - Test: `cmake --build build --target t-E4_1 && ctest --test-dir build -R t-E4_1`
  - Must come after 27.6a (regeneration produces the new file first).

### 27.6 — Regenerate E4_1 C++ and verify

- [ ] **27.6a** Regenerate E4_1.cpp and E4_1.t.cpp from the polynomial construction
  - The codegen pipeline writes to `scripts/stencil_gen/output/E4_1.cpp` and
    `scripts/stencil_gen/output/E4_1.t.cpp` via the test suite (see
    `TestE4CodeGeneration.test_write_output` at line 658 and
    `TestE4TestFileGeneration.test_write_test_output` at line 808 in
    `tests/test_e4_cut_cell.py`).
  - Steps:
    1. Run: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "test_write_output or test_write_test_output" --timeout=300`
    2. Copy output: `cp scripts/stencil_gen/output/E4_1.cpp src/stencils/E4_1.cpp`
       and `cp scripts/stencil_gen/output/E4_1.t.cpp src/stencils/E4_1.t.cpp`
    3. Verify the generated E4_1.cpp does NOT contain `std::clamp`, `psi_eps`,
       or `alpha[1] >= 197` guards.
    4. Verify the generated E4_1.cpp contains polynomial expressions for rows
       0-3 (no `1.0/psi` or `1.0/(psi - 1)` divisions) and rational
       expressions for row 4 with a common denominator.
  - Build and test: `cmake --build build --target t-E4_1 && ctest --test-dir build -R t-E4_1`
  - Also run the full C++ test suite to check for regressions:
    `cmake --build build && ctest --test-dir build`
  - Must come after 27.4b (polynomial construction is verified).

### 27.7 — E2_1 regression

- [ ] **27.7a** Verify E2_1 is unaffected (it already works correctly):
  - The polynomial construction changes only affect the `zeros` path in
    `derive_cut_cell_scheme`. E2_1 uses `nextra=1` with no zeros, so it
    takes the conservation path which is unchanged.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "E2" --timeout=120`
  - Also verify C++ E2 tests: `ctest --test-dir build -R "t-E2"`
