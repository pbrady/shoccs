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

## Procedure (Paper's Ideal Structure)

The paper constructs the stencil as follows (for reference):

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

## Implementation Strategy

Our implementation differs from the paper's procedure in step 2: instead of
solving Taylor + conservation simultaneously (which creates a bilinear system
in weight × row-entry products — see "Bilinearity Constraint" below), we:

1. Use the **polynomial ansatz** (`solve_temo_row_polynomial` from 27.2a) for
   boundary rows (i=0..R-2). 27.1a confirmed that the existing QQ(ψ) solve
   produces rational entries with Vandermonde-type denominators that do NOT
   simplify to polynomials after `cancel()`. The polynomial ansatz eliminates
   these denominators by construction.
2. Use the existing `solve_temo_row` (QQ(ψ) solve) for the near-interior row
   (i=R-1), which is expected to be rational with a benign denominator.
3. Build conservation equations from the TEMO output (existing code).
4. Solve conservation using a **fraction-clearing** approach (new
   `solve_conservation_fraction_free`) that converts rational equations
   to polynomial form before solving, avoiding ψ(ψ-1) denominators.
5. Apply the conservation solution via `xreplace` (existing code).

The result should match the paper's structure: polynomial boundary rows,
rational near-interior row with nonvanishing denominator, no singularities
on [0,1].

---

## Dependency Graph

```
27.1a → 27.1b → 27.1c (decision gate)
                  ├── boundary rows polynomial after cancel → skip 27.2 ─┐
                  └── need polynomial ansatz → 27.2a → 27.2b ────────────┤
                                                                         ├→ 27.3a → 27.3b → 27.4a ─┬→ 27.4b → 27.6a → 27.5a
                                                                         │                           └→ 27.7a
```

### Bilinearity Constraint (cross-cutting)

The conservation equations involve products `w_k * B_l[k, j_tf]` where `B_l[k, j_tf]`
is linear in alpha symbols. When both `w_k` AND `alpha_m` are solve targets, these
terms are bilinear (e.g., `w_1 * alpha_1 * f(ψ)`). For E4_1 with zeros={3,4},
the bilinear unknown×unknown product is `w_1 * alpha_1` (row 1's stencil entries
depend on alpha_1, and w_1 weights that row). This rules out treating the combined
system as a simple linear system.

The primary approach in 27.3a handles bilinearity by fraction-clearing the
equations (converting from rational to polynomial form) and then letting SymPy's
`solve` use Gröbner basis methods on the 5-equation, 5-unknown polynomial system.
This is feasible because the system is small. If this proves too slow, Fallback A
in 27.3a uses sequential elimination (solve alphas first as a linear sub-problem,
then solve the resulting nonlinear-in-w system). Note: the sequential approach does
NOT linearize the weight solve — after alpha substitution and fraction-clearing,
the weight equations are polynomial of degree ≤ 3 in w (not linear), because the
alpha solutions are rational in w.

The existing test `test_e4_1_conservation_constant_weights_infeasible_r5` (line
1481 of `test_e4_cut_cell.py`) demonstrates a related technique
(theta-linearization + fraction clearing + ψ-coefficient extraction) for a
different purpose (proving infeasibility of constant weights).

## Items

### 27.1 — Understand the polynomial structure

- [x] **27.1a** Diagnostic: verify boundary rows are polynomial after QQ(ψ) solve
  - **Result:** Boundary rows are NOT polynomial after `cancel()`. They have
    Vandermonde-type denominators: `(ψ+1)(ψ+2)(ψ+3)`, `2(ψ+1)`, `2(ψ+2)`,
    `6(ψ+3)`. All denominators are nonvanishing on [0,1] (benign).
  - Test class `TestPolynomialStructure` added near end of
    `scripts/stencil_gen/tests/test_e4_cut_cell.py`.
  - `test_boundary_rows_have_vandermonde_denominators` — PASS (confirms
    denominators exist and are Vandermonde-type/benign)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "TestPolynomialStructure" --timeout=300`

- [x] **27.1b** Verify degree bound for polynomial entries
  - Adapted for rational entries: checks numerator degree ≤ 7, denominator
    degree ≤ 3, and numerator is linear in each alpha symbol.
  - `test_numerator_degree_bound` — PASS
  - `test_entries_linear_in_alphas` — PASS (numerators linear in alpha,
    denominators free of alpha)

- [x] **27.1c** Decision gate: polynomial ansatz needed?
  - **Decision outcome:** Boundary rows are NOT polynomial after cancel.
    The polynomial ansatz (27.2a) IS required. Proceed with 27.2a.
  - The Vandermonde-type denominators are benign (nonvanishing on [0,1]),
    but the paper expects polynomial boundary rows. The polynomial ansatz
    will eliminate these denominators entirely.

### 27.2 — Implement polynomial boundary row solve

- [x] **27.2a** Add `solve_temo_row_polynomial` function
  - **Conditional:** Only needed if 27.1c determines polynomial ansatz is required.
    If boundary rows from QQ(ψ) solve already simplify to polynomials, skip this.
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (added after `solve_temo_row`
    at ~line 1355)
  - Signature: `solve_temo_row_polynomial(i, V, rhs, prescribed, psi, symbols) -> RowSolveResult`
  - **Implementation notes:**
    - d_max is computed dynamically from the max ψ-degree in V (= q for E4_1),
      giving d_max = max_v_degree + 1 = 4.
    - **Key discovery:** When the prescribed dict includes extra columns (beyond
      the zeroed column) as degree-1 polynomials, the system becomes square with
      a unique rational solution — no polynomial of degree d_max satisfies it.
      The function handles this by automatically converting extra prescribed
      columns to **endpoint constraints**: instead of forcing the entry to a
      specific degree-1 polynomial, it only constrains the values at ψ=0 and
      ψ=1. This makes the system underdetermined (30 unknowns, 30 equations
      for E4_1 with 2 extra cols × 2 endpoints), enabling polynomial solutions.
    - The first prescribed column (by index, = zeroed column) is kept as a true
      prescription. All subsequent prescribed columns are converted to limit
      constraints.
    - Verified: all 4 boundary rows produce polynomial entries of degree ≤ 4,
      satisfy Taylor accuracy symbolically, and match B_l_1 at ψ=1 and B_d at ψ=0.
    - All 98 existing tests pass (1 xfail expected). No regressions.
    - Added `expand` and `fraction` to the module-level imports in temo.py.
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

### 27.3 — Fraction-free conservation solve

- [ ] **27.3a** Implement `solve_conservation_fraction_free` function
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (new function, add after
    `solve_cut_cell_conservation` at ~line 1507)
  - Signature: `solve_conservation_fraction_free(equations: list[Expr], solve_for: list[Symbol], psi: Symbol) -> dict[Symbol, Expr]`
  - **Root cause of current problem:** The current `solve_cut_cell_conservation`
    (line 1507) passes rational-in-ψ equations directly to SymPy's `solve`,
    which can introduce ψ(ψ-1) factors in solution denominators through its
    internal elimination steps. Even though the conservation equations are
    consistent identities in ψ, SymPy's generic solver may choose an
    elimination order that creates spurious poles.
  - **Fix:** Clear ψ-dependent denominators before solving. The fraction-
    clearing is the key improvement — it converts rational-in-ψ equations
    to polynomial-in-ψ equations, preventing SymPy from introducing bad
    denominators during elimination.
  - **Algorithm (primary approach — fraction-clear + full solve):**

    **Step 1 — Clear fractions:**
    ```python
    cleared = []
    for eq in equations:
        num, den = fraction(cancel(eq))
        cleared.append(expand(num))
    ```
    After clearing, each equation is a polynomial in ψ (and the solve_for
    symbols + free params). Denominators are discarded because they are
    nonvanishing on [0,1] (products of Vandermonde factors like (ψ+1)(ψ+2)).

    **Step 2 — Solve the full system:**
    ```python
    sol = solve(cleared, solve_for, dict=True)
    assert len(sol) == 1
    sol = sol[0]
    ```
    SymPy's `solve` on polynomial inputs uses Gröbner basis methods, which
    tend to produce solutions with minimal denominators. Since the input
    polynomials have no ψ-dependent denominators to begin with, the solver
    cannot introduce ψ(ψ-1) factors through cross-multiplication.

    **Step 3 — Verify and return:**
    ```python
    # Verify: all original equations evaluate to 0
    for i, eq in enumerate(equations):
        residual = cancel(eq.subs(sol))
        assert residual == 0, f"Equation {i}: residual={residual}"

    return sol
    ```

  - **Bilinearity note:** The conservation equations contain bilinear
    terms `w_k * alpha_j` (see "Bilinearity Constraint" section above).
    For E4_1 with zeros={3,4}, the bilinear unknown×unknown products are
    `w_1 * alpha_1` (from row 1) — both are in `solve_for`. SymPy's
    `solve` can handle small bilinear polynomial systems (5 equations in
    5 unknowns) directly. The primary approach lets SymPy manage the
    bilinearity internally after fraction-clearing removes the rational
    structure that caused the original problem.

  - **Dimension check for E4_1 with zeros={3,4}:**
    - 5 conservation equations (one per grid column 1..5 in the T-frame)
    - solve_for = [alpha_0, alpha_1, w_1, w_2, w_3] (5 unknowns)
    - Free params: alpha_2, w_4 (not in solve_for)
    - After solve: alpha_2 renamed to alpha_0, w_4 renamed to alpha_1 → 2 free

  - **Why this avoids ψ(ψ-1) denominators:**
    - Clearing fractions (Step 1) removes all existing ψ-dependent
      denominators from the equations before solving.
    - The cleared equations are polynomials in (ψ, alpha, w), so the
      solver operates entirely in polynomial arithmetic.
    - Solution denominators can only arise from the Gröbner basis
      elimination, which produces factors of the polynomial coefficients
      — these are Vandermonde-family factors (nonvanishing on [0,1]),
      not ψ(ψ-1).

  - **Fallback A — Sequential elimination:** If the primary approach
    produces bad denominators or is too slow (timeout), split the solve
    into two stages:
    ```python
    alpha_unknowns = [s for s in solve_for if s.name.startswith('alpha')]
    weight_unknowns = [s for s in solve_for if s.name.startswith('w')]
    # Stage 1: linear in alpha (treating w as parameters)
    alpha_sol = solve(cleared, alpha_unknowns, dict=True)[0]
    # Stage 2: substitute alphas, clear new fractions, solve for w
    remaining = [expand(eq.subs(alpha_sol)) for eq in cleared]
    remaining_cleared = [expand(fraction(cancel(eq))[0]) for eq in remaining]
    weight_sol = solve(remaining_cleared, weight_unknowns, dict=True)[0]
    # Combine
    full_sol = {s: cancel(e.subs(weight_sol)) for s, e in alpha_sol.items()}
    full_sol.update(weight_sol)
    ```
    **Important caveat:** Stage 2 is NOT linear in weights. The alpha
    solutions from Stage 1 are rational in w (from Cramer's rule on the
    bilinear system), so after substitution and fraction-clearing, the
    remaining equations are polynomial of degree ≤ 3 in w. SymPy's
    `solve` handles this via Gröbner bases, but it may be slower than
    the primary approach. Use all 5 cleared equations (not just remaining)
    to give `solve` maximum information for elimination.

  - **Fallback B — ψ-coefficient extraction:** If both primary and
    Fallback A still produce bad denominators, expand each cleared equation
    as a ψ-polynomial, collect coefficients by ψ powers, and solve the
    resulting ψ-free algebraic system. This uses the technique from the
    existing `test_e4_1_conservation_constant_weights_infeasible_r5` test
    (lines 1541-1551 of `test_e4_cut_cell.py`). However, this forces the
    solution to be ψ-independent, which may not match the expected
    ψ-dependent weights. Use only if the ψ-coefficient system is consistent.

  - **Reuse from existing code:** The existing `build_cut_cell_conservation_system`
    (line 1431) is reused unchanged. The existing `solve_cut_cell_conservation`
    (line 1507) is NOT reused (it is the function being replaced). The
    existing test at line 1481
    (`test_e4_1_conservation_constant_weights_infeasible_r5`) demonstrates
    the fraction-clearing technique that this function productionizes.
  - **Size estimate:** ~20-30 lines for primary approach, ~40-60 if
    fallback A is also implemented inline.
  - Must come after 27.1c decision. If 27.1c says "boundary rows already
    polynomial", the approach works directly (TEMO entries have benign
    denominators). If 27.1c says "polynomial ansatz needed", must come
    after 27.2b (ensures boundary rows are polynomial before conservation).

- [ ] **27.3b** Validate the fraction-free conservation solve
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (new test class
    `TestFractionFreeConservation`)
  - Tests:
    1. `test_solution_no_psi_poles`: For each symbol in the conservation
       solution dict, extract `fraction(cancel(sol[s]))` and verify the
       denominator has no ψ or (ψ-1) factors. Specifically:
       ```python
       for s, expr in sol.items():
           num, den = fraction(cancel(expr))
           if psi in den.free_symbols:
               p = Poly(den, psi)
               assert p.eval(0) != 0, f"{s}: denominator vanishes at psi=0"
               assert p.eval(1) != 0, f"{s}: denominator vanishes at psi=1"
       ```
    2. `test_matches_current_output`: After xreplace with the fraction-free
       solution, the stencil entries should be equivalent (symbolically) to
       the current code's output. For specific alpha/psi values, numerical
       evaluation should match to machine precision.
    3. `test_conservation_holds`: Same as existing `test_conservation_holds`
       in `TestE4CutCellSchemeWithZeros` (line 1076) — verify that weighted
       column sums satisfy conservation identically in ψ.
    4. `test_remaining_free_params`: The solution leaves exactly alpha_2 and
       w_4 as free parameters (2 total for E4_1 with zeros={3,4}).
    5. `test_solve_for_completeness`: Verify that the solution dict contains
       all 5 solve_for symbols (alpha_0, alpha_1, w_1, w_2, w_3) and that
       substituting the solution into ALL 5 conservation equations gives
       zero residual (not just a subset).
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "TestFractionFreeConservation" --timeout=300`
  - Must come after 27.3a.

### 27.4 — Full E4_1 construction

- [ ] **27.4a** Integrate polynomial ansatz and fraction-free conservation into `derive_cut_cell_scheme`
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - **Modify `construct_cut_cell_stencil`** (line 1353) and
    **`derive_cut_cell_scheme`** (the `scheme.zeros` branch at lines
    2267-2312). Three targeted changes:

    **Change 1 (critical — wires 27.2a into the pipeline):** In
    `construct_cut_cell_stencil`, modify the row loop (line 1412) to call
    `solve_temo_row_polynomial` for boundary rows (i < R-1) and the
    existing `solve_temo_row` for the near-interior row (i = R-1):
    ```python
    for i in range(R):
        V, rhs_vec = build_temo_vandermonde(i, T, q, nu, psi)
        prescribed = identify_prescribed_entries(...)
        if i < R - 1:
            result = solve_temo_row_polynomial(
                i, V, rhs_vec, prescribed, psi, alpha_syms
            )
        else:
            result = solve_temo_row(
                i, V, rhs_vec, prescribed, psi, K, alpha_syms
            )
        rows.append(result.coeffs)
    ```
    Without this change, the polynomial ansatz from 27.2a would never be
    called and boundary rows would remain rational. This is the key
    integration point.

    **Change 2:** Add a cancel step after TEMO to simplify the near-interior
    row (boundary rows are already polynomial from Change 1). Insert after
    line 2278 (`R, T = floating.shape`), before conservation build:
    ```python
    # Cancel near-interior row to simplify rational entries
    for j in range(T):
        floating[R - 1, j] = cancel(floating[R - 1, j])
    ```

    **Change 3:** Replace the conservation solve call at line 2285:
    ```python
    # Current:
    sol = solve_cut_cell_conservation(eqs, solve_for)
    # New:
    sol = solve_conservation_fraction_free(eqs, solve_for, psi)
    ```

    The rest of the zeros path (weight extraction at line 2289, alpha
    renaming at lines 2291-2304, assembly at lines 2308-2312) remains
    completely unchanged. The `xreplace(sol)` at line 2287 also stays —
    the difference is that `sol` now contains expressions with benign
    denominators instead of ψ(ψ-1) factors.

  - The non-zeros paths (E2_1, E2_2, generic conservative at lines 2314-2433)
    remain completely unchanged.
  - **Imports:** No new import needed in `temo.py` (the new function is
    defined in the same file as `derive_cut_cell_scheme`). The test file
    `test_e4_cut_cell.py` only needs the import if 27.3b tests call the
    function directly — add it to the import block (around line 37) in
    that case.
  - **Size estimate:** ~15-25 lines changed across both functions. The bulk
    is the ~20-60 line new function from 27.3a (depending on whether
    fallback A is included inline).
  - Must come after 27.3b (conservation solve validated before wiring into
    full construction).

- [ ] **27.4b** Validate the full E4_1 stencil
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (new test class
    `TestPolynomialFullStencil`)
  - Tests:
    1. `test_all_entries_well_defined`: For all 5 rows × 7 cols, verify no
       ψ or (ψ-1) factors in denominators:
       ```python
       for i in range(R):
           for j in range(T):
               num, den = fraction(cancel(m[i, j]))
               if psi in den.free_symbols:
                   p = Poly(den, psi)
                   assert p.eval(0) != 0, f"[{i},{j}]: pole at psi=0"
                   assert p.eval(1) != 0, f"[{i},{j}]: pole at psi=1"
       ```
       For rows 0-3, entries should be purely polynomial (den is constant).
       For row 4, denominators should be Vandermonde-family polynomials only
       (nonvanishing on [0,1]).
    2. `test_taylor_accuracy_all_rows`: All 5 rows satisfy Taylor accuracy
       (reuse pattern from `TestE4CutCellSchemeWithZeros.test_taylor_accuracy`
       at line 1056).
    3. `test_conservation_column_sums`: Conservation holds for all columns
       (reuse pattern from `TestE4CutCellSchemeWithZeros.test_conservation_holds`
       at line 1076).
    4. `test_psi_0_limit`: At ψ→0, entries approach degenerate stencil values.
       Substitute ψ=0 and alpha values, verify entries are finite and match.
    5. `test_psi_1_limit`: At ψ=1, entries match uniform limit B_l(1).
    6. `test_free_parameter_count`: Exactly 2 free parameters after zeros +
       conservation (alpha_0, alpha_1 in final naming).
    7. `test_matches_derive_cut_cell_scheme`: Calling `derive_cut_cell_scheme(E4_1, psi)`
       uses the modified zeros path and produces the correct result.
    8. `test_weights_well_defined`: All 5 weights have no ψ(ψ-1) denominator
       factors. `weights[0] = psi`, others are rational with benign
       denominators.
  - Also update existing tests in `TestE4CutCellSchemeWithZeros` (lines 1022+):
    add a `test_no_singularities` method that verifies all floating/dirichlet
    entries and weights are finite at ψ=0 and ψ=1 (with representative alpha
    values). This serves as a regression guard against reintroducing poles.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v --timeout=300`
  - Must come after 27.4a.

### 27.5 — Remove clamping and singularity guards

- [ ] **27.5a** Verify ψ-clamping and alpha guards are absent from regenerated E4_1.cpp
  - **This is a post-27.6a verification step**, not a manual edit. The
    regenerated E4_1.cpp should already be free of singularity guards because
    the fraction-free conservation eliminates the root cause.
  - File: `src/stencils/E4_1.cpp` (after 27.6a copies the regenerated file)
  - Verify the following are ABSENT:
    - `psi_eps` and `std::clamp` (currently lines 98-99 and 214-215)
    - `alpha[1] < 197.0 / 288.0` constructor check (currently lines 35-38)
    - The singularity-explanation comment block (currently lines 17-28)
    - Any `1.0/psi` or `1.0/(psi - 1)` patterns in floating/dirichlet methods
    - Division by `t13` where `t13 = psi - 1` (currently line 110)
    - Division by `psi` in `t17 = t16/psi` (currently line 113)
  - **Note:** The codegen pipeline (`scripts/stencil_gen/stencil_gen/codegen.py`)
    does NOT hardcode any guards — the `psi_eps`/`std::clamp` and `alpha[1]`
    check in the current E4_1.cpp were added manually after code generation.
    After regeneration, these should NOT reappear. If the regenerated code
    DOES contain `1.0/psi`-type divisions (from CSE of rational entries), the
    fix is in the stencil derivation (27.3a/27.4a), not the codegen template.
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
  - **Concrete verification of the generated code:** Compare the current
    E4_1.cpp (323 lines) against the regenerated version:
    - Current code has `1.0 / (t40)` where `t40 = psi + 3` (line 137),
      `1.0 / (t32)` where `t32 = psi + 2` (line 129), `1/(alpha[1]*t13)`
      where `t13 = psi - 1` (line 110), and `t17 = t16/psi` (line 113).
      These are the Vandermonde-type and conservation-induced denominators.
    - After regeneration, rows 0-3 should only use polynomial arithmetic
      (`psi * ...`, `psi*psi * ...`, etc.) with no division by psi-dependent
      expressions. Row 4 should have ONE common denominator (an 8th-degree
      polynomial in ψ), not multiple separate denominators.
    - The constructor should NOT have the `alpha[1] < 197.0 / 288.0` check.
    - The `psi_eps` / `std::clamp` lines (current lines 98-99) should be absent.
  - **Note on codegen:** The `generate_stencil_cpp` function (in
    `scripts/stencil_gen/stencil_gen/codegen.py`) uses SymPy's `cse()` to
    produce CSE temporaries. The polynomial structure should produce simpler
    CSE trees (fewer divisions), so the generated code may be shorter.
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
  - Must come after 27.4a (regression guard runs after code changes are applied).
