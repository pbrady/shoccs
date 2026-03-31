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
   to polynomial form before solving.
5. Apply the conservation solution via `xreplace` (existing code).

**27.3a update:** Fraction-clearing does NOT eliminate ψ(ψ-1) denominators
from the alpha solutions. The bilinear structure of the conservation system
(solving for both alpha AND weight unknowns) inherently produces
ψ-dependent alpha solutions with ψ(ψ-1) poles. The weight solutions are
ψ-independent. This means the current approach (steps 1-5) produces
polynomial boundary rows PRE-conservation, but AFTER `xreplace(sol)` the
final entries reacquire ψ(ψ-1) denominators via the alpha substitution.
To truly match the paper's structure (no singularities on [0,1]), the
conservation solve must NOT solve for alphas — they should remain as
user-specified constants. See 27.3c decision gate.

---

## Dependency Graph

```
27.1a → 27.1b → 27.1c (decision gate)
                  ├── boundary rows polynomial after cancel → skip 27.2 ─┐
                  └── need polynomial ansatz → 27.2a → 27.2b ────────────┤
                                                                         ├→ 27.3a → 27.3b → 27.3c (decision gate) → 27.4a ─┬→ 27.4b → 27.6a → 27.5a
                                                                         │                                                   └→ 27.7a
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

- [x] **27.2b** Test polynomial boundary rows for E4_1
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (new test class
    `TestPolynomialBoundaryRows`)
  - Tests written (all 6 PASS):
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
  - Also added `identify_prescribed_entries` and `solve_temo_row_polynomial`
    to the test file's top-level imports.
  - All 104 existing tests pass (1 xfail expected). No regressions.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "TestPolynomialBoundaryRows" --timeout=300`
  - Must come after 27.2a.

### 27.3 — Fraction-free conservation solve

- [x] **27.3a** Implement `solve_conservation_fraction_free` function
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (added after
    `solve_cut_cell_conservation` at ~line 1688)
  - Signature: `solve_conservation_fraction_free(equations: list[Expr], solve_for: list[Symbol], psi: Symbol) -> dict[Symbol, Expr]`
  - **Implementation:** Clears ψ-dependent denominators from the
    conservation equations before passing to SymPy's `solve`.
  - **Key finding:** Fraction-clearing alone does NOT eliminate ψ(ψ-1)
    denominators from the alpha solutions. The ψ(ψ-1) factors are
    **inherent to the bilinear system structure**, not a solver artifact.
    Detailed investigation (see below) confirmed this.
  - **What was tried and why it failed:**
    1. **Primary approach (fraction-clear + full solve):** Works
       correctly — produces the same results as the old solver. But the
       alpha solutions still have `psi*w_4*(psi-1)` and
       `w_4*(psi-1)*(...)` denominators. These are NOT spurious — they
       are the unique solution to the bilinear system.
    2. **Fallback A (sequential elimination):** Solving for alphas first
       (linear in alpha, treating w as params) fails with 0 solutions
       because SymPy's `solve` doesn't detect the linear-in-alpha
       structure through the bilinear products. Using
       `linear_eq_to_matrix` to extract the linear system gives an
       overdetermined 5×2 system that is inconsistent for generic w.
       Picking 2 equations gives Cramer's-rule denominators with
       `psi*w_4*(psi-1)` — same issue.
    3. **Weights-first approach:** Solving for weights only (linear
       system) + compatibility conditions for alpha. The weight solutions
       have psi in denominator but NOT (psi-1). The compatibility
       conditions, after ψ-coefficient extraction, yield 11 constraints
       for 2 alpha unknowns — INCONSISTENT. This proves ψ-independent
       alpha solutions DO NOT EXIST for E4_1 with zeros={3,4}.
    4. **Evaluation check:** The old solver's alpha_0 genuinely varies
       with psi (e.g., alpha_0 ranges from ~51 at psi=0.1 to ~305 at
       psi=0.9 for specific parameter values). The psi-dependence is
       real, not removable.
  - **Structure of the solution (factored denominators):**
    - `alpha_0 den = 12*psi*w_4*(psi - 1)` — poles at psi=0 and psi=1
    - `alpha_1 den = w_4*(psi-1)*(12*psi^3 + 90*psi^2 + 648*psi + 288*w_4 - 197)` — pole at psi=1
    - `w_1, w_2, w_3 den = 1` — no poles (psi-independent!)
  - **Conclusion:** The conservation solve with zeros={3,4} and the
    current solve_for=[alpha_0, alpha_1, w_1, w_2, w_3] inherently
    produces psi-dependent alpha solutions with psi(psi-1) denominators.
    Eliminating these singularities requires changing the PROBLEM
    STRUCTURE, not the solver. Potential approaches for future work:
    - Use a different parameterization that avoids solving for alphas
      in conservation (match the paper's approach where α^u are
      user-specified constants, not solve targets)
    - Change the zeros configuration or nextra to allow constant-alpha
      solutions
    - Use a norm formulation with psi-dependent weights that absorbs
      the singularity
  - All 111 existing tests pass (1 xfail expected). No regressions.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "TestFractionFreeConservation" --timeout=300`

- [x] **27.3b** Validate the fraction-free conservation solve
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (new test class
    `TestFractionFreeConservation`)
  - Tests written (all 6 PASS):
    1. `test_weight_denominators_benign`: Weight solutions (w_1, w_2, w_3)
       have psi-free denominators (= 1). Weights are psi-independent.
    2. `test_alpha_denominators_documented`: Documents that alpha solutions
       have psi(psi-1) denominator factors — inherent to the bilinear
       structure, not a solver deficiency.
    3. `test_matches_old_solver`: Fraction-free solution matches the old
       `solve_cut_cell_conservation` numerically at several (psi, alpha_2,
       w_4) evaluation points.
    4. `test_conservation_holds`: All 5 conservation equations evaluate
       to 0 after substitution.
    5. `test_remaining_free_params`: Solution leaves exactly {psi, alpha_2,
       w_4} as free symbols in each solved expression.
    6. `test_solve_for_completeness`: Solution dict contains all 5
       solve_for symbols.
  - Also added `solve_conservation_fraction_free` to the test file's
    top-level imports.
  - All 111 existing tests pass (1 xfail expected). No regressions.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "TestFractionFreeConservation" --timeout=300`
  - Must come after 27.3a.

- [x] **27.3c** Decision gate: conservation solve approach after bilinearity finding
  - **Context:** 27.3a proved that solving conservation for both alpha AND
    weight unknowns (`solve_for=[alpha_0, alpha_1, w_1, w_2, w_3]`)
    inherently produces ψ(ψ-1) denominator factors in the alpha solutions.
    Fraction-clearing does not help — the ψ-dependence is real, not a
    solver artifact. After `xreplace(sol)`, boundary row entries (polynomial
    before conservation) reacquire ψ(ψ-1) denominators via the substituted
    alpha values.
  - **Investigation results (27.3c):** Tested all three approaches with
    POLYNOMIAL boundary rows (from `solve_temo_row_polynomial`). Key findings:

    **Finding 1 — Polynomial ansatz introduces extra free parameters:**
    The polynomial ansatz for boundary rows produces 12 extra free symbols
    `c_{i}_{6}_{k}` (i=0..3, k=2..4) from the underdetermined polynomial
    system for column 6 (endpoint-constrained). These propagate into columns
    2-5 of all boundary rows and appear in conservation equations 1-4.
    The conservation equations involve bilinear `w_k * c_{i}_6_{k}` products,
    making the system nonlinear in (w, c_*) jointly.

    **Finding 2 — Approach (A) is INFEASIBLE:**
    Weights-only conservation solve is INCONSISTENT for E4_1 with polynomial
    boundary rows, regardless of parameter settings:
    - With c_*=0, alpha=0: Rank(A)=4, Rank([A|b])=5 for 4 weights (INCONSISTENT)
    - With c_*=0, alpha=(1/10,1/5,1/3): same ranks (INCONSISTENT)
    - With c_*=0, alpha=(1/2,-1/4,3/7): same ranks (INCONSISTENT)
    - With c_*=0, alpha symbolic: same ranks (INCONSISTENT)
    - Variant 2 (3 weights): Rank(A)=3, Rank([A|b])=4 (INCONSISTENT)
    - Variant 3 (4 weights): Rank(A)=4, Rank([A|b])=5 (INCONSISTENT)
    Including c_* as solve targets does NOT help because the system is
    bilinear in w x c_* (e.g., `w_1 * c_1_6_2 * psi^2` terms in Eq 1).

    **Finding 3 — OLD stencil (rational boundary rows) also INCONSISTENT:**
    The weights-only solve is also inconsistent with rational boundary rows
    (from `solve_temo_row`): Rank(A)=4, Rank([A|b])=5 for 4 weights. This
    is not a polynomial-ansatz-specific issue — it is fundamental to E4_1
    with zeros={3,4}.

    **Finding 4 — Original bilinear approach works with polynomial rows:**
    `solve_for=[alpha_0, alpha_1, w_1, w_2, w_3]` with c_*=0 produces a
    unique solution. Alpha denominators contain ψ(ψ-1) factors (as expected
    from 27.3a). Weight solutions are psi-independent (w_1, w_2, w_3 have
    den=1). The denominator structure is:
    - `alpha_0 den = 6*psi*w_4*(psi-1)*(degree-6 poly in psi with w_4)`
    - `alpha_1 den = w_4*(psi-1)*(same degree-6 poly)`
    - `w_1, w_2, w_3 den = 1`
    Note: the degree-6 polynomial factor is new compared to the rational
    boundary row case (which had simpler denominators). This is because
    the polynomial boundary rows produce different conservation equations.

  - **Decision: (B) Accept ψ(ψ-1) denominators.**
    Approach (A) is ruled out: the 5-equation conservation system for E4_1
    with zeros={3,4} CANNOT be satisfied by weights alone — regardless of
    alpha values, c_* values, or boundary row construction method. The
    system fundamentally requires solving for at least 2 alpha parameters
    alongside the weights, which produces ψ(ψ-1) denominators.

    The polynomial ansatz for boundary rows (27.2a) remains valuable:
    boundary rows are polynomial PRE-conservation, giving cleaner
    intermediate expressions. After conservation xreplace, the final
    stencil reacquires ψ(ψ-1) denominators via alpha substitution.
    Runtime clamping (as in current E4_1.cpp) is still required.

    **Implication for 27.4a:** Use the original `solve_for=[alpha_0,
    alpha_1, w_1, w_2, w_3]` with c_*=0 (set the extra polynomial
    coefficients to zero). The polynomial ansatz gives simpler pre-
    conservation expressions but the final result has the same singularity
    structure. Change 1 (polynomial boundary rows) and Change 3
    (fraction-free solve) from 27.4a still apply. The `psi_eps`/`std::clamp`
    guards in E4_1.cpp remain necessary.
  - Must come after 27.3b. Must resolve BEFORE 27.4a proceeds.

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
    completely unchanged. The `xreplace(sol)` at line 2287 also stays.

    **27.3c resolved: approach (B).**  Change 3 (fraction-free solve) does
    NOT eliminate ψ(ψ-1) denominators — the alpha solutions inherently
    contain these factors regardless of solver.  The `solve_for` list
    remains `[alpha_0, alpha_1, w_1, w_2, w_3]` as before.  Change 3 is
    still worth applying for robustness (prevents accidental denominator
    blow-up in the solver) even though the final result is identical.
    The polynomial ansatz (Change 1) gives cleaner PRE-conservation
    expressions; the polynomial boundary rows need c_*=0 substitution
    before conservation (add `floating = floating.xreplace({c: 0 for c in
    c_syms})` after the row loop, where c_syms are the residual polynomial
    coefficients).  Runtime `psi_eps`/`std::clamp` guards remain necessary.

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
  - Must come after 27.3c (decision gate determines which changes apply).

- [ ] **27.4b** Validate the full E4_1 stencil
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (new test class
    `TestPolynomialFullStencil`)
  - **NOTE (27.3c resolved: approach B):** Boundary rows have ψ(ψ-1)
    denominators after alpha substitution (from conservation solve).
    Tests should document these as known limitations. The polynomial
    ansatz gives polynomial PRE-conservation entries, but the final
    stencil has the same singularity structure as before.
  - Tests:
    1. `test_all_entries_well_defined`: For all 5 rows × 7 cols, check
       denominators. Document which entries have ψ(ψ-1) poles (expected
       after alpha substitution from conservation solve — approach B).
    2. `test_taylor_accuracy_all_rows`: All 5 rows satisfy Taylor accuracy
       (reuse pattern from `TestE4CutCellSchemeWithZeros.test_taylor_accuracy`
       at line 1056).
    3. `test_conservation_column_sums`: Conservation holds for all columns
       (reuse pattern from `TestE4CutCellSchemeWithZeros.test_conservation_holds`
       at line 1076).
    4. `test_psi_0_limit`: At ψ→0, entries approach degenerate stencil values.
       Substitute ψ=0 and alpha values, verify entries are finite and match.
    5. `test_psi_1_limit`: At ψ=1, entries match uniform limit B_l(1).
    6. `test_free_parameter_count`: 2 free parameters (alpha_0, alpha_1
       in final naming, same as current — 27.3c chose approach B).
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

- [ ] **27.5a** Verify ψ-clamping and alpha guards in regenerated E4_1.cpp
  - **This is a post-27.6a verification step**, not a manual edit.
  - **27.3c chose approach (B):** The regenerated E4_1.cpp will STILL need
    `psi_eps`/`std::clamp` guards because ψ(ψ-1) denominators persist.
  - File: `src/stencils/E4_1.cpp` (after 27.6a copies the regenerated file)
  - Verify the following guards are PRESENT (manually re-add after regen
    if the codegen pipeline does not emit them):
    - `psi_eps` and `std::clamp` for ψ near 0 and 1
    - `alpha[1] < 197.0 / 288.0` constructor check (or equivalent bound
      for the new denominator structure)
  - **Note:** The codegen pipeline (`scripts/stencil_gen/stencil_gen/codegen.py`)
    does NOT hardcode any guards — the `psi_eps`/`std::clamp` and `alpha[1]`
    check in the current E4_1.cpp were added manually after code generation.
    After regeneration, these must be re-added manually.
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
