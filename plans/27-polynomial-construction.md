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

## Dependency Graph

```
27.1a → 27.1b → 27.1c (decision gate)
                  ├── boundary rows polynomial after cancel → skip 27.2, go to 27.3a
                  └── need polynomial ansatz → 27.2a → 27.2b ─┐
                                                               ├→ 27.3a → 27.3b → 27.4a → 27.4b ─┬→ 27.6a → 27.5a
                                                               │                                    └→ 27.7a
```

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
  - **Implementation detail:** Use the existing fixture pattern from the file.
    Call `derive_uniform_boundary_for_temo(E4_1, zeros=set(E4_1.zeros))` to
    get the uniform result, then `construct_cut_cell_stencil(uniform.B_u,
    uniform.interior, 2, 3, 1, 0, psi)`. For each entry in rows 0-3, do:
    ```python
    num, den = fraction(cancel(entry))
    assert not den.has(psi), f"Row {i} col {j} has psi-dependent denominator"
    ```
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

- [ ] **27.1c** Decision gate: polynomial ansatz needed?
  - **After 27.1a/27.1b pass or fail**, record the outcome as a comment in
    this plan file under this item.
  - **If boundary rows ARE polynomial after cancel:** The existing `solve_temo_row`
    with QQ(ψ) produces correct polynomial entries. Skip 27.2a entirely —
    proceed directly to 27.3a (the conservation approach is the actual problem).
    In this case, 27.3a should use the existing `construct_cut_cell_stencil`
    output for boundary rows and only restructure the near-interior solve.
  - **If boundary rows are NOT polynomial after cancel:** The polynomial ansatz
    (27.2a) is required. Proceed with 27.2a.
  - **Decision outcome:** _(to be filled in after 27.1a/27.1b run)_

### 27.2 — Implement polynomial boundary row solve

- [ ] **27.2a** Add `solve_temo_row_polynomial` function
  - **Conditional:** Only needed if 27.1c determines polynomial ansatz is required.
    If boundary rows from QQ(ψ) solve already simplify to polynomials, skip this.
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
  - **Concrete SymPy implementation for steps 3-6:**
    ```python
    # Step 2: Create polynomial unknowns for each free column
    c_syms = {}  # {col_j: [c_j_0, c_j_1, ..., c_j_dmax]}
    all_c = []
    for j in free_cols:
        c_j = [Symbol(f"c_{i}_{j}_{k}") for k in range(d_max + 1)]
        c_syms[j] = c_j
        all_c.extend(c_j)

    # Build polynomial expressions for free columns
    poly_unknowns = {}
    for j, c_j in c_syms.items():
        poly_unknowns[j] = sum(c_j[k] * psi**k for k in range(d_max + 1))

    # Step 3: Form residuals  V * x - rhs  as polynomials in psi
    residuals = []
    for eq_idx in range(n_eqs):
        expr = -rhs[eq_idx, 0]
        for j in range(T):
            if j in prescribed:
                expr += V[eq_idx, j] * prescribed[j]
            else:
                expr += V[eq_idx, j] * poly_unknowns[j]
        residuals.append(expand(expr))

    # Step 4-5: Collect by psi powers -> coefficient equations
    equations = []
    for res in residuals:
        p_res = Poly(res, psi)
        for coeff in p_res.all_coeffs():
            equations.append(coeff)  # each must be zero

    # Step 6: Solve for c_{j,k} symbols (treating alpha_syms as parameters)
    sol = solve(equations, all_c, dict=True)
    ```
    Note: `V[eq_idx, j]` contains `(-(psi+i))^k / k!` in col 0, so the
    product `V[eq_idx, j] * poly_unknowns[j]` is a polynomial in ψ of
    degree up to `q + d_max`. After `expand`, each residual is a polynomial
    in ψ whose coefficients are linear in `c_{j,k}` and `alpha_syms`.
  - **Key difference from `solve_temo_row`:** No QQ(ψ) fraction field. The ψ
    variable is expanded out, and we solve a larger but purely rational system.
    For E4_1 row 0: n_eqs=4, T=7, 1 prescribed col → 5 free cols × (d_max+1)=5
    coefficients = 25 unknowns from ~4×(4+4+1)≈36 coefficient equations.
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
    `solve_temo_row_polynomial` or after `solve_cut_cell_conservation` at
    ~line 1545)
  - Signature: `solve_near_interior_with_conservation(boundary_rows: Matrix, B_u, interior, p, q, nu, psi, alpha_syms, zeros) -> tuple[list, list, Expr]`
    - Returns: `(row_coeffs, weights, denominator_e0)` where row_coeffs are
      rational in ψ with common denominator e0
  - **Algorithm:**
    1. Start with the polynomial boundary rows (from 27.2 or from existing
       `construct_cut_cell_stencil` if 27.1c found them already polynomial)
       as a partial R×T matrix (rows 0..r-1 filled, row r unknown).
    2. For row r (near-interior), prescribe the zeroed column:
       `α_{r,1}(ψ) = ψ * B_l_1[r,1]` (Category A, same as current code).
       Reuse `identify_prescribed_entries(r, r, ...)` to get this.
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
  - **Concrete implementation for steps 3-6 (the combined system):**
    For E4_1: R=5, T=7, r=4 (row index of near-interior), n_eqs=4 (q+1),
    row r has 1 prescribed col (col 1, zeroed) → 6 free cols.
    Conservation gives T-2=5 equations (cols 1..5 of the T-frame).
    Weight unknowns: w_1..w_4 (4 symbols; w_0=ψ is fixed).
    ```python
    # Step 3: Taylor for row r
    V_r, rhs_r = build_temo_vandermonde(r, T, q, nu, psi)
    prescribed_r = identify_prescribed_entries(r, ...)
    # Move prescribed to RHS
    taylor_rhs = rhs_r.copy()
    for j, val in prescribed_r.items():
        for k in range(n_eqs):
            taylor_rhs[k, 0] -= V_r[k, j] * val
    free_cols_r = [j for j in range(T) if j not in prescribed_r]
    # Taylor unknowns: row_r_free = [x_j for j in free_cols_r]

    # Step 4: Conservation (reuse existing function on partial matrix)
    # Build partial matrix: rows 0..r-1 = boundary_rows, row r = symbolic
    row_r_syms = [Symbol(f"xr_{j}") for j in range(T)]
    for j, val in prescribed_r.items():
        row_r_syms[j] = val  # prescribed entries are known
    partial = boundary_rows.copy()
    partial = partial.row_join(Matrix([row_r_syms]))  # or build R×T
    # Now build conservation equations using the row_r symbols
    eqs_cons, w_syms = build_cut_cell_conservation_system(
        partial_matrix, R, T, p, nu, interior, psi
    )

    # Step 5: Combine into one system
    # Taylor: n_eqs equations in free_cols_r unknowns (6 for E4_1)
    # Conservation: 5 equations in xr_free + w_1..w_4 (6 + 4 = 10)
    # Total: 9 equations, 10 unknowns → 1 remaining free parameter
    # Plus alpha_syms are free parameters from boundary rows.
    all_unknowns = [row_r_syms[j] for j in free_cols_r] + list(w_syms)

    # Step 6: Solve combined system
    all_eqs = []
    for k in range(n_eqs):
        eq = sum(V_r[k, j] * row_r_syms[j] for j in free_cols_r) - taylor_rhs[k, 0]
        all_eqs.append(eq)
    all_eqs.extend(eqs_cons)
    sol = solve(all_eqs, all_unknowns, dict=True)
    ```
    **Dimension check for E4_1:** 4 Taylor eqs + 5 conservation eqs = 9 eqs;
    6 free row-r entries + 4 weight unknowns = 10 unknowns. This leaves 1
    free parameter (a weight, which becomes alpha_1 after renaming). Combined
    with the alpha from boundary rows, we get 2 total free params as expected.
  - **Key difference from current approach:** Conservation and Taylor for
    row r are solved SIMULTANEOUSLY, not sequentially. This avoids the
    ψ*(ψ-1) denominator that arises from sequential conservation substitution.
    The current code (lines 2266-2312 of temo.py) does:
    `construct_cut_cell_stencil` (all R rows via QQ(ψ)) →
    `build_cut_cell_conservation_system` → `solve_cut_cell_conservation` →
    `xreplace`. The new approach only solves rows 0..r-1 first, then row r
    simultaneously with conservation.
  - **Reuse from existing code:**
    - `build_temo_vandermonde` — reused directly for row r's Taylor system
    - `identify_prescribed_entries` — reused for row r's prescribed cols
    - `build_cut_cell_conservation_system` — can be reused if we pass a
      partial matrix with symbolic row-r entries. Alternatively, build the
      conservation equations inline (the function is ~30 lines).
    - `solve_cut_cell_conservation` — NOT reused (it solves conservation
      alone; we need the combined solve)
  - **Size estimate:** ~100-130 lines. The current `build_cut_cell_conservation_system`
    and `solve_cut_cell_conservation` can be partially reused, but the
    simultaneous solve requires restructuring.
  - Must come after 27.1c decision. If 27.1c says "boundary rows already
    polynomial", use existing `construct_cut_cell_stencil` output for rows
    0..r-1. If 27.1c says "polynomial ansatz needed", must come after 27.2b
    (boundary rows validated before using them as near-interior input).

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
  - Add a new function `construct_cut_cell_stencil_polynomial(B_u, interior, p, q, nu, nextra, psi, zeros) -> CutCellResult`
    that:
    1. Computes B_l_1 (uniform limit) and B_d (degenerate) — same as current
       `construct_cut_cell_stencil` lines 1404-1405
    2. Solves rows 0..r-1 using either `solve_temo_row_polynomial` (if 27.1c
       required it) or the existing `solve_temo_row` (if boundary rows are
       already polynomial after cancel)
    3. Calls `solve_near_interior_with_conservation` for row r
    4. Assembles the complete 5×7 stencil matrix
    5. Handles alpha renaming (maps internal alpha_syms + free weight to
       final `alpha_0`, `alpha_1`)
    6. Returns a `CutCellResult` via `assemble_cut_cell_result`
  - **Modify `derive_cut_cell_scheme`** (the `scheme.zeros` branch at lines
    2267-2312 of temo.py). Replace the current 7-step sequence:
    ```python
    # Current (lines 2267-2312):
    if scheme.zeros:
        uniform = derive_uniform_boundary_for_temo(scheme, zeros=set(scheme.zeros))
        floating_result = construct_cut_cell_stencil(...)    # all R rows
        eqs, w_syms = build_cut_cell_conservation_system(...)
        solve_for = list(uniform.alpha_symbols[:2]) + list(w_syms[:3])
        sol = solve_cut_cell_conservation(eqs, solve_for)
        floating = floating.xreplace(sol)                    # <-- introduces ψ(ψ-1) denoms
        weights = [psi] + [sol[w] for w in w_syms[:3]] + [w_syms[3]]
        # ... renaming ...
    ```
    With:
    ```python
    # New:
    if scheme.zeros:
        uniform = derive_uniform_boundary_for_temo(scheme, zeros=set(scheme.zeros))
        return construct_cut_cell_stencil_polynomial(
            uniform.B_u, uniform.interior,
            scheme.p, scheme.q, scheme.nu, scheme.nextra, psi,
            zeros=scheme.zeros, alpha_symbols=alpha_symbols,
        )
    ```
    The new function encapsulates all the conservation logic internally
    (no external xreplace step).
  - The non-zeros paths (E2_1, E2_2, generic conservative at lines 2314-2433)
    remain completely unchanged.
  - **Imports:** Add `construct_cut_cell_stencil_polynomial` to the test
    file's import block (line 32 of `test_e4_cut_cell.py`).
  - **Size estimate:** ~60-80 lines for the new function, ~15 lines to
    simplify the `scheme.zeros` branch in `derive_cut_cell_scheme`.
  - Must come after 27.3b (near-interior row validated before wiring into
    full construction).

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

- [ ] **27.5a** Verify ψ-clamping and alpha guards are absent from regenerated E4_1.cpp
  - **This is a post-27.6a verification step**, not a manual edit. The
    regenerated E4_1.cpp should already be free of singularity guards because
    the polynomial construction eliminates the root cause.
  - File: `src/stencils/E4_1.cpp` (after 27.6a copies the regenerated file)
  - Verify the following are ABSENT:
    - `psi_eps` and `std::clamp` (currently lines 98-99)
    - `alpha[1] < 197.0 / 288.0` constructor check (currently lines 35-38)
    - The singularity-explanation comment block (currently lines 17-28)
    - Any `1.0/psi` or `1.0/(psi - 1)` patterns in floating/dirichlet methods
  - **If any guards remain** in the codegen output (e.g., because the codegen
    template has hardcoded guards), remove them from the codegen template in
    `scripts/stencil_gen/stencil_gen/codegen.py` and re-run 27.6a.
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
