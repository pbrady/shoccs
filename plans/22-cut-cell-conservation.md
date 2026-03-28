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
The current TEMO pipeline solves each row of the cut-cell stencil B_l(ψ) independently for Taylor accuracy, then assembles the result. It does NOT enforce the discrete conservation constraint `Σ_i w_i · B[i,j] = 0` for interior columns. This affects BOTH E2_1 and E4_1: E2_1's existing tests only check columns 3,4 (where conservation holds trivially), but columns 0,1,2 have non-zero residuals with unit weights. E4_1 violates conservation on all columns with nonzero IC. Both schemes require non-trivial ψ-dependent weights and alpha constraints for full conservation (see CORRECTION in §E2_1 analysis below).

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
- **(A) Augmented consistency**: formulate conservation as requiring all 4×4 minors of [A(α)|b(α)] to vanish for all ψ; solve the resulting nonlinear α-system, then recover weights. **RESULT: FAILS** (22.3a-i). No α satisfies all minor conditions.
- **(B) Direct parametric solve**: parameterize w_i as rational functions of ψ with unknown coefficients, substitute into conservation, match ψ-coefficients, solve the resulting polynomial system in (weight coefficients, α). **RESULT: FAILS** (22.3a-ii). Mathematically equivalent to (A) — pointwise inconsistency at every tested (ψ, α) proves no weight function can satisfy conservation within the α-parameterized stencil space.
- **(C) Entry-level unknowns**: instead of working with α, treat the 8 free stencil entries directly as unknowns alongside weights, giving a system that may be solvable despite higher dimensionality. **RESULT: FAILS** (22.3a-iii). The conservation system is structurally infeasible in the full 8D (and 12D) entry space, for all weight parameterizations tested.

**All three approaches (A, B, C) have failed.** The conservation constraint for E4_1 (R=4, T=7, p=2, q=3, nu=1) appears to be fundamentally incompatible with the Taylor accuracy constraints at the current stencil dimensions. See 22.3a-iii results below for details. **Next step**: Investigate approach (D) — increase stencil dimensions (wider T or more boundary rows R) to provide additional degrees of freedom.

- [x] **22.3a-i** Investigate approach (A) — augmented matrix minor conditions:
  - Compute all C(6,4)=15 minor determinants of [A(ψ,α)|b(ψ,α)] for E4_1 symbolically (all 4 alpha free, not fixing alpha_3=0).
  - Extract ψ-coefficient equations from each minor.
  - Determine if the resulting α-system has any solution (check consistency).
  - If consistent, solve for α and verify the conservation system becomes consistent with those α values.
  - **Key question to answer**: does a valid (α₀,α₁,α₂,α₃) exist such that the conservation system A(ψ,α)w=b(ψ,α) is consistent for all ψ?
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (`TestApproachAMinorConditions`)
  - **RESULT: Approach (A) FAILS.** All 15 minors are nonzero. Extracting ψ-coefficients yields 40 polynomial equations in 4 alpha unknowns (39 unique). The system is heavily overdetermined and inconsistent:
    - From the 6 degree-1-in-ψ minors (rows involving 0,1,2): partial solution α₁=1/3, α₂=−4α₃−1/6.
    - Substituting into remaining equations yields alpha_3-only equations with contradictory values: −197/768, −11/48, −7/24, etc.
    - `sympy.solve()` on the full 39-equation system returns `[]` (no solution).
    - **Conclusion:** No choice of (α₀,α₁,α₂,α₃) can make the conservation system consistent for all ψ with constant weights. The bilinear coupling between weights and alpha truly requires ψ-dependent weights (approach B) or entry-level unknowns (approach C).

- [x] **22.3a-ii** Investigate approach (B) — parametric weight functions:
  - Parameterize weights as w_i = p_i(ψ)/q(ψ) where q is the common TEMO denominator (ψ+1)(ψ+2)(ψ+3) and p_i are polynomials in ψ of degree ≤ 3 with unknown rational coefficients.
  - Substitute into all 6 conservation equations, clear denominators.
  - Collect all ψ-coefficient equations.
  - Determine if the resulting system (in weight coefficients and alpha) is solvable.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (`TestApproachBParametricWeights`)
  - **RESULT: Approach (B) FAILS.** Three independent tests confirm this:
    1. **Common denominator verified:** All B_l entries share common psi-denominator 12·(ψ+1)(ψ+2)(ψ+3) (degree 3). Rows 0–2 each have 6·(ψ+1)(ψ+2)(ψ+3); row 3 has constant denom 12.
    2. **Pointwise inconsistency:** At every tested (ψ, α) combination (3 alpha choices × 3 psi values = 9 points), rank([A|b]) = 4 > rank(A) = 3. The system has no solution at those specific ψ values. No weight FUNCTION w(ψ) can produce a valid w(ψ₀) where no w exists.
    3. **Parametric coefficient system inconsistent:** For α=0, the psi-coefficient system in weight polynomial coefficients c is inconsistent at every tested degree (3, 5, 7). The rank gap is always exactly 1: rank([M|b]) = rank(M) + 1. The failure is fundamental, not a degree limitation.
    - **Mathematical proof of equivalence with approach A:** If approach B had a solution (c*, α*), then w(ψ) = p(ψ)/q(ψ) would satisfy A(ψ,α*)·w(ψ) = b(ψ,α*) for all ψ where q(ψ)≠0. This implies rank([A|b]) ≤ 3 at infinitely many ψ, forcing all 4×4 minors to vanish as polynomials — contradicting approach A's result. QED.
    - **Conclusion:** Both approaches A and B fail because the conservation system is structurally inconsistent within the α-parameterized stencil space. The 4 alpha parameters cannot simultaneously satisfy the 3 excess conservation constraints. Approach (C) — entry-level unknowns — is needed to break out of the α parameterization and work in the full 8-dimensional free-entry space.

- [x] **22.3a-iii** Investigate approach (C) — entry-level unknowns:
  - **Method:** Fix α=0 in B_u, prescribe only Category A (column 1), introduce 2 beta symbols per row via `solve_temo_row`. Gives B_l(ψ, e_{i,0}, e_{i,1}) with 8 constant betas. Build conservation equations, apply theta linearization (θ_{i,k} = w_i · e_{i,k} for i≥1), clear ψ-denominators, collect ψ-coefficients, solve scalar linear system.
  - **Results — persistent rank gap = 1 across ALL formulations:**
    1. **Constant weights + 8 betas (Category A):** 21 scalar equations in 11 unknowns. rank(A) = 7, rank([A|b]) = 8. Gap = 1.
    2. **Constant weights + 12 betas (no Category A):** 17 scalar equations in 15 unknowns. rank(A) = 9, rank([A|b]) = 10. Gap = 1.
    3. **Polynomial weights degree 1–3 + 8 betas (theta linearization):** Gap = 1 at all degrees. deg=1: 21 eqs / 20 unknowns, rank 10/11. deg=2: 26/29, rank 15/16. deg=3: 32/38, rank 20/21.
    4. **Pointwise check (specific ψ, random betas):** At ψ = 1/4, 1/2, 3/4 with 5 random beta vectors: conservation system H·w = b always has rank(H) = 3, rank([H|b]) = 4. No constant weights satisfy conservation for ANY beta choice.
    5. **Two-phase solve (solve 3 eqs for w, constrain betas from residuals):** Found beta constraints (e_{3,0}=1, e_{3,1}=−1/6, e_{1,0}=−2e_{2,0}−10, e_{1,1}=5/2−2e_{2,1}, with e_{0,*} and e_{2,*} free). BUT: at the constrained beta values, the weight solution is 0/0 (both numerator and denominator of w vanish identically). The full 6×3 system at the constrained betas has rank(A)=3, rank([A|b])=4 — still inconsistent.
    6. **ψ-dependent betas, constant weights:** At each ψ₀, the conservation equations become 6 linear equations in 8 betas. BUT rank = 2, augmented rank = 3 — only 2 of 8 betas affect the conservation sums, and the system is still inconsistent.
    7. **Nonlinear solve (bilinear system, no theta):** `sympy.solve` and `nonlinsolve` on the 21 quadratic scalar equations (constant w + constant e) both return 0 solutions.
  - **Root cause:** The Taylor accuracy constraints for each row create algebraic relationships between the stencil entries at different T-frame columns. These relationships force the 6×3 coefficient matrix H (= B_l[1:4, 0:6]^T) to have a left null space that is ALWAYS incompatible with the RHS vector (target − IC − ψ·B_l[0,:]). The left null space projection of b is a nonzero rational expression in (ψ, betas) that cannot be made zero by any beta choice. This rank gap = 1 is structural — it arises from the fixed ratio R=4, T=7, p=2 and the Taylor Vandermonde structure, not from the parameterization.
  - **Conclusion:** Approach (C) FAILS. The conservation constraint for E4_1 is **structurally infeasible** within the current stencil dimensions (R=4, T=7). No combination of Taylor-accurate stencil entries and quadrature weights (constant or ψ-dependent, of any polynomial degree) can satisfy all T−1=6 conservation column sums simultaneously.
  - **Implication for the plan:** All three identified approaches (A, B, C) have been exhaustively investigated and found infeasible. The conservation problem requires a fundamentally different strategy. Possible directions:
    - **(D) Increase stencil width T:** A wider stencil (T=8 or T=9) provides more columns and more free entries per row, potentially resolving the rank gap. This changes the interior contribution IC and the Vandermonde structure.
    - **(E) Increase boundary rows R:** More boundary rows (R=5) provide more weight unknowns and more entries, potentially making the conservation system solvable. This changes the near-interior handling.
    - **(F) Relax Taylor accuracy:** Use fewer Taylor constraints per row (lower q), freeing more DOF for conservation. This sacrifices accuracy order.
    - **(G) Accept approximate conservation:** Minimize the conservation residual instead of enforcing it exactly. This gives a "best effort" stencil that may be sufficient for practical convergence.
    - **(H) Column-selective conservation:** Enforce conservation only on a subset of columns (e.g., those with nonzero IC) and accept that other columns have small residuals.
  - File: investigatory work done in ad-hoc Python scripts (no test file committed — the investigation used interactive exploration rather than formal test classes)

- [ ] **22.3a-iv** *(Review follow-up)* Commit reproducible test evidence for approach (C):
  - Approaches A and B have formal test classes (`TestApproachAMinorConditions`, `TestApproachBParametricWeights`) in `test_e4_cut_cell.py` that allow independent verification of their infeasibility claims. Approach C has no committed test artifact — the 7 sub-results documented in 22.3a-iii were obtained via ad-hoc scripts.
  - Add `TestApproachCEntryLevelUnknowns` to `test_e4_cut_cell.py` reproducing at minimum: (1) the rank gap = 1 for constant weights + 8 betas (result #1), (2) the pointwise rank check at specific (ψ, beta) values (result #4), and (3) the nonlinear solve returning 0 solutions (result #7). These three cover the key structural claim from different angles.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

- [ ] **22.3a-v** *(Review follow-up)* Investigate approach (D) — increased stencil dimensions:
  - All three approaches (A, B, C) failed for E4_1 at R=4, T=7. The plan identifies directions D–H but has no concrete investigation item. This item gates all downstream work (22.3b–22.7a are BLOCKED until a viable approach is found).
  - Start with approach (D): increase T to 8 or 9 while keeping R=4. For each candidate T value:
    - Compute the new DOF budget: rows × (T − prescribed − Taylor_constraints) free entries vs. T−1 conservation equations and R−1 weight unknowns.
    - Build the conservation system symbolically (reuse `build_cut_cell_conservation_system` from 22.2a) and check the rank gap. If rank gap = 0, the dimension change resolves the infeasibility.
    - If T increase alone doesn't work, also try R=5 with the original T=7 (approach E).
  - Record findings and select the viable approach before unblocking 22.3b.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

> **NOTE (review of 173c879):** Items 22.3b through 22.7a below are **BLOCKED** — they assume conservation is solvable at the current E4_1 dimensions (R=4, T=7), which 22.3a-i/ii/iii proved infeasible. These items remain as-is for when a viable approach (from 22.3a-v) is identified; they will need revision to match the chosen approach's stencil dimensions and DOF structure.

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
  - **Temporarily broken tests:** After this item, `construct_cut_cell_stencil` enforces conservation by default, which changes the E4_1 stencil. The following tests in `test_e4_cut_cell.py` will fail until fixed by 22.4b/22.4c/22.4d: `test_e4_1_alpha_count` (line 659), `test_e4_1_matches_manual_pipeline` (line 665), `test_e4_1_custom_alphas` (line 707), `TestE4CodeGeneration.e4_spec` fixture (line 323), `TestE4TestFileGeneration.e4_spec` fixture (line 461). **E2_1 tests will also be affected** — conservation enforcement will constrain some E2_1 alpha parameters and produce non-trivial weights (see CORRECTION in E2_1 analysis, §22.5a). E2_1 tests that assert 4 free alpha symbols or compare against the pre-conservation stencil will need updates. E2_2 tests may be unaffected (TBD — depends on 22.6a results).

- [ ] **22.3c** Handle the quadrature weight output:
  - The conservation solve produces ψ-dependent quadrature weights w_0=ψ, w_1(ψ,α), ..., w_{R-1}(ψ,α)
  - **`CutCellResult` dataclass (line 1290):** Add `weight_solutions: dict | None = None` field (maps `w_i → expr(psi, alpha_remaining)`).
  - **`assemble_cut_cell_result()` (line 1905):** Accept `weight_solutions` parameter (added in 22.3b) and store in `CutCellResult.weight_solutions`.
  - **Test:** For E2_1, verify weight_solutions gives non-trivial ψ-dependent weights (NOT w_i=1 — the 22.3a investigation showed that full conservation for E2_1 requires non-trivial weights and alpha constraints; see CORRECTION in the E2_1 analysis). For E4_1, verify weights are rational functions of (ψ, α_remaining).
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

- [ ] **22.5a** Verify E2_1 conservation enforcement and update existing tests:
  - **CORRECTION (from 22.3a investigation):** E2_1's conservation residuals are NOT trivially zero for columns 0, 1, 2 when using unit weights. Verified: columns 0, 1, 2 have non-zero residuals with `w_i=1`; only column 3 has a zero residual. Full conservation for E2_1 requires non-trivial ψ-dependent weights AND alpha constraints, just like E4_1. The E2_1 stencil WILL change after conservation enforcement (some alpha parameters will be constrained).
  - **Existing tests that WILL need updates:**
    - `test_temo.py::TestE2_1Integration::test_conservation_symbolic` (line 1596) and `test_conservation_numeric` (line 1535) — these only check columns `j in [3, 4]` where conservation holds trivially. After conservation enforcement, extend these to verify ALL T−1=4 columns (j=0..3) using the solved weights from `result.weight_solutions`.
    - `test_e4_cut_cell.py::TestDeriveCutCellScheme::test_e2_1_reproduces_existing` (line 715) — this compares `derive_cut_cell_scheme(E2_1, psi)` against the manual pipeline. After conservation enforcement, the stencil changes (some alphas are constrained), so the manual path must also enforce conservation. Update `ur.alpha_symbols` to use `stencil.alpha_symbols` (post-conservation reduced list).
  - **New test to add:** `test_e2_1_weight_solutions` in `test_temo.py::TestE2_1Integration`:
    - Call `construct_cut_cell_stencil(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1, psi)` and verify `result.weight_solutions` is populated.
    - Verify weights are non-trivial ψ-dependent rational functions (NOT w_i=1).
    - Verify `result.alpha_symbols` has fewer than 4 alpha symbols (some constrained by conservation).
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
**ORIGINALLY CHOSEN: (a) Treat alphas as parameters, solve for w's.** **STATUS: DEAD END** — the conservation system A(ψ,α)·w = b(ψ,α) is structurally inconsistent within the α-parameterized stencil space. Both approach A (minor conditions) and approach B (parametric weights) confirm this independently. The α parameterization constrains the stencil to a 4-dimensional manifold that does not intersect the conservation constraint surface. **Approach (C) also fails:** the full 8D entry space (bypassing α) has the same structural rank gap = 1 for the conservation system. The infeasibility is inherent to the stencil dimensions (R=4, T=7), not the parameterization. **Resolution pending:** requires changing stencil dimensions or accepting approximate conservation.

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
- **Approach A (minors):** 15 minor determinants → 40 polynomial equations in 4 α unknowns. System is inconsistent (no solution).
- **Approach B (parametric weights):** 37–55 psi-coefficient equations in 12–24 weight coefficients + 4 α. System is inconsistent for all tested α (rank gap = 1). The failure is independent of weight polynomial degree.
- **Approach C (entry-level unknowns):** Investigated in 22.3a-iii. The theta-linearized system has 21–32 scalar equations in 11–38 unknowns (depending on weight polynomial degree). All formulations show rank gap = 1 — the system is structurally infeasible. The nonlinear (bilinear) system was also tested via SymPy's `solve` and `nonlinsolve` and confirmed infeasible.
- **Status: BLOCKED** — all three approaches (A, B, C) fail for E4_1 at current dimensions (R=4, T=7). Conservation requires increasing stencil dimensions or accepting approximate enforcement.

## Key Implementation Insight

The α parameterization (free parameters from the per-row Taylor solve) is a 4-dimensional manifold in an 8-dimensional space of free stencil entries. Approaches A and B proved that this manifold does not intersect the conservation constraint surface. Approach C extended the investigation to the full 8D (and 12D) entry space and proved that the conservation constraint surface does not intersect the Taylor accuracy manifold AT ALL for the current E4_1 dimensions (R=4, T=7).

**The structural infeasibility** arises because the Taylor Vandermonde constraints for each row create algebraic dependencies between stencil entries at different columns. These dependencies force the 6×3 conservation coefficient matrix (from rows 1–3 across the 6 conservation columns) to have a left null space whose projection onto the RHS is always nonzero. The rank gap = 1 is invariant across all parameterizations: constant weights, polynomial weights of any degree, ψ-dependent betas, and removal of the Category A prescription.

**Resolving conservation for E4_1 requires changing the stencil dimensions** — either increasing T (wider stencil), increasing R (more boundary rows), or accepting approximate conservation. This is a Phase 22+ decision that affects the stencil structure and C++ implementation.
