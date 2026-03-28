# Phase 21: E4 Cut-Cell Stencil Generation

**Goal:** Generate working E4_1 (1st derivative, 4th order) cut-cell stencil C++ code using the SymPy pipeline, then validate it. This requires generalizing the TEMO pipeline from E2-only to arbitrary order by connecting the general boundary solver (`boundary.py`) to the TEMO construction (`temo.py`).

**Depends on:** Phase 20 (complete — all modules implemented)

**Read first:**
- `scripts/stencil_gen/stencil_gen/temo.py` (current TEMO pipeline — hardcoded to E2 via `derive_e2_uniform_boundary`)
- `scripts/stencil_gen/stencil_gen/boundary.py` (general boundary solver — handles arbitrary p, already validated for E4u/E6u/E8u)
- `scripts/stencil_gen/stencil_gen/conservation.py` (SBP conservation solver)
- `scripts/stencil_gen/stencil_gen/codegen.py` (C++ code generation)
- `plans/stencil-derivation-math-reference.md` (Section 4: TEMO construction procedure)
- `src/stencils/E4u_1.cpp` (existing uniform E4 stencil — the uniform base for E4_1 cut-cell)
- `src/stencils/E2_1.cpp` (existing E2 cut-cell — structural reference for what E4_1 will look like)
- `src/stencils/stencil.hpp` (Stencil concept and factory declarations)

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/ -v
cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v
```

---

## Current State

The SymPy pipeline (Phase 20) is complete with 231 passing tests and 7 modules:
- `interior.py` — derives interior coefficients for any (s, p, nu)
- `boundary.py` — derives uniform boundary stencils for any (p, nu, s=0), validated for E4u, E6u, E8u
- `conservation.py` — solves SBP conservation constraints
- `temo.py` — constructs cut-cell B_l(psi) stencils, BUT only for E2 (p=1)
- `codegen.py` — generates C++ structs and test files
- `printer.py` — SymPy-to-C++ expression printer

**The gap:** `temo.py` uses `derive_e2_uniform_boundary()` which is hardcoded for p=1, q=1. The general `boundary.py`'s `solve_boundary_row()` handles arbitrary (i, t, q, nu) but the TEMO pipeline doesn't use it. The key work is:
1. Fix `compute_dimensions` (D-R25: `r = p+1+nextra`, not `q+1+nextra`)
2. Build a general `derive_uniform_boundary_for_temo` using `solve_boundary_row` with TEMO dimensions (r=3, t=6 for E4 vs r=3, t=4 for E2)
3. Handle E4's zero-constrained alpha entries via alpha distribution convention
4. Run the existing TEMO pipeline with the wider E4 boundary and validate
5. Generate, validate, and possibly register the E4_1 C++ stencil

**E4_1 scheme parameters (from Table 1 of Brady & Livescu 2021):**
- p=2, q=3, s=0, nextra=0, nu=1
- Uniform base: r=3 rows, t=6 columns (from p+1+nextra=3, p+q+1+nextra=6)
- Cut-cell: R=4 rows, T=7 columns (r+1 × t+1)
- Free parameters: alpha^u_{04}, alpha^u_{14}, alpha^u_{24}, alpha^u_{25}
- Zero-constrained: alpha^u_{05}=0, alpha^u_{15}=0
- **Note:** The Eq. 11b formula `r = q+1+nextra` gives r=4 (wrong). The correct formula is `r = p+1+nextra = 3`. See D-R25 in `plans/meta.md`.

---

## Items

### 21.0 — Fix `compute_dimensions` for E4 (prerequisite)

- [x] **21.0a** Fix `compute_dimensions()` in `scripts/stencil_gen/stencil_gen/temo.py`:
  - The function already takes `p` as its first parameter (signature: `compute_dimensions(p, q, s, nextra, nu)`)
  - Change the code on line 59: replace `r = q + 1 + nextra` with `r = p + 1 + nextra` (per D-R25)
  - The column formula `t = p + q + 1 + nextra` on line 58 is correct and unchanged
  - Update the docstring formula on line 29 to say "r = p + 1 + nextra" instead of "r = q + 1 + nextra"
  - Remove the "Note: verified for E2 schemes only" caveat (now verified for E4 too)
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (lines 24–72)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "Dimensions"`
  - Expected dimension changes:
    - E2_1 (p=1, q=1, nextra=1): **unchanged** (r=3, t=4, R=4, T=5, X=0) because p+1+nextra = q+1+nextra = 3
    - E2_2 (p=1, q=1, nextra=0): **unchanged** (r=2, t=3, R=2, T=4, X=2) because p+1+nextra = q+1+nextra = 2
    - E4_1 (p=2, q=3, nextra=0): r changes 4→3, R changes 5→4 → **(r=3, t=6, R=4, T=7, X=0)**
    - E4_2 (p=2, q=3, nextra=0): r changes 4→3, R changes 4→3, X changes 4→3 → **(r=3, t=6, R=3, T=7, X=3)**
  - **Must update test assertions**: `test_first_derivative_no_neumann` uses `E4_1.dims().X == 0` (no change). `test_second_derivative_has_neumann` checks `dims_e4_2.X == dims_e4_2.R` — both change from 4 to 3, so the equality still holds. No test changes needed unless there are explicit E4 dimension value checks elsewhere.

- [x] **21.0b** Add explicit E4 dimension value tests (review finding — the 21.0a fix is untested for E4):
  - The existing E4 tests only check relational properties (`X==0`, `X==R`) that hold regardless of whether `r = q+1+nextra` or `r = p+1+nextra`. No test verifies the E4 dimension values actually changed. E2 schemes have exact `Dimensions` tuple assertions (test_temo.py lines 48, 53); E4 needs the same.
  - Add to `tests/test_temo.py::TestDimensions`:
    - `test_e4_1_dimensions`: `assert E4_1.dims() == Dimensions(r=3, t=6, R=4, T=7, X=0)`
    - `test_e4_2_dimensions`: `assert E4_2.dims() == Dimensions(r=3, t=6, R=3, T=7, X=3)`
  - File: `scripts/stencil_gen/tests/test_temo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "e4" `

### 21.1 — Bridge boundary.py → temo.py UniformResult

- [x] **21.1a** Add a general `derive_uniform_boundary_for_temo(scheme: SchemeParams, alpha_symbols=None)` function in `temo.py` that:
  1. Computes TEMO dimensions using the **corrected** `compute_dimensions` (21.0a): r = p+1+nextra, t = p+q+1+nextra
  2. For nu=1: r_eff = r; for nu=2: r_eff = r - 1
  3. Computes `n_free_per_row = t - (q + 1)` (= 2 for E4_1)
  4. Creates alpha symbols using this concrete algorithm:
     ```
     Alpha Distribution Algorithm (depends on nextra):
     n_free = t - (q + 1)
     alpha_idx = 0

     Case nextra == 0 (no conservation row — E4_1, E4_2):
       For i = 0..(r_eff - 2):   # early rows
         active = 1
         free[i] = [alpha_{alpha_idx}] + [S.Zero] * (n_free - 1)
         alpha_idx += 1
       For i = r_eff - 1:        # last row
         active = min(n_free, 2)
         free[i] = [alpha_{alpha_idx}..alpha_{alpha_idx+active-1}]
                   + [S.Zero] * (n_free - active)
         alpha_idx += active

     Case nextra > 0 (conservation-constrained last row — E2_1):
       For i = 0..(r_eff - 2):   # early rows
         free[i] = [alpha_{alpha_idx}..alpha_{alpha_idx+n_free-1}]
         alpha_idx += n_free
       For i = r_eff - 1:        # last row — placeholder for conservation
         free[i] = [phi_0..phi_{n_free-1}]   (temporary symbols)
     ```
     Concrete results:
     - **E4_1** (nextra=0, n_free=2, r_eff=3): `[[alpha_0, 0], [alpha_1, 0], [alpha_2, alpha_3]]` → 4 alphas
     - **E2_1** (nextra=1, n_free=2, r_eff=3): `[[alpha_0, alpha_1], [alpha_2, alpha_3], [phi_0, phi_1]]` → 4 alphas + 2 phi (resolved by conservation)
  5. Calls `boundary.solve_boundary_row(i, t, q, nu, free_symbols[i])` for each row i = 0..r_eff-1.
     - **Note:** `solve_boundary_row` builds a Taylor system with `q+1` equations (via `build_taylor_system`). For E4_1 (q=3, nu=1), `q+1 = 4 = max(q+1, nu+1)`, so this matches the existing `_build_uniform_vandermonde` equation count. For nu=2 schemes where `q+1 < nu+1`, this will produce fewer equations; see 21.5a for how to handle that case.
  6. **Conservation step (nextra > 0 only):**
     - If nextra > 0: substitute phi symbols using column-sum conservation. For each interior column `j` in range `[p+1, t)` (= columns `[q+1, t)` in the `solve_boundary_row` output since `n_det = q+1`):
       ```
       For j in range(p+1, t):
         col_sum_upper = sum(B_u[i, j] for i in range(r_eff - 1))
         phi_idx = j - (q + 1)   # maps column j to phi symbol index
         subs[phi[phi_idx]] = -col_sum_upper
       B_u = B_u.subs(subs)
       ```
       This resolves all phi symbols in the last row, leaving only the alpha symbols from earlier rows.
       - For **E2_1** (p=1, t=4, q=1): columns j=2,3 are conserved → 2 phi symbols resolved, matching `derive_e2_uniform_boundary` lines 248–254.
       - For **E4_1** (nextra=0): this step is skipped entirely — no phi symbols exist.
     - If nextra == 0: no conservation step — all rows retain their free alphas.
  7. Gets interior coefficients via `interior.derive_interior(s, p, nu)` + `full_gamma_array()`
  8. Assembles the r_eff × t Matrix and packages into `UniformResult`
  - **Key insight:** boundary.py's `derive_boundary(p, nu)` uses `r = 2p-1` (SBP minimal rows) which gives r=3, t=5 for E4. The TEMO needs t=6 (wider). But `solve_boundary_row(i, t, q, nu, free)` accepts arbitrary t, so call it with the TEMO t=6. This is NOT a format conversion from `BoundaryResult` — it's a fresh derivation using the lower-level `solve_boundary_row` function.
  - **Relationship to E4u_1:** The TEMO 3×6 boundary extends the uniform 3×5 boundary (E4u_1) by one column. Row 0 and Row 1's first 5 coefficients match E4u_1 exactly (column 5 is zero). Row 2 has 2 free alphas (alpha_2, alpha_3) instead of being conservation-constrained like E4u_1's row 2.
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - **Regression test:** verify `derive_uniform_boundary_for_temo(E2_1)` produces the same `UniformResult` as the existing `derive_e2_uniform_boundary(nu=1)`:
    - Both should use alpha symbols `alpha_0, alpha_1, alpha_2, alpha_3` (in that order)
    - Compare `B_u` matrices entry-by-entry via `cancel(new[i,j] - old[i,j]) == 0`
    - Compare `interior` lists exactly
    - If the new function produces differently named alphas (e.g., from `solve_boundary_row`'s internal naming), add a symbol-remapping step before constructing `UniformResult`
    - The new function should accept an optional `alpha_symbols` parameter (like `derive_e2_uniform_boundary` does) to allow the caller to control symbol naming

- [x] **21.1b** Test the bridge for E4_1:
  - Call `derive_uniform_boundary_for_temo(E4_1)`
  - Verify: B_u shape is (3, 6) — r_eff=3 rows, t=6 columns
  - Verify: 4 free alpha symbols (matching Table 1: alpha^u_{04}, alpha^u_{14}, alpha^u_{24}, alpha^u_{25})
  - Verify: B_u[0, 5] == 0 and B_u[1, 5] == 0 (zero constraints from alpha distribution)
  - Verify: B_u[2, 4] and B_u[2, 5] are the free alpha_2, alpha_3 symbols
  - Verify: first 5 columns of rows 0,1 match E4u_1.cpp's nbs_floating coefficients symbolically (at alpha_0, alpha_1)
  - Verify: interior coefficients are [1/12, -2/3, 0, 2/3, -1/12]
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (new file)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "uniform"`

### 21.2 — Handle E4 zero-constrained entries

- [ ] **21.2a** Verify zero constraints are handled by the alpha distribution in 21.1a:
  - The zero constraints (alpha^u_{05}=0, alpha^u_{15}=0) are automatically handled by the alpha distribution convention in `derive_uniform_boundary_for_temo`: rows 0 and 1 use `free_symbols = [alpha_k, S.Zero]`, placing zero in the last free column (position 5).
  - This item is a **verification-only** step, not an implementation step (the work is done in 21.1a).
  - Test: assert `B_u[0, 5] == 0` and `B_u[1, 5] == 0` as part of 21.1b tests.
  - If the alpha distribution convention doesn't produce these zeros (unlikely), add explicit `S.Zero` entries in 21.1a's free_symbols lists.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "zero"`

### 21.3 — E4_1 TEMO construction

- [ ] **21.3a** Run the full TEMO pipeline for E4_1:
  - Get `B_u` (3×6) and `interior` from `derive_uniform_boundary_for_temo(E4_1)`
  - Call `construct_cut_cell_stencil(B_u, interior, p=2, q=3, nu=1, nextra=0, psi)`
  - **Corrected dimensions:** output is a **4×7** matrix (R=4 rows, T=7 columns), not 5×7
  - Verify: output matrix has shape (4, 7), entries are rational functions in psi and alpha_{0..3}
  - Verify: at psi=1, rows 0–2 reduce to B_u embedded in T-frame (wall=0, cols 1–6 = B_u), and row 3 (near-interior) is the interior stencil [0, 0, 1/12, -2/3, 0, 2/3, -1/12] (centered at x_3; cols 0,1 zero because wall and x_0 are outside the stencil reach)
  - Verify: at psi=0, satisfies degenerate constraints (DP1: B_d[i, j+1] = B_u[i, j] for j≥1; DP2: B_u[i,0] assigned to wall, x_0 zeroed for nu=1)
  - Verify: Taylor accuracy holds for each row at a sample psi (e.g., psi=1/2): evaluate the stencil and check it reproduces f'(x_i) for polynomials of degree ≤ q=3
  - Verify: no beta symbols in the result (nextra=0 should prescribe all excess columns via limit interpolation, eliminating betas)
  - **Overdetermined near-interior row analysis** — both `build_degenerate_stencil` and `solve_uniform_limit` produce overdetermined systems for row 3 (3 unknowns, 4 equations). The code handles this by solving the first 3 equations and verifying the 4th for consistency. Concrete column categorization:
    ```
    build_degenerate_stencil (psi=0 limit, row 3):
      deltas = [-3, -3, -2, -1, 0, 1, 2]
      zeroed_col = {1}              (x_0 zeroed, nu=1)
      conservation cols = {4, 5, 6} (j in range(p+2=4, T=7))
      known = {1, 4, 5, 6}
      unknown = {0, 2, 3}           → 3 unknowns, 4 equations

    solve_uniform_limit (psi=1 limit, row 3):
      deltas = [-4, -3, -2, -1, 0, 1, 2]
      conservation cols: j in [2,7) where j<=R-p=2 or j>=p+2=4
        j=2 ✓(≤2), j=3 ✗, j=4 ✓(≥4), j=5 ✓, j=6 ✓
      fixed = {2, 4, 5, 6}
      unknown = {0, 1, 3}           → 3 unknowns, 4 equations

    identify_prescribed_entries (general psi, all 4 rows):
      For ALL rows (nu=1): zeroed_col = 1 (x_0), n_eqs = max(q+1,nu+1) = 4

      Row 0 (boundary, i=0):
        prescribed[1] = psi * B_u[0, 0]           (Category A)
        free_cols = [0, 2, 3, 4, 5, 6]  → n_free=6, n_excess = 6-4 = 2
        extra_cols = [5, 6]  (highest 2)
        prescribed[5] = psi*B_l_1[0,5] + (1-psi)*B_d[0,5]
        prescribed[6] = psi*B_l_1[0,6] + (1-psi)*B_d[0,6]
        solve_cols = [0, 2, 3, 4]  → 4 unknowns = 4 equations ✓

      Rows 1, 2 (boundary, i=1,2): same structure as row 0
        prescribed = {1, 5, 6}, solve_cols = [0, 2, 3, 4] → exactly determined

      Row 3 (near-interior, i=3):
        prescribed[1] = psi * B_l_1[3, 1]          (Category A, target from B_l_1)
        free_cols = [0, 2, 3, 4, 5, 6]
        extra_cols = [5, 6]
        prescribed[5] = psi*B_l_1[3,5] + (1-psi)*B_d[3,5]  (psi-dependent)
        prescribed[6] = psi*B_l_1[3,6] + (1-psi)*B_d[3,6]  (psi-dependent)
        solve_cols = [0, 2, 3, 4]  → 4 unknowns = 4 equations ✓

      Summary: all rows exactly determined, no betas introduced
    ```
    The overdetermined systems are expected to be consistent because the conservation values are derived from boundary rows that satisfy the Taylor accuracy conditions. If either raises `RuntimeError("Overdetermined system inconsistent...")`, the alpha distribution or conservation logic in 21.1a needs debugging.
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "temo"`

- [ ] **21.3b** Performance check:
  - The full E4_1 derivation (uniform boundary + TEMO + conservation) should complete in < 10 seconds
  - E4_1 has 28 floating coefficients (4×7), each a rational function of psi and 4 alphas — larger than E2_1 (4×5=20 coefficients with 4 alphas)
  - If slow (>10s), profile and apply QQ(psi) field optimizations from 20.5e
  - If the `solve_in_field` calls are the bottleneck, the E4 system has n_eqs=4 per row (vs E2's n_eqs=2), so the Vandermonde solves are 4× larger
  - Test: add a timing assertion or measure with `time.time()`

### 21.4 — E4_1 C++ code generation

- [ ] **21.4a** Fix codegen constructor/factory for non-uniform single-param-array stencils (prerequisite):
  - **Bug:** `codegen.py`'s `_emit_struct_preamble()` (line ~292) guards the single-span constructor with `if spec.is_uniform and len(spec.param_arrays) == 1:`. For E4_1 (`is_uniform=False`, 1 param array), neither this nor the `elif len(spec.param_arrays) > 1:` branch fires, so **no constructor is emitted** (only the default). The same guard appears in `_emit_factory()` (line ~485): for 1 non-uniform param array, the `else` branch fires and generates a malformed factory with duplicate parameter names.
  - **Fix:** In both `_emit_struct_preamble` and `_emit_factory`, change the condition from `spec.is_uniform and len(spec.param_arrays) == 1` to `len(spec.param_arrays) == 1`. This matches the pattern used in E2_1.cpp (non-uniform, single `alpha` array, single-span constructor).
  - **Concrete changes:**
    - `_emit_struct_preamble` line ~292: `if spec.is_uniform and len(spec.param_arrays) == 1:` → `if len(spec.param_arrays) == 1:`
    - `_emit_factory` line ~485: `elif spec.is_uniform and len(spec.param_arrays) == 1:` → `elif len(spec.param_arrays) == 1:`
  - File: `scripts/stencil_gen/stencil_gen/codegen.py`
  - Test: add a test in `test_codegen.py` that creates a `StencilGenSpec(is_uniform=False, param_arrays={"alpha": 4}, ...)` and verifies the generated code contains `E4_1(std::span<const real> a)` constructor and `make_E4_1(std::span<const real> alpha)` factory
  - Also verify existing E4u_1 uniform test still passes: `cd scripts/stencil_gen && uv run pytest tests/test_codegen.py -v`

- [ ] **21.4b** Generate the E4_1 C++ stencil struct:
  - Use `codegen.generate_stencil_cpp()` to produce `E4_1.cpp`
  - The struct should have: **P=2, R=4, T=7, X=0** (corrected: R=4, not 5)
  - Member array: `std::array<real, 4> alpha` (4 free params)
  - `param_arrays = {"alpha": 4}` in `StencilGenSpec`
  - Methods: `interior()`, `nbs_floating()` (**4×7=28 coefficients**), `nbs_dirichlet()` (**3×7=21 emitted coefficients**)
  - CSE will be needed (expressions will be complex rational functions of psi and alpha)
  - The `StencilGenSpec` for E4_1:
    ```python
    StencilGenSpec(
        name="E4_1", P=2, R=4, T=7, X=0,
        derivative_order=1, is_uniform=False,
        param_arrays={"alpha": 4},
        interior_coeffs=interior,  # from derive_interior(0, 2, 1)
        floating_coeffs=list(floating_matrix),  # 28 entries row-major; SymPy Matrix flattens row-major
        dirichlet_coeffs=dirichlet_flat,         # 28 entries; see construction note below
    )
    ```
  - **IMPORTANT: `dirichlet_coeffs` format.** The codegen's `_emit_nbs_methods` slices `spec.dirichlet_coeffs[spec.T:]` to skip row 0. So `spec.dirichlet_coeffs` must have **R×T = 28 entries**, not (R-1)×T = 21. Construct as:
    ```python
    # CutCellResult.dirichlet is (R-1)×T = 3×7 = 21 entries (rows 1..3)
    # Prepend T=7 zeros as placeholder for row 0 (wall row, dropped by Dirichlet BC)
    dirichlet_flat = [Integer(0)] * 7 + list(result.dirichlet)
    ```
    This matches the existing E4u_1 spec pattern (`[Integer(0)] * 5 + e4u_dirichlet_coeffs`) and the polyE2_1 spec pattern (`[Integer(0)] * 4 + ...`).
  - Write generated code to `scripts/stencil_gen/output/E4_1.cpp` (not into src/stencils/ yet)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: verify generated code has correct structure (check for P=2, R=4, T=7 constants)

- [ ] **21.4c** Generate the E4_1 test file:
  - Use `codegen.generate_test_cpp()` to produce test data
  - Pick specific alpha values (e.g., alpha = [0.1, -0.05, 0.02, 0.01]) and psi values (0.3, 0.7, 1.0) for test cases
  - Use `codegen.compute_test_values()` to evaluate expected coefficients numerically
  - Write to `scripts/stencil_gen/output/E4_1.t.cpp`
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 21.5 — Assemble the full `derive_and_generate` pipeline

- [ ] **21.5a** Add a high-level `derive_cut_cell_scheme(scheme: SchemeParams, psi)` function in `temo.py`:
  - **Signature:** `def derive_cut_cell_scheme(scheme: SchemeParams, psi: Symbol) -> CutCellResult`
  - Orchestrates the full pipeline:
    1. `dims = compute_dimensions(scheme.p, scheme.q, scheme.s, scheme.nextra, scheme.nu)` — corrected dimensions
    2. `uniform = derive_uniform_boundary_for_temo(scheme)` → `UniformResult` (B_u, interior, alpha_symbols)
    3. `floating_result = construct_cut_cell_stencil(uniform.B_u, uniform.interior, scheme.p, scheme.q, scheme.nu, scheme.nextra, psi)` → `StencilResult`
    4. If `scheme.nu == 2` (Neumann needed):
       - `B_uN, eta_u = derive_uniform_neumann(uniform.interior, scheme.p, scheme.q, scheme.nu)`
       - `neumann_main, eta = construct_neumann_stencil(uniform.B_u, B_uN, eta_u, uniform.interior, scheme.p, scheme.q, scheme.nu, scheme.nextra, psi)`
       - Else: `neumann_main, eta = None, None`
    5. `return assemble_cut_cell_result(floating_result.matrix, neumann_main, eta, dims, uniform.alpha_symbols)`
  - Returns `CutCellResult` with all coefficient matrices
  - **For E4_1 (nu=1):** steps 4's Neumann branch is skipped. The result has `neumann=None, eta=None, dims.X=0`.
  - Works for E2_1, E2_2, E4_1 (any scheme in the Table 1 family with corrected dimensions)
  - **Dispatch strategy for step 1 (Decision):** Use `derive_uniform_boundary_for_temo` for ALL schemes (E2 and E4), not just E4+. The 21.1a regression check (21.1a step: "verify `derive_uniform_boundary_for_temo(E2_1)` produces the same `UniformResult` as `derive_e2_uniform_boundary(nu=1)`") ensures the general function handles E2's nextra>0 conservation case correctly. If the 21.1a regression check fails (e.g., alpha naming differs), fix the general function rather than keeping `derive_e2_uniform_boundary` as a fallback — the goal is ONE code path. The old `derive_e2_uniform_boundary` remains in `temo.py` but is no longer called by the pipeline; it can be deprecated in a future cleanup phase.
  - **Known limitation for nu=2 schemes:** `solve_boundary_row` uses `q+1` equations from `build_taylor_system`. For E2_2 (q=1, nu=2), this gives only 2 equations while the existing `derive_e2_uniform_boundary` uses `max(q+1, nu+1)=3`. If `derive_uniform_boundary_for_temo` uses `solve_boundary_row` directly for E2_2, it will build a different Taylor system. Fix: either (a) build the Taylor system inline with `max(q+1, nu+1)` equations instead of calling `solve_boundary_row`, or (b) modify `solve_boundary_row` to accept an optional `n_eqs` parameter. Since E4_1 (nu=1, q=3) has `q+1 = max(q+1, nu+1) = 4`, this is NOT an issue for Phase 21's primary target.
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **21.5b** Validate that the generalized pipeline still reproduces E2_1 and E2_2:
  - Run `derive_cut_cell_scheme(E2_1, psi)` and compare `CutCellResult.floating` against the existing E2_1 test data (exact symbolic equality)
  - Run `derive_cut_cell_scheme(E2_2, psi)` and compare against existing E2_2 test data
  - No regressions in existing test suite: `cd scripts/stencil_gen && uv run pytest tests/ -v`
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/ -v`

### 21.6 — Register E4_1 in the solver (optional)

- [ ] **21.6a** Copy generated E4_1.cpp and E4_1.t.cpp to `src/stencils/`:
  - Copy `scripts/stencil_gen/output/E4_1.cpp` → `src/stencils/E4_1.cpp`
  - Copy `scripts/stencil_gen/output/E4_1.t.cpp` → `src/stencils/E4_1.t.cpp`
  - **stencil.hpp** (line ~278, after `make_E4u_1`): Add declaration:
    ```cpp
    stencil make_E4_1(std::span<const real>);
    ```
  - **CMakeLists.txt** (`src/stencils/CMakeLists.txt`):
    - Add `E4_1.cpp` to the `shoccs-stencils` source list (after `E4u_1.cpp`, line ~5)
    - Add `add_unit_test(E4_1 "stencils" shoccs-stencils)` (after E4u_1 test, line ~17)
  - **stencil.cpp** (`src/stencils/stencil.cpp`, line ~48, in the `order == 1` branch):
    Add a new `else if` branch after the `E4u` case (between lines 50 and 51):
    ```cpp
    } else if (type == "E4") {
        logger(spdlog::level::info, "E4 cut-cell first scheme chosen");
        return make_E4_1(alpha);
    } else if (type == "E6u") {
    ```
  - Build and run: `cmake --build build --target t-E4_1 && ctest --test-dir build -R t-E4_1`
  - Also verify no regressions: `ctest --test-dir build -L stencils`
  - Files: `src/stencils/E4_1.cpp`, `src/stencils/E4_1.t.cpp`, `src/stencils/stencil.hpp`, `src/stencils/stencil.cpp`, `src/stencils/CMakeLists.txt`

---

## Key Risks

1. **Expression swell:** E4_1 has 28 floating coefficients (4×7, vs E2_1's 20=4×5), each a rational function of psi and 4 alphas. CSE output could be 2000+ lines. This is expected and handled by the codegen module.

2. **TEMO pipeline correctness for wider stencils:** The TEMO functions (`build_degenerate_stencil`, `solve_uniform_limit`, `construct_cut_cell_stencil`) were developed and tested only for E2 schemes (3×4 and 1×3 B_u matrices). With E4_1's 3×6 B_u, the column counts, conservation column ranges, and Vandermonde systems are all larger. Both `build_degenerate_stencil` and `solve_uniform_limit` produce **overdetermined** near-interior row systems for E4_1 (3 unknowns, 4 equations). The detailed column analysis is in item 21.3a. The `solve_temo_row` path (general psi) is exactly determined (4 solve cols = 4 equations). If the overdetermined systems are inconsistent, the code raises `RuntimeError` — the fix would be in the conservation column selection logic or the alpha distribution from 21.1a.

3. **Performance:** The `solve_in_field` calls use 4×4 Vandermonde systems (vs 2×2 for E2), and there are 4 rows to solve. Each solve involves DomainMatrix LU factorization in QQ(psi). Total derivation time should stay under 10 seconds.

4. **Dimension formula (D-R25):** The corrected formula `r = p+1+nextra` has been verified for E2 and E4 schemes against Table 1 and existing C++ code. However, it has NOT been verified for E6 or E8 TEMO schemes (which don't exist yet). If these are attempted later, the formula should be re-verified.

5. **`solve_boundary_row` equation count for nu=2:** `boundary.solve_boundary_row` uses `q+1` equations (via `build_taylor_system`), while `temo._build_uniform_vandermonde` uses `max(q+1, nu+1)`. For E4_1 (nu=1, q=3) these are equal (both 4). For E2_2 (nu=2, q=1) they differ: 2 vs 3. The new `derive_uniform_boundary_for_temo` must account for this when handling nu=2 schemes (see 21.5a). This is NOT a risk for Phase 21's primary target (E4_1) but affects E2 backward compatibility in 21.5b.

6. **Codegen single-param non-uniform bug (discovered during plan refinement):** `codegen.py`'s `_emit_struct_preamble` and `_emit_factory` guard single-param-array constructor/factory generation with `spec.is_uniform and len(spec.param_arrays) == 1`. For E4_1 (`is_uniform=False`, 1 param array), the constructor is not emitted and the factory generates duplicate parameter names. Fix in 21.4a (prerequisite). Low risk — the fix is a one-line condition change in two places and existing E4u_1 uniform tests confirm no regression.
