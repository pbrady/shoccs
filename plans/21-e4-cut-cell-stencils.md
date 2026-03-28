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

- [ ] **21.0a** Fix `compute_dimensions()` in `scripts/stencil_gen/stencil_gen/temo.py`:
  - The function already takes `p` as its first parameter (signature: `compute_dimensions(p, q, s, nextra, nu)`)
  - Single-line change on line 29: replace `r = q + 1 + nextra` with `r = p + 1 + nextra` (per D-R25)
  - The column formula `t = p + q + 1 + nextra` on line 28 is correct and unchanged
  - Update the docstring to say "r = p + 1 + nextra" instead of "r = q + 1 + nextra"
  - Remove the "Note: verified for E2 schemes only" caveat (now verified for E4 too)
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (lines 24–72)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "Dimensions"`
  - Expected dimension changes:
    - E2_1 (p=1, q=1, nextra=1): **unchanged** (r=3, t=4, R=4, T=5, X=0) because p+1+nextra = q+1+nextra = 3
    - E2_2 (p=1, q=1, nextra=0): **unchanged** (r=2, t=3, R=2, T=4, X=2) because p+1+nextra = q+1+nextra = 2
    - E4_1 (p=2, q=3, nextra=0): r changes 4→3, R changes 5→4 → **(r=3, t=6, R=4, T=7, X=0)**
    - E4_2 (p=2, q=3, nextra=0): r changes 4→3, R changes 4→3, X changes 4→3 → **(r=3, t=6, R=3, T=7, X=3)**
  - **Must update test assertions**: `test_first_derivative_no_neumann` uses `E4_1.dims().X == 0` (no change). `test_second_derivative_has_neumann` checks `dims_e4_2.X == dims_e4_2.R` — both change from 4 to 3, so the equality still holds. No test changes needed unless there are explicit E4 dimension value checks elsewhere.

### 21.1 — Bridge boundary.py → temo.py UniformResult

- [ ] **21.1a** Add a general `derive_uniform_boundary_for_temo(scheme: SchemeParams, alpha_symbols=None)` function in `temo.py` that:
  1. Computes TEMO dimensions using the **corrected** `compute_dimensions` (21.0a): r = p+1+nextra, t = p+q+1+nextra
  2. For nu=1: r_eff = r; for nu=2: r_eff = r - 1
  3. Computes `n_free_per_row = t - (q + 1)` (= 2 for E4_1)
  4. Creates alpha symbols following the boundary.py convention:
     - Rows 0..(r_eff-2): 1 active alpha + (n_free-1) zeros
     - Row r_eff-1 (penultimate/last non-conservation row): min(n_free, 2) active alphas + rest zeros
     - Total: (r_eff - 1) * 1 + min(n_free, 2) alphas (= 4 for E4_1, same for E2_1)
  5. Calls `boundary.solve_boundary_row(i, t, q, nu, free_symbols)` for each row i = 0..r_eff-1
  6. **No conservation-constrained last row** for E4_1 (nextra=0, r_eff=3 rows all have free params). For E2_1 (nextra=1), follows the existing `derive_e2_uniform_boundary` approach with conservation on the last row.
     - Specifically: if nextra > 0, there IS an extra row (row r_eff-1) that needs conservation. Use simple column sums `sum_i B_u[i,j] = 0` for j >= p+1 to determine the last row's phi symbols.
     - If nextra = 0, no conservation step — all rows retain their free alphas.
  7. Gets interior coefficients via `interior.derive_interior(s, p, nu)` + `full_gamma_array()`
  8. Assembles the r_eff × t Matrix and packages into `UniformResult`
  - **Key insight:** boundary.py's `derive_boundary(p, nu)` uses `r = 2p-1` (SBP minimal rows) which gives r=3, t=5 for E4. The TEMO needs t=6 (wider). But `solve_boundary_row(i, t, q, nu, free)` accepts arbitrary t, so call it with the TEMO t=6. This is NOT a format conversion from `BoundaryResult` — it's a fresh derivation using the lower-level `solve_boundary_row` function.
  - **Relationship to E4u_1:** The TEMO 3×6 boundary extends the uniform 3×5 boundary (E4u_1) by one column. Row 0 and Row 1's first 5 coefficients match E4u_1 exactly (column 5 is zero). Row 2 has 2 free alphas (alpha_2, alpha_3) instead of being conservation-constrained like E4u_1's row 2.
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: verify `derive_uniform_boundary_for_temo(E2_1)` produces the same `UniformResult` as the existing `derive_e2_uniform_boundary(nu=1)` (regression check)

- [ ] **21.1b** Test the bridge for E4_1:
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
  - Verify: at psi=1, rows 0–2 reduce to B_u embedded in T-frame (wall=0, cols 1–6 = B_u), and row 3 (near-interior) is the interior stencil (with wall=0)
  - Verify: at psi=0, satisfies degenerate constraints (DP1: B_d[i, j+1] = B_u[i, j] for j≥1; DP2: B_u[i,0] assigned to wall, x_0 zeroed for nu=1)
  - Verify: Taylor accuracy holds for each row at a sample psi (e.g., psi=1/2): evaluate the stencil and check it reproduces f'(x_i) for polynomials of degree ≤ q=3
  - Verify: no beta symbols in the result (nextra=0 should prescribe all excess columns via limit interpolation, eliminating betas)
  - **Potential issue:** `construct_cut_cell_stencil` internally calls `build_degenerate_stencil` and `solve_uniform_limit`, which use the `B_u.rows` to determine R = r_eff + 1. With corrected r_eff=3, this gives R=4. Verify these functions work correctly for the 3×6 B_u (they were only tested with 3×4 for E2_1 and 1×3 for E2_2).
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "temo"`

- [ ] **21.3b** Performance check:
  - The full E4_1 derivation (uniform boundary + TEMO + conservation) should complete in < 10 seconds
  - E4_1 has 28 floating coefficients (4×7), each a rational function of psi and 4 alphas — larger than E2_1 (4×5=20 coefficients with 4 alphas)
  - If slow (>10s), profile and apply QQ(psi) field optimizations from 20.5e
  - If the `solve_in_field` calls are the bottleneck, the E4 system has n_eqs=4 per row (vs E2's n_eqs=2), so the Vandermonde solves are 4× larger
  - Test: add a timing assertion or measure with `time.time()`

### 21.4 — E4_1 C++ code generation

- [ ] **21.4a** Generate the E4_1 C++ stencil struct:
  - Use `codegen.generate_stencil_cpp()` to produce `E4_1.cpp`
  - The struct should have: **P=2, R=4, T=7, X=0** (corrected: R=4, not 5)
  - Member array: `std::array<real, 4> alpha` (4 free params)
  - `param_arrays = {"alpha": 4}` in `StencilGenSpec`
  - Methods: `interior()`, `nbs_floating()` (**4×7=28 coefficients**), `nbs_dirichlet()` (**3×7=21 coefficients**)
  - CSE will be needed (expressions will be complex rational functions of psi and alpha)
  - The `StencilGenSpec` for E4_1:
    ```python
    StencilGenSpec(
        name="E4_1", P=2, R=4, T=7, X=0,
        derivative_order=1, is_uniform=False,
        param_arrays={"alpha": 4},
        interior_coeffs=interior,  # from derive_interior(0, 2, 1)
        floating_coeffs=list(floating_matrix),  # 28 entries row-major
        dirichlet_coeffs=list(dirichlet_matrix),  # 21 entries row-major
    )
    ```
  - Write generated code to `scripts/stencil_gen/output/E4_1.cpp` (not into src/stencils/ yet)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: verify generated code has correct structure (check for P=2, R=4, T=7 constants)

- [ ] **21.4b** Generate the E4_1 test file:
  - Use `codegen.generate_test_cpp()` to produce test data
  - Pick specific alpha values (e.g., alpha = [0.1, -0.05, 0.02, 0.01]) and psi values (0.3, 0.7, 1.0) for test cases
  - Use `codegen.compute_test_values()` to evaluate expected coefficients numerically
  - Write to `scripts/stencil_gen/output/E4_1.t.cpp`
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 21.5 — Assemble the full `derive_and_generate` pipeline

- [ ] **21.5a** Add a high-level `derive_cut_cell_scheme(scheme: SchemeParams, psi)` function in `temo.py`:
  - Orchestrates the full pipeline:
    1. `derive_uniform_boundary_for_temo(scheme)` → `UniformResult` (B_u, interior, alpha_symbols)
    2. `construct_cut_cell_stencil(B_u, interior, p, q, nu, nextra, psi)` → `StencilResult` (floating matrix)
    3. If nu=2: `derive_uniform_neumann(interior, p, q, nu)` + `construct_neumann_stencil(...)` → neumann matrix + eta
    4. `assemble_cut_cell_result(floating, neumann, eta, dims, alpha_symbols)` → `CutCellResult`
  - Returns `CutCellResult` with all coefficient matrices
  - Works for E2_1, E2_2, E4_1 (any scheme in the Table 1 family with corrected dimensions)
  - For E2_1/E2_2: should produce identical results to the current inline pipeline
  - For E4_1: produces the new 4×7 cut-cell stencil
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **21.5b** Validate that the generalized pipeline still reproduces E2_1 and E2_2:
  - Run `derive_cut_cell_scheme(E2_1, psi)` and compare `CutCellResult.floating` against the existing E2_1 test data (exact symbolic equality)
  - Run `derive_cut_cell_scheme(E2_2, psi)` and compare against existing E2_2 test data
  - No regressions in existing test suite: `cd scripts/stencil_gen && uv run pytest tests/ -v`
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/ -v`

### 21.6 — Register E4_1 in the solver (optional)

- [ ] **21.6a** Copy generated E4_1.cpp and E4_1.t.cpp to `src/stencils/`:
  - Add `stencil make_E4_1(std::span<const real> alpha);` declaration to `stencil.hpp` (4 alpha params, matching E2_1 pattern)
  - Add E4_1.cpp to `src/stencils/CMakeLists.txt` source list
  - Add test via `add_unit_test()` in `src/stencils/CMakeLists.txt`
  - Register in the Lua stencil factory (`stencil::from_lua`) so `type = "E4"` with `order = 1` creates E4_1
  - Build and run: `cmake --build build --target t-E4_1 && ctest --test-dir build -R t-E4_1`
  - Files: `src/stencils/E4_1.cpp`, `src/stencils/E4_1.t.cpp`, `src/stencils/stencil.hpp`, `src/stencils/CMakeLists.txt`

---

## Key Risks

1. **Expression swell:** E4_1 has 28 floating coefficients (4×7, vs E2_1's 20=4×5), each a rational function of psi and 4 alphas. CSE output could be 2000+ lines. This is expected and handled by the codegen module.

2. **TEMO pipeline correctness for wider stencils:** The TEMO functions (`build_degenerate_stencil`, `solve_uniform_limit`, `construct_cut_cell_stencil`) were developed and tested only for E2 schemes (3×4 and 1×3 B_u matrices). With E4_1's 3×6 B_u, the column counts, conservation column ranges, and Vandermonde systems are all larger. Particular attention needed for:
   - `solve_uniform_limit`: conservation column selection logic (lines 826–832) uses conditions `j <= R-p` and `j >= p+2` — verify these produce the right conserved columns for p=2, R=4.
   - `build_degenerate_stencil`: the near-interior row solve (step 3) has unknown/known column partitioning that depends on r and p — verify for r=3, p=2.
   - `identify_prescribed_entries`: verified — each row has T=7 cols, 1 zeroed (cat A), leaving 6 free. n_eqs=4, so 2 excess cols prescribed by limit interpolation. Final: 4 solve cols, 4 equations → exactly determined, no betas. (Excess = p+nextra = 2+0 = 2 per row, same as E2_1's p+nextra = 1+1 = 2.)

3. **Performance:** The `solve_in_field` calls use 4×4 Vandermonde systems (vs 2×2 for E2), and there are 4 rows to solve. Each solve involves DomainMatrix LU factorization in QQ(psi). Total derivation time should stay under 10 seconds.

4. **Dimension formula (D-R25):** The corrected formula `r = p+1+nextra` has been verified for E2 and E4 schemes against Table 1 and existing C++ code. However, it has NOT been verified for E6 or E8 TEMO schemes (which don't exist yet). If these are attempted later, the formula should be re-verified.
