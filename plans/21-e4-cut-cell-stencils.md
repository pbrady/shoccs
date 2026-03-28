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

**The gap:** `temo.py` uses `derive_e2_uniform_boundary()` which is hardcoded for p=1, q=1. The general `boundary.py` already handles arbitrary p, but the TEMO pipeline doesn't use it. The key work is:
1. Bridge `boundary.py` output into `temo.py`'s `UniformResult` format
2. Handle E4's different sizing (r=4, t=6 vs E2's r=3, t=4) and free parameter structure
3. Handle E4's zero-constrained alpha entries (Table 1: alpha^u_{05}=0, alpha^u_{15}=0)
4. Generate, validate, and possibly register the E4_1 C++ stencil

**E4_1 scheme parameters (from Table 1 of Brady & Livescu 2021):**
- p=2, q=3, s=0, nextra=0, nu=1
- Uniform base: r=4 rows, t=6 columns (from q+1+nextra=4, p+q+1+nextra=6)
- Cut-cell: R=5 rows, T=7 columns
- Free parameters: alpha^u_{04}, alpha^u_{14}, alpha^u_{24}, alpha^u_{25}
- Zero-constrained: alpha^u_{05}=0, alpha^u_{15}=0

---

## Items

### 21.1 — Bridge boundary.py → temo.py UniformResult

- [ ] **21.1a** Add a general `derive_uniform_boundary_for_temo(scheme: SchemeParams, psi: Symbol, alpha_symbols=None)` function in `temo.py` that:
  1. Calls `boundary.derive_boundary(p=scheme.p, nu=scheme.nu)` to get the general boundary result
  2. Extracts the r_eff × t coefficient matrix as a SymPy Matrix (r_eff = r for nu=1, r-1 for nu=2)
  3. Extracts interior coefficients from the boundary result
  4. Extracts quadrature weights from conservation (or computes them)
  5. Packages everything into the existing `UniformResult` namedtuple
  6. Uses the existing alpha symbols from `boundary.py` or remaps to caller-provided symbols
  - The key challenge: `boundary.py`'s `BoundaryResult` stores rows with `coefficients` and `free_params`, but `temo.py`'s `UniformResult` expects a flat Matrix and separate lists. This is a format conversion, not a mathematical change.
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: verify `derive_uniform_boundary_for_temo(E2_1, psi)` produces the same `UniformResult` as the existing `derive_e2_uniform_boundary(nu=1)`

- [ ] **21.1b** Test the bridge for E4_1:
  - Call `derive_uniform_boundary_for_temo(E4_1, psi)`
  - Verify: r_eff=4 rows, t=6 columns
  - Verify: 4 free alpha symbols (matching Table 1: alpha^u_{04}, alpha^u_{14}, alpha^u_{24}, alpha^u_{25})
  - Verify: substituting the known E4u_1 optimized alpha values reproduces E4u_1.cpp floating coefficients
  - Verify: conservation holds (W^T B = 0 for interior columns)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "uniform"`

### 21.2 — Handle E4 zero-constrained entries

- [ ] **21.2a** Handle the zero constraints from Table 1:
  - E4_1 has alpha^u_{05}=0 and alpha^u_{15}=0 (last column of rows 0 and 1 forced to zero)
  - This may already be handled by `boundary.py` if q=3 with t=6 gives the right underdetermined structure
  - If not, add explicit zero constraints in the bridge function
  - Verify numerically: the uniform boundary at these positions is indeed zero
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (extend 21.1a if needed)
  - Test: verify that the symbolic coefficients at positions (0,5) and (1,5) are exactly zero

### 21.3 — E4_1 TEMO construction

- [ ] **21.3a** Run the full TEMO pipeline for E4_1:
  - Call `construct_cut_cell_stencil(B_u, interior, p=2, q=3, nu=1, nextra=0, psi)`
  - Verify: output is 5×7 matrix of rational functions in psi and alpha
  - Verify: at psi=1, reduces to uniform E4 stencil (plus wall column of zeros)
  - Verify: at psi=0, satisfies degenerate constraints (Design Principles 1 & 2)
  - Verify: Taylor accuracy holds for each row at arbitrary psi
  - Verify: conservation holds for all psi
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "temo"`

- [ ] **21.3b** Performance check:
  - The full E4_1 derivation (uniform boundary + TEMO + conservation) should complete in < 5 seconds
  - If slow, profile and apply QQ(psi) field optimizations from 20.5e
  - Test: add a timing assertion

### 21.4 — E4_1 C++ code generation

- [ ] **21.4a** Generate the E4_1 C++ stencil struct:
  - Use `codegen.generate_stencil_cpp()` to produce `E4_1.cpp`
  - The struct should have: P=2, R=5, T=7, X=0
  - Member array: `std::array<real, 4> alpha` (4 free params)
  - Methods: `interior()`, `nbs_floating()` (5×7=35 coefficients), `nbs_dirichlet()` (4×7=28 coefficients)
  - CSE will be needed (expressions will be complex rational functions of psi and alpha)
  - Write generated code to `scripts/stencil_gen/output/E4_1.cpp` (not into src/stencils/ yet)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: verify generated code compiles (or at least has correct structure)

- [ ] **21.4b** Generate the E4_1 test file:
  - Use `codegen.generate_test_cpp()` to produce test data
  - Pick specific alpha values and psi values for test cases
  - Write to `scripts/stencil_gen/output/E4_1.t.cpp`
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 21.5 — Assemble the full `derive_and_generate` pipeline

- [ ] **21.5a** Add a high-level `derive_cut_cell_scheme(scheme: SchemeParams, psi)` function:
  - Orchestrates: uniform boundary → TEMO construction → assembly → codegen
  - Returns the `CutCellResult` with all coefficient matrices
  - This replaces the scheme-specific code paths in `assemble_cut_cell_result`
  - Works for E2_1, E2_2, E4_1, E4_2 (any scheme in the Table 1 family)
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **21.5b** Validate that the generalized pipeline still reproduces E2_1 and E2_2:
  - Run `derive_cut_cell_scheme(E2_1, psi)` and compare against existing test data
  - Run `derive_cut_cell_scheme(E2_2, psi)` and compare against existing test data
  - No regressions allowed
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/ -v`

### 21.6 — Register E4_1 in the solver (optional)

- [ ] **21.6a** Copy generated E4_1.cpp and E4_1.t.cpp to `src/stencils/`:
  - Add `make_E4_1` declaration to `stencil.hpp`
  - Add E4_1.cpp to `src/stencils/CMakeLists.txt`
  - Add test via `add_unit_test()`
  - Build and run: `cmake --build build --target t-E4_1 && ctest --test-dir build -R t-E4_1`
  - File: `src/stencils/E4_1.cpp`, `src/stencils/E4_1.t.cpp`, `src/stencils/stencil.hpp`, `src/stencils/CMakeLists.txt`

---

## Key Risks

1. **Expression swell:** E4_1 has 35 floating coefficients (vs E2_1's 20), each a rational function of psi and 4 alphas. CSE output could be 3000+ lines. This is expected and handled by the codegen module.

2. **Conservation system size:** The E4_1 conservation system is larger (6 equations vs E2_1's 4). The QQ(psi) field approach should handle this, but verify performance.

3. **Boundary solver output format:** The `boundary.py` result format may not directly match what `temo.py` expects. The bridge function (21.1a) handles this conversion.
