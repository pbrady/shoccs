# Phase 26: E4_1 Cut-Cell Conservation via Zero-Constrained Alphas

**Goal:** Implement full symbolic cut-cell conservation for E4_1 by applying the paper's zero constraints (α^u_{05}=0, α^u_{15}=0) before solving the conservation system. This was proven tractable: SymPy solves the system in ~1.2 seconds when the zeros are applied first, producing clean polynomial weights in ψ.

**Depends on:** Phase 24 (uniform conservation), Phase 25 (limit verification)

**Read first:**
- `scripts/stencil_gen/stencil_gen/temo.py` — `derive_uniform_boundary_for_temo`, `construct_cut_cell_stencil`, `build_cut_cell_conservation_system`
- `scripts/stencil_gen/stencil_gen/conservation.py` — `build_conservation_system`, `solve_conservation`
- `scripts/stencil_gen/tests/test_e4_cut_cell.py` — existing E4_1 tests including xfail conservation test
- `plans/stencil-derivation-math-reference.md` (Section 4.4: Cut-Cell Conservation)

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "conservation" --timeout=300
cd scripts/stencil_gen && uv run pytest tests/ -v --timeout=300
```

---

## Background

The paper (Table 1) specifies for E4_1:
- Free parameters: α^u_{04}, α^u_{14}, α^u_{24}, α^u_{25}
- **Zeros: α^u_{05}=0, α^u_{15}=0**

In the pipeline's alpha numbering (alpha_0..alpha_4 across 4 boundary rows), the zeros correspond to **alpha_3=0 and alpha_4=0** — the free parameters in column 5 of rows 0 and 1. These zeros must be applied BEFORE building the cut-cell stencil and solving conservation.

Research confirmed: with alpha_3=alpha_4=0, `sympy.solve()` produces a clean single-branch solution in ~1.2 seconds:
- w_1, w_2, w_3 are degree-3 polynomials in ψ (linear in w_4)
- alpha_0 is a rational function of ψ and w_4
- alpha_1 depends on ψ, w_4, and alpha_2
- 2 remaining free parameters: alpha_2 and w_4

## The Procedure

1. Build uniform boundary B_u at TEMO dimensions (r=4, t=6) with `solve_boundary_row`
2. Assign 5 alpha symbols but **set alpha_3=0, alpha_4=0** (per Table 1 zeros)
3. Apply uniform conservation (already done in Phase 24) — constrains last row
4. Build cut-cell B_l(ψ) via TEMO from the zero-constrained B_u (now has 3 free alphas)
5. Build cut-cell conservation system: 5 equations in 4 weights + 3 alphas = 7 unknowns
6. Solve with `sympy.solve()` — feasible with zeros applied (~1.2s)
7. Back-substitute to get weights and constrained alphas as functions of ψ and remaining free params
8. The result: a conservative E4_1 stencil with ~2 free parameters for numerical optimization

---

## Items

### 26.1 — Apply zero constraints in `derive_uniform_boundary_for_temo`

- [ ] **26.1a** Add a `zeros` parameter to `derive_uniform_boundary_for_temo`:
  - The parameter specifies which alpha indices to set to zero (e.g., `zeros={3, 4}` for E4_1)
  - After creating alpha symbols and solving boundary rows, substitute zeros: `B_u = B_u.subs({alpha_3: 0, alpha_4: 0})`
  - Remove zeroed symbols from the alpha_symbols list in UniformResult
  - For E4_1: this reduces from 5 alphas to 3 (alpha_0, alpha_1, alpha_2)
  - The SchemeParams or a new config should store the zero indices per Table 1:
    - E2_1: no zeros (nextra=1 provides enough DOF)
    - E4_1: zeros at indices corresponding to α^u_{05}, α^u_{15}
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: `uv run pytest tests/test_temo.py -v -k "E2" --timeout=60` (E2 unchanged)

- [ ] **26.1b** Test E4_1 uniform boundary with zeros:
  - Call `derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})`
  - Verify: B_u has 3 free alpha symbols (was 5 without zeros, 4 after conservation)
  - Verify: B_u[0, 5] == 0 and B_u[1, 5] == 0 (the zeroed columns)
  - Verify: Taylor accuracy still holds for all 4 rows
  - Verify: uniform conservation still holds
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 26.2 — Build zero-constrained cut-cell stencil

- [ ] **26.2a** Build E4_1 cut-cell with zero-constrained B_u:
  - Call `construct_cut_cell_stencil` with the zero-constrained B_u (3 alphas)
  - Verify: 5×7 matrix with entries that are rational functions of (ψ, alpha_0, alpha_1, alpha_2)
  - Verify: correct limits at ψ=0 and ψ=1
  - Verify: Taylor accuracy at intermediate ψ
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 26.3 — Solve cut-cell conservation

- [ ] **26.3a** Solve the cut-cell conservation system with zeros applied:
  - Build conservation equations: `build_cut_cell_conservation_system(B_l, R, T, p, nu, interior, psi)`
  - The system now has 5 equations, 7 unknowns (4 weights + 3 alphas)
  - Call `sympy.solve(eqs, [alpha_0, alpha_1, w_1, w_2, w_3])` treating alpha_2 and w_4 as free
  - Expected: single-branch solution in ~1-2 seconds
  - Verify: all 5 conservation equations evaluate to exactly 0 upon substitution
  - File: `scripts/stencil_gen/stencil_gen/temo.py` (add `solve_cut_cell_conservation` function)
  - Test: `uv run pytest tests/test_e4_cut_cell.py -v -k "solve_conservation" --timeout=300`

- [ ] **26.3b** Apply conservation solution to the stencil:
  - Substitute the solved alpha_0, alpha_1, w_1, w_2, w_3 back into B_l(ψ)
  - The resulting stencil has entries that are rational functions of (ψ, alpha_2, w_4)
  - These 2 remaining free parameters are for numerical optimization
  - Verify: conservation column sums are symbolically zero for all ψ
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

### 26.4 — Validate the conservative stencil

- [ ] **26.4a** Remove xfail from conservation test:
  - The test `test_e4_1_conservation_fails` should now PASS
  - Verify: `Σ_i w_i(ψ) · B[i,j](ψ) = 0` as a symbolic identity in ψ for all interior columns
  - Test at specific (ψ, alpha_2, w_4) values as well
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

- [ ] **26.4b** Verify Taylor accuracy is preserved:
  - Each row must still satisfy q+1=4 Taylor accuracy equations
  - Test at ψ=0.3, 0.5, 0.7 with random alpha_2, w_4 values
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

- [ ] **26.4c** Verify ψ limits are preserved:
  - ψ=1: wall column zeros for boundary rows, embedding matches uniform B_u
  - ψ=0: degenerate design principles satisfied
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 26.5 — Integrate into `derive_cut_cell_scheme`

- [ ] **26.5a** Update `derive_cut_cell_scheme` to use zeros and cut-cell conservation:
  - When scheme has zeros specified (E4_1), apply them before TEMO construction
  - After TEMO, solve cut-cell conservation
  - Return the fully conservative CutCellResult
  - The result's `alpha_symbols` should list only the remaining free params
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **26.5b** E2_1 regression test:
  - E2_1 has no zeros — its conservation is already handled by the nextra=1 mechanism
  - Verify E2_1 results are unchanged
  - File: `scripts/stencil_gen/tests/test_temo.py`

### 26.6 — Re-generate E4_1 C++ code

- [ ] **26.6a** Generate conservative E4_1 C++ stencil:
  - Use codegen to produce E4_1.cpp with the conservation-enforced stencil
  - The struct should have: P=2, R=5, T=7, 2 free parameters (alpha_2, w_4)
  - The generated code will have larger CSE output (conservation adds rational complexity)
  - Write to `scripts/stencil_gen/output/E4_1.cpp`
  - Copy to `src/stencils/E4_1.cpp`
  - Build: `cmake --build build --target t-E4_1`
  - Test: `ctest --test-dir build -R t-E4_1`
  - File: `src/stencils/E4_1.cpp`, `src/stencils/E4_1.t.cpp`

### 26.7 — Update memory and plans

- [ ] **26.7a** Update the stencil derivation memory with conservation resolution:
  - Conservation IS symbolically solvable when zeros are applied first
  - The zeros from Table 1 are essential — they eliminate bilinear branching
  - File: memory files
