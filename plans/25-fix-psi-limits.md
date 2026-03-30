# Phase 25: Verify Cut-Cell Stencil Behavior at ψ=0 and ψ=1

**Goal:** Verify and document that the cut-cell stencil B_l(ψ) satisfies:
- **ψ=1 (uniform limit):** B_l(1) embeds the uniform boundary B^u exactly — rows 0..r-1 of cut-cell equal B_u (shifted by 1 for wall column), wall column = 0 for boundary rows.
- **ψ=0 (degenerate limit):** B_l(0) satisfies Design Principles 1 (TEM) and 2 (continuity) — the zeroed column (x_0 for ν=1) is all zeros, and the wall column equals the uniform stencil's column 0.
- **Near-interior row:** Uses conservation+Taylor closure when the interior stencil doesn't fit in the T-frame (expected behavior for E2_1 and E4_1).
- **Codegen:** Install Phase 24's updated E4_1 C++ output (4-alpha conservation) into `src/stencils/`.

**Depends on:** Phase 24

**Read first:**
- `scripts/stencil_gen/stencil_gen/temo.py`:
  - `construct_cut_cell_stencil` (line 1335) — main TEMO procedure
  - `solve_temo_row` (line 1245) — per-row Taylor solve
  - `solve_uniform_limit` (line 1000) — computes B_l(1) for prescriptions
  - `build_degenerate_stencil` (line 536) — computes B_l(0) for prescriptions
  - `identify_prescribed_entries` (line 1164) — determines Category A + limit-interpolated entries
- `scripts/stencil_gen/tests/test_temo.py` — E2_1 limit tests
- `scripts/stencil_gen/tests/test_e4_cut_cell.py` — E4_1 limit tests
- `plans/stencil-derivation-math-reference.md` (Section 4.2-4.3: Design Principles and B_l(ψ))

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/ -v
cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "limit"
```

---

## Problem Analysis (Revised)

### Original hypothesis (DISPROVED)

The original plan hypothesized two bugs:
1. ψ=1 embedding doesn't match (alpha symbols shifted between rows)
2. Near-interior row has nonzero wall entries at ψ=1

### Actual findings

Both limits are **already correct**. Verified 2026-03-30 by running existing tests
and manual inspection:

**ψ=1 (uniform limit) — CORRECT:**
- Boundary rows 0..r-1: wall column = 0, cols 1..T-1 = B_u rows 0..r-1 with same alpha symbols.
  - E2_1: verified by `TestConstructCutCellStencil::test_e2_1_uniform_limit` (test_temo.py:1059)
  - E4_1: verified by `TestE4TEMOConstruction::test_uniform_limit_rows_0_3_embed_Bu` (test_e4_cut_cell.py:242)
- Near-interior row: uses conservation+Taylor closure (NOT raw interior stencil) because the interior stencil overflows the T-frame. This is correct behavior:
  - E2_1 (p=1, r=3): interior_end = r+p+1 = 5 > t=4, so closure needed. Wall = -4 at ψ=1.
  - E4_1 (p=2, r=4): interior_end = r+p+1 = 7 > t=6, so closure needed. Wall = -20/3 at ψ=1.
  - Verified by `TestE4TEMOConstruction::test_uniform_limit_row4_not_interior` (test_e4_cut_cell.py:256)
  - Taylor accuracy at ψ=1 confirmed for all rows (symbolic tests hold for all ψ).

**ψ=0 (degenerate limit) — CORRECT:**
- B_l(0) matches build_degenerate_stencil (satisfies Design Principles 1 and 2).
  - E2_1: verified by `TestConstructCutCellStencil::test_e2_1_degenerate_limit` (test_temo.py:1041)
  - E4_1: verified by `TestE4TEMOConstruction::test_degenerate_limit` (test_e4_cut_cell.py:281)
- Zeroed column (col 1 for ν=1) is all zeros at ψ=0.
- Wall column carries B_u column 0 values.

**Why the TEMO construction produces correct limits:**
`identify_prescribed_entries` (temo.py:1164) prescribes:
1. Category A (zeroed column): `psi * target` — at ψ=1 gives `target`, at ψ=0 gives 0.
2. Limit-interpolated excess columns: `psi * B_l_1[i,j] + (1-psi) * B_d[i,j]` — gives B_l_1 at ψ=1, B_d at ψ=0.
3. Remaining columns are Taylor-solved by `solve_temo_row`. At ψ=1, the prescribed entries carry enough constraint from B_u's alpha symbols to uniquely determine the remaining entries to match B_l_1 (because B_u satisfies the same Taylor accuracy system).

**Only remaining work:** Phase 24 updated E4_1 codegen output (4-alpha conservation)
but did not install it into `src/stencils/`. The generated files at
`scripts/stencil_gen/output/E4_1.cpp` (R=5, 4 alphas) differ from the installed
`src/stencils/E4_1.cpp` (R=4, 5 alphas).

---

## Items

### 25.1 — Limit verification tests [ALREADY VERIFIED]

- [x] **25.1a** ψ=1 embedding verified for E2_1:
  - `TestConstructCutCellStencil::test_e2_1_uniform_limit` (test_temo.py:1059) — checks TEMO at ψ=1 matches `solve_uniform_limit`, which embeds B_u by construction.
  - `TestE2_1Integration::test_degeneracy_psi1` (test_temo.py:1585) — checks wall=0 for boundary rows, Taylor accuracy and conservation for near-interior row at ψ=1.
  - All tests PASS.

- [x] **25.1b** ψ=1 embedding verified for E4_1:
  - `TestE4TEMOConstruction::test_uniform_limit` (test_e4_cut_cell.py:229) — full matrix matches `solve_uniform_limit`.
  - `TestE4TEMOConstruction::test_uniform_limit_rows_0_3_embed_Bu` (test_e4_cut_cell.py:242) — boundary rows cols 1..6 = B_u rows 0..3.
  - `TestE4TEMOConstruction::test_uniform_limit_row4_not_interior` (test_e4_cut_cell.py:256) — near-interior row matches closure (not raw interior stencil).
  - All tests PASS.

- [x] **25.1c** ψ=0 degenerate verified for E2_1 and E4_1:
  - E2_1: `TestConstructCutCellStencil::test_e2_1_degenerate_limit` (test_temo.py:1041) — PASS.
  - E2_1: `TestE2_1Integration::test_degeneracy_psi0` (test_temo.py:1574) — PASS.
  - E4_1: `TestE4TEMOConstruction::test_degenerate_limit` (test_e4_cut_cell.py:281) — PASS.
  - Zeroed column all zeros, wall column = B_u col 0.

### 25.2 — Diagnose the limit mismatch [NOT NEEDED]

- [x] **25.2a** No limit mismatch exists — diagnosis not needed. The TEMO construction correctly produces both limits via Category A prescriptions and limit-interpolated excess columns.

- [x] **25.2b** No fix needed — the current approach (Category A + limit interpolation for excess columns + Taylor solve for remaining) already guarantees correct limits by construction.

### 25.3 — Fix the TEMO construction [NOT NEEDED]

- [x] **25.3a** No modification needed — `identify_prescribed_entries` and `solve_temo_row` produce correct limit behavior.

- [x] **25.3b** No alternative fix needed.

### 25.4 — Validate both limits after fix [ALREADY VERIFIED]

- [x] **25.4a** All limit tests pass (no xfail markers needed):
  - ψ=1: wall column zeros for boundary rows, embedding matches B_u with same alphas.
  - ψ=0: TEM and continuity satisfied.
  - Test: `cd scripts/stencil_gen && uv run pytest tests/ -v -k "limit or degenerate"`

- [x] **25.4b** Taylor accuracy at intermediate ψ values verified:
  - `TestE4TEMOConstruction::test_taylor_accuracy_symbolic` (test_e4_cut_cell.py:294) — symbolic verification for all ψ.
  - `TestE4TEMOConstruction::test_taylor_accuracy_at_half` (test_e4_cut_cell.py:318) — ψ=1/2.
  - `TestE2_1Integration::test_taylor_accuracy_numeric` (test_temo.py) — ψ=0.3, 0.5, 0.7.
  - All tests PASS.

### 25.5 — Verify E2_1 against existing C++ [ALREADY VERIFIED]

- [x] **25.5a** E2_1 numerical validation:
  - `TestE2_1Integration` (test_temo.py) covers Dirichlet, floating, and Neumann at multiple ψ values against C++ reference data.
  - `TestE2_2Integration` (test_temo.py) — same for E2_2.
  - All tests PASS.

### 25.6 — Install Phase 24 E4_1 codegen output

- [ ] **25.6a** Copy updated E4_1.cpp to src/stencils:
  - Source: `scripts/stencil_gen/output/E4_1.cpp` (9436 bytes, R=5, 4 alphas, conservation-applied)
  - Target: `src/stencils/E4_1.cpp` (6281 bytes, R=4, 5 alphas, pre-conservation)
  - Key changes: R changed from 4 to 5, alpha array from 5 to 4 elements, conservation substitution applied, additional near-interior row coefficients.
  - Command: `cp scripts/stencil_gen/output/E4_1.cpp src/stencils/E4_1.cpp`

- [ ] **25.6b** Copy updated E4_1.t.cpp to src/stencils:
  - Source: `scripts/stencil_gen/output/E4_1.t.cpp` (6190 bytes)
  - Target: `src/stencils/E4_1.t.cpp` (5139 bytes)
  - Command: `cp scripts/stencil_gen/output/E4_1.t.cpp src/stencils/E4_1.t.cpp`

- [ ] **25.6c** Build and run E4_1 C++ tests:
  - Build: `cmake --build build --target t-E4_1`
  - Test: `ctest --test-dir build -R t-E4_1`
  - If build fails, check that the generated code uses correct includes and matches the stencil struct interface (see existing `src/stencils/E2_1.cpp` for reference).
  - File: `src/stencils/E4_1.cpp`, `src/stencils/E4_1.t.cpp`

- [ ] **25.6d** Run full stencil test suite to verify no regressions:
  - `ctest --test-dir build -L stencils`
  - Ensure existing E2_1, E2_2 tests still pass.
