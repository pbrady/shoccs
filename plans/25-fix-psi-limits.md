# Phase 25: Fix Cut-Cell Stencil Behavior at ψ=0 and ψ=1

**Goal:** Ensure the cut-cell stencil B_l(ψ) satisfies:
- **ψ=1 (uniform limit):** B_l(1) embeds the uniform boundary B^u exactly — rows 1..r of cut-cell equal rows 0..r-1 of B^u (shifted by 1 for wall column), wall column = 0, and the near-interior row matches the interior stencil.
- **ψ=0 (degenerate limit):** B_l(0) satisfies Design Principles 1 (TEM) and 2 (continuity) — the zeroed column (x_0 for ν=1) is all zeros, and the wall column equals the uniform stencil's column 0.

Currently both limits have bugs — the ψ=1 embedding doesn't match (alpha symbols are shifted between rows) and the near-interior row has nonzero wall entries at ψ=1.

**Depends on:** Phase 24

**Read first:**
- `scripts/stencil_gen/stencil_gen/temo.py`:
  - `construct_cut_cell_stencil` (line 1335) — main TEMO procedure
  - `solve_temo_row` (line 1245) — per-row Taylor solve, introduces beta symbols for underdetermined rows
  - `solve_uniform_limit` (line 1000) — computes B_l(1) for prescriptions
  - `build_degenerate_stencil` (line 536) — computes B_l(0) for prescriptions
  - `identify_prescribed_entries` — determines which entries of each row are prescribed vs solved
- `scripts/stencil_gen/tests/test_temo.py` — existing E2_1 limit tests (if any)
- `scripts/stencil_gen/tests/test_e4_cut_cell.py` — E4_1 tests
- `plans/stencil-derivation-math-reference.md` (Section 4.2-4.3: Design Principles and B_l(ψ))

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/ -v --timeout=120
cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "limit" --timeout=120
```

---

## Problem Analysis

### ψ=1 limit (uniform)
At ψ=1, B_l(1) should embed B^u:
- Cut-cell row 0: wall=0, cols 1..T-1 = B_u row 0 (same alpha symbols)
- Cut-cell row i+1: wall=0, cols 1..T-1 = B_u row i (same alpha symbols)
- Near-interior row r: interior stencil embedded (wall=0)

**Current bug:** `solve_temo_row` solves each row from scratch using Taylor accuracy with ψ-dependent deltas. The alpha symbols from B_u enter only through `prescribed` entries (from `identify_prescribed_entries`). If the prescriptions don't fully constrain the row, `solve_temo_row` introduces new beta symbols or solves for different combinations of the alphas. At ψ=1, the Taylor-solved entries should reduce to B_u entries, but they use the wrong alpha symbols (e.g., cut-cell row 1 uses alpha_2,alpha_3 where uniform row 0 uses alpha_0,alpha_1).

**Root cause:** The prescribed entries likely only fix a subset of columns (e.g., the zeroed column and the extra columns from nextra), leaving the Taylor system to solve the remaining columns. But the Taylor system at arbitrary ψ produces expressions in (ψ, alpha) that don't reduce to B_u's expressions at ψ=1 because the alpha symbols in the Taylor solve are free parameters of the psi-dependent system, not the B_u alphas.

**The paper's approach:** The cut-cell stencil coefficients are defined by:
1. Category A (zeroed column): α_{i,δ}(ψ) = ψ · target (sends one column to zero at ψ=0)
2. Category B (limit-interpolated): α_{ij}(ψ) = ψ·B_l_1[i,j] + (1-ψ)·B_d[i,j] — linear interpolation between the two limits
3. Category C (Taylor-determined): the remaining entries are solved from the accuracy system

**The fix:** MORE entries should be Category B (limit-interpolated) rather than Category C (Taylor-solved). If all non-Category-A entries are limit-interpolated, then at ψ=1 they equal B_l_1 (which embeds B_u) and at ψ=0 they equal B_d (which satisfies design principles). The Taylor accuracy is then a VERIFICATION, not a determination.

Alternatively, the current approach works IF `identify_prescribed_entries` prescribes enough entries so that the Taylor solve at arbitrary ψ uniquely determines the remaining entries to be consistent with B_u's alphas. This requires checking that the prescribed entries carry the alpha symbols through correctly.

### ψ=0 limit (degenerate)
Row 0 at ψ=0 should NOT be all zeros — it's the degenerate boundary point whose stencil equals B_u row 0 shifted to the wall column. This was verified correct for E2_1. But the near-interior row at ψ=0 may have issues.

---

## Items

### 25.1 — Add limit verification tests

- [ ] **25.1a** Add ψ=1 embedding test for E2_1:
  - Construct E2_1 cut-cell stencil
  - Substitute ψ=1
  - Verify: wall column all zeros
  - Verify: cut-cell rows 1..3 cols 1..4 exactly equal B_u rows 0..2 cols 0..3 (same alpha symbols)
  - Verify: cut-cell row 0 cols 1..4 equals B_u row 0 (extra cut-cell point at ψ=1 = boundary point 0)
  - Mark as xfail if currently failing
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py` (or `test_temo.py`)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/ -v -k "psi1_embedding" --timeout=60`

- [ ] **25.1b** Add ψ=1 embedding test for E4_1:
  - Same checks as 25.1a but for E4_1 (5×7 cut-cell vs 4×6 uniform)
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

- [ ] **25.1c** Add ψ=0 degenerate test for E2_1 and E4_1:
  - Category A: zeroed column all zeros ✓ (already verified)
  - TEM: wall column = B_u column 0 for rows 0..r-1 ✓ (verified for E2_1)
  - Row 0 at ψ=0: not all zeros (carries degenerate stencil coefficients)
  - Near-interior row at ψ=0: check it matches B_d's near-interior row
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`

### 25.2 — Diagnose the limit mismatch

- [ ] **25.2a** Trace `identify_prescribed_entries` for E2_1 row 1 (cut-cell row 2):
  - What entries does it prescribe? Which are Category A, B, or C?
  - At ψ=1, do the prescribed entries reduce to B_u's values?
  - Do the Taylor-solved (Category C) entries at ψ=1 match B_u?
  - If not: the Category C entries are the problem — they should be Category B (limit-interpolated)
  - File: add diagnostic prints or a targeted test

- [ ] **25.2b** Check if making ALL non-Category-A entries Category B (limit-interpolated) fixes ψ=1:
  - For each entry: α_{ij}(ψ) = ψ·B_l_1[i,j] + (1-ψ)·B_d[i,j]
  - This guarantees correct limits by construction
  - Then verify Taylor accuracy is still satisfied (it should be, by the paper's construction)
  - This is the cleanest fix if Taylor accuracy is automatically satisfied

### 25.3 — Fix the TEMO construction

- [ ] **25.3a** Modify `identify_prescribed_entries` (or the prescription logic in `construct_cut_cell_stencil`) to make more entries Category B:
  - Current: only a few entries are prescribed (Category A zeroed col + some nextra entries)
  - Fix: prescribe ALL entries as Category B (limit interpolation) except those that must be Category C
  - The Category C entries are those where neither limit is known — but both limits ARE known (B_l_1 and B_d are computed), so all entries can be Category B
  - If all entries are Category B, `solve_temo_row` is not needed at all — the stencil is fully determined by limit interpolation
  - But we still need to verify Taylor accuracy
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

- [ ] **25.3b** Alternative: fix `solve_temo_row` to use the correct alpha symbols:
  - Instead of introducing new beta symbols for underdetermined rows, use the alpha symbols from B_u
  - The prescription entries should carry enough alpha information to determine the row uniquely
  - This preserves the current Category C approach but ensures alpha consistency
  - File: `scripts/stencil_gen/stencil_gen/temo.py`

### 25.4 — Validate both limits after fix

- [ ] **25.4a** Remove xfail markers from limit tests:
  - ψ=1: wall column zeros, embedding matches B_u with same alphas
  - ψ=0: TEM and continuity (Design Principles 1 and 2) satisfied
  - Test: `cd scripts/stencil_gen && uv run pytest tests/ -v -k "limit" --timeout=120`

- [ ] **25.4b** Validate Taylor accuracy at intermediate ψ values:
  - At ψ=0.3, 0.5, 0.7: verify each row is accurate to order q for polynomials
  - This ensures the limit-interpolation approach doesn't break accuracy

### 25.5 — Verify E2_1 against existing C++

- [ ] **25.5a** Re-run E2_1 numerical validation against E2_1.cpp:
  - Substitute specific (ψ, alpha) values
  - Compare floating and Dirichlet coefficients
  - Any change in E2_1 output is a regression
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "E2_1" --timeout=120`

### 25.6 — Re-generate E4_1 C++ and rebuild

- [ ] **25.6a** Re-generate E4_1.cpp with limit-corrected stencil:
  - Write to `scripts/stencil_gen/output/E4_1.cpp`
  - Copy to `src/stencils/E4_1.cpp`
  - Build and test: `cmake --build build --target t-E4_1 && ctest --test-dir build -R t-E4_1`
  - File: `src/stencils/E4_1.cpp`, `src/stencils/E4_1.t.cpp`
