# Phase 24: Uniform-Mesh Conservation BEFORE TEMO Construction

**Goal:** Apply SBP conservation constraints to the uniform boundary stencil at TEMO dimensions BEFORE constructing the cut-cell stencil. This is what the paper says: "the conservation constraints must also be solved on the uniform mesh to provide appropriately constrained αu."

**Depends on:** Phase 23 (correct dimensions r=4 for E4_1)

**Read first:**
- `scripts/stencil_gen/stencil_gen/temo.py` (function `derive_uniform_boundary_for_temo` — currently does NOT apply conservation)
- `scripts/stencil_gen/stencil_gen/conservation.py` (functions `build_conservation_system` and `solve_conservation` — already work for E4u_1 at SBP dimensions)
- `scripts/stencil_gen/tests/test_boundary.py` (E4u_1 conservation tests — shows how conservation.py is used)
- `scripts/stencil_gen/stencil_gen/boundary.py` (function `solve_boundary_row` — used by temo.py)

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/ -v --timeout=120
cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "uniform_conservation" --timeout=120
```

---

## The Exact Fix

`derive_uniform_boundary_for_temo(E4_1)` currently returns a 4×6 uniform boundary with 5 free alphas and NO conservation enforcement. The fix:

1. After solving all 4 boundary rows with `solve_boundary_row`, call `build_conservation_system(r=4, t=6, p=2, rows, interior)` from conservation.py
2. This gives 5 equations, 6 unknowns (4 weights + 2 last-row free params)
3. Call `solve_conservation(eqs, w_syms, last_free, rows, r=4)` to solve
4. This constrains 1 of the 2 last-row free params, leaving 4 total free alphas
5. Return the conservation-constrained B_u and weights

This was already verified to work manually:
```python
build_conservation_system(r=4, t=6, p=2, rows, interior)
→ 5 equations, 4 weight unknowns + 2 last-row free = 6 unknowns
→ SOLVABLE (underdetermined by 1)
```

## Items

### 24.1 — Add conservation to `derive_uniform_boundary_for_temo`

- [ ] **24.1a** Modify `derive_uniform_boundary_for_temo` in `temo.py` to apply conservation:
  - After the existing loop that calls `solve_boundary_row` for each of r rows
  - Add these lines (pseudo-code):
    ```python
    from .conservation import build_conservation_system, solve_conservation

    # Apply SBP conservation to the uniform boundary at TEMO dimensions
    eqs, w_syms, last_free = build_conservation_system(r, t, p, rows, interior_coeffs)
    sol_dict, updated_rows = solve_conservation(eqs, w_syms, last_free, rows, r)

    # Build the B_u matrix from updated (conservation-constrained) rows
    B_u = Matrix([[updated_rows[i][j] for j in range(t)] for i in range(r)])

    # Extract the remaining (unconstrained) alpha symbols
    remaining_alphas = sorted(B_u.free_symbols, key=str)

    # Store weights for later use
    weights = [sol_dict.get(w, w) for w in w_syms]
    ```
  - Add `weights` field to `UniformResult` namedtuple
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "E2" --timeout=60` (E2 tests must still pass)

### 24.2 — Test E4_1 uniform conservation

- [ ] **24.2a** Add test verifying E4_1 uniform boundary has conservation:
  - Call `derive_uniform_boundary_for_temo(E4_1)`
  - Verify: B_u shape is (4, 6)
  - Verify: exactly 4 free alpha symbols remain (was 5 before conservation)
  - Verify: column sums satisfy conservation (using the returned weights)
  - Verify: substituting the optimized alpha values reproduces valid numerical coefficients
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "uniform_conservation" --timeout=60`

### 24.3 — Verify cut-cell conservation follows

- [ ] **24.3a** After TEMO construction with the conservation-constrained B_u, verify column sums:
  - Build the cut-cell stencil using `construct_cut_cell_stencil` with the new B_u
  - For each interior column j, check `Σ_i w_i(ψ) · B[i,j] = 0`
  - The weights w_i(ψ) should be derivable from the uniform weights via the TEMO extension
  - If this doesn't hold automatically: the TEMO construction may need a conservation step too
  - But the paper suggests it should follow from uniform conservation by construction
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "cutcell_conservation" --timeout=120`

### 24.4 — Update E4_1 alpha count and re-generate C++

- [ ] **24.4a** Update E4_1 tests for 4 free alphas (was 5):
  - Fix any assertions about alpha count
  - Re-generate E4_1.cpp with conservation-constrained stencil
  - File: `scripts/stencil_gen/tests/test_e4_cut_cell.py`, `scripts/stencil_gen/output/E4_1.cpp`

### 24.5 — Regression: E2_1 unchanged

- [ ] **24.5a** Verify E2_1 tests still pass:
  - E2_1 already had conservation via `derive_e2_uniform_boundary`
  - The new code path should produce identical results
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v --timeout=120`
