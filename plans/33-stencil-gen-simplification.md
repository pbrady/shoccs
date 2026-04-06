# Phase 33: stencil_gen Python Simplification

**Goal:** Remove dead code, consolidate duplicated patterns, and reduce test
boilerplate in the `scripts/stencil_gen/` SymPy-based stencil derivation pipeline.
Estimated ~200 lines of dead code removed, ~150 lines of duplication consolidated,
and test infrastructure modernized with shared fixtures.

**Depends on:** None (independent of C++ phases)

**Read first:**
- `scripts/stencil_gen/stencil_gen/phs.py` (1276 lines — dead functions, redundant branches, inlined dimension formulas)
- `scripts/stencil_gen/stencil_gen/temo.py` (3340 lines — 11 unit-RHS copies, duplicated linearization, inner imports)
- `scripts/stencil_gen/stencil_gen/interior.py` (205 lines — unused import, dead variable)
- `scripts/stencil_gen/stencil_gen/conservation.py` (167 lines — duplicated linsolve pattern, redundant filter)
- `scripts/stencil_gen/stencil_gen/taylor_system.py` (43 lines — shared taylor_coeff candidate)
- `scripts/stencil_gen/stencil_gen/codegen.py` (677 lines — duplicated h-division emission)
- `scripts/stencil_gen/stencil_gen/boundary.py` (153 lines — no-op `* 1`)
- `scripts/stencil_gen/tests/test_phs.py` (5971 lines — duplicated sweep classes)
- `scripts/stencil_gen/tests/test_temo.py` (1878 lines — uncached fixture calls)
- `scripts/stencil_gen/tests/test_boundary.py` (887 lines — re-implemented helper, duplicated pipeline)
- `scripts/stencil_gen/tests/test_interior.py` (204 lines — duplicate cross-validation)
- `scripts/stencil_gen/tests/test_codegen_e4u.py` (245 lines — duplicated pipeline runner)

**Test commands:**
```bash
cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q
# Fast subset (skip slow full-pipeline tests):
cd scripts/stencil_gen && uv run pytest tests/ -x -q -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"
```

---

## Items

### 33.1 — Dead code removal (source)

- [x] **33.1a** Remove dead `_phs_kernel_1d` function from `phs.py`:
  - Delete `_phs_kernel_1d` (lines 38-54). It is never called — superseded by `_phi_val`.
  - File: `scripts/stencil_gen/stencil_gen/phs.py`
  - Test: `uv run pytest tests/test_phs.py -x -q`

- [x] **33.1b** Remove dead `_phs_kernel_1d_deriv` function from `phs.py`:
  - Delete `_phs_kernel_1d_deriv` (lines 57-143, ~87 lines). Never called — superseded by `_eval_phs_deriv`. Includes ~30 lines of scratch-work comments.
  - File: `scripts/stencil_gen/stencil_gen/phs.py`
  - Test: `uv run pytest tests/test_phs.py -x -q`

- [x] **33.1c** ~~Remove unreachable code in `_eval_phs_deriv` in `phs.py`~~ **SKIPPED — not dead code.**
  - The numeric branch (lines 615-617) does NOT return; it falls through to lines 639-642. Only the symbolic branch (618-637) returns in all sub-paths. Lines 639-642 are the return path for numeric r_val and are reachable.

- [x] **33.1d** Remove unused imports and dead variables:
  - `interior.py` line 10: `zeros` imported but never used.
  - `interior.py` line 121: `equations_lhs = []` assigned but never appended to or read.
  - `temo.py` line 12: `Integer` imported but never used.
  - Files: `interior.py`, `temo.py`
  - Test: `uv run pytest tests/ -x -q -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"`

- [x] **33.1e** Delete `tests/test_placeholder.py`:
  - Contains only a trivial import-check test. Hundreds of substantive tests already verify imports.
  - File: `scripts/stencil_gen/tests/test_placeholder.py`
  - Test: `uv run pytest tests/ -x -q -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"`

### 33.2 — Extract shared helpers (source)

- [ ] **33.2a** Extract `_unit_rhs(n_eqs, nu)` helper in `temo.py`:
  - The pattern `Matrix(n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0))` appears 11 times in `temo.py` plus once in `taylor_system.py`.
  - Create a module-level helper and replace all 12 call sites.
  - Files: `scripts/stencil_gen/stencil_gen/temo.py`, `scripts/stencil_gen/stencil_gen/taylor_system.py`
  - Test: `uv run pytest tests/ -x -q -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"`

- [ ] **33.2b** Extract `solve_linear(A, b, unknowns)` helper:
  - The `linsolve` + unpack + dict-comprehension + `cancel()` pattern is repeated at `interior.py:198-200`, `conservation.py:127-129`, and `conservation.py:143-145`.
  - Create a shared helper (e.g., in a new `scripts/stencil_gen/stencil_gen/_util.py` or in `interior.py` and import elsewhere). Replace all 3 call sites.
  - Files: `interior.py`, `conservation.py`
  - Test: `uv run pytest tests/test_interior.py tests/test_boundary.py -x -q`

- [ ] **33.2c** Use `temo.compute_dimensions` in `phs.py` instead of inlined formulas:
  - The `t` and `r` computation from `(p, q, nu, nextra)` is inlined in 3 places in `phs.py` (`build_diff_matrix_rbf`, `build_diff_matrix_mixed_epsilon`, `build_diff_matrix_rbf_penalty`).
  - Replace with calls to `temo.compute_dimensions` (or equivalent).
  - File: `scripts/stencil_gen/stencil_gen/phs.py`
  - Test: `uv run pytest tests/test_phs.py -x -q`

- [ ] **33.2d** Extract bilinear linearization helper in `temo.py`:
  - The ~35-line pattern (introduce theta symbols, substitute, `linear_eq_to_matrix`/`linsolve`, recover originals by dividing theta by weight) is duplicated at `temo.py:1135-1170` and `temo.py:1644-1669`.
  - Extract a `_solve_linearized_bilinear(equations, w_syms, bilinear_syms)` helper.
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: `uv run pytest tests/test_temo.py tests/test_e4_cut_cell.py -x -q`

### 33.3 — Minor source cleanup

- [ ] **33.3a** Simplify `_phi_val` redundant branch in `phs.py`:
  - Both the numeric and symbolic branches at lines 573-581 compute `Abs(r) ** m`. Collapse into a single return after the `r == 0` check.
  - File: `scripts/stencil_gen/stencil_gen/phs.py`
  - Test: `uv run pytest tests/test_phs.py -x -q`

- [ ] **33.3b** Fix no-op `* 1` multiplication in `boundary.py`:
  - Line 119: `n_alpha = (r - 2) * 1 + n_active_penultimate` — remove the `* 1`.
  - File: `scripts/stencil_gen/stencil_gen/boundary.py`
  - Test: `uv run pytest tests/test_boundary.py -x -q`

- [ ] **33.3c** Remove redundant `isinstance` filter in `conservation.py`:
  - Line 66: `[s for s in boundary_rows[r-1].free_params if isinstance(s, Symbol)]` — `free_params` is already filtered to Symbol instances in `boundary.py`.
  - File: `scripts/stencil_gen/stencil_gen/conservation.py`
  - Test: `uv run pytest tests/test_boundary.py -x -q`

- [ ] **33.3d** Clean up `_symbols` alias and hoist inner imports in `temo.py`:
  - `conservation.py` line 10: `symbols as _symbols` alias is unnecessary.
  - `temo.py`: hoist repeated inner imports of `linear_eq_to_matrix`, `linsolve`, `symbols` to module level.
  - Files: `conservation.py`, `temo.py`
  - Test: `uv run pytest tests/ -x -q -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"`

### 33.4 — Test infrastructure: create conftest.py and shared helpers

- [ ] **33.4a** Create `tests/conftest.py` with shared pipeline fixture:
  - Extract `_run_pipeline(p, nu, s)` from `test_boundary.py:23-32` (identical to `test_codegen_e4u.py:27-36`) into a shared fixture/helper in `conftest.py`.
  - Add module-scoped fixtures for `e4u_pipeline` (p=2, nu=1, s=0) and common schemes.
  - Update `test_boundary.py` and `test_codegen_e4u.py` to use the shared fixture.
  - Files: `tests/conftest.py` (new), `tests/test_boundary.py`, `tests/test_codegen_e4u.py`
  - Test: `uv run pytest tests/test_boundary.py tests/test_codegen_e4u.py -x -q`

- [ ] **33.4b** Add `assert_taylor_accuracy` shared helper to conftest:
  - The Taylor accuracy check (compute moment sums, assert against expected derivative) is copy-pasted across 8 locations in `test_boundary.py`, `test_temo.py`, `test_e4_cut_cell.py`, `test_phs.py`.
  - Create `assert_taylor_accuracy(B_u, q, nu)` in `conftest.py`.
  - Update all 8 call sites to use it.
  - Files: `tests/conftest.py`, `test_boundary.py`, `test_temo.py`, `test_e4_cut_cell.py`, `test_phs.py`
  - Test: `uv run pytest tests/ -x -q -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"`

- [ ] **33.4c** Import `_interior_contribution` from source instead of re-implementing:
  - `test_boundary.py:35-43` re-implements `_interior_contribution` locally. `test_e4_cut_cell.py` already imports it from `stencil_gen.conservation`.
  - Replace the local copy with the import.
  - File: `tests/test_boundary.py`
  - Test: `uv run pytest tests/test_boundary.py -x -q`

### 33.5 — Test deduplication

- [ ] **33.5a** Remove duplicate cross-validation tests in `test_interior.py`:
  - "Test group 5: Cross-validation" (lines ~165-186) asserts the exact same gamma values as "Test group 3" above it. Delete group 5.
  - File: `tests/test_interior.py`
  - Test: `uv run pytest tests/test_interior.py -x -q`

- [ ] **33.5b** Extract shared base class for epsilon sweep tests in `test_phs.py`:
  - `TestEpsilonSweepE2` and `TestEpsilonSweepE4` have identical `_sweep`, `_print_table`, and sweep test method implementations. Only the class constants (P, Q, NEXTRA, NU) differ.
  - Extract a `_EpsilonSweepBase` class with the shared methods; have both inherit from it and set only class-level constants.
  - File: `tests/test_phs.py`
  - Test: `uv run pytest tests/test_phs.py -x -q -k "EpsilonSweep"`

- [ ] **33.5c** Cache `derive_e2_uniform_boundary` results in `test_temo.py`:
  - Multiple test classes independently call `derive_e2_uniform_boundary(nu=1)` and `derive_e2_uniform_boundary(nu=2)` — at least 18 times total with no caching.
  - Add `@pytest.fixture(scope="module")` fixtures for E2_1 and E2_2 uniform results in `conftest.py` or at the top of `test_temo.py`.
  - Update test classes to use the fixtures.
  - File: `tests/test_temo.py` (and optionally `tests/conftest.py`)
  - Test: `uv run pytest tests/test_temo.py -x -q`

- [ ] **33.5d** Replace element-wise assertion loops with `pytest.approx`:
  - Multiple tests use `for i, (got, want) in enumerate(zip(vals, ref)): assert abs(got - want) < tol` — this should be `assert vals == pytest.approx(ref, abs=tol)`.
  - Locations: `test_codegen_e4u.py:85-86`, `test_codegen.py:544-545`, `test_codegen.py:569-570`, and similar patterns in `test_boundary.py`.
  - Files: `test_codegen_e4u.py`, `test_codegen.py`, `test_boundary.py`
  - Test: `uv run pytest tests/test_codegen_e4u.py tests/test_codegen.py tests/test_boundary.py -x -q`

### 33.6 — Final verification

- [ ] **33.6a** Run full test suite:
  - `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q`
  - All tests must pass. No behavior changes.

---

## Ordering

```
33.1a-e (dead code removal) — independent, do first
33.2a (unit_rhs helper) — independent
33.2b (solve_linear helper) — independent
33.2c (compute_dimensions in phs) — independent
33.2d (linearization helper) — independent
33.3a-d (minor cleanup) — independent, do after 33.1 to avoid line-number conflicts
33.4a (conftest.py) — do before 33.4b-c and 33.5
33.4b (assert_taylor_accuracy) — requires 33.4a
33.4c (import interior_contribution) — requires 33.4a
33.5a (remove duplicate tests) — independent
33.5b (sweep base class) — independent
33.5c (cache fixtures) — can use conftest from 33.4a
33.5d (pytest.approx) — independent
33.6a (verify) — last
```

33.1 items should go first to avoid line-number drift. 33.4a must precede 33.4b-c and 33.5c.

---

## Completion Criteria

- Zero dead functions in `phs.py` (`_phs_kernel_1d`, `_phs_kernel_1d_deriv` deleted).
- No unused imports or dead variables in source files.
- Unit-RHS pattern appears once (as a helper), not 12 times.
- `linsolve` + unpack pattern consolidated into shared helper.
- Dimension computation in `phs.py` delegates to `temo.compute_dimensions`.
- `tests/conftest.py` exists with shared fixtures and helpers.
- No duplicate cross-validation tests in `test_interior.py`.
- Epsilon sweep classes share a base class in `test_phs.py`.
- All tests pass.
