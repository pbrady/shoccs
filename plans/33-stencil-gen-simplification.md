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

- [x] **33.2a** Extract `_unit_rhs(n_eqs, nu)` helper:
  - Defined in `taylor_system.py` (leaf of the dependency chain, no internal imports).
  - Imported into `temo.py` and used at all 10 former call sites.
  - `taylor_system.py` itself also uses the helper (1 call site).
  - Total: 11 duplicated expressions replaced with 11 calls to a single helper.
  - Files: `scripts/stencil_gen/stencil_gen/taylor_system.py`, `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: 481 passed, 1 xfailed (fast subset)

- [x] **33.2b** Extract `solve_linear(A, b, unknowns)` helper:
  - Created `scripts/stencil_gen/stencil_gen/_util.py` with `solve_linear()` (linsolve + unpack + cancel).
  - Replaced 3 call sites: `interior.py:197-199`, `conservation.py:127-129`, `conservation.py:143-145`.
  - Files: `_util.py` (new), `interior.py`, `conservation.py`
  - Test: 481 passed, 1 xfailed (fast subset)

- [x] **33.2c** Use `temo.compute_dimensions` in `phs.py` instead of inlined formulas:
  - Replaced 3 inlined dimension computations in `build_diff_matrix_rbf`, `build_diff_matrix_mixed_epsilon`, `build_diff_matrix_rbf_penalty` with calls to `temo.compute_dimensions(p, q, 0, nextra, nu)`.
  - File: `scripts/stencil_gen/stencil_gen/phs.py`
  - Test: 131 passed

- [x] **33.2d** Extract bilinear linearization helper in `temo.py`:
  - Extracted `_solve_with_linearization(equations, w_syms, bilinear_syms, theta_prefix)` helper.
  - Replaced 2 call sites: `solve_uniform_conservation_direct` and `derive_cut_cell_mathematica`.
  - File: `scripts/stencil_gen/stencil_gen/temo.py`
  - Test: 257 passed, 1 xfailed

### 33.3 — Minor source cleanup

- [x] **33.3a** Simplify `_phi_val` redundant branch in `phs.py`:
  - Both the numeric and symbolic branches computed `Abs(r) ** m`. Collapsed into a single return after the `r == 0` check.
  - File: `scripts/stencil_gen/stencil_gen/phs.py`
  - Test: 481 passed, 1 xfailed (fast subset)

- [x] **33.3b** Fix no-op `* 1` multiplication in `boundary.py`:
  - Removed `* 1` from `n_alpha = (r - 2) * 1 + n_active_penultimate`.
  - File: `scripts/stencil_gen/stencil_gen/boundary.py`
  - Test: 481 passed, 1 xfailed (fast subset)

- [x] **33.3c** Remove redundant `isinstance` filter in `conservation.py`:
  - `free_params` is already `list[Symbol]` from `boundary.py`. Replaced filtering comprehension with `list()`.
  - File: `scripts/stencil_gen/stencil_gen/conservation.py`
  - Test: 481 passed, 1 xfailed (fast subset)

- [x] **33.3d** Clean up `_symbols` alias and hoist inner imports in `temo.py`:
  - `conservation.py`: removed `symbols as _symbols` alias, use `symbols` directly.
  - `temo.py`: hoisted `linear_eq_to_matrix`, `linsolve`, `QQ`, `symbols` to module-level import; removed 4 inner `from sympy import` blocks; replaced `sym_solve`/`sym_symbols` aliases with direct `solve`/`symbols` calls.
  - Files: `conservation.py`, `temo.py`
  - Test: 481 passed, 1 xfailed (fast subset)

### 33.4 — Test infrastructure: create conftest.py and shared helpers

- [x] **33.4a** Create `tests/conftest.py` with shared pipeline fixture:
  - Extracted `run_pipeline(p, nu, s)` into `conftest.py` with module-scoped `e4u_pipeline`, `e6u_pipeline`, `e8u_pipeline` fixtures.
  - Removed `_run_pipeline` and fixture defs from `test_boundary.py`.
  - Removed `_run_e4u_pipeline` from `test_codegen_e4u.py`; `e4u_data` now depends on shared `e4u_pipeline` fixture.
  - Cleaned unused imports (`derive_boundary`, `build_conservation_system`, `solve_conservation`, `Symbol`, `symbols`) from `test_codegen_e4u.py`.
  - Files: `tests/conftest.py` (new), `tests/test_boundary.py`, `tests/test_codegen_e4u.py`
  - Test: 481 passed, 1 xfailed (fast subset)

- [x] **33.4b** Add `assert_taylor_accuracy` shared helper to conftest:
  - Created `_check_taylor_accuracy(B_u, q, nu)` in `conftest.py` with a session-scoped `assert_taylor_accuracy` fixture returning it.
  - The helper checks: `sum_j c_j * (j - i)^m = m! * delta_{m, nu}` for `m = 0..max(q, nu)`.
  - Updated 5 call sites (not 8 as originally estimated — `test_boundary.py` and `test_phs.py` use different patterns):
    - `test_e4_cut_cell.py`: `TestE4UniformBoundary.test_taylor_accuracy`, `TestE4UniformConservation.test_taylor_accuracy`
    - `test_temo.py`: `TestUniformBoundary.test_e2_1_taylor_accuracy_per_row`, `TestUniformBoundary.test_e2_2_taylor_accuracy`, `TestDeriveUniformBoundaryForTemo.test_e2_1_taylor_accuracy`
  - Files: `tests/conftest.py`, `test_e4_cut_cell.py`, `test_temo.py`
  - Test: 481 passed, 1 xfailed (fast subset)

- [x] **33.4c** Import `_interior_contribution` from source instead of re-implementing:
  - Replaced local re-implementation in `test_boundary.py` (9 lines) with `from stencil_gen.conservation import _interior_contribution`.
  - File: `tests/test_boundary.py`
  - Test: 41 passed

### 33.5 — Test deduplication

- [x] **33.5a** Remove duplicate cross-validation tests in `test_interior.py`:
  - Deleted "Test group 5: Cross-validation" (3 tests) — exact duplicates of test group 3.
  - File: `tests/test_interior.py`
  - Test: 29 passed

- [x] **33.5b** Extract shared base class for epsilon sweep tests in `test_phs.py`:
  - Extracted `_EpsilonSweepBase` with `_sweep`, `_print_table`, `_params_str`, and all three test methods.
  - `TestEpsilonSweepE2` and `TestEpsilonSweepE4` now inherit from it and set only P, Q, NEXTRA, NU, LABEL.
  - File: `tests/test_phs.py`
  - Test: 131 passed

- [x] **33.5c** Cache `derive_e2_uniform_boundary` results in `test_temo.py`:
  - Added `e2_1_uniform` and `e2_2_uniform` module-scoped fixtures to `conftest.py`.
  - Updated ~48 call sites across 11 test classes in `test_temo.py` to use the fixtures.
  - Only 3 special-case calls remain (custom alpha_symbols, error tests) — these can't be cached.
  - Files: `tests/conftest.py`, `tests/test_temo.py`
  - Test: 202 passed (test_temo.py + test_boundary.py + test_codegen_e4u.py + test_interior.py)

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
