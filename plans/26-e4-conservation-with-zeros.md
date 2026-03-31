# Phase 26: E4_1 Cut-Cell Conservation via Zero-Constrained Alphas

**Goal:** Implement full symbolic cut-cell conservation for E4_1 by applying zero constraints to the last row's free parameters (alpha_3=0, alpha_4=0) before running TEMO and solving conservation. This was proven tractable: SymPy solves the system in ~1.2 seconds when the zeros are applied first, producing clean polynomial weights in psi.

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
- Free parameters: alpha^u_{04}, alpha^u_{14}, alpha^u_{24}, alpha^u_{25}
- **Zeros: alpha^u_{05}=0, alpha^u_{15}=0**

### Alpha Mapping: Paper vs Pipeline

The pipeline's alpha distribution in `derive_uniform_boundary_for_temo` (lines 370-399) differs from Table 1:

| Row | Col 4 (free pos 1) | Col 5 (free pos 2) | Pipeline alpha | Paper notation |
|-----|--------------------|--------------------|----------------|----------------|
| 0   | alpha_0            | 0 (hardcoded)      | alpha_0        | alpha^u_{04} free, alpha^u_{05}=0 |
| 1   | alpha_1            | 0 (hardcoded)      | alpha_1        | alpha^u_{14} free, alpha^u_{15}=0 |
| 2   | alpha_2            | 0 (hardcoded)      | alpha_2        | alpha^u_{24} free, alpha^u_{25} free |
| 3   | alpha_3            | alpha_4            | alpha_3, alpha_4 | (conservation-determined) |

The pipeline assigns 1 free parameter per early row (col 4 only; col 5 hardcoded to 0) and 2 to the last row. The paper's zeros (alpha^u_{05}=0, alpha^u_{15}=0) are **already satisfied** by the pipeline's convention. The plan's **additional** zeros target **alpha_3=0 and alpha_4=0** — the last row's free parameters at B_u[3,4] and B_u[3,5].

Setting alpha_3=alpha_4=0 fully determines the last row by its Taylor equations alone, leaving 3 free parameters: alpha_0, alpha_1, alpha_2.

### Research Result

With alpha_3=alpha_4=0, `sympy.solve()` produces a clean single-branch solution in ~1.2 seconds:
- w_1, w_2, w_3 are degree-3 polynomials in psi (linear in w_4)
- alpha_0 is a rational function of psi and w_4
- alpha_1 depends on psi, w_4, and alpha_2
- 2 remaining free parameters: alpha_2 and w_4

## The Procedure

1. Build uniform boundary B_u at TEMO dimensions (r=4, t=6) with all 5 alpha symbols
2. **Set alpha_3=0, alpha_4=0** — substitute into B_u, remove from alpha list → 3 alphas remain
3. Build cut-cell B_l(psi) via TEMO from the zero-constrained B_u (3 free alphas)
4. Build cut-cell conservation system: 5 equations in 4 weights + 3 alphas = 7 unknowns
5. Solve with `sympy.solve()` for [alpha_0, alpha_1, w_1, w_2, w_3] — feasible (~1.2s)
6. Back-substitute to get the conservative stencil as functions of (psi, alpha_2, w_4)
7. Rename: alpha_2 → alpha_0, w_4 → alpha_1 for codegen uniformity (2 free params)

**Important:** Uniform conservation (`conserve=True` from Phase 24) is **NOT used** in this path. With alpha_3=alpha_4=0 the last row is fully determined by Taylor matching, so uniform conservation compatibility conditions are moot. Conservation is enforced exclusively at the cut-cell level via `build_cut_cell_conservation_system`.

---

## Items

### 26.1 — Apply zero constraints in `derive_uniform_boundary_for_temo`

- [x] **26.1a** Add `zeros` field to `SchemeParams` and `zeros` parameter to `derive_uniform_boundary_for_temo`:
  - **Files:** `scripts/stencil_gen/stencil_gen/temo.py`
  - **SchemeParams change (line 92):** Add `zeros: tuple[int, ...] = ()` field to the frozen dataclass. Update E4_1 definition (line 124) to:
    ```python
    E4_1 = SchemeParams(p=2, q=3, s=0, nextra=0, nu=1, zeros=(3, 4))
    ```
    E2_1, E2_2, E4_2 remain unchanged (empty `zeros` tuple, default).
  - **`derive_uniform_boundary_for_temo` change (line 315):** Add parameter `zeros: set[int] | None = None`:
    ```python
    def derive_uniform_boundary_for_temo(
        scheme: SchemeParams,
        alpha_symbols: list[Symbol] | None = None,
        conserve: bool = False,
        zeros: set[int] | None = None,
    ) -> UniformResult:
    ```
  - **Mutual exclusion check:** After parameter validation, add:
    ```python
    if zeros and conserve:
        raise ValueError(
            "zeros and conserve=True are mutually exclusive; "
            "use cut-cell conservation instead"
        )
    ```
  - **Post-hoc substitution:** After `B_u = Matrix(rows)` (line 436) and before the conservation step (line 439), insert:
    ```python
    if zeros:
        zero_subs = {alpha_symbols[k]: S.Zero for k in zeros}
        B_u = B_u.subs(zero_subs)
        alpha_symbols = [s for k, s in enumerate(alpha_symbols) if k not in zeros]
    ```
    This reduces E4_1 from 5 to 3 alpha_symbols.
  - **`alpha_symbols` interaction:** When `zeros` is used, `alpha_symbols` must be None (auto-created) or have the pre-zeros count (5 for E4_1). The returned `UniformResult.alpha_symbols` has the post-zeros count (3). This is fine because 26.5a always passes `alpha_symbols=None`.
  - **No other logic changes needed:** The nextra>0 and conserve branches are unaffected since E4_1 has nextra=0 and zeros+conserve is forbidden.
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "E2" --timeout=60` (E2 unchanged)

- [x] **26.1b** Test E4_1 uniform boundary with zeros:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Add new test class `TestE4UniformBoundaryWithZeros` with fixture:
    ```python
    @pytest.fixture(scope="class")
    def e4_zeroed(self):
        return derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})
    ```
  - Tests:
    - `test_shape`: B_u shape is (4, 6)
    - `test_three_alpha_symbols`: 3 free symbols (alpha_0, alpha_1, alpha_2)
    - `test_last_row_zeroed`: `B_u[3, 4] == 0` and `B_u[3, 5] == 0`
    - `test_early_row_col5_still_zero`: `B_u[0, 5] == B_u[1, 5] == B_u[2, 5] == 0`
    - `test_taylor_accuracy`: All 4 rows satisfy `max(q+1, nu+1)=4` Taylor equations (import `_build_uniform_vandermonde` from `stencil_gen.temo`)
    - `test_zeros_conserve_mutual_exclusion`: `derive_uniform_boundary_for_temo(E4_1, zeros={3,4}, conserve=True)` raises ValueError
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "WithZeros" --timeout=60`

### 26.1-followup — Fix `--timeout` in test commands

- [x] **26.1-followup** All test commands in this plan use `--timeout=N`, which requires `pytest-timeout`. That package is not in `scripts/stencil_gen/pyproject.toml` dependencies and is not installed. Every test command will fail with `unrecognized arguments: --timeout=...`. Fix: add `"pytest-timeout>=2.0"` to the `dependencies` list in `scripts/stencil_gen/pyproject.toml`, then run `uv sync` to install it.

### 26.2 — Build zero-constrained cut-cell stencil

- [x] **26.2a** Build and test E4_1 cut-cell with zero-constrained B_u:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Add test class `TestE4ZeroConstrainedCutCell` with fixture:
    ```python
    @pytest.fixture(scope="class")
    def e4_zeroed_cut_cell(self):
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
        )
        return stencil, ur, psi
    ```
  - Tests:
    - `test_shape`: 5x7 matrix (R=5, T=7)
    - `test_free_symbols`: `matrix.free_symbols - {psi}` yields exactly {alpha_0, alpha_1, alpha_2} — no alpha_3, alpha_4
    - `test_psi_1_limit`: At psi=1, matches the zero-constrained B_u (via `solve_uniform_limit`)
    - `test_taylor_accuracy_at_half`: At psi=1/2, all 5 rows satisfy q+1=4 Taylor equations
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "ZeroConstrainedCutCell" --timeout=120`

### 26.3 — Solve cut-cell conservation

- [x] **26.3a** Add `solve_cut_cell_conservation` function:
  - **File:** `scripts/stencil_gen/stencil_gen/temo.py` (insert after `build_cut_cell_conservation_system`, ~line 1487)
  - **Signature:**
    ```python
    def solve_cut_cell_conservation(
        equations: list[Expr],
        solve_for: list[Symbol],
    ) -> dict[Symbol, Expr]:
        """Solve pre-built cut-cell conservation equations.

        The caller is responsible for building the equations via
        ``build_cut_cell_conservation_system`` and choosing which symbols
        to solve for.  Any symbol in the equations that is NOT in
        ``solve_for`` is treated as a free parameter.

        Parameters
        ----------
        equations : list[Expr]
            Conservation equations (each must equal zero), as returned
            by ``build_cut_cell_conservation_system``.
        solve_for : list[Symbol]
            Symbols to solve for (e.g. [alpha_0, alpha_1, w_1, w_2, w_3]).

        Returns
        -------
        dict[Symbol, Expr]
            Maps each solved symbol to its expression in (psi, free_params).
        """
    ```
  - **Implementation:**
    1. `solution = sympy.solve(equations, solve_for, dict=True)` — expected: single-branch solution
    2. Assert `len(solution) == 1`, extract `sol = solution[0]`
    3. Verify: all equations evaluate to 0 after substitution (use `cancel(eq.subs(sol)) == 0`)
    4. Return `sol`
  - **Design note:** The function takes pre-built equations (not raw stencil params) so the caller can reuse the `w_syms` from `build_cut_cell_conservation_system` for weight assembly without redundant computation.
  - **Expected solve_for for E4_1:** `[alpha_0, alpha_1, w_1, w_2, w_3]` (5 unknowns for 5 equations)
  - **Expected free params:** `[alpha_2, w_4]` (2 free for optimization)
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "solve_cut_cell" --timeout=300`

- [x] **26.3b** Test conservation solution and apply to stencil:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Add test class `TestE4CutCellConservationSolution`:
    - Build zero-constrained cut-cell stencil (as in 26.2a)
    - Call `solve_cut_cell_conservation(...)` with `solve_for=[alpha_0, alpha_1, w_1, w_2, w_3]`
    - `test_solution_exists`: solution dict has 5 entries
    - `test_all_equations_satisfied`: all 5 conservation equations → 0 after substitution
    - `test_free_symbols_in_solution`: each solved expression involves only {psi, alpha_2, w_4}
    - `test_stencil_after_substitution`: apply `sol` to B_l via `B_l.xreplace(sol)`, verify resulting matrix has free symbols ⊆ {psi, alpha_2, w_4}
    - `test_conservation_column_sums`: after substitution, verify weighted column sums using the solved weights w_1..w_4 and w_0=psi
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "ConservationSolution" --timeout=300`

### 26.4 — Validate the conservative stencil

- [x] **26.4a** Update the xfail conservation test:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Renamed `test_e4_1_conservation_fails` → `test_e4_1_conservation_fails_without_zeros` (kept xfail)
  - Added `test_e4_1_conservation_with_zeros`: builds zero-constrained stencil, solves conservation, verifies symbolic + numeric column sums at psi=0.3, 0.5, 0.7
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "conservation_with_zeros" --timeout=300` ✓

- [x] **26.4b** Verify Taylor accuracy is preserved:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Added `test_e4_1_conservative_taylor_accuracy`: verifies q+1=4 Taylor moments at psi=0.3, 0.5, 0.7
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "conservative_taylor" --timeout=120` ✓

- [x] **26.4c** Verify stencil validity in the interior of (0, 1):
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - **Finding:** The conservation solution introduces poles at psi=0 and psi=1 (alpha_0 diverges at both boundaries). The stencil is valid only for psi in the open interval (0, 1), which is the physically meaningful range for cut cells.
  - Renamed test to `test_e4_1_conservative_psi_interior` (plan originally said `test_e4_1_conservative_psi_limits`, but the limits diverge)
  - Tests at psi=0.01, 0.1, 0.5, 0.9, 0.99: all entries finite, Taylor accuracy holds, conservation holds
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "conservative_psi_interior" --timeout=120` ✓
  - **Note for 26.5a:** The codegen output must document that coefficients diverge at psi=0 and psi=1. The C++ runtime should clamp psi away from boundaries or use the non-conservative stencil at full cells.

### 26.5 — Integrate into `derive_cut_cell_scheme`

- [x] **26.5a** Update `derive_cut_cell_scheme` to use zeros and cut-cell conservation:
  - **File:** `scripts/stencil_gen/stencil_gen/temo.py` (`derive_cut_cell_scheme`, line ~2172)
  - **Add a new code path** detected by `scheme.zeros` being non-empty:
    ```python
    # --- Zero-constrained cut-cell conservation path ---
    if scheme.zeros:
        # Step 1: Build uniform with zeros (no uniform conservation)
        uniform = derive_uniform_boundary_for_temo(
            scheme, zeros=set(scheme.zeros),
        )
        # Step 2: TEMO pipeline
        floating_result = construct_cut_cell_stencil(
            uniform.B_u, uniform.interior,
            scheme.p, scheme.q, scheme.nu, scheme.nextra, psi,
        )
        floating = floating_result.matrix
        # Step 3: Solve cut-cell conservation
        # solve_for = [alpha_0, alpha_1, w_1, w_2, w_3]
        # free = [alpha_2, w_4]
        eqs, w_syms = build_cut_cell_conservation_system(
            floating, dims.R, dims.T, scheme.p, scheme.nu,
            uniform.interior, psi,
        )
        solve_for = uniform.alpha_symbols[:2] + w_syms[:3]  # alpha_0, alpha_1, w_1..w_3
        sol = solve_cut_cell_conservation(eqs, solve_for)
        # Step 4: Apply solution to stencil
        floating = floating.xreplace(sol)
        # Step 5: Compute weights (w_0=psi, w_1..w_3 from sol, w_4 free)
        weights = [psi] + [sol[w] for w in w_syms[:3]] + [w_syms[3]]
        # Step 6: Rename remaining free params for codegen
        # alpha_2 -> alpha_0, w_4 -> alpha_1
        free_alpha = uniform.alpha_symbols[2]  # alpha_2
        free_w = w_syms[3]  # w_4
        n_free = 2  # E4_1 zeros path always produces exactly 2 free params
        if alpha_symbols is None:
            final = [Symbol(f"alpha_{k}") for k in range(n_free)]
        else:
            if len(alpha_symbols) != n_free:
                raise ValueError(
                    f"zeros path produces {n_free} free params, "
                    f"got {len(alpha_symbols)} alpha_symbols"
                )
            final = list(alpha_symbols)
        rename = {free_alpha: final[0], free_w: final[1]}
        floating = floating.xreplace(rename)
        weights = [cancel(w.xreplace(rename)) if hasattr(w, 'xreplace') else w
                   for w in weights]
        # Step 7: Assemble result
        return assemble_cut_cell_result(
            floating, None, None, dims, final,
            weights=weights,
        )
    ```
  - **Ordering constraint:** This block should go BEFORE the existing `if not conserve:` check (line ~2208) since `scheme.zeros` takes priority.
  - **`conserve` parameter interaction:** When `scheme.zeros` is non-empty, the zeros path is always taken regardless of `conserve`. The `conserve` parameter is silently ignored — cut-cell conservation is always applied via `solve_cut_cell_conservation`. This is intentional: the zeros approach replaces uniform conservation entirely for E4_1.
  - **The existing `conserve=True` path (lines ~2232-2327) is unchanged** — it continues to handle schemes without zeros (E2_1, E2_2, E4_2).
  - **Psi boundary divergence (from 26.4c finding):** The conservative stencil has poles at psi=0 and psi=1 (alpha_0 diverges). The result or codegen output must document that coefficients are valid only for psi in the open interval (0, 1). Consider adding a `valid_psi_range=(0, 1)` annotation to the result or a comment in the generated C++ code.
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "SchemeWithZeros" --timeout=300`
  - **Done:** Implemented zeros path before the `if not conserve:` check. The path builds zero-constrained B_u, runs TEMO, solves cut-cell conservation for [alpha_0, alpha_1, w_1, w_2, w_3], renames {alpha_2 → alpha_0, w_4 → alpha_1}, and returns 2-alpha result. All existing tests pass (88 passed, 1 xfail).

- [x] **26.5b** E2_1 regression test:
  - **File:** `scripts/stencil_gen/tests/test_temo.py`
  - E2_1 has `zeros=()` — its conservation is handled by the nextra=1 mechanism
  - Verify: `derive_cut_cell_scheme(E2_1, psi)` results are unchanged:
    - Shape, alpha count, Taylor accuracy, psi limits all match pre-Phase-26 behavior
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "E2" --timeout=120`
  - **Done:** Added `TestE2_1DeriveCutCellSchemeRegression` class with 6 tests: shape, alpha_count, dims, floating_matches_manual, taylor_accuracy, psi_limits. All 96 E2 tests pass.

- [x] **26.5c** E4_1 `derive_cut_cell_scheme` integration tests:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Add test class `TestE4CutCellSchemeWithZeros`:
    - `test_alpha_count`: 2 free alpha symbols
    - `test_shape`: floating (5,7), dirichlet (4,7)
    - `test_free_symbols`: floating matrix free_symbols ⊆ {psi, alpha_0, alpha_1}
    - `test_weights_present`: weights is not None, length 5 (R=5), psi-dependent
    - `test_taylor_accuracy`: at psi=1/2, all rows satisfy 4 Taylor equations (uses non-zero alpha values since alpha_1=w_4=0 causes divergence)
    - `test_conservation_holds`: weighted column sums using result.weights are zero
    - `test_custom_alphas`: accepts `alpha_symbols=[Symbol("a"), Symbol("b")]`
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "SchemeWithZeros" --timeout=300`
  - **Done:** All 7 tests pass.

- [x] **26.5d** Update existing E4_1 tests broken by the zeros path:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - **Why this is needed:** Once 26.5a adds `scheme.zeros` dispatch, all existing calls to `derive_cut_cell_scheme(E4_1, ...)` hit the zeros path (2 alphas) instead of the old uniform conservation path (4 alphas). The following test classes/methods must be updated:
  - **`TestDeriveCutCellScheme` (line ~654):**
    - `test_e4_1_alpha_count` (line ~674): change expected from 4 to 2
    - `test_e4_1_custom_alphas` (line ~731): pass 2 symbols instead of 4; update assertion
    - `test_e4_1_matches_manual_pipeline` (line ~680): **Rewrite entirely.** The old test verified `conservation_subs` from the uniform conservation path. The zeros path has no `conservation_subs` (it's None). Instead, verify that the result matches: (1) build zero-constrained B_u, (2) TEMO, (3) solve cut-cell conservation, (4) substitute — yielding the same floating matrix as `derive_cut_cell_scheme(E4_1, psi)`.
    - `test_e4_1_taylor_accuracy` (line ~714): No change needed (Taylor accuracy is independent of alpha count).
    - E2 tests (`test_e2_1_reproduces_existing`, `test_e2_2_reproduces_existing`): No change needed (E2 has empty zeros).
  - **`TestCutCellConservationAfterUniform` (line ~1250):**
    - This class proves that uniform-conservation-only is infeasible for E4_1 at the cut-cell level (Groebner basis = {1}). It is still a valuable regression test documenting *why* the zeros approach was necessary.
    - **Fix:** Change the fixture to use a local `SchemeParams(p=2, q=3, s=0, nextra=0, nu=1)` without zeros (avoiding the new `E4_1` constant which has `zeros=(3,4)`). This preserves the infeasibility proof without being affected by the zeros path.
    - Only the fixture `conserved_cut_cell` (line ~1264) needs updating: replace `E4_1` with the local SchemeParams.
  - **`TestE4CodeGeneration` (line ~339):**
    - Fixture `e4_spec` (line ~342): `derive_cut_cell_scheme(E4_1, psi, conserve=True)` now returns 2 alphas. Update `param_arrays={"alpha": 2}`.
    - `test_alpha_array` (line ~392): expect `std::array<real, 2> alpha` instead of 4.
  - **`TestE4TestFileGeneration` (line ~471):**
    - `ALPHA_VALUES` (line ~474): change to `{"alpha": [0.1, -0.05]}` (2 values).
    - Fixture `e4_spec` (line ~476): update `param_arrays={"alpha": 2}`.
    - `test_generate_test_file_structure` (line ~543): assert changes to `alpha = {0.1, -0.05}` (2 values).
    - All `compute_test_values` calls automatically produce new expected values.
  - **Ordering constraint:** Must be done together with or immediately after 26.5a. Tests will fail between 26.5a and 26.5d.
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v --timeout=300`
  - **Done:** Updated all affected tests:
    - `test_e4_1_alpha_count`: 4 → 2
    - `test_e4_1_custom_alphas`: 4 → 2 symbols
    - `test_e4_1_matches_manual_pipeline`: Rewritten to verify zeros + cut-cell conservation pipeline
    - `TestCutCellConservationAfterUniform`: Fixture uses local SchemeParams without zeros
    - `TestE4CodeGeneration`: param_arrays=2, std::array<real, 2>
    - `TestE4TestFileGeneration`: ALPHA_VALUES 2 values, param_arrays=2, assertion updated
    - **Psi divergence fix:** Changed all psi=1.0 in TestE4TestFileGeneration to psi=0.9 (conservative stencil has poles at psi=0 and psi=1 per 26.4c)

### 26.6 — Re-generate E4_1 C++ code

- [x] **26.6a** Re-generate E4_1 C++ stencil and test file:
  - **Prerequisite:** 26.5a + 26.5d must be complete (Python pipeline returns 2-alpha conservative stencil).
  - **Workflow:**
    1. Run Python codegen tests which write to `scripts/stencil_gen/output/`:
       `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "test_write_output or test_write_test_output" --timeout=300`
    2. Copy generated files:
       `cp scripts/stencil_gen/output/E4_1.cpp src/stencils/E4_1.cpp`
       `cp scripts/stencil_gen/output/E4_1.t.cpp src/stencils/E4_1.t.cpp`
  - **Expected C++ changes in `src/stencils/E4_1.cpp`:**
    - `std::array<real, 4> alpha` → `std::array<real, 2> alpha`
    - P=2, R=5, T=7, X=0 are unchanged
    - Coefficient expressions become rational functions of (psi, alpha[0], alpha[1]); CSE temporaries will differ
    - `nbs_floating`, `nbs_dirichlet` method signatures unchanged
  - **Expected C++ changes in `src/stencils/E4_1.t.cpp`:**
    - Lua config: `alpha = {0.1, -0.05}` (was `{0.1, -0.05, 0.02, 0.01}`)
    - All `REQUIRE_THAT(c, Approx(T{...}))` expected values change (new symbolic expressions)
    - Test structure (Floating/Dirichlet sections, psi values) is unchanged
    - Test uses psi=0.9, 0.3, 0.7 (was 1.0, 0.3, 0.7) to avoid pole at psi=1
  - **Psi boundary divergence (from 26.4c finding):** The generated `E4_1.cpp` should include a comment that coefficients diverge at psi=0 and psi=1. The C++ runtime that calls this stencil must clamp psi away from boundaries (e.g., psi ∈ [0.01, 0.99]) or fall back to the non-conservative stencil for full cells (psi=1).
  - **Build and test:**
    - `cmake --build build --target t-E4_1` ✓
    - `ctest --test-dir build -R t-E4_1` ✓ (16 assertions, all pass)
  - **Done:** Regenerated E4_1.cpp (2-alpha conservative stencil) and E4_1.t.cpp (updated test values, psi=0.9 replaces psi=1.0). Build and test pass. Note: t-E2_1 has a pre-existing floating-point precision failure unrelated to this change.

### 26.6-followup — Address singularities in generated E4_1 stencil

- [x] **26.6-followup-a** Fix runtime psi=1.0 division-by-zero:
  - **Problem:** The conservative E4_1 stencil divides by `(psi - 1)` (line 90 in `nbs_floating`, line 214 in `nbs_dirichlet`). The runtime clamp in `src/mesh/object_geometry.cpp:95` is `std::clamp(psi, 1e-12, 1.0)`, which allows psi=1.0 — causing division by zero and NaN/Inf propagation.
  - **Fix chosen:** Option 1 — changed upper clamp bound to `1.0 - snap_tol` in `object_geometry.cpp:95`. Only E4_1 has a `(psi-1)` denominator among all stencils, so the change is safe for all.
  - **Test:** Added two Catch2 sections in `E4_1.t.cpp` testing Floating and Dirichlet with `psi = 1.0 - 1e-12`, verifying all coefficients are `std::isfinite`. 95 assertions pass (was 16).
  - **Additional validation:** mesh tests (4/4) and simulation tests (1/1) pass with the clamp change.

- [ ] **26.6-followup-b** Document alpha[1] ≠ 0 constraint:
  - **Problem:** The generated stencil divides by `alpha[1]` (= renamed w_4, a quadrature weight) in both `nbs_floating` (line 90: `1/(alpha[1]*t13)`) and `nbs_dirichlet` (line 211: `1.0 / (alpha[1])`). A user setting `alpha[1] = 0` gets division by zero. This constraint is mentioned only in a test parenthetical (26.5c) but not in the C++ code or plan's singularity documentation.
  - **Fix:** Add a comment in `E4_1.cpp` near the alpha declaration documenting that `alpha[1]` must be nonzero. Consider adding a runtime assert or guard. Update the codegen template if this constraint should appear automatically.

- [ ] **26.6-followup-c** Add missing singularity comments to generated E4_1.cpp:
  - **Problem:** Plan item 26.6a specifies: "The generated E4_1.cpp should include a comment that coefficients diverge at psi=0 and psi=1." This was not done — the generated file has no such comment.
  - **Fix:** Add a comment block to `E4_1.cpp` (either manually or via the codegen template) documenting:
    - Coefficients have poles at psi=0 and psi=1 (valid only for psi ∈ (0, 1))
    - alpha[1] must be nonzero
    - The denominator `288*alpha[1] + 648*psi + 12*psi³ + 90*psi² - 197` must also be nonzero

### 26.7 — Update memory and plans

- [ ] **26.7a** Update the stencil derivation memory with conservation resolution:
  - Conservation IS symbolically solvable when last-row zeros (alpha_3=alpha_4=0) are applied first
  - Key: cut-cell conservation (not uniform conservation) enforces the SBP property
  - The zeros fully determine the last uniform boundary row, eliminating the bilinear branching that made the system infeasible
  - 2 free optimization parameters remain: alpha_2 (boundary shape), w_4 (quadrature weight)
  - File: memory files
