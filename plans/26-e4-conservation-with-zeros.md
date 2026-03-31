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

- [ ] **26.1a** Add `zeros` field to `SchemeParams` and `zeros` parameter to `derive_uniform_boundary_for_temo`:
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
  - **No other logic changes needed:** The nextra>0 and conserve branches are unaffected since E4_1 has nextra=0 and zeros+conserve is forbidden.
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "E2" --timeout=60` (E2 unchanged)

- [ ] **26.1b** Test E4_1 uniform boundary with zeros:
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
    - `test_taylor_accuracy`: All 4 rows satisfy `max(q+1, nu+1)=4` Taylor equations (use `_build_uniform_vandermonde`)
    - `test_zeros_conserve_mutual_exclusion`: `derive_uniform_boundary_for_temo(E4_1, zeros={3,4}, conserve=True)` raises ValueError
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "WithZeros" --timeout=60`

### 26.2 — Build zero-constrained cut-cell stencil

- [ ] **26.2a** Build and test E4_1 cut-cell with zero-constrained B_u:
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

- [ ] **26.3a** Add `solve_cut_cell_conservation` function:
  - **File:** `scripts/stencil_gen/stencil_gen/temo.py` (insert after `build_cut_cell_conservation_system`, ~line 1487)
  - **Signature:**
    ```python
    def solve_cut_cell_conservation(
        B_l: Matrix,
        R: int,
        T: int,
        p: int,
        nu: int,
        interior_coeffs: list,
        psi: Symbol,
        alpha_symbols: list[Symbol],
        solve_for: list[Symbol],
    ) -> dict[Symbol, Expr]:
        """Solve the cut-cell conservation system for the given unknowns.

        Assumes zeros have been applied to the uniform boundary before TEMO,
        making the system solvable.  Treats any symbol in the conservation
        system that is NOT in ``solve_for`` as a free parameter.

        Parameters
        ----------
        B_l : Matrix
            R x T cut-cell stencil from TEMO (with zero-constrained alphas).
        R, T, p, nu, interior_coeffs, psi : same as build_cut_cell_conservation_system.
        alpha_symbols : list[Symbol]
            The remaining alpha symbols (post-zeros, e.g. [alpha_0, alpha_1, alpha_2]).
        solve_for : list[Symbol]
            Symbols to solve for (e.g. [alpha_0, alpha_1, w_1, w_2, w_3]).

        Returns
        -------
        dict[Symbol, Expr]
            Maps each solved symbol to its expression in (psi, free_params).
            Free params are everything in {alpha_symbols + w_syms} \\ solve_for.
        """
    ```
  - **Implementation:**
    1. Call `build_cut_cell_conservation_system(B_l, R, T, p, nu, interior_coeffs, psi)` → `(eqs, w_syms)` where `w_syms = [w_1, w_2, w_3, w_4]`
    2. `solution = sympy.solve(eqs, solve_for, dict=True)` — expected: single-branch solution
    3. Assert `len(solution) == 1`, extract `sol = solution[0]`
    4. Verify: all 5 equations evaluate to 0 after substitution (use `cancel(eq.subs(sol)) == 0`)
    5. Return `sol`
  - **Expected solve_for for E4_1:** `[alpha_0, alpha_1, w_1, w_2, w_3]` (5 unknowns for 5 equations)
  - **Expected free params:** `[alpha_2, w_4]` (2 free for optimization)
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "solve_cut_cell" --timeout=300`

- [ ] **26.3b** Test conservation solution and apply to stencil:
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

- [ ] **26.4a** Update the xfail conservation test:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py` (line 819)
  - The existing `test_e4_1_conservation_fails` (line 823) uses the OLD pipeline (5 alphas, no zeros, naive flat weights). The infeasibility result is still valid for that configuration.
  - **Do NOT remove the xfail** — instead, rename to `test_e4_1_conservation_fails_without_zeros` and keep the xfail decorator. It documents that conservation fails WITHOUT zeros.
  - **Add a NEW test** `test_e4_1_conservation_with_zeros` that:
    1. Builds zero-constrained B_u (`zeros={3, 4}`)
    2. Runs TEMO → cut-cell stencil (3 alphas)
    3. Solves cut-cell conservation
    4. Substitutes solution into stencil
    5. Verifies `sum_i w_i(psi) * B[i,j](psi) = 0` symbolically for all interior columns, using the **solved psi-dependent weights** (not flat weights)
    6. Also verifies at specific numeric (psi, alpha_2, w_4) values: psi=0.3, 0.5, 0.7 with alpha_2=0.1, w_4=1.0
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "conservation_with_zeros" --timeout=300`

- [ ] **26.4b** Verify Taylor accuracy is preserved:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test `test_e4_1_conservative_taylor_accuracy`:
    - Use the conservative stencil (after substitution from 26.3b)
    - Evaluate at psi=0.3, 0.5, 0.7 with alpha_2=0.1, w_4=1.0
    - Verify each row satisfies q+1=4 Taylor moment equations
    - Use `build_cut_cell_deltas(i, T, psi_val)` for delta computation
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "conservative_taylor" --timeout=120`

- [ ] **26.4c** Verify psi limits are preserved:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Test `test_e4_1_conservative_psi_limits`:
    - psi=1: row 0 wall column is zero, rows 1-4 match uniform B_u (with zeros)
    - psi=0: degenerate design principles satisfied (col 0/col 1 splitting per `build_degenerate_stencil` logic)
    - Evaluate with specific alpha_2, w_4 values
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "conservative_psi_limits" --timeout=120`

### 26.5 — Integrate into `derive_cut_cell_scheme`

- [ ] **26.5a** Update `derive_cut_cell_scheme` to use zeros and cut-cell conservation:
  - **File:** `scripts/stencil_gen/stencil_gen/temo.py` (line 2172, `derive_cut_cell_scheme`)
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
        sol = solve_cut_cell_conservation(
            floating, dims.R, dims.T, scheme.p, scheme.nu,
            uniform.interior, psi, uniform.alpha_symbols, solve_for,
        )
        # Step 4: Apply solution to stencil
        floating = floating.xreplace(sol)
        # Step 5: Compute weights (w_0=psi, w_1..w_3 from sol, w_4 free)
        weights = [psi] + [sol[w] for w in w_syms[:3]] + [w_syms[3]]
        # Step 6: Rename remaining free params for codegen
        # alpha_2 -> alpha_0, w_4 -> alpha_1
        free_alpha = uniform.alpha_symbols[2]  # alpha_2
        free_w = w_syms[3]  # w_4
        if alpha_symbols is None:
            final = [Symbol("alpha_0"), Symbol("alpha_1")]
        else:
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
  - **Ordering constraint:** This block should go BEFORE the existing `if not conserve:` check (line 2208) since `scheme.zeros` takes priority.
  - **The existing `conserve=True` path (lines 2232-2327) is unchanged** — it continues to handle non-zero schemes like E4_1 without zeros (backward-compatible) or E4_2.
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v --timeout=300`

- [ ] **26.5b** E2_1 regression test:
  - **File:** `scripts/stencil_gen/tests/test_temo.py`
  - E2_1 has `zeros=()` — its conservation is handled by the nextra=1 mechanism
  - Verify: `derive_cut_cell_scheme(E2_1, psi)` results are unchanged:
    - Shape, alpha count, Taylor accuracy, psi limits all match pre-Phase-26 behavior
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_temo.py -v -k "E2" --timeout=120`

- [ ] **26.5c** E4_1 `derive_cut_cell_scheme` integration tests:
  - **File:** `scripts/stencil_gen/tests/test_e4_cut_cell.py`
  - Add test class `TestE4CutCellSchemeWithZeros`:
    - `test_alpha_count`: 2 free alpha symbols
    - `test_shape`: floating (5,7), dirichlet (4,7)
    - `test_free_symbols`: floating matrix free_symbols ⊆ {psi, alpha_0, alpha_1}
    - `test_weights_present`: weights is not None, length 5 (R=5), psi-dependent
    - `test_taylor_accuracy`: at psi=1/2, all rows satisfy 4 Taylor equations
    - `test_conservation_holds`: weighted column sums using result.weights are zero
    - `test_custom_alphas`: accepts `alpha_symbols=[Symbol("a"), Symbol("b")]`
  - **Test:** `cd scripts/stencil_gen && uv run pytest tests/test_e4_cut_cell.py -v -k "SchemeWithZeros" --timeout=300`

### 26.6 — Re-generate E4_1 C++ code

- [ ] **26.6a** Generate conservative E4_1 C++ stencil:
  - **Files:** `scripts/stencil_gen/output/E4_1.cpp` → `src/stencils/E4_1.cpp`
  - The `StencilGenSpec` will have:
    - `name="E4_1"`, `P=2`, `R=5`, `T=7`, `X=0`, `derivative_order=1`, `is_uniform=False`
    - `param_arrays={"alpha": 2}` — both alpha_2 and w_4 mapped to `alpha[0]` and `alpha[1]`
  - **C++ struct changes:**
    - `std::array<real, 4> alpha` → `std::array<real, 2> alpha` (was 4 after uniform conservation, now 2 after cut-cell conservation)
    - Constructor: `E4_1(std::span<const real> a)` still works with `copy_zero_padded`
  - **Coefficient expressions:** Entries are rational functions of (psi, alpha[0], alpha[1]). Expect larger CSE output since conservation introduces rational psi-dependency beyond what TEMO alone produces.
  - **Test file:** Update `src/stencils/E4_1.t.cpp` with new test values computed from `compute_test_values` with 2-element alpha array
  - **Build:** `cmake --build build --target t-E4_1`
  - **Test:** `ctest --test-dir build -R t-E4_1`

### 26.7 — Update memory and plans

- [ ] **26.7a** Update the stencil derivation memory with conservation resolution:
  - Conservation IS symbolically solvable when last-row zeros (alpha_3=alpha_4=0) are applied first
  - Key: cut-cell conservation (not uniform conservation) enforces the SBP property
  - The zeros fully determine the last uniform boundary row, eliminating the bilinear branching that made the system infeasible
  - 2 free optimization parameters remain: alpha_2 (boundary shape), w_4 (quadrature weight)
  - File: memory files
