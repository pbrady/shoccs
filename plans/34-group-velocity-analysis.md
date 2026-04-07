# Phase 34: Group Velocity Analysis Framework

**Goal:** Add group velocity analysis to the stencil derivation pipeline, providing a physically-motivated diagnostic that complements eigenvalue stability analysis and scales to 2D/3D varying-coefficient problems where eigenvalue computation is expensive or ill-conditioned.

**Depends on:** Phase 33 (complete), Phase 29-31 (complete -- PHS/tension/footprint investigations)

**Motivation:** Eigenvalue analysis of the full spatial operator D answers "is this scheme stable?" but becomes expensive (O(N^3)) and potentially ill-conditioned for large 2D/3D operators with varying coefficients. Group velocity analysis works directly with the stencil coefficients via modified wavenumber differentiation -- it is O(1) per stencil row, independent of grid size, and extends naturally to multiple dimensions. It also gives physical insight: which wavenumbers propagate at wrong speeds? Are parasitic modes created at boundaries? Does the cut-cell modification reverse energy propagation direction?

**Read first:**
- `scripts/stencil_gen/stencil_gen/phs.py` (existing modified wavenumber + stability analysis infrastructure)
- `scripts/stencil_gen/tests/test_phs.py` (`TestModifiedWavenumber` class, lines 3430-3750 -- the foundation we extend)
- `scripts/stencil_gen/stencil_gen/interior.py` (interior stencil derivation -- `derive_interior`, `full_gamma_array`)
- `scripts/stencil_gen/stencil_gen/boundary.py` (boundary row derivation)
- `scripts/stencil_gen/stencil_gen/temo.py` (cut-cell TEMO pipeline, psi-parameterized stencils)
- `papers/GroupVelocityInFiniteDifferenceSchemes.pdf` (Trefethen 1982 -- core theory)
- `papers/StabilityAndGroupVelocity.pdf` (Trefethen 1983 -- GKS connection)

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q
cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestInteriorGroupVelocity"
cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestBoundaryGroupVelocity"
```

**Background (from the papers):**

For the model equation u_t + u_x = 0 semi-discretized as du/dt = -D*u, where D
approximates d/dx with modified wavenumber kappa*(xi), the dispersion relation is
omega = Im(kappa*(xi)).  The group velocity is C(xi) = d(omega)/d(xi) = d(Im(kappa*))/d(xi).
For a perfect scheme, C(xi) = 1 for all xi. The group velocity error is
(2p+1) times the phase velocity error for a 2p-order scheme (corrected from
Trefethen 1982 Section 2's informal statement; the Taylor expansion gives 2p+1,
verified numerically in 34.2b). Parasitic modes near xi = pi/h can have reversed group velocity,
causing energy to propagate in the wrong direction. At boundaries, this
manifests as spontaneous radiation of energy into the domain (GKS instability,
Trefethen 1983).

## Sub-Plans

| Section | Plan File | Status | Summary |
|---------|-----------|--------|---------|
| 34.1 | (inline below) | **Complete** | Core group velocity module |
| 34.2 | (inline below) | Partially done (34.2a complete) | Interior scheme analysis |
| 34.3 | (inline below) | Partially done (34.3a-d complete) | Boundary closure analysis |
| 34.4 | `35-group-velocity-cut-cell.md` | Not started | Cut-cell psi-dependent analysis |
| 34.5 | `36-group-velocity-2d.md` | Not started | 2D/3D extension and varying coefficients |

---

## Items

### 34.1 — Core Group Velocity Module

- [x] **34.1a** Create `stencil_gen/group_velocity.py` with the core computation: ✅
  - Functions: `modified_wavenumber`, `group_velocity`, `group_velocity_exact`, `phase_velocity`, `group_velocity_error`, `GroupVelocityProfile` dataclass, `interior_group_velocity`.
  - **Sign convention fix:** The plan stated `C(xi) = -d(Im(kappa*))/d(xi)` but the correct formula is `C(xi) = +d(Im(kappa*))/d(xi)`. Derivation: for du/dt = -kappa* u, modes go as exp(-i*omega*t) with omega = Im(kappa*), so C = d(omega)/d(xi) = d(Im(kappa*))/d(xi). This gives C = cos(xi) for E2, matching Trefethen (1982). The analytical formula `Re(sum w_j * offset * exp(...))` was already correct in the plan.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`

- [x] **34.1b** Create `tests/test_group_velocity.py` with `TestCoreGroupVelocity`: ✅
  - 4 tests passing: `test_exact_scheme_unity_group_velocity` (E8 at low xi), `test_numerical_vs_analytical_gradient` (agrees to ~1e-4, limited by O(h^2) numerical diff), `test_phase_velocity_low_xi_limit`, `test_second_order_known_values` (E2 C = cos(xi) to machine precision).
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestCore"`

### 34.1-followup — Review fixes (from Ralph Wiggum review of 4993b58)

- [x] **34.1-fix-a** Fix docstring sign errors in `group_velocity.py`: ✅
  - Fixed module docstring: `omega = Im(kappa*(xi))`, `C = d(Im(kappa*))/d(xi)`.
  - Fixed `group_velocity()` docstring: positive sign.
  - Fixed `group_velocity_exact()` inline comment: positive sign.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`

- [x] **34.1-fix-b** Add a smoke test for `interior_group_velocity()`: ✅
  - Added `test_interior_group_velocity_e2` in `TestCoreGroupVelocity`.
  - Verifies: `GroupVelocityProfile` type, `order=2`, `group_velocity == cos(xi)`, `cutoff_xi ≈ pi/2`.
  - 5 tests passing.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`

### 34.1-followup-2 — Stale sign-convention comments in test file

- [x] **34.1-fix-c** Fix stale sign-convention comments in `tests/test_group_velocity.py`: ✅
  - Replaced the stream-of-consciousness comment block in `test_exact_scheme_unity_group_velocity` with a concise, correct explanation using the positive sign convention: `C(xi) = d(Im(kappa*))/d(xi)`.
  - 5 tests passing.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`

### 34.2 — Interior Scheme Group Velocity Analysis

- [x] **34.2a** ~~Add~~ `interior_group_velocity(p, nu, xi_array)` — already implemented in 34.1a. ✅
  - Calls `derive_interior(0, p, nu)` and `full_gamma_array()` to get weights.
  - Computes kappa*(xi) and C(xi) using the exact analytical formula.
  - Returns a dataclass `GroupVelocityProfile` with fields: `xi`, `kappa_star`, `phase_velocity`, `group_velocity`, `gv_error`, `order`, `cutoff_xi` (where C first goes to zero or negative).
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestInterior"`

- [x] **34.2b** Add `TestInteriorGroupVelocity` test class: ✅
  - Test `test_e2_group_velocity_is_cos_xi` -- E2 (p=1): C(xi) = cos(xi), exact.
  - Test `test_error_amplification_factor` -- verify group velocity error is **(2p+1)** times phase velocity error to leading order (corrected from plan's (2p-1); the Taylor expansion of Im(kappa*) = xi - a*xi^(2p+1)+... gives ratio (2p+1)). E2: factor 3, E4: factor 5, E6: factor 7, E8: factor 9.
  - Test `test_cutoff_wavenumber` -- for each scheme E2-E8, verify cutoff_xi (where C = 0) **increases** with order (corrected from plan's "decreases"; higher-order schemes resolve more wavenumbers before reversal). E2: pi/2, E4: ~1.80, E6: ~1.94, E8: ~2.03.
  - Test `test_group_velocity_sign_reversal` -- verify C(xi) ≤ 0 for all xi beyond cutoff_xi (parasitic regime where energy propagates backwards).
  - 9 tests passing (5 core + 4 interior).
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestInterior"`

- [x] **34.2c** Add `test_group_velocity_comparison_table` -- print a formatted table comparing E2/E4/E6/E8 interior schemes: ✅
  - Columns: scheme, order, cutoff xi/pi, |C_err| at xi=pi/4, |C_err| at xi=pi/2, min C (most negative).
  - Diagnostic/documentation test (always passes, prints useful data with -s).
  - 10 tests passing.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "test_group_velocity_comparison_table" -s`

### 34.3 — Boundary Closure Group Velocity Analysis

- [x] **34.3a** Add `boundary_group_velocity(p, q, nextra, nu, sigma, kernel, xi_array)` to `group_velocity.py`: ✅
  - For each boundary row i in [0, r), computes weights via `uniform_boundary_weights_rbf()` with given kernel/sigma.
  - Uses `compute_dimensions()` to get r and t, then `_build_profile()` (refactored shared helper) for each row.
  - Returns `dict[int, GroupVelocityProfile]` keyed by row index, with `order=q`.
  - Smoke tests: `test_boundary_gv_returns_all_rows`, `test_boundary_gv_bounded`, `test_boundary_row0_low_xi_near_unity` (3 tests in `TestBoundaryGroupVelocity`).
  - 13 tests passing (10 existing + 3 new).
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestBoundary"`

- [x] **34.3b** Add `boundary_group_velocity_classical(boundary_rows, alpha_values, order, xi_array)` to `group_velocity.py`: ✅
  - Takes the symbolic `BoundaryRow` list from `derive_boundary()` (or conservation-updated rows) and alpha values dict.
  - Evaluates stencil coefficients numerically via `xreplace(alpha_values)`, then computes C(xi).
  - Works with the classical (non-RBF) boundary stencils from Brady & Livescu.
  - Tests: `test_classical_returns_all_rows`, `test_classical_coefficients_finite`, `test_classical_row0_low_xi`, `test_classical_bounded` (4 tests in `TestBoundaryClassical`).
  - 17 tests passing (13 existing + 4 new).
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestBoundaryClassical"`

- [x] **34.3c** Extend `TestBoundaryGroupVelocity` with deeper diagnostic tests: ✅
  - Test `test_boundary_vs_interior_gv_error` -- verifies row 0 (one-sided) has larger GV error than interior at xi < pi/4. Also checks no boundary row has C < 0 at well-resolved wavenumbers. Uses pi/4 threshold (not pi/2) because one-sided boundary stencils have earlier dispersion cutoff than symmetric interior stencils.
  - Test `test_parasitic_direction_at_boundary` -- checks boundary rows don't create strongly positive C (> 5) in the parasitic regime (xi > cutoff) where interior has C < 0. Tests E2 and E4 with tension kernel.
  - Test `test_classical_e4_boundary_gv` -- classical E4 boundary stencils with known-good alpha values have C > 0 at all resolved wavenumbers (xi < pi/2). Added to `TestBoundaryClassical`.
  - 20 tests passing (17 existing + 3 new).
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestBoundary"`

### 34.3-followup — Review fixes (from Ralph Wiggum review of 256a730)

- [x] **34.3-fix-a** Fix `_build_profile` cutoff detection for non-monotonic boundary C(xi): ✅
  - Changed cutoff logic to scan from the high end: finds the last xi where C > 0, then sets cutoff to the next grid point. This gives the first xi beyond which C stays permanently non-positive.
  - Added `test_cutoff_handles_oscillating_c` in `TestBoundaryGroupVelocity`: synthetic one-sided stencil with oscillating C(xi), verifies cutoff is at the persistent crossing (not the transient dip), and C <= 0 beyond cutoff.
  - 21 tests passing (20 existing + 1 new).
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py` (lines 184-192)
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`

### 34.3-followup-2 — Review fixes (from Ralph Wiggum review of 3844807)

- [x] **34.3-fix-b** Fix stale `cutoff_xi` comment in `GroupVelocityProfile` dataclass: ✅
  - Updated comment on line 165 of `group_velocity.py` from "xi where C first goes to zero or negative" to "first xi beyond which C stays permanently non-positive", matching the semantics from 34.3-fix-a.
  - 21 tests passing.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py` (line 165)

- [x] **34.3d** Add GKS-inspired diagnostic `gks_group_velocity_check(D, xi_array)`: ✅
  - Added `GKSModeInfo` dataclass with fields: `eigenvalue`, `boundary_wavenumber`, `group_velocity`, `is_outgoing`.
  - Added `gks_group_velocity_check(D, xi_array, neutral_tol, localization_tol)`:
    - Computes eigenvalues/eigenvectors of -D_bc (D with inflow row/column removed).
    - Filters to nearly-neutral modes (|Re(lambda)| < neutral_tol * max|Re|).
    - Identifies boundary-localized modes (eigenvector energy fraction > localization_tol in boundary region).
    - Estimates dominant wavenumber via zero-padded FFT of boundary portion of eigenvector.
    - Computes interior group velocity at that wavenumber; `is_outgoing = True` if energy radiates from boundary into domain (C > 0 for left boundary, C < 0 for right boundary).
    - Skips conjugate duplicate eigenvalues.
  - Smoke-tested: E2 at sigma=10 finds 1 non-outgoing boundary mode; E4 and PHS find 0 modes (all stable, no energy radiation into domain).
  - 21 existing tests still pass.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestGKS"`

- [ ] **34.3e** Add `TestGKSDiagnostic` test class:
  - Test `test_stable_scheme_no_outgoing_modes` -- for E2 at optimal sigma (known stable), verify no outgoing boundary modes.
  - Test `test_known_unstable_extrapolation` -- construct a simple case with extrapolation BC (known GKS-unstable for leapfrog per Trefethen 1983). Verify the diagnostic detects the outgoing mode. (This uses a time-discrete analysis -- we may need a helper for leapfrog dispersion relation. If the semi-discrete framework doesn't capture this, document the limitation and mark as a future extension.)
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestGKS"`

---

## Completion Criteria

- `group_velocity.py` module exists with core functions and is importable.
- Interior group velocity matches analytical expectations (cos(xi) for E2, (2p+1)x error amplification).
- Boundary group velocity is computed for both RBF/tension and classical stencils.
- GKS-inspired diagnostic connects per-stencil analysis to full-operator modes.
- All new tests pass: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q`
- No existing tests broken: `cd scripts/stencil_gen && uv run pytest tests/ -x -q -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"`
