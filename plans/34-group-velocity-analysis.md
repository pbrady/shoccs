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

For the model equation u_t + u_x = 0 with semi-discrete spatial operator, the
modified wavenumber kappa*(xi) relates to the dispersion relation via
omega = -kappa*(xi). The group velocity is C(xi) = d(omega)/d(xi) = -d(kappa*)/d(xi).
For a perfect scheme, C(xi) = 1 for all xi. The group velocity error is
(2p-1) times the phase velocity error for a 2p-order scheme (Trefethen 1982,
Section 2). Parasitic modes near xi = pi/h can have reversed group velocity,
causing energy to propagate in the wrong direction. At boundaries, this
manifests as spontaneous radiation of energy into the domain (GKS instability,
Trefethen 1983).

## Sub-Plans

| Section | Plan File | Status | Summary |
|---------|-----------|--------|---------|
| 34.1 | (inline below) | **Review fixes pending** | Core group velocity module |
| 34.2 | (inline below) | Partially done (34.2a complete) | Interior scheme analysis |
| 34.3 | (inline below) | Not started | Boundary closure analysis |
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

- [ ] **34.1-fix-a** Fix docstring sign errors in `group_velocity.py`:
  - The module docstring (line 6-7) says `omega = -kappa*(xi)` and `C = -d(kappa*)/d(xi)`. The correct relation (as noted in 34.1a sign convention fix) is `omega = Im(kappa*)` and `C = +d(Im(kappa*))/d(xi)`.
  - The `group_velocity()` function docstring (line 50) says `C(xi) = -d(Im(kappa*))/d(xi)` but the implementation correctly computes `+d(Im(kappa*))/d(xi)`. Update the docstring to match the code.
  - These wrong-sign docstrings risk propagating the error into boundary/cut-cell work (34.3, 34.4).
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`

- [ ] **34.1-fix-b** Add a smoke test for `interior_group_velocity()`:
  - The function was implemented in 34.1a but is never imported or tested in `test_group_velocity.py`. Add a test in `TestCoreGroupVelocity` that calls `interior_group_velocity(p=1, nu=1, xi_array)` and verifies: (1) returned `GroupVelocityProfile` has correct `order=2`, (2) `group_velocity` field equals `cos(xi)` for E2, (3) `cutoff_xi` is approximately `pi/2`.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`

### 34.2 — Interior Scheme Group Velocity Analysis

- [x] **34.2a** ~~Add~~ `interior_group_velocity(p, nu, xi_array)` — already implemented in 34.1a. ✅
  - Calls `derive_interior(0, p, nu)` and `full_gamma_array()` to get weights.
  - Computes kappa*(xi) and C(xi) using the exact analytical formula.
  - Returns a dataclass `GroupVelocityProfile` with fields: `xi`, `kappa_star`, `phase_velocity`, `group_velocity`, `gv_error`, `order`, `cutoff_xi` (where C first goes to zero or negative).
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestInterior"`

- [ ] **34.2b** Add `TestInteriorGroupVelocity` test class:
  - Test `test_e2_group_velocity_is_cos_xi` -- E2 (p=1): C(xi) = cos(xi*h), exact.
  - Test `test_error_amplification_factor` -- verify group velocity error is (2p-1) times phase velocity error to leading order. For E2 (p=1): factor 1 (trivially, since C = cos(xi) and c = sin(xi)/xi, so C_err/c_err -> 3 at leading order). For E4 (p=2): factor 5. For E6 (p=3): factor 7. Check at small xi (xi < 0.3) where Taylor expansion dominates.
  - Test `test_cutoff_wavenumber` -- for each scheme E2-E8, verify cutoff_xi (where C = 0) decreases as order increases (higher-order schemes resolve more wavenumbers but have a sharper cutoff).
  - Test `test_group_velocity_sign_reversal` -- verify C(xi) < 0 for xi > cutoff_xi (parasitic regime where energy propagates backwards).
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestInterior"`

- [ ] **34.2c** Add `test_group_velocity_comparison_table` -- print a formatted table comparing E2/E4/E6/E8 interior schemes:
  - Columns: scheme, order, cutoff xi/pi, max |C_err| at xi=pi/4, max |C_err| at xi=pi/2, min C (most negative).
  - This is a diagnostic/documentation test (always passes, prints useful data).
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "test_group_velocity_comparison_table" -s`

### 34.3 — Boundary Closure Group Velocity Analysis

- [ ] **34.3a** Add `boundary_group_velocity(p, q, nextra, nu, sigma, kernel, xi_array)` to `group_velocity.py`:
  - For each boundary row i in [0, r), computes weights via `uniform_boundary_weights_rbf()` (tension kernel) or `uniform_boundary_weights()` (classical).
  - Computes kappa*(xi) and C(xi) for each row.
  - Returns `dict[int, GroupVelocityProfile]` keyed by row index.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestBoundary"`

- [ ] **34.3b** Add `boundary_group_velocity_classical(boundary_rows, xi_array)` to `group_velocity.py`:
  - Takes the symbolic `BoundaryRow` list from `derive_boundary()` and alpha values.
  - Evaluates stencil coefficients numerically at given alpha, then computes C(xi).
  - This works with the classical (non-RBF) boundary stencils from Brady & Livescu.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestBoundaryClassical"`

- [ ] **34.3c** Add `TestBoundaryGroupVelocity` test class:
  - Test `test_boundary_gv_bounded` -- for E2/E4 at optimal tension sigma, verify |C(xi)| is bounded (no blow-up) for all boundary rows and all xi.
  - Test `test_boundary_vs_interior_gv_error` -- compare group velocity error of boundary rows vs interior. Boundary rows should have larger error (they're lower order) but should not have reversed sign at well-resolved wavenumbers (xi < pi/2).
  - Test `test_parasitic_direction_at_boundary` -- check whether boundary stencils create parasitic modes with C > 0 (outgoing from boundary = GKS-unstable direction). For an inflow boundary (u_t + u_x = 0, left boundary), physical waves have C < 0 (leftward). If any boundary row creates C > 0 for wavenumbers where the interior has C < 0, that's a potential instability source.
  - Test `test_classical_e4_boundary_gv` -- compute group velocity for the classical E4 boundary stencils (from `derive_boundary(p=2, nu=1, s=0)`) with known-good alpha values. Verify no sign reversal at resolved wavenumbers.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestBoundary"`

- [ ] **34.3d** Add GKS-inspired diagnostic `gks_group_velocity_check(D, xi_array)`:
  - Given the full N x N differentiation matrix D, compute eigenvalues and eigenvectors.
  - For each eigenmode with Re(lambda) near zero, estimate the local wavenumber content near the boundary (FFT of eigenvector's first few components).
  - Check: does this mode's dominant wavenumber have positive group velocity (outgoing)?
  - Returns a list of `GKSModeInfo(eigenvalue, boundary_wavenumber, group_velocity, is_outgoing)`.
  - This bridges per-stencil group velocity analysis with the full-operator eigenvalue analysis.
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
- Interior group velocity matches analytical expectations (cos(xi) for E2, (2p-1)x error amplification).
- Boundary group velocity is computed for both RBF/tension and classical stencils.
- GKS-inspired diagnostic connects per-stencil analysis to full-operator modes.
- All new tests pass: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q`
- No existing tests broken: `cd scripts/stencil_gen && uv run pytest tests/ -x -q -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"`
