# Phase 35: Group Velocity Analysis for Cut-Cell Stencils

**Goal:** Extend group velocity analysis to psi-parameterized cut-cell stencils from the TEMO pipeline, answering: does the cut-cell modification introduce modes with reversed group velocity, and how does the group velocity spectrum vary with psi?

**Depends on:** Phase 34 (complete -- core group velocity module and boundary analysis)

**Read first:**
- `scripts/stencil_gen/stencil_gen/group_velocity.py` (core module from Phase 34)
- `scripts/stencil_gen/stencil_gen/temo.py` (`derive_cut_cell_mathematica`, `CutCellResult`, `SchemeParams`)
- `scripts/stencil_gen/tests/test_temo.py` (E2_1/E2_2 cut-cell pipeline tests -- understand the data structures)
- `scripts/stencil_gen/tests/test_e4_cut_cell.py` (E4_1 cut-cell tests)
- `papers/BradyLivscu2021.pdf` (TEMO methodology, psi parameterization)

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestCutCell"
cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestPsiSweep"
```

---

## Items

### 35.1 — Cut-Cell Group Velocity Computation

- [x] **35.1a** Add `cut_cell_group_velocity(cut_cell_result, psi_sym, psi_val, alpha_values, xi_array)` to `group_velocity.py`:
  - Accepts a precomputed `CutCellResult` (from `derive_cut_cell_scheme` or `derive_cut_cell_mathematica`) to avoid recomputing expensive symbolic derivation at each psi value.
  - For each boundary row (0 to R-1), evaluates the symbolic psi-dependent coefficients numerically.
  - Uses `_build_profile_nonuniform` with non-uniform offsets `[-(psi_val+i), -i, 1-i, ..., (T-2)-i]`.
  - Returns `dict[int, GroupVelocityProfile]` keyed by row index.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Done: all tests pass (29/29).

- [x] **35.1b** Add helpers `modified_wavenumber_nonuniform` and `group_velocity_exact_nonuniform`:
  - `modified_wavenumber_nonuniform(weights, offsets, xi_array)`: generalization for real-valued offsets.
  - `group_velocity_exact_nonuniform(weights, offsets, xi_array)`: `C(xi) = Re(sum_j w_j * offset_j * exp(i * offset_j * xi))`.
  - `_build_profile_nonuniform(weights, offsets, xi_array, order)`: constructs `GroupVelocityProfile` from nonuniform offsets.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Tests: `TestNonuniformModWavenumber` (3 tests, all pass).

- [x] **35.1c** Add `TestCutCellGroupVelocity` test class:
  - Test `test_psi_1_matches_uniform` -- at psi=1, wall coefficient is zero for rows 0..r-1, so cut-cell GV matches TEMO uniform GV to <1e-10. Verified.
  - Test `test_psi_0_degenerate_bounded` -- at psi=0, all rows have finite, bounded GV (|C| < 100). Verified.
  - Test `test_e2_1_cut_cell_gv_smooth_in_psi` -- 11-point psi sweep, |dC/dpsi| < 200 for all rows. Verified.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Tests: `TestCutCellGroupVelocity` (3 tests, all pass).

### 35.2 — Psi Sweep Analysis

- [x] **35.2a** Add `psi_sweep_group_velocity(scheme_params, psi_values, alpha_values, xi_array)` to `group_velocity.py`:
  - Sweeps over a range of psi values, computing group velocity profiles at each.
  - Returns `PsiSweepResult` dataclass with fields: `psi_values`, `profiles` (dict of dicts: `{psi: {row: GroupVelocityProfile}}`), `worst_row` (row with largest GV error), `worst_psi` (psi with largest GV error), `min_C` (most negative group velocity across all psi/rows), `has_sign_reversal` (bool: any C > 0 at wavenumbers where interior has C < 0).
  - Uses `derive_cut_cell_mathematica` for schemes with zeros (singularity-free at psi=0), `derive_cut_cell_scheme` otherwise.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Done: all tests pass (32/32).

- [x] **35.2b** Add `TestPsiSweepGroupVelocity` test class:
  - Test `test_e2_1_psi_sweep` -- sweep psi in [0, 1] at 11 points for E2_1. All profiles finite/bounded. No parasitic sign reversal at non-degenerate psi (>= 0.1) at resolved wavenumbers.
  - Test `test_e2_1_no_cfl_penalty` -- max|omega| ratio stays < 10x across all psi values (no CFL stiffness penalty).
  - Test `test_e4_1_psi_sweep` -- E4_1 via singularity-free Mathematica pipeline. All profiles finite, |C| < 500.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Tests: `TestPsiSweepGroupVelocity` (3 tests, all pass).

### 35.3 — Comparison with Eigenvalue Analysis

- [ ] **35.3a** Add `TestCutCellGVvsEigenvalue` test class:
  - Test `test_gv_predicts_eigenvalue_stability` -- for E2_1 at several psi values, compute both the group velocity diagnostic and the full eigenvalue stability check. Verify they agree: if the GV diagnostic says "no parasitic outgoing modes" then eigenvalues should have Re(lambda) <= 0.
  - Test `test_gv_cost_vs_eigenvalue_cost` -- time both analyses at N=50, 100, 200. Print a table showing that GV analysis is O(1) per stencil while eigenvalue is O(N^3). This motivates the 2D/3D extension.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestCutCellGVvsEigenvalue"`

---

## Completion Criteria

- Cut-cell group velocity computed correctly with non-uniform offsets.
- psi=1 limit matches uniform boundary analysis exactly.
- Smooth variation with psi (no discontinuities at the TEMO design points psi=0 and psi=1).
- Psi sweep identifies worst-case group velocity behavior across the full psi range.
- Cost comparison demonstrates the scaling advantage over eigenvalue analysis.
- All new tests pass: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "CutCell or PsiSweep"`
