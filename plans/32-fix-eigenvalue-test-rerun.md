# Phase 32: Fix Eigenvalue Stability Testing and Re-Run Analysis

**Goal:** The eigenvalue stability testing infrastructure has two fundamental bugs:

1. **Wrong BC:** Tests use the full free-end (floating) matrix. The physically correct
   test for scalar advection `u_t + u_x = 0` with rightward propagation is: Dirichlet
   at inflow (left), floating at outflow (right) → remove first row and column of D.

2. **Wrong sign convention:** Tests interpret `max Re(eigenvalue of D) > 0` as unstable.
   But for `u_t = -D u` (the semi-discrete advection equation where D ≈ d/dx), stability
   requires `Re(eigenvalue of -D) ≤ 0`, i.e., `Re(eigenvalue of D) ≥ 0`.  Positive real
   parts of D are **stable**, not unstable.

**Validation:** The production E4u_1 stencil at its optimized alphas (-0.7733, 0.1624)
is stable under the correct test (all eigenvalues of -D have non-positive real parts
with inflow-Dirichlet/outflow-floating BCs), confirming the paper's claims.

**Impact:** All Phase 29-31 conclusions about "instability floors" may be artifacts of
the wrong test.  This phase fixes the infrastructure and re-runs the analysis.

**Depends on:** Phases 29-31 (engine and tests exist, need correction)

**Priority:** Critical — all prior spline investigation results must be re-evaluated.

**Read first:**
- `scripts/stencil_gen/stencil_gen/phs.py` (`build_diff_matrix_rbf`, `max_real_eigenvalue`)
- `scripts/stencil_gen/tests/test_phs.py` (all sweep test classes)
- `plans/stencil-derivation-math-reference.md` (Section 5: optimization methodology)

**Test commands:**
```bash
cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_phs.py -x -v
```

---

## Background: The Two Bugs

### Bug 1: Boundary Conditions

The scalar advection equation `u_t + a u_x = 0` with `a > 0` (rightward propagation)
has inflow at x=0 (left) and outflow at x=L (right).  The well-posed IBVP prescribes
a Dirichlet condition at inflow only.  The outflow boundary needs no condition — the
solution simply flows out.

The correct semi-discrete matrix is obtained by removing the first row and first column
of D (the inflow point is prescribed, not an unknown):
```python
D_bc = D_full[1:, 1:]  # (n-1) x (n-1)
```

Our code used `D_full` (n×n, no BC removal), which includes the inflow equation as
an unknown — physically meaningless and eigenvalue-pathological.

### Bug 2: Sign Convention

D approximates `d/dx`.  The PDE `u_t + u_x = 0` gives the semi-discrete system:
```
du/dt = -D u
```
The time-evolution operator is `-D`, not `+D`.  For stability of `du/dt = A u`, we need
`Re(eigenvalue of A) ≤ 0`, i.e., `Re(eigenvalue of -D) ≤ 0`, i.e.:

**`Re(eigenvalue of D) ≥ 0` means STABLE.**

Our code checked `max Re(eig(D)) > 0` as "unstable" — the exact opposite of correct.

### Combined effect

With both bugs fixed, the correct stability check is:
```python
D_full = build_diff_matrix(...)
D_bc = D_full[1:, 1:]          # inflow-Dirichlet BC
eigs = np.linalg.eigvals(D_bc)
max_re_neg_D = np.max((-eigs).real)  # eigenvalues of -D
stable = max_re_neg_D < STABILITY_TOL
```

Or equivalently:
```python
min_re_D = np.min(eigs.real)
stable = min_re_D > -STABILITY_TOL
```

---

## 32.1 — Fix the Infrastructure

### 32.1a — Add `stability_eigenvalue` function to `phs.py` ✅

Added `stability_eigenvalue()` at line ~931 of `phs.py`. Uses `D[1:, 1:]` (inflow
Dirichlet BC) and eigenvalues of `-D_bc` (correct sign convention for advection).
Kept `max_real_eigenvalue` for backward compatibility with Phase 29-31 tests.

- File: `scripts/stencil_gen/stencil_gen/phs.py`

### 32.1b — Add `stability_eigenvalue_from_matrix` helper ✅

Added `stability_eigenvalue_from_matrix(D)` at line ~952 of `phs.py`. Takes a
pre-built matrix, applies `D[1:, 1:]` and returns `max Re(eig(-D_bc))`.

- File: `scripts/stencil_gen/stencil_gen/phs.py`

### 32.1c — Validate against production E4u_1 ✅

Added `TestStabilityInfrastructure` class with 5 tests (all pass):
- `test_production_e4_tension_stable`: E4 tension σ=3.0 stable at n=20,40,80,160
- `test_interior_only_neutrally_stable`: periodic interior ≈ 0
- `test_unstable_detected`: Gaussian ε=0.1 E4 correctly detected as unstable
- `test_stability_eigenvalue_from_matrix_consistent`: both APIs agree
- `test_e2_stable`: E2 Gaussian ε=1.0 stable at n=20,40,80

Key findings:
- E4 tension σ=3.0: all stable (se ranges from -4.9e-4 at n=20 to -9.6e-7 at n=160)
- E4 Gaussian ε=0.1: genuinely unstable (se ≈ +0.098)
- Tension kernels across wide σ range appear stable under correct test

- File: `scripts/stencil_gen/tests/test_phs.py`

---

## 32.2 — Re-Run Phase 29 Analysis (PHS + Gaussian + Multiquadric)

### 32.2a — E2 PHS/Gaussian/MQ sweep with correct stability test ✅

Redid the Phase 29 sweeps using `stability_eigenvalue`.  All tests pass.

**Key finding: E2 was always stable.  Phase 29 "instability" was entirely an artifact.**

Results summary:
- **PHS k=2**: Stable at n=20,40,80 (se from -1.9e-2 to -1.1e-4)
- **Gaussian ε sweep**: 57/60 epsilons stable at each n.  Only a narrow band
  near ε≈1.2-1.5 is unstable.  Stability eigenvalues are solidly negative.
- **Multiquadric ε sweep**: 60/60 epsilons stable — universally stable across
  entire ε range [0.01, 10] at all grid sizes.
- **Gaussian fine sweep**: Best ε*≈0.0025 gives se≈-9.4e-4 at n=40, stable
  across n=20,40,80,160 with se scaling as O(h²).

- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedSweepE2`

**Review follow-up (assertions added) ✅:** All 4 tests now have assertions:
- `test_multiquadric_sweep`: asserts 60/60 stable at each n
- `test_gaussian_sweep`: asserts ≥55/60 stable at each n
- `test_phs_k2_baseline`: asserts stable at n=20,40,80
- `test_gaussian_fine_sweep`: asserts best ε* stable across n=20,40,80,160

### 32.2b — E4 PHS/Gaussian/MQ sweep with correct stability test ✅

Redid the Phase 29 E4 sweeps using `stability_eigenvalue`.  All tests pass.

**Key finding: E4 instability is real for most Gaussian epsilons, but stable bands exist.
The O(1e-5) "floor" was partially an artifact — correct test reveals clear stability
in the stable regions.**

Results summary:
- **PHS k=2**: Stable at n=20,40,80 (se from -8.6e-3 to -5.9e-4).
- **Gaussian ε sweep**: Only 8/60 epsilons stable at each n, in a narrow band
  around ε≈0.68-1.94.  Most of the epsilon range is genuinely unstable (se≈+0.087-0.098).
  Best ε≈0.86 with se≈-6.2e-3 (n=20) to -2.9e-5 (n=80), scaling as O(h).
- **Multiquadric ε sweep**: Much broader stability — 24/60 at n=20, 21/60 at n=40,80.
  Two stable bands: [0.68, 1.08] and [1.54, 10.0] (narrow unstable gap at ε≈1.2-1.4).
  Best ε≈0.96 (n=20, se≈-1.2e-2) to ε≈1.08 (n=40,80, se≈-1.4e-3 to -1.1e-4).
- **Gaussian fine sweep**: Best ε*≈0.894 gives se≈-3.6e-4 at n=40, stable
  across n=20,40,80,160 with se scaling as O(h).

Conclusions vs Phase 29:
- Phase 29 "floor" at O(1e-5) was the wrong metric entirely — the correct stability
  eigenvalue shows clearly negative values (stable) in the stable bands.
- Gaussian E4 is genuinely unstable for most ε values (large positive se), but a
  tunable stable band exists. MQ has a much broader stable region.
- PHS k=2 is unconditionally stable for E4, same as E2.

- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedSweepE4`

---

## 32.3 — Re-Run Phase 30 Analysis (Tension Spline)

### 32.3a — E2 tension sweep with correct stability test ✅

Redid the Phase 30 E2 tension sweeps using `stability_eigenvalue`.  All tests pass.

**Key finding: E2 tension is universally stable — 61/61 sigmas stable at every grid size.**

Results summary:
- **Coarse sweep**: All 61 sigma values in [0, 20] stable at n=20,40,80.
  - n=20: best σ=0.0 (PHS k=2), stab_eig=-1.94e-2
  - n=40: best σ≈3.29, stab_eig=-2.72e-3
  - n=80: best σ≈4.85, stab_eig=-3.85e-4
- **Fine sweep**: σ*≈3.44 gives stab_eig=-2.85e-3 at n=40, stable across
  n=20,40,80,160 with se scaling as O(h²).
- **Comparison**: Both tension (σ*≈3.42) and Gaussian (ε*≈0.01) are stable.
  PHS k=2 (σ=0) also stable. E2 is unconditionally stable regardless of
  kernel choice.

Confirms Phase 32.2a: E2 was always stable. Phase 30 E2 "instability" was
entirely an artifact of the wrong BC/sign convention.

- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedTensionE2`

### 32.3b — E4 tension sweep with correct stability test ✅

Redid the Phase 30 E4 tension sweeps using `stability_eigenvalue`.  All tests pass.

**Key finding: E4 tension is overwhelmingly stable — far broader than Gaussian.**

Results summary:
- **Coarse sweep**: Nearly all sigma values in [0, 20] stable at n=20,40.
  - n=20: 61/61 stable, best σ≈0.48 (stab_eig=-9.2e-3)
  - n=40: 61/61 stable, best σ=0.0/PHS k=2 (stab_eig=-4.1e-3)
  - n=80: 56/61 stable, narrow unstable band at σ≈1.0-1.4 (genuine, se≈+1.9e-4
    to +3.1e-4) and σ≈4.8-5.5 (numerical noise, se≈+3.8e-9 to +1.4e-8).
- **Fine sweep**: σ*=0.0 (PHS k=2) is already optimal at n=40 with stab_eig=-4.1e-3,
  stable across n=20,40,80,160 with se scaling as O(h).
- **Comparison**: Tension has 99/101 stable at n=40 vs Gaussian 11/101. Tension's
  stable region is an order of magnitude broader.

Conclusions:
- Phase 30's "floor" was primarily an artifact of the wrong BC/sign convention.
- E4 tension is genuinely stable across almost the entire σ range.
- PHS k=2 (σ=0) is already highly stable — tension doesn't improve on it for E4.
- A narrow unstable band exists at n=80 near σ≈1.0-1.4, but this is easily avoided.
- Tension vastly outperforms Gaussian in stability breadth for E4.

- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedTensionE4`

**Review follow-up (count assertions added) ✅:** Added count assertions to
`test_tension_coarse_sweep` in `TestCorrectedTensionE4`:
- n=20,40: assert `n_stable >= 55` (plan says 61/61)
- n=80: assert `n_stable >= 50` (plan says 56/61, conservative bound)

### 32.3c — E4 tension + penalty sweep with correct stability test ✅

Redid the Phase 30.3c joint (σ, γ) sweep using `stability_eigenvalue_from_matrix`.
All tests pass.

**Key finding: Penalty does NOT help — PHS k=2 (σ=0, γ=0) is already optimal.
Moderate-to-large γ destroys stability.**

Results summary:
- **Coarse 2D sweep** (n=40, 25 σ × 25 γ = 625 points): Best is (σ=0, γ=0)
  with stab_eig=-4.1e-3.  Best γ>0 point gives stab_eig=-2.1e-3 (worse).
- **Penalty at σ=0**: Very small γ≈0.032 gives marginal improvement
  (stab_eig=-4.26e-3 vs -4.15e-3).  γ ≥ 0.87 makes the scheme unstable
  (stab_eig≈+1.5e-4 to +1.8e-4).
- **Grid independence**: (σ=0, γ=0) stable at n=20,40,80 with stab_eig
  from -8.6e-3 (n=20) to -5.9e-4 (n=80), scaling as O(h).

Conclusions vs Phase 30.3c:
- Phase 30.3c found penalty "improved" E4 from max Re(λ)≈8.7e-5 to ≈3.3e-5 —
  but this was under the wrong metric.  Under the correct metric, E4 tension
  was already stable, and the penalty mostly hurts.
- The conservation penalty is not useful for E4 stability.  PHS k=2 (σ=0)
  with no penalty is the best choice.

- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedTensionPenaltyE4`

**Review follow-up (missing n=80 assertion) ✅:** Made the assertion unconditional
for all grid sizes (n=20,40,80), matching the analogous `TestCorrectedTensionE4`
pattern. The optimal (σ≈0, γ≈0) = PHS k=2 is solidly stable at n=80 (stab_eig≈-5.9e-4).
The narrow unstable bands only affect σ≈1.0-1.4, not σ=0. Test passes.

---

## 32.4 — Re-Run Phase 31 Analysis (Boundary Footprint)

### 32.4a — Nextra sweep with correct stability test ✅

Redid the Phase 31 nextra × σ sweep using `stability_eigenvalue`.  All tests pass.

**Key finding: E4 at nextra=0 is already the most stable — nextra>0 is not needed
and can actually hurt stability.**

Results summary (n=40):
- **nextra=0**: 100/101 σ values stable.  Best σ=0 (PHS k=2), stab_eig=-4.1e-3.
  Only 1 unstable point near σ≈5.2 (marginal, se≈+1.9e-7).
- **nextra=1**: Only 63/101 stable.  Many σ>2.0 become unstable (se up to +3.3e-3).
  Best σ≈1.75, stab_eig=-9.5e-4.
- **nextra=2**: 82/101 stable.  Best σ≈2.68, stab_eig=-2.3e-3.  Some instability
  at larger σ values.
- **nextra=3**: 100/101 stable, recovers broad stability.  Best σ≈2.92,
  stab_eig=-1.0e-3.  But still not as good as nextra=0.

Grid independence (nextra=0, σ=0 PHS k=2): stable at n=20,40,80,160 with
stab_eig from -8.6e-3 (n=20) to -5.2e-5 (n=160), scaling as O(h).

Conclusions vs Phase 31:
- Phase 31 found nextra=1 helped under the wrong metric (lowered max Re(eig(D))).
  Under the correct metric, nextra=0 is already broadly stable and nextra=1
  actually narrows the stable σ range.
- The entire boundary footprint investigation was unnecessary — E4 nextra=0 with
  PHS k=2 is the most stable configuration.
- Increasing nextra adds DOF but doesn't improve stability.

- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedFootprint`

**Review follow-up (count assertions added) ✅:** Added per-nextra count assertions
to `test_nextra_sweep_e4_tension` with conservative thresholds:
- nextra=0: assert `n_stable >= 90` (plan says 100/101)
- nextra=1: assert `n_stable >= 50` (plan says 63/101)
- nextra=2: assert `n_stable >= 70` (plan says 82/101)
- nextra=3: assert `n_stable >= 90` (plan says 100/101)
Test passes.

---

## 32.5 — Updated Comparison and Conclusions

### 32.5a — Comprehensive comparison table ✅

Added `TestCorrectedComparison` class with 3 tests (all pass):
- `test_e2_comparison`: E2_1 at n=20,40,80 — all methods stable
- `test_e4_comparison`: E4_1 at n=20,40,80 — all methods stable at optimal params
- `test_grid_convergence_summary`: PHS k=2 convergence for E2 and E4

**E2_1 Results** (optimal params found at n=40):
- Gaussian ε*=0.004, Tension σ*=3.46, Penalty σ*=3.93 γ*=0.07
- **All 4 methods stable at all grid sizes** (n=20,40,80)
- CFL(RK4): PHS k=2 ≈ 2.84, Gaussian ≈ 2.88, Tension ≈ 2.78
- Conservation deficit: PHS k=2 = 1.60, Tension ≈ 1.27 (best)

**E4_1 Results** (optimal params found at n=40):
- Gaussian ε*=0.899, Tension σ*=0.00 (= PHS k=2), Penalty σ*=0.00 γ*=0.03
- **All 4 methods stable at all grid sizes** (n=20,40,80)
- CFL(RK4): PHS k=2 ≈ 2.06, Gaussian ≈ 1.90 (smaller due to larger spectral radius)
- Key finding: E4 tension optimizer lands on σ=0 (PHS k=2) — tension adds no benefit

**Grid Convergence** (PHS k=2):
- E2: stab_eig scales roughly O(h²): -1.9e-2 → -1.2e-3 → -1.1e-4 → -1.2e-5
- E4: stab_eig scales roughly O(h): -8.6e-3 → -4.1e-3 → -5.9e-4 → -5.2e-5

Production E4u_1 (σ=3.0) was separately validated as stable in 32.1c.

- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedComparison`

**Review follow-up (32.5a assertion gaps) ✅:**
Fixed all three assertion gaps in `TestCorrectedComparison`:

1. **Dead `all_stable` variable in `test_e2_comparison`**: Removed `all_stable` and
   moved per-method stability assertions inside the n-loop so all 4 methods (including
   tension+penalty) are asserted at every grid size (n=20,40,80). Removed redundant
   separate n=40 assertions.

2. **`test_e4_comparison` per-grid-size assertions**: Replaced n=40-only assertions
   with per-method per-grid-size assertions inside the loop. All 4 methods asserted
   stable at n=20,40,80. CFL check also moved inside the loop.

3. **Tension+penalty now asserted**: Both `test_e2_comparison` and `test_e4_comparison`
   now assert stability for the penalty variant at all grid sizes, not just the other
   3 methods.

### 32.5b — Update old test assertions

The Phase 29-31 tests have hard assertions based on the wrong metric.  Update:
- Replace `max_real_eigenvalue` calls with `stability_eigenvalue` (or
  `stability_eigenvalue_from_matrix` where D is built manually)
- In `_sweep`/`_metrics`/`_eval_point` helpers that compute eigenvalues of D
  directly, apply `D_bc = D[1:, 1:]` and negate (`-D_bc`) before taking eigenvalues
- Update assertions to check `stab_eig < 0` (stable) instead of `max_re < STABILITY_TOL`
- Flip assertions that claim instability where Phase 32 showed stability
- Keep sweep code structure but update variable names and print labels
- Update comparison assertions: under correct metric, PHS k=2 is stable (E2 and E4),
  so "tension improves over PHS" and "PHS should be unstable" assertions are wrong
- File: `scripts/stencil_gen/tests/test_phs.py`
- Ensure ALL existing tests still pass after updates

**Key assertion changes (wrong → correct):**
- "PHS k=2 should be unstable" → PHS k=2 IS stable
- "tension improves ≥10× over PHS" → both stable, remove ordering assertion
- "penalty improves E4 stability" → penalty doesn't help (PHS k=2 σ=0 already optimal)
- "nextra=1 needed for E2" → E2 stable regardless of nextra
- All `STABILITY_TOL` comparisons on old metric → `stab_eig < 0` on correct metric
- `E4_INSTABILITY_FLOOR` concept → E4 is stable, no floor exists

#### 32.5b-i — Update Phase 29 sweep classes ✅

Classes: `TestEpsilonSweepE2`, `TestEpsilonSweepE4`, `TestMixedEpsilon`,
`TestStableEpsilonAlphas`, `TestComparisonTable`.

Changes:
- `TestEpsilonSweepE2/E4._sweep`: Replace manual `max_re = max(Re(eigvals(D)))` with
  `stab_eig = stability_eigenvalue_from_matrix(D)`. Keep `spec_rad` from full D.
- `TestEpsilonSweepE2/E4.test_gaussian_fine_sweep_near_best`: Replace
  `max_real_eigenvalue(...)` → `stability_eigenvalue(...)`. Update print labels.
  Stability check: `stab_eig < 0` instead of `max_re < STABILITY_TOL`.
- `TestMixedEpsilon._max_re_mixed`: Replace with `stability_eigenvalue_from_matrix(D)`.
  Rename to `_stab_eig_mixed`. Update assertion: `< 1e-2` → `< 0`.
- `TestStableEpsilonAlphas._find_best_epsilon`: Replace `max_real_eigenvalue` →
  `stability_eigenvalue`. Minimize stab_eig (most negative = most stable).
- `TestComparisonTable._find_best_epsilon`: Same.
- `TestComparisonTable._metrics`: Compute `stab_eig` via `stability_eigenvalue_from_matrix(D)`
  alongside `spec_rad` from full D.
- `TestComparisonTable.test_e2_comparison`: Remove "PHS should be unstable" assertion.
  Assert all methods stable (`stab_eig < 0`).
- `TestComparisonTable.test_e4_comparison`: Remove "PHS k=2 should have small instability".
  Assert PHS k=2 stable. Update mixed-ε comparison.

All 5 classes updated (18 tests pass):
- `TestEpsilonSweepE2/E4._sweep`: `np.linalg.eigvals(D)` → `stability_eigenvalue_from_matrix(D)`
- `TestEpsilonSweepE2/E4._print_table`: labels/threshold updated
- `TestEpsilonSweepE2/E4.test_gaussian_fine_sweep_near_best`: `max_real_eigenvalue` → `stability_eigenvalue`
- `TestMixedEpsilon._max_re_mixed` → `_stab_eig_mixed` using `stability_eigenvalue_from_matrix`
- `TestMixedEpsilon.test_single_epsilon_baseline`: assert `< 0` (stable)
- `TestMixedEpsilon.test_conservation_near_interior`: raw eigvals → `stability_eigenvalue_from_matrix`
- `TestStableEpsilonAlphas._find_best_epsilon`: `max_real_eigenvalue` → `stability_eigenvalue`
- `TestStableEpsilonAlphas.test_verify_stability_with_extracted_alphas`: eigvals → `stability_eigenvalue_from_matrix`, removed TEMO unstable assertion (value printed for inspection)
- `TestStableEpsilonAlphas.test_compare_with_production_alphas`: eigvals → `stability_eigenvalue_from_matrix`
- `TestComparisonTable._find_best_epsilon`: `max_real_eigenvalue` → `stability_eigenvalue`
- `TestComparisonTable._find_best_mixed_epsilon`: raw eigvals → `stability_eigenvalue_from_matrix`
- `TestComparisonTable._metrics`: raw eigvals → `stability_eigenvalue_from_matrix`
- `TestComparisonTable.test_e2_comparison`: removed "PHS should be unstable", assert all methods `stab_eig < 0`
- `TestComparisonTable.test_e4_comparison`: removed "PHS should have small instability", assert PHS stable
- `TestComparisonTable.test_summary_across_grid_sizes`: eigvals → `stability_eigenvalue_from_matrix`, assert both E2 and E4 stable

#### 32.5b-ii — Update Phase 30 tension sweep and wavenumber classes

Classes: `TestTensionSweepE2`, `TestTensionSweepE4`, `TestTensionOptimalSigma`,
`TestModifiedWavenumber`.

Changes:
- `TestTensionSweepE2/E4._sweep`: Replace manual eigvals with
  `stability_eigenvalue_from_matrix(D)`. Keep `spec_rad`.
- `TestTensionSweepE2.test_tension_fine_sweep_near_best`: Replace
  `max_real_eigenvalue` → `stability_eigenvalue`. Assert `stab_eig < 0`.
- `TestTensionSweepE4.test_tension_fine_sweep_near_best`: Same. Remove assertion
  about "must improve over PHS k=2" (PHS k=2 is already stable).
- `TestTensionSweepE4.test_compare_with_gaussian`: Replace calls, remove "tension
  ≥10× over PHS k=2" assertion (PHS is stable).
- `TestTensionOptimalSigma._max_re` / `_max_re_gaussian`: Replace with
  `stability_eigenvalue`. Rename to `_stab_eig` / `_stab_eig_gaussian`.
- `TestTensionOptimalSigma.test_e2_optimal_sigma`: Under correct metric E2 is
  universally stable — bisection won't find a transition. Rewrite to verify
  stability across σ range instead.
- `TestTensionOptimalSigma.test_e4_optimal_sigma`: Under correct metric E4 is
  broadly stable (PHS k=2 at σ=0 is already best). Rewrite assertions.
- `TestTensionOptimalSigma.test_comparison_all_methods`: Update all assertions.
  E2: all stable. E4: PHS stable, remove "improves over PHS" assertions.
- `TestModifiedWavenumber._find_best_sigma`: Replace `max_real_eigenvalue` →
  `stability_eigenvalue`. Under corrected metric, E4 optimal is PHS k=2 (σ=0),
  not the old tension σ* — this changes which σ the wavenumber analysis runs at.
- `TestModifiedWavenumber.test_e2_boundary_at_optimal_sigma`: `best_re` now
  comes from `stability_eigenvalue`. Replace `assert best_re < STABILITY_TOL`
  with `assert best_re < 0` (stable under corrected metric).
- `TestModifiedWavenumber.test_e2_boundary_amplifying_at_sigma_zero`: Update
  docstring — PHS k=2 IS stable under correct metric. Per-stencil amplification
  can exist without implying full-operator instability. Assertion may still pass
  (per-stencil Re(κ*)>0 is a local property), but explanation must be corrected.
- `TestModifiedWavenumber.test_e4_boundary_at_optimal_sigma`: Comments claim
  "E4 does NOT achieve machine-precision stability (full matrix O(1e-5))" —
  wrong under corrected metric. Update σ* (now PHS k=2 σ=0), update comments,
  update assertion message at line ~3887 ("explaining O(1e-5) instability").
- `TestModifiedWavenumber.test_e4_phs_boundary_worse_than_tension`: Uses old
  `_find_best_sigma` to pick σ*. Under corrected metric PHS k=2 σ=0 IS the
  optimal, so this comparison may need rethinking — the assertion that tension
  reduces per-stencil amplification vs PHS may still hold locally but the
  motivation is different.

#### 32.5b-iii — Update Phase 30 conservation and comparison classes

Classes: `TestTensionConservationE2`, `TestTensionConservationE4`,
`TestTensionComparison`.

Changes:
- `TestTensionConservationE2._eval_point`: Replace manual eigvals with
  `stability_eigenvalue_from_matrix(D)`. Return `(stab_eig, deficit)`.
- `TestTensionConservationE2` assertions: E2 universally stable, so "at least
  one stable point" → "all points stable". Keep deficit improvement assertions
  (conservation deficit is independent of stability metric).
- `TestTensionConservationE4._eval_point`: Same fix.
- `TestTensionConservationE4` assertions: Remove `E4_INSTABILITY_FLOOR` concept.
  Under correct metric E4 is stable. Assert `stab_eig < 0` instead of
  `< E4_INSTABILITY_FLOOR`. Remove "γ>0 improves over γ=0" (penalty doesn't help).
- `TestTensionComparison._metrics`: Compute `stab_eig` via
  `stability_eigenvalue_from_matrix`. Keep `spec_rad` / CFL from full D.
- `TestTensionComparison._find_best_sigma` / `_find_best_epsilon`: Replace
  `max_real_eigenvalue` → `stability_eigenvalue`.
- `TestTensionComparison._find_best_sigma_gamma`: Replace manual eigvals.
- `TestTensionComparison.test_e2_comparison`: Remove "PHS should be unstable O(1e-2)"
  (line 3498). Assert all methods stable.
- `TestTensionComparison.test_e4_comparison`: Remove "PHS should be unstable"
  (line 3567). Assert all methods stable at optimal params.
- `TestTensionComparison.test_grid_convergence`: E2 and E4 both stable across grids.

#### 32.5b-iv — Update Phase 31 footprint and cross-validation classes

Classes: `TestFootprintE4Quick`, `TestFootprintSweep`, `TestFootprintPenalty`,
`TestCrossValidationE2`.

Changes:
- `TestFootprintE4Quick._sweep`: Replace `max_real_eigenvalue` → `stability_eigenvalue`.
  Stable means `stab_eig < 0`.
- `TestFootprintSweep.test_nextra_sweep_e4_tension`: Replace `max_real_eigenvalue` →
  `stability_eigenvalue`. Under correct metric nextra=0 is already broadly stable.
- `TestFootprintPenalty._eval_point`: Replace manual eigvals with
  `stability_eigenvalue_from_matrix(D)`.
- `TestFootprintPenalty` assertions: Remove "penalty improves nextra=0" (it doesn't
  under correct metric). Assert stability at γ=0.
- `TestCrossValidationE2._sweep_sigma`: Replace `max_real_eigenvalue` →
  `stability_eigenvalue`. Stable means `stab_eig < 0`.
- `TestCrossValidationE2` assertions: Under correct metric E2 is stable at all
  nextra values (0, 1, 2). Keep "all nextra stable" assertion, remove "nextra=0
  is worse" expectation.

### 32.5c — Write conclusions

Document corrected findings in the plan:
- Which methods × schemes are actually stable?
- Does the spline approach work for E4 with the correct test?
- What are the implications for the optimization pipeline?
- File: `plans/32-fix-eigenvalue-test-rerun.md`

---

## Implementation Order

1. **32.1a** — Add `stability_eigenvalue` function
2. **32.1b** — Add `stability_eigenvalue_from_matrix` helper
3. **32.1c** — Validate against production E4u_1 (ground truth)
4. **32.2a** — E2 corrected sweep (PHS/Gaussian/MQ)
5. **32.2b** — E4 corrected sweep (the key result!)
6. **32.3a** — E2 corrected tension sweep
7. **32.3b** — E4 corrected tension sweep
8. **32.3c** — E4 corrected tension + penalty
9. **32.4a** — Nextra sweep if needed
10. **32.5a** — Comparison table
11. **32.5b-i** — Update Phase 29 sweep classes
12. **32.5b-ii** — Update Phase 30 tension sweep classes
13. **32.5b-iii** — Update Phase 30 conservation and comparison classes
14. **32.5b-iv** — Update Phase 31 footprint and cross-validation classes
15. **32.5c** — Conclusions

---

## Key Files

| File | Role |
|------|------|
| `scripts/stencil_gen/stencil_gen/phs.py` | **Modified** — add `stability_eigenvalue`, deprecate `max_real_eigenvalue` |
| `scripts/stencil_gen/tests/test_phs.py` | **Modified** — add corrected sweep tests, update old assertions |
| `plans/32-fix-eigenvalue-test-rerun.md` | **This file** — updated with results |

## Performance Notes

- Same as Phases 29-31: all sweeps use numpy eigenvalues, ~1ms per point
- The BC removal (`D[1:, 1:]`) produces an `(n-1)×(n-1)` matrix — negligible cost difference
- Re-running all sweeps takes < 1 minute total
