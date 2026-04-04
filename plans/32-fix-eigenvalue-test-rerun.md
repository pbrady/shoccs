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

### 32.3a — E2 tension sweep with correct stability test

Redo tension σ sweep for E2.
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedTensionE2`

### 32.3b — E4 tension sweep with correct stability test

Redo tension σ sweep for E4.  This is the critical test: if the "floor" was entirely
due to the wrong sign/BC, tension splines may achieve full E4 stability.
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedTensionE4`

### 32.3c — E4 tension + penalty sweep with correct stability test

Redo (σ, γ) joint sweep for E4 with correct test.
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedTensionPenaltyE4`

---

## 32.4 — Re-Run Phase 31 Analysis (Boundary Footprint)

### 32.4a — Nextra sweep with correct stability test

Redo nextra × σ sweep for E4 with correct test.
- If E4 is now stable at nextra=0, this section confirms it and is brief.
- If E4 is STILL not stable, nextra sweep determines minimum footprint.
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedFootprint`

---

## 32.5 — Updated Comparison and Conclusions

### 32.5a — Comprehensive comparison table

For all methods × schemes × BCs:
- PHS k=2, Gaussian ε*, Tension σ*, Tension+penalty (σ*,γ*)
- E2 and E4, at n=20,40,80
- Correct `stability_eigenvalue` metric
- Also report RK4 max CFL at each configuration
- Compare with production E4u_1 stencil (ground truth)
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedComparison`

### 32.5b — Update old test assertions

The Phase 29-31 tests have hard assertions based on the wrong metric.  Update:
- Remove or comment out assertions that use `max_real_eigenvalue`
- Replace with assertions using `stability_eigenvalue`
- Keep the old sweep code but update the stability interpretation
- File: `scripts/stencil_gen/tests/test_phs.py`
- Ensure ALL existing tests still pass (may need to flip assertion directions)

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
11. **32.5b** — Update old test assertions
12. **32.5c** — Conclusions

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
