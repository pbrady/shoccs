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

### 32.1a — Add `stability_eigenvalue` function to `phs.py`

Replace `max_real_eigenvalue` with a correct function:

```python
def stability_eigenvalue(
    n: int, p: int, q: int, epsilon: float,
    kernel: str = "gaussian", nu: int = 1, nextra: int = 0,
) -> float:
    """Return the maximum real part of the eigenvalues of -D with inflow BCs.

    For the advection equation u_t + u_x = 0, the semi-discrete system is
    du/dt = -D u.  Stability requires all eigenvalues of -D_bc to have
    non-positive real parts, where D_bc is D with the inflow row/column
    removed (Dirichlet at inflow, floating at outflow).

    Returns the maximum real part of eigenvalues of -D_bc.
    A non-positive return value means the scheme is stable.
    """
    D = build_diff_matrix_rbf(n, p, q, epsilon, kernel, nu, nextra)
    D_bc = D[1:, 1:]  # remove inflow (first row and column)
    eigs = np.linalg.eigvals(-D_bc)
    return float(np.max(eigs.real))
```

Also add variants for mixed-epsilon and penalty matrices.

Keep `max_real_eigenvalue` as deprecated (or remove) to prevent confusion.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: production E4u_1 alphas at n=20,40,80 all return ≤ 0 (stable)
- Test: periodic interior-only matrix returns ≈ 0 (neutrally stable)

### 32.1b — Add `stability_eigenvalue_from_matrix` helper

For cases where the matrix is built externally:
```python
def stability_eigenvalue_from_matrix(D: np.ndarray) -> float:
    """Max Re(eigenvalue of -D_bc) where D_bc = D[1:, 1:]."""
    D_bc = D[1:, 1:]
    return float(np.max(np.linalg.eigvals(-D_bc).real))
```
- File: `scripts/stencil_gen/stencil_gen/phs.py`

### 32.1c — Validate against production E4u_1

Add test class `TestStabilityInfrastructure`:
- `test_production_e4u1_stable`: Build D from E4u_1 C++ formulas at production alphas
  (-0.7733, 0.1624), verify `stability_eigenvalue_from_matrix(D) < STABILITY_TOL` at
  n=20, 40, 80, 160.
- `test_interior_only_neutrally_stable`: Periodic interior-only matrix has
  `stability_eigenvalue ≈ 0`.
- `test_wrong_alphas_unstable`: E4u_1 at alpha=(0.1, 0.7) should be unstable
  (positive `stability_eigenvalue`), confirming the test can detect instability.
- File: `scripts/stencil_gen/tests/test_phs.py`

---

## 32.2 — Re-Run Phase 29 Analysis (PHS + Gaussian + Multiquadric)

### 32.2a — E2 PHS/Gaussian/MQ sweep with correct stability test

Redo the Phase 29 sweeps using `stability_eigenvalue`:
- PHS k=2 for E2 at n=20,40,80
- Gaussian ε sweep for E2
- Multiquadric ε sweep for E2
- Key question: was E2 always stable, or does the answer change?
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestCorrectedSweepE2`

### 32.2b — E4 PHS/Gaussian/MQ sweep with correct stability test

Same for E4:
- PHS k=2 for E4
- Gaussian ε sweep for E4
- Key question: is the O(1e-5) "floor" still present, or was it an artifact?
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
