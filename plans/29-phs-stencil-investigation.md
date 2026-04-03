# Phase 29: Gaussian RBF Stencil Investigation — Shape Parameter Optimization

**Goal:** Determine whether Gaussian RBF-FD stencils with a tunable shape parameter ε
can produce stable boundary stencils, reducing the multi-alpha optimization to a 1D
search.  Prior PHS investigation (committed) showed PHS k=2 eliminates 99.9% of
boundary instability but isn't quite stable (max Re(λ) = 0.006 for E4).  The Gaussian
RBF adds a continuous shape parameter that may close the gap.

**Depends on:** Phase 29 PHS engine (committed: `stencil_gen/phs.py`)

**Priority:** Active research — if successful, replaces expensive multi-alpha optimizer.

**Read first:**
- `scripts/stencil_gen/stencil_gen/phs.py` (existing PHS engine to extend)
- `scripts/stencil_gen/tests/test_phs.py` (existing tests to extend)
- `scripts/stencil_gen/stencil_gen/interior.py` (classical interior reference)
- `scripts/stencil_gen/stencil_gen/temo.py` (SchemeParams, E2_1, E4_1 definitions)
- `src/stencils/E2_1.cpp` (4 alphas, R=4, T=5 — production reference)
- `src/stencils/E4_1.cpp` (2 alphas, R=5, T=7 — production reference)

**Test commands:**
```bash
cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_phs.py -x -v
```

---

## Current State

Phase 29 PHS investigation complete (committed).  Key findings:
- PHS k=2 (cubic spline) gives max Re(λ) = 5.7e-14 for E2, 0.006 for E4
- Higher k → worse instability (anti-Runge effect confirmed)
- Conservation correction on a single row makes stability worse
- Cut-cell coefficients grow as O(1/psi), same as polynomial — TEMO still needed

The Gaussian RBF φ(r) = exp(-ε²r²) has a continuous shape parameter ε that
interpolates between heavy smoothing (large ε) and polynomial interpolation (ε→0).
If there exists an ε* where the boundary stencil is exactly stable, the entire
multi-alpha optimization reduces to a 1D root-finding problem.

---

## 29.5 — Gaussian RBF Stencil Engine

### 29.5a — Add Gaussian RBF kernel to `phs.py` ✅

Extend `phs_stencil_weights` to accept a `kernel` parameter.  Currently it uses
`φ(r) = |r|^(2k-1)` (PHS).  Add support for:
- `"gaussian"`: φ(r) = exp(-ε²r²), parameterized by ε
- `"multiquadric"`: φ(r) = √(1 + ε²r²)
- `"phs"`: existing |r|^(2k-1) (default, unchanged)

The augmented system structure is the same: `[Φ P; P' 0] [λ; μ] = [dΦ; dP]`.
Only the kernel matrix Φ and RHS dΦ change.

Implementation:
- Add `kernel` parameter to `phs_stencil_weights(points, x_eval, nu, q, k=None, kernel="phs", epsilon=None)`
- For Gaussian: `Φ_{ij} = exp(-ε²(x_i - x_j)²)`, `dΦ_i = D^nu exp(-ε²(x_eval - x_i)²)`
- For Multiquadric: `Φ_{ij} = √(1 + ε²(x_i - x_j)²)`, derivatives accordingly
- Factor kernel logic into `_kernel_eval(r, kernel, k, epsilon)` and `_kernel_deriv(r, nu, kernel, k, epsilon)`
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: Gaussian with ε→0 limit should approach polynomial FD (high-k PHS limit)
- Test: Gaussian weights should sum to 0 for first derivative (polynomial reproduction from P block)

**Completed:** Added `_kernel_eval`, `_kernel_deriv` (SymPy-based, all 3 kernels),
`_rbf_weights_numeric` (numpy path for Gaussian/MQ), and extended `phs_stencil_weights`
with `kernel`/`epsilon` keyword args.  PHS path uses `_kernel_eval`/`_kernel_deriv` for
the augmented system.  Gaussian/MQ dispatches to numpy for efficiency.  All 12 existing
PHS tests pass.  Smoke-tested: boundary weights vary with ε, polynomial reproduction holds.

### 29.5b — Add `uniform_boundary_weights_rbf` convenience wrapper ✅

```python
def uniform_boundary_weights_rbf(i, t, nu, q, epsilon, kernel="gaussian"):
    """Boundary row i using Gaussian RBF with shape parameter epsilon."""
```

Also add `uniform_interior_weights_rbf` for verification.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: interior weights with Gaussian at any ε should still match classical FD
  (the polynomial augmentation forces this)

**Completed:** Added `uniform_interior_weights_rbf` and `uniform_boundary_weights_rbf`
wrappers that delegate to `phs_stencil_weights` with `kernel`/`epsilon` kwargs.

### 29.5c — Tests for new kernels ✅

Add `TestGaussianRBF` class in `test_phs.py`:
- `test_polynomial_exactness`: Gaussian stencils exact for polynomials ≤ q
- `test_interior_matches_classical`: Gaussian interior = classical FD for all ε
- `test_weights_sum_to_zero`: First derivative weights sum to 0
- `test_small_epsilon_interior_matches_polynomial`: Interior weights converge to classical in flat limit
- `test_boundary_weights_bounded`: Boundary weights remain bounded across ε range
- `test_multiquadric_polynomial_exactness`: MQ kernel exact for polynomials ≤ q
- File: `scripts/stencil_gen/tests/test_phs.py`

**Completed:** Added `TestGaussianRBF` class with 6 tests (all pass).  Note: the
flat-limit test only checks interior stencils where the system is fully determined
(2p+1 = q+1); over-determined boundary systems are ill-conditioned as ε→0.

---

## 29.6 — Epsilon Stability Sweep

### 29.6a — Build differentiation matrix with RBF boundary stencils ✅

Add helper function `build_diff_matrix_rbf(n, p, q, epsilon, kernel, nu)` that:
1. Uses RBF boundary stencils for left/right boundary rows
2. Uses classical interior stencils (standard 2p+1 centered FD)
3. Right boundary = antisymmetric reflection of left (for first derivative)
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: matrix shape n×n, column sums of interior region are 0

**Completed:** Added `build_diff_matrix_rbf(n, p, q, epsilon, kernel, nu, nextra)` that
computes boundary dimensions from p, q, nextra (matching temo.compute_dimensions),
fills interior rows with classical centered FD, left boundary with RBF+poly stencils,
and right boundary with (-1)^nu reflection.  Tests: shape, interior row sums, boundary
nonzero, antisymmetry, polynomial reproduction (D @ x = 1).  All 5 tests pass.

### 29.6b — Implement `max_real_eigenvalue(n, p, q, epsilon, kernel)` diagnostic ✅

Compute eigenvalues of the differentiation matrix and return the maximum real part.
Uses numpy for numerical eigenvalue computation.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: periodic interior-only matrix has max Re(λ) ≈ 0

**Completed:** Added `max_real_eigenvalue()` wrapper that calls `build_diff_matrix_rbf`
and returns `max(Re(eigvals))`.  Tests: periodic interior-only matrix has
max Re(λ) < 1e-12, return type is float.  All pass.

### 29.6-fix — Fix `build_diff_matrix_rbf` dimension formula for nu=2

`build_diff_matrix_rbf` hardcodes the nu=1 dimension formula
(`t = p+q+1+nextra`, `r = q+1+nextra`) but the function accepts nu=2.
For nu=2, `temo.compute_dimensions` uses `t = p+2+nextra`, `r = p+1+nextra`.
Calling with nu=2 silently produces wrong boundary dimensions.

Either:
- Gate on nu and use the correct formula per `temo.compute_dimensions`, or
- Restrict to nu=1 and raise `NotImplementedError` for nu=2.

Also add a test that `build_diff_matrix_rbf` with E2_2 parameters (p=1, q=1,
nextra=0, nu=2) produces the same dimensions as `temo.compute_dimensions`.

- File: `scripts/stencil_gen/stencil_gen/phs.py`, `build_diff_matrix_rbf`

### 29.6c — Epsilon sweep for E2 (p=1, q=1, nextra=1)

Sweep ε over [0.01, 10] and record max Re(λ) at each ε for n=20,40,80.
**Important:** E2_1 requires `nextra=1` — pass it explicitly to
`max_real_eigenvalue` (default is 0, which gives wrong boundary dimensions).
- Find: is there an ε* where max Re(λ) ≤ 0?
- If yes: report ε* and the corresponding alpha values
- If no: report the minimum max Re(λ) achieved and the ε that achieves it
- Also sweep the Multiquadric kernel for comparison
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestEpsilonSweepE2`
- Output: printed table of (ε, max_Re, spectral_radius) for visual inspection
- This test uses `pytest -s` to show output; assertion is only that the sweep completes

### 29.6d — Epsilon sweep for E4 (p=2, q=3)

Same sweep for E4 boundary stencils.
- E4 is the primary production target; finding a stable ε here is the key result
- Record implied alpha values at each ε by comparing against symbolic stencil
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestEpsilonSweepE4`

### 29.6e — If stable ε found: extract and validate alpha values

If an ε* exists where the scheme is stable:
1. Extract the FD weights at ε*
2. Map them back to alpha values in the `derive_uniform_boundary_for_temo` parameterization
3. Verify: substitute those alphas into the symbolic stencil and confirm eigenvalue stability
4. Compare with the optimizer-derived alphas used in `src/stencils/E4_1.cpp`
5. Check conservation deficit at those alpha values
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestStableEpsilonAlphas`

### 29.6f — If NO stable ε found: characterize the gap

If no single ε produces stability:
1. Report the minimum instability (min over ε of max Re(λ))
2. Try mixed approach: different ε per boundary row
3. Try: Gaussian boundary rows 0..r-2 with one ε, near-interior row from conservation
4. Summarize findings and update plan with next research direction
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestMixedEpsilon`

---

## 29.7 — Comparison and Assessment

### 29.7a — Side-by-side comparison table

For the best RBF configuration found, produce a comparison table:
- PHS k=2 vs Gaussian ε* vs optimizer-derived (from C++ stencils)
- Metrics: max Re(λ), spectral radius, implied CFL with RK4, conservation deficit
- For E2 and E4
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestComparisonTable`

### 29.7b — Update plan with conclusions and next steps

Update this plan file with:
- Which kernel/parameter works best
- Whether the approach eliminates or reduces the optimization problem
- What the cut-cell implications are (does the stable ε work for non-uniform grids?)
- Concrete next steps (if successful: wire into codegen; if not: alternative directions)
- File: `plans/29-phs-stencil-investigation.md`

---

## Implementation Order

Each step produces tests that validate before moving on:

1. **29.5a** — Gaussian/Multiquadric kernel support in phs.py
2. **29.5b** — Convenience wrappers for uniform grids
3. **29.5c** — Tests for new kernels
4. **29.6a** — Differentiation matrix builder
5. **29.6b** — max_real_eigenvalue diagnostic
6. **29.6-fix** — Fix nu=2 dimension formula in build_diff_matrix_rbf
7. **29.6c** — E2 epsilon sweep (first result!) — remember nextra=1
8. **29.6d** — E4 epsilon sweep (key result!)
8. **29.6e** or **29.6f** — Extract alphas or characterize gap
9. **29.7a** — Comparison table
10. **29.7b** — Update plan with conclusions

---

## Key Files

| File | Role |
|------|------|
| `scripts/stencil_gen/stencil_gen/phs.py` | **Modified** — add Gaussian/MQ kernels + matrix builder |
| `scripts/stencil_gen/tests/test_phs.py` | **Modified** — add kernel tests + epsilon sweeps |
| `plans/29-phs-stencil-investigation.md` | **This file** — updated with results |

## Performance Notes

- `SYMPY_CACHE_SIZE=50000` for all SymPy operations
- Eigenvalue sweeps use numpy (fast) — no SymPy needed for numerics
- The epsilon sweep (29.6c/d) involves ~100-1000 eigenvalue computations at n=40,
  each taking ~1ms → total sweep < 1 second
- SymPy only needed for alpha extraction (29.6e), which is a one-time computation
