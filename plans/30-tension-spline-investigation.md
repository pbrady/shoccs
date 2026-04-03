# Phase 30: Tension Spline FD Stencils — σ-Tunable Stability

**Goal:** Implement tension spline (L-spline) kernels in the RBF-FD framework and
determine whether the tension parameter σ can produce stable boundary stencils.  The
tension spline kernel φ(r;σ) = σ|r| - 1 + e^{-σ|r|} continuously deforms between PHS
k=2 at σ=0 (nearly stable) and exponentially-fitted FD at large σ (maximally dissipative),
providing a physics-motivated 1D optimization parameter.

**Depends on:** Phase 29 RBF engine (committed: `stencil_gen/phs.py` with Gaussian/MQ)

**Priority:** Active research — tension splines have richer structure than Gaussian RBFs
because (1) they reduce to the best-performing PHS k=2 at σ=0, (2) they exactly
reproduce exponential modes e^{±σx} in addition to polynomials, and (3) the tension
parameter has a physical interpretation as characteristic decay length.

**Read first:**
- `scripts/stencil_gen/stencil_gen/phs.py` (existing RBF engine — extend with tension kernel)
- `scripts/stencil_gen/tests/test_phs.py` (existing tests — extend with tension tests)
- `scripts/stencil_gen/stencil_gen/interior.py` (classical interior reference)
- `scripts/stencil_gen/stencil_gen/temo.py` (SchemeParams, E2_1, E4_1)
- `src/stencils/E2_1.cpp` (production E2 reference)
- `src/stencils/E4_1.cpp` (production E4 reference)

**Test commands:**
```bash
cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_phs.py -x -v
```

---

## Background

### Why tension splines, not just another RBF?

The Phase 29 Gaussian RBF investigation found:
- E2_1: stable at ε*≈1.83 but non-conservative
- E4_1: min instability O(1e-4), NOT stable for any ε
- Conservation enforcement destroys stability

The tension spline kernel has three advantages over Gaussian:

1. **Correct σ=0 limit:** At σ=0, the tension kernel reduces to PHS k=2 (|r|³),
   which was the best-performing kernel (max Re(λ)=0.006 for E4).  The Gaussian at
   ε=0 degenerates to a constant — useless.  So tension splines START from the best
   known point and deform continuously.

2. **Exponential mode reproduction:** The tension spline exactly reproduces e^{±σx}
   in addition to polynomials.  For PDEs with exponential boundary layers or
   characteristic decay (screened Poisson, convection-diffusion), this matches the
   physics.  The stencil is optimal for the operator L = D⁴ - σ²D² in the native-
   space norm.

3. **Localization:** As σ increases, the kernel decays exponentially (e^{-σ|r|}),
   making the stencil progressively more local.  This naturally controls the "reach"
   of the boundary stencil into the interior.

### The kernel

The tension spline Green's function (1D, operator D⁴ - σ²D²):

    φ(r; σ) = (1/(2σ³)) · (σ|r| + e^{-σ|r|} - 1)

or equivalently (dropping the constant and normalization, which cancel in the RBF system):

    φ(r; σ) = σ|r| - 1 + e^{-σ|r|}

**Limits:**
- σ→0: φ → |r|³/6 + O(σ²) (PHS k=2, cubic spline)
- σ→∞: φ → σ|r| (linear, maximally local)

**Derivatives (for the RHS of the augmented system):**
- D¹φ = (r/(2σ)) · sinh(σ|r|)      [note: smooth at r=0, equals 0 there]
- D²φ = (sinh(σ|r|) + σ|r|cosh(σ|r|)) / (2σ)

**Numerical stability:** For small σr, use Taylor series (Horner form):
- φ = |r|³(σ²/6)(1 + (σr)²/10 + (σr)⁴/280 + ...)
- D¹φ = r|r|/2 · (1 + (σr)²/6 + (σr)⁴/120 + ...)
For large σr (>20), drop the e^{-σr} terms.

**Conditional positive definiteness:** Order 1 (needs at least constant augmentation).
With polynomial augmentation to degree q, the system is non-singular for distinct points.

---

## 30.1 — Tension Spline Kernel Implementation

### 30.1a — Add `_tension_kernel_eval` and `_tension_kernel_deriv` to `phs.py`

Implement the tension spline kernel φ(r;σ) and its derivatives D¹φ, D²φ with
proper numerical handling:
- Branch on σ|r|: use Taylor series for σ|r| < 2, exponential form for σ|r| ≥ 2
- Handle r=0 (φ=0, D¹φ=0, D²φ=0)
- Handle σ=0 (return PHS k=2 values)

Add `"tension"` as a new kernel type in `_kernel_eval` and `_kernel_deriv`.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: `φ(r;0)` matches `|r|³` (PHS k=2), `φ(r;σ)` is positive for r>0

### 30.1b — Extend `phs_stencil_weights` to support tension kernel

The existing function dispatches on `kernel` parameter.  Add the tension case:
- For `kernel="tension"`: use `_tension_kernel_eval` for Φ matrix entries,
  `_tension_kernel_deriv` for RHS entries
- The `epsilon` parameter serves as σ (the tension parameter)
- Use the numpy numeric path (like Gaussian/MQ) for efficiency
- Add `_rbf_weights_numeric` branch for `kernel="tension"`
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: tension weights match PHS k=2 when σ→0

### 30.1c — Add convenience wrappers

```python
def uniform_boundary_weights_tension(i, t, nu, q, sigma):
    """Boundary row i using tension spline with parameter sigma."""
```

Also `uniform_interior_weights_tension` for verification.
- File: `scripts/stencil_gen/stencil_gen/phs.py`

### 30.1d — Tests for tension kernel

Add `TestTensionSpline` class in `test_phs.py`:
- `test_sigma_zero_matches_phs_k2`: At σ=0 (or very small), tension weights ≈ PHS k=2
- `test_polynomial_exactness`: Exact for polynomials ≤ q at any σ
- `test_weights_sum_to_zero`: First derivative weights sum to 0
- `test_kernel_symmetry`: φ(r;σ) = φ(-r;σ) (even function)
- `test_interior_matches_classical`: Interior weights match classical FD for all σ
- `test_numerical_stability_large_sigma`: No overflow for σ up to 50 on unit grid
- File: `scripts/stencil_gen/tests/test_phs.py`

---

## 30.2 — Sigma Stability Sweep

### 30.2a — Extend `build_diff_matrix_rbf` for tension kernel

Ensure the differentiation matrix builder works with `kernel="tension"`.
May already work if the kernel dispatch in `_rbf_weights_numeric` is correct.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: matrix is n×n, interior column sums are 0, matches PHS at σ=0

### 30.2b — Sigma sweep for E2 (p=1, q=1)

Sweep σ over [0, 20] and record max Re(λ) at each σ for n=20,40,80.
- Compare with Gaussian ε sweep results from Phase 29
- Key question: does the σ=0 limit (PHS k=2) connect smoothly to a stable region?
- If yes: report σ* and compare with Gaussian ε*
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestTensionSweepE2`

### 30.2c — Sigma sweep for E4 (p=2, q=3)

Same sweep for E4 boundary stencils.
- The critical test: does tension do better than Gaussian for E4?
- Since tension starts from PHS k=2 (the best prior result, Re=0.006) and
  deforms continuously, it may find a path to stability that the Gaussian
  (starting from a different point in stencil space) missed.
- Also try per-row σ (mixed-tension, using `build_diff_matrix_mixed_epsilon`)
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestTensionSweepE4`

### 30.2d — Fine-grained search near optimal σ

If the sweep finds a minimum in max Re(λ):
1. Refine with bisection or Brent's method to find σ* precisely
2. Report the stencil weights at σ*
3. Compare max Re(λ) with Gaussian ε* and PHS k=2
- File: `scripts/stencil_gen/tests/test_phs.py`

---

## 30.3 — Soft Conservation Penalty

### 30.3a — Implement penalty-augmented RBF-FD system

Add a new function that solves the weighted least-squares problem:

    minimize  ‖Φλ + Pᵀμ - dΦ‖² + γ ‖Cλ - b_c‖²
    subject to  Pλ = dP  (polynomial exactness, hard constraint)

where C encodes conservation column-sum constraints and γ is the penalty weight.
This distributes conservation across all rows rather than dumping it on one.

Implementation: form the augmented normal equations or use a constrained least-squares
solver.  The system is still linear; the solution exists for all γ ≥ 0.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: γ=0 recovers standard RBF weights, γ→∞ approaches conservation-enforced weights

### 30.3b — Joint (σ, γ) sweep for E2

Sweep both tension parameter σ and conservation penalty γ:
- σ ∈ [0, 10], γ ∈ [0, 100]
- Find: is there a (σ*, γ*) where both max Re(λ) ≤ 0 AND conservation deficit < threshold?
- This is a 2D search, but each evaluation is ~1ms (numpy eigenvalue)
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestTensionConservation`

### 30.3c — Joint (σ, γ) sweep for E4

Same for E4.  This is the key test: can the 2D (σ, γ) space find what the 1D σ
and 1D ε spaces could not?
- File: `scripts/stencil_gen/tests/test_phs.py`

---

## 30.4 — Analysis and Conclusions

### 30.4a — Comparison table

Produce comparison of all approaches investigated:
- PHS k=2 (Phase 29 baseline)
- Gaussian ε* (Phase 29 result)
- Tension σ* (this phase)
- Tension + soft conservation (σ*, γ*)
- Metrics: max Re(λ), spectral radius, CFL with RK4, conservation deficit
- For E2 and E4
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestTensionComparison`

### 30.4b — Modified wavenumber analysis

For the best tension stencil found:
- Compute κ*(ξ;σ) = Σ w_j exp(ijξ) for the boundary rows
- Plot Re(κ*) and Im(κ*) vs ξ
- Compare with interior stencil κ*_int
- Verify: is Re(κ*_bdy) ≤ 0 for all ξ at the optimal σ?
- File: `scripts/stencil_gen/tests/test_phs.py`

### 30.4c — Update plan with conclusions

Document findings and next steps.
- File: `plans/30-tension-spline-investigation.md`

---

## Implementation Order

1. **30.1a** — Tension kernel evaluation (numerical, with Taylor/exp branching)
2. **30.1b** — Wire into `phs_stencil_weights` and `_rbf_weights_numeric`
3. **30.1c** — Convenience wrappers
4. **30.1d** — Tension kernel tests
5. **30.2a** — Diff matrix builder for tension
6. **30.2b** — E2 sigma sweep (first result: does PHS k=2 connect to stability?)
7. **30.2c** — E4 sigma sweep (key result: does tension beat Gaussian?)
8. **30.2d** — Fine-grained optimal σ search
9. **30.3a** — Soft conservation penalty implementation
10. **30.3b** — E2 (σ, γ) sweep
11. **30.3c** — E4 (σ, γ) sweep
12. **30.4a** — Comparison table
13. **30.4b** — Modified wavenumber analysis
14. **30.4c** — Update plan with conclusions

---

## Key Files

| File | Role |
|------|------|
| `scripts/stencil_gen/stencil_gen/phs.py` | **Modified** — add tension kernel + soft conservation |
| `scripts/stencil_gen/tests/test_phs.py` | **Modified** — add tension tests + sweeps |
| `plans/30-tension-spline-investigation.md` | **This file** — updated with results |

## Performance Notes

- All sweeps use numpy (no SymPy) — eigenvalue computation is ~1ms per point
- Tension kernel evaluation: slightly more expensive than Gaussian (exp + linear terms
  vs just exp) but negligible for n ≤ 10 stencil points
- The 2D (σ, γ) sweep at 100×100 resolution: ~10,000 eigenvalue computations at n=40,
  total < 10 seconds
- Taylor series branching: use 8 terms of Horner form for σ|r| < 2, verified to give
  full double-precision accuracy

## Key Mathematical References

- Schweikert (1966): original tension spline formulation
- Cline (1974): practical tension spline algorithms
- Renka (1993): TSPACK — reference implementation with numerical stability handling
- Green's function: φ(r;σ) = (σ|r| + e^{-σ|r|} - 1)/(2σ³)
- Conditional positive definiteness: order 1 (needs ≥ constant augmentation)
- σ=0 limit: PHS k=2 (|r|³/12), σ→∞ limit: linear interpolation
