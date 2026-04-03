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

**Derivatives of the simplified kernel** φ(r;σ) = σ|r| - 1 + e^{-σ|r|}:
- D¹φ = σ · sign(r) · (1 - e^{-σ|r|})   [smooth at r=0, equals 0 there]
- D²φ = σ² · e^{-σ|r|}                    [for r≠0; delta term at r=0 ignored]

**Numerical stability:** For small z = σ|r|, use Taylor series (Horner form):
- φ = (z²/2)(1 - z/3 + z²/12 - z³/60 + ...)
- D¹φ = σ · sign(r) · z · (1 - z/2 + z²/6 - z³/24 + ...)
For large σr (>20), drop the e^{-σr} terms.

**Conditional positive definiteness:** Order 1 (needs at least constant augmentation).
With polynomial augmentation to degree q, the system is non-singular for distinct points.

---

## 30.1 — Tension Spline Kernel Implementation

### 30.1a — Add `_tension_kernel_eval` and `_tension_kernel_deriv` to `phs.py` ✅

Implement the tension spline kernel φ(r;σ) and its derivatives D¹φ, D²φ with
proper numerical handling:
- Branch on σ|r|: use Taylor series for σ|r| < 2, exponential form for σ|r| ≥ 2
- Handle r=0 (φ=0, D¹φ=0, D²φ=0)
- Handle σ=0 (return PHS k=2 values)

Add `"tension"` as a new kernel type in `_kernel_eval` and `_kernel_deriv`.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: `φ(r;0)` matches `|r|³` (PHS k=2), `φ(r;σ)` is positive for r>0

**Done:** Implemented `_tension_kernel_eval` (Taylor for z<2, direct for z≥2) and
`_tension_kernel_deriv` (nu=0,1,2) with 8-term Horner series. Added `"tension"` dispatch
in `_kernel_eval` and `_kernel_deriv`.

### 30.1b — Extend `phs_stencil_weights` to support tension kernel ✅

The existing function dispatches on `kernel` parameter.  Add the tension case:
- For `kernel="tension"`: use `_tension_kernel_eval` for Φ matrix entries,
  `_tension_kernel_deriv` for RHS entries
- The `epsilon` parameter serves as σ (the tension parameter)
- Use the numpy numeric path (like Gaussian/MQ) for efficiency
- Add `_rbf_weights_numeric` branch for `kernel="tension"`
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: tension weights match PHS k=2 when σ→0

**Done:** Added `"tension"` to `phs_stencil_weights` dispatch (routes to `_rbf_weights_numeric`)
and to `_rbf_weights_numeric` (Phi and dPhi computation via element-wise kernel calls).

### 30.1c — Add convenience wrappers ✅

```python
def uniform_boundary_weights_tension(i, t, nu, q, sigma):
    """Boundary row i using tension spline with parameter sigma."""
```

Also `uniform_interior_weights_tension` for verification.
- File: `scripts/stencil_gen/stencil_gen/phs.py`

**Done:** Added `uniform_boundary_weights_tension` and `uniform_interior_weights_tension`
(thin wrappers delegating to `_rbf` variants with `kernel="tension"`).

### 30.1d — Tests for tension kernel ✅

Add `TestTensionSpline` class in `test_phs.py`:
- `test_sigma_zero_matches_phs_k2`: At σ=0 (or very small), tension weights ≈ PHS k=2
- `test_polynomial_exactness`: Exact for polynomials ≤ q at any σ
- `test_weights_sum_to_zero`: First derivative weights sum to 0
- `test_kernel_symmetry`: φ(r;σ) = φ(-r;σ) (even function)
- `test_interior_matches_classical`: Interior weights match classical FD for all σ
- `test_numerical_stability_large_sigma`: No overflow for σ up to 50 on unit grid
- File: `scripts/stencil_gen/tests/test_phs.py`

**Done:** 11 tests in `TestTensionSpline` — all pass. Additional tests beyond spec:
`test_kernel_positive_for_nonzero_r`, `test_kernel_zero_at_origin`, `test_d1_antisymmetric`,
`test_d2_symmetric`, `test_taylor_matches_direct`.

---

## 30.1-review — Follow-up items from review of Phase 30.1

### 30.1-review-a — Add σ=0 guard dispatching to PHS k=2 ✅

The plan spec 30.1a requires "Handle σ=0 (return PHS k=2 values)" but this was
not implemented.  At σ=0 the kernel returns 0 for all r, producing a singular Φ
matrix and a `LinAlgError`.  This **blocks Phase 30.2** which sweeps σ ∈ [0, 20].

**Done:** Added guard in `phs_stencil_weights` — when `kernel="tension"` and
`|epsilon| < 1e-14`, redirects to exact PHS k=2 path and converts to float.
This covers all callers (`uniform_boundary_weights_tension`,
`uniform_interior_weights_tension`, `build_diff_matrix_rbf`, etc.).
Test `test_sigma_exactly_zero_dispatches_to_phs` verifies boundary and interior
weights match PHS k=2 at σ=0.0 to machine precision.

### 30.1-review-b — Add D¹φ Taylor 8th term for branch-point accuracy ✅

The D¹φ Taylor series uses 7 terms (up to z⁶/5040) while the eval kernel and
D²φ use 8 terms.  At the branch point z=2, this gives ~0.6% discontinuity for
D¹φ vs ~0.01% for eval.  Add the z⁷/40320 term to the D¹φ Horner series for
consistent accuracy.

**Done:** Added `z * (-1.0 / 40320)` 8th term to D¹φ Horner series, matching
the 8-term depth of eval and D²φ.

### 30.1-review-c — Add nu=2 stencil weight test ✅

All stencil weight tests use nu=1.  Add a test that computes second-derivative
weights via `uniform_boundary_weights_tension(i, t, nu=2, q, sigma)` and verifies
polynomial exactness (sum w_j x_j^d = d(d-1) i^{d-2} for d ≥ 2).  This exercises
the D²φ code path through `_rbf_weights_numeric`.

**Done:** `test_nu2_polynomial_exactness` tests q=2,3 with all boundary rows,
verifying D² polynomial exactness to 1e-10.

---

## 30.2 — Sigma Stability Sweep

### 30.2a — Extend `build_diff_matrix_rbf` for tension kernel ✅

Ensure the differentiation matrix builder works with `kernel="tension"`.
May already work if the kernel dispatch in `_rbf_weights_numeric` is correct.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: matrix is n×n, interior column sums are 0, matches PHS at σ=0

**Done:** The existing code path already supports `kernel="tension"` — `build_diff_matrix_rbf`
passes `kernel` to `uniform_boundary_weights_rbf`, which delegates to `phs_stencil_weights`,
which dispatches tension to `_rbf_weights_numeric`. Updated docstrings in `build_diff_matrix_rbf`,
`build_diff_matrix_mixed_epsilon` to mention `"tension"`. Added `TestBuildDiffMatrixTension`
(8 tests): shape, interior column sums, σ=0 matches PHS k=2, polynomial reproduction,
antisymmetry, nu=2 reproduction, finite eigenvalues, and mixed-epsilon tension. All pass.

### 30.2b — Sigma sweep for E2 (p=1, q=1) ✅

Sweep σ over [0, 20] and record max Re(λ) at each σ for n=20,40,80.
- Compare with Gaussian ε sweep results from Phase 29
- Key question: does the σ=0 limit (PHS k=2) connect smoothly to a stable region?
- If yes: report σ* and compare with Gaussian ε*
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestTensionSweepE2`

**Done:** 3 tests in `TestTensionSweepE2` — all pass. Key findings:
- **YES, the σ=0 limit connects smoothly to a stable region.** max Re(λ)
  decreases monotonically from ~0.087 at σ=0 (PHS k=2) to machine precision
  (~1e-14) at σ ≈ 5.5, remaining stable for all larger σ.
- The transition is smooth: σ=1 gives max Re(λ)≈0.083, σ=3 gives ~0.018,
  σ=4.85 gives ~0.0016, σ=5.5 reaches machine precision.
- Stability is grid-independent: verified at n=20,40,80,160 with max Re(λ) < 5e-14.
- **Comparison with Gaussian ε*≈1.36:** Both achieve machine-precision stability.
  The tension kernel reaches stability at σ≈5.5 (vs ε≈1.36 for Gaussian).
  Both are effectively zero to machine precision at their optima.

---

## 30.2-review — Follow-up items from review of Phase 30.2b

### 30.2-review-a — Add regression assertions to E2 tension sweep tests ✅

The three tests in `TestTensionSweepE2` have zero `assert` statements — they only
print output.  If the tension kernel is broken, these tests still pass silently.
Add at least one assertion per test that codifies the key finding:

- `test_tension_coarse_sweep`: assert that for n=40, the minimum max Re(λ) across
  the sweep is < 1e-10 (i.e., machine-precision stability exists).
- `test_tension_fine_sweep_near_best`: assert that the fine-sweep best max Re(λ)
  is < 1e-10 and that the multi-grid check (n=20,40,80,160) all satisfy < 1e-10.
- `test_compare_with_gaussian`: assert that both tension and Gaussian bests are
  < 1e-10 (both achieve machine-precision stability for E2).

**Done:** Added assertions to all three tests:
- `test_tension_coarse_sweep`: asserts best n=40 result < STABILITY_TOL.
- `test_tension_fine_sweep_near_best`: asserts fine-sweep best < STABILITY_TOL
  AND all four grid sizes (n=20,40,80,160) satisfy < STABILITY_TOL.
- `test_compare_with_gaussian`: asserts both tension and Gaussian bests < STABILITY_TOL.

### 30.2-review-b — Use practical stability threshold instead of strict ≤ 0 ✅

The sweep tests classify stability as `max_re <= 0`.  Floating-point eigenvalue
computation yields tiny positive residuals (~1e-14) for genuinely stable operators,
so all results print "unstable" despite being machine-precision stable.  This makes
the output misleading and will make 30.2c E4 results uninterpretable (true instability
at 1e-4 and machine-precision stability at 1e-14 get the same label).

Fix: define a threshold constant (e.g., `STABILITY_TOL = 1e-10`) and use
`max_re < STABILITY_TOL` for stability classification in both `TestTensionSweepE2`
and the existing `TestEpsilonSweepE2`/`TestEpsilonSweepE4` classes.  This should be
done before 30.2c so the E4 sweep results are immediately interpretable.

**Done:** Added `STABILITY_TOL = 1e-10` module-level constant.  Updated all
`<= 0` stability checks to `< STABILITY_TOL` in `TestEpsilonSweepE2`,
`TestEpsilonSweepE4`, and `TestTensionSweepE2` (9 occurrences total).
All 9 sweep tests pass.

---

### 30.2c — Sigma sweep for E4 (p=2, q=3) ✅

Same sweep for E4 boundary stencils.
- The critical test: does tension do better than Gaussian for E4?
- Since tension starts from PHS k=2 (the best prior result, Re=0.006) and
  deforms continuously, it may find a path to stability that the Gaussian
  (starting from a different point in stencil space) missed.
- Also try per-row σ (mixed-tension, using `build_diff_matrix_mixed_epsilon`)
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestTensionSweepE4`

**Done:** 4 tests in `TestTensionSweepE4` — all pass. Key findings:
- **E4 tension is NOT machine-precision stable.** Best uniform σ gives
  max Re(λ) ≈ 5.3e-5 (at σ≈37, n=40), comparable to Gaussian's ~8e-5.
- Tension significantly improves over PHS k=2 (σ=0): 0.006 → 5e-5, a 100×
  reduction, but does not reach machine precision.
- Mixed-tension (two-group, σ_outer/σ_inner): best max Re(λ) ≈ 5.4e-5 at
  σ_outer≈12.2, σ_inner≈6.5. No improvement over uniform σ.
- The O(1e-4–1e-5) floor for E4 appears to be a fundamental barrier that
  neither Gaussian ε nor tension σ (nor mixed per-row) can breach.
- Stability is NOT grid-independent: best σ from n=40 gives larger max Re(λ)
  at n=20 (~2e-4), n=80 (~1.8e-4), n=160 (~2.6e-4).

---

## 30.2c-review — Follow-up items from review of Phase 30.2c

### 30.2c-review-a — Add regression assertions to E4 tension sweep tests ✅

Three of four tests in `TestTensionSweepE4` have zero `assert` statements —
the same problem fixed for E2 in 30.2-review-a.  Only `test_tension_coarse_sweep`
has assertions; the other three pass silently if the tension kernel breaks.

Since E4 is NOT machine-precision stable (best ≈ 5e-5), use a loose regression
threshold (e.g., 1e-3) rather than `STABILITY_TOL`:

- `test_tension_fine_sweep_near_best`: assert fine-sweep best < 1e-3 (actual ~5e-5).
  Assert that tension improves over PHS k=2 baseline (best < σ=0 value).
- `test_compare_with_gaussian`: assert both tension and Gaussian bests < 1e-3.
  Assert tension improves ≥ 10× over PHS k=2 (from ~0.006 to ~5e-5 is ~100×).
- `test_mixed_tension_two_group`: assert mixed best < 1e-3.

**Done:** Added assertions to all three tests:
- `test_tension_fine_sweep_near_best`: asserts fine-sweep best < 1e-3, and
  that tension improves over PHS k=2 baseline (σ=0).
- `test_compare_with_gaussian`: asserts both tension and Gaussian bests < 1e-3,
  and that tension improves ≥ 10× over PHS k=2.
- `test_mixed_tension_two_group`: asserts mixed best < 1e-3.
All 4 E4 sweep tests pass.

---

### 30.2d — Fine-grained search near optimal σ ✅

If the sweep finds a minimum in max Re(λ):
1. Refine with bisection or Brent's method to find σ* precisely
2. Report the stencil weights at σ*
3. Compare max Re(λ) with Gaussian ε* and PHS k=2
- File: `scripts/stencil_gen/tests/test_phs.py`

**Done:** 3 tests in `TestTensionOptimalSigma` — all pass. Key findings:

- **E2_1:** Bisection finds σ_crit ≈ 5.02 (sharp transition to stability).
  σ* = σ_crit + 1.0 = 6.02 gives max Re(λ) ≈ 1e-14 (machine precision).
  Grid-independent: all sizes n=20,40,80,160 satisfy max Re(λ) < 1e-6.
  Boundary stencil weights at σ*=6.02:
  - row 0: [-0.759, +0.680, -0.082, +0.161]
  - row 1: [-0.530, -0.007, +0.604, -0.067]
  - row 2: [+0.067, -0.604, +0.007, +0.530]

- **E4_1:** Dense sweep (400 points over [5,55]) finds noisy/oscillatory
  landscape.  Best σ*≈50 gives max Re(λ)≈6.8e-5.  Top-10 best results
  are scattered across σ ∈ [14, 55] with median 7.3e-5.  The O(1e-4)
  floor is confirmed as a fundamental barrier — no single σ achieves
  machine precision.  NOT grid-independent.

- **All-methods comparison (n=40):**
  | Scheme | PHS k=2 (σ=0) | Gaussian ε* | Tension σ* |
  |--------|----------------|-------------|------------|
  | E2_1   | 8.7e-2         | 1.3e-15 ✓   | 7.4e-16 ✓  |
  | E4_1   | 6.4e-3         | 5.7e-5      | 3.8e-5     |
  Tension improves E4 by 169× over PHS k=2 (vs Gaussian's 112×).

---

## 30.2d-review — Follow-up items from review of Phase 30.2d

### 30.2d-review-a — Assert E2 grid-independence in `test_e2_optimal_sigma` ✅

`test_e2_optimal_sigma` computes `all_stable` across n=20,40,80,160 but never
asserts on it.  The plan claims grid-independent stability at σ* and the code
already has the check — just add:

```python
assert all_stable, "E2 optimal σ not grid-independent"
```

after the grid-independence loop (line ~2518 in `test_phs.py`).

**Done:** Added `assert all_stable, "E2 optimal σ not grid-independent"` after
the existing grid-independence loop.  Test passes.

### 30.2d-review-b — Complete regression assertions in `test_comparison_all_methods` ✅

The comparison test asserts only that PHS baselines are reasonable (< 0.5 for E2,
< 0.05 for E4) but never asserts on the actual Gaussian or tension results.
Comments say "E2: both Gaussian and tension should achieve near-machine-precision"
and "E4: all methods should improve over baseline" but neither is tested.

Add assertions that:
- E2 Gaussian and tension bests are each < STABILITY_TOL (actual ≈ 1e-15).
- E4 Gaussian and tension bests are each < 1e-3 (actual ≈ 5e-5).
- E4 Gaussian and tension each improve over PHS k=2 baseline.

This requires computing the sweep results outside the print loop or saving them
during the loop for later assertion (the current structure discards per-scheme
results after each loop iteration).

**Done:** Saved per-scheme results during the loop and added 8 assertions after:
- E2: PHS baseline < 0.5, Gaussian < STABILITY_TOL, Tension < STABILITY_TOL.
- E4: PHS baseline < 0.05, Gaussian < 1e-3, Tension < 1e-3, both improve over PHS k=2.
All pass.

---

## 30.3 — Soft Conservation Penalty

### 30.3a — Implement penalty-augmented RBF-FD system ✅

Add a new function that solves the weighted least-squares problem:

    minimize  ‖Φλ + Pᵀμ - dΦ‖² + γ ‖Cλ - b_c‖²
    subject to  Pλ = dP  (polynomial exactness, hard constraint)

where C encodes conservation column-sum constraints and γ is the penalty weight.
This distributes conservation across all rows rather than dumping it on one.

Implementation: form the augmented normal equations or use a constrained least-squares
solver.  The system is still linear; the solution exists for all γ ≥ 0.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: γ=0 recovers standard RBF weights, γ→∞ approaches conservation-enforced weights

**Done:** Added `build_diff_matrix_rbf_penalty` to `phs.py`. Implementation uses
null-space projection: computes the standard RBF weights b₀, then adjusts via
b = b₀ + Z α where Z is the block-diagonal null space of the polynomial constraint
(preserving polynomial exactness as a hard constraint). The α is found by solving
(I + γ GᵀG) α = γ Gᵀr₀ where G = C Z and r₀ is the conservation deficit at b₀.
Right boundary is automatically conservative by the antisymmetric reflection.

6 tests in `TestConservationPenalty` — all pass:
- `test_gamma_zero_matches_standard_e2/e4`: γ=0 gives identical D to standard.
- `test_conservation_improves_with_gamma_e2/e4`: deficit decreases with γ and
  converges (E2: 1.22 → 0.85; E4: similar pattern).
- `test_polynomial_exactness_preserved_e2/e4`: polynomial reproduction < 1e-8
  at all γ values.

**Key finding:** Full conservation is NOT achievable while maintaining polynomial
exactness.  The null space of P has dimension t−(q+1) per row, but all rows share
the same null space, so the effective column-sum freedom is only t−(q+1) dimensions
vs t conservation equations.  The penalty reduces the deficit to a fundamental limit
(~30% reduction for E2), not to zero.  This is consistent with the TEMO approach's
need to sacrifice one boundary row's freedom for conservation.

### 30.3b — Joint (σ, γ) sweep for E2 ✅

Sweep both tension parameter σ and conservation penalty γ:
- σ ∈ [0, 10], γ ∈ [0, 100]
- Find: is there a (σ*, γ*) where both max Re(λ) ≤ 0 AND conservation deficit < threshold?
- This is a 2D search, but each evaluation is ~1ms (numpy eigenvalue)
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestTensionConservationE2`

**Done:** 3 tests in `TestTensionConservationE2` — all pass. Key findings:

- **YES, stable + improved conservation exists** for E2.  Best coarse-sweep
  point: σ≈9.3, γ=100 achieves max Re(λ) < 1e-10 AND deficit reduced 29%
  (1.20 → 0.85).
- **Stability boundary widens with σ:** at σ=6.2, stability holds only up to
  γ≈0.27; at σ=8.3, up to γ≈37; at σ=9.3, γ=100+ still stable.
- **At σ=6 (Phase 30.2d optimal):** very tight stability budget — only γ≤0.18,
  giving ~10.5% deficit improvement (1.22 → 1.09).
- **Conservation improvement saturates at ~30%:** deficit reduces from ~1.20 to
  ~0.85, consistent with the fundamental rank-limited null-space constraint
  from Phase 30.3a.
- **Grid independence NOT achieved at best combined point:** σ=8, γ=100 is
  stable at n=40 but unstable at n=20 (2.3e-3) and n=80 (9.0e-4).
- **Trade-off:** larger σ tolerates more conservation penalty but the combined
  (σ, γ) point loses grid-independence that the γ=0 result had.

---

## 30.3b-review — Follow-up items from review of Phase 30.3b

### 30.3b-review-a — Assert conservation improvement at γ > 0 in joint tests ✅

The plan's key finding is "deficit reduced 29% (1.20 → 0.85)" — that the
penalty mechanism improves conservation while maintaining stability.  But none
of the three 30.3b tests assert that a stable point with γ > 0 has better
deficit than the γ=0 baseline.  If `build_diff_matrix_rbf_penalty` silently
stopped applying the penalty (returning γ=0 weights regardless of γ), all
tests would still pass:

- `test_joint_sweep_coarse`: asserts stable points exist and γ=0 baseline
  exists, but not that γ > 0 improves deficit.
- `test_stability_survives_moderate_penalty`: asserts γ > 0 is stable, but
  not that deficit improves.
- `test_fine_sweep_near_optimal`: asserts `best_deficit ≤ deficit_baseline`,
  but `best_gamma` could be 0 (trivially satisfying the assertion).

Fix: in `test_fine_sweep_near_optimal`, assert `best_gamma > 0` (verifying
the optimizer found a non-trivial penalty point that improves conservation).
In `test_joint_sweep_coarse`, assert that the best stable deficit at γ > 0 is
strictly less than the γ=0 baseline deficit (actual improvement is ~29%).

**Done:** Added both assertions:
- `test_joint_sweep_coarse`: tracks `best_deficit_gamma_pos` during the sweep
  and asserts it is strictly less than the γ=0 `baseline_deficit`.
- `test_fine_sweep_near_optimal`: asserts `best_gamma > 0` to verify the
  optimizer picks a non-trivial penalty point.
All 3 tests pass.

### 30.3c — Joint (σ, γ) sweep for E4 ✅

Same for E4.  This is the key test: can the 2D (σ, γ) space find what the 1D σ
and 1D ε spaces could not?
- File: `scripts/stencil_gen/tests/test_phs.py`

**Done:** 3 tests in `TestTensionConservationE4` — all pass. Key findings:

- **NO, the 2D (σ, γ) space does NOT achieve machine-precision stability for E4.**
  The O(1e-4–1e-5) barrier persists even with conservation penalty.
- **2D improves over 1D:** Best uniform σ (γ=0) gives max Re(λ) ≈ 8.7e-5;
  best (σ, γ) = (44.2, 1.7) gives max Re(λ) ≈ 3.3e-5, a ~63% improvement.
- **Conservation slightly improves:** deficit reduces from 1.71 (γ=0) to 1.65
  at best (σ, γ) point — modest ~3.5% improvement.
- **At fixed σ=37 (Phase 30.2c optimal):** best γ≈1.15 gives max Re(λ) ≈ 7.5e-5,
  slight improvement over γ=0 (1.3e-4), but landscape is noisy/oscillatory.
- **Grid independence NOT achieved:** best n=40 point gives larger max Re(λ)
  at n=20 (2.2e-4) and n=80 (1.1e-4).
- **Conclusion:** the O(1e-5) instability floor for E4 is a fundamental barrier
  that cannot be breached by 1D σ, 1D ε, mixed per-row, or 2D (σ, γ) searches.
  This is consistent with Phase 30.2c's conclusion.

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

1. **30.1a** — Tension kernel evaluation (numerical, with Taylor/exp branching) ✅
2. **30.1b** — Wire into `phs_stencil_weights` and `_rbf_weights_numeric` ✅
3. **30.1c** — Convenience wrappers ✅
4. **30.1d** — Tension kernel tests ✅
5. **30.1-review-a** — Add σ=0 guard dispatching to PHS k=2 (blocks 30.2) ✅
6. **30.1-review-b** — Add D¹φ Taylor 8th term for branch-point accuracy ✅
7. **30.1-review-c** — Add nu=2 stencil weight test ✅
8. **30.2a** — Diff matrix builder for tension ✅
9. **30.2b** — E2 sigma sweep (first result: does PHS k=2 connect to stability?) ✅
10. **30.2-review-a** — Add regression assertions to E2 tension sweep tests ✅
11. **30.2-review-b** — Use practical stability threshold (blocks 30.2c) ✅
12. **30.2c** — E4 sigma sweep (key result: does tension beat Gaussian?) ✅
13. **30.2c-review-a** — Add regression assertions to E4 tension sweep tests ✅
14. **30.2d** — Fine-grained optimal σ search ✅
15. **30.2d-review-a** — Assert E2 grid-independence in `test_e2_optimal_sigma` ✅
16. **30.2d-review-b** — Complete regression assertions in `test_comparison_all_methods` ✅
17. **30.3a** — Soft conservation penalty implementation ✅
18. **30.3b** — E2 (σ, γ) sweep ✅
19. **30.3b-review-a** — Assert conservation improvement at γ > 0 ✅
20. **30.3c** — E4 (σ, γ) sweep ✅
21. **30.4a** — Comparison table
22. **30.4b** — Modified wavenumber analysis
23. **30.4c** — Update plan with conclusions

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
