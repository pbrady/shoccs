# Scientific Findings

Non-obvious results discovered during plans 40–44 that a new agent won't derive by reading the code alone. These drive design decisions in downstream plans (45, 46, 47).

## 1. Tension-spline family is universally BL42-infeasible

**Finding:** Tension-spline boundary closures on the E4 scheme **never** pass the BL §4.2 reflecting-hyperbolic stability test (L3r), regardless of the tension parameter σ. Verified via differential-evolution optimization over σ ∈ [0.01, 50] with 898 evaluations (~384 s): DE finds the minimum at σ → lower bound with `max_spectral_abscissa ≈ 0.63`, **10 orders of magnitude above `BL42_TOL = 1e-10`**.

Sample points from the scan:

| σ | L3 max_stab_eig | L3r max_spectral_abscissa |
|---|---|---|
| 0.01 (DE-found min) | passes | 0.63 |
| 0.5 | passes (−7e−4) | 0.65 |
| 3.0 (paper's choice) | passes (−7.8e−6) | **0.95** |
| 6.0 | passes | 1.46 |
| 10.0 | passes | 1.19 |
| 20.0 | passes | 1.34 |

**Why:** The tension-spline boundary closure is stable under the 1D advection model problem (L3) but not under the energy-conserving reflecting-BC system (L3r). The continuous L3r operator has `div(c) = 0` and purely imaginary spectrum; any discrete `Re(λ) > tol` is unambiguous boundary-closure instability.

**Implication:** For applications where the user's physics involves energy-conserving boundaries (acoustic problems, standing waves, elliptic-hyperbolic systems), tension-spline closures are disqualified. Classical-α closures remain the only option. The Gaussian and multiquadric families show similar behavior (also fail L3r at their paper-cited optimal epsilon).

## 2. Classical-α landscape is multi-modal; our framework finds different basins than Brady-Livescu

**Finding:** Running `scipy.optimize.differential_evolution` on `layer_bl42.max_spectral_abscissa` for E4 classical α over `α₀ ∈ [-2, 2], α₁ ∈ [0.05, 2]` found `α = [-1.399, 0.293]` with `max_re ≈ 4e-15` — machine epsilon. Brady-Livescu's published value `α ≈ [-0.7733, 0.1624]` is in a **different basin** (~2 units away in α₀), and also passes BL42 cleanly (`max_re ≈ 3e-14`).

Both points are valid feasible optima. The paper explicitly acknowledges multi-modality: Table 4 of Brady-Livescu 2019 reports **101 found E4 schemes** from random restarts, all passing their linear+nonlinear stability tests.

**Implication:** Don't expect reproduction of BL's exact published values. The optimization framework is working correctly when it finds *any* feasible basin with machine-precision spectral abscissa. The right validation is "are the basin's scheme stable on the target test problems" not "does α₀ match BL's 4 decimal places."

## 3. Varying-coefficient operators need explicit h-scaling

**Finding:** `scripts/stencil_gen/stencil_gen/phs.py::build_diff_matrix_rbf(n, ...)` returns weights for **unit grid spacing** (integer grid {0, 1, ..., n-1}). The BL §4.3 domain is `[0, √2]²` with physical spacing `h = √2/(n-1)`. The coefficient field `c_x(x, y)` is evaluated at physical coordinates. The semi-discrete operator was initially built as `L = -(diag(c_x) * kron(I, D) + diag(c_y) * kron(D, I))` using `D` unscaled — so eigenvalue *magnitudes* were off by a factor of `(n-1)/√2` (~21 at n=31). Signs were unaffected (so qualitative stability verdicts survived the bug), but the `L7_TOL` threshold was originally calibrated against the wrongly-scaled eigenvalues.

**Fixed in commit `843c974`** by dividing D by h before assembly. `L7_TOL` recalibrated to 0.1 to separate known-stable (0.018) from known-unstable (~3.1).

**Related:** BL §4.3 has `div(c) = 1/ψ > 0` (diverging radial flow), so the continuous operator is genuinely *not* skew-symmetric — the homogeneous semi-discrete operator inherently has positive spectral abscissa on the order of `0.5 * div(c) * L_domain ~ 1`. The discrete operator faithfully reproduces this. A strictly-zero threshold would flag all physically-correct BL §4.3 schemes as unstable.

## 4. BL §4.2 continuous eigenvalues are ±i(2k−1)π/2, not ±ikπ

**Finding (plan 44.1a correction):** The continuous spectrum of the BL §4.2 operator
```
∂u/∂t = ∂v/∂x,   ∂v/∂t = ∂u/∂x,   u(0) = 0,  v(1) = 0
```
is **`λ_k = ±i(2k-1)π/2`** for k = 1, 2, 3, ... — odd half-integer multiples of π, not integer multiples. The IC `u(x, 0) = sin(3πx/2)` is the k=2 eigenmode: `(2·2-1)π/2 = 3π/2`.

**Why it matters:** If you're validating `layer_bl42_reflecting_hyperbolic` discrete eigenvalues against the continuous reference, you need the right reference.

## 5. L1 `boundary_gv_err` is monotone across σ

**Finding:** The L1 scalar objective `boundary_gv_err` decreases monotonically as σ → 0 because the tension spline approaches PHS k=2 (which has the cleanest low-ξ dispersion for this stencil structure). Minimizing `layer1.boundary_gv_err` via scipy always drives σ to whatever lower bound you set — it's not a useful discriminating objective on its own.

**Implication:** For interior-minimum optimization, use `layer3.max_stab_eig`, `layer6.transient_growth_bound`, or `layer7.max_spectral_abscissa`. `layer1.*` is fine as a gate but bad as a primary objective.

## 6. Classical-α feasible region is below the C++ cut-cell bound

**Finding:** The C++ `src/stencils/E4_1.cpp` enforces `alpha[1] >= 197/288 ≈ 0.684` (cut-cell singularity avoidance). Brady-Livescu's feasible region sits at `α₁ ≈ 0.162` — well below the cut-cell constraint. A grid probe confirms L3-feasibility only in `α₁ ∈ [~0.08, ~0.17]` and total infeasibility once `α₁ ≥ 0.2`.

**Implication:** The Python L1–L7 cascade operates on the *uniform* feasible envelope; the C++ cut-cell constraint is separate. For BL §4.3 (uniform domain, no embedded geometry) use `type="E4u"` in Lua, which is the uniform-boundary dispatch and doesn't enforce the 197/288 bound. L8 records a `cpp_cutcell_violates_197_288` diagnostic flag when the optimizer's winner would violate that constraint.

## 7. Spline coefficients cannot be lifted to C++ codegen

**Finding:** The tension, Gaussian, and multiquadric kernels contain `exp(-σ·r)`, `exp(-(εr)²)`, or `sqrt(1 + (εr)²)` — SymPy cannot reduce these to polynomial closed form via CSE. The existing `E4_1.cpp` pattern (runtime α substituted into a CSE tree) does not work for RBF kernels.

**Solution (plan 42):** Move the 10×10 linear solve into the C++ struct *constructor*. Coefficients are computed once per simulation (not per call) and cached in an `std::array<real, 5*7>`. The `nbs_floating` hot path just reads the cache — zero per-call overhead. Verified: Python reference matches C++ coefficients to ≥14 significant digits at σ=3.0 for tension.

**Implication:** Adding new kernel families to C++ means writing the linear-solve Gaussian elimination in C++ (per kernel). Not a codegen-only task. The `StencilGenSpec.scalar_params` codegen extension is still useful for the C++ struct-field emission, but the solver body is hand-written per kernel.

## 8. Current L3r calibration across reference families

From `known_values.json["brady2d_calibration"]` after the plan-44 update:

| Family | L3 (advection) | L3r (BL §4.2) | Overall |
|---|---|---|---|
| E4_classical (BL α) | pass | **pass** (`max_re ≈ 3e-14`) | **pass through L6** |
| E2_phs_k2 | pass | pass | pass through L6 |
| E2_tension_6 | pass | fail at L1 | fail at L1 |
| E2_gaussian_2 | pass | fail at L1 | fail at L1 |
| E2_multiquadric_1 | pass | fail at L1 | fail at L1 |
| E4_phs_k2 | pass | **fail** at L3r | fail |
| E4_tension_3 | pass | **fail** at L3r (`max_re ≈ 0.95`) | fail |
| E4_gaussian_09 | pass | **fail** at L3r | fail |
| E4_multiquadric_1 | pass | **fail** at L3r | fail |

**Only two families pass through L6** in this calibration snapshot: E4_classical and E2_phs_k2. The sparse/RBF families on E4 all get caught by the new L3r layer that L3 alone misses.

**Caveat:** The E2 RBF families failing at L1 is because L1 (boundary_gv_err) has `L1_TOL = 0.05`, and the E2 boundary closure has lower accuracy than E4 so per-row GV errors are larger. This is a gate threshold that can be loosened if you care about E2 scoring.

## 9. Multi-objective trade-offs now visible

We now have three stability-informative metrics that disagree on which closures are good:

| Metric | Tension favors | Classical favors |
|---|---|---|
| `layer1.boundary_gv_err` | **Low** (~3.6e-2 at σ=3) | Higher (~1e-1) |
| `layer3.max_stab_eig` | Passes cleanly at many σ | Passes cleanly at BL α |
| `layer_bl42.max_spectral_abscissa` | **Fails universally** (~0.95 at σ=3) | Passes (~3e-14 at BL α) |
| `layer6.transient_growth_bound` | ~3.3 at σ=3 | Varies by basin (found 4.42 in one DE run) |

A user's choice depends on the physics. This trade-off space is what plan 45 (multi-objective Pareto) is designed to expose.

## 10. Non-normality diagnostics reveal "spectrally stable, practically unstable" cases

The non-normality module reveals that some schemes with `max Re(λ) < 0` still have dangerous transient growth:

- Varying-coefficient hyperbolic operators are **strongly non-normal** (Henrici departure ≥ 0.1 is typical).
- `eigenvector_condition` can exceed 10⁴ — meaning eigendecomposition is numerically meaningless and pseudospectral radius is the right metric.
- Kreiss constant `K > 10` indicates transient growth `‖exp(tL)‖` up to `e·K ~ 27` before decay kicks in — enough to trigger nonlinear blow-up in a real simulation even though the linear operator is asymptotically stable.

**Implication:** `max Re(λ) < tol` is *necessary* for stability but not *sufficient* for practical stability. Using `transient_growth_bound = e · kreiss_constant` as an optimization objective catches these cases that eigenvalue minimization alone misses.
