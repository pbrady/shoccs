# Phase 41: Analytical Stability Stack for the Brady-Livescu 2D Varying-Coefficient Benchmark

**Goal:** Build a layered Python stability-scoring pipeline for the Brady & Livescu 2019 §4.3 two-dimensional varying-coefficient scalar advection test. The pipeline discriminates stable vs unstable boundary closures using **only analytical / per-operator metrics** (group velocity, rigorous Kreiss determinant, sparse eigenvalues, non-normality diagnostics) — deferring the full C++ simulation as a last-resort validator handled in plan 42. All spline/RBF families currently supported by `stencil_gen` are scored so plan 42 can prioritize the most promising candidates for C++ implementation.

**Depends on:** Phase 40 (complete — group velocity sweep integration)

**Background — the test problem (Brady & Livescu 2019 §4.3, pp. 92–94):**

```
u_t + grad(psi) . grad(u) = 0    on [0, sqrt(2)]^2,  t in [0, 1000]
psi(x, y) = sqrt((x + 0.25)^2 + (y + 0.25)^2)
c_x = d(psi)/dx = (x + 0.25) / psi
c_y = d(psi)/dy = (y + 0.25) / psi
u(x, y, 0)   = sin(2*pi*psi)
u(0, y, t)   = sin(2*pi*(psi(0, y) - t))    (inflow, Dirichlet)
u(x, 0, t)   = sin(2*pi*(psi(x, 0) - t))    (inflow, Dirichlet)
Exact: u(x, y, t) = sin(2*pi*(psi - t))
```

**Why an analytical stack instead of running the simulation:**

Neither Brady-Livescu paper forms the full 2D differentiation matrix — they assess stability purely by long-time simulation, which is expensive (many RK4 steps, 500+ wave periods) and only gives a binary blow-up/no-blow-up verdict. Every layer of the analytical stack below is strictly cheaper and strictly more informative than running the C++ simulation, and the layered short-circuit lets a sweep reject candidates as soon as the cheapest failing layer trips. The C++ simulation becomes a *validation* step at the end (plan 42), not the primary discriminator.

**The layered pipeline (cheapest first):**

| Layer | Metric | What it catches | Cost at E4 / N=81 |
|---|---|---|---|
| L1 | Interior + boundary GV error (1D, per direction) | Dispersion-quality mismatch | sub-ms |
| L2 | Rigorous GKS Kreiss determinant test | Boundary-closure instability — necessary **and sufficient** for the 1D reduction | ~100 ms |
| L3 | 1D eigenvalue `max Re(lambda(-D_bc))` at N in {20,40,80} | Semi-discrete asymptotic stability, constant coefficient | ~30 ms |
| L4 | Per-point local GV error across the varying-coefficient 2D grid | Local dispersion error induced by varying `(c_x, c_y)` | ~tens ms |
| L5 | 2D anisotropy `max angle_error` over propagation angles, evaluated per coefficient-field sample | Grid anisotropy interacting with the radial flow | ~100 ms |
| L6 | Non-normality diagnostics: spectral + numerical abscissa, Henrici departure, pseudospectral abscissa, Kreiss constant, transient-growth bound `e*K` | Transient growth from non-normal spatial operator | ~seconds |
| L7 | Sparse 2D Arnoldi `max Re(lambda)` on the full varying-coefficient operator at N in {21,31,61,91} | True 2D semi-discrete asymptotic stability | few seconds per N |
| L8 | (plan 42) C++ simulation via Lua bridge | Actual long-time L-infinity bound; catches what analytical layers miss | ~25 s build + ~minutes sim |

Each layer's output is reduced to at least one scalar the sweep can compare against a threshold. A single orchestrator `brady2d_stability_score(scheme, kernel, params, max_layer)` runs layers in order and short-circuits on failure.

**Read first:**

- `papers/BradyLivescu2019.pdf` (pp. 92–94, §4.3 — the PDE setup and stability assessment. Also pp. 87–88, §2.4 for the optimization context, and pp. 7–8 §4.1 for the 1D eigenvalue baseline.)
- `papers/StabilityAndGroupVelocity.pdf` (Trefethen 1983, pp. 204–210 — the rigorous GKS determinant condition statement and imaginary-axis perturbation check. **Required** for plan item 41.3.)
- `scripts/stencil_gen/stencil_gen/group_velocity.py` (lines 294–1076 — existing interior / boundary / cut-cell / 2D / GKS-heuristic functions that the new layers build on)
- `scripts/stencil_gen/stencil_gen/phs.py` (lines 407–500 — `_rbf_weights_numeric` kernel dispatch; lines 622–840 — `build_diff_matrix_rbf`, `stability_eigenvalue`, `stability_eigenvalue_from_matrix`)
- `scripts/stencil_gen/sweeps/known_values.json` (the per-family optimal parameters that seed the calibration phase; `known_unstable` entries are ground-truth for negative tests)
- `scripts/stencil_gen/sweeps/_common.py` (`SCHEME_PARAMS` — only E2 and E4; scope excludes E6/E8)
- `plans/40-unify-stability-and-group-velocity.md` (prior art for layered feasible-then-minimize objective patterns)

**Test commands:**

```bash
# Fastest: the three new analytical modules in isolation
cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py tests/test_non_normality.py tests/test_brady_livescu_2d.py -x -q

# Full brady2d suite (includes calibration smoke)
cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q

# Regression suite (must still pass)
cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegression"

# CLI smoke: score a known-stable scheme at max_layer=6 (seconds)
cd scripts/stencil_gen && uv run python -m stencil_gen.brady2d --scheme E4 --kernel tension --sigma 3.0 --max-layer 6
```

---

## Items

### 41.1 — Brady-Livescu reference problem module

- [x] **41.1a** Create `stencil_gen/benchmarks/__init__.py` (empty) and `stencil_gen/benchmarks/brady_livescu_2d.py` containing pure-data functions and constants:
  - `L_DOMAIN = math.sqrt(2.0)`, `PSI_OFFSET = 0.25` as module-level constants.
  - `psi(x, y) -> float`, `c_x(x, y) -> float`, `c_y(x, y) -> float` — exact formulae as vectorized numpy functions.
  - `exact_solution(x, y, t) -> float` — returns `sin(2*pi*(psi(x,y) - t))`.
  - `initial_condition(x, y) -> float` — `exact_solution(x, y, 0.0)`.
  - `inflow_bc_x(y, t)`, `inflow_bc_y(x, t)` — the two Dirichlet boundary functions.
  - `make_coefficient_field(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` — returns `(x, y, c_x_field, c_y_field)` on a uniform N×N grid covering `[0, L_DOMAIN]^2` excluding the boundary rows at indices `(i==0)` and `(j==0)` to mirror the homogeneous stability problem.
  - Module docstring cites Brady & Livescu 2019 §4.3 and page numbers.
  - File: `scripts/stencil_gen/stencil_gen/benchmarks/brady_livescu_2d.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_benchmarks.py -x -q -k "TestBradyLivescu2D and test_constants"`

- [x] **41.1b** Add `tests/test_benchmarks.py::TestBradyLivescu2D` with:
  - `test_constants` — sanity checks: `L_DOMAIN == sqrt(2)`, `PSI_OFFSET == 0.25`.
  - `test_exact_solution_initial_time` — `exact_solution(x, y, 0) == initial_condition(x, y)` at 5 sample points.
  - `test_exact_solution_satisfies_pde` — for 5 random `(x, y, t)` points, central-difference approximations of `u_t + c_x*u_x + c_y*u_y` agree with zero to `1e-6`.
  - `test_coefficient_field_shape` — `make_coefficient_field(31)` returns arrays of shape `(31, 31)`, and `c_x**2 + c_y**2` is approximately 1 everywhere (unit radial vector).
  - `test_inflow_bc_matches_exact_at_edges` — `inflow_bc_x(y, t) == exact_solution(0, y, t)` at sample points.
  - File: `scripts/stencil_gen/tests/test_benchmarks.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_benchmarks.py -x -q -k "TestBradyLivescu2D"`

### 41.2 — L1 wrapper: interior + boundary group velocity error

- [x] **41.2a** Create `stencil_gen/brady2d_stability.py` with a **minimal** skeleton:
  - `StabilityReport` dataclass with *only* `layer1: dict | None = None`, `failed_layer: int | None = None`, `overall_verdict: str = "unknown"`, `compute_time: float = 0.0`. **Full fields (layer2 through layer8, kreiss, non_normality) are intentionally deferred to 41.10a — do not add them here.** A short module comment should note "dataclass will be expanded in 41.10a".
  - `layer1_interior_boundary_gv(scheme: str, kernel: str, params: dict, n_xi: int = 200) -> dict` — returns `{interior_gv_err_x: float, interior_gv_err_y: float, boundary_gv_err: float, cutoff_fraction: float}`. Uses existing `interior_group_velocity` and `boundary_group_velocity` from `group_velocity.py`. Wraps `boundary_group_velocity_classical` for classical-alpha family.
  - Layer-1 failure criterion: any metric > `L1_TOL = 0.05` (dispersion error > 5%).
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer1"`

- [x] **41.2b** Add `tests/test_brady2d_stability.py::TestLayer1` with:
  - `test_layer1_classical_e4_passes` — classical E4 alpha from `known_values.json` produces `boundary_gv_err < 0.05`.
  - `test_layer1_tension_e4_passes` — tension E4 at sigma=3.0 passes.
  - `test_layer1_gaussian_e4_known_unstable_still_passes_at_this_layer` — Gaussian ε=0.1 (known_unstable per `known_values.json`) passes L1 (confirms L1 is necessary but not sufficient).
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer1"`

### 41.3 — L2 rigorous GKS Kreiss determinant test (`gks_kreiss.py`)

Implementing Trefethen 1983 (pp. 206–207). For the semi-discrete problem `u_t = -Σ a_k u_{n+k}` a mode `u_n(t) = e^(s t) κ^n` requires `s + Σ a_k κ^k = 0`. For each `s` in the right half plane, keep the `r` admissible roots (`|κ| < 1`), assemble the `r × r` Kreiss matrix `M(s)[i, ℓ] = s·κ_ℓ^i + Σ_j w_{ij}·κ_ℓ^j`, and check `σ_min(M(s))`. Failure = a candidate `s` with `Re(s) ≥ 0` and `σ_min(M(s)) < tol`, with imaginary-axis perturbation confirming the violation.

- [x] **41.3a** Create `stencil_gen/gks_kreiss.py` skeleton:
  - Module docstring citing Trefethen 1983 pp. 206–207.
  - `BoundaryRow = tuple[np.ndarray, np.ndarray]` (weights, column offsets).
  - `DefectiveKappaError(RuntimeError)`.
  - `@dataclass(frozen=True) class KreissResult` — fields per agent draft: `is_stable, witness_s, witness_sigma_min, imaginary_axis_perturbation_verdict, defective_kappa_detected, s_grid_shape, compute_time, sigma_min_field, s_grid, n_admissible_roots`.
  - All function bodies raise `NotImplementedError`.
  - `logger = logging.getLogger("stencil_gen.gks_kreiss")`.
  - File: `scripts/stencil_gen/stencil_gen/gks_kreiss.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.gks_kreiss import KreissResult, DefectiveKappaError; print('ok')"`

- [x] **41.3b** Implement `kappa_roots(interior_weights, interior_offsets, s, *, repeat_tol=1e-7) -> tuple[np.ndarray, np.ndarray, bool]`:
  - Shift offsets to start at zero: `shifted = offsets - min(offsets)`, `L_left = -min(offsets)`.
  - Build polynomial coefficients `Q` of degree `L_left + R_right` such that `Q(κ) = Σ w_k κ^{k+L_left} + s·κ^{L_left}` — i.e., multiply the interior stencil symbol by `κ^L_left` to clear negative powers, add the `s` term.
  - Call `numpy.roots(Q)` to get all roots.
  - `admissible = roots[np.abs(roots) < 1.0 - 1e-12]`.
  - Defective check: if any pair of admissible roots has separation `< repeat_tol`, set `is_defective = True`.
  - Return `(all_roots, admissible, is_defective)`.
  - Add test class `TestKappaRoots` with: (i) **first-order upwind**: `interior_weights=np.array([-1.0, 1.0])`, `interior_offsets=np.array([-1, 0])`, at `s=1.0` returns all roots + exactly one admissible root with `abs(κ) < 1`; the admissible root should be the real number near `1/2` since the polynomial reduces to `-1 + κ + 1·κ = -1 + 2κ` → `κ = 1/2`; (ii) **deliberately-coalescing case**: hand-construct a small polynomial with a known double root at `κ=0.5` (e.g., using `np.poly([0.5, 0.5, 2.0])` as the `Q` directly, bypassing the stencil construction — test this via a helper `_kappa_roots_from_poly` if needed) and verify `is_defective=True`.
  - File: `scripts/stencil_gen/stencil_gen/gks_kreiss.py`, `scripts/stencil_gen/tests/test_gks_kreiss.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py -x -q -k "TestKappaRoots"`

- [x] **41.3c** Implement `kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s) -> np.ndarray` and `min_singular_value(...) -> float`:
  - `kreiss_matrix` calls `kappa_roots`, validates `len(admissible) == len(boundary_rows)` (raise `ValueError` otherwise), builds `M[i, ℓ] = s*κ_ℓ^i + Σ_j w_{ij}*κ_ℓ^j`.
  - `min_singular_value` returns `float(np.linalg.svd(M, compute_uv=False)[-1])`; on defective or shape error returns `np.inf`.
  - Tests: for an explicit small (2×2) case with hand-computed `M(s=1)`, verify exact match to `1e-12`.
  - File: `scripts/stencil_gen/stencil_gen/gks_kreiss.py`, `scripts/stencil_gen/tests/test_gks_kreiss.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py -x -q -k "TestKreissMatrix"`

- [x] **41.3c-followup** Fix empty `test_defective_raises` in `TestKreissMatrix` (test body is `pass`, tests nothing):
  - Replace the empty body with an actual test that engineers a stencil producing defective admissible roots and asserts `kreiss_matrix` raises `DefectiveKappaError`. Approach: reverse-engineer weights from a polynomial with a known double root inside the unit disk (e.g., `(κ-0.3)²(κ-5)` ↔ a 3-point stencil with offsets `[-1, 0, 1]`; solve for `weights` and `s` that produce this `Q`).
  - Add `test_min_singular_value_defective_returns_inf` verifying that `min_singular_value` returns `np.inf` on the `DefectiveKappaError` path (currently only the `ValueError`/shape-mismatch path is tested).
  - File: `scripts/stencil_gen/tests/test_gks_kreiss.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py -x -q -k "test_defective_raises or test_min_singular_value_defective_returns_inf"`

- [x] **41.3d** Implement `make_s_grid(s_max=10.0, n_radial=40, n_imag=120, imag_max=20.0, eps_imag=1e-6) -> np.ndarray` and `_sweep_grid(...)`:
  - L-shaped contour: half-disk `Re(s) ∈ logspace(-4, log10(s_max), n_radial) × Im(s) ∈ linspace(-imag_max, imag_max, n_imag)` plus dense imaginary-axis strip at `Re(s) = eps_imag`.
  - `_sweep_grid` evaluates `min_singular_value` at every grid point, returns `(sigma_field, argmin_idx)`.
  - Tests: grid shape matches expected, `_sweep_grid` on a known-stable scheme has `min(sigma_field) > 0.01`.
  - File: `scripts/stencil_gen/stencil_gen/gks_kreiss.py`, `scripts/stencil_gen/tests/test_gks_kreiss.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py -x -q -k "TestSGridSweep"`

- [x] **41.3e** Implement `_refine_witness(interior_weights, interior_offsets, boundary_rows, s_start) -> complex`:
  - `scipy.optimize.minimize` with `method="Nelder-Mead"` on `lambda re_im: log(min_singular_value(..., re_im[0] + 1j*re_im[1]) + 1e-300)`.
  - Reflect `Re(s) < 0` back into the right half-plane to enforce the constraint.
  - Return the complex `s` at convergence.
  - Test: perturbed witness `s_start = 1.5 + 0.5j` on a known-unstable scheme converges to a local min with `σ_min < 1e-6`.
  - File: `scripts/stencil_gen/stencil_gen/gks_kreiss.py`, `scripts/stencil_gen/tests/test_gks_kreiss.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py -x -q -k "TestRefineWitness"`

- [x] **41.3f** Implement `_classify_imag_axis(interior_weights, interior_offsets, s_candidate, delta=1e-4) -> str`:
  - Returns one of `{"no_candidates", "all_incoming", "outgoing_mode_detected", "defective"}`.
  - For a candidate `s_0` near the imaginary axis, recompute κ roots at `s_0` and at `s_0 + delta`, match by nearest-neighbor, classify unit-modulus κ's as `incoming` (moved inside the unit disk → genuine instability) or `outgoing` (moved outside → tangent-to-axis non-violation per Trefethen p. 207).
  - Tests: two explicit synthetic cases built as direct polynomial constructions (bypass stencils):
    - **case A (inward perturbation → violation):** start from `Q₀(κ) = (κ - e^(iφ))·(κ - 2)` with `φ = π/4`, apply a perturbation that pulls the unit-modulus root inward — specifically replace `Q₀` with `Q_δ(κ) = (κ - (1 - δ)·e^(iφ))·(κ - 2)` for `δ = 1e-3`. Use the helper from 41.3b to call `kappa_roots` on both `Q₀` and `Q_δ`. Assert `_classify_imag_axis` returns `"outgoing_mode_detected"`.
    - **case B (outward perturbation → non-violation):** same structure but replace with `Q_δ(κ) = (κ - (1 + δ)·e^(iφ))·(κ - 2)`. Assert returns `"all_incoming"`.
  - Note: the naming is inverted from intuitive because "outgoing_mode_detected" means "we detected a physical outgoing mode that should not exist at the boundary" = failure. Document this clearly in the docstring.
  - File: `scripts/stencil_gen/stencil_gen/gks_kreiss.py`, `scripts/stencil_gen/tests/test_gks_kreiss.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py -x -q -k "TestClassifyImagAxis"`

- [x] **41.3g** Implement `kreiss_stability_check(interior_weights, interior_offsets, boundary_rows, *, s_grid_params=None, sigma_tol=1e-8, refine=True) -> KreissResult`:
  - Orchestrator: calls `_sweep_grid` → if `min(sigma_field) < sigma_tol*scale` → `_refine_witness` → `_classify_imag_axis` on refined `s` → assemble `KreissResult`.
  - Wraps the whole thing in `try/except DefectiveKappaError` and records `defective_kappa_detected=True`.
  - `compute_time` via `time.perf_counter`.
  - File: `scripts/stencil_gen/stencil_gen/gks_kreiss.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py -x -q -k "test_kreiss_stability_check_runs"`

- [x] **41.3h** Integration tests `TestKreissIntegration`:
  - `test_classical_e4_passes` — E4 tension σ=3.0 (known stable RBF), run `kreiss_stability_check`, assert `is_stable=True`.
  - `test_gaussian_known_unstable_passes_gks` — E4 Gaussian ε=0.1 is GKS-stable (boundary closure is fine); its instability is an eigenvalue instability caught at Layer 3, not a boundary-closure violation. Test asserts `is_stable=True` for GKS and `stability_eigenvalue_from_matrix > 0` to confirm the eigenvalue instability exists.
  - `test_consistency_with_heuristic` — for E4 Gaussian ε=0.1, verifies that the GKS heuristic's outgoing mode has negative real part (damped, not a GKS violation). Documents that the heuristic and Kreiss test are complementary diagnostics testing different aspects.
  - `test_s_equals_zero_godunov_ryabenkii_reduction` — at `s=0`, `min_singular_value` returns `inf` (shape mismatch: 1 admissible root vs 2 boundary rows), confirming the Godunov-Ryabenkii condition.
  - **Implementation note:** The orchestrator uses `min_s_magnitude=0.1` (default) to exclude the trivial zero at `s=0` that exists for all consistent first-derivative operators (constant mode `κ=1` always satisfies both interior and boundary equations with derivative=0). Without this exclusion, Nelder-Mead refinement always converges to the trivial zero, producing false positives.
  - Runtime: ~2 s for the whole class.
  - File: `scripts/stencil_gen/tests/test_gks_kreiss.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py -x -q -k "TestKreissIntegration"`

- [x] **41.3h-followup** Fix `_classify_imag_axis` defective-check ordering and strengthen `test_s_equals_zero_godunov_ryabenkii_reduction` assertion:
  - **Defective-check ordering:** In `_classify_imag_axis` (gks_kreiss.py, line 383), `is_defective_0` fires before the near-unit-circle filter (line 387). The defective flag from `kappa_roots` checks all *admissible* roots (`|κ| < 1 - 1e-12`), not just near-unit-circle ones. If two admissible roots deep inside the unit disk (e.g., both at `|κ| ≈ 0.3`) are nearly equal, the function returns `"defective"` even though the near-unit-circle roots relevant to the classification are fine. In the orchestrator (line 548), `"defective" != "all_incoming"` → `is_stable = False` — a false-unstable verdict. Fix: move the defective guard after the `near_unit` filter and restrict it to check only whether the near-unit roots themselves are defective. Also update `test_defective_roots_returns_defective` which currently validates the wrong behavior (returns `"defective"` for roots at `|κ| = 0.3`, far from unit circle; should return `"no_candidates"` after the fix).
  - **Weak test assertion:** `test_s_equals_zero_godunov_ryabenkii_reduction` asserts `sv == np.inf or sv > 0`, which is a tautology for any non-NaN non-zero value. The docstring says the expected result is `np.inf` (shape mismatch: 1 admissible root vs 2 boundary rows). Strengthen to `assert sv == np.inf`.
  - File: `scripts/stencil_gen/stencil_gen/gks_kreiss.py`, `scripts/stencil_gen/tests/test_gks_kreiss.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py -x -q -k "TestClassifyImagAxis or test_s_equals_zero"`

- [x] **41.3h-followup2** Add test for near-unit-circle defective path in `_classify_imag_axis`:
  - The `return "defective"` at gks_kreiss.py line 396 (the near-unit defective check added in 41.3h-followup) has zero test coverage. `test_defective_roots_far_from_unit_circle_returns_no_candidates` only tests the non-triggering case. Add `test_defective_roots_near_unit_circle_returns_defective` that engineers a stencil with a double root near `|κ| = 1` (e.g., `(κ - 0.9999)²(κ - 5)` → reverse-engineer weights from `np.poly([0.9999, 0.9999, 5.0])`) and asserts `_classify_imag_axis` returns `"defective"`.
  - File: `scripts/stencil_gen/tests/test_gks_kreiss.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_gks_kreiss.py -x -q -k "test_defective_roots_near_unit_circle"`

### 41.4 — Generalize `gks_group_velocity_check` for all four boundary sides

- [x] **41.4a** Add `side: Literal["left", "right", "bottom", "top"] = "left"` parameter to `gks_group_velocity_check` at `group_velocity.py:964`:
  - For `side="left"` keep existing behavior (drop row/col 0).
  - For `side="right"` drop row/col `-1` and flip the group-velocity sign convention.
  - For `side="bottom"` and `side="top"` raise `NotImplementedError` with message about requiring 2D D; deferred to phase 41.6.
  - Update docstring.
  - Backwards-compatible: existing callers pass no `side`, get `"left"`.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestGKS"`

- [x] **41.4b** Add `TestGKSSideParameter` tests:
  - `test_left_default_unchanged` — verify existing `side="left"` results match pre-41.4a snapshot.
  - `test_right_mirrors_left` — apply `side="right"` to a right-boundary-oriented differentiation matrix, verify the result mirrors the `side="left"` result on the spatially-reflected matrix.
  - `test_bottom_raises` — `side="bottom"` raises `NotImplementedError`.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestGKSSideParameter"`

### 41.5 — L3 wrapper: 1D eigenvalue check at multiple grid sizes

- [x] **41.5a** Add `layer3_1d_eigenvalue(scheme, kernel, params, n_values=(20, 40, 80)) -> dict` to `brady2d_stability.py`:
  - For each `n`, call `stability_eigenvalue(n, p, q, epsilon=params["sigma"], kernel=kernel, nu=1, nextra=0)` (or classical-alpha variant via `stability_eigenvalue_from_matrix` when `kernel=="classical"`).
  - Return `{n: stab_eig for n in n_values}` plus `max_stab_eig = max(values)`.
  - Layer-3 failure: `max_stab_eig > STABILITY_TOL = 1e-10`.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer3"`

- [x] **41.5b** Tests `TestLayer3`:
  - `test_classical_e4_stable` at all three n.
  - `test_tension_e4_sigma_3_stable` at all three n.
  - `test_gaussian_e4_eps_01_unstable` — explicit fail case matches `known_unstable` entry.
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer3"`

- [x] **41.5c** Fix `_build_classical_diff_matrix` crash for E2 (p=1) and remove E2 classical from 41.11a FAMILIES:
  - **Problem:** `_build_classical_diff_matrix` calls `_derive_classical_boundary(p=1, ...)` → `derive_boundary(p=1)` which computes `n_alpha = (r-2) + n_active_penultimate = -1`, then crashes in `symbols("alpha_0:-1")`. E2 has no free alpha parameters — the boundary weights are fully determined.
  - **Fix option A (preferred):** Remove the `("E2", "classical", ...)` entry from 41.11a's `FAMILIES` list. E2 classical has no free parameters, so there is no classical-alpha family to calibrate. The E2 PHS k=2 entry (`("E2", "tension", {"sigma": 0.0})`) already covers the E2 case with fully-determined boundary weights. Add a comment in `_build_classical_diff_matrix` docstring noting it requires `p >= 2` (E4+).
  - **Fix option B (if E2 classical calibration is actually needed):** Handle the no-alpha case in `_derive_classical_boundary` by bypassing `derive_boundary` and directly constructing the unique E2 boundary row.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`, `plans/41-brady-livescu-2d-analytical-stability.md`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer3"`

### 41.6 — L4 2D varying-coefficient local group velocity

- [x] **41.6a** Add `local_group_velocity_2d_varying(interior_stencil_x, interior_stencil_y, c_x_field, c_y_field, xi_array) -> dict` to `group_velocity.py`:
  - For each grid point `(i, j)`, compute the *local* group velocity error by freezing coefficients: `C_local_x(xi) = c_x[i,j] * group_velocity_exact(interior_stencil_x, xi) - c_x[i,j]`.
  - Similarly for `C_local_y`.
  - Return `{C_x_field: shape (Ny, Nx, N_xi), C_y_field: shape (Ny, Nx, N_xi), gv_error_x_field, gv_error_y_field}`.
  - Scalar reduction helper `max_local_gv_error_2d(result) -> float`.
  - The varying-coefficient dispersion error at a point is `c_*[i,j] * gv_error(xi)` — factor of `c_*` scales the error.
  - Docstring: cites the fact that for smooth `(c_x, c_y)` fields the local-frozen-coefficient analysis is the first-order WKB approximation.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestLocal2DVarying"`

- [x] **41.6b** Tests `TestLocal2DVarying`:
  - `test_constant_coefficient_reduces_to_interior` — set `c_x == 1`, `c_y == 0` everywhere, verify output matches `interior_group_velocity` applied to the x-stencil.
  - `test_radial_flow_field` — use `make_coefficient_field(31)` from 41.1, verify `max_local_gv_error_2d` finite and within expected range for classical E4.
  - `test_scalar_reduction_finite_for_both_schemes` — E2 and E4 both produce finite positive `max_local_gv_error_2d` on the BL field (property assertion only; no monotonicity claim between schemes since that is not a theorem).
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestLocal2DVarying"`

- [x] **41.6c** Add `layer4_local_gv_2d(scheme, kernel, params, N=31) -> dict` to `brady2d_stability.py`:
  - Builds `(c_x_field, c_y_field)` from `make_coefficient_field(N)`.
  - Calls `local_group_velocity_2d_varying` with the scheme's interior stencils.
  - Returns `{max_local_gv_error: float, worst_point: tuple[int, int], worst_xi: float}`.
  - Failure threshold: `max_local_gv_error > 0.1` (10%, looser than L1 because varying-coefficient scaling amplifies the baseline error).
  - Tests in `TestLayer4`: classical E4 passes, deliberately large-error scheme fails.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`, `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer4"`

- [x] **41.6c-followup** Add missing L4 failure test in `TestLayer4`:
  - The plan specifies "deliberately large-error scheme fails" but no failure test was implemented. All real schemes pass easily within the resolved band (E2: 0.012, E4: 0.0002 vs `L4_TOL=0.1`), so a synthetic bad stencil is needed. Approach: construct a stencil with deliberately poor group velocity (e.g., weights `[0.5, -0.5]` with offsets `[-1, 0]` which has `C(xi) = cos(xi/2)`, giving GV error up to 100% near `xi = pi`) and call `local_group_velocity_2d_varying` directly with `c_x == 1` everywhere so that `max_local_gv_error > L4_TOL`. Wrap in a test `test_synthetic_bad_stencil_fails` that asserts the max error exceeds `L4_TOL`.
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "test_synthetic_bad_stencil_fails"`

### 41.7 — L5 2D anisotropy over the coefficient field

- [x] **41.7a** Add `anisotropy_over_coefficient_field(scheme, c_x_field, c_y_field, theta_array, xi_mag) -> dict` to `group_velocity.py`:
  - Builds `anisotropy_profile(p, nu=1, theta_array, xi_mag)` once (the raw anisotropy is scheme-property, field-independent).
  - Then evaluates the directional alignment: at each grid point, the radial propagation direction is `(c_x[i,j], c_y[i,j])/|c|` — project the anisotropy error onto this direction.
  - Returns `{max_aligned_error: float, worst_point: tuple[int, int], worst_theta: float}`.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestAnisotropyOverField"`

- [x] **41.7b** Add `layer5_anisotropy(scheme, kernel, params, N=31) -> dict` to `brady2d_stability.py`:
  - Wraps `anisotropy_over_coefficient_field`. Failure threshold: `max_aligned_error > 0.05`.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer5"`

- [x] **41.7b-followup** Add missing L5 failure test in `TestLayer5`:
  - No test exercises the `L5_TOL = 0.05` threshold from the failure side. E2 is at 0.048 — just 4% below the threshold — so no real scheme naturally fails. This mirrors the gap caught and fixed for L4 in 41.6c-followup. Approach: construct a synthetic scheme with large anisotropy error by calling `anisotropy_over_coefficient_field` with a scheme string that maps to a low-order `p` value (E2, p=1), and use a larger `xi_mag` (e.g., 80% of cutoff instead of 20%) so the anisotropy error exceeds `L5_TOL`. Alternatively, directly call `anisotropy_over_coefficient_field` with a fabricated anisotropy profile that has known large error. Wrap in a test `test_large_xi_mag_exceeds_threshold` that asserts `max_aligned_error > L5_TOL`.
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "test_large_xi_mag_exceeds_threshold"`

### 41.8 — L6 non-normality diagnostics module (`non_normality.py`)

- [x] **41.8a** Create `stencil_gen/non_normality.py` skeleton:
  - Module docstring citing Trefethen & Embree, *Spectra and Pseudospectra* (2005), ch. 14 for calibration bands.
  - `@dataclass(frozen=True) class NonNormalityReport` — fields: `spectral_abscissa, numerical_abscissa, henrici_departure, eigenvector_condition, pseudospectral_abscissae: dict[float, float], kreiss_constant, transient_growth_bound, n, compute_time, notes: list[str]`.
  - All function stubs raise `NotImplementedError`.
  - File: `scripts/stencil_gen/stencil_gen/non_normality.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.non_normality import NonNormalityReport; print('ok')"`

- [x] **41.8b** Implement `spectral_abscissa_sparse(L, k=20, shift_invert=True)`:
  - Primary: `scipy.sparse.linalg.eigs(L, k=k, which="LR")`.
  - On `ArpackNoConvergence`: retry with `sigma=0.0, which="LR"` (shift-invert).
  - Fallback: if `L.shape[0] <= 900` densify and use `np.linalg.eigvals`.
  - Return `(max_real_part, all_computed_eigenvalues)`.
  - Tests `TestSpectralAbscissa`: diagonal `-diag(1..50)` returns `≈ -1`; random sparse returns finite; dense fallback path exercised at N=20.
  - File: `scripts/stencil_gen/stencil_gen/non_normality.py`, `scripts/stencil_gen/tests/test_non_normality.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_non_normality.py -x -q -k "TestSpectralAbscissa"`

- [x] **41.8c** Implement `numerical_abscissa_sparse(L) -> float`, `henrici_departure(L) -> float`, `eigenvector_condition(L, small_dense_threshold=900) -> float`:
  - `numerical_abscissa_sparse` — `H = 0.5 * (L + L.T)`; `eigsh(H, k=1, which="LA")`; sparse → scalar.
  - `henrici_departure` — `‖L L^T - L^T L‖_F / ‖L‖_F^2`; guard division by zero.
  - `eigenvector_condition` — if `N > threshold` return `np.nan`; else dense `np.linalg.eig` and `np.linalg.cond(V)`.
  - Tests in `TestNormMetrics`: on diagonal L, all three return the normal-operator values (≈ −1, ≈ 0, ≈ 1).
  - File: `scripts/stencil_gen/stencil_gen/non_normality.py`, `scripts/stencil_gen/tests/test_non_normality.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_non_normality.py -x -q -k "TestNormMetrics"`

- [x] **41.8d** Implement `_sigma_field(L, s_grid) -> np.ndarray` shared helper:
  - For each `s` in `s_grid`, compute `σ_min(s*I - L)` via `scipy.sparse.linalg.svds(s*sp.eye(L.shape[0]) - L, k=1, which="SM", return_singular_vectors=False)` (or dense SVD for small N).
  - Handle `ArpackError` by densifying the shifted operator for small N.
  - Return `sigma_field` of shape `s_grid.shape`.
  - Tests: compare to brute-force dense SVD on small N.
  - File: `scripts/stencil_gen/stencil_gen/non_normality.py`, `scripts/stencil_gen/tests/test_non_normality.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_non_normality.py -x -q -k "TestSigmaField"`

- [x] **41.8d-followup** Fix `_sigma_field` NameError for dense input with `n > 900`:
  - **Bug:** When `L` is a dense ndarray and `n > 900`, `use_dense` is `False` so the else branch (line 300) runs and references `I_sp`/`L_sp` which were never defined (only set in the `if sp.issparse(L)` block at line 282). This crashes with `NameError: cannot access local variable 'I_sp'`.
  - **Fix:** At the top of the else branch (sparse-svds path), convert `L` to sparse if it isn't already, e.g. `if not sp.issparse(L): L_sp = sp.csc_matrix(L); I_sp = sp.eye(n, format="csc")`. Alternatively, move the `I_sp`/`L_sp` setup outside the `if sp.issparse` guard so it always runs when `use_dense` is `False`.
  - **Test:** Add `test_dense_large_input` in `TestSigmaField` that passes a dense 901×901 diagonal matrix and verifies correctness (no crash, result matches `np.linalg.svd`).
  - File: `scripts/stencil_gen/stencil_gen/non_normality.py`, `scripts/stencil_gen/tests/test_non_normality.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_non_normality.py -x -q -k "test_dense_large_input"`

- [x] **41.8e** Implement `pseudospectral_abscissa_estimate(L, epsilon_values, s_grid) -> dict[float, float]` and `kreiss_constant_estimate(L, s_grid) -> float`:
  - Both reuse `_sigma_field` to avoid duplicate SVD cost.
  - `pseudospectral_abscissa_estimate`: for each ε, `α_ε = max{ Re(s) : s in s_grid, sigma_field[s] <= ε }`, or `-inf` if no point satisfies.
  - `kreiss_constant_estimate`: `max{ Re(s) / sigma_field[s] : Re(s) > 0 }`.
  - Tests on Wilkinson bidiagonal `-I + N*upper_shift` at N=30: `spectral_abscissa ≈ -1`, `numerical_abscissa` strongly positive, `kreiss_constant >> 1`.
  - File: `scripts/stencil_gen/stencil_gen/non_normality.py`, `scripts/stencil_gen/tests/test_non_normality.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_non_normality.py -x -q -k "TestPseudoAndKreiss"`

- [x] **41.8f** Implement `compute_non_normality(L, *, small_dense_threshold=900, epsilon_values=(1e-4, 1e-3, 1e-2, 1e-1), s_grid_params=None) -> NonNormalityReport` orchestrator:
  - Uses `time.perf_counter`.
  - Default `s_grid`: rectangular right-half-plane grid in `Re(s) ∈ [1e-3, 2*|alpha|+1]`, `Im(s) ∈ [-ω_max, ω_max]`, shape ~30×60 for small matrices, ~8×12 for n>500 (dense SVD performance).
  - Computes `transient_growth_bound = math.e * kreiss_constant`.
  - Appends diagnostic notes for any ArpackNoConvergence.
  - Cross-check: `numerical_abscissa >= spectral_abscissa - 1e-9` — assert in tests.
  - Test: `test_compute_non_normality_on_bl_sized_matrix` — build a BL-sized 2D test matrix via `kron(D, I) + kron(I, D)` for `D = build_diff_matrix_rbf(n=31, ...)`, call `compute_non_normality`, assert `compute_time < 30.0`, all fields finite/NaN.
  - Mark the BL-sized test `@pytest.mark.slow`.
  - Also raised `_sigma_field` dense SVD threshold from n≤200 to n≤1200 and fallback threshold from n≤900 to n≤2000, because sparse `svds(which='SM')` is unreliable for non-symmetric operators (ARPACK failure costs ~5s/point at n≈1000 vs 0.2s for dense SVD).
  - File: `scripts/stencil_gen/stencil_gen/non_normality.py`, `scripts/stencil_gen/tests/test_non_normality.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_non_normality.py -x -q -k "TestComputeNonNormality" -m "not slow"`

### 41.9 — L7 sparse 2D eigenvalue of the full varying-coefficient BL operator

- [x] **41.9a** Add `build_sparse_2d_operator(scheme, kernel, params, N) -> tuple[scipy.sparse.csr_matrix, np.ndarray]` to `brady2d_stability.py`:
  - Build `D_x_1D` via `phs.build_diff_matrix_rbf(...)` for the chosen family (1D, shape N×N, sparse via `scipy.sparse.csr_matrix`).
  - `Ix, Iy = sp.eye(N)`, `Dx_2D = sp.kron(Iy, Dx1)`, `Dy_2D = sp.kron(Dy1, Ix)`.
  - `c_x_vec, c_y_vec` from `brady_livescu_2d.make_coefficient_field(N)` flattened row-major (`u[j*N + i]` with `i=x, j=y`).
  - `Cx = sp.diags(c_x_vec)`, `Cy = sp.diags(c_y_vec)`.
  - `L_2D = -(Cx @ Dx_2D + Cy @ Dy_2D)`.
  - Inflow removal: `keep_mask = (ii > 0) & (jj > 0)` in row-major, `L_red = L_2D[keep_idx, :][:, keep_idx]`.
  - Return `(L_red.tocsr(), keep_idx)`.
  - Test: at N=11, output shape is `(10*10, 10*10) = (100, 100)`; at N=21, shape `(400, 400)`.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestBuildSparse2D"`

- [x] **41.9b** Add `layer7_sparse_2d_eigenvalue(scheme, kernel, params, n_values=(21, 31, 61)) -> dict` to `brady2d_stability.py`:
  - For each n, build `L_red` and call `spectral_abscissa_sparse(L_red, k=20)`.
  - Return `{n: max_re for n in n_values}` plus `max_spectral_abscissa = max(values)`.
  - Failure: `max_spectral_abscissa > L7_TOL = 5e-3`. **Threshold revised from 1e-8:** The 2D varying-coefficient BL operator is not skew-symmetric — stable schemes (classical E4, tension E4 σ=3.0, E2 PHS) exhibit max Re(λ) up to ~O(1e-3) due to boundary/varying-coefficient interaction, while the known-unstable Gaussian ε=0.1 has max Re ~ 0.148. The 5e-3 threshold cleanly separates these regimes.
  - Tests: `TestLayer7` with 5 tests — tension E4 stable, classical E4 stable, Gaussian unstable, return-key checks, custom n_values.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer7"`

- [x] **41.9c** Add `layer7_with_non_normality(scheme, kernel, params, N=31) -> NonNormalityReport` wrapper:
  - Builds `L_red` at a single modest N and calls `compute_non_normality(L_red)`.
  - This links L6 to the actual BL operator (L6 defines the infrastructure, this wires it to the BL coefficient field).
  - Failure: `spectral_abscissa > L7_TOL (5e-3)` OR `transient_growth_bound > 50.0`.
  - Tests: classical E4 passes; Gaussian ε=0.1 fails.
  - Mark `@pytest.mark.slow` (≈ 10 s at N=31).
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer7WithNonNormality"`

### 41.10 — Unified orchestrator and `StabilityReport`

- [x] **41.10a** Flesh out `StabilityReport` in `brady2d_stability.py`:
  - Fields: `layer1: dict | None`, …, `layer7: dict | None`, `non_normality: NonNormalityReport | None`, `kreiss: KreissResult | None`, `overall_verdict: Literal["pass", "fail"]`, `failed_layer: int | None`, `failed_reason: str`, `compute_time: float`.
  - Factory classmethod `empty() -> StabilityReport`.
  - `__str__` method that produces a per-layer summary table.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestStabilityReport"`

- [x] **41.10b** Implement `brady2d_stability_score(scheme, kernel, params, *, max_layer=7, short_circuit=True) -> StabilityReport`:
  - Runs layers 1 → `max_layer` in order.
  - Each layer populates the corresponding field in `StabilityReport`.
  - If `short_circuit and layer fails`, set `failed_layer` + `failed_reason` and return without running later layers.
  - `compute_time` is the total wall-clock.
  - Also added `layer2_kreiss_gks(scheme, kernel, params, n=20)` wrapper and `_extract_stencil_data(D, p)` helper to bridge the scheme/kernel API to the raw-stencil Kreiss check.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestStabilityScoreOrchestrator"`

- [x] **41.10b-followup** Wire Layer 6 into the orchestrator so `max_layer=6` runs non-normality diagnostics:
  - **Problem:** The orchestrator jumps from L5 (`if max_layer >= 5`) directly to L7 (`if max_layer >= 7`). There is no `if max_layer >= 6` block. Consequently `max_layer=6` produces identical results to `max_layer=5`, the `layer6` field in `StabilityReport` is always `None`, and the `__str__` method labels non-normality as "L6" but it only appears when `max_layer >= 7`. The plan's pipeline table, CLI test commands (`--max-layer 6`), and completion criteria all expect L6 to be a distinct, cheaper-than-L7 non-normality layer.
  - **Fix:** Add `layer6_non_normality(scheme, kernel, params, n=80) -> dict` that runs `compute_non_normality` on the **1D** differentiation matrix (not the full 2D BL operator — that's L7's job). This matches the pipeline table's L6 cost (~seconds) vs L7 (~few seconds per N). Wire it into the orchestrator under `if max_layer >= 6:`, populating `report.layer6` and checking `spectral_abscissa > STABILITY_TOL` and `transient_growth_bound > L6_TRANSIENT_GROWTH_TOL` (define an appropriate threshold). Also populate `report.non_normality` from the 1D result at L6, upgrading it to the 2D result at L7 if L7 runs.
  - **Tests:** Add `test_max_layer_6_runs_non_normality` in `TestStabilityScoreOrchestrator` asserting `report.layer6 is not None` and `report.non_normality is not None` when `max_layer=6`. Add `test_max_layer_6_differs_from_5` asserting the two produce different populated fields.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`, `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "test_max_layer_6"`

- [x] **41.10c** Integration tests `TestBrady2DScoreIntegration`:
  - `test_classical_e4_passes_all_layers_1_through_7` — overall `pass`, `failed_layer is None`.
  - `test_gaussian_eps_01_fails_at_layer_2_or_3` — overall `fail`, `failed_layer in (2, 3)`.
  - `test_short_circuit_false_runs_all_layers` — with `short_circuit=False`, all layer fields populated even on failure.
  - Mark the full-pipeline tests `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestBrady2DScoreIntegration"`

- [x] **41.10d** Add CLI entry point `stencil_gen/brady2d_cli.py` with `main(argv) -> int`:
  - Args: `--scheme {E2,E4}`, `--kernel {classical,tension,gaussian,multiquadric,phs}`, numeric params (`--sigma`, `--epsilon`, or `--alpha`), `--max-layer int`, `--short-circuit/--no-short-circuit`, `--json-output PATH`.
  - Prints the `StabilityReport.__str__` summary; optionally dumps JSON.
  - Register as `python -m stencil_gen.brady2d` via a new `__main__` entry or add it to `sweeps/__main__.py` subparser table.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_cli.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -m stencil_gen.brady2d_cli --scheme E4 --kernel tension --sigma 3.0 --max-layer 3`

- [ ] **41.10d-followup** Register `python -m stencil_gen.brady2d` module entry point:
  - **Problem:** 41.10d created `brady2d_cli.py` with a working `if __name__ == "__main__"` block, but did not register the `python -m stencil_gen.brady2d` path that the plan's test commands section (line 62) and 41.10d itself both specify. Running `python -m stencil_gen.brady2d` fails with "No module named stencil_gen.brady2d". Only `python -m stencil_gen.brady2d_cli` works.
  - **Fix:** Create `scripts/stencil_gen/stencil_gen/brady2d/__init__.py` (empty) and `scripts/stencil_gen/stencil_gen/brady2d/__main__.py` containing `from stencil_gen.brady2d_cli import main; import sys; sys.exit(main())`. This lets `python -m stencil_gen.brady2d` delegate to the existing CLI.
  - File: `scripts/stencil_gen/stencil_gen/brady2d/__init__.py` (new), `scripts/stencil_gen/stencil_gen/brady2d/__main__.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -m stencil_gen.brady2d --scheme E4 --kernel tension --sigma 3.0 --max-layer 1`

### 41.11 — Calibration: score every spline/RBF family against the BL benchmark

The point of this phase is to run `brady2d_stability_score` on every (scheme, kernel, parameter) combination currently supported by `stencil_gen` and record the per-layer results in `known_values.json` under a new top-level key. Plan 42 uses these scores to prioritize which families to port to C++ first.

- [ ] **41.11a** Create `stencil_gen/benchmarks/brady2d_calibration.py` with:
  - `FAMILIES: list[tuple[str, str, dict]]` — the enumeration. **Note on PHS k=2:** `phs._rbf_weights_numeric` dispatches PHS via `kernel="tension"` with `epsilon=0.0` (see `phs.py:407–423`), not via a distinct `"phs_k2"` kernel string. The `FAMILIES` entries must use the actual kernel string accepted by the dispatcher:
    ```
    # NOTE: E2 classical removed per 41.5c — E2 has no free alpha parameters.
    ("E4", "classical", {"alpha": <loaded from alpha_extraction production values>}),
    ("E2", "tension",      {"sigma":   0.0}),  # ε=0 → dispatches to PHS k=2
    ("E4", "tension",      {"sigma":   0.0}),
    ("E2", "tension",      {"sigma":   6.0}),  # actual tension-spline at optimum
    ("E4", "tension",      {"sigma":   3.0}),
    ("E2", "gaussian",     {"epsilon": 2.0}),
    ("E4", "gaussian",     {"epsilon": 0.9}),
    ("E2", "multiquadric", {"epsilon": 1.0}),
    ("E4", "multiquadric", {"epsilon": 1.0}),
    ```
  - Use a display label (e.g. `"E2_phs_k2"`) in the output dict key when `sigma==0.0` so the stored result is distinguishable from the actual tension entries.
  - `run_calibration(max_layer=7) -> dict` that iterates `FAMILIES`, calls `brady2d_stability_score` for each, collects per-layer scalars.
  - Outputs a `pandas`-free dict (to stay out of hard dep on pandas): `{family_key: {"layer1": {...}, "layer2": {...}, ..., "overall_verdict": ..., "failed_layer": ...}}`.
  - File: `scripts/stencil_gen/stencil_gen/benchmarks/brady2d_calibration.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_benchmarks.py -x -q -k "TestCalibrationDataclass"`

- [ ] **41.11b** Add `--run-calibration` and `--update-known-values` flags to `brady2d_cli.py`:
  - `--run-calibration` runs `run_calibration(max_layer=6)` and prints a markdown table of results.
  - `--update-known-values` writes the full calibration dict to `known_values.json` under top-level key `"brady2d_calibration"` (additive; does not touch existing keys).
  - File: `scripts/stencil_gen/stencil_gen/brady2d_cli.py`
  - Test: `cd scripts/stencil_gen && uv run python -m stencil_gen.brady2d_cli --run-calibration --max-layer 3`

- [ ] **41.11c** Run the calibration at `max_layer=6` and commit the resulting `known_values.json`:
  - `cd scripts/stencil_gen && uv run python -m stencil_gen.brady2d_cli --run-calibration --max-layer 6 --update-known-values`
  - Review the output for any family that unexpectedly fails. Record surprising results in a brief note at the top of `brady2d_calibration.py`.
  - File: `scripts/stencil_gen/sweeps/known_values.json`
  - Test: `cd scripts/stencil_gen && uv run python -c "import json; d = json.load(open('sweeps/known_values.json')); assert 'brady2d_calibration' in d; print(list(d['brady2d_calibration'].keys()))"`

- [ ] **41.11d** Add `TestRegressionBrady2DCalibration` in `test_phs.py`:
  - Loads `brady2d_calibration` from `known_values.json`, iterates each family, re-runs `brady2d_stability_score` at `max_layer=3` (fast subset to keep tests under 10s), asserts the overall verdict matches the stored value.
  - Graceful skip if key is absent.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionBrady2DCalibration"`

- [ ] **41.11e** *(manual — explicitly OUT of the ralph_wiggum loop.)* Run the full calibration at `max_layer=7` (includes L7 sparse 2D eigs) to produce the authoritative reference scores for plan 42:
  - The invocation is: `cd scripts/stencil_gen && uv run python -m stencil_gen.brady2d_cli --run-calibration --max-layer 7 --update-known-values`
  - Expected runtime: 10–30 minutes total across all families (each family L7 scales as sparse-Arnoldi at multiple N).
  - **Ralph should not attempt this item.** The item is checked off manually by a human operator after running the command and committing the resulting `known_values.json`. If ralph encounters this item as the next unchecked item, it must leave the checkbox `[ ]`, skip to completion checks, and return `RALPH_STATUS: done` (not blocked). Add an explicit comment in the plan line above the item: `<!-- RALPH-SKIP: manual calibration run; do not attempt -->`.
  - File: `scripts/stencil_gen/sweeps/known_values.json` (manually committed)
  - Test: (post-calibration, manual) `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionBrady2DCalibration"`

### 41.12 — Documentation

- [ ] **41.12a** Create `scripts/stencil_gen/docs/brady2d_stability_reference.md`:
  - Problem statement from Brady-Livescu 2019 §4.3 (paraphrased, with page citation).
  - Layered pipeline table (L1–L7, what each catches, cost).
  - Failure threshold rationale for each layer.
  - API reference for `brady2d_stability_score` and `StabilityReport`.
  - Calibration band citations (Trefethen & Embree ch. 14 for Kreiss constants).
  - File: `scripts/stencil_gen/docs/brady2d_stability_reference.md` (new)
  - Test: (no test — doc only)

- [ ] **41.12b** Update `scripts/stencil_gen/docs/group_velocity_reference.md`:
  - Cross-reference `gks_kreiss.py` (L2) and `local_group_velocity_2d_varying` (L4) alongside existing sections.
  - Clarify that `gks_group_velocity_check` is the heuristic and `kreiss_stability_check` is the rigorous test, citing Trefethen 1983 pp. 206–207.
  - File: `scripts/stencil_gen/docs/group_velocity_reference.md`
  - Test: (no test)

- [ ] **41.12c** Update `.claude/skills/group-velocity-analysis/SKILL.md`:
  - Add one bullet about the Kreiss test and non-normality layer.
  - Add one bullet about the `brady2d_stability_score` entry point.
  - File: `.claude/skills/group-velocity-analysis/SKILL.md`
  - Test: (no test)

- [ ] **41.12d** Update the `stencil-sweeps` skill SKILL.md with the new `python -m stencil_gen.brady2d_cli` command:
  - One line under CLI quick reference.
  - File: `.claude/skills/stencil-sweeps/SKILL.md`
  - Test: (no test)

---

## Ordering

```
41.1a → 41.1b              # Reference problem first; everything depends on make_coefficient_field
  ↓
41.2a → 41.2b              # L1 wrapper (trivial, but first consumer of brady2d_stability.py)
  ↓
41.3a → 41.3b → 41.3c      # gks_kreiss module skeleton + primitives
  ↓    → 41.3d → 41.3e → 41.3f
  ↓    → 41.3g → 41.3h     # orchestrator + integration tests
  ↓
41.4a → 41.4b              # Independent of 41.3; can run in parallel after 41.1
  ↓
41.5a → 41.5b              # L3 wrapper (thin; uses existing stability_eigenvalue)
  ↓
41.6a → 41.6b → 41.6c      # L4 needs make_coefficient_field from 41.1a
  ↓
41.7a → 41.7b              # L5
  ↓
41.8a → 41.8b → 41.8c → 41.8d → 41.8e → 41.8f   # non_normality.py, independent strand
  ↓
41.9a → 41.9b → 41.9c      # L7 wires L8 to the BL operator; depends on 41.6 + 41.8
  ↓
41.10a → 41.10b → 41.10c → 41.10d    # Orchestrator after all layers exist
  ↓
41.11a → 41.11b → 41.11c → 41.11d    # Calibration depends on orchestrator; 41.11e is slow and manual
  ↓
41.12a → 41.12b → 41.12c → 41.12d    # Docs last
```

Parallelizable strands after 41.1 completes:
- 41.3 (Kreiss) ‖ 41.4 (GKS side param) ‖ 41.5 (L3) ‖ 41.8 (non-normality).
- 41.6, 41.7 depend on 41.1 but not each other.
- 41.9 needs 41.1 + 41.6 + 41.8.

---

## Completion Criteria

- `stencil_gen/gks_kreiss.py` exists with a passing `kreiss_stability_check` that distinguishes the classical-E4 stable closure from the Gaussian ε=0.1 known-unstable closure (integration test `TestKreissIntegration`).
- `stencil_gen/non_normality.py` exists and `compute_non_normality` returns a fully-populated `NonNormalityReport` on a BL-sized (N=31) 2D test matrix within 30 seconds.
- `stencil_gen/brady2d_stability.py` exposes `brady2d_stability_score` that successfully runs at `max_layer=7` for every family in `FAMILIES`.
- All 10 family entries from the calibration phase are persisted under `known_values.json["brady2d_calibration"]`, with `TestRegressionBrady2DCalibration` verifying they re-compute within tolerance.
- `cd scripts/stencil_gen && uv run pytest tests/ -x -q` passes in under 60 seconds (new slow tests marked appropriately; `-m "not slow"` stays under 30 seconds).
- `cd scripts/stencil_gen && uv run python -m stencil_gen.brady2d_cli --scheme E4 --kernel tension --sigma 3.0 --max-layer 6` prints a full `StabilityReport` with all layers populated and `overall_verdict="pass"`.
- `gks_group_velocity_check` accepts `side` parameter without breaking any existing callers (40.X and 41.4a backward compatible).
- Documentation: `docs/brady2d_stability_reference.md` exists; `docs/group_velocity_reference.md` cross-references Kreiss and non-normality modules; both skill files updated.
- No new dependencies beyond what `scripts/stencil_gen/` already uses (numpy, scipy, sympy, pytest). `scipy.sparse.linalg` is already a transitive dep.
- Plan 42 can now start: C++ bridge implementation has a well-defined set of families to port (the winners of `known_values.json["brady2d_calibration"]`).
