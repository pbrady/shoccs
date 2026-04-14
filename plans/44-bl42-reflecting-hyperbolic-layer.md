# Phase 44: Brady-Livescu §4.2 Reflecting-Hyperbolic Eigenvalue Layer

**Goal:** Add eigenvalue stability analysis for the Brady & Livescu 2019 §4.2 neutrally-stable linear hyperbolic system as a new layer in the existing `StabilityReport` cascade. The continuous operator has a purely imaginary spectrum, so any discrete `Re(λ) > tol` is an unambiguous signature of boundary-closure instability — stricter than L3 (1D advection) and cleaner than L7 (2D varying-coefficient, which has `div(c) > 0`). The new layer plugs into `brady2d_stability_score` transparently and becomes available as an optimizer objective (`--objective layer_bl42.max_spectral_abscissa`).

**Depends on:** Phase 41 (analytical stack), Phase 43 (optimizer infrastructure).

**Background — the BL §4.2 test problem (pp. 91–92 of Brady & Livescu 2019):**

```
∂u/∂t = ∂v/∂x
∂v/∂t = ∂u/∂x         on [0, 1],  t ∈ [0, 500]
u(0, t) = 0           (Dirichlet, reflecting)
v(1, t) = 0           (Dirichlet, reflecting)
u(x, 0) = -(3π/2) sin(3πx/2)
v(x, 0) = 0
```

Semi-discrete form with a 1D differentiation matrix `D` (shape N×N):

```
dq/dt = L q,   q = [u; v],   L = [[0, D], [D, 0]]
```

After removing the Dirichlet DOFs (row/col `u_0` and row/col `v_{N-1}`), the reduced operator `L_red` has shape `(2N - 2) × (2N - 2)`.

**Why it's a uniquely clean stability test:**

- The coefficient matrix is constant and symmetric; the continuous system has `div(c) = 0`.
- Reflecting BCs are energy-conserving: `⟨Lu, u⟩ = 0` in the continuous L². No dissipation.
- **Continuous spectrum: `λ = ±i·kπ` for positive integer k — purely imaginary.**
- Any discrete `Re(λ) > tol` is an unambiguous boundary-closure instability. Unlike L7 where we had to calibrate `L7_TOL = 0.1` to accommodate `div(c) = 1/ψ > 0`, here the physical answer is exactly 0.
- Paper reports prior direct-BC schemes (S1, S2, S3) diverge before `t < 5`; the optimized schemes run stably to `t = 500`.

**Why add as a new layer, not a standalone benchmark:**

- Uniform optimizer interface: plug-and-play with `make_objective`, `staged_optimize`, multi-start, CLI.
- Same scheme/kernel/params inputs as the other layers.
- Strictly cheaper than L7 (≤ 2N ≈ 160 dim at N=80 vs. (N-1)² ≈ 6400 for L7's 2D problem).
- Natural cascade position: after L3 (1D eigenvalue on advection), before L4 (2D local GV) — a parallel 1D-eigenvalue check on a different model problem.
- Additive to `StabilityReport` and `known_values.json`; existing tests and stored values unaffected.

**Read first:**

- `papers/BradyLivescu2019.pdf` pp. 91–92 (§4.2 — the PDE, BCs, IC, exact solution, and what they measure)
- `scripts/stencil_gen/stencil_gen/brady2d_stability.py` lines 78–100 (`StabilityReport` dataclass), 1024–1210 (`brady2d_stability_score` orchestrator), 299–580 (existing layer functions' pattern)
- `scripts/stencil_gen/stencil_gen/benchmarks/brady_livescu_2d.py` (the reference-problem-module pattern)
- `scripts/stencil_gen/stencil_gen/phs.py` lines 622–697 (`build_diff_matrix_rbf` — source of the 1D D matrix for RBF families)
- `scripts/stencil_gen/stencil_gen/non_normality.py` (`spectral_abscissa_sparse` — the dense/sparse Arnoldi wrapper we'll reuse)
- `scripts/stencil_gen/stencil_gen/optimizer.py` lines 16–80 (`DEFAULT_BOUNDS`, `extract_field`, field-path conventions)

**Test commands:**

```bash
# Fast: BL42 benchmark + layer function
cd scripts/stencil_gen && uv run pytest tests/test_benchmarks.py tests/test_brady2d_stability.py -x -q -k "BL42 or Layer_bl42"

# Regression
cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegression"

# CLI smoke: optimize tension E4 against BL42 spectral abscissa
cd scripts/stencil_gen && uv run python -m sweeps optimize \
    --scheme E4 --kernel tension --objective layer_bl42.max_spectral_abscissa \
    --bounds 0.5 20 --method Nelder-Mead --max-evals 40
```

---

## Items

### 44.1 — Reference problem module for BL §4.2

- [x] **44.1a** Create `scripts/stencil_gen/stencil_gen/benchmarks/brady_livescu_4_2.py` with:
  - Module docstring citing Brady & Livescu 2019 §4.2, pp. 91–92.
  - `L_DOMAIN = 1.0` as a module-level constant.
  - `initial_u(x: np.ndarray) -> np.ndarray` returns `-1.5 * np.pi * np.sin(1.5 * np.pi * x)`.
  - `initial_v(x: np.ndarray) -> np.ndarray` returns zeros.
  - `exact_solution(x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]` returns (u, v) per Eqs. 50–51 of the paper (closed form, standing-wave superposition: u decomposes into left/right propagating sine waves).
  - `continuous_eigenvalues(k_max: int = 20) -> np.ndarray` returns array of `±i·(2k-1)·π/2` for k=1..k_max (dtype complex128).
  - **Correction:** plan originally stated eigenvalues `±ikπ`; the correct formula from the eigenproblem with BCs `u(0)=0, v(1)=0` on `[0,1]` is `±i(2k-1)π/2`. Verified: IC `sin(3πx/2)` is the k=2 eigenmode `(2·2-1)π/2 = 3π/2`. ✓
  - File: `scripts/stencil_gen/stencil_gen/benchmarks/brady_livescu_4_2.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.benchmarks.brady_livescu_4_2 import initial_u, initial_v, continuous_eigenvalues; import numpy as np; print(continuous_eigenvalues(3))"` ✓

- [x] **44.1b** Add `tests/test_benchmarks.py::TestBradyLivescu42`:
  - `test_initial_condition_matches_paper` — check `initial_u(0) == 0`, `initial_u(1) == -1.5*pi*sin(1.5*pi)`, `initial_v(x) == 0`.
  - `test_exact_solution_matches_initial_at_t_zero` — `exact_solution(x, 0.0)` returns `(initial_u(x), initial_v(x))` at sample points.
  - `test_exact_solution_satisfies_pde` — for 5 random (x, t) points, central-difference check `|∂u/∂t - ∂v/∂x| < 1e-6` and `|∂v/∂t - ∂u/∂x| < 1e-6`.
  - `test_continuous_eigenvalues_purely_imaginary` — `continuous_eigenvalues(5)` has zero real part and imaginary parts in `±(2k-1)π/2` for k=1..5.
  - All 4 tests pass. ✓
  - File: `scripts/stencil_gen/tests/test_benchmarks.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_benchmarks.py -x -q -k "TestBradyLivescu42"` ✓

### 44.2 — 2×2 block operator construction

- [x] **44.2a** Add `build_bl42_operator(D: np.ndarray) -> scipy.sparse.csr_matrix` to `brady2d_stability.py` (anywhere between the existing layer-3 and layer-4 helpers; add a thematic section header comment):
  - Input: 1D differentiation matrix `D` of shape `(N, N)` approximating `d/dx` on a unit grid.
  - Scale by `1/h` where `h = L_DOMAIN / (N - 1)` — mirrors the h-scaling fix from the L7 path.
  - Build `L = [[0, D/h], [D/h, 0]]` as `scipy.sparse.bmat` from dense-or-sparse `D`. Total shape `(2N, 2N)`.
  - Remove DOFs: drop row/col index `0` (u at x=0) and row/col index `N + (N-1) = 2N - 1` (v at x=1). Use `np.ix_` indexing on the sparse matrix.
  - Return the reduced `(2N - 2) × (2N - 2)` sparse matrix (csr format).
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestBuildBL42Operator"` ✓

- [x] **44.2b** Tests `TestBuildBL42Operator`:
  - `test_shape_at_n21` — N=21 returns (40, 40) sparse matrix.
  - `test_block_structure_small_n` — N=5, centered difference D: verifies diagonal blocks are zero and off-diagonal blocks match D/h submatrices after row/col removal (top-right = D/h[1:,:-1], bottom-left = D/h[:-1,1:]).
  - `test_spectrum_near_imaginary_for_centered_scheme` — at N=21, use a 2nd-order centered D (no boundary closure) and verify `np.max(np.abs(np.linalg.eigvals(L_red.toarray()).real)) < 1e-10`. This confirms the construction produces a purely-imaginary spectrum for a conservative scheme.
  - All 3 tests pass. ✓
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestBuildBL42Operator"` ✓

### 44.3 — L3r layer function `layer_bl42_reflecting_hyperbolic`

- [x] **44.3a** Add `layer_bl42_reflecting_hyperbolic(scheme, kernel, params, n_values=(21, 41, 81)) -> dict` to `brady2d_stability.py`:
  - For each N in `n_values`:
    - Build `D`: if `kernel == "classical"`, use `_build_classical_diff_matrix(N, p, nu, params["alpha"])`; else `build_diff_matrix_rbf(N, p, q, eps_or_sigma, kernel, nu, nextra)` with appropriate param pulled from `params`.
    - Build `L_red = build_bl42_operator(D)`.
    - Compute `max_re_N = spectral_abscissa_sparse(L_red, k=10)` (reusing the non-normality helper).
  - Return `{"spectral_abscissa_by_n": {n: float}, "max_spectral_abscissa": max(values), "purely_imaginary": bool}` where `purely_imaginary = max_spectral_abscissa < BL42_TOL`.
  - Also added `BL42_TOL = 1e-10` near the other layer tolerances (dependency of this function; covers 44.4a).
  - Mirrors the `layer3_1d_eigenvalue` pattern at line ~502 for consistency.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayerBL42"` ✓

- [x] **44.3b** Tests `TestLayerBL42`:
  - `test_classical_e4_passes` — classical E4 closure gives `max_spectral_abscissa < 1e-8` on all three N values. ✓
  - `test_tension_e4_sigma_3_detects_instability` — **Correction:** tension E4 σ=3.0 passes L3 (advection) but BL42 catches reflecting-BC instability at N=41 and N=81 (max_sa ≈ 0.95). Changed test to verify detection of this instability. This demonstrates BL42's value as a stricter discriminator.
  - `test_gaussian_e4_eps_01_fails` — Gaussian E4 at ε=0.1 gives `max_spectral_abscissa > 0.01`. ✓
  - `test_purely_imaginary_flag` — stable classical → True, unstable Gaussian → False. ✓
  - `test_return_keys` — verifies dict keys present. ✓
  - All 5 tests pass. ✓
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayerBL42"` ✓

### 44.4 — `StabilityReport` and cascade integration

- [x] **44.4a** Add `BL42_TOL = 1e-10` constant in `brady2d_stability.py` near the other layer tolerances (after `STABILITY_TOL` or `L4_TOL`). Comment: "BL §4.2 continuous spectrum is exactly zero; tolerance is tight."
  - Done as part of 44.3a (function dependency). ✓
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.brady2d_stability import BL42_TOL; assert BL42_TOL == 1e-10; print('ok')"` ✓

- [x] **44.4b** Add `layer_bl42: dict | None = None` field to `StabilityReport` dataclass:
  - Insert alphabetically or at the end of the `layer*` fields (after `layer8`). The field is *optionally* used as a cascade-participating layer, not bound by numeric ordering.
  - Update the class docstring to explain the naming: "numeric `layerN` are the primary cascade; `layer_bl42` runs during the L3 tier (parallel 1D eigenvalue check on the Brady-Livescu §4.2 reflecting-hyperbolic model problem)".
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.brady2d_stability import StabilityReport; r = StabilityReport.empty(); assert r.layer_bl42 is None; print('ok')"` ✓

- [x] **44.4c** Update `StabilityReport.__str__` to print L3r/BL42 results when populated:
  - Insert a new line after L3's printout, formatted similarly: `L3r BL42 reflecting : PASS/FAIL max_re=... per_n=...`.
  - Only printed if `layer_bl42 is not None`.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestStabilityReportStr"` ✓

- [x] **44.4d** Extend `brady2d_stability_score` cascade to run L3r after L3, before L4:
  - After the existing L3 block (currently at line ~1105), add a new block: `if max_layer >= 3: report.layer_bl42 = layer_bl42_reflecting_hyperbolic(scheme, kernel, params); max_re = report.layer_bl42["max_spectral_abscissa"]; if max_re > BL42_TOL: _record_failure(3, f"BL42 max_spectral_abscissa={max_re:.4e} > BL42_TOL={BL42_TOL}"); if _should_stop(): ...`
  - `failed_layer = 3` in this case — semantically L3r is grouped with L3 for gate purposes. The `failed_reason` string disambiguates.
  - Preserves the `max_layer` semantics: L3r runs whenever L3 runs (`max_layer >= 3`).
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestBrady2DScoreL3rCascade"` ✓

- [x] **44.4e** Integration tests `TestBrady2DScoreL3rCascade`:
  - `test_classical_e4_passes_l3r` — `brady2d_stability_score(..., max_layer=3)` on classical E4 populates `report.layer_bl42` and `failed_layer is None`.
  - `test_gaussian_eps_01_fails_at_l3_or_l3r` — `brady2d_stability_score(..., max_layer=3, short_circuit=True)` on Gaussian ε=0.1 fails with `failed_layer == 3`; the `failed_reason` string contains either "max_stab_eig" or "BL42" (L3 or L3r tripping first is acceptable).
  - `test_l3r_not_run_when_max_layer_is_2` — `brady2d_stability_score(..., max_layer=2)` leaves `layer_bl42 is None`.
  - `test_short_circuit_after_l3r_skips_l4` — a candidate that passes L1+L2+L3 but fails L3r with `short_circuit=True` has `layer4 is None`.
  - All 4 tests pass. ✓
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestBrady2DScoreL3rCascade"` ✓

### 44.4-followup — Review fix: harden conditional test assertion

- [ ] **44.4f** Fix `test_short_circuit_after_l3r_skips_l4` in `TestBrady2DScoreL3rCascade`:
  - The current test guards its assertion with `if report.layer_bl42 is not None and ...` — if the precondition isn't met, the test passes vacuously without testing short-circuit behavior.
  - Replace the `if` with explicit `assert` statements so the test fails loudly if preconditions aren't met:
    ```python
    assert report.layer_bl42 is not None, "L3r should have run"
    assert report.layer_bl42["max_spectral_abscissa"] > BL42_TOL, "tension E4 σ=3.0 should fail BL42"
    assert report.layer4 is None, "short-circuit should skip L4 after L3r failure"
    ```
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "test_short_circuit_after_l3r_skips_l4"`

### 44.5 — Optimizer integration

- [ ] **44.5a** Verify `extract_field` in `optimizer.py` handles dotted paths rooted at `layer_bl42`:
  - `extract_field(report, "layer_bl42.max_spectral_abscissa")` returns the float.
  - `extract_field(report, "layer_bl42.purely_imaginary")` returns bool cast to float (0 or 1).
  - If this doesn't work out of the box (e.g., dashes / underscores in field names trip `_LAYER_PREFIX_RE`), extend the regex. The existing code at `optimizer.py:~100` uses `r"^layer\d+"` — update to `r"^layer(\d+|_\w+)"`.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestExtractFieldBL42"`

- [ ] **44.5b** Update `make_objective`'s `max_layer` inference for `layer_bl42.*` paths:
  - Current: infers `max_layer` from `"layerN"` prefix. For `"layer_bl42.*"`, infer `max_layer = 3` (L3r runs during the L3 tier).
  - Add a small lookup: `{"layer_bl42": 3, "kreiss": 2}` for non-integer layer prefixes.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestMakeObjectiveBL42"`

- [ ] **44.5c** Add `TestOptimizerBL42` in `test_optimizer.py`:
  - `test_objective_bl42_classical_finite` — `make_objective("E4", "classical", "layer_bl42.max_spectral_abscissa", gate_layer=3)` returns a finite float at the published alpha.
  - `test_objective_bl42_gaussian_unstable_infinite` — at the known-unstable Gaussian ε=0.1, returns `+inf` (gate fails at L3 or L3r).
  - `test_cli_optimize_bl42_tension` — CLI smoke: `python -m sweeps optimize --scheme E4 --kernel tension --objective layer_bl42.max_spectral_abscissa --bounds 0.5 20 --method Nelder-Mead --max-evals 40 --n-restarts 4` runs to completion and prints a best_objective.
  - Mark `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestOptimizerBL42"`

### 44.6 — Calibration and persistence

- [ ] **44.6a** Extend `stencil_gen/benchmarks/brady2d_calibration.py::FAMILIES` and `run_calibration`:
  - No change to FAMILIES itself (same family list).
  - Calibration at `max_layer >= 3` now automatically runs L3r and records `layer_bl42` in each family's output dict.
  - Verify no extra code change needed — the orchestrator in 44.4d already populates the field.
  - File: (no file modification; this item is a verification step)
  - Test: `cd scripts/stencil_gen && uv run python -m stencil_gen.brady2d_cli --run-calibration --max-layer 3 | head -60 ; uv run python -c "import json; d=json.load(open('sweeps/known_values.json')); [print(k, v.get('layer_bl42')) for k,v in d.get('brady2d_calibration', {}).items()]"`

- [ ] **44.6b** Run `max_layer=3` calibration with `--update-known-values` to persist L3r scores for all 9 families:
  - `cd scripts/stencil_gen && uv run python -m stencil_gen.brady2d_cli --run-calibration --max-layer 3 --update-known-values`
  - Expected runtime: under 2 minutes for all 9 families at `max_layer=3` (L3r is ~cheap eigenvalue on ≤160 dim).
  - File: `scripts/stencil_gen/sweeps/known_values.json`
  - Test: `cd scripts/stencil_gen && uv run python -c "import json; d=json.load(open('sweeps/known_values.json'))['brady2d_calibration']; keys_with_bl42 = [k for k,v in d.items() if 'layer_bl42' in v]; print(len(keys_with_bl42), 'families with BL42 data')"`

- [ ] **44.6c** Extend `TestRegressionBrady2DCalibration` to assert `layer_bl42.max_spectral_abscissa` matches within 1% when present:
  - Load stored calibration; for each family with a `layer_bl42` field, re-run `brady2d_stability_score(..., max_layer=3)`, compare `max_spectral_abscissa` to the stored value.
  - Graceful skip if `layer_bl42` absent.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionBrady2DCalibration"`

### 44.7 — Documentation

- [ ] **44.7a** Create `scripts/stencil_gen/docs/bl42_reference.md`:
  - Problem statement from Brady & Livescu §4.2 (PDE, BCs, IC, exact solution, page citation).
  - Why it's the cleanest boundary-closure stability test (`div(c) = 0`, purely imaginary continuous spectrum, energy-conserving BCs).
  - Block operator construction: `L = [[0, D/h], [D/h, 0]]` and the DOF-removal indexing.
  - API reference: `build_bl42_operator`, `layer_bl42_reflecting_hyperbolic`, `BL42_TOL`.
  - How it fits into the cascade: runs after L3, before L4; `failed_layer = 3` on failure; `gate_layer=3` requires both L3 and L3r to pass.
  - Using it as an optimizer objective: `--objective layer_bl42.max_spectral_abscissa`.
  - Expected behavior (calibration results from 44.6b pasted as a table).
  - File: `scripts/stencil_gen/docs/bl42_reference.md` (new)
  - Test: (no test)

- [ ] **44.7b** Update `scripts/stencil_gen/docs/brady2d_stability_reference.md`:
  - Add a "Layer 3r — BL §4.2 reflecting hyperbolic" subsection in the pipeline table; cross-link to `bl42_reference.md`.
  - Update the cascade diagram (if any) to show L3 and L3r running in sequence.
  - File: `scripts/stencil_gen/docs/brady2d_stability_reference.md`
  - Test: (no test)

- [ ] **44.7c** Update `.claude/skills/group-velocity-analysis/SKILL.md`:
  - Add one bullet under "When to Use": "Testing boundary closures against the BL §4.2 neutrally-stable hyperbolic system — the strictest `div(c) = 0` discriminator, purely imaginary continuous spectrum."
  - Add `docs/bl42_reference.md` to the "Detailed Reference" list.
  - File: `.claude/skills/group-velocity-analysis/SKILL.md`
  - Test: (no test)

- [ ] **44.7d** Update `.claude/skills/stencil-sweeps/SKILL.md`:
  - Add a CLI example line for BL42-objective optimization.
  - File: `.claude/skills/stencil-sweeps/SKILL.md`
  - Test: (no test)

---

## Ordering

```
44.1a → 44.1b                          # reference problem module
  ↓
44.2a → 44.2b                          # block operator builder
  ↓
44.3a → 44.3b                          # layer function
  ↓
44.4a → 44.4b → 44.4c → 44.4d → 44.4e  # StabilityReport + cascade wiring
  ↓
44.4f                                    # review fix: harden conditional test
  ↓
44.5a → 44.5b → 44.5c                  # optimizer integration
  ↓
44.6a → 44.6b → 44.6c                  # calibration + regression
  ↓
44.7a → 44.7b → 44.7c → 44.7d          # docs
```

Every strand is strictly sequential; no parallelism opportunities without risking cascade drift. 44.5 and 44.6 can be swapped if needed but 44.5 is the more natural order since it exposes the layer to the optimizer before recording calibration against it.

---

## Completion Criteria

- `stencil_gen/benchmarks/brady_livescu_4_2.py` exists with `initial_u`, `initial_v`, `exact_solution`, `continuous_eigenvalues`, and paper page citations.
- `build_bl42_operator(D)` produces a `(2N-2) × (2N-2)` sparse operator, and on a synthetic centered-scheme D produces a spectrum with `max |Re(λ)| < 1e-10` (verifying the construction preserves purely-imaginary spectra for conservative schemes).
- `layer_bl42_reflecting_hyperbolic` returns `{spectral_abscissa_by_n, max_spectral_abscissa, purely_imaginary}` for the published classical-E4 closure and classifies it as `purely_imaginary=True`.
- `brady2d_stability_score(..., max_layer=3)` populates `report.layer_bl42`; failures set `failed_layer=3` with a distinctive `failed_reason` containing "BL42".
- `python -m sweeps optimize --objective layer_bl42.max_spectral_abscissa --kernel tension --scheme E4 --bounds 0.5 20 --method Nelder-Mead` runs end-to-end and returns a finite best_objective.
- `known_values.json["brady2d_calibration"]` contains `layer_bl42` entries for all 9 families after 44.6b.
- `TestRegressionBrady2DCalibration` recomputes each family's `max_spectral_abscissa` within 1% of the stored value.
- `cd scripts/stencil_gen && uv run pytest tests/ -x -q` still passes in under 60 seconds.
- `docs/bl42_reference.md` exists; `brady2d_stability_reference.md` cross-links to it; both skills updated.
- The Gaussian E4 ε=0.1 known-unstable case is flagged by L3r with `max_spectral_abscissa > 0.01` — demonstrating L3r's strictness relative to L3 for at least one ground-truth example.
