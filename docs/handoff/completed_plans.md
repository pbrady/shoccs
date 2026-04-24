# Completed Plans Summary (Plans 40–44)

All five plans complete. Mid-air-but-blocked items: four skill-file edits (harness-blocked) and two explicitly deferred plan-42 follow-ups (E2 spline families, cut-cell spline families).

## Table of Contents

1. [Plan 40 — Unify Stability and Group Velocity](#plan-40)
2. [Plan 41 — Brady-Livescu 2D Analytical Stability](#plan-41)
3. [Plan 42 — C++ Bridge: Runtime-Parameterized Stencils](#plan-42)
4. [Plan 43 — Stability Optimization Framework](#plan-43)
5. [Plan 44 — BL §4.2 Reflecting-Hyperbolic Layer](#plan-44)
6. [Git Commit Mapping and Mid-Air Items](#git-mapping)

---

## Plan 40 — Unify Stability and Group Velocity {#plan-40}

**Goal:** Add group velocity (GV) as a secondary sweep objective alongside eigenvalue stability using the feasible-then-minimize pattern; expose GKS-style diagnostics as advisory output.

**Status: complete.**

### Deliverables

| Path | Purpose |
|---|---|
| `scripts/stencil_gen/sweeps/gv_objectives.py` | Five scalar GV helpers: `interior_gv_error_max`, `interior_cutoff_fraction`, `boundary_gv_error_max`, `cutcell_gv_min_C`, `gv_score_from_matrix`; `print_gks_advisory` |
| `scripts/stencil_gen/sweeps/gv_stability_pareto.py` | Pareto-frontier sweep subcommand for (stab_eig, gv_error) trade-offs |
| `scripts/stencil_gen/sweeps/tension_sweep.py` | Added `--include-gv`, `--check-gks`, additive `tension.gv_error` / `tension_gv` JSON keys |
| `scripts/stencil_gen/sweeps/epsilon_sweep.py` | Same GV/GKS augmentation for Gaussian/multiquadric kernels |
| `scripts/stencil_gen/sweeps/tension_penalty_sweep.py` | GV as third objective in `eval_point`; `tension_penalty.gv_error` / `tension_penalty_gv` JSON keys |
| `scripts/stencil_gen/sweeps/footprint_sweep.py` | Per-nextra GV optima; `E4_nextra{nx}_tension_{N}.gv_error` and `*_tension_gv` entries |
| `scripts/stencil_gen/tests/test_sweep_gv_objectives.py` | 32+ tests covering all merge/bit-exact/GKS paths |

### Public API

```bash
uv run python -m sweeps tension --scheme E4 --include-gv --check-gks
uv run python -m sweeps gv-stability-pareto --scheme E2 --param tension --n-points 61
uv run python -m sweeps tension --scheme E4 --include-gv --update-known-values
```

```python
from sweeps.gv_objectives import boundary_gv_error_max, gv_score_from_matrix, print_gks_advisory
```

### Deferred

- **40.5f**: Cut-cell GV diagnostics (`min_C` / `has_sign_reversal` for TEMO schemes) — deferred to a standalone `sweeps/cutcell_gv_sweep.py`. The footprint sweep uses `boundary_gv_error_max`, not `cutcell_gv_min_C`.

### Corrections / issues noted inline

- **40.8c**: The additive `{primary}.gv_error` originally stored the GV at the *GV-optimum* sigma, not the *stability-optimum* sigma — semantic mismatch. Fixed; `TestRegressionGV` uses `GV_TOLERANCE_STRICT = 1.001` on primary entries.
- **40.8d**: `boundary_gv_error_max` was called at un-rounded sigma; up to 2.5% drift at small epsilon. Fixed: round first, then evaluate.
- **`TestRegressionGV`** is dormant until `--include-gv --update-known-values` populates the `*_gv` keys.

---

## Plan 41 — Brady-Livescu 2D Analytical Stability {#plan-41}

**Goal:** Build the 7-layer Python analytical stability cascade for the BL §4.3 2D varying-coefficient advection benchmark; score all spline/RBF families to prioritize C++ porting.

**Status: complete.**

### Deliverables

| Path | Purpose |
|---|---|
| `scripts/stencil_gen/stencil_gen/benchmarks/brady_livescu_2d.py` | Reference problem: `psi`, `c_x/y`, `exact_solution`, `make_coefficient_field` |
| `scripts/stencil_gen/stencil_gen/benchmarks/brady2d_calibration.py` | `FAMILIES` list + `run_calibration(max_layer)` |
| `scripts/stencil_gen/stencil_gen/gks_kreiss.py` | Trefethen 1983 Kreiss determinant: `kreiss_stability_check`, `KreissResult`, `DefectiveKappaError` |
| `scripts/stencil_gen/stencil_gen/non_normality.py` | `compute_non_normality`, `NonNormalityReport`, `spectral_abscissa_sparse`, `numerical_abscissa_sparse`, `henrici_departure`, `kreiss_constant_estimate`, `pseudospectral_abscissa_estimate` |
| `scripts/stencil_gen/stencil_gen/brady2d_stability.py` | `StabilityReport`, `brady2d_stability_score`, `layer1..layer7` functions |
| `scripts/stencil_gen/stencil_gen/brady2d_cli.py` | CLI: `python -m stencil_gen.brady2d_cli` |
| `scripts/stencil_gen/stencil_gen/group_velocity.py` | Added `side=` param to `gks_group_velocity_check`, plus `local_group_velocity_2d_varying`, `anisotropy_over_coefficient_field` |
| `scripts/stencil_gen/docs/brady2d_stability_reference.md` | Full pipeline reference |
| `scripts/stencil_gen/sweeps/known_values.json` | `brady2d_calibration` key: per-family L1–L6 scores |

### Public API

```python
from stencil_gen.brady2d_stability import brady2d_stability_score, StabilityReport
report = brady2d_stability_score("E4", "tension", {"sigma": 3.0}, max_layer=6)
```

```bash
uv run python -m stencil_gen.brady2d_cli --scheme E4 --kernel tension --sigma 3.0 --max-layer 6
```

### Layer pipeline

L1 GV error → L2 Kreiss GKS → L3 1D eigenvalue → L4 2D local GV → L5 anisotropy → L6 non-normality (1D) → L7 sparse 2D eigenvalue → L8 (plan 42).

### Corrections / issues noted inline

- **L7 threshold**: `L7_TOL = 0.1` (not 1e-8 or 5e-3). The 2D varying-coefficient operator is not skew-symmetric because `div(c) = 1/ψ > 0`; stable schemes show `max Re(λ) ~ O(1e-2)`; threshold calibrated to separate from the known-unstable Gaussian at ~3.1. See `scientific_findings.md`.
- **h-scaling bug fix**: `build_diff_matrix_rbf` returns weights for unit grid spacing; `L7` must divide by `h = L_DOMAIN / (N-1)`. Fixed in commit `843c974` after the original plan-41 run.
- **E2 classical removed from FAMILIES**: `_build_classical_diff_matrix` requires p≥2; E2 has no free α parameters to optimize.
- **Layer 6 was missing from the orchestrator initially** (review pass found jump L5 → L7). Fixed at 41.10b-followup.

---

## Plan 42 — C++ Bridge: Runtime-Parameterized Stencils {#plan-42}

**Goal:** Close the Python ↔ C++ loop — add three runtime-parameterized spline families to the C++ stencil library (construction-time linear solve, no rebuild per sweep point), wire L8 into `brady2d_stability_score`.

**Status: complete** for all mandatory items. Two items explicitly deferred.

### Deliverables

| Path | Purpose |
|---|---|
| `lua-configs/brady_livescu_4_3.lua` | BL §4.3 Lua template (with `--{{N}}--`, `--{{T_FINAL}}--`, `--{{SCHEME_TABLE}}--` markers) |
| `lua-configs/brady_livescu_4_3_n61.lua`, `brady_livescu_4_3_long.lua` | Standalone BL §4.3 variants |
| `scripts/stencil_gen/stencil_gen/cpp_bridge.py` | `make_brady2d_lua`, `run_cpp_brady2d`, `BridgeResult` |
| `scripts/stencil_gen/stencil_gen/codegen.py` | `StencilGenSpec.scalar_params` field + emission |
| `scripts/stencil_gen/stencil_gen/printer.py` | `build_symbol_map(scalar_params=)` |
| `src/stencils/tension_E4u_1.cpp` | Runtime-param tension: constructor-time 10×10 Gaussian elim, `sigma` from Lua |
| `src/stencils/gaussian_E4u_1.cpp`, `multiquadric_E4u_1.cpp` | Same pattern for Gaussian and multiquadric kernels |
| `src/stencils/stencil.cpp` | Dispatch for `"tension_E4u"`, `"gaussian_E4u"`, `"multiquadric_E4u"` |
| `src/stencils/{tension,gaussian,multiquadric}_E4u_1.t.cpp` | Catch2 tests: Floating + Dirichlet + right=true |
| `scripts/stencil_gen/tests/fixtures/tension_e4u1_reference.py` | `REFERENCE_TENSION_E4U1_SIGMA3_COEFFS` (5×7 Python reference) |
| `scripts/stencil_gen/stencil_gen/brady2d_stability.py` | `layer8_cpp_simulation`, `L8_FINAL_LINF_TOL`; `brady2d_stability_score(..., max_layer=8)` |
| `scripts/stencil_gen/sweeps/brady2d_sweep.py` | `brady2d` sweep subcommand with `--validate-with-cpp` |
| `docs/brady2d_cpp_bridge_reference.md` | Bridge architecture reference |

### Public API

```bash
./build/src/app/shoccs lua-configs/brady_livescu_4_3_n61.lua
ctest --test-dir build -R "t-tension_E4u_1|t-gaussian_E4u_1|t-multiquadric_E4u_1"

uv run python -m sweeps brady2d --scheme E4 --kernel tension --param-range 2 4 3 \
    --max-layer 7 --validate-with-cpp
```

Lua scheme tables:
```lua
scheme = { order = 1, type = "tension_E4u", sigma = 3.0 }
scheme = { order = 1, type = "gaussian_E4u", epsilon = 0.9 }
scheme = { order = 1, type = "multiquadric_E4u", epsilon = 1.0 }
```

### Deferred

- **42.10a**: E2 spline families (`tension_E2u_1`, etc.) — clone-and-rename once E4 families are proven.
- **42.10b**: Cut-cell variants (`tension_E4_1`) — requires psi-dependent coefficient cache; separate follow-up plan.

### Corrections / issues noted inline

- **Runtime-param strategy**: Spline coefficients are *not* symbolically lifted to C++ (SymPy cannot CSE `exp(-σ·r)`). Solution: 10×10 Gaussian-elim linear solve at *struct construction* (once per simulation), coefficients cached in an `std::array<real, 5*7>`. See `scientific_findings.md`.
- **Dimensions**: `build_diff_matrix_rbf(p=2, q=3, nu=1, nextra=0)` produces r=4, t=6 (not r=5, t=7 in the original plan draft). The 5×7 cached block in C++ pads row 4 (classical interior stencil) and col 6 (zeros for rows 0–3).
- **Use `"E4u"` not `"E4"`**: classical α ≈ [-0.77, 0.16] violates the C++ cut-cell `alpha[1] >= 197/288` constraint; BL §4.3 is uniform-domain.
- **Stencil coefficient match precision**: Python vs. C++ agrees to ≥14 significant digits for all three kernels.

---

## Plan 43 — Stability Optimization Framework {#plan-43}

**Goal:** Wrap `brady2d_stability_score` as a scipy objective; add local (Nelder-Mead/COBYQA), global (SHGO/DE), multi-start, and staged cheap-inner + expensive-validator pipelines; persistence to `known_values.json["brady2d_optima"]`; classical-α E4 basin survey.

**Status: complete** for all mandatory items. Two skill-file edits blocked; see "mid-air".

### Deliverables

| Path | Purpose |
|---|---|
| `scripts/stencil_gen/stencil_gen/optimizer.py` | Full optimization module — see `framework_architecture.md` for the API surface |
| `scripts/stencil_gen/stencil_gen/benchmarks/alpha_basin_survey.py` | `run_survey`, `format_survey_table` for multi-seed classical-α diversity study |
| `scripts/stencil_gen/sweeps/optimize.py` | CLI driver |
| `scripts/stencil_gen/tests/test_optimizer.py` | 94+ tests |
| `scripts/stencil_gen/tests/test_phs.py` | `TestRegressionBrady2DOptima` regression class (dormant until optima are persisted) |
| `scripts/stencil_gen/docs/optimization_reference.md` | Full optimizer API reference |

### Public API

```bash
uv run python -m sweeps optimize \
    --scheme E4 --kernel tension \
    --objective layer3.max_stab_eig \
    --gate-layer 3 --bounds 0.5 20 \
    --method Nelder-Mead --max-evals 200 --n-restarts 10

uv run python -m sweeps optimize \
    --scheme E4 --kernel classical \
    --objective layer6.transient_growth_bound \
    --method staged --validator-max-layer 6 --top-k 5 \
    --validate-with-cpp --update-known-values

uv run python -m stencil_gen.brady2d_cli --alpha-basin-survey --n-seeds 20
```

```python
from stencil_gen.optimizer import make_objective, run_staged_optimize, DEFAULT_BOUNDS
f = make_objective("E4", "tension", "layer3.max_stab_eig", gate_layer=3)
result = run_staged_optimize("E4", "classical", "layer6.transient_growth_bound",
                             DEFAULT_BOUNDS[("E4", "classical")])
```

### Deferred

- Multi-objective Pareto (pymoo NSGA-II) — plan 45.
- Multi-fidelity BO — plan 46.
- Brady-Livescu 1D Euler reproduction — plan 47.
- E2 classical-α (4D) optimization — future.
- `tension-penalty` / `mixed-epsilon` kernels through the layered cascade (`brady2d_stability_score` routes only `classical`, `tension`, `gaussian`, `multiquadric`).

### Corrections / issues noted inline

- **`layer1.boundary_gv_err` is monotone** — minimizing it drives σ to the lower bound. Use `layer3.max_stab_eig` or `layer6.transient_growth_bound` for non-trivial interior minima.
- **`DEFAULT_BOUNDS[("E4", "classical")]` uses `[(-2,2),(0.05,2)]`** — the C++ cut-cell constraint `alpha[1] >= 197/288` is intentionally *not* enforced because BL feasible region sits at α₁ ≈ 0.16. L8 records `cpp_cutcell_violates_197_288` flag for diagnostic.
- **Persistence schema** (43.8c): `gate_layer`, `max_layer`, `validator_max_layer` are persisted alongside results so `TestRegressionBrady2DOptima` can rebuild `make_objective` deterministically.
- **Gate-layer for L3r/L6/L7 objectives**: When the objective *is* the value of a given layer, `gate_layer` must be strictly less than that layer. Not auto-inferred; user must pass explicitly. See `known_limitations.md` and `next_steps.md` for the auto-infer follow-up.

---

## Plan 44 — BL §4.2 Reflecting-Hyperbolic Layer {#plan-44}

**Goal:** Add L3r stability layer based on BL §4.2 neutrally-stable linear hyperbolic system (purely imaginary continuous spectrum, energy-conserving reflecting BCs) to the `StabilityReport` cascade.

**Status: complete** for all mandatory items. Two skill-file edits blocked.

### Deliverables

| Path | Purpose |
|---|---|
| `scripts/stencil_gen/stencil_gen/benchmarks/brady_livescu_4_2.py` | Reference problem: `initial_u/v`, `exact_solution`, `continuous_eigenvalues` — note eigenvalues are `±i(2k-1)π/2`, not `±ikπ` |
| `scripts/stencil_gen/stencil_gen/brady2d_stability.py` | `build_bl42_operator(D)` → `(2N-2)×(2N-2)` sparse; `layer_bl42_reflecting_hyperbolic`; `BL42_TOL = 1e-10`; `StabilityReport.layer_bl42` field; L3r block in cascade |
| `scripts/stencil_gen/stencil_gen/optimizer.py` | `_FIELD_LAYER_ALIAS["layer_bl42"] = 3` so `make_objective` infers `max_layer=3` |
| `scripts/stencil_gen/sweeps/known_values.json` | `brady2d_calibration` updated: 7 families fail at L3r; 2 families (E4_classical, E2_phs_k2) have full L1–L6 + L3r data |
| `scripts/stencil_gen/docs/bl42_reference.md` | Problem statement, operator construction, API, cascade position |

### Public API

```python
from stencil_gen.brady2d_stability import (
    build_bl42_operator, layer_bl42_reflecting_hyperbolic,
    BL42_TOL, brady2d_stability_score,
)
report = brady2d_stability_score("E4", "tension", {"sigma": 3.0}, max_layer=3)
# report.layer_bl42 = {"spectral_abscissa_by_n": {21, 41, 81}, "max_spectral_abscissa", "purely_imaginary"}
```

```bash
uv run python -m sweeps optimize \
    --scheme E4 --kernel tension \
    --objective layer_bl42.max_spectral_abscissa \
    --gate-layer 2 --bounds 0.5 20 \
    --method Nelder-Mead --max-evals 40 --n-restarts 4
```

### Corrections / issues noted inline

- **Eigenvalue formula corrected**: original plan stated `±ikπ`; correct formula from the BC eigenproblem is `±i(2k-1)π/2`. Verified: IC `sin(3πx/2)` is the k=2 eigenmode.
- **Major finding**: Tension E4 σ=3.0 passes L3 (1D advection) but **fails L3r** (`max_sa ≈ 0.95` at N=41/81). BL42 is a strictly stricter discriminator. Tension family is **universally infeasible** for BL42 across σ ∈ [0.01, 50] (verified via 898-eval DE run). See `scientific_findings.md`.
- **`known_values.json` verdicts updated**: E4_phs_k2, E4_tension_3, E4_gaussian_09, E4_multiquadric_1 now show `"fail"` at `failed_layer=3`. Only E4_classical and E2_phs_k2 pass.
- **Calibration overwrite** (44.6b vs 44.6d): Running `--run-calibration --max-layer 3` after a prior `max_layer=6` run overwrote layer4–6 data. Re-ran at `max_layer=6` in 44.6d to restore. Guard against this by not running lower-depth calibrations on top of higher-depth data without `--merge`.

---

## Git Commit Mapping and Mid-Air Items {#git-mapping}

### Plan 44 commits (most recent first, all complete)
`2fa1293` 44.7a-b → `c1f3c0a` 44.6d → `25d021a` 44.6a-c → `2bd36cb` 44.5d → `5787ff4` 44.5a-c → `dd23473` 44.4o-p → `f83bcbc` 44.4n → `e9effa2` 44.4g-j → `a873f4a` 44.4f → `09421d2` 44.4b-e → `737009b` 44.3a-b+44.4a → `d228df5` 44.2a-b → `a68ff29` 44.1a-b

Full chain: `git log --oneline 843c974..HEAD` from the h-scaling fix onward.

### Mid-Air / Blocked Items

| Item | Status | Blocker |
|---|---|---|
| 43.11c `.claude/skills/stencil-sweeps/SKILL.md` | blocked | Harness write-blocks `.claude/skills/**`. Manually completed this session. |
| 43.11d `.claude/skills/group-velocity-analysis/SKILL.md` | blocked | Same. Manually completed this session. |
| 44.7c `.claude/skills/group-velocity-analysis/SKILL.md` | blocked | Same. Manually completed this session. |
| 44.7d `.claude/skills/stencil-sweeps/SKILL.md` | blocked | Same. Manually completed this session. |

To unblock automation: add `"Edit(/workspace/.claude/skills/**)": "allow"` and `"Write(/workspace/.claude/skills/**)": "allow"` to `.claude/settings.local.json`. See `operating_conventions.md`.
