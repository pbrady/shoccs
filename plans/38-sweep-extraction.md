# Phase 38: Extract Research Sweeps from Test Suite

**Goal:** Move parameter space exploration sweeps out of pytest into standalone scripts under `scripts/stencil_gen/sweeps/`, leaving the test suite for regression verification only. The sweeps discover optimal parameters; the tests verify them. These are conceptually different activities and belong in separate places.

**Depends on:** Phase 37 (complete — `@pytest.mark.slow` markers identify which tests are sweeps)

**Read first:**
- `scripts/stencil_gen/tests/test_phs.py` (22 slow classes — the sweeps to extract)
- `scripts/stencil_gen/tests/test_e4_cut_cell.py` (slow symbolic validation classes)
- `scripts/stencil_gen/tests/test_group_velocity.py` (1 slow method-level mark)
- `scripts/stencil_gen/stencil_gen/phs.py` (`stability_eigenvalue`, `build_diff_matrix_rbf` — the core functions sweeps call)
- `scripts/stencil_gen/tests/conftest.py` (`--run-slow` infrastructure from Phase 37)

**Test commands:**
```bash
# Default fast suite (should remain ~35s, unaffected by extraction)
cd scripts/stencil_gen && uv run pytest tests/ -x -q

# Run a specific sweep
cd scripts/stencil_gen && uv run python -m sweeps.epsilon_sweep --scheme E2

# Run all sweeps and update known values
cd scripts/stencil_gen && uv run python -m sweeps.run_all

# Verify known values haven't drifted
cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegression"
```

**Design principle:** Each sweep script should:
1. Run the parameter exploration (the expensive part)
2. Print a human-readable results table to stdout
3. Write discovered optimal values to `sweeps/known_values.json`
4. Exit — no pytest assertions, no test infrastructure

The regression tests in `test_phs.py` (`TestRegressionE2Stability`, etc.) then load from `known_values.json` and spot-check stability at those values. If a sweep discovers new optimal parameters, updating the JSON automatically updates what the regression tests verify.

---

## Items

### 38.1 — Sweep Infrastructure

- [x] **38.1a** Create `scripts/stencil_gen/sweeps/` package with `__init__.py` and `__main__.py`:
  - `__init__.py`: package docstring explaining the sweep/regression separation
  - `__main__.py`: CLI entry point with subcommands (`epsilon`, `tension`, `footprint`, `comparison`, `all`)
  - `_common.py`: shared helpers — `SweepResult` dataclass, `print_table()` formatter, `save_known_values()` / `load_known_values()` for JSON I/O
  - File: `scripts/stencil_gen/sweeps/__init__.py`, `sweeps/__main__.py`, `sweeps/_common.py`
  - Test: `cd scripts/stencil_gen && uv run python -c "from sweeps._common import SweepResult; print('ok')"`

- [x] **38.1b** Create `scripts/stencil_gen/sweeps/known_values.json`:
  - Extract the hard-coded values from `TestRegressionE2Stability`, `TestRegressionE4Stability`, `TestRegressionFootprint`, `TestRegressionComparison` into a structured JSON file.
  - Structure:
    ```json
    {
      "E2_1": {
        "tension": {"sigma": 6.0, "stable_at": [20, 40, 80]},
        "gaussian": {"epsilon": 2.0, "stable_at": [40]},
        "phs_k2": {"stable_at": [20, 40, 80, 160]}
      },
      "E4_1": {
        "tension": {"sigma": 3.0, "stable_at": [20, 40, 80, 160]},
        "gaussian": {"epsilon": 0.9, "stable_at": [40]},
        "multiquadric": {"epsilon": 1.0, "stable_at": [40]},
        "phs_k2": {"stable_at": [20, 40, 80, 160]},
        "known_unstable": [{"gaussian": {"epsilon": 0.1, "n": 20}}]
      },
      "footprint": {
        "E4_nextra0_phs": {"stable_at": [20, 40, 80, 160]},
        "E4_nextra0_tension_3": {"stable_at": [40]},
        "E4_nextra1_phs": {"stable_at": [40]},
        "E4_nextra2_phs": {"stable_at": [40]}
      }
    }
    ```
  - File: `scripts/stencil_gen/sweeps/known_values.json`
  - Test: `cd scripts/stencil_gen && uv run python -c "import json; json.load(open('sweeps/known_values.json')); print('ok')"`

- [x] **38.1c** Update `TestRegressionE2Stability`, `TestRegressionE4Stability`, `TestRegressionFootprint`, `TestRegressionComparison` to load from `known_values.json` instead of hard-coding values:
  - Add a `conftest.py` fixture or module-level loader for the JSON.
  - Each regression test reads its expected values from the JSON.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegression"`

### 38.2 — Epsilon Sweep Scripts

- [x] **38.2a** Create `sweeps/epsilon_sweep.py` — extract from `TestCorrectedSweepE2`, `TestCorrectedSweepE4`, `TestEpsilonSweepE2`, `TestEpsilonSweepE4`:
  - Functions: `run_epsilon_sweep(scheme, kernel, n_values, n_eps)` — the core sweep logic
  - CLI: `--scheme E2/E4`, `--kernel gaussian/multiquadric`, `--n-values 20,40,80`, `--n-eps 60`
  - Output: prints stability table, identifies stable band, writes best epsilon to `known_values.json`
  - File: `scripts/stencil_gen/sweeps/epsilon_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps.epsilon_sweep --scheme E2 --kernel gaussian --n-eps 10` (quick smoke test with few points)

- [x] **38.2a-fix** Wire `--update-known-values` through the `__main__.py` dispatcher for the `epsilon` subcommand:
  - Add `sub_eps.add_argument("--update-known-values", action="store_true")` to the epsilon subparser.
  - Pass `*(["--update-known-values"] if args.update_known_values else [])` in the dispatch call to `eps_main`.
  - Apply the same pattern to each future sweep subcommand as it is wired in.
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps epsilon --help` (should show `--update-known-values`)

- [ ] **38.2b** Create `sweeps/mixed_epsilon_sweep.py` — extract from `TestMixedEpsilon`:
  - Functions: `run_mixed_epsilon_sweep(scheme, kernel, n_groups, n_eps_per_group)`
  - Handles: single-epsilon baseline, 2-group 2D sweep, per-row coordinate descent
  - File: `scripts/stencil_gen/sweeps/mixed_epsilon_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps.mixed_epsilon_sweep --scheme E4 --n-eps 5`

### 38.3 — Tension Sweep Scripts

- [ ] **38.3a** Create `sweeps/tension_sweep.py` — extract from `TestCorrectedTensionE2`, `TestCorrectedTensionE4`, `TestTensionSweepE2`, `TestTensionSweepE4`:
  - Functions: `run_tension_sweep(scheme, n_values, n_sigma)` — coarse + fine sweep
  - CLI: `--scheme E2/E4`, `--n-sigma 61`, `--sigma-max 20`
  - Output: sigma stability table, best sigma, stable count
  - File: `scripts/stencil_gen/sweeps/tension_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps.tension_sweep --scheme E2 --n-sigma 10`

- [ ] **38.3b** Create `sweeps/tension_penalty_sweep.py` — extract from `TestCorrectedTensionPenaltyE4`, `TestTensionConservationE2`, `TestTensionConservationE4`:
  - Functions: `run_tension_penalty_sweep(scheme, n_sigma, n_gamma)` — 2D (sigma, gamma) joint sweep
  - Output: stability/conservation landscape, best (sigma*, gamma*) pairs
  - File: `scripts/stencil_gen/sweeps/tension_penalty_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps.tension_penalty_sweep --scheme E4 --n-sigma 5 --n-gamma 5`

### 38.4 — Footprint and Comparison Scripts

- [ ] **38.4a** Create `sweeps/footprint_sweep.py` — extract from `TestCorrectedFootprint`, `TestFootprintE4Quick`, `TestFootprintSweep`, `TestFootprintPenalty`:
  - Functions: `run_footprint_sweep(nextra_values, n_sigma, n_gamma)` — nextra × sigma (× gamma) sweep
  - Output: nextra comparison table, best parameters per nextra
  - File: `scripts/stencil_gen/sweeps/footprint_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps.footprint_sweep --n-sigma 10`

- [ ] **38.4b** Create `sweeps/comparison.py` — extract from `TestCorrectedComparison`, `TestComparisonTable`, `TestTensionComparison`, `TestTensionOptimalSigma`:
  - Functions: `run_comparison(schemes, methods, n_values)` — multi-method comparison at each scheme
  - Runs: find best epsilon (Gaussian, MQ), best sigma (tension), best (sigma,gamma), PHS k=2 baseline
  - Output: formatted comparison tables matching the paper's format
  - File: `scripts/stencil_gen/sweeps/comparison.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps.comparison --scheme E2`

### 38.5 — Alpha Extraction Script

- [ ] **38.5a** Create `sweeps/alpha_extraction.py` — extract from `TestStableEpsilonAlphas`:
  - Functions: `extract_alphas(scheme, kernel, epsilon)` — compute boundary stencil coefficients at optimal epsilon
  - Compares extracted alphas with production values from `src/operators/gradient.t.cpp`
  - Output: alpha values, conservation deficit, comparison with production
  - File: `scripts/stencil_gen/sweeps/alpha_extraction.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps.alpha_extraction --scheme E2`

### 38.6 — Remove Extracted Sweep Tests

- [ ] **38.6a** Remove Phase 29/30/31 slow test classes from `test_phs.py` (fully superseded by Phase 32 + now extracted):
  - Remove: `TestEpsilonSweepE2`, `TestEpsilonSweepE4`, `TestMixedEpsilon`, `TestStableEpsilonAlphas`, `TestComparisonTable`, `TestTensionSweepE2`, `TestTensionSweepE4`, `TestTensionOptimalSigma`, `TestTensionConservationE2`, `TestTensionConservationE4`, `TestTensionComparison`, `TestFootprintE4Quick`, `TestFootprintSweep`, `TestFootprintPenalty`, `TestCrossValidationE2`
  - Also remove `_EpsilonSweepBase` base class (only used by removed subclasses)
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q`

- [ ] **38.6b** Remove Phase 32 slow test classes from `test_phs.py` (now extracted to sweep scripts):
  - Remove: `TestCorrectedSweepE2`, `TestCorrectedSweepE4`, `TestCorrectedTensionE2`, `TestCorrectedTensionE4`, `TestCorrectedTensionPenaltyE4`, `TestCorrectedFootprint`, `TestCorrectedComparison`
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q`

- [ ] **38.6c** Remove `_dense_sweep_min` and `_bisect_threshold` module-level helpers from `test_phs.py`:
  - These are only used by slow classes (now removed). Move to `sweeps/_common.py` if the sweep scripts need them.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q`

### 38.7 — Cleanup and Verification

- [ ] **38.7a** Verify default test suite is faster after removal (target: <20s):
  - The removed classes were already skipped via `@pytest.mark.slow`, so collection time should decrease slightly but runtime should be similar.
  - Run: `cd scripts/stencil_gen && uv run pytest tests/ --durations=10 -q`
  - File: N/A (verification only)

- [ ] **38.7b** Verify sweep scripts reproduce the known values:
  - Run each sweep script with reduced resolution and confirm it finds values near the known optima.
  - Run: `cd scripts/stencil_gen && uv run python -m sweeps.run_all --quick`
  - File: N/A (verification only)

- [ ] **38.7c** Update CLAUDE.md with sweep script documentation:
  - Add sweep commands to the stencil_gen section.
  - Document the sweep → known_values.json → regression test workflow.
  - File: `CLAUDE.md`
  - Test: N/A (documentation)

---

## Ordering

```
38.1a-c (infrastructure + known_values.json) — do first
38.2a-b, 38.3a-b, 38.4a-b, 38.5a (sweep scripts) — independent, can parallelize
38.6a-c (remove old tests) — do after 38.2-38.5 are verified
38.7a-c (cleanup) — do last
```

---

## Completion Criteria

- `sweeps/` package exists with standalone scripts for all parameter explorations.
- `sweeps/known_values.json` is the single source of truth for optimal parameters.
- Regression tests in `test_phs.py` load from `known_values.json`, not hard-coded values.
- All 22 slow test classes in `test_phs.py` are removed (functionality lives in sweep scripts).
- Default test suite runs in <20s (no slow tests to even skip/collect).
- `uv run python -m sweeps.run_all --quick` reproduces known values.
- Sweep scripts have `--help` with clear usage documentation.
