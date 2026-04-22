# Phase 45: Multi-Objective Pareto Optimization (NSGA-II)

**Goal:** Add a multi-objective optimizer to the stencil stability framework that produces Pareto fronts over 2–3 stability metrics simultaneously. The existing single-objective drivers (Nelder-Mead, DE, staged) collapse fundamentally conflicting metrics onto a scalar; plan 44 exposed real trade-offs (tension closures have excellent `layer1.boundary_gv_err` but universally fail `layer_bl42.max_spectral_abscissa`; classical closures have weaker L1 but pass L3r) that a scalar optimizer cannot represent. This plan adds a pymoo-backed NSGA-II driver, a `make_multi_objective` factory, a `pareto` CLI subcommand, per-run JSON persistence under `sweeps/pareto_fronts/`, and regression coverage. It also ships a small prerequisite fix (`gate_layer` auto-infer in `make_objective`) that today's single-objective CLI users also benefit from.

**Depends on:** Phase 41 (stability cascade), Phase 43 (optimizer framework, `make_objective`), Phase 44 (L3r layer — the new discriminator objective).

**Background — why multi-objective:**

From `docs/handoff/scientific_findings.md` (discovery #9) and plan 44's calibration:

| Metric | Measures | Tension E4 σ=3 | Classical E4 (BL α) |
|---|---|---|---|
| `layer1.boundary_gv_err` | dispersion quality | ~3.6e-2 (excellent) | ~1e-1 |
| `layer_bl42.max_spectral_abscissa` | reflecting-BC stability | ~0.95 (fails) | ~3e-14 (passes) |
| `layer6.transient_growth_bound` | non-normal growth | ~3.3 | varies by basin |

No single scalar captures "best closure." The user's physics (acoustics vs. advection-dominated vs. transient-growth-sensitive) decides. A Pareto front makes the trade-off concrete and surveyable.

**Multi-objective formulation:**

```
minimize  F(x) = [f_1(x), f_2(x), ..., f_m(x)]
subject to  x ∈ [lb, ub]
```

Pareto-dominance: `x` dominates `y` iff `F_i(x) ≤ F_i(y)` for all `i` and strictly `<` for at least one. The Pareto front is the set of non-dominated `x` — no better on one axis without being worse on another. NSGA-II (Deb et al. 2002) evolves a population toward this front via fast non-dominated sort + crowding-distance survival. At convergence, the population *is* (an approximation of) the Pareto front.

**Why pymoo:**

- Standard implementation; widely used; API stable at 0.6.
- Ships NSGA-II, NSGA-III, hypervolume indicator, reference-direction helpers.
- `ElementwiseProblem` interface composes cleanly with our existing `make_objective` pattern — one vector-valued objective wrapping our scalar-producing layer extractor.
- No PyTorch/TensorFlow dependency (unlike BoTorch, Trieste — deferred to plan 46).

**What this plan does NOT do:**

- **NSGA-III for 4+ objectives.** The MVP is 2–3 objectives. NSGA-III is a one-argument change (`NSGA3(ref_dirs=...)`) once the infrastructure exists; deferred as a follow-up if demand emerges.
- **Weighted scalarization** (`--objective "0.5*a + 0.5*b"`). Deferred; Pareto is the primary interface.
- **Multi-fidelity Bayesian optimization.** Plan 46.
- **Constraint-aware Pareto** (`pymoo.Problem.evaluate` with `G` array). Feasibility is already encoded via sentinel `+inf` from the gate.
- **Plots.** No matplotlib. Fronts are written as JSON; visualization is downstream work.
- **Non-`classical`/`tension`/`gaussian`/`multiquadric` kernels.** Same scope as plan 43.
- **3D or tensor-product 3D extensions.** L7 is 2D by construction.

**Read first:**

- `scripts/stencil_gen/stencil_gen/optimizer.py` lines 74–82 (`DEFAULT_BOUNDS`), 87–131 (`OptimizeResult`), 193–238 (`extract_field`), 240–262 (`_LAYER_PREFIX_RE`, `_FIELD_LAYER_ALIAS`, `_infer_max_layer`), 265–330 (`make_objective`)
- `scripts/stencil_gen/sweeps/optimize.py` lines 1–80 (CLI arg parsing, `_resolve_bounds`, `_KERNEL_DIM`), and the full file for the `--update-known-values` and `--validate-with-cpp` patterns
- `scripts/stencil_gen/sweeps/_common.py` full file (`load_known_values`, `save_known_values`, `SCHEME_PARAMS`)
- `scripts/stencil_gen/sweeps/__main__.py` full file (subcommand registration pattern)
- `scripts/stencil_gen/sweeps/gv_stability_pareto.py` full file (existing Pareto-*scan* sweep — **complementary** to the new NSGA-II driver, not a replacement; preserved as-is)
- `scripts/stencil_gen/stencil_gen/brady2d_stability.py` lines 80–100 (`StabilityReport`), 582–635 (`layer_bl42_reflecting_hyperbolic`), 1140–1290 (`brady2d_stability_score` orchestrator)
- `scripts/stencil_gen/tests/test_phs.py` lines 1358–1510 (`_KNOWN` loading + `TestRegression*` skip-when-absent pattern)
- `docs/handoff/scientific_findings.md` (entirety; shapes what objectives are worth pairing)
- pymoo 0.6 docs: https://pymoo.org/getting_started/part_2.html, https://pymoo.org/algorithms/moo/nsga2.html, https://pymoo.org/misc/indicators.html

**Test commands:**

```bash
# Fast: pareto unit tests (mocked objectives, synthetic NSGA-II)
cd scripts/stencil_gen && uv run pytest tests/test_pareto.py tests/test_optimizer.py -x -q -k "Pareto or MultiObjective or GateLayerInfer"

# Slow integration: real NSGA-II run on E4 classical, tiny budget (pop=12, gen=4)
cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_pareto.py -x -q -k "Integration" --run-slow

# CLI smoke: 2D Pareto front on E4 classical, cheap inner layers only
cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python -m sweeps pareto \
    --scheme E4 --kernel classical \
    --objectives layer1.boundary_gv_err layer_bl42.max_spectral_abscissa \
    --bounds -2 2 0.05 2 --pop-size 20 --n-gen 10 --seed 1 --persist
```

---

## Items

### 45.0 — Prerequisites: nlopt platform guard + gate_layer auto-infer

- [x] **45.0a** Fix `pyproject.toml` so `uv sync` succeeds on aarch64. No nlopt version has ever had an aarch64 Linux wheel on PyPI — not a regression, just a permanent gap in upstream wheel coverage. The existing `nlopt>=2.7` hard dependency blocks `uv sync` entirely on aarch64 containers, so plan 45 cannot proceed until this is fixed. Preserve nlopt availability by adding a `[tool.uv.sources]` override routing `nlopt` to the upstream Python-bindings git repo (same codebase that ships the PyPI wheels):
  ```toml
  [tool.uv.sources]
  # nlopt has no aarch64 wheels on PyPI. Build from upstream Python-bindings
  # source (same codebase that publishes the PyPI wheels). Requires swig + cmake,
  # both present in .devcontainer/Dockerfile base-system stage. First-sync build
  # cost ~7s; cached on subsequent syncs.
  nlopt = { git = "https://github.com/DanielBok/nlopt-python.git", tag = "2.10.0" }
  ```
  Keep `nlopt>=2.7` in `[project] dependencies` — uv reads the source override and routes accordingly. Verified working on aarch64: `uv pip install 'nlopt @ git+https://github.com/DanielBok/nlopt-python.git@2.10.0'` builds and installs in ~7s, `LN_BOBYQA` converges correctly on a Rosenbrock sanity problem.
  - File: `scripts/stencil_gen/pyproject.toml`
  - Test: `cd scripts/stencil_gen && uv sync && uv run python -c "import nlopt, pymoo; from pymoo.algorithms.moo.nsga2 import NSGA2; from pymoo.core.problem import ElementwiseProblem; from pymoo.indicators.hv import HV; print('nlopt', nlopt.__version__, 'pymoo', pymoo.__version__)"`

- [ ] **45.0b** Add `gate_layer` auto-inference to `make_objective`. Change signature at `optimizer.py:265` from `gate_layer: int = 3` to `gate_layer: int | None = None`. After `max_layer` is resolved (line 307), insert:
  ```python
  if gate_layer is None:
      gate_layer = max(max_layer - 1, 0)
  ```
  Rationale: for an objective living in layer N (e.g., `layer_bl42.*` → layer 3, `layer6.*` → layer 6), the right gate is N-1; current hard-coded `gate_layer=3` makes L6/L7 objectives *never gate* and L3r objectives *gate themselves* (causing the 2634-eval inf DE run described in `next_steps.md`). The `max(..., 0)` floor handles the degenerate `max_layer=1` case (no gate; the objective is the only layer). Preserve the `max_layer < gate_layer` validation at line 308.
  - File: `scripts/stencil_gen/stencil_gen/optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "GateLayerInfer"`

- [ ] **45.0c** Add `TestGateLayerInfer` class to `tests/test_optimizer.py` with 5 tests:
  - `test_default_gate_for_layer6_objective` — `make_objective("E4", "classical", "layer6.transient_growth_bound")` with no `gate_layer` kwarg infers `gate_layer=5`, `max_layer=6`.
  - `test_default_gate_for_bl42_objective` — same for `layer_bl42.max_spectral_abscissa` → `gate_layer=2`, `max_layer=3`.
  - `test_default_gate_for_layer1_objective_no_gate` — `layer1.boundary_gv_err` → `gate_layer=0`, `max_layer=1` (the no-gate degenerate case; verify the returned closure still works and returns finite values at known-feasible points).
  - `test_explicit_gate_layer_preserved` — passing `gate_layer=3` explicitly with `layer6.*` objective overrides the auto-infer (use a sentinel-check via calling the closure at a known-L3-feasible-but-L6-strict point and confirming no inf from gating).
  - `test_no_auto_infer_raises_on_unknown_field` — `make_objective("E4", "classical", "bogus_field")` still raises `ValueError` from `_infer_max_layer` unchanged.
  - File: `scripts/stencil_gen/tests/test_optimizer.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_optimizer.py -x -q -k "TestGateLayerInfer"`

### 45.1 — `pareto` module: `ParetoResult`, `ParetoPoint`, `make_multi_objective`

- [ ] **45.1a** Create `scripts/stencil_gen/stencil_gen/pareto.py` with two frozen dataclasses:
  ```python
  @dataclass(frozen=True)
  class ParetoPoint:
      x: np.ndarray              # shape (n_var,), float64
      params: dict               # params_from_vector(kernel, x)
      objectives: np.ndarray     # shape (n_obj,), float64
      report: dict               # _report_to_dict(StabilityReport) from optimizer.py:700

  @dataclass(frozen=True)
  class ParetoResult:
      front: tuple[ParetoPoint, ...]     # non-dominated members
      objective_fields: tuple[str, ...]  # e.g. ("layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa")
      scheme: str
      kernel: str
      bounds: tuple[tuple[float, float], ...]
      method: str                        # "NSGA-II" (or "NSGA-III" when 45 extended)
      pop_size: int
      n_gen: int
      n_evals: int
      seed: int
      compute_time: float
      hv_trace: tuple[float, ...]        # hypervolume per generation
      ref_point: tuple[float, ...]       # reference point used for HV
      extras: dict                       # driver-specific diagnostics
  ```
  Module docstring cites Deb et al. 2002 (NSGA-II). Import `numpy as np` and `from dataclasses import dataclass`. Place next to `optimizer.py`, not inside it — this keeps `optimizer.py` focused on scalar drivers.
  - File: `scripts/stencil_gen/stencil_gen/pareto.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_pareto.py -x -q -k "TestParetoDataclasses"`

- [ ] **45.1b** Add `make_multi_objective(scheme, kernel, report_fields, *, gate_layer=None, max_layer=None) -> Callable[[np.ndarray], np.ndarray]` to the same module:
  - Accepts a sequence of dotted-path fields (length ≥ 2).
  - Auto-infers `max_layer` as `max(_infer_max_layer(f) for f in report_fields)` and `gate_layer` as `max_layer - 1` (reusing 45.0b's floor).
  - Returns a closure `f(x: np.ndarray) -> np.ndarray` of shape `(len(report_fields),)`.
  - On any gate trip, parameter-vector-shape mismatch, or exception from `brady2d_stability_score`, returns `np.full(len(report_fields), _PARETO_SENTINEL)` where `_PARETO_SENTINEL = 1e12` (finite — hypervolume indicators reject `+inf`; per pymoo-research agent finding #6).
  - On success, extracts each field via `extract_field` (reused from `optimizer.py`) and returns the resulting `np.ndarray`.
  - Expose `_PARETO_SENTINEL` as a module-level constant.
  - File: `scripts/stencil_gen/stencil_gen/pareto.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_pareto.py -x -q -k "TestMakeMultiObjective"`

- [ ] **45.1c** Tests in `tests/test_pareto.py` (new file):
  - `TestParetoDataclasses`: construct a `ParetoPoint` and a `ParetoResult` with 3 members; assert frozen (raises `FrozenInstanceError` on assignment); assert `front` accepts tuples but not lists (or coerces — pick one; suggest: require tuple for immutability).
  - `TestMakeMultiObjective::test_shape_matches_field_count` — 2 fields → 2-vector; 3 fields → 3-vector.
  - `TestMakeMultiObjective::test_sentinel_on_gate_trip` — passing an infeasible param vector (e.g., `α = [5.0, 5.0]` for E4 classical, violating L1/L3) returns `[1e12, 1e12]`, not `[inf, inf]` or a partially-finite vector.
  - `TestMakeMultiObjective::test_sentinel_on_shape_mismatch` — wrong-length `x` returns sentinel vector without raising.
  - `TestMakeMultiObjective::test_finite_on_known_feasible_point` — `α = [-0.7733, 0.1624]` (BL published optimum) returns all-finite vector for `["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"]`.
  - `TestMakeMultiObjective::test_gate_layer_auto_inferred_from_max_field` — confirms `gate_layer = max(max_layer) - 1` in a case where fields span multiple layers.
  - File: `scripts/stencil_gen/tests/test_pareto.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_pareto.py -x -q -k "TestParetoDataclasses or TestMakeMultiObjective"`

### 45.2 — NSGA-II driver (pymoo `ElementwiseProblem` + `HVCallback`)

- [ ] **45.2a** Add `run_nsga2(scheme, kernel, report_fields, bounds, *, pop_size=40, n_gen=50, seed=1, ref_point=None, gate_layer=None, max_layer=None, verbose=False) -> ParetoResult` to `pareto.py`:
  - Builds a private `_StabilityProblem(ElementwiseProblem)` inner class whose `__init__` sets `n_var=len(bounds)`, `n_obj=len(report_fields)`, `n_ieq_constr=0`, `xl`/`xu` from `bounds`. Its `_evaluate(self, x, out, *args, **kwargs)` calls the `make_multi_objective` closure and writes `out["F"] = f(x)`.
  - Constructs `NSGA2(pop_size=pop_size)` with pymoo defaults (SBX crossover, polynomial mutation — acceptable per pymoo-research agent finding #2).
  - Computes `ref_point` automatically if `None`: run one cheap pre-evaluation of 20 uniform-random feasible points, take `1.1 * max` per column clipped to `[0, _PARETO_SENTINEL)`. Store the chosen ref_point in the `ParetoResult`.
  - Wraps `HV(ref_point=ref_point)` indicator and a `_HVCallback` (see 45.2b) to record per-generation hypervolume on the current non-dominated set (`algorithm.opt`, not full history — per pymoo-research finding #7).
  - Calls `minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=verbose, callback=callback)`.
  - Builds `ParetoResult` from `res.X`, `res.F`, filtering out sentinel rows (any row with `F[i] >= _PARETO_SENTINEL` for any `i` is excluded from `front`; logged in `extras["n_sentinel_filtered"]`).
  - File: `scripts/stencil_gen/stencil_gen/pareto.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_pareto.py -x -q -k "TestRunNSGA2"`

- [ ] **45.2b** Add `_HVCallback(pymoo.core.callback.Callback)` inner class (or private module class) storing `self.data["hv"]` and `self.data["n_nds"]` per generation. On each `notify(algorithm)`, call `algorithm.opt.get("F")`, filter finite rows, and append `hv_indicator(filtered)` (or `0.0` if empty). Expose via `result.algorithm.callback.data["hv"]` for extraction into `ParetoResult.hv_trace`.
  - File: `scripts/stencil_gen/stencil_gen/pareto.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_pareto.py -x -q -k "TestHVCallback"`

- [ ] **45.2c** Tests in `tests/test_pareto.py`:
  - `TestRunNSGA2::test_determinism_same_seed` — two runs with identical `seed`, `pop_size`, `n_gen`, `bounds`, `report_fields` on a synthetic 2-objective analytic problem (e.g., ZDT1 wrapped into our API via a fake `brady2d_stability_score`) produce identical `front` (set equality on `objectives` arrays within 1e-12).
  - `TestRunNSGA2::test_non_dominated_front` — the returned `front` is Pareto-verified (no member dominates another in the returned set). Use a pure-numpy dominance check.
  - `TestRunNSGA2::test_hv_trace_monotone_nondecreasing` — NSGA-II with elitism ⇒ `hv_trace` is non-decreasing (within `1e-10` tolerance for numerical noise).
  - `TestRunNSGA2::test_sentinel_rows_excluded` — mock a problem where half the population is infeasible; verify `front` contains only finite rows and `extras["n_sentinel_filtered"] > 0`.
  - `TestHVCallback::test_per_gen_count_matches_n_gen` — after `n_gen=5` run, `len(hv_trace) == 5`.
  - `TestRunNSGA2::test_integration_classical_alpha_2d` — `@pytest.mark.slow`; real `brady2d_stability_score` on E4 classical with 2 objectives (`layer3.max_stab_eig`, `layer6.transient_growth_bound`), `pop_size=12, n_gen=4, seed=1`; verify: (a) `len(front) >= 3` (tiny budget but NSGA-II keeps a spread), (b) `hv_trace[-1] > 0`, (c) front is non-dominated, (d) all points have `params["alpha"]` shape-2.
  - File: `scripts/stencil_gen/tests/test_pareto.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_pareto.py -x -q -k "TestRunNSGA2 or TestHVCallback"` (integration test gated by `--run-slow`)

### 45.3 — `pareto` CLI subcommand + `__main__.py` registration

- [ ] **45.3a** Create `scripts/stencil_gen/sweeps/pareto.py` CLI module mirroring `sweeps/optimize.py`. Argparse surface:
  - `--scheme {E2,E4}` (required)
  - `--kernel {classical,tension,gaussian,multiquadric}` (required)
  - `--objectives FIELD [FIELD ...]` (nargs=2+, required; each must be a valid dotted path)
  - `--bounds LO HI [LO HI ...]` (pairs; if omitted, `DEFAULT_BOUNDS[(scheme,kernel)]`)
  - `--pop-size N` (default 40)
  - `--n-gen N` (default 50)
  - `--seed N` (default 1)
  - `--ref-point V [V ...]` (optional; otherwise auto-computed per 45.2a)
  - `--gate-layer N` (optional; otherwise auto-inferred per 45.0b)
  - `--max-layer N` (optional; otherwise auto-inferred)
  - `--persist` (boolean; write JSON to `sweeps/pareto_fronts/<scheme>_<kernel>_<mangled>.json`)
  - `--validate-with-cpp` (boolean; re-run top-K front members at L8; K = `len(front)`, capped at 10)
  - `--verbose` (forwards to pymoo)
  - `main(argv)` calls `run_nsga2`, prints a summary table (front size, hypervolume, top-5 members ordered by each objective), then persists if requested, then validates if requested.
  - `_mangle_objectives(fields: Sequence[str]) -> str` → replaces `.` with `_` and joins with `__`, e.g., `["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"]` → `"layer1_boundary_gv_err__layer_bl42_max_spectral_abscissa"`.
  - File: `scripts/stencil_gen/sweeps/pareto.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_pareto.py -x -q -k "TestParetoCLI"`

- [ ] **45.3b** Register the `pareto` subcommand in `sweeps/__main__.py`:
  - Add `from .pareto import main as pareto_main` at the top.
  - Add `sub_pareto = subparsers.add_parser("pareto", help="NSGA-II multi-objective Pareto front")` to the dispatch table alongside `optimize`. Hook its execution in the `if args.command == "pareto":` branch.
  - Do NOT add `pareto` to `_run_all()` (too expensive to run blindly; same exclusion as `optimize`).
  - Update the `pareto` help text to mention it's distinct from `gv-stability-pareto` (which is a 1D parametric scan, retained as-is).
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps pareto --help`

- [ ] **45.3c** Tests in `tests/test_sweep_pareto.py` (new):
  - `TestMangleObjectives::test_roundtrip_legible` — `_mangle_objectives(["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"])` returns exactly `"layer1_boundary_gv_err__layer_bl42_max_spectral_abscissa"`.
  - `TestMangleObjectives::test_order_preserved` — two orderings of the same fields give different mangled filenames.
  - `TestParetoCLI::test_argparse_accepts_minimal_invocation` — `main(["--scheme", "E4", "--kernel", "classical", "--objectives", "layer1.boundary_gv_err", "layer3.max_stab_eig", "--pop-size", "6", "--n-gen", "2", "--seed", "1"])` exits 0 with mocked `run_nsga2`.
  - `TestParetoCLI::test_argparse_rejects_single_objective` — `--objectives layer3.max_stab_eig` alone raises `SystemExit` from argparse (nargs=2+).
  - `TestParetoCLI::test_argparse_rejects_bad_bounds_parity` — odd number of `--bounds` values raises.
  - `TestParetoCLI::test_dispatch_registered` — `python -m sweeps pareto --help` exits 0 (use `subprocess.run`).
  - File: `scripts/stencil_gen/tests/test_sweep_pareto.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_pareto.py -x -q -k "TestMangleObjectives or TestParetoCLI"`

### 45.4 — Per-run JSON persistence to `sweeps/pareto_fronts/`

- [ ] **45.4a** Add I/O helpers to a new module `scripts/stencil_gen/sweeps/_pareto_io.py` (separate from `_common.py` which is `known_values.json`-focused):
  - `PARETO_FRONTS_DIR: Path = Path(__file__).parent / "pareto_fronts"`
  - `save_pareto_front(result: ParetoResult, directory: Path = PARETO_FRONTS_DIR) -> Path`:
    - `mkdir(parents=True, exist_ok=True)`.
    - Filename: `{scheme}_{kernel}_{mangled_objectives}.json`.
    - Serializes `ParetoResult` via a custom encoder that converts `np.ndarray` → list, tuples → lists, dataclasses → `asdict`. Guarantees deterministic key ordering (`json.dump(..., sort_keys=False, indent=2)` preserves insertion order; use an explicit `OrderedDict` of top-level keys: `scheme`, `kernel`, `method`, `objective_fields`, `bounds`, `pop_size`, `n_gen`, `n_evals`, `seed`, `compute_time`, `ref_point`, `hv_trace`, `front`, `extras`).
    - Returns the written path.
  - `load_pareto_front(path: Path) -> dict`: reads and returns the raw JSON (regression test rebuilds `make_multi_objective` from `objective_fields` and re-evaluates `x` vectors; doesn't need full `ParetoResult` reconstruction).
  - `iter_pareto_fronts(directory: Path = PARETO_FRONTS_DIR) -> Iterator[Path]`: glob `*.json` for the regression test to discover.
  - File: `scripts/stencil_gen/sweeps/_pareto_io.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_pareto.py -x -q -k "TestParetoIO"`

- [ ] **45.4b** Wire `--persist` into `sweeps/pareto.py::main`: after `run_nsga2` returns, call `save_pareto_front(result)` and print the written path. Without `--persist`, no file is written (print to stdout only). Create the `sweeps/pareto_fronts/` directory with a placeholder `.gitkeep` so the empty directory is tracked. Do NOT add it to `.gitignore` (matches `output/` convention per the sweeps-agent finding).
  - File: `scripts/stencil_gen/sweeps/pareto.py`; `scripts/stencil_gen/sweeps/pareto_fronts/.gitkeep` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_pareto.py -x -q -k "TestParetoIO"`

- [ ] **45.4c** Tests in `tests/test_sweep_pareto.py`:
  - `TestParetoIO::test_save_creates_file_at_mangled_path` — construct a synthetic 3-member `ParetoResult`, call `save_pareto_front` with `tmp_path`, verify file exists with the expected name.
  - `TestParetoIO::test_roundtrip_preserves_objectives` — save then `load_pareto_front`, verify `objective_fields`, `front[i]["objectives"]`, `front[i]["x"]` round-trip within exact equality for floats.
  - `TestParetoIO::test_serializer_handles_numpy_arrays` — `ParetoResult` with `np.ndarray` fields serializes without `TypeError`.
  - `TestParetoIO::test_iter_discovers_multiple_files` — write 2 synthetic fronts, verify `iter_pareto_fronts(tmp_path)` yields both.
  - File: `scripts/stencil_gen/tests/test_sweep_pareto.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_pareto.py -x -q -k "TestParetoIO"`

### 45.5 — C++ validation of front members

- [ ] **45.5a** In `sweeps/pareto.py::main`, when `--validate-with-cpp` is set: after the `ParetoResult` is constructed, iterate up to `min(len(front), 10)` members (sorted by first objective ascending for reproducibility), call `brady2d_stability_score(scheme, kernel, pt.params, max_layer=8, layer8_N=31, layer8_t_final=5.0)` for each, and append a `cpp_validation` dict to `ParetoResult.extras` with the shape:
  ```python
  {"cpp_validation": [
      {"x": [...], "params": {...}, "l8_final_linf": float, "l8_stable": bool,
       "cpp_cutcell_violates_197_288": bool, "wall_time_s": float},
      ...
  ]}
  ```
  Skip validation for kernels outside `_CPP_SUPPORTED_KERNELS = ("classical", "tension", "gaussian", "multiquadric")` (reuse constant from `sweeps/optimize.py:44`). Log a clear "skipped (kernel not C++-supported)" if applicable. Do NOT abort the whole run if any individual L8 call fails; record `l8_error: "<exception message>"` in that entry.
  - File: `scripts/stencil_gen/sweeps/pareto.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_pareto.py -x -q -k "TestValidateWithCpp"`

- [ ] **45.5b** Tests in `tests/test_sweep_pareto.py`:
  - `TestValidateWithCpp::test_skips_on_unsupported_kernel` — synthetic run with a kernel not in `_CPP_SUPPORTED_KERNELS` (mock); verify `extras["cpp_validation"]` absent or has a "skipped" record.
  - `TestValidateWithCpp::test_caps_at_10_members` — synthetic `ParetoResult` with 25 members; verify `len(extras["cpp_validation"]) == 10`.
  - `TestValidateWithCpp::test_records_per_member_shape` — with `brady2d_stability_score` monkeypatched to return a known `StabilityReport`, verify each entry has the specified keys.
  - `TestValidateWithCpp::test_failure_records_error_not_raises` — monkeypatch `brady2d_stability_score` to raise; verify validation continues and records `l8_error` for the failing member.
  - File: `scripts/stencil_gen/tests/test_sweep_pareto.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_pareto.py -x -q -k "TestValidateWithCpp"`

### 45.6 — Calibration runs + regression test

- [ ] **45.6a** Run two seeded Pareto sweeps with `--persist` to populate `sweeps/pareto_fronts/`:
  ```bash
  cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python -m sweeps pareto \
      --scheme E4 --kernel classical \
      --objectives layer1.boundary_gv_err layer_bl42.max_spectral_abscissa \
      --bounds -2 2 0.05 2 --pop-size 40 --n-gen 30 --seed 1 --persist

  cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python -m sweeps pareto \
      --scheme E4 --kernel tension \
      --objectives layer1.boundary_gv_err layer6.transient_growth_bound \
      --bounds 0.5 20 --pop-size 20 --n-gen 20 --seed 1 --persist
  ```
  Commit the resulting two JSON files. These are the baseline fronts the regression test verifies. Document wall times in the commit message. Expected front sizes: classical 2D should produce ~15–30 non-dominated points; tension 1D is constrained by being on a 1D curve so ~5–10 points.
  - File: `scripts/stencil_gen/sweeps/pareto_fronts/E4_classical_layer1_boundary_gv_err__layer_bl42_max_spectral_abscissa.json`, `scripts/stencil_gen/sweeps/pareto_fronts/E4_tension_layer1_boundary_gv_err__layer6_transient_growth_bound.json`
  - Test: `cd scripts/stencil_gen && uv run python -c "from pathlib import Path; import json; [print(p.name, len(json.loads(p.read_text())['front'])) for p in Path('sweeps/pareto_fronts').glob('*.json')]"`

- [ ] **45.6b** Add `TestRegressionBrady2DPareto` class to `tests/test_phs.py`, modeled on `TestRegressionBrady2DOptima` (lines 2020–2099 per agent finding #8):
  - Module-level load: `_PARETO_FRONTS = list((Path(__file__).resolve().parent.parent / "sweeps" / "pareto_fronts").glob("*.json"))` once at import, filter to only readable JSONs.
  - Class-level `@pytest.fixture(autouse=True)` that `pytest.skip`s if `_PARETO_FRONTS` is empty.
  - `test_each_front_member_objectives_match` — iterate each front file; for each member in `front`, rebuild `make_multi_objective(scheme, kernel, objective_fields)`, evaluate at `x`, assert `np.allclose(recomputed, stored_objectives, rtol=1e-2, atol=1e-8)` (1% relative — matches `TestRegressionBrady2DOptima`'s tolerance).
  - `test_each_front_is_non_dominated` — verify no member dominates another within the stored front (guards against corrupt persistence).
  - Skip individual members whose `objectives` contain the sentinel `1e12` (those shouldn't be in `front` but guard anyway).
  - Add `@pytest.mark.slow` since rebuilding `make_multi_objective` on 15-30 members × 2 fronts ≈ 1–2 minutes.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionBrady2DPareto" --run-slow`

### 45.7 — Documentation + meta.md decision + skill updates

- [ ] **45.7a** Create `scripts/stencil_gen/docs/pareto_reference.md`. Sections: "Problem" (multi-objective formulation, Pareto dominance math), "Why pymoo", "API" (`make_multi_objective`, `run_nsga2`, `ParetoResult`, `ParetoPoint`), "CLI" (`python -m sweeps pareto` with 3 example invocations — one each for classical 2D, tension 1D, and a 3-objective classical-α run), "Persistence" (per-file schema under `sweeps/pareto_fronts/`, filename mangling), "Ref-point selection" (automatic via 1.1× pilot max; how to override), "Cascade integration" (how `gate_layer`/`max_layer` auto-infer per 45.0b cooperates with the multi-objective factory), "Relationship to `gv_stability_pareto.py`" (one paragraph: NSGA-II = optimizer; gv_stability_pareto = read-only 1D scan with dominance filter — both retained).
  - File: `scripts/stencil_gen/docs/pareto_reference.md` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from pathlib import Path; p = Path('docs/pareto_reference.md'); assert p.exists() and p.stat().st_size > 2000, 'pareto_reference.md missing or too small'"`

- [ ] **45.7b** Cross-link from existing docs:
  - In `scripts/stencil_gen/docs/optimization_reference.md`: add a "Multi-objective (plan 45)" section at the bottom pointing to `pareto_reference.md` and summarizing in 3 sentences: "For conflicting-metric trade-offs use `python -m sweeps pareto`. Scalar drivers in this reference are the building block; multi-objective wraps them into a vector-valued objective and runs NSGA-II. Fronts persist per-run under `sweeps/pareto_fronts/`."
  - In `scripts/stencil_gen/docs/brady2d_stability_reference.md`: add a line near the top noting that any field documented here is a valid element of `--objectives` in `sweeps pareto`.
  - In `docs/handoff/MASTER.md`: update section "Key artifacts to cite/reference" list to include `pareto_reference.md`.
  - File: `scripts/stencil_gen/docs/optimization_reference.md`, `scripts/stencil_gen/docs/brady2d_stability_reference.md`, `docs/handoff/MASTER.md`
  - Test: `cd scripts/stencil_gen && grep -l pareto_reference docs/*.md; grep -l pareto_reference ../../docs/handoff/MASTER.md`

- [ ] **45.7c** Add a decision entry `D-Opt-1` to `plans/meta.md` capturing four cross-cutting choices: (a) pymoo NSGA-II chosen over pure-numpy (library stability + community support); (b) per-run JSON persistence under `sweeps/pareto_fronts/` chosen over a `known_values.json["brady2d_pareto"]` key (scale, concurrency, clean git diffs); (c) finite sentinel `1e12` over `+inf` for gate-fail (pymoo hypervolume and `ftol` termination both break on inf); (d) `gate_layer` auto-infer as `max_layer - 1` applies to both scalar `make_objective` and multi-objective `make_multi_objective` (consistent behavior).
  - File: `plans/meta.md`
  - Test: `grep -q "D-Opt-1" plans/meta.md`

- [ ] **45.7d** Update `.claude/skills/stencil-sweeps/SKILL.md` with a `pareto` subcommand example and link to `pareto_reference.md`.
  - **Blocked:** `.claude/skills/` path is write-protected by tool permissions. Needs manual edit after ralph completes.
  - File: `.claude/skills/stencil-sweeps/SKILL.md`
  - Test: (no test)

- [ ] **45.7e** Update `.claude/skills/group-velocity-analysis/SKILL.md` with a note that multi-objective Pareto is now the right tool when `layer1.boundary_gv_err` disagrees with stability-layer objectives (plan 45).
  - **Blocked:** same permission constraint as 45.7d.
  - File: `.claude/skills/group-velocity-analysis/SKILL.md`
  - Test: (no test)

---

## Ordering

```
45.0a                                   # nlopt platform fix (unblocks uv sync)
  ↓
45.0b → 45.0c                           # gate_layer auto-infer + tests
  ↓
45.1a → 45.1b → 45.1c                   # pareto module: dataclasses + factory + tests
  ↓
45.2a → 45.2b → 45.2c                   # NSGA-II driver + HV callback + tests
  ↓
45.3a → 45.3b → 45.3c                   # CLI subcommand + dispatch + tests
  ↓
45.4a → 45.4b → 45.4c                   # per-run JSON persistence
  ↓
45.5a → 45.5b                           # C++ validation
  ↓
45.6a → 45.6b                           # calibration runs + regression
  ↓
45.7a → 45.7b → 45.7c → 45.7d → 45.7e   # docs + meta + skills
```

Strictly sequential within each group. 45.5 (C++ validation) can be deferred without breaking the rest; if `build/src/app/shoccs` is absent, mark 45.5a/b as conditional-skip items. 45.4 and 45.5 can swap (persistence vs. validation). 45.3c tests depend on 45.3a+b both landing.

---

## Completion Criteria

- `uv sync` succeeds on aarch64 and x86_64; `import pymoo` works; the CI smoke (`uv run pytest tests/ -x -q`) still passes in under 90 seconds (15 s budget added for new `test_pareto.py` fast tests).
- `make_objective(scheme, kernel, "layer_bl42.max_spectral_abscissa")` with no `gate_layer` kwarg returns a closure that produces finite values at known-L2-feasible, L3r-evaluable points (auto-infer working end-to-end, fixing the `+inf` trap from `docs/handoff/known_limitations.md`).
- `run_nsga2("E4", "classical", ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"], DEFAULT_BOUNDS[("E4","classical")], pop_size=12, n_gen=4, seed=1)` returns a `ParetoResult` with `len(front) >= 3` and a non-decreasing `hv_trace`.
- `python -m sweeps pareto --help` lists `--objectives`, `--pop-size`, `--n-gen`, `--seed`, `--persist`, `--validate-with-cpp` among others.
- `sweeps/pareto_fronts/` directory exists (committed via `.gitkeep`), contains 2 calibration JSONs from 45.6a, each with a non-empty `front` and a finite `hv_trace[-1]`.
- `TestRegressionBrady2DPareto` (marked `@pytest.mark.slow`) passes: every stored front member's objectives recompute within 1% of the stored values; no stored front contains dominated members.
- `scripts/stencil_gen/docs/pareto_reference.md` exists (>2 KB) and is cross-linked from `optimization_reference.md` and `brady2d_stability_reference.md`; `docs/handoff/MASTER.md` lists it.
- `plans/meta.md` contains `D-Opt-1` capturing the four decisions.
- The Pareto front for E4 classical against `[layer1.boundary_gv_err, layer_bl42.max_spectral_abscissa]` exhibits a visible trade-off: at least one member has low BL42 (`max_spectral_abscissa < 1e-10`) with relatively high GV error, and at least one has low GV error with higher BL42 — concrete visualization of the trade-off predicted in `docs/handoff/scientific_findings.md` §9.
- Skill files `.claude/skills/stencil-sweeps/SKILL.md` and `.claude/skills/group-velocity-analysis/SKILL.md` updated (harness-blocked; requires human session).
- `uv run pytest tests/ -x -q` (fast suite only) still passes cleanly in the normal time budget.
