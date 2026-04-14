# Phase 42: C++ Closed-Loop Bridge with Runtime-Parameterized Spline Stencils

**Goal:** Close the loop between the Python analytical stability stack (plan 41) and the C++ solver for the Brady-Livescu 2019 §4.3 two-dimensional varying-coefficient scalar wave test. A sweep can now vary a boundary closure parameter (sigma, epsilon, or classical alpha), run `brady2d_stability_score` through L1–L7 analytically, and — for promising survivors — validate by executing the compiled C++ simulation with the new parameters passed via the Lua config, **without rebuilding the C++ code per sweep point**. Add runtime-parameterized spline boundary families (tension, Gaussian, multiquadric) to the C++ stencil library so the families scored in plan 41 can be validated end-to-end.

**Depends on:** Phase 41 (must be complete — this phase consumes `brady2d_stability_score`, the `StabilityReport` dataclass, and `known_values.json["brady2d_calibration"]`)

**Background — key architectural observation:**

Agent C's investigation (during plan 41 design) found that the existing `E4_1.cpp` already has the runtime-parameterized pattern we need: `std::array<real, 2> alpha` is a struct field populated from Lua, and the `nbs_floating` CSE expressions treat `alpha[0]`, `alpha[1]` as runtime inputs. The same pattern can be extended to a scalar `real sigma` field for spline families. **No rebuild per sweep point** once a family's `.cpp` file is compiled.

**Critical constraint — spline coefficients are not symbolically closed-form in sigma.** The tension / Gaussian / multiquadric boundary closures require a small linear solve against the RBF kernel matrix, and the kernel involves `exp(-sigma * r)` which SymPy cannot reduce to polynomial closed form. Two viable strategies:

1. **Construction-time linear solve.** The struct constructor takes `sigma` from Lua, builds the small (r × r) kernel system in C++, solves it once via hand-coded Gaussian elimination (or Eigen's `LLT`), caches the resulting coefficients in an `std::array<real, R*T>` member, and `nbs_floating` reads the cache. **Works for uniform (non-cut-cell) boundaries** since the solve runs once per simulation.
2. **Precomputed tabulation.** Generate a `constexpr` table of coefficients at a grid of sigma values at Python codegen time, bilinearly interpolate in C++ at construction time.

Plan 42 uses strategy (1) — construction-time runtime solve — because it is exact, requires no precomputed table, and the Brady-Livescu 2D test uses a uniform (non-cut-cell) rectangular domain where the boundary closure does not depend on psi.

**What plan 42 does NOT do:**

- No cut-cell parameterization (non-uniform psi boundaries). Brady-Livescu §4.3 is uniform. Cut-cell runtime parameterization is a separate follow-up plan.
- No E6/E8 schemes. `SCHEME_PARAMS` only has E2 and E4; E6/E8 would require Python-side derivation work first.
- No optimization-inside-C++. The C++ code is a validator invoked by the Python sweep; optimization logic stays in Python.

**Read first:**

- `plans/41-brady-livescu-2d-analytical-stability.md` (every deliverable of plan 41 — this plan builds directly on `brady2d_stability_score`, `StabilityReport`, and `known_values.json["brady2d_calibration"]`)
- `src/stencils/E4_1.cpp` (the runtime-parameterized struct pattern — note lines 31–41, the constructor with `copy_zero_padded` and `alpha[1]` validity check, and the `nbs_floating` body where `alpha[0]`, `alpha[1]` appear as array subscripts inside CSE temporaries)
- `src/stencils/stencil.cpp` (the `stencil::from_lua` dispatch table at lines 8–88 — note `read_alpha` at lines 20–23 and the existing `type == "E4"` branch at line 50)
- `src/stencils/stencil.hpp` (the interface the new struct must implement: `make_stencil`, `query`, `nbs`, `interior`)
- `src/stencils/CMakeLists.txt` (the single `add_library(shoccs-stencils ...)` source list)
- `src/systems/scalar_wave.cpp` (the `neg_G_at` function at lines 28–34 — confirm it already computes `−(x−c)/|x−c|`, which is the Brady-Livescu radial flow when `center={-0.25,-0.25,0}` and `radius=0`; and `solution_at` at lines 35–40, which matches `sin(2*pi*(|x−c| − r − t))` — exactly the paper's exact solution)
- `scalar_wave.lua` (the closest-existing scalar wave Lua config — note `scheme.type`, `scheme.alpha`, `system.center`, `system.radius`, `mesh.domain_bounds`, `domain_boundaries.xmin/ymin`, `step_controller.max_time`, `integrator.type`)
- `scripts/stencil_gen/stencil_gen/codegen.py` (line 509 `generate_stencil_cpp(spec)`; line 233 `StencilGenSpec`; `param_arrays` field)
- `scripts/stencil_gen/stencil_gen/phs.py` (lines 407–500 `_rbf_weights_numeric` — the reference Python implementation of the RBF linear solve that the C++ construction-time solver must match)
- `scripts/stencil_gen/stencil_gen/brady2d_stability.py` (from plan 41 — the Python side this plan connects to)
- `scripts/stencil_gen/sweeps/known_values.json` (the `brady2d_calibration` top-level key produced by plan 41.11)

**Test commands:**

```bash
# C++ unit tests for new stencil families
ctest --test-dir build -R "t-tension_E4u_1|t-gaussian_E4u_1|t-multiquadric_E4u_1"

# Python bridge unit tests
cd scripts/stencil_gen && uv run pytest tests/test_cpp_bridge.py -x -q

# End-to-end closed-loop test (slow; runs one short simulation)
cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer8 and not test_full_simulation"

# Brady-Livescu Lua smoke (standalone variants — NOT the template, which is not directly runnable)
./build/src/app/shoccs lua-configs/brady_livescu_4_3_n61.lua   # produces logs/system.csv
./build/src/app/shoccs lua-configs/brady_livescu_4_3_long.lua  # N=31, t=100 for long-time stability
```

---

## Items

### 42.1 — Brady-Livescu Lua config (first deliverable — independent of any C++ changes)

- [x] **42.1a** Create `lua-configs/brady_livescu_4_3.lua`:
  - `mesh.index_extents = {31, 31}` (start small; N=81 is the paper's max, add a separate `_n81.lua` later)
  - `mesh.domain_bounds = {math.sqrt(2), math.sqrt(2)}`
  - `domain_boundaries = { xmin="dirichlet", ymin="dirichlet" }`
  - `shapes = {}` (no embedded geometry — this is the uniform-domain test)
  - `system = { type = "scalar wave", center = {-0.25, -0.25, 0}, radius = 0, max_error = 10.0 }`
  - `integrator = { type = "rk4" }`
  - `step_controller = { max_time = 10.0, cfl = { hyperbolic = 0.8 } }` (short time for smoke; plan 42 validates on short runs, not the 1000-period paper run)
  - `scheme = { order = 1, type = "E4u", alpha = {<classical E4 alpha from known_values.json>} }` — **correction:** `type = "E4u"` (uniform variant), not `"E4"` (cut-cell variant). `E4_1` requires `alpha[1] >= 197/288 ≈ 0.684` to avoid an interior denominator singularity, but the classical alpha values from `E4u_1.t.cpp` have `alpha[1] = 0.162`. Brady-Livescu §4.3 is uniform-domain, so `E4u` is the correct variant. All subsequent items (42.3a etc.) must also use `"E4u"` as the Lua scheme type.
  - `io = { write_every_step = 10 }`
  - File: `lua-configs/brady_livescu_4_3.lua` (new)
  - Test: `./build/src/app/shoccs lua-configs/brady_livescu_4_3.lua && python3 -c "import csv; rows=[r for r in csv.reader(open('logs/system.csv')) if r and r[0] and r[0][0].isdigit()]; assert rows, 'no data rows'; linf=float(rows[-1][3]); assert 0 < linf < 10, f'linf out of range: {linf}'; print(f'ok linf={linf}')"` → **PASSED** (`linf=0.00175` at t=10.0, N=31)
  - Note: shoccs binary lives at `build/src/app/shoccs`, not `build/shoccs`. Follow-up 42.2a/42.2b must use the correct path in `SHOCCS_BINARY`.

- [x] **42.1b** Create companion configs:
  - `lua-configs/brady_livescu_4_3_n61.lua` (N=61, same everything else)
  - `lua-configs/brady_livescu_4_3_long.lua` (N=31, `max_time = 100.0` for stability check)
  - Both reuse the same pattern; keep them as thin variants.
  - File: `lua-configs/brady_livescu_4_3_n61.lua`, `lua-configs/brady_livescu_4_3_long.lua` (new)
  - Test: `./build/src/app/shoccs lua-configs/brady_livescu_4_3_n61.lua` → **PASSED** (ran cleanly to t=10, wall=0.296s, 531 steps). Long variant also verified: runs to t=100 cleanly (wall=1.2s, 2652 steps at N=31).

### 42.2 — Python → C++ bridge (first cut: classical-alpha stencils only)

- [x] **42.2a** Create `scripts/stencil_gen/stencil_gen/cpp_bridge.py` with:
  - `REPO_ROOT: Path` computed via `Path(__file__).parents[3]`.
  - `LUA_TEMPLATE_DIR = REPO_ROOT / "lua-configs"`.
  - `BRADY_LIVESCU_TEMPLATE = LUA_TEMPLATE_DIR / "brady_livescu_4_3.lua"`.
  - `SHOCCS_BINARY = REPO_ROOT / "build" / "src" / "app" / "shoccs"` (verified from 42.1a — the binary lives under `build/src/app/`, not `build/`).
  - `@dataclass class BridgeResult: final_linf: float, linf_trace: np.ndarray, t_trace: np.ndarray, stable: bool, wall_time_s: float, exit_code: int, stderr: str`.
  - `make_brady2d_lua(scheme_type: str, params: dict, *, N: int, t_final: float, template: Path = BRADY_LIVESCU_TEMPLATE) -> str` — **strategy: explicit placeholder token substitution.** Add to 42.1a's template file the exact markers `--{{N}}--`, `--{{T_FINAL}}--`, `--{{SCHEME_TABLE}}--`; `make_brady2d_lua` does `template.read_text().replace(...)` on these three markers with the appropriate values. Returns the Lua source as a string. No regex, no Lua AST parsing.
  - File: `scripts/stencil_gen/stencil_gen/cpp_bridge.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_cpp_bridge.py -x -q -k "TestMakeBrady2DLua"` → **PASSED** (14 tests: path constants, marker substitution for alpha/sigma/epsilon params, balanced braces, BridgeResult defaults). End-to-end sanity: rendered template fed to `./build/src/app/shoccs` runs to completion with exit 0.
  - Note: params dict now recognizes three shapes — `{"alpha": [...]}` for classical schemes, `{"sigma": float}` for tension, `{"epsilon": float}` for gaussian/multiquadric. The scheme_table emitter handles all three. `scheme_type` is passed through verbatim to Lua `type=...`; mapping to Lua strings is 42.3a/42.7a's responsibility.
  - Note: classical E4u alpha (alpha[1] ≈ 0.162) violates E4_1's interior denominator constraint (alpha[1] >= 197/288 ≈ 0.684), so 42.3a's dispatch must map `("E4", "classical") → "E4u"` (NOT `"E4"`). Plan text at 42.3a currently says `"E4"`; update to `"E4u"` when implementing 42.3a. Same for 42.2c's smoke test: use `scheme_type="E4u"` or the simulation will fail.

- [x] **42.2b** Implement `run_cpp_brady2d(scheme_type: str, params: dict, *, N: int = 31, t_final: float = 10.0, timeout: float = 300.0) -> BridgeResult`:
  - Writes the Lua config to a `tempfile.TemporaryDirectory` and invokes `subprocess.run([str(binary), lua_path], cwd=tmp, capture_output=True, text=True, timeout=timeout, check=False)`.
  - On nonzero exit / timeout / missing CSV / empty CSV, returns `BridgeResult` with `final_linf=nan`, `stable=False`, and a diagnostic in `stderr`/`exit_code`.
  - **`BridgeResult.final_linf` default changed to `float("nan")`** (42.2a note resolved) — callers always get a well-formed object even on the error path.
  - Parses CSV with columns `Timestamp=0, Time=1, Step=2, Linf=3`; `stable = isfinite(final_linf) and final_linf < 10.0`.
  - **Concurrency:** each call uses its own `tempfile.TemporaryDirectory` as cwd — shoccs writes `logs/system.csv` under that tempdir, so concurrent invocations don't race on `REPO_ROOT/logs/`. A dedicated test (`test_run_is_isolated_to_tempdir`) guards this invariant.
  - Also exposes `binary` and `template` keyword arguments so tests can swap in a fake shoccs script.
  - File: `scripts/stencil_gen/stencil_gen/cpp_bridge.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_cpp_bridge.py -x -q -k "TestRunCppBrady2D"` → **PASSED** (7 tests: success, nonzero exit, unstable Linf, missing CSV, empty CSV, timeout, tempdir isolation). Full `test_cpp_bridge.py` run: 22 passed.

- [x] **42.2c** Smoke test `TestCppBridgeSmoke`:
  - Skip if `SHOCCS_BINARY` does not exist (`pytest.skip("shoccs binary not built")`).
  - `test_classical_e4u_short_run` — `run_cpp_brady2d(scheme_type="E4u", params={"alpha": [-0.7733323791884821, 0.1623961700641681]}, N=21, t_final=1.0)` returns `stable=True` and `final_linf < 1.0`. **Correction (from 42.2a's note):** use `scheme_type="E4u"` (uniform variant), NOT `"E4"` — the classical alpha values in `known_values.json` have `alpha[1] ≈ 0.162` which violates `E4_1`'s interior denominator constraint (`alpha[1] >= 197/288 ≈ 0.684`) and would abort the simulation. Brady-Livescu §4.3 is uniform-domain.
  - Mark `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/tests/test_cpp_bridge.py` (new)
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_cpp_bridge.py -x -q -k "TestCppBridgeSmoke" --run-slow` → **PASSED** (1 test; real shoccs run at N=21, t_final=1.0 completes in ~0.15 s per invocation). Full `test_cpp_bridge.py` with `--run-slow`: 23 passed.

### 42.3 — L8 integration: C++ simulation as the final validation layer

- [x] **42.3a** Add `layer8_cpp_simulation(scheme: str, kernel: str, params: dict, *, N: int = 31, t_final: float = 10.0) -> dict` to `brady2d_stability.py`:
  - Maps `(scheme, kernel)` → Lua `scheme.type` string. For plan 42 first cut, only `("E4", "classical")` → `"E4u"` is supported (uniform variant — see 42.2a's note for the constraint); other kernels raise `NotImplementedError` (filled in by 42.5+).
  - Calls `run_cpp_brady2d` with the appropriate params.
  - Returns `{final_linf, stable, wall_time_s, bridge_result}` — includes the full `BridgeResult` so callers can inspect `exit_code`/`stderr`/traces on failure without a second call.
  - Layer-8 failure: `not stable` OR `final_linf > 1.0` at `t_final=10.0`. Threshold exposed as `L8_FINAL_LINF_TOL = 1.0` module constant (consumed by 42.3b).
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer8 and classical"` → **PASSED** (1 fast dispatch test + 1 slow end-to-end smoke skipped without `--run-slow`). Full `TestLayer8` class with `--run-slow`: 7 passed (dispatch, unstable propagation, unsupported-kernel/scheme NotImplementedError, threshold constant check, defaults, real shoccs E4u short run with `final_linf<1.0`).

- [x] **42.3b** Extend `brady2d_stability_score` to accept `max_layer=8` and the `StabilityReport` dataclass to carry a `layer8: dict | None` field:
  - When `max_layer >= 8` and earlier layers pass, call `layer8_cpp_simulation`.
  - Layer-8 failure sets `failed_layer=8`, `failed_reason=...`.
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestStabilityScoreL8"` → **PASSED** (8 tests: dispatch to L8 on pass, forwarding of `layer8_N`/`layer8_t_final`, `failed_layer=8` on `stable=False`, `failed_layer=8` on `final_linf > L8_FINAL_LINF_TOL`, L8 skipped under short-circuit when earlier layer fails, L8 not run at `max_layer=7`, `__str__` L8 pass/fail lines). `TestStabilityReport` default-values test updated to assert `layer8 is None`; `TestStabilityScoreOrchestrator` still passes.
  - Note: added `layer8_N: int = 31` and `layer8_t_final: float = 10.0` kwargs to `brady2d_stability_score` so callers can configure the C++ run without a separate wrapper (used by 42.3c's integration test at N=21, t_final=1.0).

- [x] **42.3c** Integration test `TestBrady2DL8ClassicalE4`:
  - Runs `brady2d_stability_score(scheme="E4", kernel="classical", params={"alpha": [...]}, max_layer=8)` end-to-end.
  - Asserts `overall_verdict == "pass"`, `failed_layer is None`, `layer8.stable is True`.
  - Mark `@pytest.mark.slow` (runs a ~30 s C++ simulation at N=21, t_final=5).
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestBrady2DL8ClassicalE4" --run-slow` → **PASSED** (1 slow test; real shoccs run at N=21, t_final=1.0 completes in ~20 s including layer 1–7 analytic work. Classical E4u short run lands `final_linf < L8_FINAL_LINF_TOL` cleanly).
  - Note: used `layer8_N=21, layer8_t_final=1.0` (not the plan's draft t_final=5) to keep the slow test under 30 s; the orchestrator keywords added in 42.3b make this ergonomic.

### 42.4 — Codegen: add scalar runtime parameter support to `StencilGenSpec`

- [x] **42.4a** Add `scalar_params: list[str] = field(default_factory=list)` to `StencilGenSpec` in `scripts/stencil_gen/stencil_gen/codegen.py:233`:
  - After the existing `param_arrays` field declaration.
  - Update the class docstring noting scalar vs array distinction.
  - File: `scripts/stencil_gen/stencil_gen/codegen.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_codegen.py -x -q -k "TestStencilGenSpec"` → **PASSED** (3 tests: default empty, accepts list, per-instance independence via `field(default_factory=list)`). Full `tests/test_codegen.py` still green (32 passed).
  - Note: placed `scalar_params` at the end of the optional-defaults block (after `interp_T`) rather than literally adjacent to `param_arrays`, because inserting a `field(...)`-default field between the non-default required fields and the `has_interp=False` defaults would violate dataclass ordering. Semantically it is still the "companion to `param_arrays`" and the docstring now documents the pairing explicitly.

- [x] **42.4b** Extend `_emit_struct_preamble` to emit `real {name};` for each `scalar_params` entry:
  - After the `param_arrays` emission block, add a loop over `spec.scalar_params` emitting a `real` field declaration.
  - Update constructor emission: when `scalar_params` is non-empty, emit a constructor overload `StructName(real {param0}, real {param1}, ...)`.
  - **Test scope limited to struct preamble only.** The `TestScalarParamsEmission` test must assert only that the emitted C++ text contains `"real sigma;"` as a field and a matching constructor signature string — **do not** assert on any expression body or interior coefficients that would depend on 42.4c's symbol-map update. Those are validated in 42.4d's end-to-end test.
  - File: `scripts/stencil_gen/stencil_gen/codegen.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_codegen.py -x -q -k "TestScalarParamsEmission"` → **PASSED** (6 tests: single/multiple scalar field emission, single/multiple constructor signature, default ctor preserved, no-scalars negative check). Full `tests/test_codegen.py` still green (38 passed).
  - Note: scalar constructor emits body-assignment (`sigma = sigma_;`) rather than an init list, matching the `copy_zero_padded` body style already used for array-param constructors. Parameter is named `{name}_` to avoid shadowing the member.

- [x] **42.4c** Extend `build_symbol_map` in `scripts/stencil_gen/stencil_gen/printer.py:68` to accept `scalar_params: list[str]`:
  - For each scalar param name, map `Symbol(name) → name` (no subscript).
  - Update callers in `codegen.py` to pass `spec.scalar_params` through.
  - File: `scripts/stencil_gen/stencil_gen/printer.py`, `scripts/stencil_gen/stencil_gen/codegen.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_printer.py -x -q -k "TestScalarParams"` → **PASSED** (6 tests: single/multiple scalars, alongside array, no-subscript regression, default `None`, default-omitted backward-compat). Full `test_printer.py` + `test_codegen.py` still green (54 passed).
  - Note: `scalar_params` is keyword-only with default `None` (treated as empty) to preserve the existing two-positional-arg call site in `generate_stencil_cpp` prior to this change; callers new and old both work. `codegen.py:547` now passes `spec.scalar_params` as a keyword argument alongside `has_psi=not spec.is_uniform`.

- [x] **42.4d** Tests `TestScalarParamsCodegenEndToEnd`:
  - Build a synthetic `StencilGenSpec(name="TestStruct", scalar_params=["sigma"], interior_coeffs=[sympy.Symbol("sigma") * sympy.Symbol("h")], ...)`.
  - Call `generate_stencil_cpp(spec)` and assert the output contains `real sigma;` as a field, a constructor taking `real sigma`, and `sigma` (not `sigma[0]`) inside the interior body.
  - File: `scripts/stencil_gen/tests/test_codegen.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_codegen.py -x -q -k "TestScalarParamsCodegenEndToEnd"` → **PASSED** (3 tests: field + ctor + printer subscript-free body together, scalar inside compound expression, two-scalar parallel path). Full `test_codegen.py` still green (41 passed).
  - Note: plan text sketched putting `sigma*h` in `interior_coeffs`, but `generate_interior_method` uses the `Rational(c)` / `float(c)` fast paths and cannot accept free symbols. The expression path that actually uses `StencilCodePrinter` (and so the scalar symbol map wired in 42.4c) is `nbs_floating` / `nbs_dirichlet` — the test places the scalar symbol there. Test-scope docstring documents this divergence.

### 42.5 — First spline family in C++: `tension_E4u_1` (construction-time runtime solve, hand-written)

The strategy: the constructor takes `real sigma` from Lua, builds the small (r=5, t=7) tension-spline RBF system in C++ using a hand-coded Gaussian elimination specialized to the tension kernel, caches the resulting 5×7 boundary coefficient matrix in a struct field, and `nbs_floating` reads from the cache. The reference implementation is Python's `phs._rbf_weights_numeric(..., kernel="tension")`.

**Note:** The Python-emits-C++ codegen path (originally scoped as 42.5a / 42.5e) is explicitly deferred to a follow-up plan. Plan 42 ships with a hand-written `tension_E4u_1.cpp`. Rationale: emitting a correct Gaussian elimination as C++ text from Python adds significant scope and the hand-written version is a one-time effort per kernel.

- [x] **42.5a** Generate the Python reference coefficient array for `tension_E4u_1` at `sigma=3.0`:
  - Run `cd scripts/stencil_gen && uv run python -c "from stencil_gen.phs import build_diff_matrix_rbf; import numpy as np; D = build_diff_matrix_rbf(n=20, p=2, q=3, epsilon=3.0, kernel='tension', nu=1, nextra=0); np.set_printoptions(precision=17, suppress=False); print(repr(D[:5, :7]))"` to obtain the 5×7 boundary coefficient block at h=1/(n-1). Capture the exact numeric output.
  - Add the captured array as a constant `REFERENCE_TENSION_E4U1_SIGMA3_COEFFS` in a new test helper file (e.g. `scripts/stencil_gen/tests/fixtures/tension_e4u1_reference.py`) for later reuse. Also hand-paste it as a `std::array<real, 35>` into the Catch2 test file in 42.5f.
  - File: `scripts/stencil_gen/tests/fixtures/__init__.py` (new), `scripts/stencil_gen/tests/fixtures/tension_e4u1_reference.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from tests.fixtures.tension_e4u1_reference import REFERENCE_TENSION_E4U1_SIGMA3_COEFFS; print(len(REFERENCE_TENSION_E4U1_SIGMA3_COEFFS))"` → **PASSED** (shape=(5,7), size=35; live build_diff_matrix_rbf regeneration matches fixture to ~7e-18 max abs error).
  - Note: row 4 of the 5×7 block is the classical E4 centered stencil `[0, 0, 1/12, -2/3, 0, 2/3, -1/12]` (row index 4 falls in the interior region of build_diff_matrix_rbf, which only decorates rows 0..3 with RBF+polynomial augmentation for `q=3`). The plan's spec of r=5 boundary rows is preserved — row 4 is cached alongside the true boundary rows and the C++ will simply copy it into the output span.

> **Corrected dimension context (from 42.5a review):** The actual `build_diff_matrix_rbf(p=2, q=3, nu=1, nextra=0)` geometry is `r=4, t=6` (via `compute_dimensions`), not `r=5, t=7` as originally drafted. The solver therefore produces a 4×6 = 24-entry block (rows 0..3 × cols 0..5). The fixture (and the C++ `cached_coeffs`) pad this to 5×7 = 35 entries by appending: (a) row 4 = the classical E4 centered stencil at h=1, `[0, 0, 1/12, -2/3, 0, 2/3, -1/12]`, hardcoded; (b) col 6 = 0 for rows 0..3. Keep the `cached_coeffs` layout at 5×7 so the unit test in 42.5f can compare byte-for-byte against the fixture, but do not route row 4 or col 6 (rows 0..3) through the runtime solver. 42.6a/b/e/f inherit the same correction.
>
- [x] **42.5b** Create `src/stencils/tension_E4u_1.cpp` struct skeleton (no solver body yet):
  - Struct layout mirroring `E4u_1` but with `real sigma` field replacing `std::array<real, 2> alpha`, and a new `std::array<real, 5 * 7> cached_coeffs` member (5 rows × 7 cols; see the corrected-dimension note above — row 4 is the hardcoded classical E4 interior stencil, col 6 is zero for rows 0..3).
  - Constructor `tension_E4u_1(real sigma_in) : sigma(sigma_in) { /* solver call inserted in 42.5d */ }`.
  - `interior`, `query`, `query_max`, method stubs for `nbs_floating` and `nbs` matching `E4u_1`'s signatures but with empty (TODO) bodies that simply fill the output span with zeros and return it.
  - Does **not** register in `stencil.cpp` yet — that's 42.5e.
  - **Must compile** once added to CMakeLists in 42.5c.
  - File: `src/stencils/tension_E4u_1.cpp` (new)
  - Test: `cmake --build build --target shoccs-stencils` → **PASSED** (compiles cleanly as part of `libshoccs-stencils.a`).
  - Note: used `P=2, R=5, T=7, X=0` (matching E4_1's boundary region sizing) so `nbs_floating` writes 35 = R*T entries, which matches the `cached_coeffs` layout specified in 42.5a/42.5d. `interior()` is hardcoded to the classical E4 stencil (same coefficients as `E4u_1::interior`), since the boundary closure is the only RBF-specific piece. Skeleton methods zero-fill the output span; solver body comes in 42.5d. Struct is not yet a `stencil`-concept satisfying type that `from_lua` can dispatch to — that wiring is 42.5e.

- [x] **42.5c** Add `tension_E4u_1.cpp` to `src/stencils/CMakeLists.txt` build list:
  - Append `tension_E4u_1.cpp` to the `add_library(shoccs-stencils ...)` source list. No test target yet.
  - Do not yet register in `stencil::from_lua` (42.5e).
  - File: `src/stencils/CMakeLists.txt`
  - Test: `cmake --build build --target shoccs-stencils` → **PASSED** (single-file incremental build: `Building CXX object .../tension_E4u_1.cpp.o` → `Linking CXX static library libshoccs-stencils.a`, no warnings from the new file).

- [x] **42.5d** Implement `solve_tension_coefficients(real sigma, std::array<real, 5*7>& out)` as a static/anonymous-namespace helper in `tension_E4u_1.cpp`:
  - Builds the **6×6** tension-spline kernel matrix entries `K[i,j] = sigma*|x_i - x_j| - 1 + exp(-sigma*|x_i - x_j|)` where `x_i = i` for i in 0..5 (reference grid, h=1, t=6 collocation points).
  - Augments with the polynomial block: **4 polynomial rows and 4 polynomial columns** for q=3 (monomials `1, x, x^2, x^3`), matching `phs._rbf_weights_numeric`'s construction — augmented system is **(t + q+1) × (t + q+1) = 10 × 10**.
  - For each boundary row `i` in 0..3 (r=4 boundary rows), build the RHS `[D^1 φ(i - x_j) for j in 0..5 ; D^1 x^k at x=i for k in 0..3]` and solve the 10×10 system via plain Gaussian elimination with partial pivoting (fits on the stack). Extract the first 6 components of the solution as the boundary-row weights `w_ij` for j in 0..5.
  - Layout into the 5×7 `out` buffer (row-major, `out[i*7 + j]`):
    - Rows 0..3, cols 0..5: the solved RBF+poly weights `w_ij` at h=1.
    - Rows 0..3, col 6: `0.0` (zero-padding; fixture captures these as exact zeros because `build_diff_matrix_rbf` only writes t=6 columns per boundary row).
    - Row 4 (interior, not from the solver): hardcoded classical E4 first-derivative stencil `{0, 0, 1.0/12.0, -2.0/3.0, 0, 2.0/3.0, -1.0/12.0}` at h=1.
  - Wire the call into the `tension_E4u_1` constructor (replacing the 42.5b stub comment). Row 4 and col 6 of rows 0..3 can be written unconditionally before the solver runs.
  - Update `nbs_floating` to copy `cached_coeffs` into the output span, apply `1/h` scaling, then the `right` flip logic matching `E4u_1.cpp:~103`.
  - File: `src/stencils/tension_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs-stencils` → **PASSED** (clean incremental build, no warnings). Standalone numerical check: reproducing the solver in a throwaway `main` at sigma=3.0 produces the 5×7 block matching `REFERENCE_TENSION_E4U1_SIGMA3_COEFFS` to ≥14 significant digits row-for-row (byte-for-byte equality on rows 2..4). Full fixture-vs-solver assertion is the job of 42.5f's Catch2 test.
  - Note: `nbs_floating` copies the cached 35-entry block, divides by h, then applies the `right` flip (negate + reverse) matching `E4u_1.cpp:97`. `nbs_dirichlet` drops the first row (cols 0..6) of the cached block and applies the same h scaling + right flip, giving the 4×7=28-entry Dirichlet closure consistent with `E4u_1.cpp:105`'s "drop first row" pattern. Shared Taylor-vs-direct kernel evaluation mirrors `phs._tension_kernel_eval`/`_deriv` (z<2 threshold) so the C++ matches Python across the full sigma range the bridge will use.

- [x] **42.5e** Register `tension_E4u_1` in `stencil.hpp` and `stencil.cpp`:
  - Add `stencil make_tension_E4u_1(real sigma)` factory declaration in `stencil.hpp`.
  - In `stencil::from_lua` at `stencil.cpp`, **insert a new `else if (type == "tension_E4u")` branch immediately after the `type == "E8u"` branch and before the final `logger(...err...)` fallthrough**. The branch reads `real sigma = m["sigma"].get_or(3.0)` and calls `make_tension_E4u_1(sigma)`. Verify no existing branches are affected by diffing the file after the edit.
  - Also add a `make_tension_E4u_1` factory definition at the bottom of `tension_E4u_1.cpp`.
  - File: `src/stencils/stencil.hpp`, `src/stencils/stencil.cpp`, `src/stencils/tension_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs` → **PASSED** (full chain compiles and links; `libshoccs-stencils.a` rebuilds and `src/app/shoccs` relinks with no warnings). End-to-end smoke: a minimal 2D scalar-wave Lua with `scheme = { order=1, type="tension_E4u", sigma=3.0 }` runs to `t=1.0` at N=21, producing log line `builder: tension_E4u first scheme chosen (sigma = 3)` and completing cleanly (wall=0.79s).
  - Note: branch order in `stencil.cpp:57-64` — appended to the `order == 1` `else if` chain after `E8u` so the new type is only matched when `order == 1` (matching plan 42's uniform-1st-derivative scope). Factory at `tension_E4u_1.cpp:278` is a one-liner mirroring `make_E4u_1`.

- [x] **42.5f** Add unit test `t-tension_E4u_1`:
  - Add `add_unit_test(tension_E4u_1 "stencils" shoccs-stencils)` to `src/stencils/CMakeLists.txt`.
  - Create `src/stencils/tension_E4u_1.t.cpp` with three Catch2 tests:
    - `TEST_CASE("tension_E4u_1 construction at sigma=3")` — instantiate at `sigma=3.0`, verify no exception.
    - `TEST_CASE("tension_E4u_1 coefficients match Python reference")` — hard-code the 35-entry reference array from 42.5a's fixture as a `std::array<real, 35>`, assert each cached_coeffs entry matches within `1e-12`.
    - `TEST_CASE("tension_E4u_1 nbs_floating fills output span")` — call `nbs_floating(h=0.1, psi=1.0, c, right=false)` with a 35-element output buffer, verify the returned span has size 35 and all entries are finite.
  - File: `src/stencils/CMakeLists.txt`, `src/stencils/tension_E4u_1.t.cpp` (new)
  - Test: `cmake --build build --target t-tension_E4u_1 && ctest --test-dir build -R t-tension_E4u_1` → **PASSED** (3 Catch2 cases; test executable reports `All tests passed (assertions in 3 test cases)` and ctest reports `100% tests passed, 0 tests failed out of 1` in 0.01 s).
  - Note: the test drives the stencil through `from_lua` (mirroring `E4u_1.t.cpp`'s pattern) rather than instantiating the `tension_E4u_1` struct directly, because the struct lives in the .cpp's anonymous-namespace scope and is not exposed in the header. With `h=1.0` and `right=false`, `nbs_floating` copies `cached_coeffs` verbatim into the output span, so the reference-match test at `h=1.0` equivalently asserts `cached_coeffs == REFERENCE_SIGMA3_COEFFS` within `1e-12`. The third test additionally spot-checks the `1/h` scaling at `h=0.1` on the classical-interior row (row 4) so regressions in the `nbs_floating` transform get caught without rederiving the whole block at a second h.

- [x] **42.5g** Extend `t-tension_E4u_1` to cover the `Dirichlet` and `right=true` paths:
  - Brady-Livescu §4.3 uses `xmin/ymin = "dirichlet"` (and mirrored max faces), so `nbs_dirichlet` and the `right=true` negate+reverse branch in `nbs_floating`/`nbs_dirichlet` (`tension_E4u_1.cpp:251-253, 266-268`) are on this phase's hot path but currently untested. 42.6d/42.6h clone these tests, so closing the gap here prevents it from propagating to gaussian/multiquadric.
  - Add `TEST_CASE("tension_E4u_1 Dirichlet query and nbs at sigma=3")`:
    - Assert `query(bcs::Dirichlet)` returns `(p=2, r=4, t=7, x=0)`.
    - Call `nbs(h=1.0, bcs::Dirichlet, psi=1.0, right=false, c, ex)` with a 28-entry buffer; assert the 28 output values equal `REFERENCE_SIGMA3_COEFFS[7..34]` (rows 1..4 of the 5×7 block, matching `nbs_dirichlet`'s "drop first row" behavior) within `1e-12`.
  - Add `TEST_CASE("tension_E4u_1 right=true flips Floating block")`:
    - Call `nbs(h=1.0, bcs::Floating, psi=1.0, right=true, c, ex)` with a 35-entry buffer; assert the output equals `-REFERENCE_SIGMA3_COEFFS` reversed end-to-end within `1e-12` (i.e. `c[i] == -REFERENCE_SIGMA3_COEFFS[34 - i]`).
  - File: `src/stencils/tension_E4u_1.t.cpp`
  - Test: `cmake --build build --target t-tension_E4u_1 && ctest --test-dir build -R t-tension_E4u_1` → **PASSED** (5 Catch2 cases; ctest reports `100% tests passed, 0 tests failed out of 1` in 0.01 s). Dirichlet case asserts `query` returns `(2, 4, 7, 0)` and the 28-entry `nbs` output matches `REFERENCE_SIGMA3_COEFFS[7..35]`; `right=true` case builds the negated-reversed expected block via an explicit loop and asserts elementwise within `1e-12`.

### 42.6 — Second and third spline families: `gaussian_E4u_1`, `multiquadric_E4u_1`

Each family follows the same 4-item pattern as 42.5b–f (minus the split reference-generation step, which is combined into the first item per family). The kernel function changes; everything else mirrors `tension_E4u_1`.

- [x] **42.6a** Generate the Python reference for `gaussian_E4u_1` at `epsilon=0.9` and create `src/stencils/gaussian_E4u_1.cpp` skeleton + CMake registration:
  - Generate reference: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.phs import build_diff_matrix_rbf; import numpy as np; D = build_diff_matrix_rbf(n=20, p=2, q=3, epsilon=0.9, kernel='gaussian', nu=1, nextra=0); np.set_printoptions(precision=17); print(repr(D[:5, :7]))"`.
  - Add as `REFERENCE_GAUSSIAN_E4U1_EPS09_COEFFS` to `tests/fixtures/gaussian_e4u1_reference.py`.
  - Create `src/stencils/gaussian_E4u_1.cpp` as a clone of `tension_E4u_1.cpp` with `real epsilon` instead of `real sigma` and empty solver (stub).
  - Append `gaussian_E4u_1.cpp` to `src/stencils/CMakeLists.txt`.
  - File: `scripts/stencil_gen/tests/fixtures/gaussian_e4u1_reference.py` (new), `src/stencils/gaussian_E4u_1.cpp` (new), `src/stencils/CMakeLists.txt`
  - Test: `cmake --build build --target shoccs-stencils` → **PASSED** (clean incremental build: `Building CXX object .../gaussian_E4u_1.cpp.o` → `Linking CXX static library libshoccs-stencils.a`, no compiler warnings). Fixture regenerates exactly (max_abs_diff = 0.0) against `build_diff_matrix_rbf(kernel='gaussian', epsilon=0.9)`.
  - Note: skeleton stub `solve_gaussian_coefficients` zero-fills rows 0..3 and hardcodes the row-4 classical E4 stencil; the real solve body comes in 42.6b. Struct is not yet dispatchable from Lua — registration is 42.6c. `<cstddef>` include was omitted from the skeleton (no `std::size_t` references yet); 42.6b adds it back when `gauss_solve<>` is introduced.

- [x] **42.6b** Implement `solve_gaussian_coefficients(real epsilon, std::array<real, 5*7>& out)` in `gaussian_E4u_1.cpp`:
  - Same 10×10 augmented system and 5×7 output layout as `solve_tension_coefficients` (see corrected dimension note above 42.5b — 6-point kernel + 4-column polynomial augmentation; rows 0..3 × cols 0..5 are solved, row 4 is the hardcoded classical E4 stencil, col 6 of rows 0..3 is zero).
  - Kernel function: `gaussian_kernel(r, eps) = exp(-(eps*r)*(eps*r))`. RHS uses `D^1 φ(i - x_j) = -2*eps^2*(i-x_j) * exp(-(eps*(i-x_j))^2)`.
  - Wire into the constructor; populate `nbs_floating` to read from `cached_coeffs`.
  - File: `src/stencils/gaussian_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs-stencils` → **PASSED** (clean incremental build, no warnings). Standalone numerical check: reproducing the solver in a throwaway `main` at `epsilon=0.9` yields max abs diff of `1.8e-15` vs `REFERENCE_GAUSSIAN_E4U1_EPS09_COEFFS` across all 35 entries. `nbs_floating` was already wired to copy `cached_coeffs` in 42.6a's skeleton, so no further changes needed. Full fixture-vs-solver Catch2 assertion is 42.6d's job.
  - Note: structure is a direct clone of `solve_tension_coefficients` with the kernel functions swapped. `gaussian_phi(r, eps) = exp(-(eps*r)^2)` and `gaussian_dphi(r, eps) = -2*eps^2*r*exp(-(eps*r)^2)` — no small-z Taylor path needed because the Gaussian is smooth and well-conditioned at r=0 (unlike tension, which has z<2 Taylor for accuracy). The `gauss_solve<N, NRHS>` template is duplicated across `tension_E4u_1.cpp` and `gaussian_E4u_1.cpp` (and will be duplicated again in 42.6f); factoring to a shared header is out of scope for plan 42 — its in-anonymous-namespace placement means there's no ODR risk.

- [x] **42.6c** Register `gaussian_E4u_1` in `stencil.hpp` / `stencil.cpp` dispatch (type string `"gaussian_E4u"`, reading `real epsilon = m["epsilon"].get_or(0.9)`, inserted immediately after the `tension_E4u` branch from 42.5e):
  - File: `src/stencils/stencil.hpp`, `src/stencils/stencil.cpp`, `src/stencils/gaussian_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs-exe` → **PASSED** (clean full link of `src/app/shoccs` after incremental compile of `gaussian_E4u_1.cpp` + `stencil.cpp`). End-to-end smoke: a minimal 2D scalar-wave Lua with `scheme = { order=1, type="gaussian_E4u", epsilon=0.9 }` runs to `t=1.0` at N=21, emitting `builder: gaussian_E4u first scheme chosen (epsilon = 0.9)` and completing cleanly in 1.7 s.
  - Note: dispatch branch inserted in `stencil.cpp` immediately after `tension_E4u`; factory `make_gaussian_E4u_1` declared in `stencil.hpp:283` (after `make_tension_E4u_1`) and defined at `gaussian_E4u_1.cpp:239` before the closing `} // namespace ccs::stencils`, mirroring the `tension_E4u_1.cpp:277` pattern. Unit tests covering this dispatch (via `from_lua`) are 42.6d's job.

- [x] **42.6d** Add `t-gaussian_E4u_1` unit test mirroring `t-tension_E4u_1` (including the 42.5g Dirichlet + `right=true` cases):
  - Append `add_unit_test(gaussian_E4u_1 "stencils" shoccs-stencils)` to CMakeLists.
  - Create `src/stencils/gaussian_E4u_1.t.cpp` with the five Catch2 tests from 42.5f + 42.5g, hard-coded against the 42.6a reference.
  - File: `src/stencils/CMakeLists.txt`, `src/stencils/gaussian_E4u_1.t.cpp` (new)
  - Test: `cmake --build build --target t-gaussian_E4u_1 && ctest --test-dir build -R t-gaussian_E4u_1` → **PASSED** (5 Catch2 cases; ctest reports `100% tests passed, 0 tests failed out of 1` in 0.01 s). Mirrors `tension_E4u_1.t.cpp` exactly with `epsilon=0.9` fixture substituted for `sigma=3.0` and the Lua script using `type="gaussian_E4u"` + `epsilon = ...`.

- [x] **42.6e** Generate the Python reference for `multiquadric_E4u_1` at `epsilon=1.0` and create `src/stencils/multiquadric_E4u_1.cpp` skeleton + CMake registration (mirrors 42.6a):
  - Generate reference via the same one-liner with `kernel='multiquadric'`, store in `tests/fixtures/multiquadric_e4u1_reference.py`.
  - Clone skeleton from `gaussian_E4u_1.cpp`, rename struct and kernel function.
  - File: `scripts/stencil_gen/tests/fixtures/multiquadric_e4u1_reference.py` (new), `src/stencils/multiquadric_E4u_1.cpp` (new), `src/stencils/CMakeLists.txt`
  - Test: `cmake --build build --target shoccs-stencils` → **PASSED** (clean incremental build: `Building CXX object .../multiquadric_E4u_1.cpp.o` → `Linking CXX static library libshoccs-stencils.a`, no compiler warnings). Fixture regenerates against `build_diff_matrix_rbf(kernel='multiquadric', epsilon=1.0)` with max abs diff = 6.9e-18.
  - Note: skeleton stub `solve_multiquadric_coefficients` zero-fills rows 0..3 and hardcodes the row-4 classical E4 stencil; the real solve body comes in 42.6f. Struct is not yet dispatchable from Lua — registration is 42.6g. No `make_multiquadric_E4u_1` factory yet; that lands with 42.6g alongside the `stencil.hpp` declaration, following the 42.6a→42.6c pattern.

- [x] **42.6f** Implement `solve_multiquadric_coefficients` with kernel `sqrt(1 + (eps*r)^2)` and RHS `D^1 φ(i - x_j) = eps^2*(i - x_j) / sqrt(1 + (eps*(i - x_j))^2)`; wire into constructor (mirrors 42.6b — same 10×10 system and 5×7 layout with row 4 hardcoded and col 6 zero for rows 0..3):
  - File: `src/stencils/multiquadric_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs-stencils` → **PASSED** (clean incremental build of `multiquadric_E4u_1.cpp` + relink of `libshoccs-stencils.a`, no warnings). Standalone numerical check: reproducing the solver in Python at `epsilon=1.0` yields max abs diff of `1.7e-15` vs `REFERENCE_MULTIQUADRIC_E4U1_EPS1_COEFFS` across all 35 entries. `nbs_floating` already reads from `cached_coeffs` (wired in 42.6e's skeleton). Full fixture-vs-solver Catch2 assertion is 42.6h's job.
  - Note: structure is a direct clone of `solve_gaussian_coefficients` with kernel swapped to `multiquadric_phi(r,eps) = sqrt(1 + (eps*r)^2)` and `multiquadric_dphi(r,eps) = eps^2 * r / sqrt(1 + (eps*r)^2)`. Like gaussian, no small-r Taylor path needed (kernel is smooth at r=0 — `phi(0)=1`, `dphi(0)=0`). The `gauss_solve<N,NRHS>` template is once again duplicated in-anonymous-namespace; shared-header refactor remains out of scope for plan 42.

- [x] **42.6g** Register `multiquadric_E4u_1` in `stencil.hpp` / `stencil.cpp` dispatch (type `"multiquadric_E4u"`, default `epsilon=1.0`, inserted after `gaussian_E4u` branch):
  - File: `src/stencils/stencil.hpp`, `src/stencils/stencil.cpp`, `src/stencils/multiquadric_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs` → **PASSED** (full chain compiles and links; `libshoccs-stencils.a` rebuilds and `src/app/shoccs` relinks with no warnings). End-to-end smoke: a minimal 2D scalar-wave Lua with `scheme = { order=1, type="multiquadric_E4u", epsilon=1.0 }` runs to `t=1.0` at N=21, emitting `builder: multiquadric_E4u first scheme chosen (epsilon = 1)` and completing cleanly.
  - Note: dispatch branch inserted in `stencil.cpp` immediately after `gaussian_E4u`; factory `make_multiquadric_E4u_1` declared in `stencil.hpp` (after `make_gaussian_E4u_1`) and defined at the end of `multiquadric_E4u_1.cpp` before the closing `} // namespace ccs::stencils`, mirroring the gaussian pattern. Unit tests covering this dispatch (via `from_lua`) are 42.6h's job.

- [x] **42.6h** Add `t-multiquadric_E4u_1` unit test (mirrors 42.6d — five Catch2 tests including the Dirichlet + `right=true` cases from 42.5g):
  - File: `src/stencils/CMakeLists.txt`, `src/stencils/multiquadric_E4u_1.t.cpp` (new)
  - Test: `cmake --build build --target t-multiquadric_E4u_1 && ctest --test-dir build -R t-multiquadric_E4u_1` → **PASSED** (5 Catch2 cases; ctest reports `100% tests passed, 0 tests failed out of 1` in 0.01 s). Clone of `gaussian_E4u_1.t.cpp` with `epsilon=1.0` fixture substituted and Lua script using `type="multiquadric_E4u"`.

### 42.7 — Closed-loop bridge for spline families

- [x] **42.7a** Extend `layer8_cpp_simulation` in `brady2d_stability.py` to handle the new families:
  - Dispatch table:
    - `("E4", "classical")` → Lua `type="E4u"`, pass `alpha`
    - `("E4", "tension")` → Lua `type="tension_E4u"`, pass `sigma`
    - `("E4", "gaussian")` → Lua `type="gaussian_E4u"`, pass `epsilon`
    - `("E4", "multiquadric")` → Lua `type="multiquadric_E4u"`, pass `epsilon`
  - E2 variants deferred (pattern is identical; do in a follow-up phase if needed).
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer8Dispatch"` → **PASSED** (4 tests: 3 parametrized spline-kernel dispatch cases + classical regression). `_scheme_table_for` in `cpp_bridge.py` already routed `sigma`/`epsilon` through to Lua, so 42.7a is purely a dispatch-table addition. Existing `test_unsupported_kernel_raises` changed its example from `tension` to `bogus` since `tension` is now a supported dispatch key. Full `test_brady2d_stability.py` run: 79 passed, 8 skipped.
  - Note: the plan's draft dispatch table said `("E4", "classical") → "E4"`; corrected to `"E4u"` to match 42.3a's existing mapping. The `("E4", "tension") → "tension_E4u"` line is wired via the same `_scheme_table_for` shape already exercised by 42.2a's placeholder tests, so 42.7b is effectively a no-op extension (scheme_table emitter already handles sigma/epsilon).

- [x] **42.7b** Extend `make_brady2d_lua` in `cpp_bridge.py` to emit the new scheme tables:
  - For `tension_E4u` and `gaussian_E4u`/`multiquadric_E4u`, emit `scheme = { order = 1, type = "<name>", sigma = <val> }` or `epsilon = <val>` respectively.
  - File: `scripts/stencil_gen/stencil_gen/cpp_bridge.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_cpp_bridge.py -x -q -k "TestMakeBrady2DLuaSpline"` → **PASSED** (8 tests: parametrized sigma/epsilon emission across all three spline families, balanced-brace sanity for each family, float-precision preservation via `repr()`). Full `test_cpp_bridge.py` run: 30 passed, 1 skipped (slow smoke).
  - Note: as 42.7a predicted, the emitter itself (`_scheme_table_for` in `cpp_bridge.py`) already handled `sigma`/`epsilon` from 42.2a — 42.7b was purely a test addition confirming the end-to-end rendering contract. Also hardened the test that the non-selected scalar ("sigma" when passing epsilon, vice versa) is never emitted in the scheme slice, and that the order=1 declaration survives the substitution.

- [x] **42.7c** Integration test `TestLayer8EndToEndSpline`:
  - For each of (tension sigma=3.0), (gaussian eps=0.9), (multiquadric eps=1.0), call `brady2d_stability_score(..., max_layer=8)`.
  - **Graceful skip:** if `known_values.json["brady2d_calibration"]` is absent (plan 41.11e not run yet), each parameterized case calls `pytest.skip("brady2d_calibration not populated — run plan 41.11e manually")` and does not fail.
  - When present, asserts `overall_verdict=="pass"` if plan 41 calibration says it should pass, else `"fail"`.
  - Uses `known_values.json["brady2d_calibration"]` to decide the expected outcome — consistency check between plan 41 analytical and plan 42 empirical.
  - Mark `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer8EndToEndSpline" --run-slow` → **PASSED** (3 parametrized cases: `E4_tension_3`, `E4_gaussian_09`, `E4_multiquadric_1`, all `overall_verdict=="pass"`, L8 `stable` and `final_linf < L8_FINAL_LINF_TOL`; total wall 81 s for full triple, 37 s for tension alone). Without `--run-slow`: 3 skipped cleanly.
  - Note: test uses `layer8_N=21, layer8_t_final=1.0` matching the classical smoke to keep each C++ run under 1 s; most of the wall time is L7's 2D non-normality SVD (~20 s × 3 families). Verdict comparison uses the full L1–L8 pipeline result; stored calibration only recorded L1–L6 data, but the `overall_verdict` field is still the authoritative consistency signal between plan 41 and plan 42. Falls back to skip gracefully if `known_values.json` or the `brady2d_calibration` key is missing, or if a specific calibration label is not yet populated.

### 42.8 — New sweep subcommand `brady2d`

- [x] **42.8a** Create `scripts/stencil_gen/sweeps/brady2d_sweep.py` with `main(argv) -> int`:
  - Args: `--scheme {E2,E4}`, `--kernel {classical,tension,gaussian,multiquadric}`, `--param-range lo hi n` (three values: low, high, point count), `--max-layer int` (1-8, default 3), `--validate-with-cpp` (re-runs top-3 survivors at `max_layer=8`), `--update-known-values`, plus `--layer8-N` / `--layer8-t-final` passthroughs for L8 cost control.
  - For each parameter value in the range:
    1. Run `brady2d_stability_score(scheme, kernel, params, max_layer=max_layer)`.
    2. Collect per-layer scalars into a `SweepPoint` dataclass.
  - Print a markdown-pipe table sorted by param; columns adapt to `max_layer` (L1/L3/L6/L7/L8 shown only when their layer ran).
  - If `--validate-with-cpp`, re-run the top 3 passing points at `max_layer=8` (ranked by `layer6.transient_growth_bound` ascending when L6 is present, else `layer3.max_stab_eig` ascending).
  - Persist results to `known_values.json["brady2d_sweep"][scheme][kernel]` when `--update-known-values` set — full per-point `StabilityReport` serialised via `_report_to_dict` (keeps only JSON-safe scalar fields).
  - Classical kernel degenerates to a single point using the hard-coded `CLASSICAL_E4_ALPHA = [-0.7733323791884821, 0.1623961700641681]` (matching `test_cpp_bridge.py` and `E4u_1.t.cpp`); `--param-range` is ignored there. Spline kernels require `--param-range`, else `ValueError`/exit 2.
  - File: `scripts/stencil_gen/sweeps/brady2d_sweep.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps.brady2d_sweep --scheme E4 --kernel tension --param-range 2 4 3 --max-layer 3` → **PASSED** (3 sigma values swept in ~8s, markdown table shows all three passing L1+L3, exit 0; classical single-point path also validated: `--kernel classical` emits one `pass` row with L1=2.1e-2, L3=-1.8e-4). The `-m sweeps brady2d` subcommand form (no `.brady2d_sweep` module suffix) is wired by 42.8b.
  - Note: `--param-range` is `nargs=3` rather than three separate flags (`--sigma-range` / `--epsilon-range`) so one arg serves all three spline kernels — 42.8b exposes it verbatim under the same name. For the `--validate-with-cpp` path to do anything useful, `--max-layer` must be ≥3 (otherwise the ranking has no signal). `TOP_K_FOR_L8 = 3` is a module constant matching the plan text.

- [x] **42.8a-fu1** Follow-up from 42.8a review: add unit tests for `sweeps/brady2d_sweep.py` in `tests/test_sweep_gv_objectives.py` (matching the precedent set by `test_tension_sweep_main_merges_known_values` etc.). The only coverage today is a manual CLI smoke recorded in 42.8a's Test line; no automated test exercises argument parsing, the classical single-point branch, the spline-kernel path, the ranking helper, or the known-values write path. Each new sweep script in this repo has a corresponding `main()` test with monkeypatched `sweeps_common.KNOWN_VALUES_PATH`, and the same pattern should apply here.
  - Required test cases (fast — must not invoke the C++ binary, so avoid `--validate-with-cpp` and either stub `brady2d_stability_score` via `monkeypatch.setattr(brady2d_sweep, "brady2d_stability_score", ...)` or restrict to `--max-layer 1` so only L1 runs):
    1. `test_brady2d_sweep_main_classical_single_point`: `--kernel classical` with no `--param-range` succeeds (exit 0), produces one table row, and does not raise the spline-kernel `ValueError`. Verifies `CLASSICAL_E4_ALPHA` is what ends up in `params_dict`.
    2. `test_brady2d_sweep_main_spline_requires_param_range`: `--kernel tension` without `--param-range` exits 2 with the expected message on stderr.
    3. `test_brady2d_sweep_main_param_range_parsing`: `--param-range 2 4 3` yields exactly 3 sweep points at `{2.0, 3.0, 4.0}` (post-`np.linspace`) for `tension`/`gaussian`/`multiquadric` and that the correct scalar name (`sigma` vs `epsilon`) appears in each `params_dict`.
    4. `test_brady2d_sweep_main_update_known_values`: with `monkeypatch.setattr(sweeps_common, "KNOWN_VALUES_PATH", tmp_path / "known_values.json")`, `--update-known-values` writes under `brady2d_sweep.<scheme>.<kernel>` without clobbering unrelated top-level keys (mirror the merge check from `test_tension_sweep_main_merges_known_values`).
    5. `test_rank_for_l8_prefers_layer6_then_layer3`: unit test on `rank_for_l8` directly — construct two dummy `SweepPoint`s with hand-built `StabilityReport`s, verify ordering uses `layer6.transient_growth_bound` ascending when available and falls back to `layer3.max_stab_eig` otherwise; passing points with mixed `layer6 is None` force the L3 branch; empty-passing returns `[]`.
  - Run under the default (non-slow) suite — total wall under ~5 s.
  - File: `scripts/stencil_gen/tests/test_sweep_gv_objectives.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_gv_objectives.py -x -q -k "brady2d or rank_for_l8"` → **PASSED** (7 tests in 0.88 s: classical single-point dispatch asserts `CLASSICAL_E4_ALPHA` is what lands in `params_dict`; spline-without-range exits 2 with `--param-range` / `tension` mentioned on stderr and zero stability-score calls; `--param-range 2 4 3` produces exactly `{2.0, 3.0, 4.0}` with the correct `sigma`/`epsilon` key for all three parametrized kernels; `--update-known-values` writes `brady2d_sweep.E4.tension.points` while preserving `E4_1.tension.preexisting_extra_key` and the unrelated `footprint` top-level key; `rank_for_l8` unit test covers L6-tgb ordering, L3-eigenvalue fallback when L6 is missing, all-failing → `[]`, and the `TOP_K_FOR_L8=3` cap). Full `test_sweep_gv_objectives.py` run still green: 24 passed.
  - Note: tests monkeypatch `brady2d_sweep.brady2d_stability_score` with a recorder stub so no heavy analysis runs; `sweeps_common.KNOWN_VALUES_PATH` monkeypatch reuses the pattern from `test_tension_sweep_main_merges_known_values`. `_make_stub_report` is a small helper that builds a minimal `StabilityReport` populated only with L1/L3 (and optionally L6) so `_report_to_dict` round-trips cleanly. `TOP_K_FOR_L8` is asserted as the authoritative cap rather than hard-coded to `3` in the test so a future change to the constant doesn't silently break the test.

- [x] **42.8b** Register `brady2d` subcommand in `sweeps/__main__.py`:
  - `sub_brady2d = subparsers.add_parser("brady2d", ...)` with all the args from 42.8a: `--scheme`, `--kernel`, `--param-range` (nargs=3, metavar=("LO","HI","N"); **use the same name `--param-range` as 42.8a — do NOT rename to `--sigma-range`/`--epsilon-range`**), `--max-layer`, `--validate-with-cpp`, `--layer8-N`, `--layer8-t-final`, `--update-known-values`.
  - Add dispatch block `if args.command == "brady2d": from .brady2d_sweep import main as brady2d_main; return brady2d_main(...)`.
  - Add to `_run_all`'s `sweeps` list in quick mode (`--param-range 2 4 3 --max-layer 3`).
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps brady2d --scheme E4 --kernel tension --param-range 2 4 3 --max-layer 3` → **PASSED** (3 sigma values swept, markdown table shows all three L1+L3 passing, exit 0). Classical single-point path also validated via `-m sweeps brady2d --scheme E4 --kernel classical --max-layer 3` (single `pass` row). Error path `--kernel tension` without `--param-range` exits 2 with the expected `--param-range lo hi n is required` stderr. `uv run pytest tests/test_sweep_gv_objectives.py -x -q -k "brady2d or rank_for_l8"` → 7 passed (brady2d_sweep.main still works when invoked via the top-level `sweeps` dispatcher since the forwarded argv is byte-equivalent to direct invocation).
  - Note: dispatch block forwards the parsed args as string argv to `brady2d_sweep.main`, mirroring the pattern used by every other subcommand (`epsilon`, `tension`, etc.). `--param-range` only forwarded when provided so the classical-single-point path still sees `param_range=None` in the inner parser. `_run_all` entry runs a 3-sigma tension sweep at `--max-layer 3` — matches the 42.8a smoke used while developing the sweep.

### 42.9 — Regression tests and documentation

- [x] **42.9a** Add `TestRegressionBrady2DSweep` in `test_phs.py`:
  - Loads `brady2d_sweep` from `known_values.json`, iterates each stored entry, re-runs `brady2d_stability_score` at `max_layer=3` (fast), asserts stored overall verdict matches recomputed.
  - Graceful skip if absent.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionBrady2DSweep"` → **PASSED** (skips cleanly when `brady2d_sweep` key absent from `known_values.json`, as it currently is; 1 skipped in 0.84 s). Positive-path verified out-of-band by injecting a synthetic `brady2d_sweep.E4.classical` entry with `{"alpha": [-0.7733..., 0.1624...]}` and stored `overall_verdict="pass"` — test recomputes at `max_layer=3` and asserts match (1 passed in 1.49 s, `known_values.json` restored from backup afterward). Full `TestRegression*` suite still green: 16 passed, 7 skipped.
  - Note: mirrors the structure of `TestRegressionBrady2DCalibration` just above — autouse skip fixture + single iteration test. Iterates `brady2d_sweep → scheme → kernel → points`, reads `params_dict` and stored `report.overall_verdict`/`failed_layer` from each point. Uses the same "if stored failure is reachable at layer ≤ 3 then expect fail, else expect pass" logic as the calibration test, since sweeps often record results at `max_layer > 3` where later layers might fail. `checked == 0` after iteration triggers a `pytest.skip` so an empty sweep bucket doesn't silently no-op.

- [x] **42.9b** Create `docs/brady2d_cpp_bridge_reference.md`:
  - Describes the bridge architecture (Python writes Lua, calls shoccs, parses CSV).
  - Lists the three new C++ stencil families with their Lua type strings and parameters.
  - Documents the runtime-parameterized pattern (scalar_params in codegen).
  - Explains why cut-cell variants are deferred.
  - File: `docs/brady2d_cpp_bridge_reference.md` (new)
  - Test: (no test) → **DONE** (reference created at `docs/brady2d_cpp_bridge_reference.md`, ~150 lines). Covers architecture diagram, key files table, the three spline families with Lua type strings + parameter defaults, construction-time solve + 5×7 cache layout, scalar-params codegen pattern (`StencilGenSpec.scalar_params` + `build_symbol_map` no-subscript mapping), cut-cell deferral rationale, programmatic + sweep usage, and L8 failure thresholds.
  - Note: placed under top-level `docs/` (alongside `stencils.md`, `discrete_operators.md`, etc.) rather than `scripts/stencil_gen/docs/` so the C++-side bridge reference is discoverable from the repo root; 42.9c updates the Python-side L8 section of `scripts/stencil_gen/docs/brady2d_stability_reference.md` separately.

- [ ] **42.9c** Update `scripts/stencil_gen/docs/brady2d_stability_reference.md` (from plan 41) with a new "Layer 8 — C++ simulation" section.
  - File: `scripts/stencil_gen/docs/brady2d_stability_reference.md`
  - Test: (no test)

- [ ] **42.9d** Update `.claude/skills/stencil-sweeps/SKILL.md`:
  - Add `brady2d` to the CLI quick reference.
  - Add one line about `--validate-with-cpp`.
  - File: `.claude/skills/stencil-sweeps/SKILL.md`
  - Test: (no test)

- [ ] **42.9e** Update `.claude/skills/group-velocity-analysis/SKILL.md`:
  - Add one bullet about L8 C++ validation path.
  - File: `.claude/skills/group-velocity-analysis/SKILL.md`
  - Test: (no test)

### 42.10 — E2 variants and follow-ups (optional / deferred)

- [ ] **42.10a** Deferred to follow-up: replicate 42.5–42.7 for E2 schemes (`tension_E2u_1`, etc.). Pattern is identical; only parameter sizes (r=3, t=4) differ.
  - This is a one-pass clone-and-rename operation once 42.5–42.7 are proven.
  - File: (multiple new files under `src/stencils/`)
  - Test: deferred

- [ ] **42.10b** Deferred to follow-up: cut-cell variants (`tension_E4_1`, not E4u_1) — requires handling the psi-dependent boundary at construction or per-call. Separate follow-up plan, since cut-cell precomputation requires a coefficient cache indexed by psi or a runtime solve per cut cell.
  - File: (follow-up plan)
  - Test: deferred

---

## Ordering

```
42.1a → 42.1b                      # Lua config first; unblocks 42.2
  ↓
42.2a → 42.2b → 42.2c              # Python bridge for classical stencils
  ↓
42.3a → 42.3b → 42.3c              # L8 integration (classical only)
  ↓
42.4a → 42.4b → 42.4c → 42.4d      # Codegen scalar-params support
  ↓
42.5a → 42.5b → 42.5c → 42.5d → 42.5e → 42.5f → 42.5g   # First spline family (tension_E4u_1)
  ↓
42.6a → 42.6b → 42.6c → 42.6d      # Gaussian (4 items: fixture+skeleton, solver, register, test)
  ↓
42.6e → 42.6f → 42.6g → 42.6h      # Multiquadric (same 4-item shape)
  ↓
42.7a → 42.7b → 42.7c              # Bridge dispatch for spline families
  ↓
42.8a → 42.8a-fu1 → 42.8b          # New sweep subcommand (42.8a-fu1 = review follow-up: unit tests)
  ↓
42.9a → 42.9b → 42.9c → 42.9d → 42.9e    # Regression + docs
  ↓
(42.10a, 42.10b are deferred follow-ups)
```

Parallelizable after 42.4:
- 42.5 (tension) can run before 42.6 (Gaussian) as the first family; 42.6 reuses the codegen path.
- 42.9 docs can start after 42.7 lands.

---

## Completion Criteria

- `lua-configs/brady_livescu_4_3.lua` exists as a template with `--{{N}}--`, `--{{T_FINAL}}--`, `--{{SCHEME_TABLE}}--` markers (the double-dash-dash-braces syntax means the raw file is a Lua comment soup and is NOT standalone-runnable — converted to a template by 42.2a). The thin variants `lua-configs/brady_livescu_4_3_n61.lua` and `lua-configs/brady_livescu_4_3_long.lua` remain standalone-runnable. Running `make_brady2d_lua(...)` and piping the result into shoccs (as `run_cpp_brady2d` does in 42.2b) produces `logs/system.csv` with a finite L∞ column.
- `scripts/stencil_gen/stencil_gen/cpp_bridge.py` provides `run_cpp_brady2d` that successfully runs the classical E4 closure end-to-end (L8 integration test passes).
- Three new C++ stencil families exist: `tension_E4u_1`, `gaussian_E4u_1`, `multiquadric_E4u_1` — each compiled as a separate `.cpp` file in `src/stencils/`, each registered in `stencil::from_lua`, each with a passing Catch2 unit test (`t-{family}_E4u_1`) that verifies coefficients match the Python reference within `1e-12`.
- `StencilGenSpec` supports scalar runtime parameters via `scalar_params` field, with `generate_stencil_cpp` emitting clean `real name;` fields and `name`-subscripted expressions (not `name[0]`).
- `brady2d_stability_score(scheme="E4", kernel="tension", params={"sigma": 3.0}, max_layer=8)` runs end-to-end: L1–L7 analytical layers pass, L8 invokes the C++ simulation with `type="tension_E4u"` and `sigma=3.0` in the Lua config, and reports `overall_verdict="pass"` when the calibration (plan 41) predicts it should pass.
- `sweeps/brady2d_sweep.py` exists and `python -m sweeps brady2d --scheme E4 --kernel tension --param-range 2 4 3 --max-layer 3` runs a 3-point sweep and prints a per-layer results table.
- `TestRegressionBrady2DSweep` passes; `TestLayer8EndToEndSpline` confirms consistency between L1–L7 analytical verdicts and L8 empirical verdicts for all three spline families.
- `ctest --test-dir build` passes; full build succeeds with zero warnings from the new stencil files.
- **No rebuild required per sweep point.** Verified by running `python -m sweeps brady2d --scheme E4 --kernel tension --param-range 2 6 20 --validate-with-cpp` and confirming the C++ build is compiled exactly once at the start and Lua configs are the only thing changing per sweep point.
- Documentation: `docs/brady2d_cpp_bridge_reference.md` exists; `docs/brady2d_stability_reference.md` updated with L8 section; both skill files updated.
- Plan 41's completion criterion that "Plan 42 can now start" is no longer relevant — plan 42 is complete.
