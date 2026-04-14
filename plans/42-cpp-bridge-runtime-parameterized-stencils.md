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

- [ ] **42.4c** Extend `build_symbol_map` in `scripts/stencil_gen/stencil_gen/printer.py:68` to accept `scalar_params: list[str]`:
  - For each scalar param name, map `Symbol(name) → name` (no subscript).
  - Update callers in `codegen.py` to pass `spec.scalar_params` through.
  - File: `scripts/stencil_gen/stencil_gen/printer.py`, `scripts/stencil_gen/stencil_gen/codegen.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_printer.py -x -q -k "TestScalarParams"`

- [ ] **42.4d** Tests `TestScalarParamsCodegenEndToEnd`:
  - Build a synthetic `StencilGenSpec(name="TestStruct", scalar_params=["sigma"], interior_coeffs=[sympy.Symbol("sigma") * sympy.Symbol("h")], ...)`.
  - Call `generate_stencil_cpp(spec)` and assert the output contains `real sigma;` as a field, a constructor taking `real sigma`, and `sigma` (not `sigma[0]`) inside the interior body.
  - File: `scripts/stencil_gen/tests/test_codegen.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_codegen.py -x -q -k "TestScalarParamsCodegenEndToEnd"`

### 42.5 — First spline family in C++: `tension_E4u_1` (construction-time runtime solve, hand-written)

The strategy: the constructor takes `real sigma` from Lua, builds the small (r=5, t=7) tension-spline RBF system in C++ using a hand-coded Gaussian elimination specialized to the tension kernel, caches the resulting 5×7 boundary coefficient matrix in a struct field, and `nbs_floating` reads from the cache. The reference implementation is Python's `phs._rbf_weights_numeric(..., kernel="tension")`.

**Note:** The Python-emits-C++ codegen path (originally scoped as 42.5a / 42.5e) is explicitly deferred to a follow-up plan. Plan 42 ships with a hand-written `tension_E4u_1.cpp`. Rationale: emitting a correct Gaussian elimination as C++ text from Python adds significant scope and the hand-written version is a one-time effort per kernel.

- [ ] **42.5a** Generate the Python reference coefficient array for `tension_E4u_1` at `sigma=3.0`:
  - Run `cd scripts/stencil_gen && uv run python -c "from stencil_gen.phs import build_diff_matrix_rbf; import numpy as np; D = build_diff_matrix_rbf(n=20, p=2, q=3, epsilon=3.0, kernel='tension', nu=1, nextra=0); np.set_printoptions(precision=17, suppress=False); print(repr(D[:5, :7]))"` to obtain the 5×7 boundary coefficient block at h=1/(n-1). Capture the exact numeric output.
  - Add the captured array as a constant `REFERENCE_TENSION_E4U1_SIGMA3_COEFFS` in a new test helper file (e.g. `scripts/stencil_gen/tests/fixtures/tension_e4u1_reference.py`) for later reuse. Also hand-paste it as a `std::array<real, 35>` into the Catch2 test file in 42.5f.
  - File: `scripts/stencil_gen/tests/fixtures/__init__.py` (new), `scripts/stencil_gen/tests/fixtures/tension_e4u1_reference.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from tests.fixtures.tension_e4u1_reference import REFERENCE_TENSION_E4U1_SIGMA3_COEFFS; print(len(REFERENCE_TENSION_E4U1_SIGMA3_COEFFS))"`

- [ ] **42.5b** Create `src/stencils/tension_E4u_1.cpp` struct skeleton (no solver body yet):
  - Struct layout mirroring `E4u_1` but with `real sigma` field replacing `std::array<real, 2> alpha`, and a new `std::array<real, 5 * 7> cached_coeffs` member.
  - Constructor `tension_E4u_1(real sigma_in) : sigma(sigma_in) { /* solver call inserted in 42.5d */ }`.
  - `interior`, `query`, `query_max`, method stubs for `nbs_floating` and `nbs` matching `E4u_1`'s signatures but with empty (TODO) bodies that simply fill the output span with zeros and return it.
  - Does **not** register in `stencil.cpp` yet — that's 42.5e.
  - **Must compile** once added to CMakeLists in 42.5c.
  - File: `src/stencils/tension_E4u_1.cpp` (new)
  - Test: `cmake --build build --target shoccs-stencils` (just needs to compile; no functional test yet)

- [ ] **42.5c** Add `tension_E4u_1.cpp` to `src/stencils/CMakeLists.txt` build list:
  - Append `tension_E4u_1.cpp` to the `add_library(shoccs-stencils ...)` source list. No test target yet.
  - Do not yet register in `stencil::from_lua` (42.5e).
  - File: `src/stencils/CMakeLists.txt`
  - Test: `cmake --build build --target shoccs-stencils`

- [ ] **42.5d** Implement `solve_tension_coefficients(real sigma, std::array<real, 5*7>& out)` as a static/anonymous-namespace helper in `tension_E4u_1.cpp`:
  - Builds the 7×7 tension-spline kernel matrix entries `K[i,j] = sigma*|x_i - x_j| - 1 + exp(-sigma*|x_i - x_j|)` where `x_i = i` for i in 0..6 (reference grid, h=1).
  - Augments with the polynomial block (5 polynomial columns for q=3 → `1, x, x^2, x^3`, and 1 row/col for the order-1 derivative constraint — matches `phs._rbf_weights_numeric`'s construction).
  - Solves via plain Gaussian elimination with partial pivoting, 7×7 fits on the stack.
  - Fills `out[i*7 + j] = w_ij` where `w_ij` is the weight for boundary row `i` at column `j`.
  - Wire the call into the `tension_E4u_1` constructor (replacing the 42.5b stub comment).
  - Update `nbs_floating` to copy `cached_coeffs` into the output span, apply `1/h` scaling, then the `right` flip logic matching `E4u_1.cpp:~103`.
  - File: `src/stencils/tension_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs-stencils` (still just compiles; functional test in 42.5f)

- [ ] **42.5e** Register `tension_E4u_1` in `stencil.hpp` and `stencil.cpp`:
  - Add `stencil make_tension_E4u_1(real sigma)` factory declaration in `stencil.hpp`.
  - In `stencil::from_lua` at `stencil.cpp`, **insert a new `else if (type == "tension_E4u")` branch immediately after the `type == "E8u"` branch and before the final `logger(...err...)` fallthrough**. The branch reads `real sigma = m["sigma"].get_or(3.0)` and calls `make_tension_E4u_1(sigma)`. Verify no existing branches are affected by diffing the file after the edit.
  - Also add a `make_tension_E4u_1` factory definition at the bottom of `tension_E4u_1.cpp`.
  - File: `src/stencils/stencil.hpp`, `src/stencils/stencil.cpp`, `src/stencils/tension_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs-exe` (full chain compiles and links)

- [ ] **42.5f** Add unit test `t-tension_E4u_1`:
  - Add `add_unit_test(tension_E4u_1 "stencils" shoccs-stencils)` to `src/stencils/CMakeLists.txt`.
  - Create `src/stencils/tension_E4u_1.t.cpp` with three Catch2 tests:
    - `TEST_CASE("tension_E4u_1 construction at sigma=3")` — instantiate at `sigma=3.0`, verify no exception.
    - `TEST_CASE("tension_E4u_1 coefficients match Python reference")` — hard-code the 35-entry reference array from 42.5a's fixture as a `std::array<real, 35>`, assert each cached_coeffs entry matches within `1e-12`.
    - `TEST_CASE("tension_E4u_1 nbs_floating fills output span")` — call `nbs_floating(h=0.1, psi=1.0, c, right=false)` with a 35-element output buffer, verify the returned span has size 35 and all entries are finite.
  - File: `src/stencils/CMakeLists.txt`, `src/stencils/tension_E4u_1.t.cpp` (new)
  - Test: `cmake --build build --target t-tension_E4u_1 && ctest --test-dir build -R t-tension_E4u_1`

### 42.6 — Second and third spline families: `gaussian_E4u_1`, `multiquadric_E4u_1`

Each family follows the same 4-item pattern as 42.5b–f (minus the split reference-generation step, which is combined into the first item per family). The kernel function changes; everything else mirrors `tension_E4u_1`.

- [ ] **42.6a** Generate the Python reference for `gaussian_E4u_1` at `epsilon=0.9` and create `src/stencils/gaussian_E4u_1.cpp` skeleton + CMake registration:
  - Generate reference: `cd scripts/stencil_gen && uv run python -c "from stencil_gen.phs import build_diff_matrix_rbf; import numpy as np; D = build_diff_matrix_rbf(n=20, p=2, q=3, epsilon=0.9, kernel='gaussian', nu=1, nextra=0); np.set_printoptions(precision=17); print(repr(D[:5, :7]))"`.
  - Add as `REFERENCE_GAUSSIAN_E4U1_EPS09_COEFFS` to `tests/fixtures/gaussian_e4u1_reference.py`.
  - Create `src/stencils/gaussian_E4u_1.cpp` as a clone of `tension_E4u_1.cpp` with `real epsilon` instead of `real sigma` and empty solver (stub).
  - Append `gaussian_E4u_1.cpp` to `src/stencils/CMakeLists.txt`.
  - File: `scripts/stencil_gen/tests/fixtures/gaussian_e4u1_reference.py` (new), `src/stencils/gaussian_E4u_1.cpp` (new), `src/stencils/CMakeLists.txt`
  - Test: `cmake --build build --target shoccs-stencils`

- [ ] **42.6b** Implement `solve_gaussian_coefficients(real epsilon, std::array<real, 5*7>& out)` in `gaussian_E4u_1.cpp`:
  - Same Gaussian elimination structure as `solve_tension_coefficients` but with kernel `gaussian_kernel(r, eps) = exp(-(eps*r)*(eps*r))`.
  - Wire into the constructor; populate `nbs_floating` to read from `cached_coeffs`.
  - File: `src/stencils/gaussian_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs-stencils`

- [ ] **42.6c** Register `gaussian_E4u_1` in `stencil.hpp` / `stencil.cpp` dispatch (type string `"gaussian_E4u"`, reading `real epsilon = m["epsilon"].get_or(0.9)`, inserted immediately after the `tension_E4u` branch from 42.5e):
  - File: `src/stencils/stencil.hpp`, `src/stencils/stencil.cpp`, `src/stencils/gaussian_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs-exe`

- [ ] **42.6d** Add `t-gaussian_E4u_1` unit test mirroring `t-tension_E4u_1`:
  - Append `add_unit_test(gaussian_E4u_1 "stencils" shoccs-stencils)` to CMakeLists.
  - Create `src/stencils/gaussian_E4u_1.t.cpp` with the same three Catch2 tests, hard-coded against the 42.6a reference.
  - File: `src/stencils/CMakeLists.txt`, `src/stencils/gaussian_E4u_1.t.cpp` (new)
  - Test: `cmake --build build --target t-gaussian_E4u_1 && ctest --test-dir build -R t-gaussian_E4u_1`

- [ ] **42.6e** Generate the Python reference for `multiquadric_E4u_1` at `epsilon=1.0` and create `src/stencils/multiquadric_E4u_1.cpp` skeleton + CMake registration (mirrors 42.6a):
  - Generate reference via the same one-liner with `kernel='multiquadric'`, store in `tests/fixtures/multiquadric_e4u1_reference.py`.
  - Clone skeleton from `gaussian_E4u_1.cpp`, rename struct and kernel function.
  - File: `scripts/stencil_gen/tests/fixtures/multiquadric_e4u1_reference.py` (new), `src/stencils/multiquadric_E4u_1.cpp` (new), `src/stencils/CMakeLists.txt`
  - Test: `cmake --build build --target shoccs-stencils`

- [ ] **42.6f** Implement `solve_multiquadric_coefficients` with kernel `sqrt(1 + (eps*r)^2)`; wire into constructor (mirrors 42.6b):
  - File: `src/stencils/multiquadric_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs-stencils`

- [ ] **42.6g** Register `multiquadric_E4u_1` in `stencil.hpp` / `stencil.cpp` dispatch (type `"multiquadric_E4u"`, default `epsilon=1.0`, inserted after `gaussian_E4u` branch):
  - File: `src/stencils/stencil.hpp`, `src/stencils/stencil.cpp`, `src/stencils/multiquadric_E4u_1.cpp`
  - Test: `cmake --build build --target shoccs-exe`

- [ ] **42.6h** Add `t-multiquadric_E4u_1` unit test (mirrors 42.6d):
  - File: `src/stencils/CMakeLists.txt`, `src/stencils/multiquadric_E4u_1.t.cpp` (new)
  - Test: `cmake --build build --target t-multiquadric_E4u_1 && ctest --test-dir build -R t-multiquadric_E4u_1`

### 42.7 — Closed-loop bridge for spline families

- [ ] **42.7a** Extend `layer8_cpp_simulation` in `brady2d_stability.py` to handle the new families:
  - Dispatch table:
    - `("E4", "classical")` → Lua `type="E4"`, pass `alpha`
    - `("E4", "tension")` → Lua `type="tension_E4u"`, pass `sigma`
    - `("E4", "gaussian")` → Lua `type="gaussian_E4u"`, pass `epsilon`
    - `("E4", "multiquadric")` → Lua `type="multiquadric_E4u"`, pass `epsilon`
  - E2 variants deferred (pattern is identical; do in a follow-up phase if needed).
  - File: `scripts/stencil_gen/stencil_gen/brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer8Dispatch"`

- [ ] **42.7b** Extend `make_brady2d_lua` in `cpp_bridge.py` to emit the new scheme tables:
  - For `tension_E4u` and `gaussian_E4u`/`multiquadric_E4u`, emit `scheme = { order = 1, type = "<name>", sigma = <val> }` or `epsilon = <val>` respectively.
  - File: `scripts/stencil_gen/stencil_gen/cpp_bridge.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_cpp_bridge.py -x -q -k "TestMakeBrady2DLuaSpline"`

- [ ] **42.7c** Integration test `TestLayer8EndToEndSpline`:
  - For each of (tension sigma=3.0), (gaussian eps=0.9), (multiquadric eps=1.0), call `brady2d_stability_score(..., max_layer=8)`.
  - **Graceful skip:** if `known_values.json["brady2d_calibration"]` is absent (plan 41.11e not run yet), each parameterized case calls `pytest.skip("brady2d_calibration not populated — run plan 41.11e manually")` and does not fail.
  - When present, asserts `overall_verdict=="pass"` if plan 41 calibration says it should pass, else `"fail"`.
  - Uses `known_values.json["brady2d_calibration"]` to decide the expected outcome — consistency check between plan 41 analytical and plan 42 empirical.
  - Mark `@pytest.mark.slow`.
  - File: `scripts/stencil_gen/tests/test_brady2d_stability.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_brady2d_stability.py -x -q -k "TestLayer8EndToEndSpline"`

### 42.8 — New sweep subcommand `brady2d`

- [ ] **42.8a** Create `scripts/stencil_gen/sweeps/brady2d_sweep.py` with `main(argv) -> int`:
  - Args: `--scheme {E2,E4}`, `--kernel {classical,tension,gaussian,multiquadric}`, `--param-range lo hi n` (e.g. `--sigma-range 1.0 10.0 10`), `--max-layer int`, `--validate-with-cpp` (sets `max_layer=8` for winners), `--update-known-values`.
  - For each parameter value in the range:
    1. Run `brady2d_stability_score(scheme, kernel, params, max_layer=max_layer)`.
    2. Collect per-layer scalars into a results table.
  - Print a markdown table sorted by param.
  - If `--validate-with-cpp`, re-run at `max_layer=8` for the top 3 survivors (by lowest L6 transient_growth_bound or similar ranking metric).
  - Persist results to `known_values.json["brady2d_sweep"][scheme][kernel]` when `--update-known-values` set.
  - File: `scripts/stencil_gen/sweeps/brady2d_sweep.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps brady2d --scheme E4 --kernel tension --sigma-range 2 4 3 --max-layer 3`

- [ ] **42.8b** Register `brady2d` subcommand in `sweeps/__main__.py`:
  - `sub_brady2d = subparsers.add_parser("brady2d", ...)` with all the args from 42.8a.
  - Add dispatch block `if args.command == "brady2d": from .brady2d_sweep import main as brady2d_main; return brady2d_main(...)`.
  - Add to `_run_all`'s `sweeps` list in quick mode (`--sigma-range 2 4 3 --max-layer 3`).
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps brady2d --scheme E4 --kernel tension --sigma-range 2 4 3 --max-layer 3`

### 42.9 — Regression tests and documentation

- [ ] **42.9a** Add `TestRegressionBrady2DSweep` in `test_phs.py`:
  - Loads `brady2d_sweep` from `known_values.json`, iterates each stored entry, re-runs `brady2d_stability_score` at `max_layer=3` (fast), asserts stored overall verdict matches recomputed.
  - Graceful skip if absent.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionBrady2DSweep"`

- [ ] **42.9b** Create `docs/brady2d_cpp_bridge_reference.md`:
  - Describes the bridge architecture (Python writes Lua, calls shoccs, parses CSV).
  - Lists the three new C++ stencil families with their Lua type strings and parameters.
  - Documents the runtime-parameterized pattern (scalar_params in codegen).
  - Explains why cut-cell variants are deferred.
  - File: `docs/brady2d_cpp_bridge_reference.md` (new)
  - Test: (no test)

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
42.5a → 42.5b → 42.5c → 42.5d → 42.5e → 42.5f   # First spline family (tension_E4u_1)
  ↓
42.6a → 42.6b → 42.6c → 42.6d      # Gaussian (4 items: fixture+skeleton, solver, register, test)
  ↓
42.6e → 42.6f → 42.6g → 42.6h      # Multiquadric (same 4-item shape)
  ↓
42.7a → 42.7b → 42.7c              # Bridge dispatch for spline families
  ↓
42.8a → 42.8b                      # New sweep subcommand
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
- `sweeps/brady2d_sweep.py` exists and `python -m sweeps brady2d --scheme E4 --kernel tension --sigma-range 2 4 3 --max-layer 3` runs a 3-point sweep and prints a per-layer results table.
- `TestRegressionBrady2DSweep` passes; `TestLayer8EndToEndSpline` confirms consistency between L1–L7 analytical verdicts and L8 empirical verdicts for all three spline families.
- `ctest --test-dir build` passes; full build succeeds with zero warnings from the new stencil files.
- **No rebuild required per sweep point.** Verified by running `python -m sweeps brady2d --sigma-range 2 6 20 --validate-with-cpp` and confirming the C++ build is compiled exactly once at the start and Lua configs are the only thing changing per sweep point.
- Documentation: `docs/brady2d_cpp_bridge_reference.md` exists; `docs/brady2d_stability_reference.md` updated with L8 section; both skill files updated.
- Plan 41's completion criterion that "Plan 42 can now start" is no longer relevant — plan 42 is complete.
