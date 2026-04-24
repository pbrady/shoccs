# Known Limitations and Deferred Work

Issues that are understood but not fixed, and scope that was explicitly punted.

## Optimizer

### `gate_layer` doesn't auto-infer from the objective

**Problem:** When `--objective` is `layer_bl42.max_spectral_abscissa`, `layer6.transient_growth_bound`, or `layer7.max_spectral_abscissa`, the default `gate_layer=3` causes the cascade to short-circuit at the gate *before* the objective's layer is computed. Every evaluation returns `+inf`; the optimizer can't find a gradient.

**Workaround:** Pass `--gate-layer` explicitly, one tier below the objective's layer. Example: for `--objective layer_bl42.max_spectral_abscissa`, pass `--gate-layer 2` to gate on L1/L2 only, letting L3r's failure become the objective value instead of a gate trip.

**Fix:** Small (~30 lines) — auto-infer in `stencil_gen/optimizer.py::make_objective`. See `next_steps.md` — scheduled as item 45.0 or a standalone micro-plan.

### Monotone objective pitfall

**Problem:** `layer1.boundary_gv_err` decreases monotonically as σ → 0 (tension→PHS k=2 in that limit). Optimizing against it drives σ to whatever lower bound is set, which isn't usefully interior.

**Workaround:** Use `layer3.max_stab_eig`, `layer6.transient_growth_bound`, or `layer7.max_spectral_abscissa` as the primary objective. These have real interior minima.

**Fix:** Documentation-only. The optimizer can't know the shape of the objective a priori.

### Only 4 kernels routed through layered cascade

**Problem:** `brady2d_stability_score` routes only `classical`, `tension`, `gaussian`, `multiquadric`. `tension-penalty` and `mixed-epsilon` kernels have their own sweep modules but aren't in the main score orchestrator.

**Workaround:** None — use the kernel-specific sweeps directly.

**Fix:** Extend `params_from_vector` / `vector_from_params` and the layer functions to accept these kernels. ~100 lines of work. No active demand.

### E2 classical-α is 4D and uncharted

**Problem:** `DEFAULT_BOUNDS` does not include `("E2", "classical")`. The 4D (α[0..3]) feasible region has no sweep coverage, no bounds established. Optimizing here would need a feasibility pre-scan first.

**Workaround:** Skip. User noted "stability of 2nd order is inconsequential" (session 2026-04-14).

**Fix:** A future plan could do (a) a random-sample feasibility pre-scan, (b) establish bounds, (c) run optimization.

## C++ side

### No E2 spline stencil families

**Problem:** `tension_E4u_1`, `gaussian_E4u_1`, `multiquadric_E4u_1` exist; no E2 equivalents. Plan 42 deferred these as 42.10a.

**Workaround:** Use E4 families for BL-style tests. E2 is low accuracy anyway.

**Fix:** Clone-and-rename once someone cares. Each family is ~150 lines of C++.

### No cut-cell runtime-parameterized spline families

**Problem:** Cut-cell geometry needs coefficients that depend on `psi` (the cut-cell fraction, 0 < psi < 1). Our runtime-param C++ structs (plan 42) do the RBF linear solve at construction time — which is fine for uniform boundaries where psi=1 is implicit, but wrong for per-cell variable psi.

**Workaround:** Use the classical-α cut-cell family `E4_1` (cut-cell coefficients are CSE-expanded symbolic functions of alpha and psi). Spline cut-cell is unavailable.

**Fix:** Plan 42.10b (deferred). Needs a design decision on coefficient caching (psi-indexed lookup table? Per-call solve with caching on repeated psi values?). Substantial scope.

### `alpha[1] >= 197/288` constraint

**Problem:** The C++ `E4_1` struct enforces `alpha[1] >= 197/288 ≈ 0.684` in its constructor (cut-cell singularity avoidance). Brady-Livescu's feasible region has `α₁ ≈ 0.16`, well below this bound.

**Workaround:** When validating optimizer winners against C++, use `"E4u"` (uniform boundary, no psi dependency) instead of `"E4"` (cut-cell). L8 records `cpp_cutcell_violates_197_288` diagnostic flag.

**Fix:** Not needed if we're only doing uniform-domain tests. For cut-cell + classical-α, the constraint is a legitimate physics requirement.

## Harness / environment

### `.claude/skills/**` writes blocked

**Problem:** The harness's permission layer rejects all writes to `.claude/skills/**`, regardless of tool (Edit/Write/Bash) and regardless of `--dangerously-skip-permissions`. Plans 41.12c/d, 43.11c/d, 44.7c/d all got blocked on this.

**Workaround:** After ralph returns `RALPH_STATUS: blocked`, manually do the Skill.md edits in an interactive session.

**Fix:** Not yet tested — add to `.claude/settings.local.json`:
```json
{
  "permissions": {
    "allow": [
      "Edit(/workspace/.claude/skills/**)",
      "Write(/workspace/.claude/skills/**)"
    ]
  }
}
```
See `operating_conventions.md` for details.

### nlopt may or may not install cleanly

**Problem:** Before the recent Dockerfile update, `pip install nlopt` failed with a build error (missing `swig`). The Dockerfile now includes `swig`, but we haven't verified the fresh build's nlopt works.

**Workaround:** Use scipy's COBYQA (`method="COBYQA"` in scipy 1.14+, confirmed present at scipy 1.17 in this container) — it's the derivative-free trust-region equivalent of BOBYQA and works without nlopt.

**Fix:** Verify post-rebuild: `cd scripts/stencil_gen && uv sync && uv run python -c "import nlopt; print(nlopt.version)"`. If it fails, move nlopt from `dependencies` to `[project.optional-dependencies]` in `pyproject.toml`.

## Calibration / test artifacts

### `brady2d_sweep` and `brady2d_optima` keys not populated

**Problem:** Plan 43 defined the persistence schema for `known_values.json["brady2d_optima"]` and plan 42 defined `brady2d_sweep`. Neither key has been populated by running `--update-known-values` or `--persist` in a real sweep.

**Workaround:** The corresponding regression tests (`TestRegressionBrady2DOptima`) skip gracefully when the key is absent.

**Fix:** Run a single seeded optimization with `--update-known-values` to populate at least one entry, verify the regression test activates. Nothing complex; just hasn't been done.

### `TestRegressionGV` is dormant

**Problem:** Plan 40 added the GV-optimal persistence path (`--include-gv --update-known-values`). The tests are gated on `known_values.json["{kernel}_gv"]` entries being present. No production run has persisted these, so `TestRegressionGV` currently skips all 4 methods.

**Workaround:** Same — gracefully skipped.

**Fix:** Run `python -m sweeps tension --scheme E4 --include-gv --update-known-values` + Gaussian + MQ equivalents to populate. Non-urgent.

### `brady2d_calibration` `max_layer` sensitivity

**Problem:** `python -m stencil_gen.brady2d_cli --run-calibration --max-layer N --update-known-values` *overwrites* existing entries. If you ran at `max_layer=6` previously and then run at `max_layer=3`, you lose layer4–6 data. Caught and fixed during plan 44 (item 44.6d).

**Workaround:** Always run calibration at the highest `max_layer` you care about. Don't mix depths.

**Fix:** Add `--merge` flag that preserves higher-depth fields when lower-depth results are being written. ~20 lines. Minor priority.

## Framework gaps

### No multi-objective optimization (yet)

Deferred to plan 45. Single-objective + feasibility cliff covers ~80% of use cases cleanly.

### No multi-fidelity Bayesian optimization

Deferred to plan 46. The manual cascade (staged cheap-inner + expensive-validator) is a pragmatic approximation.

### No Brady-Livescu 1D Euler objective

Deferred to plan 47. Our analytical cascade is stricter than BL's linear test but cannot see nonlinear blow-up modes that only a full Euler simulation can exhibit.

### No 3D / tensor-product beyond 2D

L7 is 2D by construction. Most applications targeting 3D would go through the compiled C++ solver via the bridge.

### E6, E8 schemes

No Python derivation pipeline for them. `SCHEME_PARAMS` in `sweeps/_common.py` has only E2 and E4. The C++ side has `E6u_1` and `E8u_1` structs, so if you derive the weights in Python the bridge could score them. Real work, no active demand.

## Miscellaneous sharp edges

### Set `SYMPY_CACHE_SIZE=50000`

Before *any* SymPy-heavy command (stencil derivation, calibration, any brady2d run). Default 1000 causes catastrophic slowdowns. The session memory (`/home/user/.claude/projects/-workspace/memory/stencil_derivation_status.md`) has more context.

### Do not run `shoccs` in concurrent parallel threads from the same tempdir

The bridge uses `tempfile.TemporaryDirectory` per call to avoid this. If you hand-roll parallelism around `run_cpp_brady2d`, preserve the isolation.

### Plan files occasionally get linter-modified between iterations

This is fine and expected (review passes add follow-up items, correct stale criteria). The system-reminder notification lets you know. Don't try to revert these.

### Git CLAUDE.md and project rules override session instructions

`/workspace/CLAUDE.md` tells you to set `SYMPY_CACHE_SIZE=50000`, use `uv run`, use `cmake --build build`, etc. Don't override these without reason. The user can be annoyed by cargo-culted reconfiguration.
