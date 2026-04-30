# Next Steps

> **Snapshot status (2026-04-15, refreshed 2026-04-30).** This file was
> drafted during plan-45 scoping and commits its pre-plan-45 state.
> Subsequent plans have landed:
> - Plan 45 (multi-objective Pareto via NSGA-II) landed 2026-04-24; see
>   `plans/45-pareto-optimization.md` and `pareto_reference.md`.
> - Plan 46 (hardening) landed (see `plans/46-hardening.md`).
> - Plan 47 (multi-fidelity Bayesian optimization via BoTorch) landed
>   2026-04-30; see `plans/47-mfbo.md` and
>   `scripts/stencil_gen/docs/mfbo_reference.md`.
> Any reference below to plans 45–47 as "queued", to `gate_layer`
> auto-infer as unimplemented, to multi-objective optimization as "not
> yet available", or to multi-fidelity BO as "deferred" is stale. **The
> next plan in queue is Plan 48 (Brady-Livescu 1D Euler reproduction).**

Three queued plans (45, 46, 47) and one small follow-up fix. Plan 45 is the most natural next step — the handoff conversation was actively designing it when we ran out of context.

## Immediate small follow-up: gate_layer auto-infer

**Discovered during session (not yet implemented):** When an optimization objective lives in a specific layer (e.g., `--objective layer_bl42.max_spectral_abscissa` or `--objective layer7.max_spectral_abscissa`), the default `gate_layer=3` is wrong. For `layer_bl42.*` objectives specifically, if the BL42 check fails at that value, the cascade short-circuits and the objective field is never extracted — every evaluation returns `+inf` and the optimizer can't see a gradient.

**Symptom seen in session:** A 20-minute DE run on tension E4 with `layer_bl42.max_spectral_abscissa` as the objective returned `best_objective = inf` across 2634 evaluations. The fix is manually passing `--gate-layer 2`.

**Proposed fix (~30 lines in `stencil_gen/optimizer.py`):** In `make_objective`, when `report_field` starts with `"layer_bl42."`, `"layer6."`, or `"layer7."`, auto-infer `gate_layer = <that layer's tier> - 1`. User can still override explicitly. Unblocks the common "optimize against my favorite stability metric" case.

Could be a standalone ~2-item micro-plan or rolled into plan 45 as item 45.0.

---

## Plan 45 — Multi-Objective Pareto Optimization

**Why next:** Plan 44 delivered three stability-informative metrics with real, visible trade-offs:

| Metric | Measures | Favors tension | Favors classical |
|---|---|---|---|
| `layer1.boundary_gv_err` | Dispersion quality | Low | Higher |
| `layer_bl42.max_spectral_abscissa` | Reflecting-BC stability | **Fails universally** | Passes |
| `layer6.transient_growth_bound` | Non-normal growth | Varies | Varies by basin |

A single-objective optimizer collapses these onto one metric. Pareto fronts expose the trade-off and inform the user's choice based on their problem.

**Dependencies:** `pymoo>=0.6` (just added to `pyproject.toml`; devcontainer rebuilt with `swig` to satisfy optional `nlopt` build).

### Proposed scope

**Core (MVP):**
1. `stencil_gen/pareto.py` with `ParetoResult` dataclass — set of `(params, objective_vector)` non-dominated points.
2. `make_multi_objective(scheme, kernel, objective_fields, gate_layer)` factory returning `f(x) -> np.ndarray`.
3. `run_nsga2(f, bounds, n_objectives, pop_size, n_gen, seed)` wrapping `pymoo.algorithms.moo.nsga2.NSGA2`.
4. CLI `python -m sweeps pareto --scheme E4 --kernel classical --objectives layer1.boundary_gv_err layer_bl42.max_spectral_abscissa --bounds -2 2 0.05 2 --n-gen 50`.
5. Persistence: write `sweeps/pareto_fronts/<scheme>_<kernel>_<objectives>.json` per-run (keeps known_values.json clean; fronts are naturally large).
6. Hypervolume reporting as a scalar progress metric.
7. Regression test: verify ≥3 stored Pareto points re-compute their objectives within tolerance.

**Extensions (include if scope permits):**
8. NSGA-III for ≥4 objectives (same API, different import).
9. Weighted scalarization complementary mode — `--objective "0.5*layer6.tgb + 0.5*layer1.gv_err"` — useful for "I care 80% about this, 20% about that" without a full Pareto study.
10. Simple ASCII scatter plot for 2D fronts (scaled to terminal cells).

**Out of scope:**
- Constraint-aware Pareto (pymoo `Problem.evaluate(F, G)` for explicit inequalities) — the L1–L3 cascade handles constraints adequately.
- 3D matplotlib plots (optional deps).
- Multi-fidelity multi-objective (defer to plan 47).

### Three open decisions (from session)

1. **Persistence schema:** write Pareto fronts to `known_values.json["brady2d_pareto"]` (all in one file), or to `sweeps/pareto_fronts/<scheme>_<kernel>_<objs>.json` per-run (cleaner, scales better)? **Session recommendation: separate files.**

2. **gate_layer auto-infer follow-up:** include as `45.0` (prerequisite) or punt to a standalone micro-plan? **Session recommendation: include as 45.0.**

3. **NSGA-II vs fallback:** if `pip install pymoo` fails in the rebuilt container, use a ~200-line pure-numpy NSGA-II implementation (well-documented, Deb et al. 2002). With the rebuilt container, pymoo should work — but worth sketching the fallback in the plan in case it doesn't.

### Expected plan size

Similar to plan 44: ~250–300 lines, ~20 items across 6–7 phases. Phases will look like:
- 45.0 gate_layer auto-infer fix
- 45.1 pareto module skeleton + ParetoResult
- 45.2 make_multi_objective factory
- 45.3 NSGA-II driver (pymoo-based)
- 45.4 CLI + persistence
- 45.5 Regression tests
- 45.6 Documentation + skill updates

---

## Plan 46 — Hardening (this plan)

**Status:** Active. Plan 45's review pass surfaced several recurring categories of bugs (CLI surfaces vs. library defaults, schema completeness, sibling non-determinism, vacuous tests, silent fallbacks). Most were fixed inline as in-plan follow-ups. Plan 46 addresses the **siblings** — the same patterns elsewhere in the codebase that the review pass didn't get to — and activates dormant `TestRegression*` infrastructure that has been sitting idle since plans 40–43 because the populating sweeps were never run. The result is a tighter regression net before plan 47 (Multi-Fidelity BO) starts piling on optimizer machinery. See `plans/46-hardening.md` for the item list.

---

## Plan 47 — Multi-Fidelity Bayesian Optimization (DONE 2026-04-30)

**Status:** landed. Plan 47 chose **BoTorch** (clean aarch64 wheels,
NumPy 2 compat) over Emukit/Trieste. The driver
(`stencil_gen/bo.py::run_mfbo`) fits a BoTorch ICM-GP surrogate jointly
over `(x, m)` (no Kennedy-O'Hagan AR1, since L3 ↔ L3r are different
physics rather than a refinement chain) and picks the next `(x, m)` via
cost-aware qMFKG.  CLI: `python -m sweeps bo`.  Per-run JSONs persist
under `sweeps/bo_runs/<scheme>_<kernel>_<mangled>_<seed>.json`.  Full
details, schema, CLI examples, and the AugmentedBranin synthetic
acceptance results (47.3k.4) live in
`scripts/stencil_gen/docs/mfbo_reference.md`.  See `plans/47-mfbo.md`
for the implementation history.

---

## Plan 48 — Brady-Livescu 1D Euler Reproduction (next)

**Why:** The paper's actual optimization objective is a full 1D nonlinear Euler RK4 simulation — a two-phase score where phase 1 = "did it stay stable to t_c" and phase 2 = "how monotone is the boundary energy" (via total-variation deviation). They cite finding **101 E4 schemes**, **16 E6**, **3 E8**, **1079 T4**, **16 T6**, **25 T8** from random restarts, all passing their linear *and* nonlinear tests. Reproducing this objective lets us:
- Validate our framework against a published reference.
- Discover schemes using the paper's exact discriminator.
- Extend easily to E6/E8 (which we currently can't score because there's no Python derivation pipeline for them yet).

**Cost:** needs a working Python 1D Euler RK4 solver (6 full simulations per objective evaluation, runs to t=10.5). Not trivial — the BL2D bridge uses the compiled C++ `shoccs` for L8. For 1D Euler we'd either:
- (a) Wire a new Lua config + `--system 1d-euler-gaussian-bump` bridge through the existing C++ solver, OR
- (b) Write a small Python 1D Euler solver (~200 lines of RK4 on a shock-capturing scheme).

**Option (a)** is much less work if the C++ solver already has a 1D Euler system with reflecting walls and a Gaussian bump IC. Worth checking `src/systems/` before planning.

**Scope:** medium-large depending on (a) vs (b). Adds `layer9_1d_euler_nonlinear` (ow, a new layer!) and extends the optimizer with a new objective class `blowup_or_monotonicity_score`.

---

## Plan backlog (smaller, no urgency)

- **E2 classical-α 4D optimization.** The E2_1 TEMO closure has 4 free α parameters (α[0..3]) and zero enforced bounds. Feasibility region is uncharted. A proper scoping task is to first run a feasibility pre-scan (random samples + L3 gate) to understand the shape of the feasible region before attempting optimization. ~50 lines of plan.

- **E6u / E8u Python derivation.** `SCHEME_PARAMS` in `_common.py` has only E2 and E4 today. Extending to E6/E8 means generating new boundary stencils via `temo.py` with (p=3, q=5, nextra=0) for E6 and (p=4, q=7, nextra=0) for E8. Reopens the old Brady-Livescu alpha-counting formula. The C++ side already has `E6u_1` and `E8u_1` structs (fixed-α or accepting arrays) — so the Python side is the bottleneck. ~150 lines of plan; could unlock plan-47 coverage of E6/E8 optimization.

- **Cut-cell runtime parameterization (C++).** Plan 42.10b: `tension_E4_1`, `gaussian_E4_1`, `multiquadric_E4_1` for cut-cell geometry. Each `nbs_floating` call has a different psi → either psi-indexed coefficient cache or per-call linear solve. Substantial scope; needs a design decision on the caching strategy before writing the plan.

- **Parameter-space visualization.** Add `matplotlib` as an optional dep, emit PDF/PNG plots of Pareto fronts, α-basin scatter, sweep landscapes. Nice-to-have for paper figures; not needed for the core work. ~80 lines of plan.

- **Brady-Livescu 2D C++ codegen bridge.** Today's bridge generates Lua + reads CSV. If we ever want to *generate new stencil families* as part of the optimization (not just vary parameters), we'd need to plug the `codegen` + build cycle into the loop. Would enable "let the optimizer propose a new boundary closure" research. Long-term.

---

## Where the next agent should start

If the user says "continue" without further direction, the right first action is:

1. Read `MASTER.md` (this folder's entry point) and skim `completed_plans.md`, `framework_architecture.md`, `scientific_findings.md` for context.
2. Confirm the BoTorch stack is in place: `cd scripts/stencil_gen && uv run python -c "import botorch, torch, gpytorch; print(botorch.__version__, torch.__version__, gpytorch.__version__)"`.
3. Plans 45–47 have all landed (see snapshot at the top of this file). The next plan in queue is **Plan 48 — Brady-Livescu 1D Euler reproduction**; scope it in more detail, decide between option (a) (Lua bridge for 1D Euler) and option (b) (Python RK4 solver), and get the user's sign-off before writing the plan.
4. Write and execute plan 48 via ralph_wiggum.

If plan 48 isn't the right next step for the user's current goal, `next_steps.md` surfaces the alternative backlog items.
