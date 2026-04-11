# Phase 40: Unify Stability and Group Velocity in the Sweep Optimization Pipeline

**Goal:** Bring group velocity (GV) analysis into the sweep optimization pipeline as a secondary objective alongside the existing eigenvalue stability check, so that optimal stencil parameters minimize GV error among the stable feasible set. Also expose GKS-style group-velocity diagnostics as advisory output.

**Depends on:** Phase 39 (complete — sweep extraction and code simplification done)

**Background — current state of the two pipelines:**

| Aspect | Eigenvalue stability | Group velocity |
|---|---|---|
| Canonical API | `phs.stability_eigenvalue(n,p,q,eps,kernel,nu,nextra)` → `max Re(λ(-D_bc))` (`phs.py:807`) | `interior_group_velocity(p,nu,xi)`, `boundary_group_velocity(...)`, `psi_sweep_group_velocity(...)`, `gks_group_velocity_check(D,xi)` (`group_velocity.py:504,533,718,964`) |
| Threshold / scalar | `< STABILITY_TOL = 1e-10` (binary feasible/infeasible) | `profile.gv_error` array, `profile.cutoff_xi`, `PsiSweepResult.min_C`, `has_sign_reversal` (no canonical scalar yet) |
| Cost per call | `np.linalg.eigvals` on dense `n×n` (O(n³), n≤160 → sub-ms) | O(N_xi × stencil_width) numpy ops (microseconds) |
| Sweep integration | All 6 sweeps consume it via `sweeps/_common.py` constants | **Zero** — no sweep imports `group_velocity` |

**Multi-objective prior art:** `tension_penalty_sweep.py` (`eval_point` at lines 36–58) already handles two objectives via the *feasible-then-minimize* pattern: stability is a hard constraint (`se < STABILITY_TOL`); among feasible points, the conservation deficit is minimized. **This is the exact pattern we will reuse for GV.** GV becomes the secondary objective (or third, in penalty sweeps), never overriding stability.

**Why not GV-as-stability-replacement:** `gks_group_velocity_check` is *necessary not sufficient* for instability (test docstring at `test_group_velocity.py:1015–1017`). It will be exposed as an advisory diagnostic, not a feasibility gate.

**`known_values.json` compatibility:** The regression tests in `test_phs.py` (lines ~1358–1630) only access named keys (`{kernel}.epsilon`, `tension.sigma`, `params`, `stable_at`). Adding new keys (e.g. `tension.gv_error`, `tension_gv`, etc.) is non-breaking; existing tests will ignore them.

**Read first:**
- `scripts/stencil_gen/sweeps/_common.py` (`STABILITY_TOL`, `SCHEME_PARAMS`, `SweepResult`, helpers)
- `scripts/stencil_gen/sweeps/tension_penalty_sweep.py` (the multi-objective prior art — `eval_point`, `run_joint_sweep_coarse`)
- `scripts/stencil_gen/sweeps/tension_sweep.py` (simplest 1D sweep — testbed for new GV objective)
- `scripts/stencil_gen/sweeps/epsilon_sweep.py` (parallel structure)
- `scripts/stencil_gen/sweeps/__main__.py` (CLI dispatch — where new subcommands go)
- `scripts/stencil_gen/sweeps/known_values.json` (schema, current keys)
- `scripts/stencil_gen/stencil_gen/group_velocity.py` (lines 504–730 — interior/boundary/cut-cell APIs)
- `scripts/stencil_gen/stencil_gen/phs.py` (lines 807–840 — `stability_eigenvalue` and `_from_matrix`)
- `scripts/stencil_gen/tests/test_phs.py` (lines 1358–1630 — `TestRegression*` JSON loaders)

**Test commands:**
```bash
# Targeted: just the new objectives module
cd scripts/stencil_gen && uv run pytest tests/test_sweep_gv_objectives.py -x -q

# Smoke: each sweep with --include-gv at minimal resolution (each <30s)
cd scripts/stencil_gen && uv run python -m sweeps tension --scheme E2 --n-sigma 5 --include-gv
cd scripts/stencil_gen && uv run python -m sweeps epsilon --scheme E2 --kernel gaussian --n-eps 5 --include-gv

# Regression tests (must still pass)
cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegression"
```

---

## Items

### 40.1 — Shared GV objective helpers in `sweeps/`

- [x] **40.1a** Create `sweeps/gv_objectives.py` with five thin scalar wrappers around the existing `group_velocity` API:
  - `interior_gv_error_max(p: int, nu: int = 1, n_xi: int = 200) -> float` — calls `interior_group_velocity`, returns `float(np.max(np.abs(profile.gv_error)))`.
  - `interior_cutoff_fraction(p: int, nu: int = 1, n_xi: int = 200) -> float` — returns `profile.cutoff_xi / np.pi` (1.0 = ideal, lower = earlier parasitic onset).
  - `boundary_gv_error_max(p, q, nextra, nu, sigma, kernel, n_xi=200) -> float` — calls `boundary_group_velocity`, returns the max over rows of `np.max(np.abs(prof.gv_error))`.
  - `cutcell_gv_min_C(scheme_params, psi_values, alpha_values, n_xi=200) -> tuple[float, bool]` — calls `psi_sweep_group_velocity`, returns `(result.min_C, result.has_sign_reversal)`.
  - `gv_score_from_matrix(D: np.ndarray, n_xi: int = 200) -> dict` — extracts boundary rows from a constructed `D` matrix and returns `{"max_gv_error": ..., "min_cutoff_xi": ...}` (this is the entry point sweeps that already build `D` will use to avoid rebuilding). Uses a leading-rows scanner (row's leftmost nonzero column == 0) to identify the left-boundary block from D alone.
  - Each function has a one-line docstring describing its semantics. Module docstring explains the "feasible-then-minimize" contract.
  - File: `scripts/stencil_gen/sweeps/gv_objectives.py` (new) — **done**
  - Verified: `uv run python -c "from sweeps.gv_objectives import ...; ..."` returns finite positive values; `gv_score_from_matrix(D)["max_gv_error"]` matches `boundary_gv_error_max` to machine precision for the E2 tension case at sigma=6.0.

- [x] **40.1b** Add unit tests for the five helpers in `tests/test_sweep_gv_objectives.py`:
  - For each helper, one assertion that it returns a finite positive float (or correct tuple) at known inputs (E2 interior, E4 boundary at the stored optimum sigma from `known_values.json`).
  - Note: `@pytest.mark.fast` is **not** applied because the project's `pyproject.toml` only registers a `slow` marker; unmarked tests already run in the default suite, and adding an unregistered marker would emit pytest warnings. Tests are simply not marked `slow`.
  - File: `scripts/stencil_gen/tests/test_sweep_gv_objectives.py` (new) — **done**
  - Verified: `uv run pytest tests/test_sweep_gv_objectives.py -x -q` → 7 passed in 0.92s. `tests/test_phs.py -k TestRegression` → 15 passed (no regressions).
  - Follow-up for 40.7b / 40.8b: this file already exists; those items just append tests.

- [x] **40.1c** Add the missing `cutcell_gv_min_C` unit test (gap from 40.1b — only four of the five helpers were actually tested):
  - Review pass on commit `8fb9f68` confirmed `tests/test_sweep_gv_objectives.py` has zero references to `cutcell_gv_min_C`. Plan item 40.1b explicitly required "For each helper, one assertion that it returns a finite positive float (or correct tuple) at known inputs"; this helper's tuple contract is currently uncovered, and downstream item 40.5a depends on it.
  - Use a small fixture: `from stencil_gen.temo import E2_1` for `scheme_params`, an empty `alpha_values={}` (the helper's `psi_sweep_group_velocity` defaults missing alpha symbols to 0 — see `group_velocity.py:763`), and `psi_values=np.linspace(0.05, 0.95, 5)` to keep runtime under one second.
  - Assertions: returned object is a 2-tuple; `np.isfinite(min_C)`; `isinstance(has_sign_reversal, bool)` (not `np.bool_`, since the helper wraps with `bool(...)`).
  - File: `scripts/stencil_gen/tests/test_sweep_gv_objectives.py` — **done**
  - Verified: `uv run pytest tests/test_sweep_gv_objectives.py -x -q` → 8 passed in 0.88s. The new test asserts only `np.isfinite(min_C)` (not strictly positive) since `psi_sweep_group_velocity` returns the most-negative C across the psi sweep — at E2_1 with empty alpha it's −32.3.

### 40.2 — Integrate GV into `tension_sweep` (testbed sweep)

- [x] **40.2a** Add `--include-gv` CLI flag to `tension_sweep.py`:
  - Default `False` so existing invocations behave identically.
  - When set, after the existing stability scan, evaluate `boundary_gv_error_max(p, q, nextra, nu, sigma, "tension")` for every coarse-grid sigma and store alongside the `(sigma, stab_eig)` tuples.
  - Argument plumbing only — no behavior change yet beyond a new column. Keep the change <60 lines.
  - File: `scripts/stencil_gen/sweeps/tension_sweep.py` — **done**
  - Implementation: `sweep_stability` now returns `(stab_results, gv_by_sigma)` where `gv_by_sigma: dict[float, float] | None` is populated once per sigma (grid-size-independent) when `include_gv=True`, else `None`. `run_tension_sweep` accepts `include_gv=` kwarg and threads it through; the returned summary dict gains a `gv_by_sigma` key. `main()` adds `--include-gv`, and `sweeps/__main__.py` adds the same flag on the tension subparser with passthrough. No existing output format changed — 40.2b owns the printing work.
  - Verified: `uv run python -m sweeps tension --scheme E2 --n-sigma 5 --include-gv` runs end-to-end without error; `uv run python -m sweeps tension --scheme E2 --n-sigma 5` (no flag) behaves identically to before; `uv run pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -k "TestRegression or gv_objectives"` → 23 passed (no regressions).

- [x] **40.2b** Print GV error column in `tension_sweep` table when `--include-gv` is set:
  - Extend `print_sweep_table` call (or wrap output) so the per-(n, sigma) table gains a `gv_err` column.
  - Among feasible (`stab_eig < STABILITY_TOL`) sigmas, log a "Best by GV error: sigma=…, gv_err=…" line in addition to the existing "widest stable range" output.
  - File: `scripts/stencil_gen/sweeps/tension_sweep.py` — **done**
  - Implementation: `print_sweep_table` in `_common.py` gained an optional `gv_by_param` kwarg (additive, default `None`); when present, the per-n table renders a fourth `gv_err` column. `tension_sweep.py` passes `gv_by_sigma` through and, after `report_stable_ranges`, computes the intersection of stable sigmas across all grid sizes (the feasible set) and prints `Best feasible by GV error: sigma=…, gv_err=…`. The feasible set being empty is reported, not crashed.
  - Verified: `uv run python -m sweeps tension --scheme E2 --n-sigma 5 --include-gv` shows the new column and `Best feasible by GV error: sigma=20.000000, gv_err=2.074641e+00`. The same command without `--include-gv` is byte-identical to the prior format (no column added). `uv run pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -k "TestRegression or gv_objectives"` → 23 passed.

- [x] **40.2c** Persist GV-optimal sigma to `known_values.json` under a new sub-key:
  - When `--update-known-values` AND `--include-gv` are both set, write `kv[scheme_key]["tension"]["gv_error"] = best_gv_err` and `kv[scheme_key]["tension_gv"] = {"sigma": ..., "gv_error": ..., "stable_at": [...]}`.
  - This is additive: the existing `tension.sigma` (widest-stability optimum) is unchanged. The new `tension_gv.sigma` is the GV-optimal feasible sigma.
  - File: `scripts/stencil_gen/sweeps/tension_sweep.py` — **done**
  - Implementation: `run_tension_sweep` now cross-checks the GV-optimal feasible sigma at grid sizes `{20,40,80,160}` (mirroring the existing stability-optimum cross-check) and returns `gv_sigma`, `gv_error`, `gv_stable_at` in the summary. `main()` writes `tension.gv_error` (additive on the existing entry) and a new top-level `tension_gv` dict only when both `--update-known-values` and `--include-gv` are set; without `--include-gv`, the existing `tension` entry is written without the `gv_error` key (unchanged behavior). Without `--update-known-values`, nothing is written.
  - Verified: `uv run python -m sweeps tension --scheme E2 --n-sigma 5 --include-gv --update-known-values` wrote both `E2_1.tension.gv_error = 2.074641…` and `E2_1.tension_gv = {sigma: 20.0, gv_error: 2.074641…, stable_at: [20,40,80,160]}`; `known_values.json` restored from pre-test snapshot after verification (the low-resolution `--n-sigma 5` run would have overwritten the stability optimum at 6.0 with 3.483637, so this smoke run was not committed to the JSON). `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -k "gv_objectives or TestRegression"` → 23 passed. The no-flag invocation (`--n-sigma 5` without `--include-gv`) still produces the same output as before.

- [x] **40.2d** Fix non-additive overwrite of `tension.gv_error` in `tension_sweep.main()`:
  - Review pass on commit `74df8c7` confirmed: when `--update-known-values` is invoked **without** `--include-gv`, `main()` rebuilds `tension_entry = {"sigma": ..., "stable_at": ...}` from scratch and assigns it to `kv[scheme_key]["tension"]`, silently dropping any `gv_error` key that a prior `--include-gv` run had populated. The 40.2c verification never exercised this round-trip — it only tested `--include-gv --update-known-values` once and then restored the JSON.
  - Fix: `main()` now reads the existing `kv[scheme_key].get("tension", {})` and `kv[scheme_key].get("tension_gv", {})` and merges only the keys this invocation owns (`sigma`, `stable_at`, and conditionally `gv_error`). Earlier-run keys survive a subsequent non-GV `--update-known-values` invocation.
  - File: `scripts/stencil_gen/sweeps/tension_sweep.py` — **done**
  - Verified: snapshot `known_values.json`; run `uv run python -m sweeps tension --scheme E2 --n-sigma 5 --include-gv --update-known-values` (writes `E2_1.tension.gv_error = 2.074641…` and `E2_1.tension_gv`), then `uv run python -m sweeps tension --scheme E2 --n-sigma 5 --update-known-values` (no `--include-gv`). After step 2, `E2_1.tension.gv_error` is still `2.074641143264805` and `E2_1.tension_gv` is unchanged. Snapshot restored. `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -k "TestRegression or gv_objectives"` → 23 passed.
  - **Carry-over note:** the same merge pattern must be used when implementing 40.3c (epsilon sweep), 40.4c (tension-penalty), and 40.5c (footprint) — they all face the identical "additive on existing entry" contract.

- [x] **40.2e** Add an automated regression test for the `tension_sweep.main()` merge pattern:
  - Review pass on commit `f9ce69b` confirmed the 40.2d fix is only exercised by manual "snapshot → run twice → inspect JSON → restore" verification. There is no test that will fail if a future refactor (or the carry-over implementations in 40.3c/40.4c/40.5c) re-introduces a non-merging assignment to `kv[scheme_key]["tension"]`. The 40.2c bug was exactly this kind of silent regression; without a test, 40.2d's guarantee is review-dependent.
  - Implementation: `test_tension_sweep_main_merges_known_values` in `tests/test_sweep_gv_objectives.py` monkeypatches `sweeps._common.KNOWN_VALUES_PATH` to a `tmp_path / "known_values.json"`, seeds it via `_seed_kv()` with `{"E2_1": {"tension": {"sigma": 6.0, "stable_at": [20,40,80], "gv_error": 1.234, "preexisting_extra_key": "survive"}, "tension_gv": {"sigma": 5.5, "gv_error": 1.234, "stable_at": [20,40]}}}`, then runs `tension_sweep.main` twice: first without `--include-gv` (asserts `tension.gv_error == 1.234`, `tension.preexisting_extra_key == "survive"`, and `tension_gv` unchanged byte-for-byte), then with `--include-gv` (asserts `preexisting_extra_key` still present, `tension.gv_error` refreshed to a finite float, and `tension_gv` refreshed). Note: tension_sweep does not re-export `KNOWN_VALUES_PATH`, so monkeypatching only `_common` is sufficient — the plan's "and the re-export in tension_sweep" was incorrect. The seed helper is a module-local `_seed_kv(path)`; the carry-over items 40.3c/40.4c/40.5c can extract it into a parameterized helper if/when their sweeps land.
  - File: `scripts/stencil_gen/tests/test_sweep_gv_objectives.py` — **done**
  - Verified: `uv run pytest tests/test_sweep_gv_objectives.py -x -q` → 9 passed in 1.09s. `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -x -q -k "TestRegression or gv_objectives or sweep_gv"` → 24 passed in 1.33s. The two `tension_sweep.main` invocations together add ~1s to the test runtime, well under the 2s budget.

- [x] **40.2f** Extend the 40.2e regression test to also pin the `tension_gv` merge:
  - Review pass on commit `7071482` confirmed: `_seed_kv` places the `preexisting_extra_key` sentinel only on the `tension` entry, not on `tension_gv`. After the `--include-gv` invocation, the test asserts `set(tension_gv) == {"sigma", "gv_error", "stable_at"}`, which passes whether `tension_sweep.main()` merges into the existing `tension_gv` dict (as 40.2d requires) or overwrites it with a fresh `{"sigma":..., "gv_error":..., "stable_at":...}` literal. Only half of the 40.2d fix is actually pinned. The carry-over items (40.3c/40.4c/40.5c) will replicate this same merge pattern for `{kernel}_gv`, `tension_penalty`, and `footprint.*`, so a robust template matters.
  - Implementation: `_seed_kv` now seeds `tension_gv` with an extra `"preexisting_gv_extra": "survive_gv"` key. The first (non-GV) invocation's exact-equality assertion now includes the sentinel, pinning that nothing touches `tension_gv` on that path. The second (GV) invocation's assertion was loosened from `set(tension_gv) == {"sigma","gv_error","stable_at"}` to `{"sigma","gv_error","stable_at"} <= set(tension_gv)` and gained `assert tension_gv["preexisting_gv_extra"] == "survive_gv"`, which fails iff `tension_sweep.main()` replaces the existing `tension_gv` dict with a fresh literal.
  - File: `scripts/stencil_gen/tests/test_sweep_gv_objectives.py` — **done**
  - Verified: `uv run pytest tests/test_sweep_gv_objectives.py -x -q -k "merges_known_values"` → 1 passed in 1.07s against the current merging implementation. Mutation check: temporarily replaced lines 284–288 of `tension_sweep.py` with `kv[scheme_key]["tension_gv"] = {"sigma": summary["gv_sigma"], "gv_error": summary["gv_error"], "stable_at": summary["gv_stable_at"]}` and the test failed with `KeyError: 'preexisting_gv_extra'` at the new assertion (then restored). `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -k "TestRegression or gv_objectives or sweep_gv or merges"` → 24 passed in 1.33s.

### 40.3 — Integrate GV into `epsilon_sweep` (Gaussian / multiquadric kernels)

- [x] **40.3a** Add `--include-gv` flag to `epsilon_sweep.py`, mirroring 40.2a:
  - `boundary_gv_error_max(p, q, nextra, nu, eps, kernel)` is the per-point call.
  - File: `scripts/stencil_gen/sweeps/epsilon_sweep.py` — **done**
  - Implementation: `sweep_stability` now returns `(results, gv_by_eps)` where `gv_by_eps: dict[float, float] | None` is populated once per epsilon (grid-size-independent) when `include_gv=True`, else `None`. The GV call uses `sigma=float(eps)` since `boundary_gv_error_max`'s internal parameter name is `sigma` but `boundary_group_velocity` uses it as the generic RBF shape parameter for all kernels (gaussian/multiquadric/tension — see `group_velocity.py:556`). `run_epsilon_sweep` accepts `include_gv=` kwarg and threads it through; the returned summary dict gains a `gv_by_eps` key. `main()` adds `--include-gv`, and `sweeps/__main__.py` adds the same flag on the epsilon subparser with passthrough. Argument plumbing only — the output format is unchanged (40.3b owns printing the new column and the "best feasible by GV error" line).
  - Verified: `uv run python -m sweeps epsilon --scheme E2 --kernel gaussian --n-eps 5 --include-gv` runs end-to-end without error; same command with `--kernel multiquadric` also runs; `uv run python -m sweeps epsilon --scheme E2 --kernel gaussian --n-eps 5` (no flag) produces identical output to before. `uv run pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -x -q -k "TestRegression or gv_objectives or sweep_gv or merges"` → 24 passed in 1.35s (no regressions).

- [x] **40.3b** Print GV column and "best feasible by GV" line in `epsilon_sweep`:
  - Same structure as 40.2b.
  - File: `scripts/stencil_gen/sweeps/epsilon_sweep.py` — **done**
  - Implementation: `run_epsilon_sweep` now passes `gv_by_param=gv_by_eps` to `print_sweep_table` (the kwarg added in 40.2b is reused). After `report_stable_ranges`, when `gv_by_eps is not None` it intersects per-grid stable epsilons (`se < STABILITY_TOL`) across every `n` in the sweep to form the feasible set, then prints `Best feasible by GV error: eps=…, gv_err=…` for the smallest GV error in that set, or `(no eps stable at every grid size)` if empty. The summary dict and the cross-check across `{20,40,80,160}` are deliberately untouched — those belong to 40.3c.
  - Verified: `uv run python -m sweeps epsilon --scheme E2 --kernel multiquadric --n-eps 5 --include-gv` shows the new column and `Best feasible by GV error: eps=10.000000, gv_err=2.143617e+00`. Same for `--kernel gaussian` (`eps=1.778279, gv_err=1.868048e+00`). `diff` of the no-flag vs `--include-gv` outputs at `--n-eps 5 --kernel gaussian` shows only the additive `gv_err` column and the new "Best feasible" line — every other line is byte-identical. `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -x -q -k "TestRegression or gv_objectives or sweep_gv or merges"` → 24 passed in 1.32s.

- [x] **40.3c** Persist `{kernel}_gv` keys in `known_values.json` for epsilon sweep:
  - `kv[scheme_key][kernel]["gv_error"] = ...` (additive on existing entry).
  - `kv[scheme_key]["{kernel}_gv"] = {"epsilon": ..., "gv_error": ..., "stable_at": [...]}` for the GV-optimal feasible epsilon.
  - File: `scripts/stencil_gen/sweeps/epsilon_sweep.py` — **done**
  - Implementation: `run_epsilon_sweep` now cross-checks the GV-optimal feasible epsilon at grid sizes `{20,40,80,160}` (mirroring the existing stability-optimum cross-check) and returns `gv_epsilon`, `gv_error`, `gv_stable_at` in the summary. `main()` uses the 40.2d merge pattern: reads the existing `kv[scheme_key].get(args.kernel, {})` and `kv[scheme_key].get(f"{args.kernel}_gv", {})` and merges only the keys this invocation owns (`epsilon`, `stable_at`, and conditionally `gv_error`). Without `--include-gv`, the existing `{kernel}` entry is written without the `gv_error` key, but any pre-existing `gv_error`/`{kernel}_gv` from an earlier `--include-gv` run survives. Without `--update-known-values`, nothing is written.
  - Verified: snapshot `known_values.json`; `uv run python -m sweeps epsilon --scheme E2 --kernel gaussian --n-eps 5 --include-gv --update-known-values` wrote `E2_1.gaussian.gv_error = 1.8680479…` and `E2_1.gaussian_gv = {epsilon: 1.778279, gv_error: 1.8680479…, stable_at: [20,40,80,160]}`; subsequent `... --update-known-values` (no `--include-gv`) kept both keys. Also verified with `--kernel multiquadric` → `E2_1.multiquadric_gv.epsilon = 10.0`. Snapshot restored (the low-resolution `--n-eps 5` run is not committed to JSON — a full-resolution sweep should populate the real values). The no-flag path (`--n-eps 5` without `--include-gv`, no `--update-known-values`) produces output identical to before this item. `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -x -q -k "TestRegression or gv_objectives or sweep_gv or merges"` → 24 passed in 1.34s.
  - **Carry-over for 40.4c / 40.5c:** same additive merge pattern (`dict(kv[scheme_key].get(key, {}))` then assign specific keys) applies to `tension_penalty` and `footprint.*` entries. Consider adding a regression test analogous to `test_tension_sweep_main_merges_known_values` when those items land.

- [x] **40.3d** Add an automated regression test for `epsilon_sweep.main()` merge pattern (mirrors 40.2e/40.2f for tension):
  - Review pass on commit `b6e4a75` confirmed that 40.3c implements the additive merge pattern in `epsilon_sweep.main()` for both `kv[scheme_key][args.kernel]` and `kv[scheme_key][f"{args.kernel}_gv"]`, but the only verification is the same manual "snapshot → run twice → inspect JSON → restore" workflow that 40.2e was created to retire for tension. Without an automated test, a future refactor of `epsilon_sweep.main()` (or a copy-paste from a non-merging template) can silently re-introduce the exact bug 40.2d fixed for tension.
  - Implementation: `test_epsilon_sweep_main_merges_known_values` added in `tests/test_sweep_gv_objectives.py`, mirroring `test_tension_sweep_main_merges_known_values`. Uses a sibling `_seed_kv_epsilon(path, kernel="gaussian")` helper (does not refactor the existing tension-specific `_seed_kv` — keeping the helpers parallel makes future carry-over for 40.4c/40.5c easier to scaffold). Monkeypatches `sweeps_common.KNOWN_VALUES_PATH` to `tmp_path / "known_values.json"`, seeds `{"E2_1": {"params": ..., "gaussian": {epsilon, stable_at, gv_error: 1.234, preexisting_extra_key: "survive"}, "gaussian_gv": {epsilon, gv_error, stable_at, preexisting_gv_extra: "survive_gv"}}}`, then runs `epsilon_sweep.main` twice with `--scheme E2 --kernel gaussian --n-eps 5 --n-values 20`: first without `--include-gv` (asserts `gaussian.gv_error == 1.234`, `gaussian.preexisting_extra_key == "survive"`, `gaussian_gv` byte-equal to seed), then with `--include-gv` (asserts `gaussian.preexisting_extra_key == "survive"`, `gaussian.gv_error` is finite float, `gaussian_gv["preexisting_gv_extra"] == "survive_gv"`, and `{"epsilon","gv_error","stable_at"} <= set(gaussian_gv)`).
  - File: `scripts/stencil_gen/tests/test_sweep_gv_objectives.py` — **done**
  - Verified: `uv run pytest tests/test_sweep_gv_objectives.py -x -q` → 10 passed in 1.28s. Mutation check 1: replaced the `kernel_entry = dict(kv[scheme_key].get(args.kernel, {}))` block with a non-merging `kernel_entry = {"epsilon": ..., "stable_at": ...}` literal — test failed at `kernel_entry["gv_error"] == 1.234` with `KeyError: 'gv_error'` (restored). Mutation check 2: replaced the `gv_entry = dict(kv[scheme_key].get(gv_key, {}))` block with a non-merging `gv_entry = {"epsilon": ..., "gv_error": ..., "stable_at": ...}` literal — test failed at `gv_entry["preexisting_gv_extra"] == "survive_gv"` with `KeyError: 'preexisting_gv_extra'` (restored). Final regression sweep: `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -x -q -k "TestRegression or gv_objectives or sweep_gv or merges"` → 25 passed in 1.53s.
  - **Carry-over for 40.4c / 40.5c:** when those land, replicate this pattern. Consider extracting `_seed_kv` / `_seed_kv_epsilon` into a single parameterized `_seed_kv_with_keys(path, scheme_key, primary_key, secondary_key)` helper at that point — three near-identical seeders is the threshold to deduplicate.

### 40.4 — Extend `tension_penalty_sweep` to track GV as a third objective

- [x] **40.4a** Augment `eval_point()` in `tension_penalty_sweep.py` to return `(stab_eig, deficit, gv_error)`:
  - Use `gv_objectives.gv_score_from_matrix(D)["max_gv_error"]` so we don't rebuild D.
  - All callers updated to unpack three values.
  - File: `scripts/stencil_gen/sweeps/tension_penalty_sweep.py` — **done**
  - Implementation: `eval_point` now imports `gv_score_from_matrix` from `.gv_objectives`, computes `gv_error = float(gv_score_from_matrix(D)["max_gv_error"])` from the same `D` matrix it already builds (no rebuild), and returns the 3-tuple `(se, deficit, gv_error)`. The return-type annotation is updated to `tuple[float, float, float]`. All four call sites in this file (`run_joint_sweep_coarse`, `run_penalty_effect`, `run_fine_sweep` main loop, `run_fine_sweep` grid-independence loop) unpack into `se, deficit, _gv` — the `_gv` is intentionally discarded; 40.4b owns plumbing it through to the accumulator and printer. `grep eval_point` confirms no other file calls it. Argument plumbing only — no behavior change yet beyond the additional return value.
  - Verified: `uv run python -m sweeps tension-penalty --scheme E2 --n-sigma 5 --n-gamma 5` runs end-to-end with output identical to before this item (same coarse-sweep, penalty-effect, and fine-sweep numbers). `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -x -q -k "TestRegression or gv_objectives or sweep_gv or merges"` → 25 passed in 1.49s.

- [x] **40.4b** In `run_joint_sweep_coarse`, accumulate `best_stable_gv` (smallest GV error among feasible points):
  - New accumulator alongside `best_stable_deficit`. Same feasible-then-minimize pattern.
  - Print a third "best stable GV error" line at end of phase 1 output.
  - File: `scripts/stencil_gen/sweeps/tension_penalty_sweep.py` — **done**
  - Implementation: `run_joint_sweep_coarse` now captures the already-returned `gv` from `eval_point` (renaming `_gv` → `gv` in the loop) and maintains `best_stable_gv`, `best_stable_gv_sigma`, `best_stable_gv_gamma` accumulators updated inside the existing `se < STABILITY_TOL` branch (no extra pass over the grid). After the existing "Best stable point (lowest deficit)" block, the function prints a parallel "Best stable point (lowest GV error)" block. The returned summary dict gains `best_stable_gv_sigma`, `best_stable_gv_gamma`, and `best_stable_gv` (None when no feasible point exists); existing keys are untouched. `run_penalty_effect` and `run_fine_sweep` still discard their `_gv` — they are not the phase-1 accumulator and plan 40.4c owns the persistence work.
  - Verified: `uv run python -m sweeps tension-penalty --scheme E2 --n-sigma 5 --n-gamma 5` prints `Best stable point (lowest GV error): sigma=20.0000, gamma=0.0000, gv_error=2.074641e+00` (matches the 40.2b tension-only result at the same coarse grid). Phase 2 and phase 3 output is byte-identical to before. `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -x -q -k "TestRegression or gv_objectives or sweep_gv or merges"` → 25 passed in 1.47s.

- [x] **40.4c** Persist three-way optimum to `known_values.json`:
  - Review pass on commit `5cd827e` confirmed: 40.4b added `best_stable_gv` / `best_stable_gv_sigma` / `best_stable_gv_gamma` to the dict returned by `run_joint_sweep_coarse`, but `run_tension_penalty_sweep` (lines 350–356) does NOT thread any of those keys into its own returned summary. Without that plumbing 40.4c cannot read the value in `main()`.
  - Implementation: `run_tension_penalty_sweep` now reads `coarse["best_stable_gv_sigma"]`, `coarse["best_stable_gv_gamma"]`, `coarse["best_stable_gv"]` and threads them into its returned summary under `gv_sigma`, `gv_gamma`, `gv_error` (rounded to 6 decimals for sigma/gamma, raw float for gv_error). When the coarse sweep found no feasible point, all three are `None` — no unpacking errors at downstream call sites. `best_sigma` / `best_gamma` (sourced from `fine`) remain the eigenvalue-stability optimum.
  - `main()` applies the 40.2d/40.3c merge pattern to `kv[scheme_key]["tension_penalty"]`: read `tp_entry = dict(kv[scheme_key].get("tension_penalty", {}))`, assign `sigma`/`gamma`/`stable_at`, and conditionally set `gv_error` when `summary["gv_error"] is not None`. No `--include-gv` flag — `eval_point` always computes `gv_error` (40.4a decision), so the only gating is "coarse sweep found a feasible point". A second top-level `tension_penalty_gv` entry is written using the same merge pattern when `summary["gv_sigma"] is not None`, with keys `sigma`/`gamma`/`gv_error`. Note: tension-penalty's coarse sweep is single-grid (n=40 only), so `tension_penalty_gv` has **no `stable_at` list** in this commit — **40.4d (next item) closes that gap by adding an explicit cross-grid stability re-check at the GV optimum**, and 40.4e's regression test must then assume `stable_at` is present.
  - File: `scripts/stencil_gen/sweeps/tension_penalty_sweep.py` — **done**
  - Verified: snapshot `known_values.json`; `uv run python -m sweeps tension-penalty --scheme E2 --n-sigma 5 --n-gamma 5 --update-known-values` wrote `E2_1.tension_penalty = {sigma: 4.0, gamma: 0.1, stable_at: [20,40,80], gv_error: 2.074641…}` and `E2_1.tension_penalty_gv = {sigma: 20.0, gamma: 0.0, gv_error: 2.074641…}`. Snapshot restored (the low-resolution `--n-sigma 5 --n-gamma 5` run is not committed — a full-resolution sweep should populate the real values). Without `--update-known-values` the smoke run produces output identical to before (byte-for-byte through phases 1/2/3; only new end-of-phase-1 "Best stable point (lowest GV error)" block from 40.4b). `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -x -q -k "TestRegression or gv_objectives or sweep_gv or merges"` → 25 passed in 1.46s.
  - **Carry-over for 40.5c:** same additive merge pattern applies to `footprint.*` entries. 40.4e will automate the regression test for this merge.

- [ ] **40.4d** Cross-grid stability check for the persisted `tension_penalty_gv` (sigma, gamma) optimum:
  - Review pass on commit `a5e7f5d` confirmed: 40.4c persists `kv[scheme_key]["tension_penalty_gv"] = {"sigma": ..., "gamma": ..., "gv_error": ...}` with **no `stable_at` field at all**, diverging from the convention set by 40.2c (`tension_gv.stable_at = [20,40,80,160]`) and 40.3c (`{kernel}_gv.stable_at = [...]`). The plan's justification ("coarse sweep is single-grid (n=40 only) — `tension_penalty_gv` has **no `stable_at` list**") does not hold up: the coarse sweep being single-grid means GV was *computed* at n=40, but verifying that the GV-optimal `(sigma, gamma)` pair is stable at additional grid sizes is just one extra `eval_point(nn, sigma, gamma, ...)` call per `nn`, not a re-execution of the coarse sweep. Without that check, a future consumer loading `tension_penalty_gv` from `known_values.json` has no guarantee the point is stable at any grid size other than n=40.
  - For the E2 verification in 40.4c the GV optimum happened to be `(sigma=20.0, gamma=0.0)`, which is identical to the tension-only sweep optimum that 40.2c already cross-checked at `{20,40,80,160}` — so the missing check was masked. For E4 (or any future scheme) where the GV optimum has `gamma > 0`, no such serendipitous coverage exists, and the persisted point could be silently unstable at higher resolution.
  - Implementation: in `run_tension_penalty_sweep` (after `run_joint_sweep_coarse` returns and before building the summary dict), if `coarse["best_stable_gv_sigma"] is not None`, loop over `nn in sorted({20, 40, 80, 160})` calling `eval_point(nn, gv_sigma, gv_gamma, p=p, q=q, nextra=nextra, nu=nu)` and collect the `nn` values whose `se < STABILITY_TOL` into `gv_stable_at`. Print the per-grid status the same way `tension_sweep.run_tension_sweep` does (lines 207–220). Thread `gv_stable_at` through the returned summary as `gv_stable_at`. In `main()`, when writing `tp_gv_entry`, set `tp_gv_entry["stable_at"] = summary["gv_stable_at"]` (additive — preserves any pre-existing `stable_at` only if the new check yields the same shape, but since the merge pattern is `dict(...).get(...)` followed by key assignment, the new value overwrites the old one for this specific key, which is correct because the new check is authoritative).
  - File: `scripts/stencil_gen/sweeps/tension_penalty_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps tension-penalty --scheme E2 --n-sigma 5 --n-gamma 5 --update-known-values` should print the per-grid stability re-check for the GV optimum and write `E2_1.tension_penalty_gv.stable_at = [20, 40, 80, 160]`. Then snapshot-restore `known_values.json`. The full regression sweep `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -x -q -k "TestRegression or gv_objectives or sweep_gv or merges"` must remain green.

- [ ] **40.4e** Add an automated regression test for `tension_penalty_sweep.main()` merge pattern (mirrors 40.2e/40.2f and 40.3d):
  - Without this, the 40.4c additive-merge guarantee is review-dependent and the same overwrite bug that 40.2d had to fix can be silently re-introduced by a future refactor or by 40.5c's copy-paste.
  - Implementation: `test_tension_penalty_sweep_main_merges_known_values` in `tests/test_sweep_gv_objectives.py`. Add a sibling `_seed_kv_tension_penalty(path)` helper (do not refactor the existing `_seed_kv` / `_seed_kv_epsilon` yet — three near-identical seeders is the threshold to deduplicate, and 40.5c will likely add a fourth, at which point a single parameterized helper makes sense). Seed `{"E2_1": {"params": ..., "tension_penalty": {"sigma": 6.0, "gamma": 0.0, "stable_at": [20,40,80], "gv_error": 1.234, "preexisting_extra_key": "survive"}, "tension_penalty_gv": {"sigma": 5.5, "gamma": 0.1, "gv_error": 1.234, "preexisting_gv_extra": "survive_gv"}}}`. Monkeypatch `sweeps._common.KNOWN_VALUES_PATH` to `tmp_path / "known_values.json"`. Run `tension_penalty_sweep.main(["--scheme","E2","--n-sigma","5","--n-gamma","5","--update-known-values"])` once and assert: `tension_penalty["preexisting_extra_key"] == "survive"`, `tension_penalty["gv_error"]` is a finite float (refreshed), and `tension_penalty_gv["preexisting_gv_extra"] == "survive_gv"`. After 40.4d also assert `set(tension_penalty_gv["stable_at"]) <= {20,40,80,160}` and that it is a non-empty list (the GV optimum at n=5,n_gamma=5 should be feasible at least at n=40, the coarse-sweep grid). There is no separate "non-GV" invocation path here (unlike 40.2e/40.3d) because tension-penalty has no `--include-gv` flag.
  - **Mutation check before committing:** temporarily replace the merge in `main()` with a non-merging literal and confirm the test fails at the `preexisting_extra_key` assertion; restore. Repeat for the `tension_penalty_gv` literal. Both mutations must produce a `KeyError`.
  - File: `scripts/stencil_gen/tests/test_sweep_gv_objectives.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_gv_objectives.py -x -q -k "tension_penalty_sweep_main_merges"` — must pass, and the full regression sweep `pytest tests/test_sweep_gv_objectives.py tests/test_phs.py -x -q -k "TestRegression or gv_objectives or sweep_gv or merges"` must remain green.

### 40.5 — Cut-cell GV integration with `footprint_sweep`

- [ ] **40.5a** Add `--include-gv` flag to `footprint_sweep.py`:
  - For each (scheme, nextra) combination, after the existing stability scan, call `cutcell_gv_min_C(scheme_params, psi_values=np.linspace(0.05, 0.95, 19), alpha_values, n_xi=200)`.
  - Reuse the alpha_values already extracted by the sweep — do not re-derive.
  - File: `scripts/stencil_gen/sweeps/footprint_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps footprint --n-sigma 5 --include-gv`

- [ ] **40.5b** Print `min_C` and `has_sign_reversal` columns in the footprint table when `--include-gv` is set:
  - Sign reversal is the parasitic-mode signature for cut-cell stencils — deserves a visible flag.
  - Among feasible footprints (existing stability gate), report "Best footprint by min_C" alongside the existing "Smallest stable footprint" line.
  - File: `scripts/stencil_gen/sweeps/footprint_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps footprint --n-sigma 5 --include-gv`

- [ ] **40.5c** Persist cut-cell GV summary in `known_values.json` under each footprint entry:
  - `kv["footprint"]["{key}"]["min_C"] = ...`, `kv["footprint"]["{key}"]["has_sign_reversal"] = ...`
  - File: `scripts/stencil_gen/sweeps/footprint_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps footprint --n-sigma 5 --include-gv --update-known-values`

### 40.6 — New `gv-stability-pareto` sweep subcommand

- [ ] **40.6a** Create `sweeps/gv_stability_pareto.py` exporting `main(argv) -> int`:
  - For a chosen scheme + parameter (sigma for tension, epsilon for Gaussian/MQ), compute `(stab_eig, gv_error)` over a fine 1D grid.
  - Output: a markdown table sorted by sigma, plus a final "Pareto-optimal points" section listing non-dominated (stable, low gv) entries.
  - No `--update-known-values` — this sweep is for research / docs only.
  - Keep total file <200 lines.
  - File: `scripts/stencil_gen/sweeps/gv_stability_pareto.py` (new)
  - Test: `cd scripts/stencil_gen && uv run python -c "from sweeps.gv_stability_pareto import main; raise SystemExit(main(['--scheme','E2','--param','tension','--n-points','11']))"`

- [ ] **40.6b** Register `gv-stability-pareto` subcommand in `sweeps/__main__.py`:
  - Add `subparsers.add_parser("gv-stability-pareto", ...)` block with args `--scheme`, `--param {tension,gaussian,multiquadric}`, `--n-points`.
  - Add `if args.command == "gv-stability-pareto":` dispatch with lazy import.
  - Add to `_run_all`'s `sweeps` list with quick-mode resolution.
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps gv-stability-pareto --scheme E2 --param tension --n-points 11`

### 40.7 — GKS group-velocity check as advisory diagnostic

- [ ] **40.7a** Add `--check-gks` flag to `tension_sweep.py` and `epsilon_sweep.py`:
  - When set, after picking the optimum parameter, build the `D` matrix at the optimum and call `gks_group_velocity_check(D, xi_array)`.
  - Print any returned `GKSModeInfo` entries with `is_outgoing=True` as `WARNING: outgoing boundary mode at xi=…` lines.
  - **Does not** alter the optimum or update `stable_at`. Purely advisory output.
  - Files: `scripts/stencil_gen/sweeps/tension_sweep.py`, `scripts/stencil_gen/sweeps/epsilon_sweep.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps tension --scheme E2 --n-sigma 5 --check-gks`

- [ ] **40.7b** Add a smoke test for `--check-gks` on a known-stable scheme:
  - Test asserts the call exits 0 (no crash) and produces no false-positive errors.
  - File: `scripts/stencil_gen/tests/test_sweep_gv_objectives.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_gv_objectives.py -x -q -k "gks_advisory"`

### 40.8 — Regression tests for GV-augmented `known_values.json`

- [ ] **40.8a** Add `TestRegressionGV` class in `test_phs.py`:
  - Loads `known_values.json` (gracefully skipping if absent).
  - For each scheme that has a `tension_gv` or `{kernel}_gv` entry: rebuild the D matrix at the stored sigma/epsilon and assert `gv_score_from_matrix(D)["max_gv_error"] <= stored_value * 1.1` (10% tolerance for floating-point variation).
  - Mark all class methods with `@pytest.mark.fast`.
  - File: `scripts/stencil_gen/tests/test_phs.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_phs.py -x -q -k "TestRegressionGV"`

- [ ] **40.8b** Add a smoke test that exercises `gv_score_from_matrix` on a small precomputed D:
  - Hardcoded tiny matrix, deterministic result.
  - File: `scripts/stencil_gen/tests/test_sweep_gv_objectives.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_sweep_gv_objectives.py -x -q -k "from_matrix"`

### 40.9 — Documentation updates

- [ ] **40.9a** Update `scripts/stencil_gen/docs/sweeps_reference.md` with a new "Group velocity objectives" section:
  - Document `--include-gv`, `--check-gks`, and the new `gv-stability-pareto` subcommand.
  - Document the new `known_values.json` keys: `*.gv_error`, `tension_gv`, `{kernel}_gv`, `footprint.*.min_C`.
  - Explain the feasible-then-minimize contract: stability is hard, GV is soft secondary.
  - File: `scripts/stencil_gen/docs/sweeps_reference.md`
  - Test: (no test — documentation only; rendered check by reading the file)

- [ ] **40.9b** Update `scripts/stencil_gen/docs/group_velocity_reference.md` with a "Sweep integration" section:
  - Cross-reference the helpers in `sweeps/gv_objectives.py`.
  - Note that `gks_group_velocity_check` is exposed as `--check-gks` advisory only (necessary not sufficient).
  - File: `scripts/stencil_gen/docs/group_velocity_reference.md`
  - Test: (no test)

- [ ] **40.9c** Update the `stencil-sweeps` skill at `.claude/skills/stencil-sweeps/SKILL.md`:
  - Add `gv-stability-pareto` to the CLI quick reference.
  - Add a one-line bullet about `--include-gv` and `--check-gks`.
  - File: `.claude/skills/stencil-sweeps/SKILL.md`
  - Test: (no test)

### 40.10 — Quick-mode integration in `_run_all`

- [ ] **40.10a** Have `sweeps all --quick` exercise the new GV path on at least one sweep:
  - In `__main__.py`'s `_run_all` list, augment the `tension` entry with `--include-gv` when `--quick` is set, so the smoke path covers the new objective wiring.
  - This guards against silent regressions in the GV integration.
  - File: `scripts/stencil_gen/sweeps/__main__.py`
  - Test: `cd scripts/stencil_gen && uv run python -m sweeps all --quick`

---

## Ordering

```
40.1a → 40.1b → 40.1c   (objectives module + tests; everything else depends on these; 40.1c closes a coverage gap)
  ↓
40.2a → 40.2b → 40.2c → 40.2d → 40.2e → 40.2f   (tension_sweep testbed; 40.2d is a correctness fix from review of 74df8c7; 40.2e is an automated regression test from review of f9ce69b; 40.2f closes a tension_gv merge-coverage gap from review of 7071482)
  ↓
40.3a → 40.3b → 40.3c → 40.3d   (epsilon_sweep mirrors tension_sweep — do once tension is verified; 40.3d is an automated regression test for the merge pattern from review of b6e4a75, mirroring 40.2e/40.2f)
  ↓
40.4a → 40.4b → 40.4c → 40.4d → 40.4e   (tension_penalty 3-way objective; 40.4c expanded after review of 5cd827e to spell out the summary plumbing + additive merge pattern; 40.4d added after review of a5e7f5d to fix the missing cross-grid stability check on the persisted `tension_penalty_gv` entry; 40.4e adds the merge regression test mirroring 40.2e/40.2f/40.3d)
  ↓
40.5a → 40.5b → 40.5c   (cut-cell footprint integration)
  ↓
40.6a → 40.6b           (new pareto sweep subcommand)
  ↓
40.7a → 40.7b           (GKS advisory)
  ↓
40.8a → 40.8b           (regression tests for new known_values keys)
  ↓
40.9a → 40.9b → 40.9c   (docs — only after API is stable)
  ↓
40.10a                   (final integration smoke in _run_all --quick)
```

Independent branches that may be parallelized after 40.1 lands:
- 40.2 / 40.3 / 40.4 / 40.5 are all independent additive changes to separate sweep files.
- 40.7 (GKS advisory) can be done in parallel with 40.5.

---

## Completion Criteria

- `sweeps/gv_objectives.py` exists with five documented helpers and unit tests.
- Every existing sweep (`tension`, `epsilon`, `tension-penalty`, `footprint`) accepts `--include-gv` and produces a GV column without changing the eigenvalue-based optimum.
- `tension_sweep` and `epsilon_sweep` accept `--check-gks` and report any outgoing boundary modes as warnings.
- `python -m sweeps gv-stability-pareto --scheme E2 --param tension --n-points 11` runs and prints a Pareto table.
- `known_values.json` contains additive `*.gv_error`, `tension_gv`, `{kernel}_gv`, and `footprint.*.min_C` keys for at least the E2 scheme — populated by running each sweep with `--include-gv --update-known-values`.
- `TestRegressionGV` class in `test_phs.py` passes against the populated keys.
- `cd scripts/stencil_gen && uv run pytest tests/ -x -q` still passes in <30s (no new slow tests added to the default suite).
- `cd scripts/stencil_gen && uv run python -m sweeps all --quick` runs end-to-end without error and exercises the new GV path on at least one sweep.
- Documentation (`sweeps_reference.md`, `group_velocity_reference.md`, `stencil-sweeps` skill) is updated.
- Eigenvalue stability remains the *only* hard feasibility gate everywhere; GV is exclusively a secondary objective or advisory diagnostic. Existing optima in `known_values.json` are unchanged unless `--update-known-values` is rerun with `--include-gv`.
