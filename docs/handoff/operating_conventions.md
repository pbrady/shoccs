# Operating Conventions

Practical patterns established over plans 40–44 that a new agent should follow (or consciously decide to deviate from).

## Running ralph_wiggum

The repo has a `ralph_wiggum.sh` at the root. It non-interactively executes implementation plan files iteration by iteration. Each iteration runs one plan item, then a review pass that can modify the plan (add follow-ups, fix stale notes, etc.).

**Launch pattern:**

```bash
# Launch from repo root, NOT from subdirectories
cd /workspace
./ralph_wiggum.sh --plan plans/NN-plan-name.md --mode work --max-iterations 50 --allow-dirty
```

- `--mode work`: executes the plan's `[ ]` items in order. (Other mode: `--mode review` just reviews recent commits.)
- `--max-iterations 50`: cap. Plans typically finish in 10–40 iterations.
- `--allow-dirty`: required because our worktree often has untracked `.claude/skills/**` files, notes, etc. Safe.
- **CWD matters:** Launch from `/workspace`. Previous attempt to run from `scripts/stencil_gen/` failed with exit code 127 (`no such file or directory: ./ralph_wiggum.sh`).

**Run it in the background:**

Use the `Bash` tool with `run_in_background=true`. The harness sends a completion notification when done; do not poll git log or output files while waiting (see `feedback_no_polling` memory).

**Status protocol:** Each plan item must end with:
```
RALPH_STATUS: committed|done|blocked
RALPH_SUMMARY: one-line summary
```
Items that ralph finds legitimately blocked (e.g., `.claude/skills/**` writes, see below) return `RALPH_STATUS: blocked`, which exits with a nonzero code. That's not a failure of ralph — it's "please handle the blocker manually."

## Known hang: `run_in_background=true` + zsh `kill -0` polling

**Pattern observed 2026-04-22/23, hung ralph twice for ~20 h each:**

Claude (the work-pass subprocess ralph spawns) sometimes invokes pytest with `run_in_background=true` and then writes a *Bash tool call* containing a polling wait-loop of the form:

```bash
while kill -0 $(pgrep -f "pytest ..." | head -1) 2>/dev/null; do sleep 5; done
tail -50 /tmp/.../baa0kus93.output
```

The bug: when `pgrep` returns nothing (test already finished), `$(pgrep ... | head -1)` expands to empty. **In zsh, `kill -0` with no/empty argument defaults to the current process group and returns 0 (success)**, so the loop sleeps another 5 s and polls again, forever. The pytest tests themselves finish normally (verified: `130 passed, 11 skipped in 178.73 s` on the occurrence we diagnosed) — only the wait wrapper hangs. From the outside this looks like a stuck `claude -p` invocation with ~1% CPU rate.

**Mitigation in ralph (as of this doc):** `build_work_prompt` and `build_review_prompt` in `ralph_wiggum.sh` now explicitly tell the subprocess:
- Tests run in the FOREGROUND — never set `run_in_background=true` on pytest/ctest/test commands.
- If something truly must be backgrounded, wait via `TaskOutput` / `BashOutput` with `block=true`, not a shell polling loop.
- If a shell poll is truly unavoidable, quote the PID and guard on emptiness:
  ```bash
  pid=$(pgrep -f "pattern" | head -1)
  while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do
    sleep 5; pid=$(pgrep -f "pattern" | head -1)
  done
  ```

**Detection cheat sheet:** if ralph stops committing but the `claude` process is alive, check `ps --forest` for a child shell running a `kill -0` / `sleep` loop and `ls -la /proc/<claude-pid>/fd/` for an open `iteration-NNNN-work.log`. `<1 % CPU rate` over many minutes with an established connection to api.anthropic.com is the signature. Kill the child shell (`kill <shell-pid>`) to unstick the current iteration, or kill the whole ralph bash to restart.

## Harness block on `.claude/skills/**`

**Pattern observed in plans 41, 43, 44:** All four skill-file update items (41.12c/d, 43.11c/d, 44.7c/d) returned `RALPH_STATUS: blocked`. The harness's permission layer rejects all writes to `.claude/skills/**` regardless of tool (`Edit`, `Write`, `Bash cp`) and regardless of `--dangerously-skip-permissions`. The permission layer is enforced outside the tool's authority.

**Workaround so far:** After ralph returns blocked, manually Edit the skill files in an interactive session. The four skills files that have been hand-updated this way:
- `.claude/skills/stencil-sweeps/SKILL.md`
- `.claude/skills/group-velocity-analysis/SKILL.md`

**To unblock for automation** (not yet attempted): add to `.claude/settings.local.json`:

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

Then re-run the blocked items. Not tested; may or may not survive the harness's policy overlay.

## Plan file conventions

Plans live in `/workspace/plans/NN-plan-name.md`, numbered sequentially. Each plan follows a consistent skeleton (copy from an existing one when drafting):

### Structure

```markdown
# Phase NN: Short title

**Goal:** 1-3 sentence problem statement.

**Depends on:** Phase N-1 (if any).

**Background:** ~200 words explaining why this matters, with concrete
inputs/outputs.

**What this plan does NOT do:** (explicit non-goals — crucial for scope)

**Read first:** list of files the implementer should read first.

**Test commands:** the narrow commands to run per item.

---

## Items

### NN.1 — Section title

- [ ] **NN.1a** One-sentence description of what this item does.
  - More detail as needed (implementation hints, inputs/outputs, edge cases).
  - File: `path/to/file.ext`
  - Test: `command that verifies this item`

- [ ] **NN.1b** ...

---

## Ordering

```
NN.1a → NN.1b → NN.2a ...
```

---

## Completion Criteria

(bulleted list of observable outcomes — used by ralph to decide the plan is
done)
```

### Item sizing

An item must fit in one Claude Code session (≤300 lines of diff, ≤2 files). If an item touches 3+ files, split it. If it's unusually subtle (research code, new algorithm), break into skeleton + body + tests as separate items. Review passes frequently add `NN.Mx-followup` items for gaps; that's normal.

### Test commands per item

Must be narrow (single test file or `-k "TestFoo"` filter). The review pass runs every item's Test command, so slow tests compound. Mark slow tests with `@pytest.mark.slow` and exclude via `-m "not slow"` in the test command.

### Corrections from the review pass

The review pass is surprisingly good at catching:
- Stale completion criteria that don't match what landed
- Vacuous tests (e.g., `if x:` guards without assertions)
- Bounds / tolerance mismatches against committed code
- Missing gate/field plumbing

It creates `NN.My-followup` items for these. Ralph then executes them next iteration. Expect 20–40% of iterations to be these follow-ups, not the originally-planned items.

## Python environment

- **`uv` manages the `scripts/stencil_gen/` virtualenv.** Dependencies go in `scripts/stencil_gen/pyproject.toml`. Run `cd scripts/stencil_gen && uv sync` to install.
- **Run commands via `uv run`**: `uv run python -m stencil_gen.brady2d_cli ...`, `uv run pytest ...`.
- **Set `SYMPY_CACHE_SIZE=50000`** before any command that exercises SymPy (the stencil derivation pipeline): `SYMPY_CACHE_SIZE=50000 uv run ...`. Default cache size of 1000 causes severe slowdowns.
- **New deps as of this session**: `pymoo>=0.6`, `nlopt>=2.7`. The devcontainer Dockerfile was updated to include `swig` (needed for nlopt's build). Container was just rebuilt.

## C++ build and run

- Build directory: `/workspace/build`. Built with Ninja + Clang by default.
- `cmake --build build` — full build.
- `cmake --build build --target t-tension_E4u_1` — single test.
- `ctest --test-dir build -R "t-foo"` — run matching tests.
- `./build/src/app/shoccs path/to/config.lua` — run the simulation binary. Writes `logs/system.csv` with per-step metrics (`Time`, `Step`, `Linf`, ...).
- **Stencil coefficient match precision**: Python vs C++ agrees to ≥14 significant digits — use a tight tolerance (1e-12) in Python-reference regression tests.
- **New C++ stencil families go in**: `src/stencils/*.cpp`, register in `src/stencils/stencil.cpp::from_lua` dispatch, add to `src/stencils/CMakeLists.txt`, write unit test `src/stencils/*.t.cpp`.

## Python ↔ C++ bridge

The bridge (`stencil_gen/cpp_bridge.py`) is subprocess-based: Python writes a Lua config into a temp directory, invokes `shoccs`, parses `logs/system.csv`. Key points:

- Template Lua has `--{{N}}--`, `--{{T_FINAL}}--`, `--{{SCHEME_TABLE}}--` markers (explicit placeholder tokens, not Lua AST parsing).
- Each bridge call uses an isolated `tempfile.TemporaryDirectory` so concurrent runs don't collide.
- The bridge parses the CSV with explicit header handling; don't rely on `cut -d, -f4` shell pipelines (header rows can drift).

## `known_values.json` schema

Located at `scripts/stencil_gen/sweeps/known_values.json`. Write additively — regression tests access specific keys, not walk unknown keys, so adding top-level keys is non-breaking. Existing top-level keys:
- `E2_1`, `E4_1`: optimal kernel params (older sweeps)
- `footprint`: nextra variants
- `brady2d_calibration`: full `StabilityReport` serializations per family (plan 41, updated plan 44)
- `brady2d_sweep`, `brady2d_optima`: schema defined but not yet populated

Any new plan that persists results should add a new top-level key (e.g., `brady2d_pareto`). Don't shadow existing keys.

## Git hygiene

- Ralph creates many commits (~30–90 per plan). Each is tagged `NN.Ma` or `NN.Ma-followup`. The messages are informative.
- Don't squash or rebase these during a ralph run.
- Before launching ralph: commit or stash any unrelated changes so the `--allow-dirty` baseline is clean. The script uses `git_status_snapshot` for drift detection.
- **Don't run `git push`** without the user's explicit OK. The convention in this repo is commit-locally-only during plan execution.

## Testing philosophy

- **Regression tests** (`TestRegression*` in `test_phs.py`) consume `known_values.json` and re-verify stored optima within a tolerance. They skip gracefully when their key is absent. This lets plans persist values opportunistically without breaking tests.
- **Unit tests** go in `tests/test_<module>.py`. Use pytest markers: `@pytest.mark.slow` for anything > 1 s. Default runs skip slow tests; `-m slow` enables them.
- **Run narrow filters** via `-k "TestClassName"` or `-k "test_specific_function"`. Don't rely on running the full suite per plan item.

## Session memory files

The user uses a memory system at `/home/user/.claude/projects/-workspace/memory/` (per the auto-memory section of the system prompt). Relevant entries already present:
- `feedback_no_polling.md`: don't poll git log/status while waiting for a background task; wait for the notification.
- `stencil_derivation_status.md`: E4_1 generated, conservation is a soft constraint for q≥2.
- `user_numerical_methods.md`: user is expert in high-order FD; prefers strong BC enforcement, not SBP-SAT.
- `planning_checklist.md`: solver integration checklist lessons from the HYPRE work.

Check the memory index at `/home/user/.claude/projects/-workspace/memory/MEMORY.md` before making assumptions about user preferences.

## When in doubt

- **Ask one or two sharp decision questions before writing long plans.** The user repeatedly prefers "propose scope + flag decisions" over "write 500 lines and hope."
- **Split off risky items.** If a plan item might block on harness permissions, an optional dep, or a build system issue, make it the last item so the plan can be marked complete at the earlier checkpoint.
- **Verify runtime behavior** end-to-end, not just unit tests. Plan 43's optimizer had 94 passing unit tests but the CLI invocation exposed a real edge case (monotone objective) that unit tests alone would have missed. Run a representative CLI smoke after any significant plan completes.
