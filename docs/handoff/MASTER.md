# Handoff — Stencil Stability & Optimization Framework

> **This is the entry point.** Read this file first. Drill into the linked files as needed.

## What this codebase is

SHOCCS is a C++ Cartesian cut-cell high-order PDE solver (`/workspace/src/`). The user is Peter Brady — author of the Brady-Livescu 2019 and 2021 papers cited throughout. The Python-side tool at `/workspace/scripts/stencil_gen/` began as a SymPy stencil derivation pipeline and has grown (plans 40–44) into a layered stability-scoring and optimization framework that:

- Evaluates any scheme/kernel/parameter choice against a cascade of **eight** stability layers (L1 GV error → L2 rigorous Kreiss → L3 1D eigenvalue → L3r BL §4.2 reflecting-hyperbolic → L4 2D local GV → L5 anisotropy → L6 non-normality → L7 sparse 2D eigenvalue → L8 full C++ simulation)
- Optimizes boundary closures via scipy (Nelder-Mead, COBYQA, SHGO, differential_evolution, staged multi-fidelity)
- Validates winners end-to-end by running the compiled C++ `shoccs` solver with the discovered parameters passed through Lua

## Recent state (2026-04-15)

- Plans 40–44 are all complete. Git log from commit `843c974` forward shows the full trail.
- The devcontainer was just rebuilt with `pymoo>=0.6` and `nlopt>=2.7` added to `scripts/stencil_gen/pyproject.toml` (plus `swig` added to the Dockerfile for nlopt's build).
- Plan 45 (multi-objective Pareto via NSGA-II) was actively being scoped when this handoff was written.

## File guide (this folder)

Read in this order if you want the full picture:

1. **[framework_architecture.md](framework_architecture.md)** — what modules live where, their APIs, their dependency graph, and the C++/Lua/JSON integration surface. ~4000 words. The reference an agent needs to navigate the code.

2. **[completed_plans.md](completed_plans.md)** — one-section-per-plan summary of what plans 40–44 delivered, public entry points, and inline corrections discovered during execution. ~3500 words. Skip to a specific plan's section when you need to understand what that plan did.

3. **[scientific_findings.md](scientific_findings.md)** — 10 non-obvious discoveries from the work that a new agent cannot derive by reading code alone. Examples: tension-spline fails BL42 universally (verified via 898-eval DE); classical-α has multiple feasible basins (BL found 101); L7 operator has physically-correct positive spectral abscissa from `div(c) > 0`; BL §4.2 eigenvalues are `±i(2k-1)π/2` not `±ikπ`. Read this before drawing conclusions from any stability number.

4. **[next_steps.md](next_steps.md)** — plan 45 (queued), plans 46 and 47 (deeper queue), small follow-ups and backlog. Open decisions flagged per plan. Start here when the user says "continue" or "what's next."

5. **[operating_conventions.md](operating_conventions.md)** — how to launch ralph_wiggum, plan-file structure, the `.claude/skills/**` permission block, Python env conventions (`uv run`, `SYMPY_CACHE_SIZE`), C++ build commands, git hygiene. Read before touching anything.

6. **[known_limitations.md](known_limitations.md)** — deferred work, known sharp edges, incomplete features (monotone-objective pitfall, gate_layer auto-infer TODO, nlopt uncertainty, uncharted E2 classical-α 4D space, etc.).

## 30-second orientation

**The framework's top-level entry point:**

```python
from stencil_gen.brady2d_stability import brady2d_stability_score
report = brady2d_stability_score("E4", "classical", {"alpha": [-0.7733, 0.1624]}, max_layer=6)
print(report.overall_verdict, report.failed_layer)
```

**The optimizer's top-level entry point:**

```bash
uv run python -m sweeps optimize \
    --scheme E4 --kernel tension \
    --objective layer6.transient_growth_bound \
    --bounds 0.1 20 --method Nelder-Mead --n-restarts 8
```

**The C++ bridge's top-level entry point:**

```python
from stencil_gen.cpp_bridge import run_cpp_brady2d
result = run_cpp_brady2d("tension_E4u", {"sigma": 3.0}, N=31, t_final=10.0)
```

**The CLI dispatch lives at `python -m sweeps <subcmd>`**, with 11 subcommands registered: `epsilon`, `tension`, `tension-penalty`, `footprint`, `comparison`, `alpha`, `mixed-epsilon`, `gv-stability-pareto`, `brady2d`, `optimize`, `all`.

## Key artifacts to cite/reference

- **Plans:** `/workspace/plans/40-*.md` through `44-*.md`. All items `[x]` except four `.claude/skills/**` edits that were manually completed this session. Plan 45 doesn't exist yet.
- **Reference docs** (in `/workspace/scripts/stencil_gen/docs/`):
  - `brady2d_stability_reference.md` — L1–L8 cascade
  - `bl42_reference.md` — BL §4.2 reflecting-hyperbolic layer (L3r)
  - `optimization_reference.md` — optimizer framework
  - `pareto_reference.md` — multi-objective NSGA-II Pareto extension (plan 45)
  - `group_velocity_reference.md` — the GV module
  - `pipeline_reference.md` — end-to-end stencil derivation
  - `sweeps_reference.md` — sweep subcommands
  - `testing_reference.md` — test organization
- **C++ bridge doc:** `/workspace/docs/brady2d_cpp_bridge_reference.md`
- **Papers:** `/workspace/papers/BradyLivescu2019.pdf`, `BradyLivscu2021.pdf`, `StabilityAndGroupVelocity.pdf`, `GroupVelocityInFiniteDifferenceSchemes.pdf`
- **`known_values.json`:** `/workspace/scripts/stencil_gen/sweeps/known_values.json` — persistent calibration/sweep/optima data

## What to do if the user says "continue"

1. Read [MASTER.md](MASTER.md) (you're here), skim [next_steps.md](next_steps.md).
2. Verify the devcontainer rebuild took: `cd scripts/stencil_gen && uv run python -c "import pymoo; print(pymoo.__version__)"`.
3. If the user wants plan 45 (multi-objective Pareto): propose the scope from [next_steps.md](next_steps.md), resolve the three open decisions noted there, then write and execute the plan via ralph_wiggum.
4. If the user wants something else: [next_steps.md](next_steps.md) lists plans 46, 47, and backlog items. Confirm direction before writing.

## What to do if the user asks about a specific topic

- "How does the analytical stack work?" → [framework_architecture.md](framework_architecture.md), section "brady2d_stability.py"
- "What does plan N deliver?" → [completed_plans.md](completed_plans.md), section for plan N
- "Why does X fail / pass?" → [scientific_findings.md](scientific_findings.md)
- "How do I run this?" / "What's the CLI?" → [operating_conventions.md](operating_conventions.md)
- "Is this feature implemented?" → [known_limitations.md](known_limitations.md) (deferred/incomplete things) or [framework_architecture.md](framework_architecture.md) (things that exist)
- "What's next?" → [next_steps.md](next_steps.md)

## Session-context distillation

Three substantive findings from the session that should shape future work:

1. **The BL §4.2 layer (L3r) is a sharp discriminator.** All spline/RBF families (tension, Gaussian, multiquadric) that pass the traditional 1D advection stability test fail L3r. Only classical-α closures survive. This is both a scientific result and a practical design constraint for downstream consumers.

2. **Multi-objective is the right next move.** Plan 44's results exposed genuine trade-offs between three stability metrics — tension has excellent GV but fails BL42, classical has weaker GV but passes BL42 — which a Pareto front would make concrete. Plan 45 is sized, scoped, and has three open decisions (see [next_steps.md](next_steps.md)).

3. **The framework does real optimization.** It's no longer a scoring pipeline — it's a full optimizer with local/global/staged/multi-start drivers, CLI, persistence, and C++-validated winners. A multi-seed run on E4 classical-α with DE found `α = [-1.399, 0.293]` (different basin than BL's published `[-0.77, 0.16]`, both feasible), confirming the paper's multi-modality observation experimentally.

---

**One-line summary:** Python scoring cascade + optimizer + C++ bridge, five plans deep, all passing, ready for multi-objective extension via pymoo (just added to the rebuilt devcontainer).
