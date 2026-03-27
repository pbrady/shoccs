# Phase 20: SymPy Stencil Derivation Pipeline

**Goal:** Build a Python/SymPy pipeline that derives finite difference stencil coefficients (interior, uniform-mesh boundary, and TEMO cut-cell) symbolically, then generates optimized C++ code matching the existing `src/stencils/` patterns. Replace the slow Mathematica notebook workflow.

**Depends on:** None (standalone tooling, does not modify C++ solver code)

**Priority:** Active — unblocks generation of new stencil schemes (T4, T6, T8 cut-cell variants, higher-order 2nd-derivative schemes).

**Read first:**
- `plans/stencil-derivation-math-reference.md` (key equations and procedures)
- `src/stencils/stencil.hpp` (Stencil concept, `info` struct, type-erased wrapper)
- `src/stencils/E4u_1.cpp` (uniform boundary stencil — simplest complete reference)
- `src/stencils/E2_1.cpp` (cut-cell 1st derivative — primary TEMO validation target)
- `src/stencils/E2_2.cpp` (cut-cell 2nd derivative — secondary TEMO target, no free params)
- `src/stencils/polyE2_1.cpp` (cut-cell with interpolation constraints — code pattern reference only, NOT a derivation target)
- `src/stencils/E8u_1.cpp` (largest uniform stencil — 7 free params)

**Test commands:**
```bash
# Run the pipeline's own test suite
cd scripts/stencil_gen && uv run pytest -v

# Validate generated stencils match existing C++ (numerical comparison)
cd scripts/stencil_gen && uv run python -m stencil_gen validate

# Generate a stencil and diff against existing
cd scripts/stencil_gen && uv run python -m stencil_gen generate E4u_1 --diff
```

---

## Current State

- Stencil coefficients are derived in Mathematica notebooks (slow, not version-controlled)
- The C++ stencil files in `src/stencils/` were generated from Mathematica output
- No Python infrastructure exists for stencil derivation
- SymPy 1.14.0 is available; `uv` (at `/usr/local/bin/uv`) manages the Python environment
- **All Python commands must be run via `uv run` from `scripts/stencil_gen/`** — this ensures
  the correct virtualenv and dependencies are used. Never use bare `python` or `pip`.
- 7 stencil implementations exist: E2_1, E2_2, E4_2, E4u_1, E6u_1, E8u_1, polyE2_1

---

## Sub-Plans

| Section | Plan File | Status | Summary |
|---------|-----------|--------|---------|
| 20.1 | (inline below) | Done | Package scaffolding |
| 20.2 | `20.2-interior-stencils.md` | Not started | Interior stencil derivation (Taylor series matching) |
| 20.3 | `20.3-boundary-stencils.md` | Not started | Uniform mesh boundary stencils + conservation |
| 20.4 | `20.4-codegen.md` | Not started | C++ code generation (CSE, printer, struct emission) |
| 20.5 | `20.5-temo-cut-cell.md` | Not started | TEMO cut-cell extension (ψ-dependent stencils) |
| 20.6 | (inline below) | Not started | CLI and documentation |

### 20.1 — Project scaffolding (Done)

- [x] **20.1a** Package scaffold created: `scripts/stencil_gen/` with pyproject.toml, __init__.py, __main__.py
  - SymPy and pytest in regular dependencies, managed by uv
  - Placeholder test passing: `cd scripts/stencil_gen && uv run pytest -v`

### 20.6 — CLI and documentation

- [ ] **20.6a** Implement CLI in `__main__.py`:
  - `uv run python -m stencil_gen generate <scheme>` — derive and emit C++ to stdout
  - `uv run python -m stencil_gen validate` — run numerical validation against existing C++
  - `uv run python -m stencil_gen list` — list available schemes
  - File: `scripts/stencil_gen/__main__.py`

- [ ] **20.6b** Add README with usage examples:
  - How to generate a new stencil scheme
  - How to validate against existing C++
  - Performance expectations (E8 < 0.5s)
  - File: `scripts/stencil_gen/README.md`

---

## Performance Budget

| Operation | Target | Strategy |
|-----------|--------|----------|
| Interior derivation (any order) | <0.01s | Tiny linear system, linsolve() |
| E4 uniform boundary | <0.05s | linsolve() per row, manual conservation |
| E8 uniform boundary | <0.5s | Same, larger system |
| E2 TEMO cut-cell | <0.05s | QQ(psi) fraction field |
| E4 TEMO cut-cell | <0.1s | QQ(psi) fraction field |
| CSE + code generation | <0.01s | sympy.cse() on final expressions |

## Key SymPy Performance Rules

1. **Always** use `linsolve()`, never `solve()` (12x faster)
2. **Always** use `Rational`, never float (exact arithmetic)
3. **Always** use `xreplace()`, never `subs()` for simple substitutions (3x faster)
4. **Use** `QQ(psi)` fraction field for TEMO substitutions (24,000x faster than naive)
5. **Never** call `simplify()` — use `cancel()` instead (11x faster)
6. **Call** `cancel()` once per row after solve, not per operation
7. **Apply** `cse()` only at the final code generation step
