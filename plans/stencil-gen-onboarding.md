# Stencil Generation Pipeline: Onboarding Guide

> **Purpose:** Get a fresh agent up to speed on the SymPy stencil derivation pipeline,
> its current state, what works, what's broken, and what to read.

## Project Context

SHOCCS is a C++ cut-cell solver for PDEs. The stencils (finite difference coefficients)
come from two papers by Brady & Livescu:
- **2019 paper** (`papers/BradyLivescu2019.pdf`): Uniform mesh boundary stencils with
  conservation and stability optimization
- **2021 paper** (`papers/BradyLivscu2021.pdf`): TEMO cut-cell extension — stencils
  parameterized by ψ ∈ [0,1] (fluid line fraction)

A SymPy pipeline in `scripts/stencil_gen/` derives these stencils symbolically and
generates C++ code. The pipeline was built over Phases 20-27.

## What Works

- **Interior stencils** (`interior.py`): Taylor series matching for any (s, p, ν). Correct.
- **Uniform boundary stencils** (`boundary.py`, `conservation.py`): Validated against
  E4u_1, E6u_1, E8u_1 C++ code. Correct.
- **C++ code generation** (`codegen.py`, `printer.py`): Generates struct, tests, CSE. Works.
- **E2 cut-cell stencils** (`temo.py`): E2_1 and E2_2 validated against existing C++ code.
  These work because q=1 makes conservation exactly satisfiable.
- **370 Python tests**, all passing. C++ E4_1 builds and passes.

## What's Broken

The **E4_1 cut-cell stencil** (4th order, 1st derivative) has **singularities at ψ=0 and
ψ=1** in the near-interior row entries. The paper's Appendix A (Table A.4) shows the
actual E4_1 coefficients have NO singularities — boundary rows are polynomials in ψ,
and the near-interior row has a rational denominator `e0(ψ)` that is nonvanishing on [0,1].

The paper explicitly says: "Note that the αu terms are specified rather than solved for.
This avoids any singularities in the coefficients in the range ψ ∈ [0,1]."

Our pipeline produces entries with `ψ·(ψ-1)` denominator factors, requiring runtime
clamping that violates the TEMO design principle of singularity-free construction.

**Root cause:** The solution procedure (documented in `plans/stencil-gen-procedure.md`)
does not correctly follow the paper's construction. See that document for details.

## Files to Read (in order)

### 1. Mathematical Reference
- `plans/stencil-derivation-math-reference.md` — Key equations from both papers

### 2. The Papers (critical sections)
- `papers/BradyLivscu2021.pdf` pages 8-12 (Section 3: Construction of embedded stencils)
  - Section 3.1: Design Principles (TEM + continuity)
  - Section 3.2: Application of design principles (B_l(ψ) construction)
  - Section 3.3: Discrete conservation constraints
  - Section 3.4: Optimization for stability
- `papers/BradyLivscu2021.pdf` pages 28-30 (Appendix A: actual E4_1 coefficients + supplementary data description)
- `papers/BradyLivescu2019.pdf` pages 3-6 (Section 2: uniform mesh construction + conservation)

### 3. Pipeline Code
- `scripts/stencil_gen/stencil_gen/temo.py` — The TEMO construction (1500+ lines). Key functions:
  - `derive_uniform_boundary_for_temo` (~line 315): Builds the uniform boundary B_u
  - `construct_cut_cell_stencil` (~line 1353): Main TEMO procedure
  - `solve_temo_row` (~line 1263): Per-row Taylor solve (produces RATIONAL functions — the problem)
  - `solve_temo_row_polynomial` (~line 1349): Polynomial ansatz for boundary rows (Phase 27 fix)
  - `solve_conservation_fraction_free` (~line 1507): Conservation solve with zeros
  - `build_cut_cell_conservation_system` (~line 1430): Builds the conservation equations
- `scripts/stencil_gen/stencil_gen/boundary.py` — Uniform boundary row solver
- `scripts/stencil_gen/stencil_gen/conservation.py` — SBP conservation constraints
- `scripts/stencil_gen/stencil_gen/interior.py` — Interior stencil derivation
- `scripts/stencil_gen/stencil_gen/codegen.py` — C++ code generation

### 4. Existing C++ Stencils (reference implementations)
- `src/stencils/E2_1.cpp` — Working cut-cell stencil (q=1, conservation works)
- `src/stencils/E4_1.cpp` — Generated E4_1 (has singularity guards — WRONG)
- `src/stencils/E4u_1.cpp` — Uniform E4 boundary (correct, validated)
- `src/stencils/stencil.hpp` — Stencil concept and interface

### 5. Plans and Decision History
- `plans/stencil-gen-procedure.md` — **READ THIS**: documents what the pipeline does
  and where it diverges from the paper
- `plans/27-polynomial-construction.md` — Latest attempt (polynomial boundary rows)
- `plans/22-cut-cell-conservation.md` — Conservation investigation history
- `plans/meta.md` — Decision log (D-R18, D-R22 through D-R25)

### 6. Tests
- `scripts/stencil_gen/tests/test_temo.py` — E2_1 and E2_2 integration tests
- `scripts/stencil_gen/tests/test_e4_cut_cell.py` — E4_1 tests (including singularity tests)
- `scripts/stencil_gen/tests/test_boundary.py` — Uniform boundary validation

## Key Parameters

| Scheme | p | q | nextra | r | t | R | T | Free α | Zeros |
|--------|---|---|--------|---|---|---|---|--------|-------|
| E2_1 | 1 | 1 | 1 | 3 | 4 | 4 | 5 | 4 | none |
| E4_1 | 2 | 3 | 0 | 4 | 6 | 5 | 7 | 4 | α^u_{05}=0, α^u_{15}=0 |

Dimension formulas (Eq. 11): `t = p + q + 1 + nextra`, `r = q + 1 + nextra` (1st deriv only).

## Running the Pipeline

```bash
cd scripts/stencil_gen
uv run pytest tests/ -v --timeout=300          # all tests
uv run pytest tests/test_temo.py -v -k "E2"   # E2 tests only
uv run python -m stencil_gen list              # list schemes
```

## The Core Question

How should the TEMO construction produce cut-cell stencil entries that are:
1. **Polynomial in ψ** for boundary rows (no denominators)
2. **Rational with benign denominator** for the near-interior row (denominator nonvanishing on [0,1])
3. **Conservative** (SBP column sums = 0 for interior columns)
4. **Taylor-accurate** (order q for each boundary row)
5. **Singularity-free** on ψ ∈ [0,1]

The paper achieves all five. Our pipeline achieves 3 and 4 but fails 1, 2, and 5.
