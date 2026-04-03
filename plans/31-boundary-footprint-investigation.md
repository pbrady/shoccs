# Phase 31: Boundary Footprint Sweep — nextra as Stability Hyperparameter

**Goal:** Determine whether increasing the boundary closure footprint (nextra) enables
the tension spline kernel to achieve full eigenvalue stability for E4.  Phases 29-30
held nextra=0 fixed, giving only 2 extra DOF per boundary row — the tightest possible
closure.  The E2 case that achieves machine-precision stability uses nextra=1.  Sweeping
nextra ∈ {0, 1, 2, 3} may reveal that the O(1e-5) instability floor is not fundamental
to E4 but rather an artifact of insufficient boundary DOF.

**Depends on:** Phase 30 tension spline engine (committed: `stencil_gen/phs.py`)

**Priority:** High — if nextra>0 enables E4 stability, the entire optimization problem
for 4th-order boundary stencils is solved by the tension spline approach.

**Read first:**
- `scripts/stencil_gen/stencil_gen/phs.py` (tension kernel, `build_diff_matrix_rbf`, `max_real_eigenvalue`)
- `scripts/stencil_gen/tests/test_phs.py` (existing sweeps for reference)
- `scripts/stencil_gen/stencil_gen/temo.py` (`compute_dimensions`, `SchemeParams` — nextra parameter)

**Test commands:**
```bash
cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_phs.py -x -v -k "Footprint or Phase31"
```

---

## Background

### The nextra parameter

The TEMO dimension formulas for nu=1 (first derivative) are:
- `t = p + q + 1 + nextra` (boundary stencil width)
- `r = q + 1 + nextra` (number of boundary rows)

Each boundary row has `t - (q+1) = p + nextra` free parameters beyond polynomial
accuracy.  These are the DOF that the RBF kernel fills.

| Scheme | p | q | nextra | r | t | Extra DOF/row | Total extra DOF |
|--------|---|---|--------|---|---|---------------|-----------------|
| E2_1   | 1 | 1 | 1      | 3 | 4 | 2             | 6               |
| E4 nx=0| 2 | 3 | 0      | 4 | 6 | 2             | 8               |
| E4 nx=1| 2 | 3 | 1      | 5 | 7 | 3             | 15              |
| E4 nx=2| 2 | 3 | 2      | 6 | 8 | 4             | 24              |
| E4 nx=3| 2 | 3 | 3      | 7 | 9 | 5             | 35              |

The DOF growth is rapid: nextra=2 gives 3× the total DOF of nextra=0.  More DOF means
the RBF kernel has more room to distribute smoothness while satisfying accuracy constraints.

### Why this matters

The Phase 30 conclusion that the E4 instability floor (~1e-5) is "fundamental" was based
on testing only nextra=0.  But the E2 case (which IS stable) uses nextra=1 by default.
The question is: does increasing nextra for E4 provide enough additional flexibility for
the tension spline to find a stable boundary closure?

---

## 31.1 — Generalize Matrix Builder for Arbitrary nextra

### 31.1a — Extend `build_diff_matrix_rbf` to accept `nextra` parameter

Currently `build_diff_matrix_rbf` computes dimensions using a hardcoded formula.
Extend it to accept an explicit `nextra` parameter (defaulting to 0 for backward
compatibility).  The only change is the dimension calculation:
- `t = p + q + 1 + nextra`
- `r = q + 1 + nextra`

Everything else (boundary row construction, interior stencil, right-side reflection)
stays the same — the RBF system naturally handles wider stencils on more points.

Also extend `build_diff_matrix_mixed_epsilon` and `build_diff_matrix_rbf_penalty`
similarly.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: nextra=0 gives identical results to current code (regression)
- Test: nextra=1 gives a larger matrix with more boundary rows

### 31.1b — Similarly extend `max_real_eigenvalue`

Pass `nextra` through to the matrix builder.
- File: `scripts/stencil_gen/stencil_gen/phs.py`
- Test: E2_1 with nextra=1 matches existing Phase 30 results

---

## 31.2 — Nextra Sweep for E4 with Tension Kernel

### 31.2a — Quick diagnostic: E4 tension at nextra=1

Before a full sweep, test the single most important case:
- E4 with nextra=1, tension kernel, sweep σ ∈ [0, 50]
- If max Re(λ) < STABILITY_TOL at some σ: the hypothesis is confirmed
- If not: check whether the instability floor dropped significantly from 3e-5
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestFootprintE4Quick`
- Test: `test_nextra_1_sigma_sweep` — print table and assert the sweep completes

### 31.2b — Full nextra × σ sweep for E4

Sweep nextra ∈ {0, 1, 2, 3} × σ ∈ [0, 50] at n=40:
- For each (nextra, σ): compute max Re(λ)
- Report the minimum instability per nextra
- Key question: does the instability floor decrease monotonically with nextra?
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestFootprintSweep`
- Test: `test_nextra_sweep_e4_tension` — full sweep with printed comparison table
- Assertion: nextra=1 instability floor < nextra=0 floor (minimal expected improvement)

### 31.2c — Nextra sweep for E4 with tension + soft conservation penalty

Repeat with the (σ, γ) joint optimization from Phase 30:
- For each nextra: find optimal (σ*, γ*) pair
- Does the penalty help more at larger nextra? (more DOF to trade)
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestFootprintPenalty`
- Test: `test_nextra_penalty_e4` — sweep nextra × σ × γ

### 31.2d — If stable nextra found: identify the minimum boundary footprint

If some nextra achieves stability:
1. Report the minimum nextra that works
2. Extract the stencil weights at (nextra*, σ*)
3. Compute the implied alpha values
4. Compare the boundary stencil dimensions with existing C++ stencils
5. Assess: is the wider stencil acceptable for production use?
   (wider boundary = more computation per boundary point, but boundary points
   are a small fraction of total grid)
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestStableFootprint`

---

## 31.3 — Cross-Scheme Validation

### 31.3a — Verify E2 is unaffected

Confirm that E2_1 (which already has nextra=1) retains its machine-precision stability
at the existing optimal σ.  Also check nextra=0 for E2 (should be worse).
- File: `scripts/stencil_gen/tests/test_phs.py`

### 31.3b — Try E6 and E8 if E4 succeeds

If wider footprint enables E4 stability, test the same approach on higher-order schemes:
- E6: p=3, q=5 — how much nextra is needed?
- E8: p=4, q=7 — scales further?
- This determines if the approach generalizes or if each order needs more nextra
- File: `scripts/stencil_gen/tests/test_phs.py`, class `TestFootprintHigherOrder`

---

## 31.4 — Analysis and Conclusions

### 31.4a — Scaling analysis

Quantify how the minimum nextra for stability scales with scheme order:
- E2 (q=1): nextra=1 → total extra DOF = 6
- E4 (q=3): nextra=? → total extra DOF = ?
- E6 (q=5): nextra=? → total extra DOF = ?
- Is there a pattern?  (e.g., nextra = q - 1, or total DOF ~ q²)
- File: `scripts/stencil_gen/tests/test_phs.py`

### 31.4b — Cost assessment

For the stable E4 configuration:
- How many extra boundary points compared to nextra=0?
- What is the impact on stencil sparsity and bandwidth?
- Is the computational cost acceptable for production?
- Compare: wider boundary closure (one-time cost per grid setup) vs
  multi-alpha optimization (expensive, repeated)
- Document in plan

### 31.4c — Update plan with conclusions

Summarize findings and determine next steps:
- If successful: plan Phase 32 for integrating tension stencils into codegen/C++
- If not: document the remaining gap and alternative directions
- File: `plans/31-boundary-footprint-investigation.md`

---

## Implementation Order

1. **31.1a** — Extend matrix builders for nextra parameter
2. **31.1b** — Extend max_real_eigenvalue for nextra
3. **31.2a** — Quick E4 nextra=1 diagnostic (first result!)
4. **31.2b** — Full nextra × σ sweep (key result: does floor drop?)
5. **31.2c** — Nextra × σ × γ sweep with penalty
6. **31.2d** — Extract stencil weights if stable nextra found
7. **31.3a** — E2 cross-validation
8. **31.3b** — E6/E8 if E4 works
9. **31.4a** — Scaling analysis
10. **31.4b** — Cost assessment
11. **31.4c** — Conclusions

---

## Key Files

| File | Role |
|------|------|
| `scripts/stencil_gen/stencil_gen/phs.py` | **Modified** — add nextra to matrix builders |
| `scripts/stencil_gen/tests/test_phs.py` | **Modified** — add footprint sweep tests |
| `plans/31-boundary-footprint-investigation.md` | **This file** — updated with results |

## Performance Notes

- Each eigenvalue computation at n=40 takes ~1ms
- The nextra × σ sweep at 4 × 100 resolution: ~400 evals, < 1 second
- The nextra × σ × γ sweep at 4 × 50 × 20: ~4000 evals, < 5 seconds
- Wider stencils (nextra=3, t=9) produce slightly larger augmented systems (14×14 vs 11×11)
  but this is negligible for the RBF solve
- Main cost is the eigenvalue computation which scales as O(n³) — still fast at n=40
