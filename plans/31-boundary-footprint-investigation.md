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

### 31.1a — Extend `build_diff_matrix_rbf` to accept `nextra` parameter ✅

Already implemented in Phase 30.  All three matrix builders (`build_diff_matrix_rbf`,
`build_diff_matrix_mixed_epsilon`, `build_diff_matrix_rbf_penalty`) accept `nextra`
with correct dimension formulas.

### 31.1b — Similarly extend `max_real_eigenvalue` ✅

Already implemented.  `max_real_eigenvalue` passes `nextra` through to the builder.

---

## 31.2 — Nextra Sweep for E4 with Tension Kernel

### 31.2a — Quick diagnostic: E4 tension at nextra=1 ✅

- Test: `TestFootprintE4Quick::test_nextra_1_sigma_sweep` — PASSED
- **Result:** nextra=1 did NOT achieve stability.  Instability floor dropped only
  marginally from 7.7e-5 (nx=0) to 6.7e-5 (nx=1), a 1.15× improvement.
- The hypothesis that nextra=1 alone would unlock stability is **not confirmed**.

### 31.2b — Full nextra × σ sweep for E4 ✅

- Test: `TestFootprintSweep::test_nextra_sweep_e4_tension` — PASSED
- **Key result:** the instability floor is **flat at ~5e-5 across all nextra values**.

  | nextra | t | r | extra DOF | best σ | min max Re(λ) | status |
  |--------|---|---|-----------|--------|---------------|--------|
  | 0      | 6 | 4 | 8         | 38.6   | 5.5e-5        | unstable |
  | 1      | 7 | 5 | 15        | 29.8   | 4.9e-5        | unstable |
  | 2      | 8 | 6 | 24        | 2.9    | 7.2e-5        | unstable |
  | 3      | 9 | 7 | 35        | 15.0   | 5.0e-5        | unstable |

- The floor does NOT decrease monotonically with nextra.  nextra=1 is marginally
  better than nextra=0, but nextra=2 is worse, and nextra=3 is about equal.
- **Conclusion:** The O(1e-5) instability floor is NOT an artifact of insufficient
  boundary DOF.  It appears to be fundamental to the tension kernel + E4 combination,
  regardless of footprint size.

### 31.2c — Nextra sweep for E4 with tension + soft conservation penalty ✅

- Test: `TestFootprintPenalty::test_nextra_penalty_e4` — PASSED
- **Result:** No (nextra, σ, γ) combination achieves machine-precision stability.

  | nextra | t | r | extra DOF | γ=0 floor  | best σ*  | best γ*  | (σ,γ) floor | improvement |
  |--------|---|---|-----------|------------|----------|----------|-------------|-------------|
  | 0      | 6 | 4 | 8         | 8.2e-5     | 7.64     | 0.77     | 5.6e-5      | 31.8%       |
  | 1      | 7 | 5 | 15        | 9.6e-5     | 6.39     | 0.17     | 2.9e-5      | 70.1%       |
  | 2      | 8 | 6 | 24        | 9.5e-6     | 6.39     | 0.00     | 9.5e-6      | 0.0%        |
  | 3      | 9 | 7 | 35        | 6.2e-5     | 22.4     | 46.4     | 2.2e-5      | 64.8%       |

- **Key observations:**
  1. nextra=2 at γ=0 achieves the overall best floor (9.5e-6) — penalty does not help here.
  2. nextra=1 sees the largest penalty benefit (70% improvement), consistent with the idea
     that moderate DOF + penalty works best.
  3. nextra=2 γ=0 is ~6× better than nextra=0 γ=0, so nextra DOES matter for pure tension.
     But this contradicts 31.2b, suggesting the σ range matters — the penalty sweep used
     a finer grid near low σ values (logspace from 0.01) vs 31.2b's linear grid.
  4. Despite the improved floor at nextra=2, the O(1e-5) barrier is NOT breached.
- **Conclusion:** Neither nextra alone (31.2b) nor nextra+penalty (31.2c) can achieve
  machine-precision stability for E4 with the tension kernel.

### 31.2d — If stable nextra found: identify the minimum boundary footprint

N/A — no stable nextra was found.  Skipping per plan logic.

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

1. ✅ **31.1a** — Extend matrix builders for nextra parameter (already existed)
2. ✅ **31.1b** — Extend max_real_eigenvalue for nextra (already existed)
3. ✅ **31.2a** — Quick E4 nextra=1 diagnostic → nextra=1 marginally better, NOT stable
4. ✅ **31.2b** — Full nextra × σ sweep → floor flat at ~5e-5 across all nextra
5. ✅ **31.2c** — Nextra × σ × γ sweep with penalty → floor persists at ~1e-5
6. N/A **31.2d** — No stable nextra found; skipped
7. **31.3a** — E2 cross-validation
8. **31.3b** — E6/E8 if E4 works (likely N/A)
9. **31.4a** — Scaling analysis (likely N/A)
10. **31.4b** — Cost assessment (likely N/A)
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
