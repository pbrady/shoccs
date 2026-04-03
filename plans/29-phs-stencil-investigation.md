# Plan: Phase 29 — Polyharmonic Spline (PHS) Stencil Investigation

## Context

The stencil pipeline uses expensive numerical optimization to choose free parameters
(alphas) in boundary stencils. Literature research reveals that polyharmonic spline
interpolation (PHS+poly) provides a closed-form, no-optimization procedure for computing
FD weights — including boundary stencils — by minimizing smoothness functionals subject
to polynomial accuracy constraints. Each free alpha in our stencils corresponds to one
interior knot in an equivalent spline space.

This investigation will determine whether PHS-derived stencils match or outperform
optimizer-derived stencils, potentially replacing the expensive optimization loop
with a direct analytical construction.

**Approach**: Build a standalone PHS stencil module, compare against known-good stencils
on E2_1 (simplest) and E4_1 (production), check stability via total positivity and
eigenvalue analysis.

---

## 29.1 — PHS+Poly Stencil Derivation Engine

**File**: `scripts/stencil_gen/stencil_gen/phs.py` (new module)

Implement the core PHS+poly system for computing FD weights on arbitrary 1D point sets.

### 29.1a: Core PHS system builder

Given points `{x_0, ..., x_{n-1}}`, derivative order `nu`, evaluation point `x_eval`,
polynomial degree `q`, and PHS order `k`:

Build and solve the augmented system:

```
[Φ  P] [λ]   [d_Φ]
[P' 0] [μ] = [d_P]
```

where:
- `Φ_{ij} = φ(|x_i - x_j|)` with `φ(r) = r^(2k-1)` (1D PHS kernel)
- `P_{ij} = x_j^i` for `i = 0..q` (polynomial reproduction)
- `d_Φ_i = D^nu φ(|x_eval - x_i|)` (derivative of PHS kernel at eval point)
- `d_P_i = D^nu x_eval^i` (derivative of monomials at eval point)
- `λ` are the FD weights

**Function**: `phs_stencil_weights(points, x_eval, nu, q, k) -> list[Rational]`

### 29.1b: Uniform grid specialization

For the uniform grid case, provide a convenience wrapper:
- `uniform_interior_weights(p, nu, k, q) -> list` — interior stencil from 2p+1 points
- `uniform_boundary_weights(i, t, nu, k, q) -> list` — boundary row i from t points

Verify: for sufficiently high `k`, the interior weights should match the classical
`derive_interior` results (since polynomial interpolation is the high-k limit of PHS).

### 29.1c: Cut-cell grid specialization

For a cut-cell grid with wall at position `-psi` relative to grid point 0:
- `cut_cell_weights(i, T, nu, k, q, psi) -> list[Expr]` — symbolic in psi

This builds the PHS system on the non-uniform point set
`{-psi, 0, 1, 2, ..., T-2}` (normalized by h).

---

## 29.2 — Comparison with Known Stencils

### 29.2a: E2_1 uniform comparison

Compare PHS boundary stencils against the known E2_1 uniform boundary.

- E2_1: p=1, q=1, nu=1, t=4, r=3 (3 boundary rows, 4 columns each)
- Each boundary row has 1 free alpha
- PHS with k=2 (cubic) and k=3 should produce specific alpha values
- Compare to: (a) the optimizer-derived values used in `src/stencils/E2_1.cpp`,
  (b) the symbolic stencil from `derive_uniform_boundary_for_temo(E2_1)`

**Test**: `TestPHSvsE2Uniform` — check that PHS weights satisfy Taylor accuracy,
compare alpha values, check if they match known good values.

### 29.2b: E4_1 uniform comparison

Same comparison for E4_1: p=2, q=3, nu=1, t=6, r=4.
- 4 boundary rows, each with up to 2 free params (after zeros)
- Compare PHS-derived alphas to the Mathematica-optimized values from Table A.4

### 29.2c: Interior stencil verification

Verify that PHS interior stencils match classical stencils:
- For `k >= q+1`: PHS should recover polynomial interpolation → classical FD
- For `k < q+1`: PHS gives different (more dissipative) interior stencils
- Test for E2 (p=1) and E4 (p=2)

---

## 29.3 — Stability Analysis (Non-SBP)

### 29.3a: Total positivity check

Implement a total positivity diagnostic:
- Build the full differentiation matrix D (boundary + interior) for size n
- Form the iteration matrix `M = I + dt * D` for a range of dt values
- Check TP via Neville elimination: factor M into bidiagonal matrices,
  check if all multipliers are non-negative
- Compare TP properties of PHS-derived vs optimizer-derived stencils

**Function**: `neville_tp_check(M) -> (is_tp, min_multiplier)`

### 29.3b: Eigenvalue analysis

For the full differentiation matrix with PHS boundary stencils:
- Compute eigenvalues for a range of grid sizes (n=20, 40, 80, 160)
- Check: all eigenvalues should have non-positive real part (stability)
- Compare spectral radius of PHS vs optimizer stencils
- Plot eigenvalue loci for different PHS orders k

### 29.3c: CFL number comparison

For the 1D advection equation u_t + u_x = 0 with RK4 time integration:
- Compute maximum stable CFL number for PHS stencils vs optimizer stencils
- Vary PHS order k to find the optimal k for CFL

---

## 29.4 — Cut-Cell PHS Stencils

### 29.4a: PHS cut-cell stencil as function of psi

Build cut-cell stencils using PHS on the non-uniform grid:
- Points: `{-psi, 0, 1, ..., T-2}` for each row
- Compare structure to the Mathematica-derived cut-cell stencils
- Check: are PHS cut-cell entries polynomial in psi? Rational? What denominators?

### 29.4b: Small-cell stability (psi → 0)

The critical test: as psi → 0, do PHS cut-cell stencils remain well-behaved?
- Evaluate stencil coefficients at psi = 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0
- Check eigenvalues of the full operator at each psi
- Compare to the Mathematica workflow stencils at the same psi values

### 29.4c: Conservation check

Do PHS cut-cell stencils satisfy conservation (SBP column-sum property)?
- If yes: PHS provides conservation for free
- If no: how large is the conservation violation? Can it be fixed by
  adjusting the PHS order k or adding a conservation constraint to the
  PHS system?

---

## Implementation Order (Ralph Loop Sequence)

Each step produces tests that validate before moving on:

1. **29.1a** — Core PHS solver (pure math, no stencil knowledge needed)
2. **29.1b** — Uniform grid wrappers + verify interior matches classical FD
3. **29.2c** — Interior verification (confirms PHS engine is correct)
4. **29.2a** — E2_1 boundary comparison (first real comparison)
5. **29.2b** — E4_1 boundary comparison
6. **29.3b** — Eigenvalue analysis (does PHS give stable stencils?)
7. **29.3a** — Total positivity diagnostic
8. **29.4a** — Cut-cell PHS stencils
9. **29.4b** — Small-cell stability
10. **29.4c** — Conservation check

---

## Key Files

| File | Role |
|------|------|
| `scripts/stencil_gen/stencil_gen/phs.py` | **New** — PHS stencil engine |
| `scripts/stencil_gen/tests/test_phs.py` | **New** — PHS tests |
| `scripts/stencil_gen/stencil_gen/interior.py` | Reference for classical interior stencils |
| `scripts/stencil_gen/stencil_gen/boundary.py` | Reference for classical boundary stencils |
| `scripts/stencil_gen/stencil_gen/temo.py` | Reference for cut-cell construction |
| `plans/29-phs-stencil-investigation.md` | This plan (copy to plans/) |

## Verification

```bash
cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_phs.py -x -v
```

Each step adds tests that must pass before proceeding. The eigenvalue and CFL
analyses (29.3) produce numerical results that we inspect manually rather than
assert on — they're exploratory.

## Expected Outcomes

- **Best case**: PHS with some `k` produces stencils that are stable, conservative,
  and match or beat optimizer stencils. The optimization loop can be replaced.
- **Likely case**: PHS provides a good starting point (specific alpha values) that
  is close to optimal, dramatically reducing the optimization search space.
- **Worst case**: PHS stencils are not stable without additional constraints, but
  the investigation reveals which spline properties correlate with stability,
  guiding future optimization strategies.
