# Stencil Generation: Current Procedure vs Paper's Procedure

> **Purpose:** Document exactly what the SymPy pipeline does to generate cut-cell
> stencils, where it diverges from the paper's construction, and why this produces
> singularities that the paper avoids.

---

## The Paper's Procedure (Brady & Livescu 2021, Section 3)

The paper constructs the cut-cell boundary stencil B_l(ψ) in this order:

### Step 1: Uniform Boundary Stencil B^u

Build the r × t uniform boundary stencil using the 2019 paper's procedure:
- Each row satisfies Taylor accuracy (order q)
- Conservation (SBP) constraints determine quadrature weights and constrain
  some free parameters
- The remaining free parameters α^u are constant numbers chosen by numerical
  optimization
- For E4_1: r=4, t=6, 4 free α^u (Table 1)

### Step 2: Design Principles → Degenerate Stencil B^d

Determine the degenerate (ψ=0) stencil B^d from B^u using:
- **Principle 1 (TEM):** α^d_{i,wall} + α^d_{i,0} = α^u_{i,0} and
  α^d_{i,j} = α^u_{i,j} for j ≥ 1
- **Principle 2 (Continuity):** Choose variant B^{d,0} (for 1st derivatives):
  wall gets α^u_{i,0}, x_0 gets zero

B^d has size (r+1) × (t+1). It is fully determined by B^u and the design principles.

### Step 3: Construct B_l(ψ) — The Critical Step

The paper (Section 3.2) says:

> "The free parameter choices are therefore:
> α_{iδ} = ψ·α^u_{iδ} for i < r
> α_{r,r-p} = ψ·γ_{-p}
> where δ indicates the appropriate coefficient to send to zero"

And for extra free parameters:

> "α_{ij} = ψ·α^u_{ij} + (1-ψ)·α^u_{i-1,j-1}"

**Critical:** The paper then says:

> "Note that the αu terms are SPECIFIED rather than solved for. This avoids
> any singularities in the coefficients in the range ψ ∈ [0,1] and satisfies
> the design principles by construction."

This means: the α^u values are treated as KNOWN CONSTANTS (not unknowns).
The ψ-dependent entries are written directly in terms of these constants.
They are NOT solved from a Taylor system with ψ as a symbolic variable.

### Step 4: Conservation Constraints

Conservation is enforced (Section 3.3). The system has t+1 equations in r+1
unknowns (quadrature weights). Some of the α^u free parameters are consumed
by conservation. For E4_1 with zeros α^u_{05}=0, α^u_{15}=0: 4 free α^u
remain after conservation.

### Step 5: Optimization

The remaining free α^u values are chosen by numerical optimization on the
1D Euler equations (Section 3.4). This produces specific numerical values
for each α^u.

### What the Paper's Appendix A Shows

The actual E4_1 stencil (Table A.4) has:
- **Boundary rows (i=0..3):** Each entry is a polynomial in ψ of degree ≤ 4
  with coefficients that are NUMBERS (specific α^u values substituted)
- **Near-interior row (i=4):** Entries are rational: polynomial/e0(ψ), where
  e0 is a degree-8 polynomial that is NONVANISHING on [0,1]
- **No singularities** anywhere on [0,1]

---

## What Our Pipeline Does (and Where It Diverges)

### Our Step 1: Uniform Boundary — CORRECT

`derive_uniform_boundary_for_temo(E4_1, zeros={3,4})` builds the 4×6 uniform
boundary using `solve_boundary_row` from `boundary.py`. This matches the paper's
Step 1 correctly. The result has 3 free α symbols (α_0, α_1, α_2) after applying
the Table 1 zeros.

### Our Step 2: Degenerate Stencil — CORRECT

`build_degenerate_stencil` applies Design Principles 1 and 2 to get B^d. This
matches the paper's Step 2.

### Our Step 3: B_l(ψ) Construction — DIVERGES HERE

`construct_cut_cell_stencil` does this for each row:

1. Computes the Vandermonde Taylor system with ψ-dependent first column
2. Prescribes Category A entries (zeroed column = ψ · target)
3. For nextra > 0: prescribes extra columns via limit interpolation
4. **SOLVES the remaining entries from the Taylor system** using `solve_temo_row`
   or `solve_temo_row_polynomial`

**The problem is in step 4.** The Taylor system has a Vandermonde matrix with
entries like `(-(ψ+i))^k / k!` in the first column. When you solve this system:
- In QQ(ψ) (fraction field): you get rational functions with denominators like
  `(ψ+1)(ψ+2)(ψ+3)` from the Vandermonde determinant. These are benign (nonvanishing
  on [0,1]).
- With the polynomial ansatz (`solve_temo_row_polynomial`): you clear the Vandermonde
  denominator and get polynomial entries. This matches boundary rows correctly.

**But the near-interior row** (row r) is where it breaks. This row has more unknowns
than Taylor equations (underdetermined), so conservation is used to fill the gap.
The conservation equations couple this row to all other rows, creating a bilinear
system (products of weight unknowns × alpha parameters). Solving this bilinear
system introduces `ψ·(ψ-1)` denominators.

### Why Our Approach Produces Singularities

The singularities come from solving conservation for BOTH weights AND alphas
simultaneously. Specifically:

1. The conservation equations `Σ_i w_i · B[i,j] = 0` contain products `w_i · α_k`
   (bilinear terms) because boundary row entries B[i,j] depend on α_k.

2. Solving for α_0, α_1 in terms of ψ makes them ψ-DEPENDENT rational functions
   with `ψ·(ψ-1)` denominators. But the paper says α^u values are CONSTANTS
   (independent of ψ).

3. When these ψ-dependent α values are substituted back into the stencil entries,
   the polynomial boundary rows reacquire rational denominators.

### What the Paper Does Differently

The paper treats the α^u values as **known constants** throughout the construction.
It does NOT solve conservation symbolically for α — instead, conservation is
enforced through the sizing formula (Eq. 11) which ensures enough rows/columns
exist, and the specific α^u values are chosen NUMERICALLY by the optimizer to
satisfy both stability and conservation simultaneously.

The key difference:
- **Paper:** α^u are constants → stencil entries are polynomials in ψ (or rational
  with benign denominator for the near-interior row) → conservation verified
  numerically for each candidate α set during optimization
- **Our pipeline:** α^u are symbolic unknowns → conservation system is bilinear →
  solving for α produces ψ-dependent α → singularities

### The Near-Interior Row

The paper's near-interior row (row 4 of E4_1) has entries that are:
`polynomial(ψ, α^u) / e0(ψ, α^u)` where e0 is degree 8 in ψ.

When specific α^u values are substituted (from the optimizer), e0 evaluates to
a polynomial in ψ alone that is nonvanishing on [0,1]. But symbolically, e0
depends on both ψ and α, and for SOME α values e0 could vanish on [0,1].
The optimizer must avoid such α values.

Our pipeline tries to produce symbolic expressions where conservation holds for
ALL α — but this is impossible (proven via Gröbner basis analysis). The paper
only requires conservation for the SPECIFIC α values found by the optimizer.

---

## Summary of the Divergence

| Aspect | Paper | Our Pipeline |
|--------|-------|-------------|
| α^u values | Constants (from optimizer) | Symbolic unknowns |
| Conservation | Verified numerically per α | Enforced symbolically for all α |
| Boundary row entries | Polynomials in ψ | Polynomials in ψ (after Phase 27 fix) |
| Near-interior row | Rational with benign denom | Rational with ψ·(ψ-1) denom |
| Singularities | None | At ψ=0 and ψ=1 |

## The Path Forward

The pipeline should:
1. Build boundary row entries as **polynomials in ψ** with symbolic α^u coefficients
   (Phase 27 achieved this via `solve_temo_row_polynomial`)
2. Build the near-interior row entries as **rational functions of ψ and α^u**, where
   the denominator depends on both ψ and α^u
3. **NOT solve conservation symbolically for α** — instead, provide the symbolic
   stencil with conservation as a CONSTRAINT that the optimizer must satisfy
4. The optimizer evaluates the symbolic stencil at candidate (α^u, ψ) values and
   checks: (a) stability on Euler test problems, (b) conservation column sums ≈ 0,
   (c) denominator e0(ψ, α) ≠ 0 on [0,1]

Alternatively: solve conservation for the NEAR-INTERIOR ROW ENTRIES ONLY (not for α),
treating boundary rows as known polynomial inputs. This would produce the near-interior
entries as rational functions of (ψ, α) with a denominator that is the determinant of
the conservation+Taylor system for that single row — analogous to the paper's e0.
The boundary rows remain polynomial, and the α^u remain free constants.

---

## Key References in Code

| Function | File | Line | What it does |
|----------|------|------|-------------|
| `derive_uniform_boundary_for_temo` | temo.py | ~315 | Builds uniform B_u with zeros |
| `construct_cut_cell_stencil` | temo.py | ~1353 | Main TEMO construction |
| `solve_temo_row` | temo.py | ~1263 | Per-row Taylor solve (RATIONAL — bad for near-interior) |
| `solve_temo_row_polynomial` | temo.py | ~1349 | Polynomial ansatz (GOOD for boundary rows) |
| `solve_conservation_fraction_free` | temo.py | ~1507 | Conservation solve (produces singularities) |
| `build_cut_cell_conservation_system` | temo.py | ~1430 | Builds conservation equations |
| `solve_uniform_limit` | temo.py | ~1000 | Computes B_l(1) for limit interpolation |
| `build_degenerate_stencil` | temo.py | ~536 | Computes B_l(0) via design principles |
