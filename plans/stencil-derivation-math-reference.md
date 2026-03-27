# Stencil Derivation: Mathematical Reference

> **Read first.** This document summarizes the key equations from Brady & Livescu
> (2019, "High-order multiblock embedded boundary finite difference methods for
> the wave equation and Euler equations") and (2021, "Foundations for
> high-order, conservative cut-cell methods: stable discretizations on
> degenerate meshes") that the stencil derivation pipeline implements.
> Implementation details (SymPy, code generation) belong in the plan files.

---

## 1. Interior Stencil Formula

The general implicit (compact) finite difference approximation to the d-th
derivative on a uniform grid with spacing h (Eq. 6, 2019):

    sum_{k=-s}^{s} delta_k f^(d)_{i+k} = (1/h^d) sum_{j=-p}^{p} gamma_j f_{i+j}

where s controls the left-hand side (LHS) bandwidth and p controls the
right-hand side (RHS) bandwidth. Conventions:

- delta_0 = 1 (normalization)
- delta_{-k} = delta_k (LHS symmetry for all derivatives)
- 1st derivative: gamma_{-j} = -gamma_j, gamma_0 = 0 (antisymmetry)
- 2nd derivative: gamma_{-j} = gamma_j (symmetry)

The unknowns are determined by Taylor-expanding f_{i+j} and f^(d)_{i+k} about
point i and matching coefficients through the desired order. The interior
scheme has formal order 2p (explicit) or 2(p+s) (compact).

Scheme families in this codebase:

| Family | s | LHS | Example |
|--------|---|-----|---------|
| E (explicit) | 0 | identity (delta_0=1 only) | E2, E4, E6, E8 |
| T (tridiagonal compact) | 1 | delta_{-1} f' + f' + delta_1 f' | T4, T6, T8 |

---

## 2. Uniform Mesh Boundary Stencils

Near a domain boundary (left side, rows i = 0, ..., r-1), the interior stencil
cannot be centered. A one-sided boundary closure replaces it (Eq. 25, 2019):

    sum_{k=0}^{2s} delta_{i,k} f^(d)_{k} = (1/h^d) sum_{j=0}^{t-1} gamma_{i,j} f_j

where:

- t = r + p (total boundary stencil width)
- The LHS has 2s+1 terms (for compact schemes) or just f^(d)_i (for explicit)
- Order constraint: q = 2(p + s) - 1 (one less than interior)

Taylor-expanding about x_0 yields q+1 linear equations in t unknowns. Since
t > q+1, the system is underdetermined and the solution has free parameters
alpha_{i,0}, alpha_{i,1}, ... that parameterize the null space.

Right boundary: obtained from the left boundary stencil by reversal. For the
1st derivative, negate all coefficients and reverse their order. For the 2nd
derivative, reverse order only (no sign change).

---

## 3. Conservation (SBP) Constraints

The boundary stencil must satisfy the summation-by-parts (SBP) property,
which requires the existence of a positive-definite diagonal norm matrix H
such that the discrete operator Q = HD satisfies Q + Q^T = B (boundary term).

Concretely, the conservation constraint (Eq. 15, 2019) is:

    W^T B_i = 0     for interior columns i = r, ..., t-1

where W is the r x t matrix of boundary stencil coefficients (one row per
boundary line i = 0, ..., r-1) and B_i is the i-th column. The quadrature
(norm) weights are w_0, ..., w_{r-1} for boundary rows and w_i = 1 for all
interior rows.

This yields a system of (r + p - 1) additional linear constraints on the
gamma_{i,j} and w_i, further reducing the number of free alpha parameters.

---

## 4. TEMO Cut-Cell Extension (2021 Paper)

When an embedded boundary intersects a grid cell at fractional position
psi in [0,1] (psi=1 is a full cell, psi->0 is fully degenerate), the
uniform boundary stencil from Section 2 must be replaced by a cut-cell
variant. The TEMO (Truncation Error Matching Optimization) procedure of
Brady & Livescu 2021 constructs this variant in a way that avoids
singularities over the full range psi in (0,1].

### 4.1 Sizing (Eq. 11)

The degenerate (cut-cell) boundary stencil B^d_l has r+1 rows and t+1
columns -- one more row and one more column than the corresponding
uniform stencil B^u_l. The extra column corresponds to the wall point
f_0; the extra row provides an additional closure line.

The dimensions are determined by (Eq. 11):

    t = p + q + 1 + nextra        (Eq. 11a)
    r = q + 1 + nextra            (Eq. 11b)

where p is the interior half-bandwidth, q is the boundary accuracy order,
and nextra adds extra rows/columns to provide free parameters available
for numerical optimization.

Concrete schemes from Table 1 of the 2021 paper:

| Scheme | p | q | nextra | r+1 rows | t+1 cols | Free parameters | Zeros |
|--------|---|---|--------|----------|----------|-----------------|-------|
| E2_1   | 1 | 1 | 1      | 4        | 5        | alpha^u_{02}, alpha^u_{03}, alpha^u_{12}, alpha^u_{13} | -- |
| E4_1   | 2 | 3 | 0      | 4        | 7        | alpha^u_{04}, alpha^u_{14}, alpha^u_{24}, alpha^u_{25} | alpha^u_{05}=0, alpha^u_{15}=0 |

### 4.2 Design Principles (Section 3.1)

The construction proceeds step-by-step:

**Step 1.** Start with the uniform boundary stencil B^u_l of size r x t
from Module 2 (Section 2 above / the 2019 paper). This stencil operates
on [f_0, f_1, ..., f_{t-1}]^T with uniform spacing h.

**Step 2.** The degenerate stencil B^d_l has size (r+1) x (t+1). It
operates on the extended vector [f_0, f_delta, f_1, f_2, ...]^T, where
f_delta is the wall point at distance psi*h from f_0 (column index
delta = 1 in the degenerate stencil). As psi -> 0, f_delta coincides
with f_0.

**Step 3 -- Design Principle 1 (TEM, Eq. 6).** The truncation error of
the degenerate stencil must match that of the uniform stencil. This
requires:

    alpha^d_{i,delta} + alpha^d_{i,0} = alpha^u_{i,0}     (Eq. 6a)
    alpha^d_{i,j} = alpha^u_{i,j}     for j in [1, t)     (Eq. 6b)

That is, the uniform coefficient alpha^u_{i,0} is split between the
original column 0 and the new wall column delta, while all other columns
are inherited unchanged.

**Step 4 -- Design Principle 2 (Continuity, Eq. 7).** Resolve the
ambiguous split from Step 3 by imposing continuity as psi -> 1. Three
variants are defined by how the split is resolved for the leading rows:

    B^{d,0}:  alpha^d_{0,delta} = alpha^u_{0,0},  alpha^d_{0,0} = 0
              alpha^d_{i,delta} = 0,               alpha^d_{i,0} = alpha^u_{i,0}  for i >= 1

    B^{d,1}:  alpha^d_{0,delta} = 0,               alpha^d_{0,0} = alpha^u_{0,0}
              alpha^d_{i,delta} = alpha^u_{i,0},   alpha^d_{i,0} = 0              for i >= 1

    B^{d,2}:  alpha^d_{0,delta} = alpha^u_{0,0},  alpha^d_{0,0} = 0
              alpha^d_{i,delta} = alpha^u_{i,0},   alpha^d_{i,0} = 0              for i >= 1

**Step 5 -- Variant selection.**

- B^{d,0} is used for 1st derivative stencils.
- B^{d,1} is used for 2nd derivative with Neumann BCs.
- B^{d,2} is used for 2nd derivative with Dirichlet BCs.

**Step 6.** Important: all schemes in this paper use alpha^d_{i,0} = 0
for all i >= 1. That is, only the wall point (column delta) carries the
split coefficient for rows i >= 1; the original column 0 is zeroed.

### 4.3 B_l(psi) Construction (Section 3.2)

The general cut-cell stencil valid for psi in [0,1] is built from the
degenerate template above by making coefficients psi-dependent. The key
rules are:

**Rule 1 -- The delta column (zeroed by Principle 2).**
For the column being set to zero by Step 4:

    alpha_{i,delta}(psi) = psi * alpha^u_{i,delta}     for i < r

This linearly fades the wall-point coupling to zero as psi -> 0.

**Rule 2 -- Last row.**
The extra (r+1)-th row ties back to the interior scheme:

    alpha_{r, r-p}(psi) = psi * gamma_{-p}

where gamma_{-p} is the interior scheme coefficient, scaled by psi.

**Rule 3 -- Extra free parameters.**
Free parameters introduced by nextra > 0 interpolate between adjacent
uniform rows:

    alpha_{i,j}(psi) = psi * alpha^u_{i,j} + (1 - psi) * alpha^u_{i-1, j-1}

**Critical property:** In Rules 1--3, the alpha^u terms are SPECIFIED
(taken from the known uniform stencil), not solved for. This is the key
insight that prevents the coefficient system from becoming singular as
psi varies over [0,1].

**Rule 4 -- Remaining (non-free) coefficients.**
All coefficients not fixed by Rules 1--3 are determined by the Taylor
accuracy system. The Taylor expansion uses non-uniform grid spacings
arising from the cut cell:

    x_{j+1} - x_j = psi * h     for j = 0  (the cut interval)
    x_{j+1} - x_j = h           for j > 0  (regular intervals)

The spacing differences entering the Taylor system are:

    Delta_{i,j} = x_j - x_i

with the above non-uniform spacings. The resulting coefficients are
rational functions of psi (no singularities for psi in (0,1]).

### 4.4 Cut-Cell Conservation (Section 3.3)

The SBP conservation constraint from Section 3 carries over with
modifications:

**Interior columns (same as 2019 paper):**

    w^T O_i = 0     for interior columns i

where O_i is column i of the full operator and w is the vector of norm
(quadrature) weights. For the cut cell, w_0 = psi.

**Extra constraint for column 0 of the degenerate stencil:**
Because coefficients from column 0 of B^u_l have been moved to column 1
(the delta column) of B^d_l, an additional conservation equation is
required (equation following Eq. 10 in the paper):

    w_0 * alpha_{0,0} = w^T_u * B^u_{l, column 0}

This ensures that the norm-weighted column sum is preserved despite the
coefficient redistribution.

**System dimensions:** The full conservation system has t+1 equations in
r+1 unknowns (the norm weights w_0, ..., w_r plus any remaining free
alpha parameters). Some of the free alpha parameters from Rules 1--3 are
consumed by these conservation constraints; the parameters that remain
unconstrained are available for spectral-radius optimization (Section 5).

### 4.5 Neumann Stencils (Eq. 8)

For the 2nd derivative with Neumann boundary conditions, the stencil
includes coupling to the prescribed derivative value f'_0 at the wall.
The approximation takes the form (Eq. 8):

    f^(nu)_i = (1 / h^{nu-1}) eta_i f'_0 + (1 / h^nu) sum_j alpha_{i,j} f_j

where eta_i are the Neumann coupling coefficients and nu = 2 is the
derivative order.

**Design principles constrain eta:** Analogous to the treatment of
alpha, the degenerate Neumann coefficients satisfy:

    eta^d_0 = eta^d_delta = eta^u_0
    eta^d_i = eta^u_i     for i >= 1

These eta coefficients produce the X (nextra) extra coefficients stored
in the C++ stencil structs:

| Scheme | nextra (X) | Meaning |
|--------|------------|---------|
| E2_2   | 2          | 2 extra eta coupling coefficients |
| E4_2   | 3          | 3 extra eta coupling coefficients |

---

## 5. Optimization (Non-Symbolic -- Out of Scope)

The free alpha parameters remaining after Taylor-matching and conservation
constraints are optimized numerically. Brady & Livescu minimize the spectral
radius on 1D Euler test problems.

This optimization is NOT part of the symbolic derivation pipeline. The
symbolic pipeline produces stencil coefficients as closed-form rational
functions of alpha (and psi for cut-cells). The alpha values are chosen
separately and passed in at construction time (see `std::span<const real>`
constructor arguments in the C++ stencil structs).

---

## 6. Scheme Inventory

### First derivative (suffix `_1`)

| Scheme | s | p | r | t | Interior order | Boundary order | Free alpha |
|--------|---|---|---|---|---------------|----------------|------------|
| E2_1 (cut-cell) | 0 | 1 | 4 | 5 | 2 | 1 | 4 |
| E4u_1 (uniform) | 0 | 2 | 3 | 5 | 4 | 3 | 2 |
| E6u_1 (uniform) | 0 | 3 | 5 | 8 | 6 | 5 | 5 |
| E8u_1 (uniform) | 0 | 4 | 7 | 11 | 8 | 7 | 7 |
| polyE2_1 (cut-cell) | 0 | 1 | 3 | 4 | 2 | 1 | 6+3+4 (f/d/i) |
| T4_1 | 1 | 1 | 3 | 4 | 4 | 3 | 2 |
| T6_1 | 1 | 2 | 4 | 6 | 6 | 5 | 3 |
| T8_1 | 1 | 3 | 6 | 9 | 8 | 7 | 6 |

### Second derivative (suffix `_2`)

| Scheme | s | p | r | t | nextra | Interior order |
|--------|---|---|---|---|--------|----------------|
| E2_2 | 0 | 1 | 2 | 4 | 2 | 2 |
| E4_2 | 0 | 2 | 3 | 5 | 3 | 4 |

Note: `nextra` (the X field in `info`) counts additional rows for Neumann
boundary conditions in second-derivative stencils.

---

## 7. Output Format

### Uniform stencils (e.g., `E4u_1.cpp`)

- **Interior coefficients:** Fixed rational numbers divided by h (or h^2).
  Stored in the `interior(h, c)` method. Size = 2p+1.
- **Boundary coefficients:** Rational functions of alpha, divided by h.
  Stored in `nbs_floating(h, psi, c, right)` and `nbs_dirichlet(...)`.
  Layout: r*t (floating) or (r-1)*t (Dirichlet) values in row-major order.
- **Right boundary:** Computed from left by negating + reversing (1st deriv)
  or reversing only (2nd deriv).

### Cut-cell stencils (e.g., `E2_1.cpp`, `polyE2_1.cpp`)

- Same structure, but boundary coefficients are rational functions of both
  psi and alpha. The nbs methods take a `real psi` parameter.
- Intermediate subexpressions (powers of psi, reciprocals of polynomials
  in psi) are hoisted into named temporaries (`t3`, `t5`, ...).

### C++ interface contract

All stencil structs satisfy the `Stencil` concept from `stencil.hpp`:

```
info query(bcs::type) const;          // {p, r, t, nextra} for given BC
info query_max() const;               // maximum {p, r, t, nextra}
interp_info query_interp() const;     // {p, t} for interpolation
span<const real> interior(h, c);      // write 2p+1 coefficients into c
span<const real> nbs(h, bc, psi, right, c, extra);  // write boundary block
```

The type-erased wrapper `ccs::stencils::stencil` holds any `Stencil` via
`any_stencil_impl<S>`. Factory functions (`make_E4u_1`, etc.) construct
concrete stencils from alpha spans.
