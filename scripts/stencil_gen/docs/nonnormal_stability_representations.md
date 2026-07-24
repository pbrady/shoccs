# Representations of non-normal cut-cell operators for stability analysis

**Status: TABLED (2026-07-23).** Research complete; no implementation planned for now.
The recommendations below lean heavily on SBP-style norm co-design, and there is an
open reservation about going down the SBP path (this project deliberately uses strong
BC enforcement, not SBP-SAT). Recorded here so the findings aren't lost when the
discussion resumes.

Provenance: two adversarially-verified deep-research runs (2026-07-22/23).
Pass 1: 24 sources fetched, 108 claims extracted, top 25 verified by 3-vote panels →
22 confirmed 3-0, 3 refuted. Pass 2 (focused follow-up): 22 sources, 109 claims,
top 25 verified → 25 confirmed (two at 2-1, one at 2-0); the synthesis stage of the
run failed on a usage limit, so the merge below was done by hand from the verified
claims. All quotes were checked verbatim against primary sources by independent
verifier agents.

---

## 1. Motivating question

SHOCCS semi-discrete spatial operators are strongly non-normal: 1D operators are
banded Toeplitz/circulant in the interior with dense one-sided boundary closures plus
sparse cut-cell correction rows; multi-D operators compose 1D operators
Kronecker-style and additionally incorporate interpolation/extrapolation operators at
embedded boundaries. Eigenvalue analysis is unreliable, and pseudospectra/GKS-style
reasoning is expensive and awkward inside an optimization loop over free stencil
parameters.

Two ideas were investigated:

1. Can the operator be usefully approximated/decomposed as a **sum of normal
   matrices** so that stability analysis becomes tractable?
2. Can **low-rank tensor / Kronecker representations** provide a friendlier space for
   stability analysis and optimization?

Follow-up question (pass 2): does the **contractivity certification** strategy used
for cut-cell DG (Taylor–Wilcox–Chan: prove the stabilization operator is
norm-contractive, so it cannot damage an energy-stable base scheme) apply to
point-value finite-difference discretizations, or is it DG-specific?

## 2. Verdict summary

- **Sum-of-normal splittings do not certify stability by themselves.** The exact
  Hermitian + skew-Hermitian split always exists, but the parts do not commute and
  nothing about `e^{tA}` follows from their spectra. The usable version of the idea
  is the **numerical range / logarithmic norm** route, and it only pays off in a
  co-designed inner product — at which point it collapses into the energy method.
- **The recommended representation** (from the literature; see reservation above):
  `A = H^{-1}(S + E)` — co-designed norm `H`, skew-symmetric part `S`, dissipative
  boundary part `E` — with the H-norm numerical abscissa
  `ω_H(A) = λ_max` of the H-symmetrized part as the cheap optimization functional,
  and cut-cell/interpolation corrections certified **compositionally** (dissipative
  base + norm-certified correction) instead of analyzing the assembled non-normal
  operator.
- **The low-rank/tensor angle came up empty** for stability analysis specifically
  (as opposed to fast solves): no surviving verified claims. Absence of evidence,
  not evidence of absence, but there is no established precedent to follow.
- **Normal-plus-low-rank perturbation framing is closed** for eigenvalue purposes:
  the localization theorem requires *both* endpoint matrices to be normal, which
  fails here (the boundary-closed base operator is itself non-normal).
- **Contractivity certification is discretization-agnostic, and historically
  FD-native** — the stable-filtering results were established for FD first and later
  translated to DG. What is DG-specific is only the easy proof that particular
  operators (modal filters, state redistribution) are contractive. For point-value
  FD, high-order interpolation is generically *not* contractive (verified: achieved
  only at orders 2 and 4; open at 6 and 8), so an FD correction operator must either
  be certified by adjoint-compatibility/telescoping in the energy estimate, or be
  *designed* as an H-orthogonal projection / convex weighted averaging.

## 3. Pass 1 — representations for non-normal operators

### 3.1 Premise confirmed (3-0)

Eigenvalue analysis is unreliable and can be actively misleading for strongly
non-normal operators: `e^{tα(A)} ≤ ‖e^{tA}‖ ≤ κ(V) e^{tα(A)}` and the eigenvector
condition number `κ(V)` can grow exponentially in physical/discretization parameters
(e.g. `κ(V) ≥ e^{d/2}` for convection–diffusion), especially when departure from
normality grows under mesh refinement.

- Trefethen & Embree, *Spectra and Pseudospectra* (Princeton, 2005)
- Benzi, Boll. UMI (2021), https://link.springer.com/article/10.1007/s40574-020-00249-2
- Trefethen, SIAM Review 39 (1997), https://people.maths.ox.ac.uk/trefethen/publication/PDF/1997_72.pdf

### 3.2 Field of values / numerical abscissa as the usable substitute (3-0)

- **Crouzeix–Palencia (SIMAX 38(2), 2017):** `W(A)` is a spectral set with universal
  constant `1+√2`: `‖f(A)‖ ≤ (1+√2) sup_{W(A)} |f|`. Consequence: if `W(A)` lies in
  the closed left half-plane, `‖e^{tA}‖ ≤ 1+√2` uniformly in `t` — dimension- and
  eigenvector-conditioning-independent. (Q = 2 conjecture still open as of
  Aug 2025.)
- **Logarithmic-norm identity:** `d/dt ‖e^{tA}‖ |_{t=0+} = ω(A) = λ_max((A+A*)/2)`.
  One well-conditioned Hermitian eigenproblem per candidate — ideal inside an
  optimization loop. In a weighted inner product this becomes `ω_H(A)`.
- Caveat: `ω_H(A) ≤ 0` is sufficient, not necessary — as a hard constraint it may
  over-constrain high-order closures. Principled relaxation: allow a small positive
  abscissa with the Crouzeix–Palencia transient bound.

### 3.3 Norm co-design precedents (3-0)

- **Sharan, Brady & Livescu** (SINUM 60(3):1331–1362, 2022, DOI 10.1137/21M1419957;
  AIAA 2020-0807; JCP 464:111341, 2022): sufficient conditions of the form
  `HA + A*H ≤ 0` co-derived with explicit boundary and cut-cell stencils; strong
  (injection) BCs; no small-cell problem; dimensionally split multi-D. Direct
  precedent for this project (self-citation-adjacent — the research run flagged
  this). Known gap: multi-D stability there rests on eigenvalue spectra plus
  numerical experiments, not a multi-D energy proof.
- **Glaubitz, Nordström & Öffner** (J. Sci. Comput. 2025, arXiv:2405.08770):
  the SBP norm matrix `P` and the operator are constructed *simultaneously* as an
  unconstrained least-squares problem (positivity via sigmoid parametrization,
  solved with off-the-shelf L-BFGS). Proof that "find the norm numerically inside
  the derivation loop" is practical.
- **Marchildon & Zingg** (JCP 2020, DOI 10.1016/j.jcp.2020.109380): free parameters
  — cubature (defines diagonal `H`), the interpolation/extrapolation operator `R`,
  and the skew part `S_ξ` — optimized *inside* the `D = H^{-1}(S+E)` parametrization
  so energy stability is preserved throughout the optimization. Their `R` is the
  structural analogue of embedded-boundary interpolation operators.

### 3.4 Compositional certification template (3-0)

**Taylor, Wilcox & Chan** (JCP 521, 2025; arXiv:2404.06630): semi-discrete energy
stability on arbitrary cut cells by (a) skew-symmetric formulation, energy stable
under arbitrary quadrature, and (b) state redistribution (SRD) proven L²-contractive
(`‖Su‖ ≤ ‖u‖`, Theorem 2.1), so composing SRD with the stable base cannot increase
energy. Template: prove the base dissipative in a designed norm; certify each
correction by a norm bound on the correction operator itself. Scope caveat: 2D
acoustic wave (linear, symmetric hyperbolic), semi-discrete.

### 3.5 Verified negative results

- **Pseudospectral certification does not scale:** the easy half of the Kreiss
  matrix theorem is a sharp necessary condition
  (`sup_t ‖e^{tA}‖ ≥ α_ε(A)/ε` — cheap falsifier, worth keeping), but the upper
  bound from pseudospectra is proportional to matrix dimension, and
  Spijker–Tracogna–Welfert (Math. Comp. 72, 2003) proved linearity is unimprovable.
  The pain with pseudospectra under refinement is a theorem, not a tooling problem.
- **Normal + low-rank localization is vacuous here:** Glebsky–Rivera (ELA 18, 2009;
  arXiv:0802.3874) requires both endpoints normal.
- **Refuted in verification (do not cite without re-checking):**
  "SBP structure alone guarantees stability" (needs SATs or verified strong
  injection to close the estimate; voted down 1-2) and "rank-1 perturbations can
  relocate the spectrum of any matrix arbitrarily" (voted down 1-2).
- **Tensor-train / Kronecker / displacement-rank formats for stability analysis**
  (pseudospectra, numerical range, Lyapunov in compressed format): no surviving
  verified claims. Open question, not a dead end.

## 4. Pass 2 — is contractivity certification DG-only?

### 4.1 The composition theorem is discretization-agnostic (3-0)

- **Nordström, "Well-posed and stable transmission problems"** (JCP 2018):
  passing a solution through an operator `X` between two energy-bounded dynamics
  admits an energy estimate **iff** `P₁ − XᵀP₂X ≥ 0` (contractivity from the
  `P₁`-norm to the `P₂`-norm). Stated for general weighted-L² inner products;
  verbatim: "No assumptions about the spatial discretisations was made, apart from
  them being semi-bounded and stable." A fully discrete version (SBP-in-time,
  Prop. 4) has the *same* condition. Because the condition is necessary and
  sufficient, a non-contractive `X` admits no energy bound in those norms — fix `X`
  or change the norm.
- **Nordström & Winters** (arXiv:2007.06948), the result Taylor–Wilcox–Chan invoke:
  hypothesis on the base scheme is only a semi-discrete energy bound. The paper
  explicitly "re-contextualize[s] stable filtering results **for finite difference
  methods** into the DG setting" — the transfer direction was FD → DG.
- **Where DG structure actually enters:** proving specific operators contractive.
  The modal-filter proof uses the Legendre basis + LGL quadrature exactness;
  the SRD proof uses SRD's structure as an average of volume-weighted L²
  projections (projections contractive; convex averages contractive by Jensen).

### 4.2 How the FD/SBP world certifies coupling operators (3-0 unless noted)

1. **Adjoint-compatibility (dominant).** Mattsson & Carpenter (SISC 32(4), 2010,
   DOI 10.1137/090750068): `H_R I_c2f = I_f2cᵀ H_L` ("SBP-preserving") makes the
   interface energy contribution telescope to exactly zero. Kozdon & Wilcox
   (SISC 2016, DOI 10.1137/15M1022823): same adjoint structure via a glue grid
   (`Pᵀ_g2f H = M P_f2g`); stability fully decoupled from accuracy. Nordström's
   transmission paper shows adjoint-compatibility plus the accompanying conditions
   is *equivalent* to the contractivity condition — the FD literature's conditions
   are contractivity in disguise.
2. **Dissipative penalties.** Choose SAT coefficients so the interface quadratic
   form is negative semidefinite (La Cognata & Nordström, BIT 2016 — 2-1 vote on
   the mechanism characterization; Mattsson–Carpenter Lemma 3.4 achieves stability
   with only adjoint-compatibility + an SPD penalty, bypassing contractivity).
3. **Genuine contractivity — the obstruction.** Mattsson–Carpenter's
   contractivity-type conditions (`H(I − I_f2c I_c2f) ≥ 0`, their eq. 12) are "a
   significant restriction," achieved **only for 2nd- and 4th-order operators;
   6th/8th order stated as an open question**. Consistent: the
   Sharan–Pantano–Bodony overset method (JCP 2018, DOI 10.1016/j.jcp.2018.01.049)
   never certifies interpolation in a norm — a block-triangular eigenvalue argument
   covers any interpolation, and when that structure fails, plain Lagrange
   interpolation needed added local artificial dissipation at donor points (2-0
   vote; one verifier ran without its safety classifier). High-order point-value
   interpolation is generically norm-amplifying.

### 4.3 Implications for SHOCCS

- **Interpolation inside the RHS** (part of the assembled operator): `‖R‖_H ≤ 1` is
  not the right ask — what is needed is that the interpolation's contribution to
  the energy estimate cancels (adjoint-compatibility/telescoping) or is
  dissipative. Adjoint-compatibility is a *linear* constraint coupling `R` and `H`
  — SymPy-pipeline-friendly, achievable at high order.
- **Correction applied as per-step postprocessing** (SRD-style): `H − SᵀHS ≥ 0` is
  the right certificate, and the verified obstruction dictates design: build the
  correction as an H-orthogonal projection / convex weighted averaging (contractive
  by construction), not polynomial extrapolation.
- Since `H` is co-designed anyway, "make the correction contractive in *some*
  admissible `H`" becomes a joint constraint in the same optimization.

### 4.4 Gaps left by pass 2

- Explicit-RK composition (products `(S·R(ΔtA))^n`): verified only for SBP-in-time
  integrators. SSP/circle-condition theory is the relevant machinery for explicit
  RK; sources were fetched but their claims were budget-capped out of verification.
- Berger–Giuliani FV SRD stability results: fetched, not verified.

## 5. Open questions (carried from both passes)

1. Can LMI/SDP or Lyapunov-based search for `H` scale to full 2D/3D cut-cell
   operators, and can `H(ψ)` be parametrized smoothly in the cut parameter so the
   pipeline derives a certified family rather than one solve per geometry?
2. Can FoV/energy certificates be evaluated in compressed (Kronecker/TT) format
   instead of on the assembled operator? (No literature found either way.)
3. Can an SRD-style contractivity certificate (`‖S‖_H ≤ 1`) be constructed for
   SHOCCS's FD interpolation/extrapolation corrections, yielding a compositional
   proof analogous to Taylor–Wilcox–Chan?
4. Is there a principled relaxation of `ω_H(A) ≤ 0` (bounded abscissa +
   Crouzeix-controlled transient; ε-pseudospectral abscissa with
   dimension-independent consequences) that avoids over-constraining high-order
   closures?
5. Does the compositional certificate survive explicit RK time stepping (§4.4)?

## 6. Why this is tabled

The literature-backed path runs through SBP-style machinery: co-designed norm
matrices, SBP-preserving adjoint-compatibility for interpolation, SAT-style
dissipative penalties as the fallback. There is a standing reservation about
adopting the SBP path in this project (strong BC enforcement is a deliberate design
choice here, and the SINUM/JCP strong-BC line shows norms can be co-designed without
SAT weak enforcement — but the broader SBP framing is still not an obviously wanted
dependency). Decision deferred; revisit against open questions 1–5 when
multi-dimensional stability work resumes.
