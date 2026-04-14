---
name: group-velocity-analysis
description: Guide to group velocity analysis for finite difference stencils. Use when discussing wave propagation, dispersion, modified wavenumber, parasitic modes, GKS stability, cut-cell group velocity, or 2D anisotropy.
---

# Group Velocity Analysis

Group velocity `C(ξ) = dω/dξ` governs how wave packets propagate through finite difference discretizations. Unlike phase velocity, group velocity determines energy transport and controls boundary stability (GKS theory).

## Quick Reference

| Function | Purpose |
|----------|---------|
| `interior_group_velocity(p, nu, xi)` | GV profile for E2-E8 interior stencils |
| `boundary_group_velocity(p, q, nextra, nu, sigma, kernel, xi)` | GV for RBF/tension boundary rows |
| `boundary_group_velocity_classical(boundary_rows, alpha_values, order, xi)` | GV for symbolic boundary stencils |
| `cut_cell_group_velocity(result, psi_sym, psi_val, alpha, xi)` | GV for TEMO cut-cell stencils |
| `psi_sweep_group_velocity(scheme, psi_values, alpha, xi)` | Sweep GV across ψ ∈ [0,1] |
| `group_velocity_2d(kappa_x_star, kappa_y_star, xi, eta, a, b)` | 2D GV vector from pre-computed modified wavenumber arrays |
| `anisotropy_profile(p, nu, theta, xi_mag)` | Grid anisotropy vs propagation angle |
| `gks_group_velocity_check(D, xi, side=)` | GKS eigenmode boundary stability diagnostic (heuristic) |
| `kreiss_stability_check(interior, offsets, brows)` | Rigorous GKS Kreiss determinant test (Trefethen 1983) |
| `local_group_velocity(weights_func, x, xi)` | Local GV for varying coefficients (1D) |
| `local_group_velocity_2d_varying(stencil_x, stencil_y, c_x, c_y, xi)` | Per-point local GV error on a 2D varying-coefficient grid |
| `anisotropy_over_coefficient_field(scheme, c_x, c_y, grid, theta, xi_mag)` | 2D anisotropy projected onto a varying flow field |
| `ray_trace_group_velocity(C_field, x, xi, ...)` | Ray tracing dx/dt = C, dξ/dt = -∂C/∂x |
| `brady2d_stability_score(scheme, kernel, params, max_layer=)` | Layered stability scoring for Brady-Livescu 2D benchmark |

## Key Facts

- **Sign convention:** `C(ξ) = d(Im(κ*))/dξ` where `κ*(ξ)` is the modified wavenumber
- **Error amplification:** Group velocity error is `(2p+1)×` the phase velocity error for a 2p-order scheme
- **Cutoff wavenumber:** Where `C(ξ) = 0` — above this, energy propagates backwards (parasitic)
- **GKS connection:** Boundary instability ⟺ outgoing modes (`C > 0`) spontaneously generated at boundary
- **Scaling advantage:** GV analysis is O(1) per stencil; eigenvalue analysis is O(N³) per grid

## When to Use

- Analyzing dispersion behavior of new stencil schemes
- Diagnosing boundary instabilities via GKS group velocity check (heuristic) or rigorous Kreiss determinant test
- Evaluating cut-cell stencils across the ψ range
- Checking 2D grid anisotropy for tensor-product operators
- Comparing GV analysis cost vs eigenvalue analysis for large problems
- Scoring boundary closures against the Brady-Livescu 2D varying-coefficient benchmark (L1-L7 layered pipeline)
- Layer 8 closed-loop validation: re-running top survivors through the compiled C++ solver (`run_cpp_brady2d` / `sweeps brady2d --validate-with-cpp`) to confirm analytical L1-L7 verdicts empirically
- Non-normality diagnostics: spectral/numerical abscissa, Kreiss constant, transient growth bound

## Detailed Reference

For complete API documentation, usage examples, and mathematical derivations, see:
- `scripts/stencil_gen/docs/group_velocity_reference.md`
- `scripts/stencil_gen/docs/brady2d_stability_reference.md` (layered stability pipeline for the Brady-Livescu 2D benchmark)
- `docs/brady2d_cpp_bridge_reference.md` (L8 C++ bridge architecture: Lua template, subprocess runner, runtime-parameterized spline families)
