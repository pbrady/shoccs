# Phase 36: 2D/3D Group Velocity and Varying Coefficients

**Goal:** Extend group velocity analysis to multi-dimensional operators and varying-coefficient problems, where eigenvalue analysis is expensive (O(N^6) for 3D) or ill-conditioned, but per-stencil group velocity remains O(1).

**Depends on:** Phase 35 (complete -- cut-cell group velocity with psi sweeps)

**Read first:**
- `scripts/stencil_gen/stencil_gen/group_velocity.py` (core module from Phases 34-35)
- `papers/GroupVelocityInFiniteDifferenceSchemes.pdf` (Trefethen 1982 -- Section 4: Group Velocity in Two Dimensions)
- `papers/StabilityAndGroupVelocity.pdf` (Trefethen 1983 -- Theorem 7: multi-dimensional GKS)
- `src/operators/` (C++ operator application -- understand how multi-dimensional operators are composed)
- `src/mesh/cartesian.hpp` (grid structure for 2D/3D)

**Test commands:**
```bash
cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "Test2D"
cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestVarying"
```

**Background (from Trefethen 1982, Section 4):**

In 2D, the group velocity becomes a vector C = grad_xi(omega) where xi = (xi, eta)
is the 2D wavenumber. For the equation u_t + a*u_x + b*u_y = 0 with dimension-by-
dimension FD application, the semi-discrete dispersion relation factors:
omega = a*kappa_x*(xi) + b*kappa_y*(eta). The group velocity vector is:
C_x = a * d(kappa_x*)/d(xi),  C_y = b * d(kappa_y*)/d(eta).

Key phenomena in 2D:
- Anisotropy: waves along grid axes vs diagonals travel at different speeds
- Group speed |C| depends on propagation angle theta
- Group propagation angle deviates from wave normal angle
- For 2nd-order schemes: |C| ~ 1 - (|xi|h)^2/24 * [3 + cos(4*theta)] (Eq. 4.8a)

For varying coefficients a(x,y), b(x,y), the analysis becomes local: at each grid
point, the stencil coefficients and local wave speed determine the local group
velocity. This is analogous to ray tracing in optics.

---

## Items

### 36.1 — 2D Group Velocity for Tensor-Product Stencils

- [x] **36.1a** Add `group_velocity_2d(kappa_x_star, kappa_y_star, xi_array, eta_array, a=1.0, b=1.0)` to `group_velocity.py`:
  - For dimension-by-dimension (tensor product) operators where the 2D dispersion relation factors as `omega = a*kappa_x*(xi) + b*kappa_y*(eta)`:
  - `C_x(xi, eta) = a * d(kappa_x*)/d(xi)` -- depends only on 1D xi analysis.
  - `C_y(xi, eta) = b * d(kappa_y*)/d(eta)` -- depends only on 1D eta analysis.
  - Computes: group speed `|C|`, group angle `theta_C = atan2(C_y, C_x)`, angle error `theta_C - theta_wave`.
  - Returns `GroupVelocity2DResult` with fields: `xi`, `eta`, `C_x`, `C_y`, `speed`, `angle`, `angle_error`.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "test_2d_basic"`

- [x] **36.1b** Add `anisotropy_profile(p, nu, theta_array, xi_mag)` to `group_velocity.py`:
  - For a given interior scheme order (E2-E8), compute the group speed and angle error as a function of wave propagation angle theta at a fixed wavenumber magnitude |xi|.
  - Uses the tensor-product dispersion relation with `a = cos(theta)`, `b = sin(theta)`.
  - Key output: speed ratio `|C|/|C_exact|` and angle deviation `theta_C - theta_wave` vs theta.
  - Trefethen's result: for 2nd-order, speed is maximized along diagonals (theta = pi/4) and minimized along axes. Verify this.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "test_anisotropy"`

- [x] **36.1c** Add `Test2DGroupVelocity` test class:
  - Test `test_axis_aligned_reduces_to_1d` -- for theta=0 (wave along x-axis), 2D group velocity C_x should match 1D group velocity. C_y = 0.
  - Test `test_diagonal_propagation` -- for theta=pi/4, verify both components are equal (C_x = C_y) by symmetry.
  - Test `test_anisotropy_e2_diagonal_fastest` -- covered by existing `test_anisotropy_axis_vs_diagonal`.
  - Test `test_anisotropy_e4_reduced` -- for E4, verify anisotropy is reduced compared to E2 (higher-order schemes reduce but don't eliminate grid anisotropy).
  - Test `test_angle_deviation_bounded` -- verify group propagation angle deviation from wave normal is bounded (< 5 degrees at xi_mag=0.7 for E2-E6).
  - Test `test_trefethen_eq_4_8a` -- for E2, verify the analytical formula `|C| ~ 1 - |xi|^2 * (3 + cos(4*theta)) / 8` matches numerical computation to leading order. (Note: coefficient is 1/8 from Taylor expansion; the plan's original 1/24 was incorrect.)
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "Test2D"` -- 10 tests pass.

### 36.2 — 2D Boundary Group Velocity

- [ ] **36.2a** Add `boundary_group_velocity_2d(p_x, p_y, boundary_rows_x, interior_y, theta_array, xi_mag)` to `group_velocity.py`:
  - At a boundary in x (left wall), the x-direction uses boundary stencils while y-direction uses interior stencils.
  - The 2D dispersion relation near the boundary is: `omega = a*kappa_x_bdy*(xi) + b*kappa_y_int*(eta)`.
  - Computes the 2D group velocity vector using boundary kappa_x* and interior kappa_y*.
  - Key question: does the boundary distort the group velocity angle, causing waves to be "bent" toward or away from the boundary?
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "test_2d_boundary"`

- [ ] **36.2b** Add `Test2DBoundaryGroupVelocity` test class:
  - Test `test_boundary_angle_distortion` -- for E2/E4 at a left boundary, compare group velocity angle at boundary rows vs interior. Quantify how much the boundary bends wave propagation.
  - Test `test_corner_region` -- at a corner (boundary in both x and y), both directions use boundary stencils. Compute 2D group velocity and check for anomalous behavior.
  - Test `test_no_outgoing_2d` -- verify no 2D modes have group velocity pointing into the domain (C dot n > 0 where n is the outward normal) at wavenumbers where the interior doesn't.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "Test2DBoundary"`

### 36.3 — Varying Coefficient Analysis

- [ ] **36.3a** Add `local_group_velocity(weights_func, x, xi_array)` to `group_velocity.py`:
  - For a varying-coefficient problem `u_t + a(x)*u_x = 0`, the stencil coefficients may be x-dependent (e.g., through a(x) scaling or through psi(x) for cut cells).
  - `weights_func(x)` returns (weights, offsets) at grid point x.
  - Computes local group velocity C(x, xi) at each grid point.
  - Returns a 2D array `C[i_x, i_xi]` of local group velocities.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "test_varying_basic"`

- [ ] **36.3b** Add `ray_trace_group_velocity(C_field, x_grid, xi_array, xi_0, x_0, t_final, dt)`:
  - Simple ray tracer: given a field of local group velocities C(x, xi), trace a ray from initial position (x_0, xi_0) following:
    - dx/dt = C(x, xi)  (group velocity)
    - dxi/dt = -dC/dx   (refraction, Trefethen Eq. 4.9b)
  - Uses simple Euler or RK4 integration.
  - Returns ray trajectory (x(t), xi(t)).
  - This is a diagnostic tool: rays that reflect back from a boundary region suggest energy trapping. Rays that converge to a point suggest caustic formation.
  - File: `scripts/stencil_gen/stencil_gen/group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "test_ray_trace"`

- [ ] **36.3c** Add `TestVaryingCoefficientGroupVelocity` test class:
  - Test `test_constant_coefficient_uniform_gv` -- with a(x) = 1 everywhere, local group velocity equals the uniform interior group velocity at all x (except near boundaries).
  - Test `test_linear_coefficient_gv_variation` -- with a(x) = 1 + 0.5*x on [0,1], verify local group velocity varies smoothly with x.
  - Test `test_sign_change_interface` -- with a(x) changing sign (a_- < 0 < a_+), verify Trefethen's prediction that the interface always has outgoing modes (Theorem 5 of Trefethen 1983). This is a known instability mechanism.
  - Test `test_ray_trace_uniform` -- in a uniform medium, rays should be straight lines at constant xi. Verify to numerical precision.
  - Test `test_ray_trace_refraction` -- in a medium with a(x) varying, rays should bend according to Snell's law analogue. Verify against analytical prediction for simple a(x) profiles.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestVarying"`

### 36.4 — Scaling Comparison: GV vs Eigenvalue

- [ ] **36.4a** Add `TestScalingComparison` test class:
  - Test `test_1d_scaling` -- time GV analysis vs eigenvalue analysis for N = 50, 100, 200, 400, 800 in 1D. Print a table showing:
    - GV time (should be ~constant, independent of N)
    - Eigenvalue time (should grow as O(N^3))
    - Both give the same stability conclusion
  - Test `test_2d_scaling_projection` -- for 2D on an NxN grid (N = 20, 40, 80), the full operator has N^2 eigenvalues. Time the eigenvalue decomposition. Then time the GV analysis (which only needs the 1D stencils, O(1)). Print the projected speedup.
  - Test `test_3d_scaling_projection` -- extrapolate to 3D on NxNxN grids. Full eigenvalue: O(N^9). GV analysis: O(1) per stencil. Print projected times to demonstrate the motivation for this work.
  - File: `scripts/stencil_gen/tests/test_group_velocity.py`
  - Test: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "TestScaling" -s`

---

## Completion Criteria

- 2D tensor-product group velocity analysis matches Trefethen's analytical predictions (anisotropy, diagonal preference, Eq. 4.8a).
- Boundary distortion of 2D group velocity is quantified.
- Varying-coefficient analysis produces smooth, physically meaningful local group velocity fields.
- Ray tracing works for simple test cases and agrees with analytical predictions.
- Scaling comparison demonstrates the practical advantage over eigenvalue analysis for multi-dimensional problems.
- All new tests pass: `cd scripts/stencil_gen && uv run pytest tests/test_group_velocity.py -x -q -k "Test2D or TestVarying or TestScaling"`
- No existing tests broken: `cd scripts/stencil_gen && uv run pytest tests/ -x -q -k "not TestMathematicaWorkflow and not TestPolynomialFullStencil and not TestE4CodeGeneration"`
