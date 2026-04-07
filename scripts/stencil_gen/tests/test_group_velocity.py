"""Tests for group velocity analysis module."""

import numpy as np
import pytest

from stencil_gen.group_velocity import (
    GKSModeInfo,
    GroupVelocity2DResult,
    GroupVelocityProfile,
    PsiSweepResult,
    boundary_group_velocity,
    boundary_group_velocity_classical,
    cut_cell_group_velocity,
    gks_group_velocity_check,
    group_velocity,
    group_velocity_2d,
    group_velocity_error,
    group_velocity_exact,
    group_velocity_exact_nonuniform,
    interior_group_velocity,
    modified_wavenumber,
    modified_wavenumber_nonuniform,
    phase_velocity,
    psi_sweep_group_velocity,
)


class TestCoreGroupVelocity:
    """Core group velocity computation tests (34.1b)."""

    N_XI = 500

    def test_exact_scheme_unity_group_velocity(self):
        """For exact differentiation (central difference of delta), C(xi) = 1."""
        # For D1 approximating d/dx, kappa*(xi) has Im(kappa*) approximating xi.
        # Group velocity: C(xi) = d(Im(kappa*))/d(xi).
        # For exact differentiation, Im(kappa*) = xi, so C = 1.
        #
        # No finite stencil gives kappa* = i*xi exactly, so we test the
        # "spectral" limit: use a high-order scheme (E8) and verify C ≈ 1
        # at low wavenumbers where the truncation error is negligible.
        from stencil_gen.interior import derive_interior, full_gamma_array

        p = 4  # E8 scheme
        coeffs = derive_interior(0, p, 1)
        w = [float(c) for c in full_gamma_array(coeffs)]
        nodes = list(range(-p, p + 1))
        xi = np.linspace(0.01, 0.5, self.N_XI)  # well-resolved wavenumbers

        C = group_velocity_exact(w, 0, nodes, xi)
        # At low xi, C should be very close to 1 for high-order scheme
        assert np.allclose(C, 1.0, atol=1e-4), (
            f"E8 group velocity should be ~1 at low xi, got max error "
            f"{np.max(np.abs(C - 1.0)):.2e}"
        )

    def test_numerical_vs_analytical_gradient(self):
        """group_velocity() and group_velocity_exact() agree within numerical tolerance."""
        # Use E2 interior stencil: weights [-1/2, 0, 1/2]
        weights = [-0.5, 0.0, 0.5]
        nodes = [-1, 0, 1]
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)

        kstar = modified_wavenumber(weights, 0, nodes, xi)
        C_numerical = group_velocity(kstar, xi)
        C_analytical = group_velocity_exact(weights, 0, nodes, xi)

        # Numerical differentiation (2nd-order central diff) agrees to ~O(h^2)
        assert np.allclose(C_numerical, C_analytical, atol=1e-4), (
            f"Numerical vs analytical group velocity max diff: "
            f"{np.max(np.abs(C_numerical - C_analytical)):.2e}"
        )

    def test_phase_velocity_low_xi_limit(self):
        """Phase velocity c(xi) -> 1 as xi -> 0 for any consistent scheme."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        for p in [1, 2, 3]:
            coeffs = derive_interior(0, p, 1)
            w = [float(c) for c in full_gamma_array(coeffs)]
            nodes = list(range(-p, p + 1))
            xi = np.linspace(0.01, 0.1, 50)

            kstar = modified_wavenumber(w, 0, nodes, xi)
            c = phase_velocity(kstar, xi)

            # At small xi, phase velocity should be close to 1
            assert abs(c[0] - 1.0) < 1e-3, (
                f"p={p}: phase velocity at xi={xi[0]:.3f} is {c[0]:.6f}, "
                f"expected ~1.0"
            )

    def test_second_order_known_values(self):
        """For E2 interior stencil [-1/2, 0, 1/2], C(xi) = cos(xi)."""
        weights = [-0.5, 0.0, 0.5]
        nodes = [-1, 0, 1]
        xi = np.linspace(0, np.pi, self.N_XI)

        # Analytical group velocity
        C = group_velocity_exact(weights, 0, nodes, xi)
        expected = np.cos(xi)

        assert np.allclose(C, expected, atol=1e-14), (
            f"E2 group velocity should be cos(xi), max error: "
            f"{np.max(np.abs(C - expected)):.2e}"
        )

        # Also verify via modified wavenumber
        kstar = modified_wavenumber(weights, 0, nodes, xi)
        # kappa* = i*sin(xi), so Im(kappa*) = sin(xi)
        assert np.allclose(np.imag(kstar), np.sin(xi), atol=1e-14)
        assert np.allclose(np.real(kstar), 0.0, atol=1e-14)

    def test_interior_group_velocity_e2(self):
        """Smoke test for interior_group_velocity(): E2 (p=1) returns correct profile."""
        xi = np.linspace(0, np.pi, self.N_XI)
        profile = interior_group_velocity(p=1, nu=1, xi_array=xi)

        # Correct type and order
        assert isinstance(profile, GroupVelocityProfile)
        assert profile.order == 2

        # E2 group velocity is cos(xi)
        assert np.allclose(profile.group_velocity, np.cos(xi), atol=1e-14)

        # Cutoff where C first goes to zero: cos(xi)=0 at xi=pi/2
        assert abs(profile.cutoff_xi - np.pi / 2) < 2 * np.pi / self.N_XI


class TestInteriorGroupVelocity:
    """Interior scheme group velocity analysis (34.2b)."""

    N_XI = 2000

    def test_e2_group_velocity_is_cos_xi(self):
        """E2 (p=1) interior scheme: C(xi) = cos(xi) exactly."""
        xi = np.linspace(0, np.pi, self.N_XI)
        profile = interior_group_velocity(p=1, nu=1, xi_array=xi)
        assert np.allclose(profile.group_velocity, np.cos(xi), atol=1e-14)

    def test_error_amplification_factor(self):
        """Group velocity error is (2p+1) times phase velocity error at leading order.

        For a 2p-th order scheme, Im(kappa*) = xi - a*xi^(2p+1) + ..., so:
          phase velocity error:  c - 1 = -a*xi^(2p) + ...
          group velocity error:  C - 1 = -(2p+1)*a*xi^(2p) + ...
        The leading-order ratio is (2p+1).

        Note: the original plan stated (2p-1), but the correct factor from
        the Taylor expansion is (2p+1).  Verified numerically below.
        """
        xi = np.linspace(0.01, 0.15, 500)
        for p in [1, 2, 3, 4]:
            profile = interior_group_velocity(p=p, nu=1, xi_array=xi)
            gv_err = profile.group_velocity - 1.0
            pv_err = profile.phase_velocity - 1.0
            # Use a point near the middle where errors are small but nonzero
            idx = len(xi) // 2
            ratio = gv_err[idx] / pv_err[idx]
            expected = 2 * p + 1
            assert abs(ratio - expected) < 0.1, (
                f"E{2*p}: gv_err/pv_err ratio = {ratio:.2f}, expected {expected}"
            )

    def test_cutoff_wavenumber(self):
        """Cutoff xi (where C = 0) increases with order.

        Higher-order schemes resolve more wavenumbers before group velocity
        reversal, so cutoff_xi moves to higher values.
        """
        xi = np.linspace(0, np.pi, self.N_XI)
        cutoffs = []
        for p in [1, 2, 3, 4]:
            profile = interior_group_velocity(p=p, nu=1, xi_array=xi)
            cutoffs.append(profile.cutoff_xi)

        # Cutoffs should be strictly increasing
        for i in range(len(cutoffs) - 1):
            assert cutoffs[i] < cutoffs[i + 1], (
                f"E{2*(i+1)} cutoff {cutoffs[i]:.4f} >= "
                f"E{2*(i+2)} cutoff {cutoffs[i+1]:.4f}"
            )

        # E2 cutoff should be pi/2
        assert abs(cutoffs[0] - np.pi / 2) < 2 * np.pi / self.N_XI

    def test_group_velocity_sign_reversal(self):
        """C(xi) < 0 for all xi beyond cutoff (parasitic regime)."""
        xi = np.linspace(0, np.pi, self.N_XI)
        for p in [1, 2, 3, 4]:
            profile = interior_group_velocity(p=p, nu=1, xi_array=xi)
            beyond = xi > profile.cutoff_xi
            assert np.any(beyond), f"E{2*p}: no points beyond cutoff"
            C_beyond = profile.group_velocity[beyond]
            assert np.all(C_beyond <= 0), (
                f"E{2*p}: found positive C beyond cutoff, "
                f"max = {np.max(C_beyond):.4e}"
            )

    def test_group_velocity_comparison_table(self, capsys):
        """Print formatted comparison table for E2/E4/E6/E8 interior schemes.

        Diagnostic test -- always passes, prints useful data with -s flag.
        """
        xi = np.linspace(0, np.pi, 2000)
        xi_quarter = np.pi / 4
        xi_half = np.pi / 2

        header = (
            f"{'Scheme':<8} {'Order':>5} {'Cutoff xi/pi':>12} "
            f"{'|C_err| xi=pi/4':>16} {'|C_err| xi=pi/2':>16} {'min C':>8}"
        )
        print()
        print(header)
        print("-" * len(header))

        for p in [1, 2, 3, 4]:
            profile = interior_group_velocity(p=p, nu=1, xi_array=xi)
            C = profile.group_velocity

            # Interpolate C at specific xi values
            idx_q = np.argmin(np.abs(xi - xi_quarter))
            idx_h = np.argmin(np.abs(xi - xi_half))
            err_q = abs(C[idx_q] - 1.0)
            err_h = abs(C[idx_h] - 1.0)
            min_C = np.min(C)

            print(
                f"E{2*p:<6} {2*p:>5} {profile.cutoff_xi/np.pi:>12.4f} "
                f"{err_q:>16.6e} {err_h:>16.6e} {min_C:>8.4f}"
            )


class TestBoundaryGroupVelocity:
    """Boundary closure group velocity analysis (34.3)."""

    N_XI = 1000

    def test_boundary_gv_returns_all_rows(self):
        """boundary_group_velocity returns a profile for each boundary row."""
        xi = np.linspace(0, np.pi, self.N_XI)
        # E2, q=1, nextra=0, tension kernel with small sigma
        profiles = boundary_group_velocity(
            p=1, q=1, nextra=0, nu=1, sigma=0.1, kernel="tension", xi_array=xi,
        )
        # For E2 nu=1: r = q+1+nextra = 2
        assert len(profiles) == 2
        for i in range(2):
            assert i in profiles
            assert isinstance(profiles[i], GroupVelocityProfile)
            assert len(profiles[i].xi) == self.N_XI
            assert profiles[i].order == 1  # boundary accuracy order = q

    def test_boundary_gv_bounded(self):
        """For E2/E4 at small sigma, |C(xi)| is bounded (no blow-up)."""
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        configs = [
            (1, 1, 0, 0.1),  # E2, q=1
            (2, 3, 0, 0.1),  # E4, q=3
        ]
        for p, q, nextra, sigma in configs:
            profiles = boundary_group_velocity(
                p=p, q=q, nextra=nextra, nu=1, sigma=sigma,
                kernel="tension", xi_array=xi,
            )
            for i, prof in profiles.items():
                C = prof.group_velocity
                assert np.all(np.isfinite(C)), (
                    f"p={p}, row {i}: non-finite group velocity"
                )
                assert np.max(np.abs(C)) < 100, (
                    f"p={p}, row {i}: |C| blow-up, max={np.max(np.abs(C)):.2e}"
                )

    def test_boundary_row0_low_xi_near_unity(self):
        """Boundary row 0 group velocity should approach 1 at low xi (consistent scheme)."""
        xi = np.linspace(0.01, 0.3, self.N_XI)
        profiles = boundary_group_velocity(
            p=2, q=3, nextra=0, nu=1, sigma=0.1, kernel="tension", xi_array=xi,
        )
        # Row 0 evaluates derivative at grid point 0 using a one-sided stencil.
        # At low xi, if the scheme is consistent, C should be near 1.
        C = profiles[0].group_velocity
        assert abs(C[0] - 1.0) < 0.5, (
            f"Boundary row 0 C at xi={xi[0]:.3f} = {C[0]:.4f}, expected ~1"
        )

    def test_cutoff_handles_oscillating_c(self):
        """cutoff_xi reflects persistent (not transient) sign reversal.

        Boundary stencils can have C(xi) that dips below zero briefly
        then recovers positive.  The cutoff should mark where C goes
        *permanently* non-positive, not the first transient dip.
        """
        from stencil_gen.group_velocity import _build_profile

        # Synthetic one-sided stencil (i_eval=0, nodes=[0..4]) whose
        # group velocity oscillates: C = 3cos(xi) - 6cos(2xi) + 6cos(3xi) - 2cos(4xi).
        # C(0) = 1, dips negative near xi~0.7, recovers strongly positive
        # around xi~pi/2, then goes permanently negative near xi~2.7.
        weights = [0.0, 3.0, -3.0, 2.0, -0.5]
        nodes = [0, 1, 2, 3, 4]
        xi = np.linspace(0, np.pi, 2000)

        profile = _build_profile(weights, 0, nodes, xi, order=1)
        C = profile.group_velocity

        # Verify oscillation: C has a transient dip below zero AND
        # later recovery to positive values before the final descent.
        first_neg = None
        for idx in range(1, len(xi)):
            if C[idx] <= 0.0:
                first_neg = float(xi[idx])
                break
        recovery = False
        if first_neg is not None:
            for idx in range(1, len(xi)):
                if xi[idx] > first_neg and C[idx] > 0.0:
                    recovery = True
                    break
        assert first_neg is not None, "expected a transient negative dip"
        assert recovery, "expected C to recover positive after first dip"

        # Cutoff must be beyond the transient dip (at the persistent crossing)
        assert profile.cutoff_xi > first_neg + 0.5, (
            f"cutoff_xi={profile.cutoff_xi:.3f} is too close to first "
            f"transient dip at xi={first_neg:.3f}"
        )

        # Beyond cutoff, C stays non-positive
        beyond = xi > profile.cutoff_xi
        if np.any(beyond):
            assert np.all(C[beyond] <= 0.0), (
                f"C has positive values beyond cutoff_xi={profile.cutoff_xi:.3f}"
            )

    def test_boundary_vs_interior_gv_error(self):
        """Boundary row 0 has larger GV error than interior; no sign reversal at low xi.

        Row 0 (fully one-sided) should have larger error than the symmetric
        interior stencil.  Rows closer to the interior may have smaller error
        because they use a wider stencil.

        At well-resolved wavenumbers (xi < pi/4), no boundary row should
        reverse the sign of C (which would mean energy propagating backwards).
        """
        xi = np.linspace(0.01, np.pi, self.N_XI)
        for p, q, nextra, sigma in [(2, 3, 0, 0.1), (3, 5, 0, 0.1)]:
            interior = interior_group_velocity(p=p, nu=1, xi_array=xi)
            boundary = boundary_group_velocity(
                p=p, q=q, nextra=nextra, nu=1, sigma=sigma,
                kernel="tension", xi_array=xi,
            )
            resolved = xi < np.pi / 4
            # Row 0 (fully one-sided) should have the largest error
            C_row0 = boundary[0].group_velocity[resolved]
            C_int = interior.group_velocity[resolved]
            row0_err = np.max(np.abs(C_row0 - 1.0))
            int_err = np.max(np.abs(C_int - 1.0))
            assert row0_err >= int_err, (
                f"p={p}: row 0 error ({row0_err:.4e}) smaller than "
                f"interior ({int_err:.4e})"
            )
            # No sign reversal at well-resolved wavenumbers for any row
            for i, prof in boundary.items():
                C_bnd = prof.group_velocity[resolved]
                assert np.all(C_bnd > 0), (
                    f"p={p}, row {i}: C < 0 at resolved xi<pi/4 "
                    f"(min={np.min(C_bnd):.4e})"
                )

    def test_parasitic_direction_at_boundary(self):
        """Check for parasitic outgoing modes at the boundary.

        For u_t + u_x = 0 on a left boundary, physical waves propagate rightward
        (C > 0).  If a boundary stencil creates a mode with C < 0 at a wavenumber
        where the interior has C > 0, that's a parasitic mode propagating energy
        out through the boundary — the hallmark of GKS instability.

        Here we check the complementary concern for an inflow boundary: modes
        with C > 0 (into the domain) at wavenumbers where the interior has C < 0
        (energy should be propagating outward).  Such modes create spontaneous
        radiation of energy into the domain.
        """
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        for p, q, nextra, sigma in [(1, 1, 0, 0.1), (2, 3, 0, 0.1)]:
            interior = interior_group_velocity(p=p, nu=1, xi_array=xi)
            boundary = boundary_group_velocity(
                p=p, q=q, nextra=nextra, nu=1, sigma=sigma,
                kernel="tension", xi_array=xi,
            )
            # In the parasitic regime (beyond interior cutoff), check if
            # boundary rows create C > 0 where interior has C < 0.
            parasitic = xi > interior.cutoff_xi
            if not np.any(parasitic):
                continue
            for i, prof in boundary.items():
                C_bnd_parasitic = prof.group_velocity[parasitic]
                # Flag if boundary creates strongly positive C in parasitic regime.
                # Small positive values are tolerable; large positive means energy
                # radiation into the domain.
                max_positive = np.max(C_bnd_parasitic) if len(C_bnd_parasitic) > 0 else 0
                # This is a diagnostic — we record but don't fail on small positives.
                # A strongly positive boundary C in the parasitic regime is suspicious.
                assert max_positive < 5.0, (
                    f"p={p}, row {i}: boundary C strongly positive ({max_positive:.2f}) "
                    f"in parasitic regime — potential GKS instability source"
                )


class TestBoundaryClassical:
    """Classical (non-RBF) boundary stencil group velocity analysis (34.3b)."""

    N_XI = 1000

    @pytest.fixture(scope="class")
    def e4_classical(self):
        """Derive E4 boundary rows with conservation and known-good alpha values."""
        from sympy import symbols

        from stencil_gen.boundary import derive_boundary
        from stencil_gen.conservation import build_conservation_system, solve_conservation

        result = derive_boundary(p=2, nu=1, s=0)
        equations, w_syms, last_free = build_conservation_system(
            result.r, result.t, 2, result.rows, result.interior_coeffs,
        )
        _, updated_rows = solve_conservation(
            equations, w_syms, last_free, result.all_free_params, result.rows,
        )
        # Known-good alpha values from E4u_1.t.cpp (same as test_boundary.py)
        a0, a1 = symbols("alpha_0 alpha_1")
        alpha_values = {a0: -0.7733323791884821, a1: 0.1623961700641681}
        return updated_rows, alpha_values

    def test_classical_returns_all_rows(self, e4_classical):
        """boundary_group_velocity_classical returns a profile for each boundary row."""
        updated_rows, alpha_values = e4_classical
        xi = np.linspace(0, np.pi, self.N_XI)
        profiles = boundary_group_velocity_classical(
            updated_rows, alpha_values, order=3, xi_array=xi,
        )
        # E4 (p=2): r = 3 boundary rows
        assert len(profiles) == 3
        for i in range(3):
            assert i in profiles
            assert isinstance(profiles[i], GroupVelocityProfile)
            assert profiles[i].order == 3

    def test_classical_coefficients_finite(self, e4_classical):
        """All evaluated coefficients are finite (no unresolved symbols)."""
        updated_rows, alpha_values = e4_classical
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        profiles = boundary_group_velocity_classical(
            updated_rows, alpha_values, order=3, xi_array=xi,
        )
        for i, prof in profiles.items():
            assert np.all(np.isfinite(prof.group_velocity)), (
                f"Row {i}: non-finite group velocity"
            )
            assert np.all(np.isfinite(prof.kappa_star)), (
                f"Row {i}: non-finite modified wavenumber"
            )

    def test_classical_row0_low_xi(self, e4_classical):
        """Classical E4 row 0 group velocity near unity at low xi."""
        updated_rows, alpha_values = e4_classical
        xi = np.linspace(0.01, 0.3, self.N_XI)
        profiles = boundary_group_velocity_classical(
            updated_rows, alpha_values, order=3, xi_array=xi,
        )
        C = profiles[0].group_velocity
        assert abs(C[0] - 1.0) < 0.5, (
            f"Classical E4 row 0 C at xi={xi[0]:.3f} = {C[0]:.4f}, expected ~1"
        )

    def test_classical_bounded(self, e4_classical):
        """Group velocity is bounded (no blow-up) for all boundary rows."""
        updated_rows, alpha_values = e4_classical
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        profiles = boundary_group_velocity_classical(
            updated_rows, alpha_values, order=3, xi_array=xi,
        )
        for i, prof in profiles.items():
            assert np.max(np.abs(prof.group_velocity)) < 100, (
                f"Row {i}: |C| blow-up, max={np.max(np.abs(prof.group_velocity)):.2e}"
            )

    def test_classical_e4_boundary_gv(self, e4_classical):
        """Classical E4 boundary stencils: no sign reversal at resolved wavenumbers.

        Uses the known-good alpha values from E4u_1.t.cpp.  At well-resolved
        wavenumbers (xi < pi/2), all boundary rows should have C > 0, meaning
        no parasitic energy reversal in the resolved regime.
        """
        updated_rows, alpha_values = e4_classical
        xi = np.linspace(0.01, np.pi, self.N_XI)
        profiles = boundary_group_velocity_classical(
            updated_rows, alpha_values, order=3, xi_array=xi,
        )
        resolved = xi < np.pi / 2
        for i, prof in profiles.items():
            C_resolved = prof.group_velocity[resolved]
            assert np.all(C_resolved > 0), (
                f"Classical E4 row {i}: C < 0 at resolved xi "
                f"(min={np.min(C_resolved):.4e}), parasitic sign reversal"
            )


class TestGKSDiagnostic:
    """GKS group velocity diagnostic tests (34.3e).

    Tests for :func:`gks_group_velocity_check`, which bridges per-stencil
    group velocity analysis with full-operator eigenvalue analysis by
    identifying boundary-localized, nearly-neutral eigenmodes whose group
    velocity radiates energy into the domain (GKS instability signature).
    """

    N_XI = 1000

    def test_stable_scheme_no_outgoing_modes(self):
        """Stable E2 scheme: no boundary modes radiate energy into the domain.

        E2 with tension kernel (universally stable for all sigma) produces no
        boundary-localized nearly-neutral eigenmodes at sigma=10, confirming
        the diagnostic returns clean results for a well-behaved scheme.
        """
        from stencil_gen.phs import build_diff_matrix_rbf

        n = 40
        xi = np.linspace(0, np.pi, self.N_XI)
        D = build_diff_matrix_rbf(
            n, p=1, q=1, epsilon=10.0, kernel="tension", nu=1, nextra=0,
        )
        modes = gks_group_velocity_check(D, xi)

        outgoing = [m for m in modes if m.is_outgoing]
        assert len(outgoing) == 0, (
            f"Stable E2 has {len(outgoing)} outgoing modes: "
            + "; ".join(
                f"lam={m.eigenvalue:.4f}, xi={m.boundary_wavenumber:.3f}, "
                f"C={m.group_velocity:.3f}"
                for m in outgoing
            )
        )

    def test_known_unstable_extrapolation(self):
        """GKS diagnostic detects parasitic boundary mode in E4 PHS.

        The original plan aimed to test with extrapolation outflow BC (known
        GKS-unstable for leapfrog, Trefethen 1983).  However, the leapfrog
        instability is time-discrete: the fully-discrete dispersion relation
        differs from the semi-discrete one, and the semi-discrete eigenvalue
        framework used here cannot capture that effect.  A time-discrete
        extension using the leapfrog dispersion relation is future work.

        Instead, we test with E4 PHS (sigma=0), which reliably produces a
        boundary-localized, nearly-neutral eigenmode whose dominant wavenumber
        is in the parasitic regime (xi near pi).  The interior group velocity
        at that wavenumber is strongly negative (C ~ -1.7), meaning energy
        flows from the right boundary into the domain.  The mode is slightly
        damped (Re ~ -0.007), so it doesn't cause exponential growth, but the
        diagnostic correctly flags the suspicious group velocity direction.
        """
        from stencil_gen.phs import build_diff_matrix_rbf

        n = 40
        xi = np.linspace(0, np.pi, self.N_XI)
        D = build_diff_matrix_rbf(
            n, p=2, q=3, epsilon=0.0, kernel="tension", nu=1, nextra=0,
        )
        modes = gks_group_velocity_check(D, xi)

        # E4 PHS produces at least one outgoing boundary mode
        outgoing = [m for m in modes if m.is_outgoing]
        assert len(outgoing) >= 1, (
            f"Expected at least 1 outgoing mode for E4 PHS, got {len(modes)} "
            f"total modes with 0 outgoing"
        )

        # The outgoing mode should be in the parasitic regime (high xi)
        # where interior group velocity is negative
        mode = outgoing[0]
        assert isinstance(mode, GKSModeInfo)
        assert mode.boundary_wavenumber > np.pi / 2, (
            f"Outgoing mode wavenumber {mode.boundary_wavenumber:.3f} should be "
            f"in parasitic regime (> pi/2)"
        )
        assert mode.group_velocity < 0, (
            f"Outgoing mode group velocity {mode.group_velocity:.3f} should be "
            f"negative (energy flowing into domain from right boundary)"
        )
        # Mode is nearly-neutral (slightly damped, not growing)
        assert mode.eigenvalue.real < 0, (
            f"Mode Re(lambda) = {mode.eigenvalue.real:.6e} should be negative "
            f"(damped, not growing)"
        )


class TestNonuniformModWavenumber:
    """Tests for modified_wavenumber_nonuniform and group_velocity_exact_nonuniform (35.1b)."""

    N_XI = 500

    def test_nonuniform_mod_wavenumber(self):
        """With integer offsets, nonuniform version matches uniform version exactly."""
        weights = [-0.5, 0.0, 0.5]
        nodes = [-1, 0, 1]
        i_eval = 0
        offsets = [n - i_eval for n in nodes]  # [-1, 0, 1]
        xi = np.linspace(0, np.pi, self.N_XI)

        kstar_uniform = modified_wavenumber(weights, i_eval, nodes, xi)
        kstar_nonuniform = modified_wavenumber_nonuniform(weights, offsets, xi)

        assert np.allclose(kstar_uniform, kstar_nonuniform, atol=1e-14), (
            f"Nonuniform should match uniform for integer offsets, "
            f"max diff = {np.max(np.abs(kstar_uniform - kstar_nonuniform)):.2e}"
        )

    def test_nonuniform_gv_matches_uniform(self):
        """With integer offsets, nonuniform GV matches uniform GV exactly."""
        weights = [-0.5, 0.0, 0.5]
        nodes = [-1, 0, 1]
        i_eval = 0
        offsets = [n - i_eval for n in nodes]
        xi = np.linspace(0, np.pi, self.N_XI)

        C_uniform = group_velocity_exact(weights, i_eval, nodes, xi)
        C_nonuniform = group_velocity_exact_nonuniform(weights, offsets, xi)

        assert np.allclose(C_uniform, C_nonuniform, atol=1e-14), (
            f"Nonuniform GV should match uniform for integer offsets, "
            f"max diff = {np.max(np.abs(C_uniform - C_nonuniform)):.2e}"
        )

    def test_nonuniform_fractional_offset_bounded(self):
        """Nonuniform with fractional offsets produces bounded, finite results."""
        # Simulate a cut-cell-like stencil with wall at -0.3
        weights = [-0.3, 0.7, 0.1, -0.5]
        offsets = [-0.3, 0.0, 1.0, 2.0]
        xi = np.linspace(0.01, np.pi, self.N_XI)

        kstar = modified_wavenumber_nonuniform(weights, offsets, xi)
        C = group_velocity_exact_nonuniform(weights, offsets, xi)

        assert np.all(np.isfinite(kstar)), "kappa* should be finite"
        assert np.all(np.isfinite(C)), "Group velocity should be finite"
        assert np.max(np.abs(C)) < 100, f"|C| blow-up: max={np.max(np.abs(C)):.2e}"


class TestCutCellGroupVelocity:
    """Cut-cell group velocity analysis (35.1c)."""

    N_XI = 1000

    @pytest.fixture(scope="class")
    def e2_1_cut_cell(self):
        """Derive E2_1 cut-cell stencil (symbolic in psi and alpha)."""
        from sympy import Symbol

        from stencil_gen.temo import E2_1, derive_cut_cell_scheme

        psi = Symbol("psi")
        result = derive_cut_cell_scheme(E2_1, psi)
        return result, psi

    def test_psi_1_matches_uniform(self, e2_1_cut_cell):
        """At psi=1, cut-cell group velocity matches uniform boundary GV.

        By TEMO construction, B(psi=1) = B_u (the uniform boundary stencil).
        At psi=1 the wall is at position -1 (uniform spacing), so the cut-cell
        stencil effectively reduces to a uniform-grid boundary stencil.

        We compare by evaluating the TEMO's own B_u at alpha=0 and verifying
        the group velocity profiles agree at resolved wavenumbers.
        """
        from stencil_gen.group_velocity import _build_profile
        from stencil_gen.temo import E2_1 as scheme, derive_uniform_boundary_for_temo

        result, psi = e2_1_cut_cell
        xi = np.linspace(0.01, np.pi, self.N_XI)
        alpha_vals = {s: 0 for s in result.alpha_symbols}

        cc_profiles = cut_cell_group_velocity(
            result, psi, psi_val=1.0, alpha_values=alpha_vals, xi_array=xi,
        )

        # At psi=1, all rows should give well-behaved GV
        for i, prof in cc_profiles.items():
            assert np.all(np.isfinite(prof.group_velocity)), (
                f"Row {i}: non-finite GV at psi=1"
            )
            # At low xi, C should be near 1 (consistent scheme)
            low_xi = xi < 0.3
            C_low = prof.group_velocity[low_xi]
            assert abs(C_low[0] - 1.0) < 0.5, (
                f"Row {i}: C at low xi = {C_low[0]:.4f}, expected ~1"
            )

        # Get the TEMO uniform boundary B_u and compute its GV.
        # At psi=1, the wall column coefficient is zero for rows 0..r-1,
        # so the cut-cell stencil exactly matches the uniform one.
        ur = derive_uniform_boundary_for_temo(scheme)
        B_u = ur.B_u
        u_alpha_vals = {s: 0 for s in ur.alpha_symbols}
        t = B_u.cols
        nodes = list(range(t))

        for i in range(B_u.rows):
            w_uni = [float(B_u[i, j].xreplace(u_alpha_vals))
                     if hasattr(B_u[i, j], 'xreplace') else float(B_u[i, j])
                     for j in range(t)]
            uni_prof = _build_profile(w_uni, i, nodes, xi, order=scheme.q)
            max_diff = np.max(np.abs(
                cc_profiles[i].group_velocity - uni_prof.group_velocity
            ))
            assert max_diff < 1e-10, (
                f"Row {i}: psi=1 cut-cell GV should match TEMO uniform "
                f"exactly (wall coeff=0), diff = {max_diff:.2e}"
            )

    def test_psi_0_degenerate_bounded(self, e2_1_cut_cell):
        """At psi=0 (degenerate mesh), group velocity is bounded and finite.

        The degenerate point collocates the wall with grid point 0, so the
        stencil degenerates gracefully (by TEMO design principle).
        """
        result, psi = e2_1_cut_cell
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        alpha_vals = {s: 0 for s in result.alpha_symbols}

        profiles = cut_cell_group_velocity(
            result, psi, psi_val=0.0, alpha_values=alpha_vals, xi_array=xi,
        )

        for i, prof in profiles.items():
            C = prof.group_velocity
            assert np.all(np.isfinite(C)), (
                f"Row {i}: non-finite GV at psi=0"
            )
            assert np.max(np.abs(C)) < 100, (
                f"Row {i}: |C| blow-up at psi=0, max={np.max(np.abs(C)):.2e}"
            )

    def test_e2_1_cut_cell_gv_smooth_in_psi(self, e2_1_cut_cell):
        """E2_1 group velocity varies smoothly with psi (no discontinuous jumps).

        Compute C(xi) at 11 evenly spaced psi values in [0, 1] and verify
        that the group velocity profile at resolved wavenumbers changes
        smoothly between adjacent values (bounded derivative dC/dpsi).
        """
        result, psi = e2_1_cut_cell
        xi = np.linspace(0.01, np.pi / 2, self.N_XI)
        alpha_vals = {s: 0 for s in result.alpha_symbols}

        psi_values = np.linspace(0.0, 1.0, 11)
        all_profiles = {}
        for pv in psi_values:
            all_profiles[pv] = cut_cell_group_velocity(
                result, psi, psi_val=float(pv), alpha_values=alpha_vals,
                xi_array=xi,
            )

        # For each row, check that adjacent psi values give bounded dC/dpsi
        R = result.floating.rows
        for i in range(R):
            for k in range(len(psi_values) - 1):
                pv0, pv1 = float(psi_values[k]), float(psi_values[k + 1])
                C0 = all_profiles[pv0][i].group_velocity
                C1 = all_profiles[pv1][i].group_velocity
                dpsi = pv1 - pv0
                max_deriv = np.max(np.abs(C1 - C0)) / dpsi
                # dC/dpsi should be finite (no discontinuity).  A smooth
                # rational function in psi can have large but bounded
                # derivatives, especially near psi=0 where the stencil
                # degenerates.  Threshold of 200 catches genuine
                # discontinuities while allowing smooth variation.
                assert max_deriv < 200, (
                    f"Row {i}: dC/dpsi too large between psi={pv0:.2f} and "
                    f"psi={pv1:.2f}, max|dC/dpsi| = {max_deriv:.1f}"
                )


class TestPsiSweepGroupVelocity:
    """Psi sweep group velocity analysis (35.2b)."""

    N_XI = 500

    def test_e2_1_psi_sweep(self):
        """Sweep psi in [0, 1] for E2_1; verify result structure and diagnostics.

        Computes group velocity at 11 psi values and verifies:
        - PsiSweepResult is well-formed with all expected fields.
        - All profiles are finite and bounded.
        - No parasitic sign reversal at well-resolved wavenumbers (xi < pi/2)
          for non-degenerate psi values (psi >= 0.1).
        """
        from stencil_gen.temo import E2_1

        xi = np.linspace(0.01, np.pi, self.N_XI)
        psi_values = np.linspace(0.0, 1.0, 11)

        result = psi_sweep_group_velocity(
            E2_1, psi_values, alpha_values={}, xi_array=xi,
        )

        assert isinstance(result, PsiSweepResult)
        assert len(result.profiles) == len(psi_values)
        assert result.min_C < float("inf")

        # All profiles should be finite and bounded
        for pv, profs in result.profiles.items():
            for row_idx, prof in profs.items():
                assert np.all(np.isfinite(prof.group_velocity)), (
                    f"psi={pv}, row {row_idx}: non-finite GV"
                )
                assert np.max(np.abs(prof.group_velocity)) < 100, (
                    f"psi={pv}, row {row_idx}: |C| blow-up"
                )

        # At non-degenerate psi (>= 0.1), boundary rows should not have
        # strongly negative C at resolved wavenumbers.  The degenerate
        # psi=0 point collocates the wall with a grid point and some rows
        # naturally produce negative C there.
        resolved = xi < np.pi / 2
        interior = interior_group_velocity(p=E2_1.p, nu=1, xi_array=xi)
        C_int = interior.group_velocity
        for pv, profs in result.profiles.items():
            if pv < 0.1:
                continue
            for row_idx, prof in profs.items():
                C_res = prof.group_velocity[resolved]
                # No parasitic sign reversal: boundary C > 0 where interior C < 0
                reversal = (C_res > 0) & (C_int[resolved] < 0)
                assert not np.any(reversal), (
                    f"psi={pv}, row {row_idx}: parasitic sign reversal "
                    f"at resolved xi"
                )

    def test_e2_1_no_cfl_penalty(self):
        """TEMO cut-cell stencil does not dramatically increase max|omega(xi)|.

        The TEMO construction avoids the CFL stiffness penalty: the maximum
        |omega| = max|Im(kappa*(xi))| should not blow up as psi -> 0.
        We verify that the ratio max|omega(psi)| / max|omega(psi=1)| stays
        bounded (< 10x) across all psi values.
        """
        from stencil_gen.temo import E2_1

        xi = np.linspace(0.01, np.pi, self.N_XI)
        psi_values = np.linspace(0.0, 1.0, 11)

        result = psi_sweep_group_velocity(
            E2_1, psi_values, alpha_values={}, xi_array=xi,
        )

        # Compute max|omega| at psi=1 as reference
        ref_profs = result.profiles[1.0]
        ref_max_omega = max(
            float(np.max(np.abs(np.imag(prof.kappa_star))))
            for prof in ref_profs.values()
        )

        # At each psi, max|omega| should not blow up
        for pv, profs in result.profiles.items():
            psi_max_omega = max(
                float(np.max(np.abs(np.imag(prof.kappa_star))))
                for prof in profs.values()
            )
            ratio = psi_max_omega / max(ref_max_omega, 1e-15)
            assert ratio < 10.0, (
                f"psi={pv}: max|omega| = {psi_max_omega:.4f} is "
                f"{ratio:.1f}x the psi=1 reference ({ref_max_omega:.4f}), "
                f"indicating CFL penalty"
            )

    def test_e4_1_psi_sweep(self):
        """Sweep psi in [0, 1] for E4_1 (stricter scheme).

        E4_1 has tighter constraints (only 2 stable schemes in the paper).
        Verify the psi sweep completes without blow-up and profiles are bounded.
        """
        from stencil_gen.temo import E4_1

        xi = np.linspace(0.01, np.pi, self.N_XI)
        psi_values = np.linspace(0.0, 1.0, 11)

        result = psi_sweep_group_velocity(
            E4_1, psi_values, alpha_values={}, xi_array=xi,
        )

        assert isinstance(result, PsiSweepResult)
        assert len(result.profiles) == len(psi_values)

        # All profiles should be finite and bounded
        for pv, profs in result.profiles.items():
            for row_idx, prof in profs.items():
                assert np.all(np.isfinite(prof.group_velocity)), (
                    f"psi={pv}, row {row_idx}: non-finite GV"
                )
                # E4_1 has a wider stencil (7 points) with non-uniform
                # offsets, so |C| can be larger at high xi than for E2.
                assert np.max(np.abs(prof.group_velocity)) < 500, (
                    f"psi={pv}, row {row_idx}: |C| blow-up, "
                    f"max={np.max(np.abs(prof.group_velocity)):.2e}"
                )


class TestCutCellGVvsEigenvalue:
    """Comparison of GV diagnostic with eigenvalue analysis (35.3a).

    Verifies that the per-stencil group velocity diagnostic agrees with
    full-operator eigenvalue stability for cut-cell configurations, and
    demonstrates the O(1)-vs-O(N^3) cost advantage of the GV approach.
    """

    N_XI = 500

    @staticmethod
    def _build_cut_cell_diff_matrix(cc_result, psi_sym, psi_val, alpha_values,
                                    scheme_params, n):
        """Build N×N diff matrix with cut-cell left boundary.

        Left boundary rows use the Dirichlet cut-cell stencil evaluated at
        *psi_val*.  Interior uses the standard centered stencil.  Right
        boundary uses the uniform RBF/tension stencil (sigma=10, stable).

        Parameters
        ----------
        cc_result : CutCellResult
            Pre-derived symbolic cut-cell stencil.
        psi_sym : Symbol
            SymPy psi symbol.
        psi_val : float
            Numeric psi value.
        alpha_values : dict
            Alpha symbol → value mapping.
        scheme_params : SchemeParams
            Scheme parameters (for p, q, nu, nextra).
        n : int
            Grid size.

        Returns
        -------
        np.ndarray
            N×N differentiation matrix.
        """
        from stencil_gen.interior import derive_interior, full_gamma_array
        from stencil_gen.phs import uniform_boundary_weights_rbf

        dims = cc_result.dims
        r, t, T = dims.r, dims.t, dims.T
        p, nu = scheme_params.p, scheme_params.nu

        subs = {psi_sym: psi_val, **alpha_values}

        # Interior stencil
        interior_coeffs = derive_interior(0, p, nu)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        D = np.zeros((n, n))

        # Left boundary: cut-cell Dirichlet stencil.
        # dirichlet has (R-1) = r rows, T columns.
        # Column 0 is the wall (u_wall = 0 for Dirichlet), columns 1..T-1
        # are grid points 0..T-2 = 0..t-1.
        F_dir = cc_result.dirichlet
        for i in range(F_dir.rows):
            for j in range(1, T):
                D[i, j - 1] = float(F_dir[i, j].xreplace(subs))

        # Interior rows: centered 2p+1 stencil
        for i in range(r, n - r):
            for k_idx, j in enumerate(range(i - p, i + p + 1)):
                D[i, j] = interior_w[k_idx]

        # Right boundary: reflected uniform stencil (sigma=0 tension =
        # polynomial-only, matching the TEMO polynomial design).
        sign = (-1.0) ** nu
        for i in range(r):
            w = uniform_boundary_weights_rbf(
                i, t, nu, scheme_params.q, 0.0, kernel="tension",
            )
            row = n - 1 - i
            for j in range(t):
                col = n - 1 - j
                D[row, col] = sign * float(w[j])

        return D

    def test_gv_predicts_eigenvalue_stability(self):
        """GV diagnostic and eigenvalue stability agree for E2_1 cut-cell.

        For E2_1 at psi = 0.1, 0.3, 0.5, 0.7, 1.0:
        - Compute GV profiles via cut_cell_group_velocity.
        - Build the N×N diff matrix and compute eigenvalues of -D.
        - Check: if no parasitic sign reversal in GV → eigenvalues stable
          (Re(lambda) <= small positive tolerance).
        """
        from sympy import Symbol

        from stencil_gen.temo import E2_1, derive_cut_cell_scheme

        psi_sym = Symbol("psi")
        cc = derive_cut_cell_scheme(E2_1, psi_sym)
        alpha_vals = {s: 0 for s in cc.alpha_symbols}

        xi = np.linspace(0.01, np.pi, self.N_XI)
        n = 40
        psi_test = [0.1, 0.3, 0.5, 0.7, 1.0]

        # Interior GV for sign-reversal detection
        interior = interior_group_velocity(p=E2_1.p, nu=1, xi_array=xi)
        C_int = interior.group_velocity

        for pv in psi_test:
            # --- Group velocity diagnostic ---
            profiles = cut_cell_group_velocity(
                cc, psi_sym, pv, alpha_vals, xi, order=E2_1.q,
            )
            # Check for parasitic sign reversal at resolved wavenumbers
            resolved = xi < np.pi / 2
            gv_has_reversal = False
            for row_idx, prof in profiles.items():
                C = prof.group_velocity
                reversal = (C[resolved] > 0) & (C_int[resolved] < 0)
                if np.any(reversal):
                    gv_has_reversal = True
                    break

            # --- Eigenvalue stability ---
            D = self._build_cut_cell_diff_matrix(
                cc, psi_sym, pv, alpha_vals, E2_1, n,
            )
            eigenvalues = np.linalg.eigvals(-D)
            max_real = float(np.max(eigenvalues.real))

            # Consistency: no GV reversal → eigenvalues stable.
            # Tolerance 1e-4: the polynomial-only right boundary produces
            # O(1e-6) positive Re at finite N — these vanish as N → ∞ and
            # are not related to the cut-cell left boundary under test.
            if not gv_has_reversal:
                assert max_real < 1e-4, (
                    f"psi={pv}: GV says no parasitic reversal but "
                    f"eigenvalue Re(lambda)_max = {max_real:.2e} > 0 "
                    f"(unstable)"
                )
            # Note: GV reversal does NOT guarantee instability (it's a
            # necessary condition from GKS theory, not sufficient), so we
            # don't assert the converse.

    def test_gv_cost_vs_eigenvalue_cost(self):
        """GV analysis is O(1) per stencil; eigenvalue analysis is O(N^3).

        Times both analyses at N=50, 100, 200 (eigenvalue) versus the GV
        computation (which is independent of N).  Verifies that eigenvalue
        cost grows super-linearly while GV cost stays constant.
        """
        import time

        from sympy import Symbol

        from stencil_gen.temo import E2_1, derive_cut_cell_scheme

        psi_sym = Symbol("psi")
        cc = derive_cut_cell_scheme(E2_1, psi_sym)
        alpha_vals = {s: 0 for s in cc.alpha_symbols}

        xi = np.linspace(0.01, np.pi, self.N_XI)
        psi_val = 0.5

        # Time GV computation (independent of N)
        t0 = time.perf_counter()
        for _ in range(3):
            cut_cell_group_velocity(
                cc, psi_sym, psi_val, alpha_vals, xi, order=E2_1.q,
            )
        gv_time = (time.perf_counter() - t0) / 3

        # Time eigenvalue computation at several N values
        eig_times = {}
        for n in [50, 100, 200]:
            t0 = time.perf_counter()
            D = self._build_cut_cell_diff_matrix(
                cc, psi_sym, psi_val, alpha_vals, E2_1, n,
            )
            np.linalg.eigvals(-D)
            eig_times[n] = time.perf_counter() - t0

        # Print comparison table
        print(f"\n{'Method':<20} {'N':<6} {'Time (ms)':>10}")
        print("-" * 38)
        print(f"{'GV (per stencil)':<20} {'—':<6} {gv_time * 1000:>10.2f}")
        for n, t in eig_times.items():
            print(f"{'Eigenvalue':<20} {n:<6} {t * 1000:>10.2f}")

        # Eigenvalue cost should grow super-linearly (at least N^2)
        t50, t200 = eig_times[50], eig_times[200]
        ratio = t200 / max(t50, 1e-10)
        # (200/50)^2 = 16, but overhead means we only require > 4x
        assert ratio > 4.0, (
            f"Eigenvalue cost ratio t(200)/t(50) = {ratio:.1f}, "
            f"expected > 4 for super-linear scaling"
        )


class Test2DGroupVelocity:
    """2D tensor-product group velocity tests (36.1)."""

    N_XI = 500

    @staticmethod
    def _e2_kappa_star(xi: np.ndarray) -> np.ndarray:
        """E2 (2nd-order central) modified wavenumber: kappa* = i*sin(xi)."""
        weights = [-0.5, 0.0, 0.5]
        nodes = [-1, 0, 1]
        return modified_wavenumber(weights, 0, nodes, xi)

    def test_2d_basic(self):
        """group_velocity_2d returns correct result structure and values for E2."""
        xi = np.linspace(0.01, np.pi - 0.01, self.N_XI)
        eta = np.linspace(0.01, np.pi - 0.01, self.N_XI)

        kx = self._e2_kappa_star(xi)
        ky = self._e2_kappa_star(eta)

        result = group_velocity_2d(kx, ky, xi, eta, a=1.0, b=1.0)

        # Check return type and field shapes
        assert isinstance(result, GroupVelocity2DResult)
        assert result.C_x.shape == (len(xi), len(eta))
        assert result.C_y.shape == (len(xi), len(eta))
        assert result.speed.shape == (len(xi), len(eta))
        assert result.angle.shape == (len(xi), len(eta))
        assert result.angle_error.shape == (len(xi), len(eta))

        # For E2, Im(kappa*) = sin(xi), so C_1d = cos(xi).
        # C_x should equal cos(xi) broadcast over eta.
        C_x_expected = np.cos(xi)
        np.testing.assert_allclose(
            result.C_x[:, len(eta) // 2], C_x_expected, atol=1e-3,
            err_msg="C_x should match cos(xi) for E2 stencil",
        )

        # C_y should equal cos(eta) broadcast over xi.
        C_y_expected = np.cos(eta)
        np.testing.assert_allclose(
            result.C_y[len(xi) // 2, :], C_y_expected, atol=1e-3,
            err_msg="C_y should match cos(eta) for E2 stencil",
        )

        # Speed at (xi, eta) = sqrt(cos^2(xi) + cos^2(eta))
        xi_2d, eta_2d = np.meshgrid(xi, eta, indexing="ij")
        speed_expected = np.sqrt(np.cos(xi_2d)**2 + np.cos(eta_2d)**2)
        np.testing.assert_allclose(
            result.speed, speed_expected, atol=1e-3,
            err_msg="Speed should be sqrt(C_x^2 + C_y^2)",
        )

    def test_2d_wave_speed_scaling(self):
        """group_velocity_2d correctly scales by wave speed coefficients a, b."""
        xi = np.linspace(0.1, 2.0, 200)
        eta = np.linspace(0.1, 2.0, 200)

        kx = self._e2_kappa_star(xi)
        ky = self._e2_kappa_star(eta)

        a, b = 2.0, 0.5
        result = group_velocity_2d(kx, ky, xi, eta, a=a, b=b)

        # C_x should be a*cos(xi), C_y should be b*cos(eta).
        # Exclude endpoints where np.gradient has reduced accuracy.
        s = slice(1, -1)
        np.testing.assert_allclose(
            result.C_x[s, 0], a * np.cos(xi[s]), atol=1e-3,
            err_msg="C_x should scale with wave speed a",
        )
        np.testing.assert_allclose(
            result.C_y[0, s], b * np.cos(eta[s]), atol=1e-3,
            err_msg="C_y should scale with wave speed b",
        )
