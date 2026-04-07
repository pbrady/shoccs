"""Tests for group velocity analysis module."""

import numpy as np
import pytest

from stencil_gen.group_velocity import (
    GroupVelocityProfile,
    boundary_group_velocity,
    group_velocity,
    group_velocity_error,
    group_velocity_exact,
    interior_group_velocity,
    modified_wavenumber,
    phase_velocity,
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
