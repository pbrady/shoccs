"""Tests for group velocity analysis module."""

import numpy as np
import pytest

from stencil_gen.group_velocity import (
    GroupVelocityProfile,
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
        # The exact first derivative in Fourier space has kappa* = i*xi,
        # so C = -d(Im(kappa*))/d(xi) = -d(xi)/d(xi) = -1... wait.
        # For u_t + u_x = 0, the exact spatial operator is d/dx, with
        # kappa*(xi) = i*xi. Then omega = -kappa* = -i*xi, and
        # C = d(Re(omega))/d(xi) = 0 for the real part... Actually:
        #
        # The convention: for D1 approximating d/dx, the modified wavenumber
        # is kappa*(xi) with Im(kappa*) approximating xi.
        # Group velocity: C(xi) = d(Im(kappa*))/d(xi) (not negative).
        #
        # Actually looking at the plan: C(xi) = -d(kappa*)/d(xi) where
        # kappa* is complex. For u_t + u_x = 0, omega = -kappa*, and
        # group velocity = d(omega)/d(xi) = -d(kappa*)/d(xi).
        # For exact: kappa* = i*xi, so d(kappa*)/d(xi) = i,
        # and C = -i... that's complex.
        #
        # The plan says C(xi) = -d(Im(kappa*))/d(xi). For exact: Im(kappa*) = xi,
        # so C = -1. That contradicts "C = 1 for perfect scheme."
        #
        # Reconciliation: the sign convention depends on the equation.
        # For u_t + u_x = 0 discretized as du/dt = -D*u, where D approximates d/dx:
        #   omega = i*kappa*(xi) [if D has kappa* = i*xi for exact]
        #   Group velocity = d(omega)/d(xi) = i * d(kappa*)/d(xi)
        #
        # Let's just use the analytical formula and verify consistency.
        # For weights [-1/2, 0, 1/2] (E2 D1), nodes [-1,0,1], i_eval=0:
        #   kappa*(xi) = -sin(xi) * i  ... wait, let me compute:
        #   kappa* = (-1/2)*exp(-i*xi) + 0 + (1/2)*exp(i*xi) = i*sin(xi)
        #   So Im(kappa*) = sin(xi), and C_exact (from group_velocity_exact):
        #   C = Re(sum w_j * offset_j * exp(i*offset*xi))
        #     = Re((-1/2)(-1)exp(-i*xi) + (1/2)(1)exp(i*xi))
        #     = Re((1/2)(exp(-i*xi) + exp(i*xi))) = Re(cos(xi)) = cos(xi)
        #
        # For the exact scheme (kappa* = i*xi):
        # We need a stencil that gives kappa* = i*xi exactly. No finite stencil
        # does this, but we can test that group_velocity_exact agrees with
        # the numerical group_velocity for a known case.
        #
        # Instead, test: for the "spectral" limit, C -> 1 at low wavenumbers.
        # Use a high-order scheme (p=4, E8) and check C ≈ 1 for small xi.
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
