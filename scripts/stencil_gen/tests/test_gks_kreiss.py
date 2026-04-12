"""Tests for the rigorous GKS Kreiss determinant stability module."""

import numpy as np
import pytest

from stencil_gen.gks_kreiss import (
    KreissResult,
    DefectiveKappaError,
    _kappa_roots_from_poly,
    kappa_roots,
)


class TestKappaRoots:
    """Tests for kappa_roots (41.3b)."""

    def test_first_order_upwind(self):
        """First-order upwind: u_t = -(u_n - u_{n-1}) has one admissible root.

        Interior stencil: weights=[-1, 1], offsets=[-1, 0].
        Characteristic eqn: s + (-1)*kappa^{-1} + 1*kappa^0 = 0
        Multiply by kappa (L_left=1): -1 + kappa + s*kappa = 0
        => kappa*(1 + s) = 1 => kappa = 1/(1+s)

        At s=1: kappa = 1/2 (admissible, |kappa| < 1).
        The polynomial is degree 1 so there's exactly one root total.
        """
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        s = 1.0

        all_roots, admissible, is_defective = kappa_roots(
            interior_weights, interior_offsets, s
        )

        # Degree-1 polynomial => exactly one root
        assert len(all_roots) == 1
        # The root should be 1/(1+s) = 0.5
        assert abs(all_roots[0] - 0.5) < 1e-12

        # One admissible root
        assert len(admissible) == 1
        assert abs(admissible[0] - 0.5) < 1e-12
        assert abs(admissible[0]) < 1.0

        # Not defective (only one root)
        assert is_defective is False

    def test_second_order_centered(self):
        """Second-order centered: u_t = -(u_{n+1} - u_{n-1})/2.

        Interior stencil: weights=[-0.5, 0.5], offsets=[-1, 1].
        Char eqn: s + (-0.5)*kappa^{-1} + 0.5*kappa = 0
        Multiply by kappa: -0.5 + s*kappa + 0.5*kappa^2 = 0

        Degree-2 polynomial => 2 roots total.
        """
        interior_weights = np.array([-0.5, 0.5])
        interior_offsets = np.array([-1, 1])
        s = 0.5 + 0.3j

        all_roots, admissible, is_defective = kappa_roots(
            interior_weights, interior_offsets, s
        )

        # Degree-2 polynomial
        assert len(all_roots) == 2

        # Verify all roots satisfy the polynomial
        # Q(kappa) = -0.5 + s*kappa + 0.5*kappa^2
        for k in all_roots:
            val = -0.5 + s * k + 0.5 * k**2
            assert abs(val) < 1e-10, f"Root {k} doesn't satisfy polynomial: Q={val}"

    def test_defective_kappa_detected_from_poly(self):
        """Deliberately-coalescing case: double root at kappa=0.5.

        Construct polynomial with known double root at 0.5 and a root at 2.0:
        Q(kappa) = (kappa - 0.5)^2 * (kappa - 2.0)
        """
        # np.poly([0.5, 0.5, 2.0]) gives coefficients [1, -3, 2.25, -0.5]
        poly_coeffs = np.poly([0.5, 0.5, 2.0])

        all_roots, admissible, is_defective = _kappa_roots_from_poly(poly_coeffs)

        # Should have 3 roots total
        assert len(all_roots) == 3

        # Two admissible roots (both near 0.5), one outside (near 2.0)
        assert len(admissible) == 2
        for k in admissible:
            assert abs(k - 0.5) < 1e-6

        # Defective because the two admissible roots are very close
        assert is_defective is True

    def test_no_admissible_roots(self):
        """All roots outside the unit disk."""
        # Q(kappa) = (kappa - 2)(kappa - 3) = kappa^2 - 5*kappa + 6
        poly_coeffs = np.poly([2.0, 3.0])

        all_roots, admissible, is_defective = _kappa_roots_from_poly(poly_coeffs)

        assert len(all_roots) == 2
        assert len(admissible) == 0
        assert is_defective is False

    def test_upwind_various_s_values(self):
        """For upwind, kappa = 1/(1+s); admissible iff Re(s) > 0 or |1+s|>1."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])

        # s=0.1: kappa = 1/1.1 ≈ 0.909, still admissible
        _, adm, _ = kappa_roots(interior_weights, interior_offsets, 0.1)
        assert len(adm) == 1
        assert abs(adm[0] - 1.0 / 1.1) < 1e-12

        # s with large Re: kappa small, definitely admissible
        _, adm, _ = kappa_roots(interior_weights, interior_offsets, 10.0)
        assert len(adm) == 1
        assert abs(adm[0]) < 0.1

    def test_purely_imaginary_s(self):
        """Upwind at s = 2j: kappa = 1/(1+2j) = (1-2j)/5."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        s = 2j

        all_roots, admissible, _ = kappa_roots(interior_weights, interior_offsets, s)
        expected = 1.0 / (1.0 + 2j)
        assert abs(all_roots[0] - expected) < 1e-12
        # |kappa| = 1/sqrt(5) ≈ 0.447 < 1
        assert len(admissible) == 1
