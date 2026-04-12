"""Tests for the rigorous GKS Kreiss determinant stability module."""

import numpy as np
import pytest

from stencil_gen.gks_kreiss import (
    KreissResult,
    DefectiveKappaError,
    _kappa_roots_from_poly,
    kappa_roots,
    kreiss_matrix,
    min_singular_value,
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


class TestKreissMatrix:
    """Tests for kreiss_matrix and min_singular_value (41.3c)."""

    def test_1x1_upwind(self):
        """First-order upwind with one boundary row gives a 1x1 Kreiss matrix.

        Interior: offsets=[-1, 0], weights=[-1, 1].
        At s=1: kappa = 1/2.
        Boundary row 0: weights=[2, -1], offsets=[0, 1] (grid points 0 and 1).
        M[0,0] = s*kappa^0 + 2*kappa^0 + (-1)*kappa^1
               = 1 + 2 - 0.5 = 2.5
        """
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([2.0, -1.0]), np.array([0, 1]))]
        s = 1.0

        M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)

        assert M.shape == (1, 1)
        assert abs(M[0, 0] - 2.5) < 1e-12

    def test_2x2_hand_computed(self):
        """Explicit 2x2 case with hand-computed M(s=1).

        Interior: offsets=[-2, -1, 0], weights=[0.1, 0.5, -0.4].
        At s=1, polynomial 0.6*kappa^2 + 0.5*kappa + 0.1 = 0.
        Roots: kappa_a = -1/3, kappa_b = -1/2 (both admissible).

        Boundary row 0: weights=[3, -2], offsets=[0, 1].
        Boundary row 1: weights=[1, 0.5, -0.5], offsets=[0, 1, 2].

        For kappa in {-1/3, -1/2} (order from np.roots may vary):
          M[0, l] = s*kappa^0 + 3*kappa^0 + (-2)*kappa^1
                  = 1 + 3 - 2*kappa = 4 - 2*kappa
          M[1, l] = s*kappa^1 + 1*kappa^0 + 0.5*kappa^1 + (-0.5)*kappa^2
                  = kappa + 1 + 0.5*kappa - 0.5*kappa^2
                  = 1 + 1.5*kappa - 0.5*kappa^2
        """
        interior_weights = np.array([0.1, 0.5, -0.4])
        interior_offsets = np.array([-2, -1, 0])
        boundary_rows = [
            (np.array([3.0, -2.0]), np.array([0, 1])),
            (np.array([1.0, 0.5, -0.5]), np.array([0, 1, 2])),
        ]
        s = 1.0

        M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)

        assert M.shape == (2, 2)

        # Get the admissible roots to know the column ordering
        _, admissible, _ = kappa_roots(interior_weights, interior_offsets, s)
        assert len(admissible) == 2

        # Verify each entry against the formula
        for ell, kappa in enumerate(admissible):
            # Row 0: M[0, l] = 4 - 2*kappa
            expected_0 = 4.0 - 2.0 * kappa
            assert abs(M[0, ell] - expected_0) < 1e-12, (
                f"M[0,{ell}]: got {M[0,ell]}, expected {expected_0}"
            )
            # Row 1: M[1, l] = 1 + 1.5*kappa - 0.5*kappa^2
            expected_1 = 1.0 + 1.5 * kappa - 0.5 * kappa**2
            assert abs(M[1, ell] - expected_1) < 1e-12, (
                f"M[1,{ell}]: got {M[1,ell]}, expected {expected_1}"
            )

    def test_2x2_exact_values(self):
        """Verify M entries for kappa = -1/3, -1/2 (same setup as above).

        For kappa = -1/3: M[0] = 4 + 2/3 = 14/3, M[1] = 1 - 1/2 - 1/18 = 4/9
        For kappa = -1/2: M[0] = 4 + 1 = 5,       M[1] = 1 - 3/4 - 1/8 = 1/8

        The exact matrix (up to column reordering) is:
            [[14/3, 5], [4/9, 1/8]]  or  [[5, 14/3], [1/8, 4/9]]
        """
        interior_weights = np.array([0.1, 0.5, -0.4])
        interior_offsets = np.array([-2, -1, 0])
        boundary_rows = [
            (np.array([3.0, -2.0]), np.array([0, 1])),
            (np.array([1.0, 0.5, -0.5]), np.array([0, 1, 2])),
        ]
        s = 1.0

        M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)
        _, admissible, _ = kappa_roots(interior_weights, interior_offsets, s)

        # Map kappa values to expected M columns
        for ell, kappa in enumerate(admissible):
            if abs(kappa - (-1.0 / 3.0)) < 1e-10:
                assert abs(M[0, ell] - 14.0 / 3.0) < 1e-12
                assert abs(M[1, ell] - 4.0 / 9.0) < 1e-12
            elif abs(kappa - (-1.0 / 2.0)) < 1e-10:
                assert abs(M[0, ell] - 5.0) < 1e-12
                assert abs(M[1, ell] - 1.0 / 8.0) < 1e-12
            else:
                pytest.fail(f"Unexpected admissible root: {kappa}")

    def test_shape_mismatch_raises(self):
        """ValueError when admissible root count != boundary row count."""
        # Upwind has 1 admissible root at s=1, but we provide 2 boundary rows
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([0])),
        ]
        with pytest.raises(ValueError, match="admissible roots"):
            kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s=1.0)

    def test_defective_raises(self):
        """DefectiveKappaError when admissible roots coalesce.

        Reverse-engineer a stencil whose characteristic polynomial is
        (kappa - 0.3)^2 * (kappa - 5) at s = 0.4, giving a double admissible
        root at kappa = 0.3.

        For offsets [-2, -1, 0, 1] with L_left=2, shifted=[0,1,2,3]:
          Q(kappa) = w_{-2} + w_{-1}*k + (w_0 + s)*k^2 + w_1*k^3
        Target:    -0.45   + 3.09*k    - 5.6*k^2       + k^3
        So w_{-2}=-0.45, w_{-1}=3.09, w_0=-6.0 (since w_0+s=-5.6), w_1=1.0.
        """
        interior_weights = np.array([-0.45, 3.09, -6.0, 1.0])
        interior_offsets = np.array([-2, -1, 0, 1])
        s = 0.4

        # Confirm kappa_roots detects defective roots
        _, admissible, is_defective = kappa_roots(
            interior_weights, interior_offsets, s
        )
        assert len(admissible) == 2
        assert is_defective is True

        # kreiss_matrix must raise DefectiveKappaError
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([0])),
        ]
        with pytest.raises(DefectiveKappaError):
            kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)

    def test_min_singular_value_1x1(self):
        """min_singular_value returns |M[0,0]| for a 1x1 matrix."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([2.0, -1.0]), np.array([0, 1]))]
        s = 1.0

        sv = min_singular_value(interior_weights, interior_offsets, boundary_rows, s)
        assert abs(sv - 2.5) < 1e-12

    def test_min_singular_value_2x2(self):
        """min_singular_value matches numpy SVD on the 2x2 hand-computed M."""
        interior_weights = np.array([0.1, 0.5, -0.4])
        interior_offsets = np.array([-2, -1, 0])
        boundary_rows = [
            (np.array([3.0, -2.0]), np.array([0, 1])),
            (np.array([1.0, 0.5, -0.5]), np.array([0, 1, 2])),
        ]
        s = 1.0

        sv = min_singular_value(interior_weights, interior_offsets, boundary_rows, s)

        # Cross-check: build M manually and compute SVD
        M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)
        expected_sv = float(np.linalg.svd(M, compute_uv=False)[-1])
        assert abs(sv - expected_sv) < 1e-12

    def test_min_singular_value_shape_mismatch_returns_inf(self):
        """min_singular_value returns inf when root count != boundary row count."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([0])),
        ]
        sv = min_singular_value(interior_weights, interior_offsets, boundary_rows, s=1.0)
        assert sv == np.inf

    def test_min_singular_value_defective_returns_inf(self):
        """min_singular_value returns inf on the DefectiveKappaError path.

        Uses the same engineered stencil as test_defective_raises:
        Q(kappa) = (kappa - 0.3)^2 * (kappa - 5) at s = 0.4.
        """
        interior_weights = np.array([-0.45, 3.09, -6.0, 1.0])
        interior_offsets = np.array([-2, -1, 0, 1])
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([0])),
        ]
        sv = min_singular_value(
            interior_weights, interior_offsets, boundary_rows, s=0.4
        )
        assert sv == np.inf
