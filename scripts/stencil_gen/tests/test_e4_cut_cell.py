"""Tests for E4_1 cut-cell stencil derivation (21.1b onwards)."""

import pytest
from sympy import Matrix, Rational, S, Symbol, cancel, simplify

from stencil_gen.temo import (
    E4_1,
    SchemeParams,
    UniformResult,
    derive_uniform_boundary_for_temo,
)


class TestE4UniformBoundary:
    """Tests for derive_uniform_boundary_for_temo with E4_1 (21.1b)."""

    @pytest.fixture
    def e4_result(self):
        """Compute E4_1 uniform boundary once for the test class."""
        return derive_uniform_boundary_for_temo(E4_1)

    def test_shape(self, e4_result):
        """E4_1 B_u has shape (3, 6) — r_eff=3 rows, t=6 columns."""
        assert e4_result.B_u.shape == (3, 6)

    def test_four_alpha_symbols(self, e4_result):
        """E4_1 has exactly 4 free alpha symbols."""
        assert len(e4_result.alpha_symbols) == 4
        # Verify they are named alpha_0..alpha_3
        for k, sym in enumerate(e4_result.alpha_symbols):
            assert sym.name == f"alpha_{k}"

    def test_zero_constraints(self, e4_result):
        """B_u[0, 5] == 0 and B_u[1, 5] == 0 (zero-constrained entries)."""
        assert e4_result.B_u[0, 5] == 0
        assert e4_result.B_u[1, 5] == 0

    def test_last_row_free_alphas(self, e4_result):
        """B_u[2, 4] and B_u[2, 5] contain alpha_2, alpha_3."""
        alpha_2 = e4_result.alpha_symbols[2]
        alpha_3 = e4_result.alpha_symbols[3]
        # These entries should involve alpha_2 and alpha_3 respectively
        assert alpha_2 in e4_result.B_u[2, 4].free_symbols
        assert alpha_3 in e4_result.B_u[2, 5].free_symbols

    def test_interior_coefficients(self, e4_result):
        """Interior coefficients are [1/12, -2/3, 0, 2/3, -1/12]."""
        expected = [Rational(1, 12), Rational(-2, 3), S.Zero,
                    Rational(2, 3), Rational(-1, 12)]
        assert e4_result.interior == expected

    def test_scheme_metadata(self, e4_result):
        """Result carries correct p, q, nu."""
        assert e4_result.p == 2
        assert e4_result.q == 3
        assert e4_result.nu == 1

    def test_rows_0_1_match_e4u_1(self, e4_result):
        """First 5 columns of rows 0, 1 match E4u_1.cpp's nbs_floating coefficients.

        E4u_1.cpp row 0 (c[0..4], before /h):
            c[0] = (6*a0 - 11)/6
            c[1] = 3 - 4*a0
            c[2] = (12*a0 - 3)/2
            c[3] = -(12*a0 - 1)/3
            c[4] = a0

        E4u_1.cpp row 1 (c[5..9], before /h):
            c[5] = (3*a1 - 1)/3
            c[6] = -(8*a1 + 1)/2
            c[7] = 6*a1 + 1
            c[8] = -(24*a1 + 1)/6
            c[9] = a1
        """
        B_u = e4_result.B_u
        a0 = e4_result.alpha_symbols[0]
        a1 = e4_result.alpha_symbols[1]

        # E4u_1 row 0 expected (5 columns)
        row0_expected = [
            (6 * a0 - 11) / S(6),
            3 - 4 * a0,
            (12 * a0 - 3) / S(2),
            -(12 * a0 - 1) / S(3),
            a0,
        ]

        # E4u_1 row 1 expected (5 columns)
        row1_expected = [
            (3 * a1 - 1) / S(3),
            -(8 * a1 + 1) / S(2),
            6 * a1 + 1,
            -(24 * a1 + 1) / S(6),
            a1,
        ]

        for j in range(5):
            diff = cancel(B_u[0, j] - row0_expected[j])
            assert diff == 0, (
                f"Row 0, col {j}: B_u={B_u[0,j]}, expected={row0_expected[j]}"
            )

        for j in range(5):
            diff = cancel(B_u[1, j] - row1_expected[j])
            assert diff == 0, (
                f"Row 1, col {j}: B_u={B_u[1,j]}, expected={row1_expected[j]}"
            )

    def test_taylor_accuracy(self, e4_result):
        """Each row satisfies Taylor matching for q+1=4 equations (polynomials up to degree 3).

        For first derivative (nu=1), row i should exactly differentiate
        monomials x^m for m = 0, 1, ..., q=3:
            sum_j c_j * (j - i)^m = delta_{m,1} * m!  (= delta_{m,1})
        """
        B_u = e4_result.B_u
        t = B_u.cols
        q = e4_result.q  # q=3

        for i in range(B_u.rows):
            row = B_u.row(i)
            for m in range(q + 1):
                moment = sum(row[j] * (j - i) ** m for j in range(t))
                if m == 1:
                    expected = 1
                else:
                    expected = 0
                assert simplify(moment - expected) == 0, (
                    f"Row {i}, moment {m}: got {simplify(moment)}, expected {expected}"
                )

    def test_no_conservation_constraint(self, e4_result):
        """E4_1 (nextra=0) has no column-sum conservation constraint.

        Column sums need NOT be zero — this confirms nextra=0 path is different
        from E2_1's nextra=1 path.
        """
        B_u = e4_result.B_u
        # Just verify that B_u doesn't have phi symbols (conservation resolved)
        free = B_u.free_symbols
        assert all("phi" not in str(s) for s in free), (
            f"Unexpected phi symbols: {free}"
        )

    def test_only_alpha_symbols_in_B_u(self, e4_result):
        """B_u contains only the expected alpha symbols, nothing else."""
        expected_syms = set(e4_result.alpha_symbols)
        actual_syms = e4_result.B_u.free_symbols
        assert actual_syms <= expected_syms, (
            f"Unexpected symbols in B_u: {actual_syms - expected_syms}"
        )

    def test_custom_alpha_symbols(self):
        """derive_uniform_boundary_for_temo(E4_1) accepts custom alpha names."""
        syms = [Symbol(f"a{k}") for k in range(4)]
        result = derive_uniform_boundary_for_temo(E4_1, alpha_symbols=syms)
        assert result.alpha_symbols == syms
        free = result.B_u.free_symbols
        assert free <= set(syms)

    def test_wrong_alpha_count_raises(self):
        """Wrong number of alpha symbols raises ValueError."""
        with pytest.raises(ValueError, match="alpha symbols"):
            derive_uniform_boundary_for_temo(E4_1, alpha_symbols=[Symbol("a")])
