"""Tests for the TEMO cut-cell stencil extension module."""

import pytest
from sympy import Matrix, Rational, Symbol, simplify

from stencil_gen.temo import (
    Dimensions,
    SchemeParams,
    UniformResult,
    compute_dimensions,
    derive_e2_uniform_boundary,
    E2_1,
    E2_2,
    E4_1,
    E4_2,
)


class TestDimensions:
    """Tests for compute_dimensions and SchemeParams.dims()."""

    def test_e2_1_dimensions(self):
        """E2_1: p=1, q=1, nextra=1, nu=1 -> r=3, t=4, R=4, T=5, X=0."""
        dims = E2_1.dims()
        assert dims == Dimensions(r=3, t=4, R=4, T=5, X=0)

    def test_e2_2_dimensions(self):
        """E2_2: p=1, q=1, nextra=0, nu=2 -> r=2, t=3, R=2, T=4, X=2."""
        dims = E2_2.dims()
        assert dims == Dimensions(r=2, t=3, R=2, T=4, X=2)

    def test_e2_1_matches_cpp(self):
        """E2_1.cpp has P=1, R=4, T=5, X=0."""
        dims = E2_1.dims()
        assert dims.R == 4
        assert dims.T == 5
        assert dims.X == 0
        assert E2_1.p == 1  # P in C++

    def test_e2_2_matches_cpp(self):
        """E2_2.cpp has P=1, R=2, T=4, X=2."""
        dims = E2_2.dims()
        assert dims.R == 2
        assert dims.T == 4
        assert dims.X == 2
        assert E2_2.p == 1  # P in C++

    def test_e2_1_uniform_dimensions(self):
        """E2_1 uniform boundary: 3 rows x 4 columns."""
        dims = E2_1.dims()
        assert dims.r == 3
        assert dims.t == 4

    def test_e2_2_uniform_dimensions(self):
        """E2_2 uniform boundary: 2 rows x 3 columns (r_eff=1 after -1)."""
        dims = E2_2.dims()
        assert dims.r == 2
        assert dims.t == 3

    def test_compute_dimensions_directly(self):
        """compute_dimensions matches SchemeParams.dims()."""
        dims = compute_dimensions(p=1, q=1, s=0, nextra=1, nu=1)
        assert dims == E2_1.dims()

    def test_invalid_nu_raises(self):
        """Unsupported derivative order raises ValueError."""
        with pytest.raises(ValueError, match="nu=3"):
            compute_dimensions(p=1, q=1, s=0, nextra=0, nu=3)

    def test_scheme_params_frozen(self):
        """SchemeParams is immutable."""
        with pytest.raises(AttributeError):
            E2_1.p = 2  # type: ignore[misc]

    def test_first_derivative_no_neumann(self):
        """1st derivative stencils have X=0 (no Neumann rows)."""
        assert E2_1.dims().X == 0
        assert E4_1.dims().X == 0

    def test_second_derivative_has_neumann(self):
        """2nd derivative stencils have X=R (Neumann rows)."""
        dims_e2_2 = E2_2.dims()
        assert dims_e2_2.X == dims_e2_2.R

        dims_e4_2 = E4_2.dims()
        assert dims_e4_2.X == dims_e4_2.R


class TestUniformBoundary:
    """Tests for derive_e2_uniform_boundary (20.5b)."""

    def test_e2_1_shape_and_free_symbols(self):
        """E2_1 B_u has shape (3, 4) and 4 free alpha symbols."""
        result = derive_e2_uniform_boundary(nu=1)
        assert result.B_u.shape == (3, 4)
        assert len(result.alpha_symbols) == 4

    def test_e2_2_fully_determined(self):
        """E2_2 B_u = [[1, -2, 1]] with no free symbols."""
        result = derive_e2_uniform_boundary(nu=2)
        assert result.B_u == Matrix([[1, -2, 1]])
        assert result.alpha_symbols == []

    def test_e2_1_interior_stencil(self):
        """E2_1 interior stencil is [-1/2, 0, 1/2]."""
        result = derive_e2_uniform_boundary(nu=1)
        assert result.interior == [Rational(-1, 2), Rational(0), Rational(1, 2)]

    def test_e2_2_interior_stencil(self):
        """E2_2 interior stencil is [1, -2, 1]."""
        result = derive_e2_uniform_boundary(nu=2)
        assert result.interior == [Rational(1), Rational(-2), Rational(1)]

    def test_e2_1_conservation(self):
        """E2_1: sum_i B_u[i, j] = 0 for interior columns j=2,3."""
        result = derive_e2_uniform_boundary(nu=1)
        B_u = result.B_u
        r_eff = B_u.rows
        for j in [2, 3]:
            col_sum = sum(B_u[i, j] for i in range(r_eff))
            assert simplify(col_sum) == 0, f"Conservation failed for column {j}"

    def test_e2_1_custom_alpha_symbols(self):
        """E2_1 accepts user-supplied alpha symbols."""
        syms = [Symbol(f"a{k}") for k in range(4)]
        result = derive_e2_uniform_boundary(nu=1, alpha_symbols=syms)
        assert result.alpha_symbols == syms
        # All entries should involve only these symbols
        free = result.B_u.free_symbols
        assert free <= set(syms)

    def test_e2_1_wrong_alpha_count_raises(self):
        """E2_1 with wrong number of alpha symbols raises ValueError."""
        with pytest.raises(ValueError, match="4 alpha symbols"):
            derive_e2_uniform_boundary(nu=1, alpha_symbols=[Symbol("a")])

    def test_e2_1_taylor_accuracy_per_row(self):
        """Each row of B_u satisfies the Taylor system (q+1=2 equations)."""
        result = derive_e2_uniform_boundary(nu=1)
        B_u = result.B_u
        t = B_u.cols
        for i in range(B_u.rows):
            row = B_u.row(i)
            # k=0: sum of row = 0 (zeroth moment for 1st derivative)
            assert simplify(sum(row)) == 0
            # k=1: sum_j alpha_j * (j - i) = 1 (first moment)
            moment1 = sum(row[j] * (j - i) for j in range(t))
            assert simplify(moment1) == 1

    def test_e2_2_taylor_accuracy(self):
        """E2_2 single row satisfies max(q+1, nu+1)=3 Taylor equations."""
        result = derive_e2_uniform_boundary(nu=2)
        row = result.B_u.row(0)
        t = result.B_u.cols
        # k=0: sum = 0
        assert sum(row) == 0
        # k=1: sum_j alpha_j * (j - 0) = 0
        assert sum(row[j] * j for j in range(t)) == 0
        # k=2: sum_j alpha_j * j^2 / 2 = 1
        assert sum(row[j] * Rational(j**2, 2) for j in range(t)) == 1

    def test_e2_1_p_q_nu(self):
        """UniformResult carries correct scheme parameters."""
        result = derive_e2_uniform_boundary(nu=1)
        assert result.p == 1
        assert result.q == 1
        assert result.nu == 1

    def test_e2_2_p_q_nu(self):
        """UniformResult carries correct scheme parameters."""
        result = derive_e2_uniform_boundary(nu=2)
        assert result.p == 1
        assert result.q == 1
        assert result.nu == 2

    def test_invalid_nu_raises(self):
        """Unsupported nu raises ValueError."""
        with pytest.raises(ValueError, match="nu=3"):
            derive_e2_uniform_boundary(nu=3)
