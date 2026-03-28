"""Tests for the TEMO cut-cell stencil extension module."""

import pytest

from stencil_gen.temo import (
    Dimensions,
    SchemeParams,
    compute_dimensions,
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
