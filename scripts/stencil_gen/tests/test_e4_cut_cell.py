"""Tests for E4_1 cut-cell stencil derivation (21.1b onwards)."""

import pathlib

import pytest
from itertools import combinations
from sympy import (
    Integer, Matrix, Rational, S, Symbol, cancel, collect, expand,
    factor, fraction, linear_eq_to_matrix, simplify, solve,
)

from stencil_gen.codegen import (
    StencilGenSpec,
    TestCase,
    compute_test_values,
    generate_stencil_cpp,
    generate_test_cpp,
)
from stencil_gen.conservation import _interior_contribution
from stencil_gen.temo import (
    E2_1,
    E2_2,
    E4_1,
    SchemeParams,
    UniformResult,
    assemble_cut_cell_result,
    build_cut_cell_conservation_system,
    build_cut_cell_deltas,
    build_degenerate_stencil,
    compute_dimensions,
    construct_cut_cell_stencil,
    derive_cut_cell_scheme,
    derive_e2_uniform_boundary,
    derive_uniform_boundary_for_temo,
    solve_uniform_limit,
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


class TestE4TEMOConstruction:
    """Tests for E4_1 full TEMO pipeline (21.3a)."""

    @pytest.fixture(scope="class")
    def e4_temo(self):
        """Run the full E4_1 TEMO pipeline once for the test class."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
        )
        return ur, result, psi

    def test_shape(self, e4_temo):
        """E4_1 cut-cell stencil has shape (4, 7) — R=4, T=7."""
        _, result, _ = e4_temo
        assert result.matrix.shape == (4, 7)

    def test_no_betas(self, e4_temo):
        """E4_1 (nextra=0) produces no beta parameters."""
        _, result, _ = e4_temo
        assert len(result.beta_info) == 0
        assert len(result.beta_symbols) == 0

    def test_entries_in_psi_alpha(self, e4_temo):
        """All entries are rational in psi and alpha_{0..3} only."""
        _, result, _ = e4_temo
        all_syms = result.matrix.free_symbols
        expected_names = {"psi"} | {f"alpha_{k}" for k in range(4)}
        actual_names = {s.name for s in all_syms}
        assert actual_names <= expected_names, (
            f"Unexpected symbols: {actual_names - expected_names}"
        )

    def test_uniform_limit(self, e4_temo):
        """At psi=1, rows 0-2 reduce to B_u in T-frame, row 3 is interior."""
        ur, result, psi = e4_temo
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0)
        m1 = result.matrix.subs(psi, 1)
        R, T = m1.shape
        for i in range(R):
            for j in range(T):
                assert simplify(m1[i, j] - B_l_1[i, j]) == 0, (
                    f"Uniform limit mismatch at [{i},{j}]: "
                    f"{cancel(m1[i, j])} != {cancel(B_l_1[i, j])}"
                )

    def test_uniform_limit_rows_0_2_embed_Bu(self, e4_temo):
        """At psi=1, rows 0-2: wall col=0, then cols 1-6 = B_u rows 0-2."""
        ur, result, psi = e4_temo
        m1 = result.matrix.subs(psi, 1)
        B_u = ur.B_u
        for i in range(3):
            # Column 0 is the wall column
            # Columns 1..6 should match B_u[i, 0..5]
            for j in range(6):
                assert simplify(m1[i, j + 1] - B_u[i, j]) == 0, (
                    f"B_u embed mismatch at row {i}, col {j}: "
                    f"{cancel(m1[i, j + 1])} != {cancel(B_u[i, j])}"
                )

    def test_uniform_limit_row3_interior(self, e4_temo):
        """At psi=1, row 3 is the interior stencil [0, 0, 1/12, -2/3, 0, 2/3, -1/12]."""
        ur, result, psi = e4_temo
        m1 = result.matrix.subs(psi, 1)
        expected_row3 = [
            S.Zero, S.Zero,
            Rational(1, 12), Rational(-2, 3), S.Zero,
            Rational(2, 3), Rational(-1, 12),
        ]
        for j in range(7):
            assert simplify(m1[3, j] - expected_row3[j]) == 0, (
                f"Row 3 interior mismatch at col {j}: "
                f"{cancel(m1[3, j])} != {expected_row3[j]}"
            )

    def test_degenerate_limit(self, e4_temo):
        """At psi=0, matches the degenerate stencil B_d."""
        ur, result, psi = e4_temo
        m0 = result.matrix.subs(psi, 0)
        B_d = build_degenerate_stencil(ur.B_u, ur.interior, p=2, q=3, nu=1)
        R, T = m0.shape
        for i in range(R):
            for j in range(T):
                assert simplify(m0[i, j] - B_d[i, j]) == 0, (
                    f"Degenerate mismatch at [{i},{j}]: "
                    f"{cancel(m0[i, j])} != {cancel(B_d[i, j])}"
                )

    def test_taylor_accuracy_symbolic(self, e4_temo):
        """Each row satisfies Taylor accuracy (q+1=4 equations) for symbolic psi.

        For first derivative (nu=1), row i should exactly differentiate
        monomials x^m for m = 0, 1, ..., q=3:
            sum_j c_j * delta_j^m = delta_{m,1} * m!  (= delta_{m,1})
        """
        _, result, psi = e4_temo
        m = result.matrix
        R, T = m.shape
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi)
            row = [m[i, j] for j in range(T)]
            for k in range(4):  # q+1 = 4
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                if k == 1:
                    expected = 1
                else:
                    expected = 0
                assert simplify(moment - expected) == 0, (
                    f"Row {i}, moment k={k}: got {simplify(moment)}, "
                    f"expected {expected}"
                )

    def test_taylor_accuracy_at_half(self, e4_temo):
        """Taylor accuracy holds at psi=1/2 (numerical check)."""
        _, result, psi = e4_temo
        m = result.matrix.subs(psi, Rational(1, 2))
        R, T = m.shape
        psi_val = Rational(1, 2)
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi_val)
            row = [m[i, j] for j in range(T)]
            for k in range(4):
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                if k == 1:
                    expected = 1
                else:
                    expected = 0
                assert simplify(moment - expected) == 0, (
                    f"Row {i}, moment k={k} at psi=1/2: "
                    f"got {simplify(moment)}, expected {expected}"
                )


class TestE4CodeGeneration:
    """Tests for E4_1 C++ code generation (21.4b)."""

    @pytest.fixture(scope="class")
    def e4_spec(self):
        """Build the full StencilGenSpec from the TEMO pipeline."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
        )
        dims = compute_dimensions(E4_1.p, E4_1.q, E4_1.s, E4_1.nextra, E4_1.nu)
        cc = assemble_cut_cell_result(
            result.matrix, None, None, dims, ur.alpha_symbols,
        )

        # floating_coeffs: R*T = 4*7 = 28 entries, row-major from cc.floating
        floating_flat = list(cc.floating)

        # dirichlet_coeffs: R*T = 28 entries (prepend T=7 zeros for row 0)
        dirichlet_flat = [Integer(0)] * 7 + list(cc.dirichlet)

        spec = StencilGenSpec(
            name="E4_1",
            P=2,
            R=4,
            T=7,
            X=0,
            derivative_order=1,
            is_uniform=False,
            param_arrays={"alpha": 4},
            interior_coeffs=ur.interior,
            floating_coeffs=floating_flat,
            dirichlet_coeffs=dirichlet_flat,
        )
        return spec

    @pytest.fixture(scope="class")
    def e4_code(self, e4_spec):
        """Generate the E4_1 C++ code."""
        return generate_stencil_cpp(e4_spec)

    def test_struct_constants(self, e4_code):
        """Generated code has P=2, R=4, T=7, X=0."""
        assert "static constexpr int P = 2;" in e4_code
        assert "static constexpr int R = 4;" in e4_code
        assert "static constexpr int T = 7;" in e4_code
        assert "static constexpr int X = 0;" in e4_code

    def test_struct_name(self, e4_code):
        """Generated code defines struct E4_1."""
        assert "struct E4_1" in e4_code

    def test_namespace(self, e4_code):
        """Generated code uses ccs::stencils namespace."""
        assert "namespace ccs::stencils" in e4_code

    def test_alpha_array(self, e4_code):
        """Generated code has std::array<real, 4> alpha member."""
        assert "std::array<real, 4> alpha;" in e4_code

    def test_constructor(self, e4_code):
        """Generated code has span constructor."""
        assert "E4_1(std::span<const real> a)" in e4_code
        assert "copy_zero_padded(a, alpha);" in e4_code

    def test_factory(self, e4_code):
        """Generated code has make_E4_1 factory."""
        assert "make_E4_1(std::span<const real> alpha)" in e4_code
        assert "return E4_1{alpha};" in e4_code

    def test_interior_method(self, e4_code):
        """Generated code has interior() method."""
        assert "interior(real h," in e4_code

    def test_nbs_floating_method(self, e4_code):
        """Generated code has nbs_floating method with 28 coefficient assignments."""
        assert "nbs_floating(real h," in e4_code
        floating_start = e4_code.index("nbs_floating(real h,")
        dirichlet_start = e4_code.index("nbs_dirichlet(real h,")
        floating_section = e4_code[floating_start:dirichlet_start]
        # R*T = 4*7 = 28 coefficient assignments
        assert floating_section.count("c[") == 28, (
            f"Expected 28 c[] assignments in floating, got {floating_section.count('c[')}"
        )

    def test_nbs_dirichlet_method(self, e4_code):
        """Generated code has nbs_dirichlet method with 21 coefficient assignments."""
        assert "nbs_dirichlet(real h," in e4_code
        dirichlet_start = e4_code.index("nbs_dirichlet(real h,")
        neumann_start = e4_code.index("nbs_neumann")
        dirichlet_section = e4_code[dirichlet_start:neumann_start]
        # (R-1)*T = 3*7 = 21 coefficient assignments
        assert dirichlet_section.count("c[") == 21, (
            f"Expected 21 c[] assignments in dirichlet, got {dirichlet_section.count('c[')}"
        )

    def test_nbs_neumann_stub(self, e4_code):
        """Generated code has nbs_neumann stub (X=0, non-uniform)."""
        assert "nbs_neumann" in e4_code

    def test_psi_named_parameter(self, e4_code):
        """Cut-cell stencil uses named psi parameter in nbs methods."""
        # In cut-cell (non-uniform), psi is a named parameter
        floating_start = e4_code.index("nbs_floating(real h,")
        floating_sig_end = e4_code.index("{", floating_start)
        floating_sig = e4_code[floating_start:floating_sig_end]
        assert "real psi" in floating_sig

    def test_cse_temporaries(self, e4_code):
        """Generated code uses CSE temporaries for complex expressions."""
        # E4_1 has rational functions of psi and alpha — CSE should produce temporaries
        floating_start = e4_code.index("nbs_floating(real h,")
        dirichlet_start = e4_code.index("nbs_dirichlet(real h,")
        floating_section = e4_code[floating_start:dirichlet_start]
        # Check for CSE temporaries (real t... = ...)
        assert "real t" in floating_section, (
            "Expected CSE temporaries in floating method"
        )

    def test_query_methods(self, e4_code):
        """Generated code has query_max and query methods."""
        assert "query_max()" in e4_code
        assert "query(bcs::type b)" in e4_code
        assert "query_interp()" in e4_code

    def test_write_output(self, e4_spec, e4_code):
        """Write generated E4_1.cpp to output directory."""
        output_dir = pathlib.Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "E4_1.cpp"
        output_path.write_text(e4_code)
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestE4TestFileGeneration:
    """Tests for E4_1 C++ test file generation (21.4c)."""

    ALPHA_VALUES = {"alpha": [0.1, -0.05, 0.02, 0.01]}

    @pytest.fixture(scope="class")
    def e4_spec(self):
        """Build the full StencilGenSpec from the TEMO pipeline."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
        )
        dims = compute_dimensions(E4_1.p, E4_1.q, E4_1.s, E4_1.nextra, E4_1.nu)
        cc = assemble_cut_cell_result(
            result.matrix, None, None, dims, ur.alpha_symbols,
        )

        floating_flat = list(cc.floating)
        dirichlet_flat = [Integer(0)] * 7 + list(cc.dirichlet)

        return StencilGenSpec(
            name="E4_1",
            P=2,
            R=4,
            T=7,
            X=0,
            derivative_order=1,
            is_uniform=False,
            param_arrays={"alpha": 4},
            interior_coeffs=ur.interior,
            floating_coeffs=floating_flat,
            dirichlet_coeffs=dirichlet_flat,
        )

    def test_compute_floating_values(self, e4_spec):
        """compute_test_values produces 28 floating coefficients at psi=1.0."""
        values = compute_test_values(
            e4_spec.floating_coeffs,
            alpha_values=self.ALPHA_VALUES,
            h=1.0,
            psi=1.0,
        )
        assert len(values) == 28  # R*T = 4*7

    def test_compute_dirichlet_values(self, e4_spec):
        """compute_test_values produces 21 Dirichlet coefficients at psi=0.7."""
        dirichlet_emitted = e4_spec.dirichlet_coeffs[e4_spec.T:]
        values = compute_test_values(
            dirichlet_emitted,
            alpha_values=self.ALPHA_VALUES,
            h=0.5,
            psi=0.7,
        )
        assert len(values) == 21  # (R-1)*T = 3*7

    def test_floating_uniform_limit_row3_interior(self, e4_spec):
        """At psi=1.0, floating row 3 is the interior stencil [0,0,1/12,-2/3,0,2/3,-1/12]/h."""
        values = compute_test_values(
            e4_spec.floating_coeffs,
            alpha_values=self.ALPHA_VALUES,
            h=2.0,
            psi=1.0,
        )
        # Row 3 = indices 21..27
        row3 = values[21:28]
        expected = [0, 0, 1 / 24, -1 / 3, 0, 1 / 3, -1 / 24]
        for i, (got, want) in enumerate(zip(row3, expected)):
            assert abs(got - want) < 1e-12, (
                f"Row 3 col {i}: got {got}, want {want}"
            )

    def test_generate_test_file_structure(self, e4_spec):
        """Generated test file has correct Catch2 structure for E4_1."""
        floating_vals = compute_test_values(
            e4_spec.floating_coeffs,
            alpha_values=self.ALPHA_VALUES,
            h=1.0,
            psi=1.0,
        )
        cases = [
            TestCase(
                bc_type="Floating",
                h=1.0,
                psi=1.0,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=floating_vals,
            ),
        ]
        code = generate_test_cpp(e4_spec, cases)
        assert 'TEST_CASE("E4_1")' in code
        assert 'type = "E4"' in code
        assert "order = 1" in code
        assert "alpha = {0.1, -0.05, 0.02, 0.01}" in code
        assert "REQUIRE(p == 2)" in code
        assert "REQUIRE(r == 4)" in code
        assert "REQUIRE(t == 7)" in code

    def test_generate_test_file_multiple_cases(self, e4_spec):
        """Generated test file has Floating and Dirichlet test blocks."""
        dirichlet_emitted = e4_spec.dirichlet_coeffs[e4_spec.T:]
        cases = [
            TestCase(
                bc_type="Floating",
                h=2.0,
                psi=1.0,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    e4_spec.floating_coeffs,
                    alpha_values=self.ALPHA_VALUES, h=2.0, psi=1.0,
                ),
            ),
            TestCase(
                bc_type="Floating",
                h=1.0,
                psi=0.3,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    e4_spec.floating_coeffs,
                    alpha_values=self.ALPHA_VALUES, h=1.0, psi=0.3,
                ),
            ),
            TestCase(
                bc_type="Dirichlet",
                h=0.5,
                psi=0.7,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    dirichlet_emitted,
                    alpha_values=self.ALPHA_VALUES, h=0.5, psi=0.7,
                ),
            ),
        ]
        code = generate_test_cpp(e4_spec, cases)
        assert code.count("REQUIRE_THAT(c,") == 3
        assert "bcs::Floating" in code
        assert "bcs::Dirichlet" in code

    def test_write_test_output(self, e4_spec):
        """Generate and write E4_1.t.cpp to output directory."""
        dirichlet_emitted = e4_spec.dirichlet_coeffs[e4_spec.T:]
        cases = [
            TestCase(
                bc_type="Floating",
                h=2.0,
                psi=1.0,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    e4_spec.floating_coeffs,
                    alpha_values=self.ALPHA_VALUES, h=2.0, psi=1.0,
                ),
            ),
            TestCase(
                bc_type="Floating",
                h=1.0,
                psi=0.3,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    e4_spec.floating_coeffs,
                    alpha_values=self.ALPHA_VALUES, h=1.0, psi=0.3,
                ),
            ),
            TestCase(
                bc_type="Dirichlet",
                h=0.5,
                psi=0.7,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    dirichlet_emitted,
                    alpha_values=self.ALPHA_VALUES, h=0.5, psi=0.7,
                ),
            ),
        ]
        code = generate_test_cpp(e4_spec, cases)

        output_dir = pathlib.Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "E4_1.t.cpp"
        output_path.write_text(code)
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestDeriveCutCellScheme:
    """Tests for derive_cut_cell_scheme high-level pipeline (21.5a)."""

    def test_e4_1_shape(self):
        """E4_1 via derive_cut_cell_scheme has R=4, T=7 floating matrix."""
        psi = Symbol("psi")
        result = derive_cut_cell_scheme(E4_1, psi)
        assert result.floating.shape == (4, 7)
        assert result.dirichlet.shape == (3, 7)
        assert result.dims.R == 4
        assert result.dims.T == 7
        assert result.dims.X == 0

    def test_e4_1_no_neumann(self):
        """E4_1 (nu=1) has no Neumann stencil."""
        psi = Symbol("psi")
        result = derive_cut_cell_scheme(E4_1, psi)
        assert result.neumann is None
        assert result.eta is None

    def test_e4_1_alpha_count(self):
        """E4_1 has 4 free alpha symbols."""
        psi = Symbol("psi")
        result = derive_cut_cell_scheme(E4_1, psi)
        assert len(result.alpha_symbols) == 4

    def test_e4_1_matches_manual_pipeline(self):
        """derive_cut_cell_scheme(E4_1) matches the manual step-by-step pipeline."""
        psi = Symbol("psi")

        # Manual pipeline (as done in earlier tests)
        ur = derive_uniform_boundary_for_temo(E4_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
        )
        dims = compute_dimensions(E4_1.p, E4_1.q, E4_1.s, E4_1.nextra, E4_1.nu)
        manual = assemble_cut_cell_result(
            stencil.matrix, None, None, dims, ur.alpha_symbols,
        )

        # High-level pipeline
        auto = derive_cut_cell_scheme(E4_1, psi)

        # Compare floating matrices entry by entry
        R, T = auto.floating.shape
        for i in range(R):
            for j in range(T):
                assert cancel(auto.floating[i, j] - manual.floating[i, j]) == 0, (
                    f"Floating mismatch at [{i},{j}]"
                )

    def test_e4_1_taylor_accuracy(self):
        """E4_1 result satisfies Taylor accuracy at psi=1/2."""
        psi = Symbol("psi")
        result = derive_cut_cell_scheme(E4_1, psi)
        m = result.floating.subs(psi, Rational(1, 2))
        R, T = m.shape
        psi_val = Rational(1, 2)
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi_val)
            row = [m[i, j] for j in range(T)]
            for k in range(4):  # q+1 = 4
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                expected = 1 if k == 1 else 0
                assert simplify(moment - expected) == 0, (
                    f"Row {i}, moment k={k}: got {simplify(moment)}"
                )

    def test_e4_1_custom_alphas(self):
        """derive_cut_cell_scheme accepts custom alpha symbols."""
        psi = Symbol("psi")
        syms = [Symbol(f"a{k}") for k in range(4)]
        result = derive_cut_cell_scheme(E4_1, psi, alpha_symbols=syms)
        assert result.alpha_symbols == syms
        assert result.floating.free_symbols <= {psi} | set(syms)

    def test_e2_1_reproduces_existing(self):
        """derive_cut_cell_scheme(E2_1) matches manual E2_1 pipeline."""
        psi = Symbol("psi")

        # Manual E2_1 pipeline using old derive_e2_uniform_boundary
        ur = derive_e2_uniform_boundary(nu=1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=1, q=1, nu=1, nextra=1, psi=psi,
        )
        dims = compute_dimensions(1, 1, 0, 1, 1)
        manual = assemble_cut_cell_result(
            stencil.matrix, None, None, dims, ur.alpha_symbols,
        )

        # High-level pipeline
        auto = derive_cut_cell_scheme(E2_1, psi)

        # Shapes must match
        assert auto.floating.shape == manual.floating.shape
        assert auto.dirichlet.shape == manual.dirichlet.shape
        assert auto.dims == manual.dims

        # Floating matrices must match entry by entry
        R, T = auto.floating.shape
        for i in range(R):
            for j in range(T):
                assert cancel(auto.floating[i, j] - manual.floating[i, j]) == 0, (
                    f"E2_1 floating mismatch at [{i},{j}]"
                )

    def test_e2_2_reproduces_existing(self):
        """derive_cut_cell_scheme(E2_2) matches manual E2_2 pipeline."""
        psi = Symbol("psi")

        # Manual E2_2 pipeline
        ur = derive_e2_uniform_boundary(nu=2)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=1, q=1, nu=2, nextra=0, psi=psi,
        )
        from stencil_gen.temo import derive_uniform_neumann, construct_neumann_stencil
        B_uN, eta_u = derive_uniform_neumann(ur.interior, 1, 1, 2)
        neumann_main, eta = construct_neumann_stencil(
            ur.B_u, B_uN, eta_u, ur.interior, 1, 1, 2, 0, psi,
        )
        dims = compute_dimensions(1, 1, 0, 0, 2)
        manual = assemble_cut_cell_result(
            stencil.matrix, neumann_main, eta, dims, ur.alpha_symbols,
        )

        # High-level pipeline
        auto = derive_cut_cell_scheme(E2_2, psi)

        # Shapes
        assert auto.floating.shape == manual.floating.shape
        assert auto.dims == manual.dims
        assert auto.neumann is not None
        assert auto.eta is not None

        # Floating
        R, T = auto.floating.shape
        for i in range(R):
            for j in range(T):
                assert cancel(auto.floating[i, j] - manual.floating[i, j]) == 0, (
                    f"E2_2 floating mismatch at [{i},{j}]"
                )

        # Neumann
        for i in range(R):
            for j in range(T):
                assert cancel(auto.neumann[i, j] - manual.neumann[i, j]) == 0, (
                    f"E2_2 neumann mismatch at [{i},{j}]"
                )

        # Eta
        for i in range(R):
            assert cancel(auto.eta[i] - manual.eta[i]) == 0, (
                f"E2_2 eta mismatch at row {i}"
            )


@pytest.mark.xfail(reason="conservation not yet enforced for E4_1")
def test_e4_1_conservation_fails():
    """E4_1 cut-cell stencil violates discrete conservation (SBP property).

    For each T-frame column j in 0..T-2, the conservation sum must be zero:
        w_0 * B[0,j] + w_1 * B[1,j] + w_2 * B[2,j] + w_3 * B[3,j] + IC(j) = target(j)
    where w_0=psi, w_i=1 for i>=1, and IC(j) is the interior contribution.
    Target: -1 for j=0 (wall), 0 for j>=1.

    This test exposes the bug: columns j=3,4,5 have nonzero IC but the boundary
    block doesn't compensate, so conservation is violated.
    """
    psi = Symbol("psi")
    ur = derive_uniform_boundary_for_temo(E4_1)
    stencil = construct_cut_cell_stencil(
        ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
    )
    m = stencil.matrix  # R=4 x T=7 matrix
    R, T = 4, 7

    for j in range(T - 1):  # j = 0..5
        # Weighted column sum: w_0=psi for row 0, w_i=1 for rows 1..R-1
        col_sum = psi * m[0, j] + m[1, j] + m[2, j] + m[3, j]

        # Interior contribution: T-frame column j -> grid-frame column j-1
        ic = _interior_contribution(j - 1, R, 2, ur.interior)
        col_sum += ic

        # Target: -1 for wall column (j=0), 0 for j>=1
        target = -1 if j == 0 else 0

        residual = cancel(col_sum - target)
        assert residual == 0, (
            f"Conservation violated at T-frame column j={j}: residual={residual}"
        )


class TestBuildCutCellConservationSystem:
    """Tests for build_cut_cell_conservation_system dimensions and IC values (22.2b)."""

    def test_e2_1_conservation_system_dimensions(self):
        """E2_1: T-1=4 equations, 3 weight unknowns, all IC values zero."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E2_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E2_1.p, q=E2_1.q, nu=E2_1.nu,
            nextra=E2_1.nextra, psi=psi,
        )
        R, T = stencil.matrix.rows, stencil.matrix.cols
        assert R == 4
        assert T == 5

        eqs, ws = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E2_1.p, nu=E2_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        assert len(eqs) == T - 1  # 4 equations
        assert len(ws) == R - 1   # 3 weight unknowns (w_1, w_2, w_3)

        # All IC values should be 0 for E2_1 (no interior row reaches T-frame cols 0..3)
        for j in range(T - 1):
            ic = _interior_contribution(j - 1, R, E2_1.p, ur.interior)
            assert ic == 0, f"E2_1 IC({j}) should be 0, got {ic}"

    def test_e4_1_conservation_system_dimensions(self):
        """E4_1: T-1=6 equations, 3 weight unknowns, nonzero IC at j=3,4,5."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E4_1.p, q=E4_1.q, nu=E4_1.nu,
            nextra=E4_1.nextra, psi=psi,
        )
        R, T = stencil.matrix.rows, stencil.matrix.cols
        assert R == 4
        assert T == 7

        eqs, ws = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E4_1.p, nu=E4_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        assert len(eqs) == T - 1  # 6 equations
        assert len(ws) == R - 1   # 3 weight unknowns (w_1, w_2, w_3)

        # Verify IC values for E4_1
        expected_ic = {
            0: Rational(0), 1: Rational(0), 2: Rational(0),
            3: Rational(1, 12), 4: Rational(-7, 12), 5: Rational(-7, 12),
        }
        for j in range(T - 1):
            ic = _interior_contribution(j - 1, R, E4_1.p, ur.interior)
            assert ic == expected_ic[j], (
                f"E4_1 IC({j}): expected {expected_ic[j]}, got {ic}"
            )

    def test_e4_1_overdetermined_system(self):
        """E4_1 has 6 equations and 3 weight unknowns -> 3 excess constraints."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E4_1.p, q=E4_1.q, nu=E4_1.nu,
            nextra=E4_1.nextra, psi=psi,
        )
        R, T = stencil.matrix.rows, stencil.matrix.cols
        eqs, ws = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E4_1.p, nu=E4_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        excess = len(eqs) - len(ws)
        assert excess == 3, f"Expected 3 excess constraints, got {excess}"

        # Verify the stencil has 4 alpha symbols that must absorb these constraints
        alpha_syms = sorted(stencil.matrix.free_symbols - {psi}, key=lambda s: s.name)
        assert len(alpha_syms) == 4, (
            f"Expected 4 alpha symbols, got {len(alpha_syms)}: {alpha_syms}"
        )


class TestApproachAMinorConditions:
    """Investigation 22.3a-i: augmented matrix minor conditions for E4_1.

    Tests that approach (A) — requiring all 4x4 minors of [A|b] to vanish
    for all psi — is infeasible for the E4_1 conservation system.
    """

    @pytest.fixture
    def e4_conservation_system(self):
        """Build the E4_1 conservation system and extract A, b."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E4_1.p, q=E4_1.q, nu=E4_1.nu,
            nextra=E4_1.nextra, psi=psi,
        )
        B_l = stencil.matrix
        R, T = B_l.rows, B_l.cols
        eqs, w_syms = build_cut_cell_conservation_system(
            B_l, R, T, p=E4_1.p, nu=E4_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        A_mat, b_vec = linear_eq_to_matrix(eqs, w_syms)
        alpha_syms = sorted(B_l.free_symbols - {psi}, key=lambda s: s.name)
        return {
            "psi": psi,
            "A": A_mat,
            "b": b_vec,
            "alpha_syms": alpha_syms,
            "w_syms": w_syms,
        }

    def test_all_15_minors_are_nonzero(self, e4_conservation_system):
        """All C(6,4)=15 4x4 minors of [A|b] are nonzero as functions of (psi, alpha)."""
        A_mat = e4_conservation_system["A"]
        b_vec = e4_conservation_system["b"]
        Aug = A_mat.row_join(b_vec)

        row_combos = list(combinations(range(6), 4))
        assert len(row_combos) == 15

        nonzero_count = 0
        for rows in row_combos:
            submat = Aug.extract(list(rows), [0, 1, 2, 3])
            det_val = cancel(submat.det())
            if det_val != 0:
                nonzero_count += 1

        assert nonzero_count == 15, (
            f"Expected all 15 minors nonzero, got {nonzero_count}"
        )

    def _collect_all_alpha_eqs(self, e4_conservation_system):
        """Helper: extract all alpha equations from psi-coefficients of all 15 minors."""
        psi = e4_conservation_system["psi"]
        A_mat = e4_conservation_system["A"]
        b_vec = e4_conservation_system["b"]
        Aug = A_mat.row_join(b_vec)

        all_alpha_eqs = []
        for rows in combinations(range(6), 4):
            submat = Aug.extract(list(rows), [0, 1, 2, 3])
            det_val = cancel(submat.det())
            if det_val == 0:
                continue
            num, _ = fraction(det_val)
            num_expanded = expand(num)
            num_collected = collect(num_expanded, psi, evaluate=False)
            for key, coeff in num_collected.items():
                c = cancel(coeff)
                if c != 0:
                    all_alpha_eqs.append(c)
        return all_alpha_eqs

    def test_no_alpha_solution_exists(self, e4_conservation_system):
        """No (alpha_0, alpha_1, alpha_2, alpha_3) makes all minors vanish for all psi.

        This proves approach (A) is infeasible: the conservation system A*w=b
        cannot be made consistent for all psi by any choice of alpha alone.
        Any partial solution from a subset of equations fails to satisfy all
        remaining equations simultaneously.
        """
        alpha_syms = e4_conservation_system["alpha_syms"]
        all_alpha_eqs = self._collect_all_alpha_eqs(e4_conservation_system)

        # The system is heavily overdetermined: many equations in 4 unknowns
        assert len(all_alpha_eqs) >= 30

        # Find a candidate solution from a small subset
        candidate = solve(all_alpha_eqs[:10], alpha_syms, dict=True)

        if candidate:
            # Verify the candidate FAILS on remaining equations
            unsatisfied = 0
            for eq in all_alpha_eqs[10:]:
                val = cancel(eq.subs(candidate[0]))
                if val != 0:
                    unsatisfied += 1
            assert unsatisfied > 0, (
                "Candidate solution unexpectedly satisfies all equations"
            )
        else:
            # Even the small subset has no solution — system is definitely inconsistent
            pass

        # Definitive check: solve the full system
        full_sol = solve(all_alpha_eqs, alpha_syms, dict=True)
        assert full_sol == [], (
            f"Expected no solution for full system, got {full_sol}"
        )

    def test_partial_solution_yields_contradictory_alpha3(self, e4_conservation_system):
        """Substituting the partial solution alpha_1=1/3, alpha_2=-4*alpha_3-1/6
        into remaining equations yields contradictory alpha_3 values.

        This is the specific mechanism by which approach (A) fails: the first
        few simplest equations constrain (alpha_1, alpha_2), but the remaining
        overdetermined equations demand different alpha_3 values.
        """
        alpha_syms = e4_conservation_system["alpha_syms"]
        a0, a1, a2, a3 = alpha_syms
        all_alpha_eqs = self._collect_all_alpha_eqs(e4_conservation_system)

        # The partial solution alpha_1=1/3, alpha_2=-4*alpha_3-1/6 comes from the
        # low-degree (degree 1 in psi) minors involving rows 0,1,2,{3,4,5}
        partial = {a1: Rational(1, 3), a2: -4 * a3 - Rational(1, 6)}
        reduced_eqs = []
        for eq in all_alpha_eqs:
            eq_sub = cancel(eq.subs(partial))
            if eq_sub != 0:
                reduced_eqs.append(eq_sub)

        assert len(reduced_eqs) > 0, "Partial solution satisfies all equations"

        # Extract the constraints on alpha_3 alone (no alpha_0 dependence)
        a3_only_eqs = [eq for eq in reduced_eqs if a0 not in eq.free_symbols]
        assert len(a3_only_eqs) >= 2, (
            f"Expected at least 2 alpha_3-only equations, got {len(a3_only_eqs)}"
        )

        # Solve each for alpha_3 — they give different values
        a3_values = set()
        for eq in a3_only_eqs:
            sols = solve(eq, a3)
            for s in sols:
                a3_values.add(s)

        assert len(a3_values) > 1, (
            f"Expected contradictory alpha_3 values, got single: {a3_values}"
        )


class TestApproachBParametricWeights:
    """Investigation 22.3a-ii: parametric weight functions for E4_1.

    Approach (B): parameterize w_i = p_i(psi)/q(psi) with polynomial
    numerators p_i and denominator q = common stencil denominator.
    Substitute into conservation equations, clear denominators, collect
    psi-coefficients, and check solvability of the resulting linear-in-c
    system.

    RESULT: Approach B FAILS.  The conservation system A(psi,alpha)*w =
    b(psi,alpha) is pointwise inconsistent — rank([A|b]) = 4 > rank(A) = 3
    at every tested (psi, alpha).  No weight function w(psi) can produce a
    valid w at those points.  The psi-coefficient system in the weight
    polynomial coefficients c is also inconsistent (rank gap = 1) regardless
    of the polynomial degree.  This follows mathematically from approach A's
    result: since no alpha makes all 4×4 minors of [A|b] vanish identically
    in psi, the system is pointwise inconsistent on a dense set.
    """

    @pytest.fixture
    def e4_data(self):
        """Build E4_1 stencil, conservation system, and derived quantities."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E4_1.p, q=E4_1.q, nu=E4_1.nu,
            nextra=E4_1.nextra, psi=psi,
        )
        B_l = stencil.matrix
        R, T = B_l.rows, B_l.cols
        eqs, w_syms = build_cut_cell_conservation_system(
            B_l, R, T, p=E4_1.p, nu=E4_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        A_mat, b_vec = linear_eq_to_matrix(eqs, w_syms)
        alpha_syms = sorted(B_l.free_symbols - {psi}, key=lambda s: s.name)

        # Common psi-only denominator of all B_l entries
        from sympy import lcm as sym_lcm
        d = S.One
        for i in range(R):
            for j in range(T):
                e = B_l[i, j]
                if e != 0:
                    _, den = fraction(cancel(e))
                    d = sym_lcm(d, den)

        return {
            "psi": psi, "eqs": eqs, "B_l": B_l,
            "alpha_syms": alpha_syms, "w_syms": w_syms,
            "R": R, "T": T,
            "A": A_mat, "b": b_vec,
            "common_denom": d,
        }

    def test_common_denominator_is_psi_plus_1_2_3(self, e4_data):
        """The common psi-denominator of E4_1 stencil entries is
        12*(psi+1)*(psi+2)*(psi+3).

        Rows 0-2 each have denominator 6*(psi+1)*(psi+2)*(psi+3);
        row 3 (the near-interior row) has constant denominator 12.
        """
        from sympy import Poly as SPoly, lcm as sym_lcm

        psi = e4_data["psi"]
        B_l = e4_data["B_l"]
        R, T = e4_data["R"], e4_data["T"]

        # Per-row denominators
        row_denoms = []
        for i in range(R):
            rd = S.One
            for j in range(T):
                e = B_l[i, j]
                if e != 0:
                    _, den = fraction(cancel(e))
                    rd = sym_lcm(rd, den)
            row_denoms.append(rd)

        # Rows 0-2 share the same denominator
        expected_row_denom = 6 * (psi + 1) * (psi + 2) * (psi + 3)
        for i in range(3):
            assert cancel(row_denoms[i] - expected_row_denom) == 0, (
                f"Row {i} denom mismatch: {factor(row_denoms[i])}"
            )

        # Row 3 has constant denominator
        assert row_denoms[3].free_symbols == set(), (
            f"Row 3 denom should be constant, got {factor(row_denoms[3])}"
        )

        # Overall common denominator
        d = e4_data["common_denom"]
        deg = SPoly(d, psi).degree()
        assert deg == 3, f"Expected degree 3, got {deg}"

    def test_pointwise_inconsistency(self, e4_data):
        """At every tested (psi, alpha), rank([A|b]) > rank(A).

        This directly proves approach B fails: if the 6×3 system A*w = b
        has no solution at a specific psi_0, no weight function w(psi) can
        produce a valid w(psi_0).
        """
        psi = e4_data["psi"]
        A_mat = e4_data["A"]
        b_vec = e4_data["b"]
        alpha_syms = e4_data["alpha_syms"]

        test_alphas = [
            {a: Integer(0) for a in alpha_syms},
            {alpha_syms[0]: Rational(11, 6), alpha_syms[1]: Rational(1, 3),
             alpha_syms[2]: Rational(-1, 6), alpha_syms[3]: Integer(0)},
            {a: Rational(i + 1, 10) for i, a in enumerate(alpha_syms)},
        ]
        psi_vals = [Rational(1, 4), Rational(1, 2), Rational(3, 4)]

        inconsistent_count = 0
        total = 0
        for alpha_vals in test_alphas:
            for psi_val in psi_vals:
                total += 1
                subs = {psi: psi_val, **alpha_vals}
                A_num = A_mat.subs(subs)
                b_num = b_vec.subs(subs)
                aug = A_num.row_join(b_num)
                r_A = A_num.rank()
                r_aug = aug.rank()
                if r_aug > r_A:
                    inconsistent_count += 1

        # ALL tested points should be inconsistent
        assert inconsistent_count == total, (
            f"Expected all {total} points inconsistent, got {inconsistent_count}"
        )

    def test_parametric_c_system_inconsistent(self, e4_data):
        """For fixed alpha=0, the psi-coefficient system in weight polynomial
        coefficients c is inconsistent at every tested degree (3, 5, 7).

        The rank gap is always exactly 1: rank([M|b]) = rank(M) + 1.
        This confirms the failure is fundamental, not a degree limitation.
        """
        from sympy import Poly as SPoly

        psi = e4_data["psi"]
        eqs = e4_data["eqs"]
        w_syms = e4_data["w_syms"]
        alpha_syms = e4_data["alpha_syms"]
        q = e4_data["common_denom"]
        alpha_vals = {a: Integer(0) for a in alpha_syms}

        for deg in [3, 5, 7]:
            c_all = []
            w_subs = {}
            for idx, w in enumerate(w_syms):
                cs = [Symbol(f"c_{idx + 1}_{k}") for k in range(deg + 1)]
                c_all.extend(cs)
                p = sum(cs[k] * psi**k for k in range(deg + 1))
                w_subs[w] = p / q

            psi_eqs = []
            for eq in eqs:
                eq_sub = eq.subs(w_subs).subs(alpha_vals)
                eq_c = cancel(eq_sub)
                num, _ = fraction(eq_c)
                poly = SPoly(expand(num), psi)
                for coeff in poly.all_coeffs():
                    c_val = cancel(coeff)
                    if c_val != 0:
                        psi_eqs.append(c_val)

            M, rhs = linear_eq_to_matrix(psi_eqs, c_all)
            aug = M.row_join(rhs)
            r_M = M.rank()
            r_aug = aug.rank()

            assert r_aug > r_M, (
                f"Degree {deg}: expected inconsistent, got rank(M)={r_M}, "
                f"rank([M|b])={r_aug}"
            )
            assert r_aug == r_M + 1, (
                f"Degree {deg}: rank gap should be 1, got {r_aug - r_M}"
            )
