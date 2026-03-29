"""Tests for E4_1 cut-cell stencil derivation (21.1b onwards)."""

import pathlib

import pytest
from sympy import (
    Integer, Matrix, Poly, Rational, S, Symbol, cancel, collect, expand,
    factor, fraction, linear_eq_to_matrix, linsolve, simplify, solve,
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
    build_temo_vandermonde,
    compute_dimensions,
    construct_cut_cell_stencil,
    derive_cut_cell_scheme,
    derive_e2_uniform_boundary,
    derive_uniform_boundary_for_temo,
    make_psi_field,
    solve_temo_row,
    solve_uniform_limit,
)


class TestE4UniformBoundary:
    """Tests for derive_uniform_boundary_for_temo with E4_1 (21.1b)."""

    @pytest.fixture
    def e4_result(self):
        """Compute E4_1 uniform boundary once for the test class."""
        return derive_uniform_boundary_for_temo(E4_1)

    def test_shape(self, e4_result):
        """E4_1 B_u has shape (4, 6) — r_eff=4 rows, t=6 columns."""
        assert e4_result.B_u.shape == (4, 6)

    def test_five_alpha_symbols(self, e4_result):
        """E4_1 has exactly 5 free alpha symbols."""
        assert len(e4_result.alpha_symbols) == 5
        # Verify they are named alpha_0..alpha_4
        for k, sym in enumerate(e4_result.alpha_symbols):
            assert sym.name == f"alpha_{k}"

    def test_zero_constraints(self, e4_result):
        """B_u[0, 5] == 0, B_u[1, 5] == 0, B_u[2, 5] == 0 (zero-constrained entries)."""
        assert e4_result.B_u[0, 5] == 0
        assert e4_result.B_u[1, 5] == 0
        assert e4_result.B_u[2, 5] == 0

    def test_last_row_free_alphas(self, e4_result):
        """B_u[3, 4] and B_u[3, 5] contain alpha_3, alpha_4."""
        alpha_3 = e4_result.alpha_symbols[3]
        alpha_4 = e4_result.alpha_symbols[4]
        # These entries should involve alpha_3 and alpha_4 respectively
        assert alpha_3 in e4_result.B_u[3, 4].free_symbols
        assert alpha_4 in e4_result.B_u[3, 5].free_symbols

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
        syms = [Symbol(f"a{k}") for k in range(5)]
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
        """E4_1 cut-cell stencil has shape (5, 7) — R=5, T=7."""
        _, result, _ = e4_temo
        assert result.matrix.shape == (5, 7)

    def test_no_betas(self, e4_temo):
        """E4_1 (nextra=0) produces no beta parameters."""
        _, result, _ = e4_temo
        assert len(result.beta_info) == 0
        assert len(result.beta_symbols) == 0

    def test_entries_in_psi_alpha(self, e4_temo):
        """All entries are rational in psi and alpha_{0..4} only."""
        _, result, _ = e4_temo
        all_syms = result.matrix.free_symbols
        expected_names = {"psi"} | {f"alpha_{k}" for k in range(5)}
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

    def test_uniform_limit_rows_0_3_embed_Bu(self, e4_temo):
        """At psi=1, rows 0-3: wall col=0, then cols 1-6 = B_u rows 0-3."""
        ur, result, psi = e4_temo
        m1 = result.matrix.subs(psi, 1)
        B_u = ur.B_u
        for i in range(4):
            # Column 0 is the wall column
            # Columns 1..6 should match B_u[i, 0..5]
            for j in range(6):
                assert simplify(m1[i, j + 1] - B_u[i, j]) == 0, (
                    f"B_u embed mismatch at row {i}, col {j}: "
                    f"{cancel(m1[i, j + 1])} != {cancel(B_u[i, j])}"
                )

    def test_uniform_limit_row4_not_interior(self, e4_temo):
        """At psi=1, row 4 is NOT the raw interior stencil — it's derived via conservation+Taylor.

        The interior stencil [1/12, -2/3, 0, 2/3, -1/12] overflows the T-frame at row 4,
        so solve_uniform_limit computes a closure row that contains alpha symbols.
        """
        ur, result, psi = e4_temo
        m1 = result.matrix.subs(psi, 1)
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0)
        # Row 4 must match the solve_uniform_limit result
        for j in range(7):
            assert simplify(m1[4, j] - B_l_1[4, j]) == 0, (
                f"Row 4 uniform limit mismatch at col {j}: "
                f"{cancel(m1[4, j])} != {cancel(B_l_1[4, j])}"
            )
        # Negative assertion: row 4 is NOT the raw interior stencil
        raw_interior_in_T = [
            S.Zero, S.Zero,
            Rational(1, 12), Rational(-2, 3), S.Zero,
            Rational(2, 3), Rational(-1, 12),
        ]
        assert any(
            simplify(m1[4, j] - raw_interior_in_T[j]) != 0 for j in range(7)
        ), "Row 4 should NOT be the raw interior stencil"

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

        # floating_coeffs: R*T = 5*7 = 35 entries, row-major from cc.floating
        floating_flat = list(cc.floating)

        # dirichlet_coeffs: R*T = 35 entries (prepend T=7 zeros for row 0)
        dirichlet_flat = [Integer(0)] * 7 + list(cc.dirichlet)

        spec = StencilGenSpec(
            name="E4_1",
            P=2,
            R=5,
            T=7,
            X=0,
            derivative_order=1,
            is_uniform=False,
            param_arrays={"alpha": 5},
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
        """Generated code has P=2, R=5, T=7, X=0."""
        assert "static constexpr int P = 2;" in e4_code
        assert "static constexpr int R = 5;" in e4_code
        assert "static constexpr int T = 7;" in e4_code
        assert "static constexpr int X = 0;" in e4_code

    def test_struct_name(self, e4_code):
        """Generated code defines struct E4_1."""
        assert "struct E4_1" in e4_code

    def test_namespace(self, e4_code):
        """Generated code uses ccs::stencils namespace."""
        assert "namespace ccs::stencils" in e4_code

    def test_alpha_array(self, e4_code):
        """Generated code has std::array<real, 5> alpha member."""
        assert "std::array<real, 5> alpha;" in e4_code

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
        """Generated code has nbs_floating method with 35 coefficient assignments."""
        assert "nbs_floating(real h," in e4_code
        floating_start = e4_code.index("nbs_floating(real h,")
        dirichlet_start = e4_code.index("nbs_dirichlet(real h,")
        floating_section = e4_code[floating_start:dirichlet_start]
        # R*T = 5*7 = 35 coefficient assignments
        assert floating_section.count("c[") == 35, (
            f"Expected 35 c[] assignments in floating, got {floating_section.count('c[')}"
        )

    def test_nbs_dirichlet_method(self, e4_code):
        """Generated code has nbs_dirichlet method with 28 coefficient assignments."""
        assert "nbs_dirichlet(real h," in e4_code
        dirichlet_start = e4_code.index("nbs_dirichlet(real h,")
        neumann_start = e4_code.index("nbs_neumann")
        dirichlet_section = e4_code[dirichlet_start:neumann_start]
        # (R-1)*T = 4*7 = 28 coefficient assignments
        assert dirichlet_section.count("c[") == 28, (
            f"Expected 28 c[] assignments in dirichlet, got {dirichlet_section.count('c[')}"
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

    ALPHA_VALUES = {"alpha": [0.1, -0.05, 0.02, 0.01, 0.005]}

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
            R=5,
            T=7,
            X=0,
            derivative_order=1,
            is_uniform=False,
            param_arrays={"alpha": 5},
            interior_coeffs=ur.interior,
            floating_coeffs=floating_flat,
            dirichlet_coeffs=dirichlet_flat,
        )

    def test_compute_floating_values(self, e4_spec):
        """compute_test_values produces 35 floating coefficients at psi=1.0."""
        values = compute_test_values(
            e4_spec.floating_coeffs,
            alpha_values=self.ALPHA_VALUES,
            h=1.0,
            psi=1.0,
        )
        assert len(values) == 35  # R*T = 5*7

    def test_compute_dirichlet_values(self, e4_spec):
        """compute_test_values produces 28 Dirichlet coefficients at psi=0.7."""
        dirichlet_emitted = e4_spec.dirichlet_coeffs[e4_spec.T:]
        values = compute_test_values(
            dirichlet_emitted,
            alpha_values=self.ALPHA_VALUES,
            h=0.5,
            psi=0.7,
        )
        assert len(values) == 28  # (R-1)*T = 4*7

    def test_floating_uniform_limit_row4_not_interior(self, e4_spec):
        """At psi=1.0, floating row 4 is NOT the raw interior stencil — it's derived via conservation+Taylor.

        Compute expected values dynamically from the 5×7 floating matrix.
        """
        values = compute_test_values(
            e4_spec.floating_coeffs,
            alpha_values=self.ALPHA_VALUES,
            h=2.0,
            psi=1.0,
        )
        # Row 4 = indices 28..34
        row4 = values[28:35]
        # Row 4 should NOT be the raw interior stencil / h
        raw_interior_over_h = [0, 0, 1 / 24, -1 / 3, 0, 1 / 3, -1 / 24]
        assert any(
            abs(got - want) > 1e-12
            for got, want in zip(row4, raw_interior_over_h)
        ), "Row 4 should NOT be the raw interior stencil"

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
        assert "alpha = {0.1, -0.05, 0.02, 0.01, 0.005}" in code
        assert "REQUIRE(p == 2)" in code
        assert "REQUIRE(r == 5)" in code
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
        """E4_1 via derive_cut_cell_scheme has R=5, T=7 floating matrix."""
        psi = Symbol("psi")
        result = derive_cut_cell_scheme(E4_1, psi)
        assert result.floating.shape == (5, 7)
        assert result.dirichlet.shape == (4, 7)
        assert result.dims.R == 5
        assert result.dims.T == 7
        assert result.dims.X == 0

    def test_e4_1_no_neumann(self):
        """E4_1 (nu=1) has no Neumann stencil."""
        psi = Symbol("psi")
        result = derive_cut_cell_scheme(E4_1, psi)
        assert result.neumann is None
        assert result.eta is None

    def test_e4_1_alpha_count(self):
        """E4_1 has 4 free alpha symbols (conservation eliminates 1)."""
        psi = Symbol("psi")
        result = derive_cut_cell_scheme(E4_1, psi)
        assert len(result.alpha_symbols) == 4

    def test_e4_1_matches_manual_pipeline(self):
        """derive_cut_cell_scheme(E4_1) equals manual pipeline + conservation subs."""
        psi = Symbol("psi")

        # Manual non-conserved pipeline (5 alphas)
        ur = derive_uniform_boundary_for_temo(E4_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
        )
        dims = compute_dimensions(E4_1.p, E4_1.q, E4_1.s, E4_1.nextra, E4_1.nu)
        manual = assemble_cut_cell_result(
            stencil.matrix, None, None, dims, ur.alpha_symbols,
        )

        # Conservative high-level pipeline (4 alphas)
        auto = derive_cut_cell_scheme(E4_1, psi)

        # Apply conservation_subs to manual, then rename surviving alphas
        assert auto.conservation_subs is not None
        manual_subbed = manual.floating.xreplace(auto.conservation_subs)
        # Rename surviving nc-alphas (alpha_0..alpha_2, alpha_4) to final (alpha_0..alpha_3)
        nc_alphas = ur.alpha_symbols  # [alpha_0..alpha_4]
        surviving = [a for a in nc_alphas if a not in auto.conservation_subs]
        rename = dict(zip(surviving, auto.alpha_symbols))
        manual_subbed = manual_subbed.xreplace(rename)

        # Compare floating matrices entry by entry
        R, T = auto.floating.shape
        for i in range(R):
            for j in range(T):
                assert cancel(auto.floating[i, j] - manual_subbed[i, j]) == 0, (
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
        """derive_cut_cell_scheme accepts custom alpha symbols (4 post-conservation)."""
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


@pytest.mark.xfail(reason="conservation not yet enforced for E4_1 cut-cell")
def test_e4_1_conservation_fails():
    """E4_1 cut-cell stencil violates discrete conservation (SBP property).

    Conservation applies to grid-point columns (T-frame cols 1..T-2):
        sum_i w_i * B[i, g+1] + IC(g) = target(g)
    where g is the grid point (g = T-frame col - 1), w_0=psi,
    w_i=1 for i>=1 (naive flat weights), and IC(g) is the interior
    contribution to grid point g.
    Target: -1 at grid point 0, 0 elsewhere.

    The T-frame col 0 is the wall (delta) column, which is NOT a grid
    point and is excluded from the SBP conservation check.
    """
    psi = Symbol("psi")
    ur = derive_uniform_boundary_for_temo(E4_1)
    stencil = construct_cut_cell_stencil(
        ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
    )
    m = stencil.matrix  # R=5 x T=7 matrix
    R, T = 5, 7

    for j_tf in range(1, T - 1):  # T-frame cols 1..5 (grid points 0..4)
        g = j_tf - 1  # grid point index
        # Weighted column sum: w_0=psi for row 0, w_i=1 for rows 1..R-1
        col_sum = psi * m[0, j_tf] + m[1, j_tf] + m[2, j_tf] + m[3, j_tf] + m[4, j_tf]

        # Interior contribution for grid point g
        ic = _interior_contribution(g, R, 2, ur.interior)
        col_sum += ic

        # Target: -1 at grid point 0, 0 elsewhere
        target = -1 if g == 0 else 0

        residual = cancel(col_sum - target)
        assert residual == 0, (
            f"Conservation violated at grid point {g} (T-frame col {j_tf}): "
            f"residual={residual}"
        )


class TestBuildCutCellConservationSystem:
    """Tests for build_cut_cell_conservation_system dimensions and IC values (22.2b)."""

    def test_e2_1_conservation_system_dimensions(self):
        """E2_1: T-2=3 equations (grid-point cols), 3 weight unknowns, all IC zero."""
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
        assert len(eqs) == T - 2  # 3 equations (grid-point cols 0..2)
        assert len(ws) == R - 1   # 3 weight unknowns (w_1, w_2, w_3)

        # All IC values should be 0 for E2_1 (no interior row reaches grid points 0..2)
        for g in range(T - 2):
            ic = _interior_contribution(g, R, E2_1.p, ur.interior)
            assert ic == 0, f"E2_1 IC(g={g}) should be 0, got {ic}"

    def test_e4_1_conservation_system_dimensions(self):
        """E4_1: T-2=5 equations (grid-point cols), 4 weight unknowns, nonzero IC at g=3,4."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E4_1.p, q=E4_1.q, nu=E4_1.nu,
            nextra=E4_1.nextra, psi=psi,
        )
        R, T = stencil.matrix.rows, stencil.matrix.cols
        assert R == 5
        assert T == 7

        eqs, ws = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E4_1.p, nu=E4_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        assert len(eqs) == T - 2  # 5 equations (grid-point cols 0..4)
        assert len(ws) == R - 1   # 4 weight unknowns (w_1, w_2, w_3, w_4)

        # Verify IC values for E4_1 at R=5 (grid points 0..4)
        expected_ic = {
            0: Rational(0), 1: Rational(0), 2: Rational(0),
            3: Rational(1, 12), 4: Rational(-7, 12),
        }
        for g in range(T - 2):
            ic = _interior_contribution(g, R, E4_1.p, ur.interior)
            assert ic == expected_ic[g], (
                f"E4_1 IC(g={g}): expected {expected_ic[g]}, got {ic}"
            )

    def test_e4_1_overdetermined_system(self):
        """E4_1 has 5 equations and 4 weight unknowns -> 1 excess constraint."""
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
        assert excess == 1, f"Expected 1 excess constraint, got {excess}"

        # Verify the stencil has 5 alpha symbols that must absorb these constraints
        alpha_syms = sorted(stencil.matrix.free_symbols - {psi}, key=lambda s: s.name)
        assert len(alpha_syms) == 5, (
            f"Expected 5 alpha symbols, got {len(alpha_syms)}: {alpha_syms}"
        )


def test_e4_1_conservation_constant_weights_infeasible_r5():
    """E4_1 conservation with constant weights is infeasible at R=5 (23.3a).

    The conservation equations are rational in psi with bilinear terms w_i * alpha_k.
    We theta-linearize (replace w_i * alpha_k -> theta_{i,k}), clear psi-denominators,
    extract psi-coefficients to get scalar linear equations, then check
    rank(M) vs rank([M|b]) via the Rouche-Capelli theorem.

    Result: rank gap = 1, meaning the system is inconsistent. Conservation with
    constant (psi-independent) weights w_1..w_4 is structurally infeasible at R=5.
    Direct symbolic solve confirms: w_i solutions are rational functions of psi,
    not constants. A psi-dependent norm formulation would be needed.
    """
    psi = Symbol("psi")
    ur = derive_uniform_boundary_for_temo(E4_1)
    stencil = construct_cut_cell_stencil(
        ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
    )
    m = stencil.matrix
    R, T = 5, 7

    # Step 1: Build conservation equations (5 eqs, 4 weight unknowns)
    eqs, w_syms = build_cut_cell_conservation_system(
        m, R, T, p=2, nu=1, interior_coeffs=ur.interior, psi=psi,
    )
    assert len(eqs) == 5
    assert len(w_syms) == 4

    # Identify alpha symbols in the stencil
    alpha_syms = sorted(m.free_symbols - {psi}, key=lambda s: s.name)
    assert len(alpha_syms) == 5

    # Step 2: Identify which alphas appear in each row and create theta symbols
    theta_syms = []
    theta_map = {}  # (row_index, alpha) -> theta symbol
    row_alpha_map = {}  # row_index -> [alphas]
    for i in range(1, R):
        row_alphas = set()
        for j in range(T):
            row_alphas.update(s for s in m[i, j].free_symbols if s in alpha_syms)
        row_alphas_sorted = sorted(row_alphas, key=lambda s: s.name)
        row_alpha_map[i] = row_alphas_sorted
        for alpha in row_alphas_sorted:
            theta = Symbol(f"th_{i}_{alpha.name}")
            theta_syms.append(theta)
            theta_map[(i, alpha)] = theta

    # Alphas that appear in row 0 (linear, since w_0=psi is a parameter)
    row0_alphas = set()
    for j in range(T):
        row0_alphas.update(s for s in m[0, j].free_symbols if s in alpha_syms)
    row0_alpha_list = sorted(row0_alphas, key=lambda s: s.name)

    # Build substitution dict for all bilinear pairs w_i * alpha_k
    subs_dict = {}
    for i in range(1, R):
        w_i = w_syms[i - 1]
        for alpha in row_alpha_map[i]:
            subs_dict[w_i * alpha] = theta_map[(i, alpha)]

    # Step 3: Clear psi-denominators, expand, theta-linearize, extract psi-coefficients
    scalar_eqs = []
    for eq in eqs:
        num, _den = fraction(cancel(eq))
        poly_num = expand(num)
        lin_num = poly_num.subs(subs_dict)
        if psi in lin_num.free_symbols:
            p_poly = Poly(lin_num, psi)
            scalar_eqs.extend(p_poly.all_coeffs())
        else:
            scalar_eqs.append(lin_num)

    # Step 4: Build linear system and check rank (Rouche-Capelli)
    lin_unknowns = list(w_syms) + row0_alpha_list + theta_syms
    M_mat, b_vec = linear_eq_to_matrix(scalar_eqs, lin_unknowns)
    M_aug = M_mat.row_join(b_vec)

    rank_M = M_mat.rank()
    rank_aug = M_aug.rank()

    # Conservation with constant weights is INFEASIBLE: rank gap = 1
    assert rank_aug - rank_M == 1, (
        f"Expected rank gap 1, got {rank_aug - rank_M} "
        f"(rank(M)={rank_M}, rank([M|b])={rank_aug})"
    )
    assert rank_M == 8
    assert rank_aug == 9

    # Cross-check: direct symbolic solve produces psi-DEPENDENT weights,
    # confirming constant weights cannot satisfy conservation for all psi
    all_unknowns = list(w_syms) + list(alpha_syms)
    sol = solve(eqs, all_unknowns, dict=True)
    assert len(sol) == 1
    for w in w_syms:
        if w in sol[0]:
            assert psi in sol[0][w].free_symbols, (
                f"{w} solution should depend on psi"
            )


class TestE4UniformConservation:
    """Tests for derive_uniform_boundary_for_temo(E4_1, conserve=True) (23.3a)."""

    @pytest.fixture(scope="class")
    def conserved(self):
        return derive_uniform_boundary_for_temo(E4_1, conserve=True)

    def test_alpha_count(self, conserved):
        """Conservation reduces E4_1 from 5 to 4 free alphas."""
        assert len(conserved.alpha_symbols) == 4
        for k, sym in enumerate(conserved.alpha_symbols):
            assert sym.name == f"alpha_{k}"

    def test_shape_unchanged(self, conserved):
        """B_u shape is still (4, 6) — conservation doesn't change dimensions."""
        assert conserved.B_u.shape == (4, 6)

    def test_weights_present(self, conserved):
        """Conservation produces 4 quadrature weights."""
        assert conserved.weights is not None
        assert len(conserved.weights) == 4

    def test_weights_depend_on_alpha_3(self, conserved):
        """All weights are rational functions of alpha_3 only."""
        alpha_3 = conserved.alpha_symbols[3]
        for w in conserved.weights:
            assert w.free_symbols <= {alpha_3}, (
                f"Weight {w} depends on {w.free_symbols}, expected only {alpha_3}"
            )

    def test_conservation_holds(self, conserved):
        """Weighted column sums satisfy the SBP conservation condition."""
        B_u = conserved.B_u
        r, t = B_u.shape
        for j in range(t - 1):
            col_sum = sum(conserved.weights[i] * B_u[i, j] for i in range(r))
            ic = _interior_contribution(j, r, conserved.p, conserved.interior)
            total = cancel(col_sum + ic)
            target = -1 if j == 0 else 0
            assert cancel(total - target) == 0, (
                f"Conservation violated at col {j}: residual={cancel(total - target)}"
            )

    def test_taylor_accuracy(self, conserved):
        """Each row still satisfies Taylor matching for q+1=4 equations."""
        B_u = conserved.B_u
        r, t = B_u.shape
        for i in range(r):
            for m in range(4):
                moment = sum(B_u[i, j] * (j - i) ** m for j in range(t))
                expected = 1 if m == 1 else 0
                assert simplify(moment - expected) == 0, (
                    f"Row {i}, moment {m}: got {simplify(moment)}, expected {expected}"
                )

    def test_rows_0_2_unchanged(self, conserved):
        """Rows 0-2 are identical to the non-conservative result."""
        ur_no_cons = derive_uniform_boundary_for_temo(E4_1, conserve=False)
        for i in range(3):
            for j in range(6):
                diff = cancel(conserved.B_u[i, j] - ur_no_cons.B_u[i, j])
                assert diff == 0, f"Row {i}, col {j} should be unchanged"

    def test_only_alpha_symbols_in_Bu(self, conserved):
        """B_u contains only the expected alpha symbols."""
        expected_syms = set(conserved.alpha_symbols)
        actual_syms = conserved.B_u.free_symbols
        assert actual_syms <= expected_syms, (
            f"Unexpected symbols in B_u: {actual_syms - expected_syms}"
        )

    def test_custom_alpha_symbols(self):
        """conserve=True accepts 4 custom alpha names."""
        syms = [Symbol(f"a{k}") for k in range(4)]
        result = derive_uniform_boundary_for_temo(E4_1, alpha_symbols=syms, conserve=True)
        assert result.alpha_symbols == syms
        assert result.B_u.free_symbols <= set(syms)

    def test_e2_1_unaffected(self):
        """E2_1 (nextra=1) is unaffected by conserve=True — uses its own conservation."""
        ur = derive_uniform_boundary_for_temo(E2_1, conserve=True)
        assert len(ur.alpha_symbols) == 4
        assert ur.weights is None  # nextra=1 conservation is inline, no explicit weights

