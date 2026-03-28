"""Tests for E4_1 cut-cell stencil derivation (21.1b onwards)."""

import pathlib

import pytest
from sympy import Integer, Matrix, Rational, S, Symbol, cancel, simplify

from stencil_gen.codegen import (
    StencilGenSpec,
    generate_stencil_cpp,
)
from stencil_gen.temo import (
    E4_1,
    SchemeParams,
    UniformResult,
    assemble_cut_cell_result,
    build_cut_cell_deltas,
    build_degenerate_stencil,
    compute_dimensions,
    construct_cut_cell_stencil,
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
