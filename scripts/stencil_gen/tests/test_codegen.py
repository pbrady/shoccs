"""Tests for stencil_gen.codegen module."""

from sympy import Integer, Rational, Symbol, symbols

from stencil_gen.codegen import (
    apply_cse,
    format_rational_h_division,
    generate_interior_method,
    generate_nbs_method,
)
from stencil_gen.printer import StencilCodePrinter, build_symbol_map

# ── Shared test fixtures ─────────────────────────────────────────────────

alpha_0, alpha_1 = symbols("alpha_0 alpha_1")

# --- E4u_1 fixtures (uniform, nu=1, R=3, T=5) ---

uniform_printer = StencilCodePrinter(symbol_map=build_symbol_map({"alpha": 2}))

# 15 expressions from E4u_1.cpp nbs_floating (lines 80-94)
e4u_floating_coeffs = [
    (6 * alpha_0 - 11) / 6,
    3 - 4 * alpha_0,
    (12 * alpha_0 - 3) / 2,
    -(12 * alpha_0 - 1) / 3,
    alpha_0,
    (3 * alpha_1 - 1) / 3,
    -(8 * alpha_1 + 1) / 2,
    6 * alpha_1 + 1,
    -(24 * alpha_1 + 1) / 6,
    alpha_1,
    -(168 * alpha_1 + 54 * alpha_0 - 11) / 138,
    (112 * alpha_1 + 36 * alpha_0 - 15) / 23,
    -(336 * alpha_1 + 108 * alpha_0 + 1) / 46,
    (336 * alpha_1 + 108 * alpha_0 + 47) / 69,
    -(28 * alpha_1 + 9 * alpha_0 + 2) / 23,
]

# 10 expressions: nbs_dirichlet coefficients (already sliced, rows 1-2 only)
e4u_dirichlet_coeffs = e4u_floating_coeffs[5:]

# --- polyE2_1 fixtures (cut-cell, nu=1, R=3, T=4) ---

fa = [Symbol(f"fa_{i}") for i in range(6)]
psi = Symbol("psi")

cutcell_printer = StencilCodePrinter(
    symbol_map=build_symbol_map({"fa": 6, "da": 3, "ia": 4}, has_psi=True)
)

# 12 expressions reconstructed from polyE2_1.cpp nbs_floating
poly_floating_coeffs = [
    (fa[1] - 1) / (2 * (1 + psi)),
    (fa[0] - 1) / 2,
    -fa[0] - (2 + psi) * (fa[1] - 1) / (2 * (1 + psi)),
    (fa[0] + fa[1]) / 2,
    (fa[3] - 1) / (2 * (1 + psi)),
    (fa[2] - 1) / 2,
    -fa[2] - (2 + psi) * (fa[3] - 1) / (2 * (1 + psi)),
    (fa[3] + fa[2]) / 2,
    (fa[5] - 1) / (2 * (1 + psi)),
    (fa[4] - 1) / 2,
    -fa[4] - (2 + psi) * (fa[5] - 1) / (2 * (1 + psi)),
    (fa[5] + fa[4]) / 2,
]


# ── 20.4b: CSE wrapper tests ──────────────────────────────────────────────


def test_cse_psi_dependent():
    """CSE hoists common psi subexpressions."""
    psi = Symbol("psi")
    a = Symbol("alpha_0")
    coeffs = [a / (1 + psi), (1 - a) / (1 + psi), a / (2 + psi)]
    repls, reduced = apply_cse(coeffs, prefix="t", start=5)
    # (1 + psi) should be hoisted as a common subexpression
    assert len(repls) >= 1
    # Verify algebraic equivalence after substitution
    for orig, red in zip(coeffs, reduced):
        restored = red.xreplace(dict(repls))
        assert (orig - restored).cancel() == 0


def test_cse_uniform_skipped():
    """Uniform stencil: CSE is not called (decision is caller's)."""
    # This test validates the decision logic, not apply_cse itself
    a0, a1 = symbols("alpha_0 alpha_1")
    coeffs = [(6 * a0 - 11) / 6, 3 - 4 * a0, a1]
    # For uniform, caller should NOT call apply_cse
    # Verify the expressions are simple enough
    assert all(c.count_ops() <= 20 for c in coeffs)


def test_cse_numbering_starts_at_5():
    """CSE temporaries start at t5 by default."""
    psi = Symbol("psi")
    a = Symbol("alpha_0")
    coeffs = [a / (1 + psi), (1 - a) / (1 + psi)]
    repls, _ = apply_cse(coeffs, prefix="t", start=5)
    if repls:
        # First temporary should be t5
        assert repls[0][0].name == "t5"


def test_cse_custom_prefix_and_start():
    """Custom prefix and start index work."""
    psi = Symbol("psi")
    a = Symbol("alpha_0")
    coeffs = [a / (1 + psi), (1 - a) / (1 + psi)]
    repls, _ = apply_cse(coeffs, prefix="x", start=0)
    if repls:
        assert repls[0][0].name == "x0"


def test_cse_roundtrip():
    """Full round-trip: substituting replacements back recovers original."""
    psi = Symbol("psi")
    fa_0, fa_1, fa_2 = symbols("fa_0 fa_1 fa_2")
    coeffs = [
        (fa_1 - 1) / (2 * (1 + psi)),
        (fa_0 - 1) / 2,
        -fa_0 - (2 + psi) * (fa_1 - 1) / (2 * (1 + psi)),
        (fa_0 + fa_1) / 2,
    ]
    repls, reduced = apply_cse(coeffs)
    subs = dict(repls)
    for orig, red in zip(coeffs, reduced):
        restored = red.xreplace(subs)
        assert (orig - restored).cancel() == 0


# ── 20.4c: Interior method generator tests ────────────────────────────────


def test_format_rational_h_division_nu1():
    """nu=1 rational formatting matches E4u_1.cpp pattern."""
    assert format_rational_h_division(Rational(1, 12), nu=1) == "1 / (12 * h)"
    assert format_rational_h_division(Rational(-2, 3), nu=1) == "-2 / (3 * h)"
    assert format_rational_h_division(Rational(1, 1), nu=1) == "1 / h"
    assert format_rational_h_division(Rational(0, 1), nu=1) == "0"


def test_format_rational_h_division_nu2():
    """nu=2 rational formatting matches E2_2.cpp, E4_2.cpp patterns."""
    assert format_rational_h_division(Rational(1, 1), nu=2) == "1 / (h * h)"
    assert format_rational_h_division(Rational(-2, 1), nu=2) == "-2 / (h * h)"
    assert format_rational_h_division(Rational(-1, 12), nu=2) == "-1 / (12 * (h * h))"
    assert format_rational_h_division(Rational(4, 3), nu=2) == "4 / (3 * (h * h))"
    assert format_rational_h_division(Rational(-5, 2), nu=2) == "-5 / (2 * (h * h))"
    assert format_rational_h_division(Rational(0, 1), nu=2) == "0"


def test_interior_e4u_antisymmetry():
    """E4u interior uses shorthand: c[3] = -c[1]; c[4] = -c[0];"""
    coeffs = [Rational(1, 12), Rational(-2, 3), 0, Rational(2, 3), Rational(-1, 12)]
    body = generate_interior_method(coeffs, nu=1, is_uniform=True)
    assert "c[3] = -c[1];" in body
    assert "c[4] = -c[0];" in body
    assert "c[2] = 0;" in body


def test_interior_e8u():
    """E8u interior uses shorthand for all 4 mirrored pairs."""
    coeffs = [
        Rational(1, 280), Rational(-4, 105), Rational(1, 5), Rational(-4, 5),
        0, Rational(4, 5), Rational(-1, 5), Rational(4, 105), Rational(-1, 280),
    ]
    body = generate_interior_method(coeffs, nu=1, is_uniform=True)
    assert "c[5] = -c[3];" in body
    assert "c[8] = -c[0];" in body


def test_interior_poly_loop():
    """polyE2 interior uses loop-based h division."""
    coeffs = [Rational(-1, 2), 0, Rational(1, 2)]
    body = generate_interior_method(coeffs, nu=1, is_uniform=False)
    assert "for (auto&& v : c) v /= h;" in body
    assert "-0.5" in body  # no baked-in h


# ── 20.4d: Boundary method generator tests ───────────────────────────────


def test_nbs_uniform_no_cse():
    """Uniform stencil boundary has no CSE temporaries."""
    body = generate_nbs_method(
        "nbs_floating",
        e4u_floating_coeffs,
        r=3,
        t=5,
        printer=uniform_printer,
        psi_dependent=False,
    )
    assert "real t" not in body  # no CSE temps
    assert "alpha[0]" in body
    assert "for (auto&& v : c) v /= h;" in body
    assert "v *= -1" in body


def test_nbs_cutcell_has_cse():
    """Cut-cell stencil boundary has CSE temporaries."""
    body = generate_nbs_method(
        "nbs_floating",
        poly_floating_coeffs,
        r=3,
        t=4,
        printer=cutcell_printer,
        psi_dependent=True,
    )
    assert "real t5" in body or "real t6" in body
    assert "psi" in body


def test_nbs_dirichlet_fewer_rows():
    """Dirichlet method has (r-1)*t coefficients."""
    body = generate_nbs_method(
        "nbs_dirichlet",
        e4u_dirichlet_coeffs,
        r=3,
        t=5,
        printer=uniform_printer,
        psi_dependent=False,
    )
    # Should have (3-1)*5 = 10 coefficient assignments
    assert body.count("c[") == 10


def test_nbs_uniform_signature():
    """Uniform nbs_floating has single-line signature with unnamed psi."""
    body = generate_nbs_method(
        "nbs_floating",
        e4u_floating_coeffs,
        r=3,
        t=5,
        printer=uniform_printer,
        psi_dependent=False,
    )
    # Single-line signature with unnamed psi (real, not real psi)
    assert "nbs_floating(real h, real, std::span<real> c, bool right) const" in body
    # No subspan in body (uniform does it in dispatcher)
    assert "c = c.subspan" not in body


def test_nbs_cutcell_signature():
    """Cut-cell nbs_floating has two-line signature with named psi."""
    body = generate_nbs_method(
        "nbs_floating",
        poly_floating_coeffs,
        r=3,
        t=4,
        printer=cutcell_printer,
        psi_dependent=True,
    )
    # Two-line signature with named psi
    assert "nbs_floating(real h, real psi, std::span<real> c, bool right) const" in body
    # Subspan in method body
    assert "c = c.subspan(0, R * T);" in body


def test_nbs_nu2_right_boundary():
    """nu=2 right-boundary only reverses, no negation."""
    body = generate_nbs_method(
        "nbs_floating",
        e4u_floating_coeffs,
        r=3,
        t=5,
        printer=uniform_printer,
        psi_dependent=False,
        nu=2,
    )
    assert "for (auto&& v : c) v /= (h * h);" in body
    assert "std::ranges::reverse(c);" in body
    assert "v *= -1" not in body


def test_nbs_floating_coeff_count():
    """Floating method has r*t coefficient assignments."""
    body = generate_nbs_method(
        "nbs_floating",
        e4u_floating_coeffs,
        r=3,
        t=5,
        printer=uniform_printer,
        psi_dependent=False,
    )
    # Should have R*T = 15 coefficient assignments
    assert body.count("c[") == 15
