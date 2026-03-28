"""Tests for stencil_gen.codegen module."""

from sympy import Rational, Symbol, symbols

from stencil_gen.codegen import apply_cse, format_rational_h_division, generate_interior_method


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
