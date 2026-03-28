"""Unit tests for the custom C++ stencil code printer."""

from sympy import Integer, Pow, Rational, Symbol, symbols

from stencil_gen.printer import StencilCodePrinter, build_symbol_map


def test_pow_reciprocal():
    p = StencilCodePrinter()
    x = Symbol("x")
    assert p.doprint(Pow(1 + x, -1)) == "1.0 / (x + 1)"


def test_pow_square():
    p = StencilCodePrinter()
    x = Symbol("x")
    assert p.doprint(Pow(x, 2)) == "x * x"


def test_pow_square_compound():
    """Compound base gets parenthesized."""
    p = StencilCodePrinter()
    x, y = symbols("x y")
    assert p.doprint(Pow(x + y, 2)) == "(x + y) * (x + y)"


def test_pow_cube():
    p = StencilCodePrinter()
    x = Symbol("x")
    assert p.doprint(Pow(x, 3)) == "x * x * x"


def test_pow_high():
    p = StencilCodePrinter()
    x = Symbol("x")
    assert p.doprint(Pow(x, 5)) == "std::pow(x, 5)"


def test_pow_neg2():
    p = StencilCodePrinter()
    x = Symbol("x")
    assert p.doprint(Pow(x, -2)) == "1.0 / (x * x)"


def test_rational():
    p = StencilCodePrinter()
    assert p.doprint(Rational(1, 12)) == "1.0 / 12"
    assert p.doprint(Rational(2, 3)) == "2.0 / 3"
    assert p.doprint(Rational(-5, 6)) == "-5.0 / 6"


def test_integer():
    p = StencilCodePrinter()
    assert p.doprint(Integer(3)) == "3"
    assert p.doprint(Rational(3, 1)) == "3"
    assert p.doprint(Integer(-1)) == "-1"


def test_symbol_map():
    smap = build_symbol_map({"alpha": 2}, has_psi=True)
    p = StencilCodePrinter(symbol_map=smap)
    alpha_0 = Symbol("alpha_0")
    psi = Symbol("psi")
    assert p.doprint(alpha_0) == "alpha[0]"
    assert p.doprint(psi) == "psi"


def test_symbol_unmapped():
    """CSE temporaries not in the map print as their name."""
    p = StencilCodePrinter(symbol_map={})
    t5 = Symbol("t5")
    assert p.doprint(t5) == "t5"
