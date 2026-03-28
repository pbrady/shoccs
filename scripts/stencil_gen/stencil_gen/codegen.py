"""C++ code generation for stencil files.

Generates .cpp and .t.cpp files matching the patterns in src/stencils/.
"""

from sympy import Expr, Rational, Symbol, cse, numbered_symbols


def apply_cse(
    coeffs: list[Expr],
    prefix: str = "t",
    start: int = 5,
) -> tuple[list[tuple[Symbol, Expr]], list[Expr]]:
    """Apply common subexpression elimination with project conventions.

    The start=5 default matches the existing naming convention in polyE2_1.cpp
    where CSE temporaries begin at t5.

    Args:
        coeffs: List of symbolic expressions to optimize.
        prefix: Prefix for generated temporary symbols.
        start: Starting index for temporary numbering.

    Returns:
        Tuple of (replacements, reduced) where replacements is a list of
        (symbol, expression) pairs and reduced is the simplified expression list.
    """
    replacements, reduced = cse(
        coeffs,
        symbols=numbered_symbols(prefix, start=start),
    )
    return replacements, reduced


def format_rational_h_division(r: Rational, nu: int) -> str:
    """Format a rational coefficient with baked-in h-division for interior methods.

    Returns a C++ expression string. Not using StencilCodePrinter because these
    are exact Rational constants, not general symbolic expressions.
    """
    if r == 0:
        return "0"
    p, q = int(r.p), int(r.q)
    h_div = "h" if nu == 1 else "(h * h)"
    if q == 1:
        return f"{p} / {h_div}"
    return f"{p} / ({q} * {h_div})"


def generate_interior_method(
    coeffs: list,
    nu: int,
    is_uniform: bool,
) -> str:
    """Generate the body of the interior() method.

    Returns the lines between { and } of the interior method, each indented
    with 8 spaces (matching 2-level indent inside struct + method).

    Args:
        coeffs: Length 2P+1 exact Rational coefficients from interior derivation.
        nu: Derivative order (1 or 2).
        is_uniform: True for uniform stencils (bake h into expressions).
    """
    P = (len(coeffs) - 1) // 2
    lines: list[str] = []
    indent = "        "

    if is_uniform:
        # Verify symmetry/antisymmetry
        if nu == 1:
            # Antisymmetric: c[2P-k] = -c[k], c[P] = 0
            use_shorthand = all(
                coeffs[2 * P - k] == -coeffs[k] for k in range(P)
            ) and coeffs[P] == 0
        else:
            # Symmetric: c[2P-k] = c[k]
            use_shorthand = all(
                coeffs[2 * P - k] == coeffs[k] for k in range(P)
            )

        # Emit first P+1 coefficients with baked-in h
        for i in range(P + 1):
            c = coeffs[i]
            if c == 0:
                lines.append(f"{indent}c[{i}] = 0;")
            else:
                lines.append(f"{indent}c[{i}] = {format_rational_h_division(Rational(c), nu)};")

        # Emit shorthand tail or explicit assignments
        if use_shorthand:
            sign = "-" if nu == 1 else ""
            for i in range(P + 1, 2 * P + 1):
                mirror = 2 * P - i
                lines.append(f"{indent}c[{i}] = {sign}c[{mirror}];")
        else:
            # Fallback: explicit assignments for remaining coefficients
            for i in range(P + 1, 2 * P + 1):
                c = coeffs[i]
                if c == 0:
                    lines.append(f"{indent}c[{i}] = 0;")
                else:
                    lines.append(f"{indent}c[{i}] = {format_rational_h_division(Rational(c), nu)};")

        lines.append("")
        lines.append(f"{indent}return c.subspan(0, 2 * P + 1);")
    else:
        # Cut-cell: loop-based h division
        lines.append(f"{indent}c = c.subspan(0, 2 * {P} + 1);")
        for i in range(2 * P + 1):
            c = coeffs[i]
            if c == 0:
                lines.append(f"{indent}c[{i}] = 0;")
            else:
                # Print as float literal with 17 significant digits
                val = float(c)
                # Use repr-style formatting but clean up
                formatted = f"{val:.17g}"
                lines.append(f"{indent}c[{i}] = {formatted};")

        if nu == 1:
            lines.append(f"{indent}for (auto&& v : c) v /= h;")
        else:
            lines.append(f"{indent}for (auto&& v : c) v /= (h * h);")
        lines.append(f"{indent}return c;")

    return "\n".join(lines)
