"""Taylor system builder for boundary stencil derivation.

Constructs the Vandermonde-like linear system from Taylor expansion matching
for a single boundary row of an SBP finite difference operator.
"""

from sympy import Matrix, Rational, factorial


def build_taylor_system(
    i: int,
    t: int,
    q: int,
    nu: int = 1,
) -> tuple[Matrix, Matrix]:
    """Build the Taylor matching system for boundary row i.

    Parameters
    ----------
    i : int
        Row index (0..r-1), the grid point being approximated.
    t : int
        Boundary stencil width (number of columns).
    q : int
        Polynomial order of the boundary scheme.
    nu : int
        Derivative order (1 for first derivative).

    Returns
    -------
    (V, rhs) where V is (q+1) x t and rhs is (q+1) x 1.
    """
    n_rows = q + 1
    V = Matrix.zeros(n_rows, t)
    rhs = Matrix.zeros(n_rows, 1)

    for k in range(n_rows):
        inv_fact = Rational(1, factorial(k))
        for j in range(t):
            V[k, j] = Rational((j - i) ** k) * inv_fact
        rhs[k] = Rational(1) if k == nu else Rational(0)

    return V, rhs
