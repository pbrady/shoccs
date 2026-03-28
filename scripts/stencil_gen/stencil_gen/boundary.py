"""Single-row boundary stencil solver.

Solves the underdetermined Taylor system for a single boundary row,
expressing coefficients as symbolic functions of free alpha parameters.
"""

from dataclasses import dataclass

from sympy import Expr, Matrix, Symbol, cancel

from stencil_gen.taylor_system import build_taylor_system


@dataclass
class BoundaryRow:
    """Result of solving one boundary row's Taylor system."""

    row_index: int  # i (0..r-1)
    coefficients: list[Expr]  # length t, each is Expr in alpha symbols
    free_params: list[Symbol]  # the alpha symbols for this row


def solve_boundary_row(
    i: int,
    t: int,
    q: int,
    nu: int,
    free_symbols: list,
) -> BoundaryRow:
    """Solve the Taylor system for boundary row i.

    Parameters
    ----------
    i : int
        Row index (0..r-1).
    t : int
        Boundary stencil width.
    q : int
        Polynomial order of boundary scheme.
    nu : int
        Derivative order.
    free_symbols : list
        Pre-created alpha symbols (or S.Zero) for the free columns.
        Length must equal t - (q + 1).

    Returns
    -------
    BoundaryRow with coefficients as symbolic expressions of the free params.
    """
    V, rhs = build_taylor_system(i, t, q, nu)

    n_det = q + 1
    n_free = t - n_det
    assert len(free_symbols) == n_free

    # Partition: first n_det columns are determined, last n_free are free
    V_det = V[:, :n_det]
    V_free = V[:, n_det:]

    alpha_vec = Matrix(free_symbols)
    rhs_adjusted = rhs - V_free * alpha_vec

    gamma_det = V_det.solve(rhs_adjusted)

    coefficients = [cancel(gamma_det[k]) for k in range(n_det)] + list(free_symbols)

    return BoundaryRow(
        row_index=i,
        coefficients=coefficients,
        free_params=[s for s in free_symbols if isinstance(s, Symbol)],
    )
