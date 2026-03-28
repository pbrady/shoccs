"""TEMO (Truncation Error Matching Optimization) cut-cell stencil extension.

Implements the TEMO procedure from Brady & Livescu (2021) for deriving
psi-parameterized cut-cell boundary stencils from uniform boundary stencils.
"""

from dataclasses import dataclass
from typing import NamedTuple


class Dimensions(NamedTuple):
    """Stencil dimensions for uniform and cut-cell cases."""

    r: int  # uniform boundary rows
    t: int  # uniform boundary columns
    R: int  # cut-cell rows
    T: int  # cut-cell columns (including wall)
    X: int  # Neumann extra rows


def compute_dimensions(p: int, q: int, s: int, nextra: int, nu: int) -> Dimensions:
    """Compute stencil dimensions from scheme parameters.

    Uses Eq. 11a/11b from Brady & Livescu (2021):
        t = p + q + 1 + nextra     (stencil width)
        r = q + 1 + nextra         (number of boundary rows)

    For cut-cell stencils:
        R = r_eff + 1, T = t + 1

    where r_eff = r for 1st derivatives, r_eff = r - 1 for 2nd derivatives
    (the last uniform boundary row overlaps with the first interior row).

    Note: verified for E2 schemes only. E4 schemes may require a different
    derivation (see D-R25 in meta.md).

    Parameters
    ----------
    p : int
        Interior half-width (RHS bandwidth).
    q : int
        Boundary accuracy order.
    s : int
        LHS half-width (0 for explicit, 1 for tridiagonal compact).
    nextra : int
        Extra rows/columns for numerical optimization.
    nu : int
        Derivative order (1 or 2).

    Returns
    -------
    Dimensions
        Named tuple (r, t, R, T, X).
    """
    t = p + q + 1 + nextra
    r = q + 1 + nextra

    if nu == 1:
        r_eff = r
    elif nu == 2:
        r_eff = r - 1
    else:
        raise ValueError(f"Unsupported derivative order nu={nu}; must be 1 or 2")

    R = r_eff + 1
    T = t + 1
    X = R if nu == 2 else 0

    return Dimensions(r=r, t=t, R=R, T=T, X=X)


@dataclass(frozen=True)
class SchemeParams:
    """Scheme parameters from Table 1 of Brady & Livescu (2021).

    Parameters
    ----------
    p : int
        Interior half-width (RHS bandwidth).
    q : int
        Boundary accuracy order.
    s : int
        LHS half-width (0 for explicit, 1 for tridiagonal compact).
    nextra : int
        Extra rows/columns for numerical optimization.
    nu : int
        Derivative order (1 or 2).
    """

    p: int
    q: int
    s: int
    nextra: int
    nu: int

    def dims(self) -> Dimensions:
        """Compute stencil dimensions for this scheme."""
        return compute_dimensions(self.p, self.q, self.s, self.nextra, self.nu)


# Pre-defined schemes from Table 1
E2_1 = SchemeParams(p=1, q=1, s=0, nextra=1, nu=1)
E2_2 = SchemeParams(p=1, q=1, s=0, nextra=0, nu=2)
E4_1 = SchemeParams(p=2, q=3, s=0, nextra=0, nu=1)
E4_2 = SchemeParams(p=2, q=3, s=0, nextra=0, nu=2)
