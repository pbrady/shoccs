"""pyshoccs: the Python array arm of the SHOCCS phase-1 thin waist.

The package consumes an export directory written by ``shoccs-export`` (data
contract: ``docs/reference/pyshoccs-export.md``) and provides

- ``tables``: the loader and the exact operator (SciPy CSR, raw COO apply),
- ``structure``: the role bits, interior masks and exception CSR of spec Section 4,
- ``mms``: the gaussian manufactured solution,
- ``heat``: the heat system (source, RHS, boundary update, stats, RK4 step),
- ``oracle``: the monomial oracle for R rows and closure rows,
- ``parity``: the five checks of spec Section 3.
"""

from .tables import Tables, Layout, load_tables, default_export_dir
from .structure import Structure, derive_structure, apply_structured
from .mms import Gaussian
from .heat import HeatSystem

__all__ = [
    "Tables",
    "Layout",
    "load_tables",
    "default_export_dir",
    "Structure",
    "derive_structure",
    "apply_structured",
    "Gaussian",
    "HeatSystem",
]
