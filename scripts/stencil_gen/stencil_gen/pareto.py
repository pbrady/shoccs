"""Multi-objective Pareto optimization wrapper around the Brady-Livescu 2D
stability pipeline (plan 45).

Where :mod:`stencil_gen.optimizer` drives *scalar* objectives (a single dotted
path into :class:`StabilityReport`), this module targets the multi-objective
case: minimise a vector ``F(x) = [f_1(x), ..., f_m(x)]`` so the population
converges toward the *Pareto front* — the set of non-dominated parameter
vectors where no axis can be improved without worsening another.

Pareto-dominance: ``x`` dominates ``y`` iff ``F_i(x) <= F_i(y)`` for all ``i``
and strictly ``<`` for at least one.  The front is the subset of evaluated
points not dominated by any other.

References
----------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). "A fast and elitist
multiobjective genetic algorithm: NSGA-II." *IEEE Transactions on Evolutionary
Computation*, 6(2), 182-197.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParetoPoint:
    """A single non-dominated member of a Pareto front.

    Attributes
    ----------
    x : np.ndarray
        Flat parameter vector of shape ``(n_var,)``, dtype ``float64``.  The
        optimiser's native representation — convert to a kernel-specific
        ``params`` dict via :func:`stencil_gen.optimizer.params_from_vector`.
    params : dict
        Kernel-specific parameter dict at ``x`` (the thing you would pass to
        :func:`brady2d_stability_score`).  Redundant with ``x`` but included
        so downstream code can consume either representation without carrying
        the kernel through.
    objectives : np.ndarray
        Vector of objective values, shape ``(n_obj,)``, dtype ``float64``.
        Aligned with :attr:`ParetoResult.objective_fields`.
    report : dict
        Serialised :class:`StabilityReport` at ``x`` (produced by
        ``_report_to_dict`` from :mod:`stencil_gen.optimizer`).  Empty dict if
        the evaluation produced no feasible report.
    """

    x: np.ndarray
    params: dict
    objectives: np.ndarray
    report: dict


@dataclass(frozen=True)
class ParetoResult:
    """Frozen record of a single multi-objective optimiser run.

    Attributes
    ----------
    front : tuple[ParetoPoint, ...]
        Non-dominated members at the end of the run.  Empty tuple if the run
        produced no feasible point.
    objective_fields : tuple[str, ...]
        Dotted-path identifiers matching :attr:`ParetoPoint.objectives`, e.g.
        ``("layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa")``.
    scheme : str
        Scheme identifier forwarded to :func:`brady2d_stability_score`.
    kernel : str
        Kernel identifier forwarded to :func:`brady2d_stability_score`.
    bounds : tuple[tuple[float, float], ...]
        Parameter bounds used for the run, one ``(lo, hi)`` pair per variable.
    method : str
        Name of the driver (``"NSGA-II"`` for plan 45; ``"NSGA-III"`` reserved
        for a future extension).
    pop_size : int
        Population size used by the evolutionary algorithm.
    n_gen : int
        Number of generations executed.
    n_evals : int
        Total number of objective evaluations across the run.
    seed : int
        RNG seed supplied to the algorithm.
    compute_time : float
        Wall-clock seconds for the run.
    hv_trace : tuple[float, ...]
        Hypervolume of the current non-dominated set at the end of each
        generation.  Length equals ``n_gen``.
    ref_point : tuple[float, ...]
        Reference point used for the hypervolume indicator, aligned with
        :attr:`objective_fields`.
    extras : dict
        Free-form additional fields (e.g. ``n_sentinel_filtered``,
        ``cpp_validation``, driver-specific diagnostics).
    """

    front: tuple[ParetoPoint, ...]
    objective_fields: tuple[str, ...]
    scheme: str
    kernel: str
    bounds: tuple[tuple[float, float], ...]
    method: str
    pop_size: int
    n_gen: int
    n_evals: int
    seed: int
    compute_time: float
    hv_trace: tuple[float, ...]
    ref_point: tuple[float, ...]
    extras: dict
