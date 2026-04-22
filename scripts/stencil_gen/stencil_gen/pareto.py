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
from typing import Callable, Sequence

import numpy as np

from stencil_gen.brady2d_stability import brady2d_stability_score
from stencil_gen.optimizer import (
    _infer_max_layer,
    extract_field,
    params_from_vector,
)

# Finite sentinel used when the multi-objective evaluation is infeasible
# (gate trip, shape mismatch, or ``brady2d_stability_score`` exception).
# pymoo's hypervolume indicator and ``ftol`` termination both reject ``+inf``,
# so we substitute a large finite number.  Downstream consumers (NSGA-II
# driver, persistence layer) filter sentinel rows out of the reported front.
_PARETO_SENTINEL: float = 1e12


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


def make_multi_objective(
    scheme: str,
    kernel: str,
    report_fields: Sequence[str],
    *,
    gate_layer: int | None = None,
    max_layer: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a feasibility-gated vector-valued objective ``f(x) -> np.ndarray``.

    Parallels :func:`stencil_gen.optimizer.make_objective` but returns a
    length-``len(report_fields)`` vector instead of a scalar, for use with
    multi-objective evolutionary algorithms (NSGA-II; see :func:`run_nsga2`).

    The returned closure converts a flat vector ``x`` into a kernel-specific
    ``params`` dict, runs :func:`brady2d_stability_score` in short-circuit
    mode up to ``max_layer``, and returns:

    - ``np.full(n_obj, _PARETO_SENTINEL)`` if any layer at or before
      ``gate_layer`` failed (feasibility cliff), if ``x`` has the wrong
      shape for the kernel, or if :func:`brady2d_stability_score` raised.
    - a vector of :func:`extract_field` values (one per ``report_fields``)
      otherwise.  Individual missing fields still produce ``+inf`` from
      :func:`extract_field`; pymoo tolerates per-element ``+inf`` on a
      partially-successful evaluation, but the sentinel path keeps
      hypervolume well-defined when the whole evaluation is infeasible.

    Parameters
    ----------
    scheme, kernel
        Forwarded to :func:`brady2d_stability_score`.
    report_fields
        Sequence of dotted-path identifiers (length ≥ 2).  Each must resolve
        via :func:`_infer_max_layer` unless ``max_layer`` is supplied
        explicitly.
    gate_layer
        Highest layer whose failure forces the sentinel vector.  Defaults to
        ``max_layer - 1`` (floored at 0) — consistent with
        :func:`make_objective`.
    max_layer
        Highest layer actually executed.  Defaults to
        ``max(_infer_max_layer(f) for f in report_fields)`` so the pipeline
        runs deep enough to populate every requested field.  Raises
        ``ValueError`` if any field's layer cannot be inferred and no
        explicit ``max_layer`` is given, or if ``max_layer < gate_layer``.
    """
    fields = tuple(report_fields)
    if len(fields) < 2:
        raise ValueError(
            f"make_multi_objective requires >= 2 report_fields, got {len(fields)}"
        )

    if max_layer is None:
        inferred_layers = []
        for f in fields:
            layer = _infer_max_layer(f)
            if layer is None:
                raise ValueError(
                    f"cannot infer max_layer from report_field={f!r}; "
                    "pass max_layer explicitly"
                )
            inferred_layers.append(layer)
        max_layer = max(inferred_layers)
    if gate_layer is None:
        gate_layer = max(max_layer - 1, 0)
    if max_layer < gate_layer:
        raise ValueError(
            f"max_layer={max_layer} is less than gate_layer={gate_layer}; "
            "raise max_layer or lower gate_layer"
        )

    n_obj = len(fields)
    sentinel_vec = np.full(n_obj, _PARETO_SENTINEL, dtype=float)

    def objective(x: np.ndarray) -> np.ndarray:
        try:
            params = params_from_vector(kernel, x)
            report = brady2d_stability_score(
                scheme,
                kernel,
                params,
                max_layer=max_layer,
                short_circuit=True,
            )
        except Exception:
            return sentinel_vec.copy()
        if report.failed_layer is not None and report.failed_layer <= gate_layer:
            return sentinel_vec.copy()
        return np.array(
            [extract_field(report, f) for f in fields], dtype=float
        )

    return objective
