"""Multi-fidelity Bayesian optimization over the cascade.

Implements plan 47: replace the hand-coded ``run_staged_optimize`` cheap-inner
+ expensive-validator heuristic with a principled multi-fidelity Bayesian
optimizer that uses a Gaussian-process surrogate over the cascade's discrete
fidelity levels and a cost-aware acquisition function.

Algorithm
---------

The optimizer chooses ``(x, m)`` jointly to maximize expected information gain
at the high-fidelity target per second of wall time.  The GP surrogate uses an
Intrinsic Coregionalization Model (ICM) kernel to learn correlations between
cascade layers from data — necessary because the cascade's L3 ↔ L3r pair tests
different physics (1D periodic advection vs. reflecting BCs), so a single
Kennedy-O'Hagan autoregressive ladder is inappropriate (see
``docs/handoff/scientific_findings.md`` finding #1).

References
----------

- Wu, J., Toscano-Palmerin, S., Frazier, P. I., & Wilson, A. G. (2020).
  *Practical Multi-fidelity Bayesian Optimization for Hyperparameter Tuning*.
  https://arxiv.org/abs/1903.04703
- BoTorch tutorial: discrete multi-fidelity BO.
  https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/
- BoTorch tutorial: cost-aware Bayesian optimization.
  https://botorch.org/docs/tutorials/cost_aware_bayesian_optimization/

This is a skeleton module — the dataclasses, factory, GP, cost model, DOE,
acquisition, and ``run_mfbo`` driver are added in subsequent items of plan 47
(47.1 onward).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch  # noqa: F401  # used in subsequent items
import botorch  # noqa: F401  # used in subsequent items

from stencil_gen.optimizer import (  # noqa: F401  # reused in subsequent items
    DEFAULT_BOUNDS,
    _FIELD_LAYER_ALIAS,
    _infer_max_layer,
    _report_to_dict,
    extract_field,
    params_from_vector,
    vector_from_params,
)


# Finite sentinel used when a multi-fidelity evaluation is infeasible (gate
# trip, shape mismatch, or ``brady2d_stability_score`` exception).  qMFKG /
# MES break on ``+inf`` (Cholesky failure during fantasy sampling); a large
# finite value keeps the GP fit well-conditioned.  Sentinel rows are filtered
# out of the training tensors before each GP refit.
_BO_SENTINEL: float = 1e12


@dataclass(frozen=True)
class BOEval:
    """A single multi-fidelity evaluation record.

    One row of the BO loop's evaluation history: the design vector ``x``, the
    fidelity ``m`` it was evaluated at, the resulting objective value, the
    measured wall time (for empirical cost calibration), and the serialised
    :class:`StabilityReport`.

    Attributes
    ----------
    x : np.ndarray
        Flat design vector of shape ``(d,)``, dtype ``float64``.  The
        optimiser's native representation — convert to a kernel-specific
        ``params`` dict via :func:`stencil_gen.optimizer.params_from_vector`.
    params : dict
        Kernel-specific parameter dict at ``x``.  Redundant with ``x`` but
        included so downstream code can consume either representation without
        carrying the kernel through.
    fidelity : int
        Cascade layer index this evaluation ran at (e.g. ``1``, ``3``, ``7``).
        This is the *external* layer number used by
        :func:`brady2d_stability_score`, not the internal contiguous fidelity
        index used by the GP/acquisition.
    value : float
        Extracted objective value at this fidelity.  Equals
        :data:`_BO_SENTINEL` if the evaluation was infeasible.
    wall_time : float
        Measured per-eval seconds (``time.perf_counter`` delta around the
        :func:`brady2d_stability_score` call).  Always positive, even on the
        sentinel path.
    report : dict
        Serialised :class:`StabilityReport` (produced by ``_report_to_dict``).
        Contains ``{"error": str(exc)}`` if the evaluation produced no
        feasible report.
    """

    x: np.ndarray
    params: dict
    fidelity: int
    value: float
    wall_time: float
    report: dict


@dataclass(frozen=True)
class BOResult:
    """Frozen record of a single multi-fidelity Bayesian optimisation run.

    Mirrors :class:`stencil_gen.pareto.ParetoResult` in spirit: an immutable
    summary plus enough raw data (full eval history, GP hyperparameters, cost
    table) to reproduce the run's recommendation off-line.

    Attributes
    ----------
    best_x : np.ndarray
        Recommended design vector at the high-fidelity target.  Selected via
        ``argmin_x μ_n(x, m=hf)`` on a Sobol' grid (posterior mean — standard
        for noisy / multi-fidelity GPs), then re-evaluated at HF to populate
        :attr:`best_objective` from real data.
    best_params : dict
        Kernel-specific parameter dict at :attr:`best_x`.
    best_objective : float
        HF objective value at :attr:`best_x` from a final real evaluation
        (NOT the GP posterior mean, which can disagree under model misspec).
    best_report : dict
        Full serialised :class:`StabilityReport` at :attr:`best_x` at HF.
    method : str
        Driver name, e.g. ``"BoTorch-qMFKG"`` (or fallback name like
        ``"BoTorch-qMFMES"`` if KG diagnostics show degeneracy).
    scheme : str
        Scheme identifier forwarded to :func:`brady2d_stability_score`.
    kernel : str
        Kernel identifier forwarded to :func:`brady2d_stability_score`.
    bounds : tuple[tuple[float, float], ...]
        Parameter bounds used for the run, one ``(lo, hi)`` pair per variable.
    fidelity_levels : tuple[int, ...]
        Sorted external layer indices in ascending cost order, e.g.
        ``(1, 3, 7)``.  Sorted so ``[-1]`` is always the HF level.
    hf_level : int
        ``max(fidelity_levels)``.  The optimiser's target.
    report_fields_by_layer : dict[int, str]
        Mapping ``layer index → dotted path``, e.g.
        ``{1: "layer1.boundary_gv_err", 7: "layer7.max_spectral_abscissa"}``.
        The HF layer's field is the optimisation target.
    cost_model : dict[int, float]
        The actual cost table used (with floor applied).  Keyed by external
        layer index, values in seconds.
    n_evals_per_fidelity : dict[int, int]
        Count of evaluations at each fidelity (initial design + acquisition
        steps + final HF re-evaluation at ``best_x``).  Keys match
        :attr:`fidelity_levels`.
    wall_time_per_fidelity : dict[int, float]
        Cumulative measured wall time at each fidelity, in seconds.
    total_compute_time : float
        Total wall-clock seconds for the run (init + GP fits + acquisition
        optimisation + objective evaluations + final HF re-evaluation).
    eval_history : tuple[BOEval, ...]
        Full per-eval log, in chronological order.  Length equals the total
        number of evaluations.
    hf_eval_history : tuple[BOEval, ...]
        Filter of :attr:`eval_history` to ``fidelity == hf_level`` only.
        Used to produce the convergence trace ``best_observed_hf_so_far``.
    gp_hyperparameters : dict
        Final GP state at convergence: lengthscale, outputscale, noise, and
        the ICM ``B = W Wᵀ + diag(κ)`` coregionalization matrix (extracted
        from ``model.covar_module.state_dict()``).  Empty dict if the GP
        never fit (e.g. all initial evals returned sentinel).
    seed : int
        RNG seed supplied to :func:`run_mfbo`.  Setting the same seed
        reproduces :attr:`best_x` to within ``1e-6``.
    converged : bool
        ``True`` if the run terminated by variance / stagnation guard;
        ``False`` if it hit budget.  Always ``False`` on error termination.
    stop_reason : str
        One of ``"budget"``, ``"variance"``, ``"stagnation"``, ``"error"``.
    extras : dict
        Free-form additional fields (e.g. ``n_sentinel_filtered``,
        ``baseline`` :class:`OptimizeResult`, ``cpp_validation`` payload).
    """

    best_x: np.ndarray
    best_params: dict
    best_objective: float
    best_report: dict
    method: str
    scheme: str
    kernel: str
    bounds: tuple[tuple[float, float], ...]
    fidelity_levels: tuple[int, ...]
    hf_level: int
    report_fields_by_layer: dict[int, str]
    cost_model: dict[int, float]
    n_evals_per_fidelity: dict[int, int]
    wall_time_per_fidelity: dict[int, float]
    total_compute_time: float
    eval_history: tuple[BOEval, ...]
    hf_eval_history: tuple[BOEval, ...]
    gp_hyperparameters: dict
    seed: int
    converged: bool
    stop_reason: str
    extras: dict


__all__: list[str] = [
    "_BO_SENTINEL",
    "BOEval",
    "BOResult",
]
