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

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import botorch  # noqa: F401  # used in subsequent items
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.transforms import Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from stencil_gen.brady2d_stability import brady2d_stability_score
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


# --- multi-fidelity objective factory ----------------------------------------


def make_multi_fidelity_objective(
    scheme: str,
    kernel: str,
    report_fields_by_layer: dict[int, str],
    *,
    gate_layer: int | None = None,
) -> Callable[[np.ndarray, int], tuple[float, float, dict]]:
    """Build a multi-fidelity objective ``f(x, m) -> (value, wall_time, report)``.

    Mirrors :func:`stencil_gen.optimizer.make_objective` but routes through a
    per-fidelity field selection and returns the wall-time + serialised report
    alongside the scalar value, so the BO loop can record per-eval cost without
    a side channel.

    Parameters
    ----------
    scheme, kernel
        Forwarded to :func:`brady2d_stability_score`.
    report_fields_by_layer
        Mapping ``{layer_index: dotted_field_path}``.  ``max(...)`` is the HF
        target; the HF field is the optimisation objective.  Cheaper layers'
        fields are surrogates that the GP correlates with the HF objective via
        the ICM coregionalization matrix.
    gate_layer
        Highest layer whose failure forces the sentinel value.  Defaults to
        ``max(min(layers) - 1, 0)`` — only layers strictly *cheaper* than the
        cheapest fidelity in ``report_fields_by_layer`` gate; the cheapest
        fidelity itself is always a usable result.  Pass ``0`` to disable
        gating entirely.

    Returns
    -------
    Callable[[np.ndarray, int], tuple[float, float, dict]]
        Closure ``f(x, m)``.  On any of:

        - ``m`` not in ``report_fields_by_layer``,
        - shape mismatch in :func:`params_from_vector`,
        - exception from :func:`brady2d_stability_score`,
        - gate trip (failed layer ≤ ``gate_layer``),

        returns ``(_BO_SENTINEL, measured_wall_time, {"error": str(...)})``.
        On success, returns
        ``(extract_field(report, field_at_m), wall_time, _report_to_dict(report))``.

    Raises
    ------
    ValueError
        At factory time, if any field's :func:`_infer_max_layer` exceeds the
        layer it is keyed under (you cannot extract ``layer7.*`` from an
        ``m=3`` run).  Also raised when ``report_fields_by_layer`` is empty.
    """
    if not report_fields_by_layer:
        raise ValueError("report_fields_by_layer must not be empty")
    for layer, field in report_fields_by_layer.items():
        inferred = _infer_max_layer(field)
        if inferred is not None and inferred > layer:
            raise ValueError(
                f"field {field!r} requires max_layer={inferred} but is keyed "
                f"under layer={layer}; cannot extract a field from a layer "
                "that is not run"
            )
    layers_sorted = sorted(report_fields_by_layer)
    if gate_layer is None:
        gate_layer = max(layers_sorted[0] - 1, 0)

    def objective(x: np.ndarray, m: int) -> tuple[float, float, dict]:
        if m not in report_fields_by_layer:
            return (
                _BO_SENTINEL,
                0.0,
                {"error": f"unknown fidelity m={m}"},
            )
        t0 = time.perf_counter()
        try:
            params = params_from_vector(kernel, x)
            report = brady2d_stability_score(
                scheme,
                kernel,
                params,
                max_layer=m,
                short_circuit=True,
            )
        except Exception as exc:
            return (
                _BO_SENTINEL,
                time.perf_counter() - t0,
                {"error": str(exc)},
            )
        wall_time = time.perf_counter() - t0
        if (
            report.failed_layer is not None
            and report.failed_layer <= gate_layer
        ):
            return (
                _BO_SENTINEL,
                wall_time,
                _report_to_dict(report),
            )
        value = extract_field(report, report_fields_by_layer[m])
        if not np.isfinite(value):
            return (
                _BO_SENTINEL,
                wall_time,
                _report_to_dict(report),
            )
        return (float(value), wall_time, _report_to_dict(report))

    return objective


# --- multi-fidelity GP surrogate ---------------------------------------------


def build_mf_gp(
    train_X: np.ndarray | torch.Tensor,
    train_Y: np.ndarray | torch.Tensor,
    fidelity_dim: int,
    num_fidelities: int,
    *,
    rank: int = 2,
) -> MultiTaskGP:
    """Build and fit an ICM-style multi-fidelity GP surrogate.

    The surrogate is a :class:`MultiTaskGP` — BoTorch's purpose-built ICM
    model — with a Matern-5/2 ARD data kernel and an Intrinsic
    Coregionalization Model (ICM) parameterisation of the layer-pair
    covariance ``B = W Wᵀ + diag(κ)``, with ``W`` of shape
    ``(num_fidelities, rank)``, learned end-to-end via marginal-likelihood
    optimisation.  This lets the data report the actual layer-pair
    correlations rather than baking in a Kennedy-O'Hagan refinement chain
    that the cascade does not satisfy (L3 ↔ L3r test different physics —
    see ``docs/handoff/scientific_findings.md`` finding #1).

    Parameters
    ----------
    train_X
        Training inputs of shape ``(N, d + 1)`` where the column at
        ``fidelity_dim`` holds integer-valued fidelity indices in
        ``{0, ..., num_fidelities - 1}``.  Accepts NumPy or torch; converted
        to ``torch.float64`` internally.
    train_Y
        Training targets of shape ``(N,)`` or ``(N, 1)``.  Sentinel rows must
        be filtered upstream — the GP only fits on finite-value rows.
    fidelity_dim
        Column index of the fidelity feature in ``train_X``.  Conventionally
        the last column (i.e. ``train_X.shape[-1] - 1``).
    num_fidelities
        Number of distinct fidelity levels (informational; the actual task
        count is inferred from values present in the fidelity column).
    rank
        Rank of the ICM coregionalization factor ``W``.  ``rank=2`` is a good
        default for 3–5 fidelities — large enough to capture non-trivial
        layer-pair correlations, small enough to remain identifiable from
        modest training data.

    Returns
    -------
    MultiTaskGP
        Fitted GP.  Hyperparameters can be inspected via:

        - ``model.covar_module.kernels[0].lengthscale`` (Matern ARD).
        - ``model.covar_module.kernels[1].covar_factor`` (W of the ICM
          matrix).
        - ``model.covar_module.kernels[1].var`` (diagonal κ of the ICM
          matrix).
        - ``model.likelihood.noise``.

    Notes
    -----
    The plan body cited :class:`SingleTaskMultiFidelityGP` as the wrapper,
    but that class always composes the user-supplied ``covar_module`` with a
    fixed :class:`LinearTruncatedFidelityKernel` (the AR1 kernel we
    explicitly want to avoid) or :class:`ExponentialDecayKernel`, so a
    custom ICM kernel is not respected.  Hand-composing ``MaternKernel *
    IndexKernel`` on a regular :class:`SingleTaskGP` worked but proved
    fragile: ``fit_gpytorch_mll`` failed on ~70% of small noise-free
    datasets due to NotPSDError during Cholesky factorisation.
    :class:`MultiTaskGP` is BoTorch's purpose-built ICM model — same kernel
    structure (Matern-on-data × IndexKernel-on-task), but with engineered
    parameter initialisation and PSD-stable parameterisation that fits
    reliably.  The MF-aware ``project`` helper that
    :class:`SingleTaskMultiFidelityGP` adds beyond a plain GP is supplied
    directly to ``qMultiFidelityKnowledgeGradient`` in 47.3a.

    Outputs are standardised via :class:`Standardize`; without it the
    marginal-likelihood optimiser fails on raw cascade scales
    (``max_stab_eig`` is ~1e-12 while ``boundary_gv_err`` is ~1e-2 — five
    orders of magnitude apart).
    """
    if num_fidelities < 1:
        raise ValueError(f"num_fidelities must be ≥ 1, got {num_fidelities}")
    if rank < 1:
        raise ValueError(f"rank must be ≥ 1, got {rank}")

    X = torch.as_tensor(train_X, dtype=torch.float64)
    Y = torch.as_tensor(train_Y, dtype=torch.float64)
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    if X.ndim != 2:
        raise ValueError(f"train_X must be 2D, got shape {tuple(X.shape)}")
    if Y.shape[0] != X.shape[0]:
        raise ValueError(
            f"train_X has {X.shape[0]} rows but train_Y has {Y.shape[0]}"
        )
    n_cols = X.shape[-1]
    if not (0 <= fidelity_dim < n_cols):
        raise ValueError(
            f"fidelity_dim={fidelity_dim} out of range for train_X with "
            f"{n_cols} columns"
        )

    n_data_dims = n_cols - 1
    data_kernel = MaternKernel(nu=2.5, ard_num_dims=n_data_dims)

    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-9))

    model = MultiTaskGP(
        train_X=X,
        train_Y=Y,
        task_feature=fidelity_dim,
        covar_module=data_kernel,
        likelihood=likelihood,
        rank=rank,
        all_tasks=list(range(num_fidelities)),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# --- cost model + cost-aware utility -----------------------------------------


# Default per-layer wall-time costs (seconds) from plan 46 measurements.  Keys
# are *external* cascade layer indices; the contiguous internal fidelity index
# 0..K-1 used by the GP/acquisition is derived by sorting the keys ascending.
# L3r is keyed at external index 5 by plan 47.4a convention (it sits between
# L3=3 and L6=6 in cost) — even though it shares ``max_layer=3`` with L3
# inside :func:`brady2d_stability_score`, the BO module treats it as a
# distinct fidelity so the ICM kernel can learn an L3-vs-L3r task correlation.
# The CLI (47.4a) translates this synthetic ``5`` to the ``layer_bl42`` field
# name when invoking the cascade.
DEFAULT_COST_TABLE: dict[int, float] = {
    1: 0.076,  # L1: GV dispersion (interior + boundary)
    3: 0.038,  # L3: 1D advection eigenvalue
    5: 0.486,  # L3r: BL §4.2 reflecting-hyperbolic spectrum
    6: 0.846,  # L6: non-normality on 1D operator
    7: 1.434,  # L7: full 2D varying-coefficient spectral abscissa
}


# Cost floor as a fraction of the most expensive layer's cost.  Caps the
# acquisition's preference for the cheapest layer when the cost ratio is so
# extreme (here ``c(L7)/c(L3) ≈ 38``) that the cost-aware utility would
# otherwise keep querying the cheapest layer indefinitely, even after the GP
# has learned that layer is uncorrelated with HF.
_DEFAULT_COST_FLOOR_RATIO: float = 0.05


def apply_cost_floor(
    cost_table: dict[int, float],
    *,
    floor_ratio: float = _DEFAULT_COST_FLOOR_RATIO,
) -> dict[int, float]:
    """Return a copy of *cost_table* with a per-entry cost floor applied.

    For each entry ``c(m)``, the floored cost is
    ``max(c(m), floor_ratio * max_n c(n))`` — any layer whose cost is below
    ``floor_ratio`` of the most expensive layer is lifted to that floor.
    Prevents qMFKG from over-exploiting the cheapest layer; see Wu et al. 2020
    §4.2 for the cost-weighted KG formulation that motivates the floor.

    Parameters
    ----------
    cost_table
        Mapping ``layer index → cost (seconds)``.  Caller's choice of layer
        indices; the function does not interpret them.
    floor_ratio
        Per-entry floor as a fraction of the most expensive layer's cost.
        Pass ``0.0`` to disable (not recommended).

    Returns
    -------
    dict[int, float]
        New dict with the same keys as *cost_table* and floored values.

    Raises
    ------
    ValueError
        If *cost_table* is empty or *floor_ratio* is negative.
    """
    if not cost_table:
        raise ValueError("cost_table must not be empty")
    if floor_ratio < 0:
        raise ValueError(f"floor_ratio must be ≥ 0, got {floor_ratio}")
    hf_cost = max(cost_table.values())
    floor = floor_ratio * hf_cost
    return {layer: max(cost, floor) for layer, cost in cost_table.items()}


def build_cost_model(
    cost_table: dict[int, float],
    fidelity_dim: int,
    *,
    floor_ratio: float = _DEFAULT_COST_FLOOR_RATIO,
) -> InverseCostWeightedUtility:
    """Build the inverse-cost-weighted utility for cost-aware MF acquisition.

    Wraps a step-function deterministic cost model in
    :class:`InverseCostWeightedUtility` so qMFKG (47.3a) weights expected
    information gain by ``1 / cost(m)``.  The deterministic model reads the
    *internal* contiguous fidelity index (integer-rounded) from column
    ``fidelity_dim`` of its input tensor and looks up the corresponding cost
    in a floored copy of *cost_table*.

    Parameters
    ----------
    cost_table
        Mapping ``external layer index → cost (seconds)``.  Sorted ascending
        to derive the internal contiguous index ``0..K-1``: e.g. for keys
        ``{1, 3, 5, 6, 7}``, internal index ``0`` ↔ layer 1, ``4`` ↔ layer 7.
    fidelity_dim
        Column index of the fidelity feature in the acquisition's ``X``
        tensor.  Conventionally the last column (``train_X.shape[-1] - 1``).
    floor_ratio
        Forwarded to :func:`apply_cost_floor`.  Default ``0.05``.

    Returns
    -------
    InverseCostWeightedUtility
        Utility with ``use_mean=True`` (the default; the deterministic cost
        model has no posterior variance, so the choice is moot — but ``True``
        matches the BoTorch discrete-MF tutorial).

    Raises
    ------
    ValueError
        If *cost_table* is empty, *floor_ratio* is negative, or
        *fidelity_dim* is negative.
    """
    if fidelity_dim < 0:
        raise ValueError(f"fidelity_dim must be ≥ 0, got {fidelity_dim}")
    floored = apply_cost_floor(cost_table, floor_ratio=floor_ratio)
    sorted_layers = sorted(floored)
    cost_lookup = torch.tensor(
        [floored[layer] for layer in sorted_layers],
        dtype=torch.float64,
    )
    n_layers = len(sorted_layers)

    def cost_fn(X: torch.Tensor) -> torch.Tensor:
        # ``X`` has shape ``(..., d + 1)``; the fidelity column holds integer
        # internal indices.  Round and clamp before lookup to defend against
        # NaN or out-of-range values from the acquisition optimiser.
        fid = X[..., fidelity_dim].round().long().clamp(0, n_layers - 1)
        lookup = cost_lookup.to(dtype=X.dtype, device=X.device)
        return lookup[fid].unsqueeze(-1)

    cost_model = GenericDeterministicModel(f=cost_fn, num_outputs=1)
    return InverseCostWeightedUtility(cost_model=cost_model, use_mean=True)


# --- initial design (DOE) ----------------------------------------------------


def build_initial_design(
    bounds: Sequence[tuple[float, float]],
    fidelity_levels: Sequence[int],
    *,
    n_init: int | None = None,
    hf_anchors: int = 3,
    mid_anchors: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a stratified Sobol' initial design for the BO loop.

    The DOE has three goals: (i) cover the design space well in the cheap
    fidelity (where evaluations are nearly free), (ii) seed enough HF data
    that the GP's posterior at the target fidelity is informative from
    iteration 0, and (iii) provide *paired* HF/cheap evaluations at the same
    ``x`` so the ICM coregionalization matrix ``B = W Wᵀ + diag(κ)`` is
    identifiable from data.  Without paired evaluations the marginal
    likelihood cannot pin down the off-diagonal task correlations (Wu et al.
    2020 §3.1; this is "Agent 2 pitfall #1" in the plan-47 design notes).

    The stratification: of ``n_init`` total evaluations, ``hf_anchors`` go to
    the HF level (paired with the first ``hf_anchors`` cheap-fidelity points
    at identical ``x``), ``mid_anchors`` go to the median-cost fidelity (at
    additional Sobol' draws), and the remaining ``n_init - hf_anchors -
    mid_anchors`` go to the cheapest fidelity.  With the defaults
    ``hf_anchors=3, mid_anchors=2`` and ``n_init = 5*d + 3`` (Loeppky et al.
    2009), a 2D problem yields 8 cheap + 2 mid + 3 HF = 13 evaluations — a
    reasonable approximation to the 70/20/10 design-intent split.

    Parameters
    ----------
    bounds
        Per-dimension ``(lo, hi)`` pairs; ``len(bounds)`` is the design
        dimension ``d``.
    fidelity_levels
        External cascade layer indices (e.g. ``(1, 3, 7)`` or
        ``(1, 3, 5, 6, 7)``).  Sorted ascending to derive the contiguous
        internal index ``0..K-1`` returned in ``fid_indices``.  The cheapest
        fidelity is index ``0``; the HF fidelity is index ``K - 1``; the mid
        fidelity is the median index ``K // 2`` when ``K >= 3``.  When
        ``K == 2``, ``mid_anchors`` is silently zeroed (no median fidelity to
        anchor on); when ``K == 1`` the entire design lives at that single
        fidelity (``hf_anchors`` and ``mid_anchors`` ignored).
    n_init
        Total number of evaluations.  Defaults to ``5*d + 3``.
    hf_anchors
        Number of HF anchor points; the first ``hf_anchors`` cheap-fidelity
        ``x``-values are replicated at the HF level for paired evaluation.
        Must satisfy ``hf_anchors <= n_init - mid_anchors`` (otherwise there
        are no cheap points to pair with).
    mid_anchors
        Number of mid-fidelity points (additional unique Sobol' draws).
        Silently zeroed when ``K < 3``.
    seed
        Seed for :class:`torch.quasirandom.SobolEngine` (the engine is
        scrambled).  Same seed → identical ``(X, fid_indices)`` output.

    Returns
    -------
    X_init : np.ndarray
        Float64 array of shape ``(n_init, d)``.  The first ``n_cheap`` rows
        are cheap-fidelity Sobol' draws; the next ``mid_anchors`` (when
        ``K >= 3``) are mid-fidelity draws; the final ``hf_anchors`` rows are
        the HF replicas (a verbatim copy of the first ``hf_anchors`` cheap
        rows).
    fid_indices : np.ndarray
        Int64 array of shape ``(n_init,)``.  Holds *internal contiguous*
        fidelity indices ``0..K-1`` aligned with ``sorted(fidelity_levels)``,
        not the external layer numbers.  The BO module is the only place that
        does this internal indexing — the caller is responsible for
        translating back to external layers when invoking the cascade.

    Raises
    ------
    ValueError
        If ``bounds`` or ``fidelity_levels`` is empty; if ``n_init`` is not
        positive; if ``hf_anchors`` or ``mid_anchors`` is negative; or if
        ``hf_anchors`` exceeds the available cheap-fidelity slot count.
    """
    if not bounds:
        raise ValueError("bounds must be non-empty")
    if not fidelity_levels:
        raise ValueError("fidelity_levels must be non-empty")
    if hf_anchors < 0:
        raise ValueError(f"hf_anchors must be ≥ 0, got {hf_anchors}")
    if mid_anchors < 0:
        raise ValueError(f"mid_anchors must be ≥ 0, got {mid_anchors}")

    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError(
            f"bounds must be a sequence of (lo, hi) pairs, got shape "
            f"{bounds_arr.shape}"
        )
    if np.any(bounds_arr[:, 0] >= bounds_arr[:, 1]):
        raise ValueError(f"bounds must satisfy lo < hi for every dim: {bounds}")

    d = bounds_arr.shape[0]
    if n_init is None:
        n_init = 5 * d + 3
    if n_init <= 0:
        raise ValueError(f"n_init must be > 0, got {n_init}")

    sorted_levels = sorted(set(fidelity_levels))
    K = len(sorted_levels)
    cheap_idx = 0
    hf_idx = K - 1
    mid_idx = K // 2  # median index; coincides with cheap_idx when K==1

    # Collapse mid into "no mid" when there is no distinct median fidelity.
    if K < 3:
        mid_anchors = 0
    if K == 1:
        # Single fidelity: ignore HF anchors (no distinct HF level to anchor).
        hf_anchors = 0

    n_cheap = n_init - hf_anchors - mid_anchors
    if n_cheap < hf_anchors:
        raise ValueError(
            f"need at least hf_anchors={hf_anchors} cheap points to pair "
            f"with HF replicas, but n_cheap = n_init - hf_anchors - "
            f"mid_anchors = {n_cheap}"
        )
    if n_cheap < 0:
        raise ValueError(
            f"n_init={n_init} too small for hf_anchors={hf_anchors} + "
            f"mid_anchors={mid_anchors}"
        )

    sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed)
    n_unique = n_cheap + mid_anchors
    raw = sobol.draw(n_unique).numpy().astype(np.float64, copy=False)

    lo = bounds_arr[:, 0]
    span = bounds_arr[:, 1] - bounds_arr[:, 0]
    X_unique = lo + raw * span  # broadcasts over n_unique rows

    X_cheap = X_unique[:n_cheap]
    X_mid = X_unique[n_cheap : n_cheap + mid_anchors]
    X_hf = X_cheap[:hf_anchors].copy()  # paired with first hf_anchors cheap x's

    X_init = np.vstack([X_cheap, X_mid, X_hf]) if (mid_anchors or hf_anchors) else X_cheap
    fid_indices = np.concatenate(
        [
            np.full(n_cheap, cheap_idx, dtype=np.int64),
            np.full(mid_anchors, mid_idx, dtype=np.int64),
            np.full(hf_anchors, hf_idx, dtype=np.int64),
        ]
    )
    return X_init, fid_indices


__all__: list[str] = [
    "_BO_SENTINEL",
    "BOEval",
    "BOResult",
    "DEFAULT_COST_TABLE",
    "apply_cost_floor",
    "build_cost_model",
    "build_initial_design",
    "build_mf_gp",
    "make_multi_fidelity_objective",
]
