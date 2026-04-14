"""Optimization wrapper around the Brady-Livescu 2D stability pipeline.

Implements the layered cascade approach of Phase 43: a cheap inner objective
built from short-circuited :func:`brady2d_stability_score` calls drives an
off-the-shelf scipy optimizer, top-k survivors are re-ranked at a higher
(more expensive) ``max_layer``, and the winner is optionally pushed through
the L8 C++ bridge for simulation-level validation.

See ``plans/43-stability-optimization-framework.md`` for the plan, scope, and
algorithm choices (Nelder-Mead, COBYQA, SHGO, differential_evolution,
Sobol-seeded multi-start).

The public API surface is:

- :class:`OptimizeResult` — frozen record returned by every ``run_*`` helper.
- :data:`DEFAULT_BOUNDS` — ``(scheme, kernel) -> list[(lo, hi)]`` fallback.
- :func:`params_from_vector` / :func:`vector_from_params` — kernel-aware
  mapping between the optimizer's flat ``x`` and the nested ``params`` dict
  that ``brady2d_stability_score`` expects.
- :func:`extract_field` — dotted-path lookup into :class:`StabilityReport`.
- :func:`make_objective` — builds a feasibility-gated ``f(x) -> float``.
- :func:`run_scipy_local`, :func:`run_scipy_shgo`, :func:`run_scipy_de`,
  :func:`multi_start_optimize`, :func:`run_staged_optimize` — the driver
  helpers.
"""

from __future__ import annotations

import operator
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy
from scipy.optimize import minimize

from stencil_gen.brady2d_stability import brady2d_stability_score


# --- bounds ------------------------------------------------------------------

# Per-(scheme, kernel) default bounds.  Matches the parameter-spaces table in
# plans/43-stability-optimization-framework.md.  Gaussian/multiquadric ranges
# are given in log-uniform space only in the UI sense; the optimizer operates
# on the raw ε value — log-sampling for multi-start is applied by Sobol scaling
# inside :func:`multi_start_optimize` when bounds span more than a decade.
#
# Scope note (plan 43.1d, option b): ``"tension-penalty"`` and
# ``"mixed-epsilon"`` are intentionally omitted.  ``brady2d_stability_score``
# and its layer helpers currently dispatch only
# ``kernel ∈ {"classical", "tension", "gaussian", "multiquadric"}``; the two
# excluded families live in dedicated sweeps (``sweeps/tension_penalty_sweep``
# and ``sweeps/mixed_epsilon_sweep``) that bypass the layered pipeline.
# Extending the layered pipeline to those kernels is deferred — see the
# "What this plan does NOT do" section of the plan file.
DEFAULT_BOUNDS: dict[tuple[str, str], list[tuple[float, float]]] = {
    ("E2", "tension"): [(0.5, 20.0)],
    ("E4", "tension"): [(0.5, 20.0)],
    ("E2", "gaussian"): [(0.1, 5.0)],
    ("E4", "gaussian"): [(0.1, 5.0)],
    ("E2", "multiquadric"): [(0.1, 5.0)],
    ("E4", "multiquadric"): [(0.1, 5.0)],
    ("E4", "classical"): [(-2.0, 2.0), (197.0 / 288.0, 2.0)],
}


# --- result record -----------------------------------------------------------

@dataclass(frozen=True)
class OptimizeResult:
    """Frozen record of a single optimizer run.

    Attributes
    ----------
    best_params : dict
        Kernel-specific params dict at the optimum (the thing you would pass
        to :func:`brady2d_stability_score`).
    best_x : np.ndarray
        Flat parameter vector at the optimum.
    best_objective : float
        Value of the objective at ``best_x``.  ``+inf`` if no feasible point
        was found.
    best_report : dict
        Serialized :class:`StabilityReport` at the optimum (empty dict if no
        feasible point was found).
    method : str
        Name of the driver ("Nelder-Mead", "COBYQA", "SHGO", "DE",
        "multi-start", "staged").
    converged : bool
        Whether the underlying scipy call reported convergence AND the best
        objective is finite.
    n_evals : int
        Total objective evaluations performed.
    compute_time : float
        Wall-clock seconds.
    history : list
        ``[(x, f), ...]`` sampled during the run.  May be empty for drivers
        that do not expose per-step callbacks.
    extras : dict
        Free-form additional fields (e.g. ``n_local_minima`` for SHGO,
        ``stage`` for staged).
    """

    best_params: dict
    best_x: np.ndarray
    best_objective: float
    best_report: dict
    method: str
    converged: bool
    n_evals: int
    compute_time: float
    history: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


# --- primitives: params <-> vector -------------------------------------------

_SCALAR_EPSILON_KERNELS = ("gaussian", "multiquadric")


def params_from_vector(kernel: str, x: np.ndarray) -> dict:
    """Convert a flat vector to the kernel-specific ``params`` dict that
    :func:`brady2d_stability_score` consumes.

    Kernel mapping (see plan 43, section "Parameter spaces in scope"):

    - ``"tension"``                     : ``x=[σ]``            → ``{"sigma": σ}``
    - ``"gaussian"`` / ``"multiquadric"``: ``x=[ε]``            → ``{"epsilon": ε}``
    - ``"classical"``                   : ``x=[α₀, α₁]``       → ``{"alpha": [α₀, α₁]}``

    The ``"tension-penalty"`` and ``"mixed-epsilon"`` families are out of
    scope for this optimizer (plan 43.1d, option b) — ``brady2d_stability_score``
    does not route those kernels.  Use the standalone
    ``sweeps/tension_penalty_sweep`` / ``sweeps/mixed_epsilon_sweep`` entry
    points for those parameter spaces.
    """
    x = np.asarray(x, dtype=float).ravel()
    if kernel == "tension":
        if x.size != 1:
            raise ValueError(f"kernel='tension' expects 1D vector, got shape {x.shape}")
        return {"sigma": float(x[0])}
    if kernel in _SCALAR_EPSILON_KERNELS:
        if x.size != 1:
            raise ValueError(f"kernel={kernel!r} expects 1D vector, got shape {x.shape}")
        return {"epsilon": float(x[0])}
    if kernel == "classical":
        if x.size != 2:
            raise ValueError(f"kernel='classical' expects 2D vector, got shape {x.shape}")
        return {"alpha": [float(x[0]), float(x[1])]}
    raise ValueError(f"unknown kernel: {kernel!r}")


def vector_from_params(kernel: str, params: dict) -> np.ndarray:
    """Inverse of :func:`params_from_vector`.

    Returns a flat ``np.ndarray`` of dtype ``float`` whose layout matches the
    convention in :func:`params_from_vector`.
    """
    if kernel == "tension":
        return np.array([float(params["sigma"])], dtype=float)
    if kernel in _SCALAR_EPSILON_KERNELS:
        return np.array([float(params["epsilon"])], dtype=float)
    if kernel == "classical":
        alpha = params["alpha"]
        if len(alpha) != 2:
            raise ValueError(
                f"kernel='classical' expects alpha of length 2, got {len(alpha)}"
            )
        return np.array([float(alpha[0]), float(alpha[1])], dtype=float)
    raise ValueError(f"unknown kernel: {kernel!r}")


# --- primitives: report field extraction -------------------------------------

def extract_field(report, dotted_path: str) -> float:
    """Dotted-path lookup into a :class:`StabilityReport`.

    The first segment resolves as an attribute on ``report`` (via
    :func:`operator.attrgetter`); remaining segments walk the nested payload
    — ``dict[key]`` when the current node is a mapping, ``getattr`` otherwise
    so dataclass-valued fields like ``kreiss`` work the same as the dict-
    valued layer payloads.  Returns ``float('inf')`` if any segment is
    missing, the layer was not run (``None``), or the final value cannot be
    coerced to ``float``.

    Examples
    --------
    >>> extract_field(report, "layer1.boundary_gv_err")
    >>> extract_field(report, "layer6.transient_growth_bound")
    >>> extract_field(report, "kreiss.witness_sigma_min")
    """
    segments = dotted_path.split(".")
    if not segments or not segments[0]:
        return float("inf")
    first, *rest = segments
    try:
        node = operator.attrgetter(first)(report)
    except AttributeError:
        return float("inf")
    if node is None:
        return float("inf")
    for seg in rest:
        if isinstance(node, dict):
            if seg not in node:
                return float("inf")
            node = node[seg]
        else:
            try:
                node = getattr(node, seg)
            except AttributeError:
                return float("inf")
        if node is None:
            return float("inf")
    try:
        return float(node)
    except (TypeError, ValueError):
        return float("inf")


# --- objective factory -------------------------------------------------------

_LAYER_PREFIX_RE = re.compile(r"^layer(\d+)\.")

# Dotted-path prefixes that alias to a populating layer.  ``kreiss`` is
# assigned in layer 2; extend this mapping if new aliased fields are added
# to :class:`StabilityReport`.
_FIELD_LAYER_ALIAS = {
    "kreiss": 2,
    "non_normality": 6,
}


def _infer_max_layer(report_field: str) -> int | None:
    """Return the layer that populates ``report_field``, or ``None`` if the
    prefix is unrecognised.  Layer-prefixed fields (``layer1.*`` …
    ``layer8.*``) are parsed directly; aliased fields such as ``kreiss.*`` are
    mapped via :data:`_FIELD_LAYER_ALIAS`.
    """
    m = _LAYER_PREFIX_RE.match(report_field)
    if m:
        return int(m.group(1))
    head, _, _ = report_field.partition(".")
    return _FIELD_LAYER_ALIAS.get(head)


def make_objective(
    scheme: str,
    kernel: str,
    report_field: str,
    *,
    gate_layer: int = 3,
    max_layer: int | None = None,
) -> Callable[[np.ndarray], float]:
    """Build a feasibility-gated objective ``f(x) -> float``.

    The returned closure converts a flat vector ``x`` to a kernel-specific
    ``params`` dict, runs :func:`brady2d_stability_score` in short-circuit
    mode up to ``max_layer``, and returns:

    - ``+inf`` if any layer at or before ``gate_layer`` failed (the
      feasibility cliff).
    - ``+inf`` if :func:`brady2d_stability_score` raised (extreme parameters
      can produce singular/ill-conditioned RBF systems).
    - :func:`extract_field` of ``report_field`` otherwise (which itself
      returns ``+inf`` when the requested dotted path is absent).

    Parameters
    ----------
    scheme, kernel, report_field
        Forwarded to :func:`brady2d_stability_score` and
        :func:`extract_field`.
    gate_layer
        Highest layer whose failure forces ``+inf`` (the feasibility gate).
        Defaults to 3, matching the cheap-inner stage of the cascade.
    max_layer
        Highest layer actually executed.  Defaults to the layer implied by
        ``report_field`` (``layer6.*`` → 6, ``kreiss.*`` → 2, …).  Raises
        ``ValueError`` if the resolved value is less than ``gate_layer`` —
        the optimiser cannot gate on layers it never runs.
    """
    if max_layer is None:
        inferred = _infer_max_layer(report_field)
        if inferred is None:
            raise ValueError(
                f"cannot infer max_layer from report_field={report_field!r}; "
                "pass max_layer explicitly"
            )
        max_layer = inferred
    if max_layer < gate_layer:
        raise ValueError(
            f"max_layer={max_layer} is less than gate_layer={gate_layer}; "
            "raise max_layer or lower gate_layer"
        )

    def objective(x: np.ndarray) -> float:
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
            return float("inf")
        if report.failed_layer is not None and report.failed_layer <= gate_layer:
            return float("inf")
        return extract_field(report, report_field)

    return objective


# --- drivers -----------------------------------------------------------------

def _probe_cobyqa_available() -> bool:
    """Probe whether ``scipy.optimize.minimize(method="COBYQA")`` is usable.

    COBYQA was added in scipy 1.14; older installations raise ``ValueError``
    when the method is requested.  The probe itself is a sub-millisecond call
    on a 1-variable identity objective.
    """
    try:
        minimize(
            lambda x: float(x[0] ** 2),
            x0=np.array([1.0]),
            method="COBYQA",
            options={"maxfev": 2},
        )
    except Exception:
        return False
    return True


_COBYQA_AVAILABLE = _probe_cobyqa_available()

_LOCAL_METHODS = ("Nelder-Mead", "COBYQA")


def run_scipy_local(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    *,
    method: str = "Nelder-Mead",
    max_evals: int = 200,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Local optimization via ``scipy.optimize.minimize``.

    Wraps the user-supplied objective ``f`` in a recorder that appends
    ``(x.copy(), fval)`` to ``history`` on every evaluation — scipy's
    per-iteration ``callback`` only samples once per simplex step, which
    would miss most feasibility-cliff evaluations.

    ``method="Nelder-Mead"`` and ``method="COBYQA"`` are both supported.
    COBYQA (derivative-free trust region, scipy ≥ 1.14) handles 1-6D problems
    with feasibility cliffs better than Nelder-Mead in practice; see plan
    43.3b.  If COBYQA is requested on a scipy build that lacks it, a clear
    ``RuntimeError`` is raised rather than the opaque internal ``ValueError``.
    """
    if method not in _LOCAL_METHODS:
        raise ValueError(
            f"run_scipy_local: method must be one of {_LOCAL_METHODS}, got {method!r}"
        )
    if method == "COBYQA" and not _COBYQA_AVAILABLE:
        raise RuntimeError(
            f"COBYQA requires scipy >= 1.14; got {scipy.__version__}"
        )
    x0 = np.asarray(x0, dtype=float).ravel()
    if len(bounds) != x0.size:
        raise ValueError(
            f"run_scipy_local: bounds length {len(bounds)} does not match x0 size {x0.size}"
        )

    history: list[tuple[np.ndarray, float]] = []

    def _recorder(x: np.ndarray) -> float:
        fval = float(f(np.asarray(x, dtype=float)))
        history.append((np.asarray(x, dtype=float).copy(), fval))
        return fval

    if method == "Nelder-Mead":
        options = {
            "xatol": tol,
            "fatol": tol,
            "maxfev": max_evals,
            "adaptive": True,
        }
    else:  # COBYQA
        options = {
            "maxfev": max_evals,
            "feasibility_tol": tol,
        }

    t0 = time.perf_counter()
    result = minimize(
        _recorder,
        x0=x0,
        method=method,
        bounds=bounds,
        options=options,
    )
    compute_time = time.perf_counter() - t0

    best_x = np.asarray(result.x, dtype=float).ravel()
    best_objective = float(result.fun)
    converged = bool(result.success) and np.isfinite(best_objective)

    # ``run_scipy_local`` is kernel-agnostic — it receives a black-box ``f``
    # and cannot map ``best_x`` back to a kernel-specific params dict.
    # Higher-level drivers (``multi_start_optimize``, ``run_staged_optimize``)
    # own the kernel and use ``dataclasses.replace`` to fill ``best_params``.
    return OptimizeResult(
        best_params={},
        best_x=best_x,
        best_objective=best_objective,
        best_report={},
        method=method,
        converged=converged,
        n_evals=int(getattr(result, "nfev", len(history))),
        compute_time=compute_time,
        history=history,
        extras={"scipy_message": str(getattr(result, "message", ""))},
    )


def multi_start_optimize(
    f: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    n_restarts: int = 10,
    *,
    method: str = "Nelder-Mead",
    seed: int = 0,
) -> OptimizeResult:
    """Sobol-seeded multi-start wrapper around :func:`run_scipy_local`.

    Implemented in 43.4a.
    """
    raise NotImplementedError("multi_start_optimize: implemented in 43.4")


def run_scipy_shgo(
    f: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    *,
    n: int = 100,
    iters: int = 3,
) -> OptimizeResult:
    """Global optimization via ``scipy.optimize.shgo``.

    Implemented in 43.5a.
    """
    raise NotImplementedError("run_scipy_shgo: implemented in 43.5")


def run_scipy_de(
    f: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    *,
    popsize: int = 15,
    maxiter: int = 100,
    seed: int = 0,
    strategy: str = "best1bin",
) -> OptimizeResult:
    """Global optimization via ``scipy.optimize.differential_evolution``.

    Implemented in 43.5b.
    """
    raise NotImplementedError("run_scipy_de: implemented in 43.5")


def run_staged_optimize(
    scheme: str,
    kernel: str,
    report_field: str,
    bounds: list[tuple[float, float]],
    *,
    inner_gate: int = 3,
    inner_max_layer: int = 3,
    validator_max_layer: int = 6,
    top_k: int = 5,
    method: str = "Nelder-Mead",
    n_restarts: int = 20,
    seed: int = 0,
) -> OptimizeResult:
    """Cheap-inner + expensive-validator staged pipeline.

    Implemented in 43.6a.
    """
    raise NotImplementedError("run_staged_optimize: implemented in 43.6")
