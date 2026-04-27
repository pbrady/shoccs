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


__all__: list[str] = []
