"""Tests for :mod:`stencil_gen.bo` (plan 47)."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from botorch.acquisition.cost_aware import InverseCostWeightedUtility

from stencil_gen.bo import (
    _BO_SENTINEL,
    BOEval,
    BOResult,
    DEFAULT_COST_TABLE,
    apply_cost_floor,
    build_cost_model,
    build_initial_design,
    build_mf_gp,
    make_multi_fidelity_objective,
)
from stencil_gen.brady2d_stability import StabilityReport


def _empty_report_with(**layers) -> StabilityReport:
    """Build a ``StabilityReport`` with the given layer payloads populated."""
    r = StabilityReport.empty()
    for name, value in layers.items():
        setattr(r, name, value)
    return r


def _make_bo_eval(*, fidelity: int = 1, value: float = 0.1) -> BOEval:
    return BOEval(
        x=np.array([-0.77, 0.16]),
        params={"alpha": [-0.77, 0.16]},
        fidelity=fidelity,
        value=value,
        wall_time=0.05,
        report={"failed_layer": None},
    )


def _make_bo_result(eval_history: tuple[BOEval, ...] = ()) -> BOResult:
    hf_history = tuple(e for e in eval_history if e.fidelity == 7)
    return BOResult(
        best_x=np.array([-0.77, 0.16]),
        best_params={"alpha": [-0.77, 0.16]},
        best_objective=0.42,
        best_report={"failed_layer": None},
        method="BoTorch-qMFKG",
        scheme="E4",
        kernel="classical",
        bounds=((-2.0, 2.0), (0.05, 2.0)),
        fidelity_levels=(1, 3, 7),
        hf_level=7,
        report_fields_by_layer={
            1: "layer1.boundary_gv_err",
            3: "layer3.max_stab_eig",
            7: "layer7.max_spectral_abscissa",
        },
        cost_model={1: 0.076, 3: 0.038, 7: 1.434},
        n_evals_per_fidelity={1: 9, 3: 3, 7: 2},
        wall_time_per_fidelity={1: 0.7, 3: 0.1, 7: 2.9},
        total_compute_time=3.7,
        eval_history=eval_history,
        hf_eval_history=hf_history,
        gp_hyperparameters={"lengthscale": [1.0, 1.0], "outputscale": 1.0},
        seed=1,
        converged=True,
        stop_reason="variance",
        extras={"n_sentinel_filtered": 0},
    )


class TestBOResult:
    """Plan 47.1a: ``BOEval`` and ``BOResult`` are frozen dataclasses."""

    def test_frozen_dataclasses(self):
        ev = _make_bo_eval()
        with pytest.raises(FrozenInstanceError):
            ev.fidelity = 99
        with pytest.raises(FrozenInstanceError):
            ev.value = 0.0

        result = _make_bo_result()
        with pytest.raises(FrozenInstanceError):
            result.best_objective = 0.0
        with pytest.raises(FrozenInstanceError):
            result.seed = 99

    def test_eval_history_is_tuple_not_list(self):
        history = (_make_bo_eval(fidelity=1), _make_bo_eval(fidelity=7, value=0.42))
        result = _make_bo_result(history)
        assert isinstance(result.eval_history, tuple)
        assert isinstance(result.hf_eval_history, tuple)
        assert all(isinstance(e, BOEval) for e in result.eval_history)
        assert len(result.hf_eval_history) == 1
        assert result.hf_eval_history[0].fidelity == 7

    def test_serializable_via_dataclasses_asdict(self):
        # Plan 47.1c: 47.4c builds the JSON encoder; for now just assert
        # `dataclasses.asdict()` succeeds on a populated BOResult.  This
        # guards against any future field that isn't natively dataclass-
        # serialisable being added without an encoder update.
        history = (_make_bo_eval(fidelity=1), _make_bo_eval(fidelity=7, value=0.42))
        result = _make_bo_result(history)
        d = dataclasses.asdict(result)
        assert d["method"] == "BoTorch-qMFKG"
        assert d["seed"] == 1
        assert d["hf_level"] == 7
        # eval_history asdict-recursively unfolds each BOEval to a dict
        assert len(d["eval_history"]) == 2
        assert d["eval_history"][0]["fidelity"] == 1
        assert d["eval_history"][1]["fidelity"] == 7


class TestMakeMultiFidelityObjective:
    """Plan 47.1b: multi-fidelity objective factory ``f(x, m) -> (value, wt, report)``."""

    def test_returns_3tuple(self, monkeypatch):
        import stencil_gen.bo as bo_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            return _empty_report_with(
                layer1={"boundary_gv_err": 0.03},
                layer3={"max_stab_eig": 1e-12},
            )

        monkeypatch.setattr(bo_mod, "brady2d_stability_score", fake_score)
        f = make_multi_fidelity_objective(
            "E4",
            "classical",
            {1: "layer1.boundary_gv_err", 3: "layer3.max_stab_eig"},
        )
        out = f(np.array([-0.77, 0.16]), 1)
        assert isinstance(out, tuple)
        assert len(out) == 3
        value, wall_time, report = out
        assert isinstance(value, float)
        assert isinstance(wall_time, float)
        assert isinstance(report, dict)
        assert value == pytest.approx(0.03)
        assert wall_time >= 0.0

    def test_sentinel_on_gate_trip(self, monkeypatch):
        # Layers (1, 3, 7) → gate_layer auto = 0; only failed_layer == 0
        # would sentinel here, which never happens (layers are 1-indexed).
        # Use layers (3, 7) → gate_layer = 2; failed_layer=1 (<= 2) gates.
        import stencil_gen.bo as bo_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            r = StabilityReport.empty()
            r.failed_layer = 1
            r.failed_reason = "synthetic L1 failure"
            return r

        monkeypatch.setattr(bo_mod, "brady2d_stability_score", fake_score)
        f = make_multi_fidelity_objective(
            "E4",
            "classical",
            {3: "layer3.max_stab_eig", 7: "layer7.max_spectral_abscissa"},
        )
        value, wall_time, report = f(np.array([5.0, 5.0]), 3)
        assert value == _BO_SENTINEL
        assert wall_time > 0.0  # measured perf_counter delta around the call
        # The serialised StabilityReport carries the fail metadata.
        assert report.get("failed_layer") == 1

    def test_sentinel_on_shape_mismatch(self):
        # E4 classical expects x of length 2.  A length-3 input causes
        # params_from_vector to raise; the closure must swallow and sentinel.
        f = make_multi_fidelity_objective(
            "E4",
            "classical",
            {1: "layer1.boundary_gv_err", 3: "layer3.max_stab_eig"},
        )
        value, wall_time, report = f(np.array([-0.77, 0.16, 99.0]), 1)
        assert value == _BO_SENTINEL
        assert wall_time >= 0.0
        assert "error" in report

    def test_unknown_fidelity_returns_sentinel(self):
        f = make_multi_fidelity_objective(
            "E4",
            "classical",
            {1: "layer1.boundary_gv_err", 3: "layer3.max_stab_eig"},
        )
        value, wall_time, report = f(np.array([-0.77, 0.16]), 99)
        assert value == _BO_SENTINEL
        assert wall_time == 0.0  # never invoked the score function
        assert "error" in report
        assert "99" in report["error"]

    def test_field_layer_validation_at_factory_time(self):
        # layer7.* is populated only when max_layer >= 7, but here we key it
        # under m=3.  The factory must reject this configuration up-front.
        with pytest.raises(ValueError, match="cannot extract"):
            make_multi_fidelity_objective(
                "E4",
                "classical",
                {3: "layer7.max_spectral_abscissa"},
            )

    def test_rejects_empty_mapping(self):
        with pytest.raises(ValueError, match="must not be empty"):
            make_multi_fidelity_objective("E4", "classical", {})

    def test_finite_at_known_feasible_point(self):
        # BL published optimum for E4 classical.  Real cascade call (no mock):
        # exercises params_from_vector → brady2d_stability_score → extract_field.
        # Restricted to L1 + L3 to keep this test in the fast suite.
        f = make_multi_fidelity_objective(
            "E4",
            "classical",
            {1: "layer1.boundary_gv_err", 3: "layer3.max_stab_eig"},
        )
        for m in (1, 3):
            value, wall_time, report = f(
                np.array([-0.7733323791884821, 0.1623961700641681]), m
            )
            assert np.isfinite(value), f"BL optimum at m={m} returned non-finite"
            assert value < _BO_SENTINEL / 2
            assert wall_time > 0.0

    def test_gate_layer_default(self, monkeypatch):
        # Default gate_layer = max(min(layers) - 1, 0).  For layers (3, 7)
        # → gate_layer = 2.  An L2 failure (== gate_layer) gates; an L3
        # failure (> gate_layer) does not.
        import stencil_gen.bo as bo_mod

        sentinel = {"failed_layer": 2, "field": 0.04}

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            r = StabilityReport.empty()
            r.failed_layer = sentinel["failed_layer"]
            r.layer3 = {"max_stab_eig": sentinel["field"]}
            return r

        monkeypatch.setattr(bo_mod, "brady2d_stability_score", fake_score)
        f = make_multi_fidelity_objective(
            "E4",
            "classical",
            {3: "layer3.max_stab_eig", 7: "layer7.max_spectral_abscissa"},
        )

        # L2 failure ≤ gate_layer=2 ⇒ sentinel.
        sentinel["failed_layer"] = 2
        value, _, _ = f(np.array([-0.77, 0.16]), 3)
        assert value == _BO_SENTINEL

        # L3 failure > gate_layer=2 ⇒ pass through to extract_field.
        sentinel["failed_layer"] = 3
        value, _, _ = f(np.array([-0.77, 0.16]), 3)
        assert value == pytest.approx(0.04)

    def test_gate_layer_explicit_override(self, monkeypatch):
        import stencil_gen.bo as bo_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            r = StabilityReport.empty()
            r.failed_layer = 1  # would be gated by default (gate_layer=0
            # implies no gating; with explicit gate_layer=5, layer1 fails
            # gate trivially)
            r.layer1 = {"boundary_gv_err": 0.04}
            r.layer3 = {"max_stab_eig": 1e-12}
            return r

        monkeypatch.setattr(bo_mod, "brady2d_stability_score", fake_score)
        f = make_multi_fidelity_objective(
            "E4",
            "classical",
            {1: "layer1.boundary_gv_err", 3: "layer3.max_stab_eig"},
            gate_layer=5,
        )
        # failed_layer=1 ≤ gate_layer=5 ⇒ sentinel even though L1 has data.
        value, _, _ = f(np.array([-0.77, 0.16]), 1)
        assert value == _BO_SENTINEL

    def test_wall_time_recorded(self, monkeypatch):
        import stencil_gen.bo as bo_mod

        def slow_score(scheme, kernel, params, *, max_layer, short_circuit):
            time.sleep(0.05)
            return _empty_report_with(layer1={"boundary_gv_err": 0.01})

        monkeypatch.setattr(bo_mod, "brady2d_stability_score", slow_score)
        f = make_multi_fidelity_objective(
            "E4",
            "classical",
            {1: "layer1.boundary_gv_err", 3: "layer3.max_stab_eig"},
        )
        t0 = time.perf_counter()
        _, wall_time, _ = f(np.array([-0.77, 0.16]), 1)
        elapsed = time.perf_counter() - t0
        assert wall_time >= 0.05
        # Reported wall time can't exceed the perf_counter delta around the
        # call (allow 5ms slack for finalization between time samples).
        assert wall_time <= elapsed + 0.005

    def test_sentinel_when_field_path_missing(self, monkeypatch):
        # Layer ran fine, no failed_layer, but the requested field path is
        # absent (extract_field returns +inf).  The factory rewrites this
        # to the finite sentinel so the GP fit stays well-conditioned.
        import stencil_gen.bo as bo_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            return _empty_report_with(layer1={"some_other_field": 0.0})

        monkeypatch.setattr(bo_mod, "brady2d_stability_score", fake_score)
        f = make_multi_fidelity_objective(
            "E4",
            "classical",
            {1: "layer1.boundary_gv_err", 3: "layer3.max_stab_eig"},
        )
        value, _, _ = f(np.array([-0.77, 0.16]), 1)
        assert value == _BO_SENTINEL


# ---------------------------------------------------------------------------
# 47.2d: GP + cost model + DOE tests
# ---------------------------------------------------------------------------


def _make_smooth_mf_dataset(
    *, n: int = 30, d: int = 2, num_fidelities: int = 3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic smooth-quadratic data for GP-fit tests.

    Per-fidelity bias produces non-trivial off-diagonal ICM correlations
    without forcing the marginal-likelihood optimiser into the noise-free
    pathology that bites a hand-composed Matern × IndexKernel SingleTaskGP.
    """
    rng = np.random.RandomState(seed)
    X_design = rng.uniform(-1.0, 1.0, size=(n, d))
    fid = rng.randint(0, num_fidelities, size=n).astype(np.float64)
    X = np.column_stack([X_design, fid])
    Y = np.array(
        [
            (X[i, 0] - 0.3) ** 2
            + (X[i, 1] + 0.2) ** 2
            + 0.05 * X[i, -1]
            for i in range(n)
        ]
    )
    return X, Y


class TestMFGP:
    """Plan 47.2a: ``build_mf_gp`` ICM multi-fidelity GP surrogate."""

    def test_gp_fits_on_synthetic_data(self):
        torch.manual_seed(0)
        X, Y = _make_smooth_mf_dataset(n=30, d=2, num_fidelities=3, seed=0)
        gp = build_mf_gp(X, Y, fidelity_dim=2, num_fidelities=3)
        gp.eval()
        with torch.no_grad():
            posterior = gp.posterior(torch.as_tensor(X, dtype=torch.float64))
            mean = posterior.mean.squeeze(-1).numpy()
        # Posterior mean at training points within 1e-3 of training Y.
        assert np.max(np.abs(mean - Y)) < 1e-3

    def test_index_kernel_correlation_matrix_psd(self):
        torch.manual_seed(0)
        X, Y = _make_smooth_mf_dataset(n=30, d=2, num_fidelities=3, seed=0)
        gp = build_mf_gp(X, Y, fidelity_dim=2, num_fidelities=3)
        # Reconstruct B = W Wᵀ + diag(var) per the documented accessor paths.
        ik = gp.covar_module.kernels[1]
        W = ik.covar_factor.detach().numpy()
        v = ik.var.detach().numpy().squeeze()
        B = W @ W.T + np.diag(np.atleast_1d(v))
        eigs = np.linalg.eigvalsh(B)
        assert np.all(eigs >= -1e-10), f"B not PSD: eigs = {eigs}"

    def test_seed_determinism(self):
        X, Y = _make_smooth_mf_dataset(n=20, d=2, num_fidelities=3, seed=0)
        torch.manual_seed(42)
        gp1 = build_mf_gp(X, Y, fidelity_dim=2, num_fidelities=3)
        ls1 = gp1.covar_module.kernels[0].lengthscale.detach().numpy().copy()
        W1 = gp1.covar_module.kernels[1].covar_factor.detach().numpy().copy()

        torch.manual_seed(42)
        gp2 = build_mf_gp(X, Y, fidelity_dim=2, num_fidelities=3)
        ls2 = gp2.covar_module.kernels[0].lengthscale.detach().numpy().copy()
        W2 = gp2.covar_module.kernels[1].covar_factor.detach().numpy().copy()

        np.testing.assert_allclose(ls1, ls2, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(W1, W2, atol=0.0, rtol=0.0)

    def test_noise_floor_respected(self):
        # Smooth quadratic is essentially noise-free; the noise constraint
        # GreaterThan(1e-9) must keep likelihood.noise above the floor so the
        # Cholesky factorisation of the kernel matrix doesn't fail.  The
        # softplus reparameterisation can return a value ~3 ULPs below 1e-9
        # at float32 precision (e.g. 9.999...e-10) — allow this tiny slack.
        from gpytorch.constraints import GreaterThan

        torch.manual_seed(0)
        X, Y = _make_smooth_mf_dataset(n=30, d=2, num_fidelities=3, seed=0)
        gp = build_mf_gp(X, Y, fidelity_dim=2, num_fidelities=3)
        constraint = gp.likelihood.noise_covar.raw_noise_constraint
        assert isinstance(constraint, GreaterThan)
        assert float(constraint.lower_bound) == pytest.approx(1e-9)
        noise = gp.likelihood.noise.detach().numpy().squeeze()
        assert noise == pytest.approx(1e-9, rel=1e-4, abs=0.0)

    def test_rejects_invalid_inputs(self):
        # Plan 47.2a's ``build_mf_gp`` validates shapes / index ranges before
        # constructing the GPyTorch model — verify the up-front error paths.
        X, Y = _make_smooth_mf_dataset(n=10, d=2, num_fidelities=3, seed=0)
        with pytest.raises(ValueError, match="num_fidelities"):
            build_mf_gp(X, Y, fidelity_dim=2, num_fidelities=0)
        with pytest.raises(ValueError, match="rank"):
            build_mf_gp(X, Y, fidelity_dim=2, num_fidelities=3, rank=0)
        with pytest.raises(ValueError, match="fidelity_dim"):
            build_mf_gp(X, Y, fidelity_dim=99, num_fidelities=3)
        with pytest.raises(ValueError, match="train_X"):
            build_mf_gp(X.ravel(), Y, fidelity_dim=2, num_fidelities=3)
        with pytest.raises(ValueError, match="train_Y"):
            build_mf_gp(X, Y[:5], fidelity_dim=2, num_fidelities=3)


class TestCostModel:
    """Plan 47.2b: ``DEFAULT_COST_TABLE`` + ``apply_cost_floor`` + ``build_cost_model``."""

    def test_default_table_matches_plan_46_measurements(self):
        # Per the plan-47 background table: L1=76 ms, L3=38 ms, L3r=486 ms,
        # L6=846 ms, L7=1434 ms.  ``DEFAULT_COST_TABLE`` keys L3r at external
        # index 5 by 47.4a convention (between L3=3 and L6=6 in cost order).
        assert DEFAULT_COST_TABLE == {
            1: 0.076,
            3: 0.038,
            5: 0.486,
            6: 0.846,
            7: 1.434,
        }

    def test_inverse_cost_weighted_utility_construction(self):
        util = build_cost_model(DEFAULT_COST_TABLE, fidelity_dim=2)
        assert isinstance(util, InverseCostWeightedUtility)
        # Cost evaluation at every internal index returns the floored cost.
        n_layers = len(DEFAULT_COST_TABLE)
        X = torch.zeros(n_layers, 3, dtype=torch.float64)
        X[:, 2] = torch.arange(n_layers, dtype=torch.float64)
        costs = util.cost_model(X).squeeze(-1).numpy()
        assert costs.shape == (n_layers,)
        assert np.all(costs > 0.0)
        # Last entry is HF cost (1.434, no floor needed).
        assert costs[-1] == pytest.approx(1.434)

    def test_cost_floor_applied(self):
        # c(L1) = 0.001, c(L7) = 1.0 ⇒ floor = 0.05 * 1.0 = 0.05; L1 lifts.
        util = build_cost_model({1: 0.001, 7: 1.0}, fidelity_dim=0)
        X = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        costs = util.cost_model(X).squeeze(-1).numpy()
        # Internal index 0 ↔ external L1; internal index 1 ↔ external L7.
        assert costs[0] == pytest.approx(0.05)  # floored
        assert costs[1] == pytest.approx(1.0)
        # ``apply_cost_floor`` is the single source of truth for the formula.
        floored = apply_cost_floor({1: 0.001, 7: 1.0})
        assert floored == {1: 0.05, 7: 1.0}

    def test_cost_floor_disabled(self):
        floored = apply_cost_floor({1: 0.001, 7: 1.0}, floor_ratio=0.0)
        assert floored == {1: 0.001, 7: 1.0}

    def test_cost_table_persisted_in_BOResult(self):
        # 47.3b's ``run_mfbo`` will store the floored cost table in
        # ``BOResult.cost_model``; for now verify the dataclass field round-
        # trips via ``dataclasses.asdict`` and is not ``None``.
        result = BOResult(
            best_x=np.array([0.0, 0.0]),
            best_params={"alpha": [0.0, 0.0]},
            best_objective=0.0,
            best_report={},
            method="BoTorch-qMFKG",
            scheme="E4",
            kernel="classical",
            bounds=((-2.0, 2.0), (0.05, 2.0)),
            fidelity_levels=(1, 3, 7),
            hf_level=7,
            report_fields_by_layer={
                1: "layer1.boundary_gv_err",
                3: "layer3.max_stab_eig",
                7: "layer7.max_spectral_abscissa",
            },
            cost_model=apply_cost_floor({1: 0.076, 3: 0.038, 7: 1.434}),
            n_evals_per_fidelity={1: 1, 3: 1, 7: 1},
            wall_time_per_fidelity={1: 0.1, 3: 0.05, 7: 1.4},
            total_compute_time=1.55,
            eval_history=(),
            hf_eval_history=(),
            gp_hyperparameters={},
            seed=0,
            converged=False,
            stop_reason="budget",
            extras={},
        )
        assert result.cost_model is not None
        # Floor active: c(L7) = 1.434 ⇒ floor = 0.0717; L3 (0.038) lifts.
        assert result.cost_model[3] == pytest.approx(0.05 * 1.434)
        d = dataclasses.asdict(result)
        assert d["cost_model"] == result.cost_model

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError, match="empty"):
            build_cost_model({}, fidelity_dim=0)
        with pytest.raises(ValueError, match="floor_ratio"):
            build_cost_model({1: 0.1}, fidelity_dim=0, floor_ratio=-0.1)
        with pytest.raises(ValueError, match="fidelity_dim"):
            build_cost_model({1: 0.1}, fidelity_dim=-1)


class TestDOE:
    """Plan 47.2c: ``build_initial_design`` stratified Sobol' DOE."""

    def test_n_init_default(self):
        # d=2 ⇒ default n_init = 5*2 + 3 = 13 (Loeppky et al. 2009).
        X, fid = build_initial_design([(-1.0, 1.0), (-1.0, 1.0)], (1, 3, 7), seed=0)
        assert X.shape == (13, 2)
        assert fid.shape == (13,)

    def test_fidelity_stratification_default_split(self):
        # n_init=13, hf_anchors=3, mid_anchors=2, K=5 ⇒ exact 8/2/3 split per
        # the kwarg-derived counts (47.2c "Done" note: ``hf_anchors`` and
        # ``mid_anchors`` are literal counts, not 70/20/10 ratio targets).
        X, fid = build_initial_design(
            [(-1.0, 1.0), (-1.0, 1.0)],
            (1, 3, 5, 6, 7),
            n_init=13,
            hf_anchors=3,
            mid_anchors=2,
            seed=0,
        )
        unique, counts = np.unique(fid, return_counts=True)
        # Internal indices 0..K-1: cheap=0, mid=K//2=2, hf=K-1=4.
        counts_by_idx = dict(zip(unique.tolist(), counts.tolist()))
        assert counts_by_idx == {0: 8, 2: 2, 4: 3}

    def test_fidelity_stratification_clean_ratio(self):
        # Reachable 70/20/10: n_init=10, hf_anchors=1, mid_anchors=2, K=5.
        X, fid = build_initial_design(
            [(-1.0, 1.0), (-1.0, 1.0)],
            (1, 3, 5, 6, 7),
            n_init=10,
            hf_anchors=1,
            mid_anchors=2,
            seed=0,
        )
        unique, counts = np.unique(fid, return_counts=True)
        counts_by_idx = dict(zip(unique.tolist(), counts.tolist()))
        assert counts_by_idx == {0: 7, 2: 2, 4: 1}

    def test_hf_anchor_paired_with_cheap(self):
        # For ICM identifiability, every HF replica must share its ``x`` with
        # at least one cheap row (paired evaluations — Wu 2020 §3.1).
        X, fid = build_initial_design(
            [(-1.0, 1.0), (-1.0, 1.0)],
            (1, 3, 7),
            n_init=13,
            hf_anchors=3,
            mid_anchors=2,
            seed=0,
        )
        hf_idx = 2  # K - 1 with K = 3
        cheap_idx = 0
        hf_rows = X[fid == hf_idx]
        cheap_rows = X[fid == cheap_idx]
        assert len(hf_rows) == 3
        matches = sum(
            1 for hr in hf_rows if any(np.allclose(hr, cr) for cr in cheap_rows)
        )
        assert matches >= 3

    def test_hf_replicas_are_independent_copies(self):
        # The HF block is a ``.copy()`` of the first ``hf_anchors`` cheap
        # rows; mutating the HF rows must not bleed into the cheap rows
        # (defends against the ``.copy()`` regressing to a view).
        X, fid = build_initial_design(
            [(-1.0, 1.0), (-1.0, 1.0)],
            (1, 3, 7),
            n_init=13,
            hf_anchors=3,
            mid_anchors=2,
            seed=0,
        )
        hf_mask = fid == 2  # K-1 with K=3
        cheap_mask = fid == 0
        cheap_before = X[cheap_mask].copy()
        X[hf_mask] = 99.0
        np.testing.assert_array_equal(X[cheap_mask], cheap_before)

    def test_seed_determinism(self):
        Xa, fa = build_initial_design(
            [(-1.0, 1.0), (-1.0, 1.0)], (1, 3, 7), seed=42
        )
        Xb, fb = build_initial_design(
            [(-1.0, 1.0), (-1.0, 1.0)], (1, 3, 7), seed=42
        )
        np.testing.assert_array_equal(Xa, Xb)
        np.testing.assert_array_equal(fa, fb)
        Xc, _ = build_initial_design(
            [(-1.0, 1.0), (-1.0, 1.0)], (1, 3, 7), seed=43
        )
        assert not np.array_equal(Xa, Xc)

    def test_bounds_respected(self):
        bounds = [(-2.0, 2.0), (0.05, 2.0)]
        X, _ = build_initial_design(bounds, (1, 3, 7), n_init=20, seed=0)
        for j, (lo, hi) in enumerate(bounds):
            assert X[:, j].min() >= lo
            assert X[:, j].max() <= hi

    def test_K1_single_fidelity_all_cheap(self):
        # ``hf_anchors`` and ``mid_anchors`` silently ignored when K==1.
        X, fid = build_initial_design(
            [(-1.0, 1.0)], (7,), n_init=5, hf_anchors=2, mid_anchors=2, seed=0
        )
        assert X.shape == (5, 1)
        assert fid.shape == (5,)
        np.testing.assert_array_equal(fid, np.zeros(5, dtype=np.int64))

    def test_K2_mid_anchors_silently_zeroed(self):
        # ``mid_anchors`` silently zeroed when K<3; only cheap+HF appear.
        X, fid = build_initial_design(
            [(-1.0, 1.0)], (1, 7), n_init=10, hf_anchors=3, mid_anchors=2, seed=0
        )
        unique = set(np.unique(fid).tolist())
        assert unique == {0, 1}  # cheap=0, hf=K-1=1; no mid
        # Total preserved: n_cheap + hf_anchors = (10 - 3 - 0) + 3 = 10.
        assert fid.shape == (10,)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"bounds": [], "fidelity_levels": (1, 7)}, "bounds"),
            ({"bounds": [(-1.0, 1.0)], "fidelity_levels": ()}, "fidelity_levels"),
            ({"bounds": [(1.0, 0.0)], "fidelity_levels": (1, 7)}, "lo < hi"),
            (
                {"bounds": [(-1.0, 1.0)], "fidelity_levels": (1, 7), "n_init": 0},
                "n_init",
            ),
            (
                {"bounds": [(-1.0, 1.0)], "fidelity_levels": (1, 7), "n_init": -1},
                "n_init",
            ),
            (
                {
                    "bounds": [(-1.0, 1.0)],
                    "fidelity_levels": (1, 7),
                    "hf_anchors": -1,
                },
                "hf_anchors",
            ),
            (
                {
                    "bounds": [(-1.0, 1.0)],
                    "fidelity_levels": (1, 7),
                    "mid_anchors": -1,
                },
                "mid_anchors",
            ),
            (
                {
                    "bounds": [(-1.0, 1.0)],
                    "fidelity_levels": (1, 3, 7),
                    "n_init": 4,
                    "hf_anchors": 3,
                    "mid_anchors": 0,
                },
                "cheap",
            ),
        ],
    )
    def test_validation_errors(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            build_initial_design(**kwargs)

    def test_fid_indices_dtype_is_int64(self):
        _, fid = build_initial_design(
            [(-1.0, 1.0), (-1.0, 1.0)], (1, 3, 7), seed=0
        )
        assert fid.dtype == np.int64
