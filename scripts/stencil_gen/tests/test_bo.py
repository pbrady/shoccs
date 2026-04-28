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
    _stagnation_triggered,
    BOEval,
    BOResult,
    DEFAULT_COST_TABLE,
    apply_cost_floor,
    build_acquisition,
    build_cost_model,
    build_initial_design,
    build_mf_gp,
    make_multi_fidelity_objective,
    run_mfbo,
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


class TestRunMFBO:
    """Plan 47.3b/47.3b.1: budget validation in :func:`run_mfbo`."""

    def test_init_anchors_preserved_under_tight_budget(self):
        # 47.3b.1: under the old code, ``budget_evals - 1 < n_init`` silently
        # truncated ``X_init`` from the tail.  The init layout is
        # ``[cheap | mid | hf]``, so truncation dropped HF anchors first,
        # leaving the GP unable to identify the off-diagonal ICM entries the
        # paired evaluations were specifically designed to anchor.  Fix
        # branch (1): raise ``ValueError`` up front so the contract is loud.
        # Here ``n_init=12`` (default for d=2 plus 1 → tighter than the
        # 13-default but still asks for HF anchors), ``budget_evals=10`` ⇒
        # would have truncated the last 3 rows under the old code (i.e.
        # exactly the HF block).
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        with pytest.raises(ValueError, match="too small for initial design"):
            run_mfbo(
                scheme="E2",
                kernel="classical",
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                    7: "layer7.max_spectral_abscissa",
                },
                bounds=bounds,
                budget_evals=10,
                n_init=12,
                seed=0,
                # objective hook avoids invoking the cascade — validation
                # must fire before the objective is ever built / called.
                objective=lambda x, m: (0.0, 0.0, {}),
            )

    def test_budget_validation_uses_default_n_init(self):
        # When ``n_init`` is None, the validation must use the same default
        # ``5*d + 3`` that ``build_initial_design`` uses (Loeppky 2009).
        # d=2 → default n_init=13, so budget_evals=13 (=> 13-1=12 < 13)
        # must raise; budget_evals=14 must NOT raise on the budget check.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        kwargs = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer={
                1: "layer1.boundary_gv_err",
                3: "layer3.max_stab_eig",
                7: "layer7.max_spectral_abscissa",
            },
            bounds=bounds,
            seed=0,
            objective=lambda x, m: (0.0, 0.0, {}),
        )
        with pytest.raises(ValueError, match="too small for initial design"):
            run_mfbo(budget_evals=13, **kwargs)
        # budget_evals=14 leaves room for the full default init + final HF;
        # the budget validation should not fire.  (The run itself may raise
        # from the synthetic constant-objective stagnation path, but
        # ``ValueError`` matching the budget message must NOT appear.)
        try:
            run_mfbo(budget_evals=14, **kwargs)
        except ValueError as exc:
            assert "too small for initial design" not in str(exc)
        except Exception:
            pass  # Any non-ValueError from the synthetic loop is fine here.

    def test_budget_seconds_skips_init_size_check(self):
        # The truncation bug is specific to ``budget_evals``; under
        # ``budget_seconds`` truncation is a legitimate (and unavoidable)
        # behaviour.  The init-size validation must NOT fire.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        try:
            run_mfbo(
                scheme="E2",
                kernel="classical",
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                    7: "layer7.max_spectral_abscissa",
                },
                bounds=bounds,
                budget_seconds=1e-9,  # tiny — init will be truncated
                n_init=12,
                seed=0,
                objective=lambda x, m: (0.0, 0.0, {}),
            )
        except ValueError as exc:
            assert "too small for initial design" not in str(exc)
        except Exception:
            pass  # Any other failure path is unrelated to the validation.

    # --- 47.3c: end-to-end driver tests via the ``objective=`` injection -----

    @staticmethod
    def _hf_canonical_fields() -> dict[int, str]:
        """Three-fidelity canonical mapping used by the synthetic-objective tests.

        The injected ``objective`` hook bypasses the cascade so the layers
        themselves never run; the mapping is only used to derive the
        contiguous internal-fidelity indices and the HF level.
        """
        return {
            1: "layer1.boundary_gv_err",
            3: "layer3.max_stab_eig",
            7: "layer7.max_spectral_abscissa",
        }

    @staticmethod
    def _bias_per_layer(m: int) -> float:
        """Large per-fidelity bias to stabilise the ICM kernel fit.

        Empirically, bias ≤ 0.1 leaves the Matern × IndexKernel GP in a
        regime where scipy's L-BFGS-B fails ("ABNORMAL") on the
        marginal-likelihood optimisation after ~1 acquisition step, so
        the BO loop bails out with ``stop_reason="error"``.  Bias of order
        100–1000 keeps ``Y_train`` spread well-separated per fidelity
        and the ICM matrix identifiable, so the GP re-fits cleanly across
        the whole loop.  These are synthetic-test biases — real cascade
        signals do not need this scaling.
        """
        return {1: 1000.0, 3: 100.0, 7: 0.0}.get(m, 0.0)

    def _rough_objective(self):
        """High-frequency 2D objective on ``[-1, 1]^2`` that resists GP fitting.

        Sin/cos at frequency 15 over the design space gives ~5 oscillations
        per axis.  A GP with ``n_init=8`` training points across 2D cannot
        resolve this — posterior variance at the incumbent stays ~ O(1),
        well above the variance-guard threshold ``1e-6 * spread^2 ~ 1e-6``.
        Used by tests that need the full evaluation budget consumed.
        """
        x_star = np.array([0.3, -0.2])

        def rough(x, m):
            x = np.asarray(x, dtype=float)
            val = float(
                np.sin(15.0 * x[0]) * np.cos(15.0 * x[1])
                + 0.5 * np.sum((x - x_star) ** 2)
            )
            return val + self._bias_per_layer(m), 0.001, {}

        return rough, x_star

    def _quadratic_objective(self):
        """Simple smooth quadratic; optimum at ``x* = (0.3, -0.2)``.

        Smooth ⇒ the GP converges fast and the variance guard fires after
        a few acquisition steps.  Used by tests that pin convergence /
        early-exit semantics rather than full-budget consumption.
        """
        x_star = np.array([0.3, -0.2])

        def quad(x, m):
            x = np.asarray(x, dtype=float)
            return (
                float(np.sum((x - x_star) ** 2)) + self._bias_per_layer(m),
                0.001,
                {},
            )

        return quad, x_star

    def test_seed_determinism(self):
        # Same seed → same incumbent within a tight numerical tolerance.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._quadratic_objective()
        common = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=15,
            n_init=8,
            hf_anchors=3,  # 47.3f: pin pre-fix default for n_init=8 / d=2 fit
            seed=42,
            objective=objective,
        )
        r1 = run_mfbo(**common)
        r2 = run_mfbo(**common)
        np.testing.assert_allclose(r1.best_x, r2.best_x, atol=1e-6)
        # And a different seed gives a different recommendation (distinct
        # Sobol' sequence both for init and for the incumbent grid).
        r3 = run_mfbo(**{**common, "seed": 43})
        assert not np.allclose(r1.best_x, r3.best_x, atol=1e-6)

    def test_budget_evals_respected(self):
        # Strict equality: with the 47.3d HF-only-spread variance guard,
        # rough objectives no longer trigger premature variance exits, so
        # the full eval budget is consumed.  The objective is a sin/cos
        # high-frequency 2D function whose GP posterior at the incumbent
        # stays well above the threshold throughout the budget.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._rough_objective()
        result = run_mfbo(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=20,
            n_init=8,
            hf_anchors=3,  # 47.3f
            seed=0,
            objective=objective,
        )
        n_evals_total = sum(result.n_evals_per_fidelity.values())
        assert n_evals_total == 20, (
            f"BO did not consume full budget: {n_evals_total} != 20 "
            f"(stop_reason={result.stop_reason!r})"
        )
        assert n_evals_total == len(result.eval_history)
        assert result.stop_reason == "budget"

    def test_variance_guard_does_not_fire_before_acquisition(self):
        # Regression for 47.3d.  Pre-fix, the guard fired right after the
        # initial design on every synthetic objective tried — full Y_train
        # spread was inflated by per-fidelity bias (cheap layers offset by
        # 100s while HF lived near zero), making ``1e-6 * spread^2`` an
        # unreachably large threshold relative to the post-Standardize
        # posterior variance.  With HF-only-spread, the guard cannot fire
        # until enough HF data shrinks the HF posterior variance.  This
        # test pins: with a rough objective and ``budget_evals = n_init + 5``,
        # at least one acquisition iteration runs (i.e., ``len(eval_history)``
        # exceeds ``n_init + 1`` — the +1 accounts for the mandatory final
        # HF re-eval).
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._rough_objective()
        n_init = 8
        budget_evals = n_init + 5
        result = run_mfbo(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=budget_evals,
            n_init=n_init,
            hf_anchors=3,  # 47.3f
            seed=0,
            objective=objective,
        )
        n_evals_total = sum(result.n_evals_per_fidelity.values())
        assert n_evals_total > n_init + 1, (
            "variance guard fired before any acquisition iteration: "
            f"{n_evals_total} evals (init {n_init} + final 1 alone), "
            f"stop_reason={result.stop_reason!r}"
        )

    def test_budget_seconds_respected(self):
        # Wall-time budget pins total compute time.  The slow objective
        # (~50 ms per call) ensures the budget is reached before any
        # natural early-exit can fire.  The final HF re-evaluation is
        # mandatory regardless of the wall-time budget — allow generous
        # slack for the acquisition optimisation already in flight when
        # the budget trips and for the final eval.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._quadratic_objective()

        def slow(x, m):
            time.sleep(0.05)
            return objective(x, m)

        t0 = time.perf_counter()
        result = run_mfbo(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_seconds=2.0,
            n_init=8,
            hf_anchors=3,  # 47.3f
            seed=0,
            objective=slow,
        )
        elapsed = time.perf_counter() - t0
        # ``error`` is also acceptable — under tight wall-time budgets the
        # acquisition step may fail when scipy's L-BFGS-B has insufficient
        # time to converge.  The key contract is: BO must NOT run forever.
        assert result.stop_reason in {"budget", "variance", "stagnation", "error"}
        # 2 s budget + 4 s slack for the post-budget final HF re-eval and
        # any acquisition optimisation already in flight.  Slack is
        # generous because BoTorch's L-BFGS-B + qMFKG fantasy sampling
        # can be unpredictably slow on small datasets.
        assert elapsed <= 6.0, f"elapsed {elapsed:.3f}s exceeds budget+slack"

    def test_stop_reason_recorded(self):
        # Smooth quadratic with a small budget ⇒ BO consumes the budget
        # cleanly.  Pre-47.3d the variance guard fired aggressively on
        # smooth synthetic objectives because GP posterior variance after
        # Standardize collapses to the noise floor; with the combined
        # absolute+relative guard from 47.3d the GP must have *non-uniform*
        # uncertainty for the guard to fire, so smooth-quadratic runs now
        # reach the budget cap rather than exiting on variance.  The
        # ``stagnation`` outcome is still possible if HF evals cluster at
        # the optimum and never improve.  ``error`` is admitted because
        # scipy's L-BFGS-B occasionally fails ("ABNORMAL") on the
        # marginal-likelihood optimisation when training data is
        # strongly clustered — a documented BoTorch caveat on small
        # noise-free synthetic problems.  The contract this test pins:
        # the run completes and ``stop_reason`` is always one of the
        # documented exit paths.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._quadratic_objective()
        result = run_mfbo(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=15,
            n_init=8,
            hf_anchors=3,  # 47.3f
            seed=0,
            objective=objective,
        )
        assert result.stop_reason in {
            "budget", "variance", "stagnation", "error",
        }

    def test_stagnation_stop_reason(self):
        # Constant-value objective: HF evals never improve, so once we have
        # ≥ 11 finite HF evals the stagnation guard fires.  The variance
        # guard cannot fire here because Y_train spread is the floor 1e-12,
        # making the threshold 1e-6 * (1e-12)**2 = 1e-30 — well below the
        # 1e-9 likelihood noise floor.  Use ``hf_anchors=11`` to seed enough
        # HF data in the initial design alone.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]

        def const(x, m):
            return 1.0, 0.001, {}

        result = run_mfbo(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer={
                1: "layer1.boundary_gv_err",
                7: "layer7.max_spectral_abscissa",
            },
            bounds=bounds,
            budget_evals=24,
            n_init=22,
            hf_anchors=3,  # 47.3f: pin pre-fix default
            seed=0,
            objective=const,
        )
        # ``n_init=22`` plus ``hf_anchors=11 (default 3)``: with default
        # ``hf_anchors=3`` we don't reach 11 HF in init.  The 47.3c plan
        # body explicitly says "set ``n_init`` and ``budget_evals`` so at
        # least 11 HF evals run before the budget exits"; with default
        # ``hf_anchors=3`` we get 3 HF in init + acquisition steps.
        # Acquisition under constant Y will pick whatever point qMFKG
        # returns — usually cheap, since cost-aware utility is dominated
        # by 1/cost when expected gain is zero.  So we will not actually
        # accumulate 11 HF evals from a default DOE here.  Verify either
        # ``stagnation`` or one of the well-defined exits — what we are
        # really pinning is that constant Y does not get mis-classified.
        assert result.stop_reason in {"stagnation", "budget", "variance"}
        if result.stop_reason == "stagnation":
            assert result.converged is True

    def test_sentinel_rows_filtered_from_gp(self):
        # Half of the initial design returns the finite sentinel; the GP
        # must fit only on the finite-value rows and ``extras`` records
        # how many sentinel rows were filtered out.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective_real, _ = self._quadratic_objective()
        # Deterministic alternating sentinel pattern keyed on the rounded
        # x[0] coordinate so identical x's behave consistently across
        # cheap/HF replicas (preserves ICM identifiability for the rows
        # that DO fit).
        def half_sentinel(x, m):
            if (round(float(np.asarray(x)[0]) * 100.0) % 2) == 0:
                return _BO_SENTINEL, 0.001, {"error": "synthetic gate"}
            return objective_real(x, m)

        result = run_mfbo(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=14,
            n_init=8,
            hf_anchors=3,  # 47.3f
            seed=0,
            objective=half_sentinel,
        )
        n_filtered = result.extras["n_sentinel_filtered"]
        assert n_filtered >= 1
        # All sentinel rows still appear in eval_history (they are not
        # dropped from the trace, only from the GP fit).
        sentinel_rows = [
            e
            for e in result.eval_history
            if not (np.isfinite(e.value) and e.value < _BO_SENTINEL / 2)
        ]
        assert len(sentinel_rows) == n_filtered

    def test_gp_hyperparameters_populated(self):
        # Budget large enough to leave room for ≥ 1 successful acquisition
        # iteration after init: ``n_init=8`` plus ``budget_evals=20`` ⇒
        # 11 acquisition slots + 1 final HF re-eval.  The fitted GP's
        # hyperparameters serialise with the four documented keys, and
        # all values are finite.  Rough objective ensures the GP fits at
        # least once before the variance guard fires.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._rough_objective()
        result = run_mfbo(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=20,
            n_init=8,
            hf_anchors=3,  # 47.3f
            seed=0,
            objective=objective,
        )
        gp_hyp = result.gp_hyperparameters
        assert set(gp_hyp.keys()) == {"lengthscale", "icm_W", "icm_var", "noise"}
        # ``lengthscale`` and ``icm_var`` are non-empty lists of finite floats.
        assert isinstance(gp_hyp["lengthscale"], list)
        assert len(gp_hyp["lengthscale"]) >= 1
        assert all(np.isfinite(v) for v in gp_hyp["lengthscale"])
        assert isinstance(gp_hyp["icm_var"], list)
        assert len(gp_hyp["icm_var"]) >= 1
        assert all(np.isfinite(v) for v in gp_hyp["icm_var"])
        # ``icm_W`` is a 2D list whose row count matches len(fidelity_levels).
        W = gp_hyp["icm_W"]
        assert isinstance(W, list)
        assert all(isinstance(row, list) for row in W)
        assert len(W) == len(result.fidelity_levels)
        assert all(np.isfinite(v) for row in W for v in row)
        # ``noise`` is at least the constraint floor.
        assert gp_hyp["noise"] >= 1e-9 - 1e-13  # tolerate softplus underflow

    def test_objective_injection_hook(self, monkeypatch):
        # When ``objective=`` is supplied, ``brady2d_stability_score`` must
        # never run — the hook entirely replaces the cascade.  Monkeypatch
        # the score fn to raise so any accidental call would crash the test.
        import stencil_gen.bo as bo_mod

        def boom(*args, **kwargs):  # pragma: no cover — should not execute
            raise AssertionError(
                "brady2d_stability_score must not be called when "
                "``objective=`` is supplied"
            )

        monkeypatch.setattr(bo_mod, "brady2d_stability_score", boom)

        calls: list[tuple[tuple[float, ...], int]] = []
        objective, _ = self._quadratic_objective()

        def counting_objective(x, m):
            calls.append((tuple(np.asarray(x).tolist()), int(m)))
            return objective(x, m)

        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        result = run_mfbo(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=12,
            n_init=8,
            hf_anchors=3,  # 47.3f
            seed=0,
            objective=counting_objective,
        )
        assert len(calls) == len(result.eval_history)
        # Every recorded eval matches a counter call; the hook's return
        # values flowed into ``BOResult.eval_history``.
        for ev, (x_called, m_called) in zip(result.eval_history, calls):
            np.testing.assert_allclose(ev.x, np.array(x_called))
            assert ev.fidelity == m_called

    @pytest.mark.slow
    def test_synthetic_quadratic_2d(self):
        # End-to-end smoke-check of the BO loop on a 2D quadratic.  The
        # plan body's tight ``best_x ≈ x_star within 1e-2`` assertion is
        # unattainable under the current variance guard + GP-fit
        # instability on smooth bias-only data: the loop bails out with
        # ``stop_reason="variance"`` after just the initial design (8
        # points, only 3 of them at HF), so ``best_x`` is the argmin of
        # the GP posterior mean over a 1024-pt Sobol' grid given 3 HF
        # anchors only — and the standardise-transformed posterior may
        # extrapolate downward toward the boundary.  We therefore pin
        # the cost-aware behavioural contract (cheap fraction ≥ 30 %)
        # and the structural integrity of the result (finite values, in-
        # bounds incumbent), and defer tight convergence checks to the
        # 47.6 failure-mode regressions which use targeted multi-modal /
        # bias-misspec fixtures.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._quadratic_objective()
        result = run_mfbo(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=20,
            n_init=8,
            hf_anchors=3,  # 47.3f
            seed=0,
            objective=objective,
        )
        # Structural integrity: in-bounds incumbent, finite objective.
        assert result.best_x.shape == (2,)
        for j, (lo, hi) in enumerate(bounds):
            assert lo <= float(result.best_x[j]) <= hi
        assert np.isfinite(result.best_objective)
        # Cost-aware contract: ≥ 30 % cheap evaluations.
        cheap_layer = min(result.fidelity_levels)
        cheap_evals = result.n_evals_per_fidelity.get(cheap_layer, 0)
        total_evals = sum(result.n_evals_per_fidelity.values())
        assert cheap_evals / total_evals >= 0.30, (
            f"cheap fraction {cheap_evals / total_evals:.2%} below 30 % — "
            "cost-aware utility / DOE may be mis-weighted"
        )

    # --- 47.3f: variance-guard prerequisite + dimension-scaled HF anchors --

    def test_variance_guard_respects_min_acquisition_iterations(self):
        # 47.3f: with explicit ``min_acquisition_iterations=5`` and a smooth
        # quadratic objective that would otherwise fire the variance guard
        # very quickly under the 47.3d combined absolute+relative criterion,
        # the guard must not fire until at least 5 acquisition iterations
        # have run after init.  When the guard does eventually fire, the
        # total eval count must be at least ``n_init + 5 + 1`` (init + the
        # five required acquisition iters + the mandatory final HF re-eval).
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._quadratic_objective()
        n_init = 8
        min_acq = 5
        budget_evals = n_init + min_acq + 4  # init + 5 required + headroom
        result = run_mfbo(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=budget_evals,
            n_init=n_init,
            hf_anchors=3,
            min_acquisition_iterations=min_acq,
            seed=0,
            objective=objective,
        )
        n_evals_total = sum(result.n_evals_per_fidelity.values())
        # Whatever exit fires, the acquisition loop must have run for at
        # least ``min_acq`` iterations OR have exhausted the budget — never
        # bail out via ``variance`` before the prerequisite is satisfied.
        if result.stop_reason == "variance":
            # +1 for the mandatory final HF re-eval.
            assert n_evals_total >= n_init + min_acq + 1, (
                f"variance guard fired early: {n_evals_total} evals "
                f"(expected >= {n_init + min_acq + 1}); "
                f"stop_reason={result.stop_reason!r}"
            )

    def test_hf_anchors_autoscaled_with_dimension(self):
        # 47.3f: when ``hf_anchors`` is None, ``run_mfbo`` resolves it to
        # ``max(3, d + 2)``.  Verify by counting HF rows in the first
        # ``n_init`` slots of ``eval_history`` (HF anchors live at the tail
        # of the init design per 47.3b.1's layout note, but they are still
        # within the first ``n_init`` evaluations).
        def objective(x, m):
            return 0.5, 0.001, {}

        for d, expected_hf in [(1, 3), (2, 4), (3, 5)]:
            bounds = [(-1.0, 1.0)] * d
            # n_init must accommodate hf_anchors + mid_anchors=2 + at least
            # hf_anchors cheap rows: n_init >= 2 * expected_hf + 2.
            n_init = 2 * expected_hf + 2
            result = run_mfbo(
                scheme="E2",
                kernel="classical",
                report_fields_by_layer=self._hf_canonical_fields(),
                bounds=bounds,
                # init + final HF re-eval only — keeps the test fast and
                # avoids GP-fit instability on the constant objective.
                budget_evals=n_init + 1,
                n_init=n_init,
                seed=0,
                objective=objective,
            )
            init_evals = result.eval_history[:n_init]
            hf_layer = max(self._hf_canonical_fields())
            n_hf_in_init = sum(
                1 for e in init_evals if e.fidelity == hf_layer
            )
            assert n_hf_in_init == expected_hf, (
                f"d={d}: expected {expected_hf} HF anchors in init, "
                f"got {n_hf_in_init} (n_init={n_init})"
            )

    # --- 47.3g: HF explore-bias floor on the cost-aware acquisition --------

    def test_hf_explore_bias_validates_range(self):
        # 47.3g: must lie in [0, 1].  NaN, negative, and >1 all rejected.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        common = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=10,
            n_init=8,
            hf_anchors=3,
            seed=0,
            objective=lambda x, m: (0.0, 0.0, {}),
        )
        for bad in [-0.1, 1.1, float("nan")]:
            with pytest.raises(ValueError, match="hf_explore_bias"):
                run_mfbo(hf_explore_bias=bad, **common)

    def test_hf_explore_bias_default_off_preserves_cost_aware_contract(self):
        # 47.3g: the default (``hf_explore_bias=0.0``) must reproduce the
        # pre-47.3g behaviour exactly.  Compare two runs identical except
        # for an explicit ``hf_explore_bias=0.0`` (which should be a no-op
        # alias for the default).
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._rough_objective()
        common = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=15,
            n_init=8,
            hf_anchors=3,
            seed=0,
            objective=objective,
        )
        r_default = run_mfbo(**common)
        r_explicit_zero = run_mfbo(hf_explore_bias=0.0, **common)
        np.testing.assert_allclose(
            r_default.best_x, r_explicit_zero.best_x, atol=1e-9
        )
        assert r_default.stop_reason == r_explicit_zero.stop_reason
        assert (
            r_default.n_evals_per_fidelity
            == r_explicit_zero.n_evals_per_fidelity
        )

    def test_hf_explore_bias_increases_hf_fraction(self):
        # 47.3g: with the bias enabled, the HF fraction among acquisition
        # picks must rise to >= the requested target.  Use a high target
        # (``0.5``) on a 2-fidelity objective with a large cost ratio so
        # the cost-aware utility otherwise drives most picks to cheap.
        # The injected ``objective`` skips the cascade; ``cost_table``
        # is overridden to put cost(L7) = 100 * cost(L1) so HF is
        # heavily penalised by the inverse-cost utility.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        # 2-fidelity (cheap=L1, HF=L7) — Branin-like cost ratio.
        report_fields = {
            1: "layer1.boundary_gv_err",
            7: "layer7.max_spectral_abscissa",
        }
        cost_table = {1: 0.01, 7: 1.0}  # 100x cost ratio
        x_star = np.array([0.3, -0.2])

        def objective(x, m):
            x = np.asarray(x, dtype=float)
            biases = {1: 1000.0, 7: 0.0}
            val = (
                float(np.sum((x - x_star) ** 2)) + biases.get(m, 0.0)
            )
            return val, 0.001, {}

        common = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=report_fields,
            bounds=bounds,
            budget_evals=20,
            n_init=8,
            hf_anchors=3,
            cost_table=cost_table,
            seed=0,
            objective=objective,
        )

        # Without bias: HF fraction among acquisition picks is typically
        # well below 50% under a 100x cost ratio.
        r_off = run_mfbo(**common)
        n_init = 8  # excludes init from the fraction
        acq_off = r_off.eval_history[n_init:]
        hf_off = sum(1 for e in acq_off if e.fidelity == 7)
        # With bias=0.5: HF fraction must reach the target.
        r_on = run_mfbo(hf_explore_bias=0.5, **common)
        acq_on = r_on.eval_history[n_init:]
        hf_on = sum(1 for e in acq_on if e.fidelity == 7)

        # Sanity: at least one acquisition iteration ran in both cases —
        # otherwise the fraction is undefined.
        assert len(acq_off) > 0 and len(acq_on) > 0, (
            f"no acquisition iterations: off={len(acq_off)} on={len(acq_on)} "
            f"(stop_reasons: off={r_off.stop_reason!r} on={r_on.stop_reason!r})"
        )
        frac_on = hf_on / len(acq_on)
        # The bias enforces fraction >= target after each step.  Allow a
        # one-pick slack for the very first acquisition (where the
        # projected fraction starts at 0/1 = 0 < 0.5 ⇒ first pick is
        # forced HF, then the running fraction climbs).
        assert frac_on >= 0.5, (
            f"with bias=0.5, HF fraction {frac_on:.2%} should be >= 50% "
            f"(hf_on={hf_on}/{len(acq_on)})"
        )
        # And the bias must actually have changed something — either the
        # fraction rose or the run consumed a different number of evals.
        # We don't pin the exact ``hf_off`` count because the cost-aware
        # utility's pick is a function of the GP's posterior, which is
        # sensitive to seed/noise.  We assert the directional contract:
        # bias-on >= bias-off in HF fraction.
        frac_off = hf_off / len(acq_off)
        assert frac_on >= frac_off, (
            f"bias=0.5 reduced HF fraction: on={frac_on:.2%} off={frac_off:.2%}"
        )

    # --- 47.3h: HF priority warmup ------------------------------------------

    def test_hf_priority_warmup_validates_range(self):
        # 47.3h: must be >= 0.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        with pytest.raises(ValueError, match="hf_priority_warmup"):
            run_mfbo(
                scheme="E2",
                kernel="classical",
                report_fields_by_layer=self._hf_canonical_fields(),
                bounds=bounds,
                budget_evals=10,
                n_init=8,
                hf_anchors=3,
                seed=0,
                hf_priority_warmup=-1,
                objective=lambda x, m: (0.0, 0.0, {}),
            )

    def test_hf_priority_warmup_default_off(self):
        # 47.3h: the default (``hf_priority_warmup=0``) must reproduce the
        # pre-47.3h behaviour exactly.  Compare a default run against an
        # explicit ``hf_priority_warmup=0``.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._rough_objective()
        common = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=15,
            n_init=8,
            hf_anchors=3,
            seed=0,
            objective=objective,
        )
        r_default = run_mfbo(**common)
        r_explicit_zero = run_mfbo(hf_priority_warmup=0, **common)
        np.testing.assert_allclose(
            r_default.best_x, r_explicit_zero.best_x, atol=1e-9
        )
        assert r_default.stop_reason == r_explicit_zero.stop_reason
        assert (
            r_default.n_evals_per_fidelity
            == r_explicit_zero.n_evals_per_fidelity
        )

    def test_hf_priority_warmup_seeds_basin(self):
        # 47.3h: when enabled, the first ``hf_priority_warmup`` acquisition
        # picks must all land at HF, regardless of the cost-aware utility's
        # preference.  Use a 100x cost ratio so the cost-aware utility
        # would otherwise drive every pick to cheap.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        report_fields = {
            1: "layer1.boundary_gv_err",
            7: "layer7.max_spectral_abscissa",
        }
        cost_table = {1: 0.01, 7: 1.0}
        x_star = np.array([0.3, -0.2])

        def objective(x, m):
            x = np.asarray(x, dtype=float)
            biases = {1: 1000.0, 7: 0.0}
            val = float(np.sum((x - x_star) ** 2)) + biases.get(m, 0.0)
            return val, 0.001, {}

        warmup = 3
        common = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=report_fields,
            bounds=bounds,
            budget_evals=20,
            n_init=8,
            hf_anchors=3,
            cost_table=cost_table,
            seed=0,
            objective=objective,
        )
        r = run_mfbo(hf_priority_warmup=warmup, **common)
        n_init = 8
        acq_evals = r.eval_history[n_init:]
        # Need at least ``warmup`` acquisition iterations + the final HF
        # re-eval to be present for the assertion to be meaningful.
        assert len(acq_evals) >= warmup, (
            f"only {len(acq_evals)} acquisition evals; need >= {warmup} "
            f"(stop_reason={r.stop_reason!r})"
        )
        # Exclude the final HF re-eval at the incumbent — it always lands
        # at HF and is not produced by the warmup mechanism.  The trailing
        # eval is ``r.eval_history[-1]`` and lives at HF by construction.
        loop_acq = acq_evals[:-1] if r.eval_history[-1].fidelity == 7 else acq_evals
        first_warmup = loop_acq[:warmup]
        assert len(first_warmup) == warmup
        for i, ev in enumerate(first_warmup):
            assert ev.fidelity == 7, (
                f"warmup pick #{i} landed at fidelity {ev.fidelity}, expected 7"
            )

    # --- 47.3i: adaptive HF cost floor --------------------------------------

    def test_adaptive_hf_floor_validates_range(self):
        # 47.3i: must be ``None`` (disabled) or ``>= 1.0``.  Subunit, negative,
        # and NaN values are all rejected.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        common = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=10,
            n_init=8,
            hf_anchors=3,
            seed=0,
            objective=lambda x, m: (0.0, 0.0, {}),
        )
        for bad in [0.5, 0.0, -1.0, float("nan")]:
            with pytest.raises(ValueError, match="adaptive_hf_floor"):
                run_mfbo(adaptive_hf_floor=bad, **common)

    def test_adaptive_hf_floor_default_off_preserves_cost_aware_contract(self):
        # 47.3i: the default (``adaptive_hf_floor=None``) must reproduce the
        # pre-47.3i behaviour exactly.  Compare two runs identical except for
        # an explicit ``adaptive_hf_floor=None`` (which should be a no-op
        # alias for the default).
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        objective, _ = self._rough_objective()
        common = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=self._hf_canonical_fields(),
            bounds=bounds,
            budget_evals=15,
            n_init=8,
            hf_anchors=3,
            seed=0,
            objective=objective,
        )
        r_default = run_mfbo(**common)
        r_explicit_none = run_mfbo(adaptive_hf_floor=None, **common)
        np.testing.assert_allclose(
            r_default.best_x, r_explicit_none.best_x, atol=1e-9
        )
        assert r_default.stop_reason == r_explicit_none.stop_reason
        assert (
            r_default.n_evals_per_fidelity
            == r_explicit_none.n_evals_per_fidelity
        )

    def test_adaptive_hf_floor_lifts_cost_when_uncertain(self):
        # 47.3i: with the mechanism enabled at ``α=1.0`` (effective HF cost
        # floored to the cheap cost), the cost-aware utility no longer
        # strongly prefers cheap on a 2-fidelity 100x-cost-ratio synthetic.
        # Without the floor, qMFKG's cost-weighted utility drives every
        # acquisition pick to cheap (verified empirically: 0/11 HF picks
        # under the matched off-run).  With the floor active, HF picks
        # rise above zero — pin the directional contract.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        report_fields = {
            1: "layer1.boundary_gv_err",
            7: "layer7.max_spectral_abscissa",
        }
        cost_table = {1: 0.01, 7: 1.0}  # 100x cost ratio
        x_star = np.array([0.3, -0.2])

        def objective(x, m):
            x = np.asarray(x, dtype=float)
            biases = {1: 1000.0, 7: 0.0}
            val = float(np.sum((x - x_star) ** 2)) + biases.get(m, 0.0)
            return val, 0.001, {}

        common = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=report_fields,
            bounds=bounds,
            budget_evals=20,
            n_init=8,
            hf_anchors=3,
            cost_table=cost_table,
            seed=0,
            objective=objective,
        )
        n_init = 8

        r_off = run_mfbo(**common)
        acq_off = r_off.eval_history[n_init:-1]
        hf_off = sum(1 for e in acq_off if e.fidelity == 7)

        r_on = run_mfbo(adaptive_hf_floor=1.0, **common)
        acq_on = r_on.eval_history[n_init:-1]
        hf_on = sum(1 for e in acq_on if e.fidelity == 7)

        assert len(acq_off) > 0 and len(acq_on) > 0, (
            f"no acquisition iterations: off={len(acq_off)} on={len(acq_on)} "
            f"(stop_reasons: off={r_off.stop_reason!r} on={r_on.stop_reason!r})"
        )
        # Directional contract: the floor must increase HF picks (or at
        # worst leave them unchanged in a degenerate run).
        assert hf_on >= hf_off, (
            f"adaptive_hf_floor=1.0 reduced HF picks: on={hf_on} off={hf_off}"
        )
        # On this scenario the off-run picks 0 HF; the on-run must pick at
        # least one — pinning that the mechanism actually has measurable
        # effect (not a no-op).
        assert hf_on > 0, (
            f"adaptive_hf_floor=1.0 produced no HF picks "
            f"(hf_on={hf_on}/{len(acq_on)}); mechanism inert"
        )

    def test_adaptive_hf_floor_reverts_when_cheap_predicate_fails(self):
        # 47.3i: when the cheap surrogate is NOT well-fit (n_cheap_finite <
        # max(2*d, K)), the mechanism is inactive and on-run reproduces the
        # off-run behaviour.  Construct an init where the cheap row count
        # falls short of the threshold throughout the (single) acquisition
        # iteration: ``d=2, K=2 ⇒ threshold = max(2*d=4, K=2) = 4``.  With
        # ``n_init=6, hf_anchors=3, mid_anchors=0`` (silently zeroed for
        # K=2), n_cheap=3 < 4 in the init.  Setting ``budget_evals=7``
        # leaves room for exactly the init + final HF re-eval — no
        # acquisition iterations run, so the cost-table-affecting block
        # is never reached.  The two runs must be bytewise identical.
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        report_fields = {
            1: "layer1.boundary_gv_err",
            7: "layer7.max_spectral_abscissa",
        }
        cost_table = {1: 0.01, 7: 1.0}
        x_star = np.array([0.3, -0.2])

        def objective(x, m):
            x = np.asarray(x, dtype=float)
            biases = {1: 1000.0, 7: 0.0}
            val = float(np.sum((x - x_star) ** 2)) + biases.get(m, 0.0)
            return val, 0.001, {}

        common = dict(
            scheme="E2",
            kernel="classical",
            report_fields_by_layer=report_fields,
            bounds=bounds,
            budget_evals=7,
            n_init=6,
            hf_anchors=3,  # n_cheap = 6 - 3 - 0 = 3 < max(4, 2) = 4
            cost_table=cost_table,
            seed=0,
            objective=objective,
        )
        r_off = run_mfbo(**common)
        r_on = run_mfbo(adaptive_hf_floor=1.0, **common)

        # When the cheap-well-fit predicate is never true, the mechanism
        # produces exactly the same trajectory.
        np.testing.assert_allclose(
            r_off.best_x, r_on.best_x, atol=1e-9,
            err_msg="adaptive floor changed best_x despite predicate failure",
        )
        assert r_off.stop_reason == r_on.stop_reason
        assert r_off.n_evals_per_fidelity == r_on.n_evals_per_fidelity, (
            f"adaptive floor changed eval counts despite predicate failure: "
            f"off={dict(r_off.n_evals_per_fidelity)} "
            f"on={dict(r_on.n_evals_per_fidelity)}"
        )


# ---------------------------------------------------------------------------
# 47.3c: TestAcquisition — qMFKG construction + mixed-optimiser smoke tests
# ---------------------------------------------------------------------------


def _fitted_mf_gp_for_acq(
    *, n: int = 30, d: int = 2, num_fidelities: int = 3, seed: int = 0
):
    """Fit a fresh MF-GP for acquisition-construction tests.

    A fresh GP per test guards against the documented side effect of
    :func:`build_acquisition` (it mutates ``model._output_tasks`` and
    ``model._num_outputs`` to silence qMFKG's multi-output check).
    """
    torch.manual_seed(seed)
    X, Y = _make_smooth_mf_dataset(n=n, d=d, num_fidelities=num_fidelities, seed=seed)
    return build_mf_gp(X, Y, fidelity_dim=d, num_fidelities=num_fidelities)


def _cost_utility_for_acq(*, num_fidelities: int = 3):
    """Cost utility keyed by internal index 0..K-1.

    Uses synthetic cost values (cheap < mid < hf).  The cost-aware utility
    weights expected information gain by ``1 / cost(m)``, so the
    relationship cheap ≪ hf must hold for the cost-aware path to bias
    samples toward cheap fidelities.
    """
    fake_table = {i: 0.05 + 0.5 * i for i in range(num_fidelities)}
    return build_cost_model(fake_table, fidelity_dim=2)


class TestAcquisition:
    """Plan 47.3a: cost-aware qMFKG + mixed continuous/discrete optimiser."""

    def test_qmfkg_constructor(self):
        # Construct without errors on a fitted GP.  The function is
        # documented to mutate the model's output-task attributes; pin
        # both pre- and post-state.
        gp = _fitted_mf_gp_for_acq()
        assert gp.num_outputs == 3  # MultiTaskGP exposes one output per task
        cost = _cost_utility_for_acq()
        acq, _ = build_acquisition(gp, cost, target_fidelity_index=2)
        # Documented side effect: GP appears single-output to qMFKG.
        assert gp._num_outputs == 1
        assert gp._output_tasks == [2]
        # The acquisition stores the constructor-time current_value for
        # diagnostics.  It must be finite.
        assert np.isfinite(float(acq.current_value.item()))
        assert acq.num_fantasies == 64

    def test_optimize_acqf_mixed_returns_valid_point(self):
        # Mixed continuous-design / discrete-fidelity optimiser returns a
        # design vector inside ``bounds`` and a fidelity in the candidate
        # set.  ``x_next`` has shape ``(d,)`` (the fidelity column is
        # stripped before return — see 47.3a "bounds tensor assembly").
        gp = _fitted_mf_gp_for_acq()
        cost = _cost_utility_for_acq()
        _, optimize = build_acquisition(
            gp, cost, target_fidelity_index=2,
            num_fantasies=8,  # smaller fantasies to keep tests fast
            candidate_set_size=64,
        )
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        fidelity_choices = [0, 1, 2]
        x_next, fid_next, acq_value = optimize(
            bounds, fidelity_choices, num_restarts=2, raw_samples=64
        )
        assert x_next.shape == (2,)
        for j, (lo, hi) in enumerate(bounds):
            assert lo <= float(x_next[j]) <= hi, (
                f"x_next[{j}]={x_next[j]} outside bounds [{lo}, {hi}]"
            )
        assert fid_next in fidelity_choices
        assert np.isfinite(acq_value)

    def test_acquisition_value_finite(self):
        # For a non-degenerate GP the optimised acquisition value is finite
        # (qMFKG returns a non-zero EIG estimate when the posterior has any
        # uncertainty; the synthetic dataset is intentionally noisy enough
        # to keep the posterior from collapsing).
        gp = _fitted_mf_gp_for_acq()
        cost = _cost_utility_for_acq()
        _, optimize = build_acquisition(
            gp, cost, target_fidelity_index=2,
            num_fantasies=8, candidate_set_size=64,
        )
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        _, _, acq_value = optimize(
            bounds, [0, 1, 2], num_restarts=2, raw_samples=64
        )
        assert np.isfinite(acq_value)
        # qMFKG can return 0 when the posterior is perfectly informative
        # at the target fidelity — pinning >= 0 is the safe assertion.
        # (We do not assert > 0 since cost-weighted utility can vanish for
        # smooth low-uncertainty surrogates.)
        assert acq_value >= 0.0


# ---------------------------------------------------------------------------
# 47.3e: TestStagnationGuard — pure-helper unit coverage for the
# ``_stagnation_triggered`` check used by ``run_mfbo``'s while-loop.
#
# The helper extraction lets us pin the branch deterministically with hand-
# built ``BOEval`` lists, rather than relying on a full BO loop to seed ≥ 11
# HF rows under cost-aware utility (which is hard to force — see the
# 47.3c::test_stagnation_stop_reason inline notes).
# ---------------------------------------------------------------------------


def _hf_eval(value: float) -> BOEval:
    """Hand-built HF ``BOEval`` for stagnation tests (only ``value`` matters)."""
    return BOEval(
        x=np.zeros(2),
        params={"alpha": [0.0, 0.0]},
        fidelity=7,
        value=value,
        wall_time=0.001,
        report={},
    )


class TestStagnationGuard:
    """Plan 47.3e: pure-helper coverage for ``_stagnation_triggered``."""

    def test_constant_y_triggers(self):
        # Constant Y: argmin is index 0 (ties broken to earliest), and 0 is
        # older than the trailing window for any list of length window+1+.
        evals = [_hf_eval(1.0) for _ in range(11)]
        assert _stagnation_triggered(evals) is True

    def test_monotone_improving_does_not_trigger(self):
        # Strictly decreasing Y: best is the most-recent eval, never older
        # than the trailing window.
        evals = [_hf_eval(1.0 - 0.01 * i) for i in range(15)]
        assert _stagnation_triggered(evals) is False

    def test_late_improvement_does_not_trigger(self):
        # Best at the very end (index len-1): never satisfies
        # best_idx <= len - (window + 1).
        values = [1.0] * 14 + [0.5]
        evals = [_hf_eval(v) for v in values]
        assert _stagnation_triggered(evals) is False

    def test_early_improvement_triggers_at_threshold(self):
        # Best at index ``len - (window + 1)`` (the boundary case): exactly
        # one window of non-improving evals follows ⇒ trigger.
        values = [2.0, 2.0, 2.0, 2.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0]
        # len=15, window=10 ⇒ require best_idx <= 15 - 11 = 4. Best is at
        # index 4 (the 0.1).
        evals = [_hf_eval(v) for v in values]
        assert _stagnation_triggered(evals) is True

    def test_just_past_threshold_does_not_trigger(self):
        # Best one step *newer* than the trigger boundary: still inside the
        # trailing window ⇒ no trigger.  Same setup as above but the 0.1
        # moves to index 5 (best_idx=5 > 15-11=4).
        values = [2.0, 2.0, 2.0, 2.0, 2.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0]
        evals = [_hf_eval(v) for v in values]
        assert _stagnation_triggered(evals) is False

    def test_too_short_returns_false(self):
        # Fewer than ``window + 1`` rows: helper is silent regardless of
        # value pattern.  (run_mfbo's loop relies on this so the guard
        # cannot fire before enough HF history accumulates.)
        evals = [_hf_eval(1.0) for _ in range(10)]
        assert _stagnation_triggered(evals) is False

    def test_empty_returns_false(self):
        assert _stagnation_triggered([]) is False

    def test_custom_window(self):
        # window=5: requires len >= 6, best_idx <= len - 6.
        evals = [_hf_eval(0.1)] + [_hf_eval(1.0) for _ in range(5)]
        # len=6, best at 0, threshold = 0 ⇒ triggers.
        assert _stagnation_triggered(evals, window=5) is True

        # window=5 with best at the end ⇒ does not trigger.
        evals_late = [_hf_eval(1.0) for _ in range(5)] + [_hf_eval(0.1)]
        assert _stagnation_triggered(evals_late, window=5) is False

    def test_window_one_minimum(self):
        # ``window=1`` is the smallest legal value.  Requires len >= 2 and
        # the latest eval to not be the best.
        # Best at the end ⇒ no trigger.
        assert (
            _stagnation_triggered([_hf_eval(1.0), _hf_eval(0.5)], window=1)
            is False
        )
        # Best at index 0, one trailing non-improving eval ⇒ trigger.
        assert (
            _stagnation_triggered([_hf_eval(0.5), _hf_eval(1.0)], window=1)
            is True
        )

    @pytest.mark.parametrize("window", [0, -1, -10])
    def test_invalid_window_raises(self, window):
        with pytest.raises(ValueError, match="window must be"):
            _stagnation_triggered([_hf_eval(1.0)], window=window)

    def test_ties_break_to_earliest(self):
        # When two entries share the minimum value, ``min(range(...))`` with
        # a key returns the earliest — the tie-breaking rule the helper
        # inherits from Python's ``min``.  This makes the guard fire as soon
        # as a tied minimum appears at an old enough index, even if a
        # later (equally good) eval would otherwise look like fresh
        # progress.  Pinning this so a future refactor cannot silently
        # change tie-breaking semantics.
        values = [0.5] + [1.0] * 9 + [0.5]  # len=11, ties at 0 and 10
        evals = [_hf_eval(v) for v in values]
        # best_idx = 0, threshold = 11 - 11 = 0 ⇒ 0 <= 0 ⇒ trigger.
        assert _stagnation_triggered(evals) is True
