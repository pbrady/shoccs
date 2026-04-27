"""Tests for :mod:`stencil_gen.bo` (plan 47)."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from stencil_gen.bo import (
    _BO_SENTINEL,
    BOEval,
    BOResult,
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
