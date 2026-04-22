"""Tests for :mod:`stencil_gen.pareto` (plan 45)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from stencil_gen.brady2d_stability import StabilityReport
from stencil_gen.pareto import (
    _PARETO_SENTINEL,
    ParetoPoint,
    ParetoResult,
    make_multi_objective,
)


def _empty_report_with(**layers) -> StabilityReport:
    """Build a ``StabilityReport`` with the given layer payloads populated."""
    r = StabilityReport.empty()
    for name, value in layers.items():
        setattr(r, name, value)
    return r


class TestParetoDataclasses:
    """Plan 45.1a: ``ParetoPoint`` and ``ParetoResult`` are frozen dataclasses."""

    @staticmethod
    def _make_point(offset: float = 0.0) -> ParetoPoint:
        return ParetoPoint(
            x=np.array([offset, offset + 1.0]),
            params={"alpha": [offset, offset + 1.0]},
            objectives=np.array([offset * 2, offset * 2 + 1]),
            report={"failed_layer": None},
        )

    @staticmethod
    def _make_result(front: tuple[ParetoPoint, ...]) -> ParetoResult:
        return ParetoResult(
            front=front,
            objective_fields=(
                "layer1.boundary_gv_err",
                "layer_bl42.max_spectral_abscissa",
            ),
            scheme="E4",
            kernel="classical",
            bounds=((-2.0, 2.0), (0.05, 2.0)),
            method="NSGA-II",
            pop_size=20,
            n_gen=10,
            n_evals=200,
            seed=1,
            compute_time=1.23,
            hv_trace=(0.1, 0.2, 0.3),
            ref_point=(1.0, 1.0),
            extras={"n_sentinel_filtered": 0},
        )

    def test_pareto_point_frozen(self):
        pt = self._make_point()
        with pytest.raises(FrozenInstanceError):
            pt.x = np.array([99.0, 99.0])
        with pytest.raises(FrozenInstanceError):
            pt.objectives = np.array([0.0, 0.0])

    def test_pareto_result_frozen(self):
        front = tuple(self._make_point(i) for i in range(3))
        result = self._make_result(front)
        assert len(result.front) == 3
        with pytest.raises(FrozenInstanceError):
            result.front = ()
        with pytest.raises(FrozenInstanceError):
            result.seed = 2

    def test_pareto_result_front_is_tuple(self):
        # Plan 45.1c picks the "require tuple for immutability" option: the
        # field is annotated ``tuple[ParetoPoint, ...]`` and constructed with
        # tuples.  Python dataclasses do not enforce annotations at runtime,
        # so this test guards the producer side (run_nsga2, load_pareto_front,
        # test fixtures) against accidentally passing a list.
        front = tuple(self._make_point(i) for i in range(3))
        result = self._make_result(front)
        assert isinstance(result.front, tuple)
        # Each member is a ParetoPoint; indexing and iteration both work.
        assert all(isinstance(p, ParetoPoint) for p in result.front)
        assert result.front[0] is front[0]


class TestMakeMultiObjective:
    """Plan 45.1b: vector-valued feasibility-gated objective."""

    def test_shape_matches_field_count_two(self, monkeypatch):
        # Two fields → shape (2,).
        import stencil_gen.pareto as pareto_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            return _empty_report_with(
                layer1={"boundary_gv_err": 0.03},
                layer3={"max_stab_eig": 1e-12},
                layer_bl42={"max_spectral_abscissa": 0.9},
            )

        monkeypatch.setattr(pareto_mod, "brady2d_stability_score", fake_score)
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )
        out = f(np.array([-0.77, 0.16]))
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out.dtype == float
        np.testing.assert_allclose(out, [0.03, 0.9])

    def test_shape_matches_field_count_three(self, monkeypatch):
        # Three fields → shape (3,).
        import stencil_gen.pareto as pareto_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            return _empty_report_with(
                layer1={"boundary_gv_err": 0.04},
                layer3={"max_stab_eig": 2e-12},
                layer_bl42={"max_spectral_abscissa": 0.8},
                layer6={"transient_growth_bound": 3.3},
            )

        monkeypatch.setattr(pareto_mod, "brady2d_stability_score", fake_score)
        f = make_multi_objective(
            "E4",
            "classical",
            [
                "layer1.boundary_gv_err",
                "layer_bl42.max_spectral_abscissa",
                "layer6.transient_growth_bound",
            ],
        )
        out = f(np.array([-0.77, 0.16]))
        assert out.shape == (3,)
        np.testing.assert_allclose(out, [0.04, 0.8, 3.3])

    def test_sentinel_on_gate_trip(self, monkeypatch):
        # Simulate a L1 failure (the gate, since max_layer=3 → gate_layer=2
        # and failed_layer=1 <= 2).  Should return a vector of sentinel
        # values, not +inf, so pymoo's hypervolume indicator stays well-
        # defined.
        import stencil_gen.pareto as pareto_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            r = StabilityReport.empty()
            r.failed_layer = 1
            r.failed_reason = "synthetic L1 failure (alpha=[5,5] non-feasible)"
            return r

        monkeypatch.setattr(pareto_mod, "brady2d_stability_score", fake_score)
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )
        out = f(np.array([5.0, 5.0]))
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out, [_PARETO_SENTINEL, _PARETO_SENTINEL])

    def test_sentinel_on_shape_mismatch(self):
        # E4 classical expects x of length 2 (alpha_0, alpha_1).  A length-3
        # input causes params_from_vector to raise; the closure must swallow
        # the exception and return the sentinel vector.
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )
        out = f(np.array([-0.77, 0.16, 99.0]))
        assert out.shape == (2,)
        np.testing.assert_array_equal(out, [_PARETO_SENTINEL, _PARETO_SENTINEL])

    def test_sentinel_on_score_exception(self, monkeypatch):
        # Any exception from brady2d_stability_score (singular RBF system,
        # numerical blow-up at extreme parameters, …) returns the sentinel
        # vector without propagating.
        import stencil_gen.pareto as pareto_mod

        def raising_score(scheme, kernel, params, *, max_layer, short_circuit):
            raise RuntimeError("simulated numerical blow-up")

        monkeypatch.setattr(pareto_mod, "brady2d_stability_score", raising_score)
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )
        out = f(np.array([-0.77, 0.16]))
        np.testing.assert_array_equal(out, [_PARETO_SENTINEL, _PARETO_SENTINEL])

    def test_finite_on_known_feasible_point(self):
        # BL published optimum for E4 classical.  With both L1 and L3r
        # populated the closure must return an all-finite vector.  This is a
        # real brady2d_stability_score call and the most expensive test in
        # this file; kept because it exercises the full
        # make_multi_objective → short-circuit cascade → extract_field chain
        # end-to-end.
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )
        out = f(np.array([-0.7733323791884821, 0.1623961700641681]))
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))
        # L1 group-velocity error is well below the L1 tolerance for the BL
        # optimum; L3r spectral abscissa passes but may be small-positive or
        # negative depending on integrator cutoff — only sign constraint is
        # "finite, populated".
        assert out[0] >= 0.0

    def test_gate_layer_auto_inferred_from_max_field(self, monkeypatch):
        # Fields span multiple layers (layer1 → 1, layer_bl42 → 3).  The
        # auto-inferred max_layer is 3 and gate_layer is 2: an L2 failure
        # gates; an L3r failure does not (that's the bl42-self-gate trap the
        # scalar analogue in 45.0b unblocked).
        import stencil_gen.pareto as pareto_mod

        captured: dict = {}

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            captured["max_layer"] = max_layer
            r = StabilityReport.empty()
            r.failed_layer = captured["failed_layer_sentinel"]
            r.layer1 = {"boundary_gv_err": 0.04}
            r.layer_bl42 = {"max_spectral_abscissa": 5.0}
            return r

        monkeypatch.setattr(pareto_mod, "brady2d_stability_score", fake_score)
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )

        # L2 failure (below gate_layer=2 → <=2 → gates to sentinel).
        captured["failed_layer_sentinel"] = 2
        out = f(np.array([-0.77, 0.16]))
        assert captured["max_layer"] == 3
        np.testing.assert_array_equal(out, [_PARETO_SENTINEL, _PARETO_SENTINEL])

        # L3 failure (at max_layer, above gate_layer=2 → does not gate; the
        # populated L1 / BL42 payload passes through extract_field).
        captured["failed_layer_sentinel"] = 3
        out = f(np.array([-0.77, 0.16]))
        np.testing.assert_allclose(out, [0.04, 5.0])

    def test_rejects_fewer_than_two_fields(self):
        with pytest.raises(ValueError, match="requires >= 2 report_fields"):
            make_multi_objective("E4", "classical", ["layer1.boundary_gv_err"])

    def test_rejects_unknown_field(self):
        # Mirrors make_objective's failure mode: unrecognised prefix raises
        # before the closure is constructed.
        with pytest.raises(ValueError, match="cannot infer max_layer"):
            make_multi_objective(
                "E4",
                "classical",
                ["layer1.boundary_gv_err", "bogus_field"],
            )
