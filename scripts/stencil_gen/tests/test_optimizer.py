"""Tests for :mod:`stencil_gen.optimizer` (plan 43)."""

from __future__ import annotations

import numpy as np
import pytest

from stencil_gen.brady2d_stability import StabilityReport
from stencil_gen.gks_kreiss import KreissResult
from stencil_gen.optimizer import (
    DEFAULT_BOUNDS,
    extract_field,
    make_objective,
    params_from_vector,
    vector_from_params,
)


class TestParamsVector:
    @pytest.mark.parametrize(
        ("kernel", "x", "expected"),
        [
            ("tension", [3.0], {"sigma": 3.0}),
            ("gaussian", [1.5], {"epsilon": 1.5}),
            ("multiquadric", [0.7], {"epsilon": 0.7}),
            ("classical", [-0.5, 197.0 / 288.0], {"alpha": [-0.5, 197.0 / 288.0]}),
        ],
    )
    def test_params_from_vector(self, kernel, x, expected):
        assert params_from_vector(kernel, np.asarray(x)) == expected

    @pytest.mark.parametrize(
        ("kernel", "params", "expected_x"),
        [
            ("tension", {"sigma": 3.0}, [3.0]),
            ("gaussian", {"epsilon": 1.5}, [1.5]),
            ("multiquadric", {"epsilon": 0.7}, [0.7]),
            ("classical", {"alpha": [-0.5, 1.0]}, [-0.5, 1.0]),
        ],
    )
    def test_vector_from_params(self, kernel, params, expected_x):
        v = vector_from_params(kernel, params)
        assert v.dtype == np.float64
        np.testing.assert_array_equal(v, np.asarray(expected_x, dtype=float))

    @pytest.mark.parametrize(
        ("kernel", "params"),
        [
            ("tension", {"sigma": 3.0}),
            ("gaussian", {"epsilon": 1.5}),
            ("multiquadric", {"epsilon": 0.7}),
            ("classical", {"alpha": [-0.5, 1.0]}),
        ],
    )
    def test_roundtrip_params_vector_params(self, kernel, params):
        v = vector_from_params(kernel, params)
        assert params_from_vector(kernel, v) == params

    @pytest.mark.parametrize(
        ("kernel", "x"),
        [
            ("tension", [1.0, 2.0]),
            ("gaussian", [1.0, 2.0]),
            ("multiquadric", []),
            ("classical", [1.0]),
        ],
    )
    def test_params_from_vector_wrong_dim(self, kernel, x):
        with pytest.raises(ValueError):
            params_from_vector(kernel, np.asarray(x, dtype=float))

    @pytest.mark.parametrize("kernel", ["tension-penalty", "mixed-epsilon"])
    def test_pruned_kernels_rejected(self, kernel):
        # Plan 43.1d (option b): these families are out of scope for the
        # layered optimizer; brady2d_stability_score does not route them.
        with pytest.raises(ValueError, match="unknown kernel"):
            params_from_vector(kernel, np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="unknown kernel"):
            vector_from_params(kernel, {"sigma": 1.0, "gamma": 2.0})

    def test_params_from_vector_unknown_kernel(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            params_from_vector("nope", np.array([1.0]))

    def test_vector_from_params_unknown_kernel(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            vector_from_params("nope", {})

    def test_classical_requires_length_two_alpha(self):
        with pytest.raises(ValueError):
            vector_from_params("classical", {"alpha": [0.1]})

    def test_default_bounds_shapes_match_roundtrip(self):
        # For each entry in DEFAULT_BOUNDS, build a midpoint vector and round-trip
        # through params_from_vector / vector_from_params.
        for (scheme, kernel), bounds in DEFAULT_BOUNDS.items():
            mid = np.array([0.5 * (lo + hi) for (lo, hi) in bounds], dtype=float)
            params = params_from_vector(kernel, mid)
            v = vector_from_params(kernel, params)
            assert v.shape == mid.shape, f"{scheme}/{kernel}: shape mismatch"
            np.testing.assert_array_equal(v, mid)


class TestExtractField:
    @staticmethod
    def _populated_report() -> StabilityReport:
        return StabilityReport(
            layer1={"boundary_gv_err": 1.25e-3},
            layer2=KreissResult(is_stable=True, witness_sigma_min=0.42),
            layer3={"max_stab_eig": -1.5e-4},
            layer6={
                "spectral_abscissa": -2.0e-3,
                "kreiss_constant": 3.7,
                "transient_growth_bound": 12.5,
                "henrici_departure": 0.01,
            },
            layer7={"max_spectral_abscissa": 5.0e-4},
            kreiss=KreissResult(is_stable=True, witness_sigma_min=0.77),
            overall_verdict="pass",
        )

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("layer1.boundary_gv_err", 1.25e-3),
            ("layer3.max_stab_eig", -1.5e-4),
            ("layer6.spectral_abscissa", -2.0e-3),
            ("layer6.kreiss_constant", 3.7),
            ("layer6.transient_growth_bound", 12.5),
            ("layer7.max_spectral_abscissa", 5.0e-4),
            ("kreiss.witness_sigma_min", 0.77),
        ],
    )
    def test_extract_populated_fields(self, path, expected):
        r = self._populated_report()
        assert extract_field(r, path) == pytest.approx(expected)

    def test_extract_unknown_first_segment_returns_inf(self):
        r = self._populated_report()
        assert extract_field(r, "layer99.foo") == float("inf")

    def test_extract_missing_key_returns_inf(self):
        r = self._populated_report()
        assert extract_field(r, "layer1.not_a_metric") == float("inf")

    def test_extract_layer_not_run_returns_inf(self):
        # layer4/layer5/layer8 were not populated above
        r = self._populated_report()
        assert extract_field(r, "layer4.max_local_gv_error") == float("inf")
        assert extract_field(r, "layer8.final_linf") == float("inf")

    def test_extract_empty_report_returns_inf(self):
        r = StabilityReport.empty()
        assert extract_field(r, "layer1.boundary_gv_err") == float("inf")
        assert extract_field(r, "kreiss.witness_sigma_min") == float("inf")

    def test_extract_empty_path_returns_inf(self):
        r = self._populated_report()
        assert extract_field(r, "") == float("inf")

    def test_extract_missing_dataclass_attr_returns_inf(self):
        r = self._populated_report()
        assert extract_field(r, "kreiss.no_such_attr") == float("inf")


class TestMakeObjective:
    def test_objective_returns_finite_on_feasible(self):
        # E4 tension σ=3.0 is the known-good sweep-derived optimum and
        # comfortably passes L1-L3.
        f = make_objective(
            "E4", "tension", "layer1.boundary_gv_err",
            gate_layer=3, max_layer=3,
        )
        val = f(np.array([3.0]))
        assert np.isfinite(val)
        assert val >= 0.0

    def test_objective_returns_inf_on_gate_failure(self):
        # A tiny gaussian ε makes the RBF matrix nearly singular: L1 or L3
        # fails and the feasibility gate forces +inf.
        f = make_objective(
            "E4", "gaussian", "layer1.boundary_gv_err",
            gate_layer=3, max_layer=3,
        )
        assert f(np.array([0.01])) == float("inf")

    def test_objective_catches_exception(self, monkeypatch):
        import stencil_gen.optimizer as opt

        def _boom(*_args, **_kwargs):
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(opt, "brady2d_stability_score", _boom)
        f = opt.make_objective(
            "E4", "tension", "layer1.boundary_gv_err",
            gate_layer=3, max_layer=3,
        )
        assert f(np.array([3.0])) == float("inf")

    def test_objective_raises_on_bad_field(self):
        # A nonsense dotted path at a feasible point returns +inf (extract_field
        # treats missing segments as inf) rather than raising.
        f = make_objective(
            "E4", "tension", "layer99.foo",
            gate_layer=3, max_layer=3,
        )
        assert f(np.array([3.0])) == float("inf")

    def test_objective_infers_max_layer_from_field(self):
        # layer6.* implies max_layer=6; gate_layer=3 (default) is < 6 so no
        # error.
        f = make_objective("E4", "tension", "layer6.spectral_abscissa")
        val = f(np.array([3.0]))
        assert np.isfinite(val)

    def test_objective_rejects_max_layer_below_gate(self):
        with pytest.raises(ValueError, match="less than gate_layer"):
            make_objective(
                "E4", "tension", "layer1.boundary_gv_err",
                gate_layer=3, max_layer=1,
            )

    def test_objective_rejects_uninferable_field_without_max_layer(self):
        with pytest.raises(ValueError, match="cannot infer max_layer"):
            make_objective("E4", "tension", "no_prefix_here")
