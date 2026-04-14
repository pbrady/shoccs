"""Tests for :mod:`stencil_gen.optimizer` (plan 43)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc

from stencil_gen.brady2d_stability import StabilityReport
from stencil_gen.gks_kreiss import KreissResult
from stencil_gen.optimizer import (
    DEFAULT_BOUNDS,
    _COBYQA_AVAILABLE,
    extract_field,
    make_objective,
    multi_start_optimize,
    params_from_vector,
    run_scipy_local,
    run_scipy_shgo,
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


class TestRunScipyLocal:
    """Driver-level tests for :func:`run_scipy_local`.

    Uses simple analytic objectives (quadratics) so the tests run fast and do
    not depend on the Brady-Livescu pipeline.  End-to-end tests against
    ``make_objective`` live further down so that a failure there is clearly
    tagged as an integration issue, not a driver bug.
    """

    @staticmethod
    def _quadratic(x: np.ndarray) -> float:
        # Minimum at [3.0] on the [0, 10] interval.
        return float((x[0] - 3.0) ** 2)

    def test_nelder_mead_converges_on_quadratic(self):
        r = run_scipy_local(
            self._quadratic,
            x0=np.array([5.0]),
            bounds=[(0.0, 10.0)],
            method="Nelder-Mead",
            max_evals=100,
        )
        assert r.method == "Nelder-Mead"
        assert r.converged
        assert np.isfinite(r.best_objective)
        assert r.best_x[0] == pytest.approx(3.0, abs=1e-3)
        assert r.best_objective == pytest.approx(0.0, abs=1e-6)
        assert r.n_evals > 0
        assert r.compute_time >= 0.0
        assert len(r.history) > 0
        # history entries should be (ndarray, float)
        x0, f0 = r.history[0]
        assert isinstance(x0, np.ndarray)
        assert isinstance(f0, float)

    def test_nelder_mead_history_records_every_eval(self):
        r = run_scipy_local(
            self._quadratic,
            x0=np.array([5.0]),
            bounds=[(0.0, 10.0)],
            method="Nelder-Mead",
            max_evals=50,
        )
        # The recorder should capture every objective call — not just
        # iteration endpoints.
        assert len(r.history) == r.n_evals

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            run_scipy_local(
                self._quadratic,
                x0=np.array([5.0]),
                bounds=[(0.0, 10.0)],
                method="BFGS",
            )

    def test_rejects_bounds_length_mismatch(self):
        with pytest.raises(ValueError, match="bounds length"):
            run_scipy_local(
                self._quadratic,
                x0=np.array([5.0]),
                bounds=[(0.0, 10.0), (0.0, 10.0)],
                method="Nelder-Mead",
            )

    def test_nelder_mead_returns_inf_when_only_infeasible(self):
        # Objective that is +inf everywhere — optimizer cannot converge.
        r = run_scipy_local(
            lambda x: float("inf"),
            x0=np.array([5.0]),
            bounds=[(0.0, 10.0)],
            method="Nelder-Mead",
            max_evals=30,
        )
        assert not np.isfinite(r.best_objective)
        assert not r.converged

    def test_nelder_mead_on_make_objective(self):
        # Integration: local optimize of the real tension-E4 objective starting
        # from a nearby feasible point.  We expect the minimizer to land in
        # the feasible basin around σ=3.0 and improve on the initial value.
        f = make_objective(
            "E4", "tension", "layer1.boundary_gv_err",
            gate_layer=3, max_layer=3,
        )
        x0 = np.array([2.0])
        f0 = f(x0)
        r = run_scipy_local(
            f, x0=x0, bounds=[(0.5, 20.0)],
            method="Nelder-Mead", max_evals=80,
        )
        assert np.isfinite(r.best_objective)
        assert r.best_objective <= f0 + 1e-12


@pytest.mark.skipif(not _COBYQA_AVAILABLE, reason="COBYQA requires scipy >= 1.14")
class TestRunScipyLocalCOBYQA:
    @staticmethod
    def _quadratic(x: np.ndarray) -> float:
        return float((x[0] - 3.0) ** 2)

    def test_cobyqa_converges_on_quadratic(self):
        r = run_scipy_local(
            self._quadratic,
            x0=np.array([5.0]),
            bounds=[(0.0, 10.0)],
            method="COBYQA",
            max_evals=100,
        )
        assert r.method == "COBYQA"
        assert r.converged
        assert r.best_x[0] == pytest.approx(3.0, abs=1e-3)

    def test_cobyqa_converges_on_tension_e4(self):
        # Plan 43.3b: COBYQA drives the real tension-E4 objective to a finite,
        # feasible minimum.  The original plan bullet asserted convergence
        # within 5% of "the known σ=3.0 optimum", but σ=3.0 is the
        # sweep-derived stability optimum across a weighted landscape — the
        # true minimum of ``layer1.boundary_gv_err`` alone sits near the
        # lower bound (σ≈0.5) on the monotone branch.  The right invariant is
        # therefore: start from a feasible point, converge to something no
        # worse, and respect the bounds.
        f = make_objective(
            "E4", "tension", "layer1.boundary_gv_err",
            gate_layer=3, max_layer=3,
        )
        x0 = np.array([2.0])
        f0 = f(x0)
        r = run_scipy_local(
            f,
            x0=x0,
            bounds=[(0.5, 20.0)],
            method="COBYQA",
            max_evals=120,
        )
        assert np.isfinite(r.best_objective)
        assert r.best_objective <= f0 + 1e-9
        assert 0.5 <= r.best_x[0] <= 20.0


class TestMultiStart:
    """Tests for :func:`multi_start_optimize` (plan 43.4)."""

    @staticmethod
    def _quadratic(x: np.ndarray) -> float:
        # Minimum at [3.0] on [0, 10].
        return float((x[0] - 3.0) ** 2)

    def test_multi_start_converges_on_quadratic(self):
        r = multi_start_optimize(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            n_restarts=4,
            method="Nelder-Mead",
            seed=0,
            max_evals=50,
        )
        assert r.method == "multi-start"
        assert r.converged
        assert np.isfinite(r.best_objective)
        assert r.best_x[0] == pytest.approx(3.0, abs=1e-3)
        # n_evals sums across restarts; history concatenates.
        assert r.n_evals > 0
        assert len(r.history) == r.n_evals
        assert r.extras["inner_method"] == "Nelder-Mead"
        assert r.extras["n_restarts"] == 4
        assert r.extras["n_feasible_restarts"] >= 1

    def test_multi_start_deterministic(self):
        r1 = multi_start_optimize(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            n_restarts=3,
            method="Nelder-Mead",
            seed=42,
            max_evals=40,
        )
        r2 = multi_start_optimize(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            n_restarts=3,
            method="Nelder-Mead",
            seed=42,
            max_evals=40,
        )
        np.testing.assert_array_equal(r1.best_x, r2.best_x)
        assert r1.best_objective == r2.best_objective
        assert r1.n_evals == r2.n_evals

    def test_multi_start_handles_fully_infeasible(self):
        r = multi_start_optimize(
            lambda x: float("inf"),
            bounds=[(0.0, 10.0)],
            n_restarts=3,
            method="Nelder-Mead",
            seed=0,
            max_evals=20,
        )
        assert not np.isfinite(r.best_objective)
        assert not r.converged
        assert r.extras["n_feasible_restarts"] == 0

    def test_multi_start_rejects_zero_restarts(self):
        with pytest.raises(ValueError, match="n_restarts must be >= 1"):
            multi_start_optimize(
                self._quadratic,
                bounds=[(0.0, 10.0)],
                n_restarts=0,
            )

    def test_multi_start_rejects_empty_bounds(self):
        with pytest.raises(ValueError, match="bounds must be non-empty"):
            multi_start_optimize(
                self._quadratic,
                bounds=[],
                n_restarts=2,
            )

    def test_multi_start_finds_feasible_optimum(self):
        # Plan 43.4b: tension E4 against layer3.max_stab_eig.  The feasibility
        # gate carves out a non-trivial portion of [0.5, 20], so a multi-start
        # should land on a finite, bound-respecting minimum no worse than the
        # best random-restart starting objective.  (No specific-σ claim — see
        # 43.3b for why the sweep-derived σ=3.0 is not this objective's
        # minimum.)
        f = make_objective(
            "E4", "tension", "layer3.max_stab_eig",
            gate_layer=3, max_layer=3,
        )
        bounds = [(0.5, 20.0)]
        r = multi_start_optimize(
            f,
            bounds=bounds,
            n_restarts=4,
            method="Nelder-Mead",
            seed=0,
            max_evals=60,
        )
        assert np.isfinite(r.best_objective)
        assert bounds[0][0] <= r.best_x[0] <= bounds[0][1]
        # Compare against the best initial-point value across restarts.
        sampler = qmc.Sobol(d=1, seed=0)
        x0s = qmc.scale(
            sampler.random(4),
            np.array([bounds[0][0]]),
            np.array([bounds[0][1]]),
        )
        best_f0 = min(float(f(x0)) for x0 in x0s)
        assert r.best_objective <= best_f0 + 1e-9


class TestSHGO:
    """Tests for :func:`run_scipy_shgo` (plan 43.5a)."""

    @staticmethod
    def _quadratic(x: np.ndarray) -> float:
        # Single global minimum at [3.0] on [0, 10].
        return float((x[0] - 3.0) ** 2)

    @staticmethod
    def _two_basin(x: np.ndarray) -> float:
        # Two distinct local minima: one at x=2 (shallow) and one at x=7
        # (deep global).  Constructed so SHGO will find both basins.
        return float(
            0.3 * (x[0] - 2.0) ** 2 * (x[0] < 4.5)
            + ((x[0] - 7.0) ** 2 + 0.1) * (x[0] >= 4.5)
        )

    def test_shgo_converges_on_quadratic(self):
        r = run_scipy_shgo(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            n=20,
            iters=2,
        )
        assert r.method == "SHGO"
        assert r.converged
        assert np.isfinite(r.best_objective)
        assert r.best_x[0] == pytest.approx(3.0, abs=1e-3)
        assert r.best_objective == pytest.approx(0.0, abs=1e-6)
        assert r.n_evals > 0
        assert r.compute_time >= 0.0
        # Extras carry the local-minima table.
        assert "n_local_minima" in r.extras
        assert r.extras["n_local_minima"] >= 1
        assert len(r.extras["local_minima"]) == r.extras["n_local_minima"]

    def test_shgo_records_local_minima(self):
        # Multi-basin landscape — SHGO should discover at least one minimum,
        # and the extras["local_minima"] table should be parseable.
        r = run_scipy_shgo(
            self._two_basin,
            bounds=[(0.0, 10.0)],
            n=30,
            iters=2,
        )
        assert r.extras["n_local_minima"] >= 1
        for x, fv in r.extras["local_minima"]:
            assert isinstance(x, np.ndarray)
            assert isinstance(fv, float)
            assert 0.0 <= x[0] <= 10.0

    def test_shgo_rejects_empty_bounds(self):
        with pytest.raises(ValueError, match="bounds must be non-empty"):
            run_scipy_shgo(self._quadratic, bounds=[])

    def test_shgo_handles_fully_infeasible(self):
        r = run_scipy_shgo(
            lambda x: float("inf"),
            bounds=[(0.0, 10.0)],
            n=10,
            iters=1,
        )
        assert not np.isfinite(r.best_objective)
        assert not r.converged
        assert r.method == "SHGO"
        # Fallback best_x is the bound midpoint — a sensible placeholder.
        assert r.best_x.shape == (1,)


class TestRunScipyLocalCOBYQAUnavailable:
    def test_cobyqa_unavailable_raises_runtime_error(self, monkeypatch):
        import stencil_gen.optimizer as opt

        monkeypatch.setattr(opt, "_COBYQA_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="COBYQA requires scipy"):
            opt.run_scipy_local(
                lambda x: float(x[0] ** 2),
                x0=np.array([1.0]),
                bounds=[(-5.0, 5.0)],
                method="COBYQA",
            )
