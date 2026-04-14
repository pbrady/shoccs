"""Tests for :mod:`stencil_gen.optimizer` (plan 43)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import qmc

from stencil_gen.brady2d_stability import StabilityReport
from stencil_gen.gks_kreiss import KreissResult
from stencil_gen.optimizer import (
    DEFAULT_BOUNDS,
    _COBYQA_AVAILABLE,
    OptimizeResult,
    extract_field,
    make_objective,
    multi_start_optimize,
    params_from_vector,
    run_scipy_de,
    run_scipy_local,
    run_scipy_shgo,
    run_staged_optimize,
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


class TestDE:
    """Tests for :func:`run_scipy_de` (plan 43.5b)."""

    @staticmethod
    def _quadratic(x: np.ndarray) -> float:
        return float((x[0] - 3.0) ** 2)

    @staticmethod
    def _rosenbrock(x: np.ndarray) -> float:
        # Classic 2D Rosenbrock, minimum at (1, 1).
        return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)

    def test_de_converges_on_quadratic(self):
        r = run_scipy_de(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            popsize=8,
            maxiter=100,
            seed=0,
        )
        assert r.method == "DE"
        assert np.isfinite(r.best_objective)
        # Population-convergence tolerance means DE may stop with success=False
        # if maxiter runs out before the population collapses, even when the
        # polish pass has already pinned the minimum — so require finite
        # convergence-to-a-known-optimum rather than ``result.success``.
        assert r.best_x[0] == pytest.approx(3.0, abs=1e-3)
        assert r.best_objective == pytest.approx(0.0, abs=1e-6)
        assert r.n_evals > 0
        assert r.compute_time >= 0.0
        assert r.extras["popsize"] == 8
        assert r.extras["maxiter"] == 100
        assert r.extras["seed"] == 0
        assert r.extras["strategy"] == "best1bin"

    def test_de_deterministic(self):
        kwargs = dict(bounds=[(-5.0, 5.0), (-5.0, 5.0)], popsize=8, maxiter=10, seed=42)
        r1 = run_scipy_de(self._rosenbrock, **kwargs)
        r2 = run_scipy_de(self._rosenbrock, **kwargs)
        assert r1.best_objective == pytest.approx(r2.best_objective, rel=0, abs=1e-12)
        assert np.allclose(r1.best_x, r2.best_x)
        assert r1.n_evals == r2.n_evals

    def test_de_records_history(self):
        r = run_scipy_de(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            popsize=5,
            maxiter=5,
            seed=0,
        )
        assert len(r.history) > 0
        for x, fv in r.history:
            assert isinstance(x, np.ndarray)
            assert isinstance(fv, float)
            assert 0.0 <= x[0] <= 10.0

    def test_de_rejects_empty_bounds(self):
        with pytest.raises(ValueError, match="bounds must be non-empty"):
            run_scipy_de(self._quadratic, bounds=[])

    def test_de_rejects_bad_popsize(self):
        with pytest.raises(ValueError, match="popsize must be >= 1"):
            run_scipy_de(self._quadratic, bounds=[(0.0, 1.0)], popsize=0)

    def test_de_rejects_bad_maxiter(self):
        with pytest.raises(ValueError, match="maxiter must be >= 1"):
            run_scipy_de(self._quadratic, bounds=[(0.0, 1.0)], maxiter=0)

    def test_de_handles_fully_infeasible(self):
        r = run_scipy_de(
            lambda x: float("inf"),
            bounds=[(0.0, 1.0)],
            popsize=4,
            maxiter=3,
            seed=0,
        )
        assert not np.isfinite(r.best_objective)
        assert r.converged is False
        assert r.best_x.shape == (1,)
        assert len(r.history) > 0


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


class TestGlobalOptimizers:
    """Integration tests for :func:`run_scipy_shgo` / :func:`run_scipy_de`
    against real :mod:`brady2d_stability` objectives (plan 43.5c)."""

    def test_shgo_finds_tension_optimum(self):
        # 1D tension E4 against layer3.max_stab_eig.  The earlier plan text
        # asserted convergence to σ=3.0; 43.3b refuted that invariant (see
        # 43.3c) — the new acceptance is only "finite feasible global minimum,
        # bound-respecting, at least one discovered basin."
        f = make_objective(
            "E4", "tension", "layer3.max_stab_eig",
            gate_layer=3, max_layer=3,
        )
        bounds = [(0.5, 20.0)]
        r = run_scipy_shgo(f, bounds=bounds, n=8, iters=1)
        assert np.isfinite(r.best_objective)
        assert bounds[0][0] <= r.best_x[0] <= bounds[0][1]
        assert r.extras["n_local_minima"] >= 1

    def test_de_finds_tension_optimum(self):
        # Same objective via differential_evolution.  Kept to a tight budget
        # (popsize=6, maxiter=8) to keep the test under ~30s; the objective's
        # feasibility cliff is narrow enough that this still reaches a finite
        # feasible minimum.
        f = make_objective(
            "E4", "tension", "layer3.max_stab_eig",
            gate_layer=3, max_layer=3,
        )
        bounds = [(0.5, 20.0)]
        r = run_scipy_de(f, bounds=bounds, popsize=6, maxiter=8, seed=0)
        assert np.isfinite(r.best_objective)
        assert bounds[0][0] <= r.best_x[0] <= bounds[0][1]

    @pytest.mark.slow
    def test_shgo_2d_classical_alpha(self):
        # 2D E4 classical-α.  NB: the plan's ``DEFAULT_BOUNDS[("E4",
        # "classical")] = [(-2, 2), (197/288, 2)]`` encodes the C++ hard
        # constraint α₁ ≥ 197/288 but the Brady-Livescu published feasible
        # point (α ≈ [-0.77, 0.16]) sits *below* that lower α₁ bound, and the
        # intersection of DEFAULT_BOUNDS with the L3-feasible region appears
        # empty in the Python pipeline (probed on an 11×8 grid).  We therefore
        # use relaxed bounds that admit the known feasible region so the
        # "SHGO finds at least one feasible local min" invariant holds.
        # Resolving the DEFAULT_BOUNDS mismatch itself is deferred to 43.9a.
        f = make_objective(
            "E4", "classical", "layer3.max_stab_eig",
            gate_layer=3, max_layer=3,
        )
        bounds = [(-1.2, -0.3), (0.05, 0.4)]
        r = run_scipy_shgo(f, bounds=bounds, n=6, iters=1)
        assert np.isfinite(r.best_objective), (
            "SHGO should land on at least one feasible minimum in the "
            "Brady-Livescu-adjacent region"
        )
        # Bound-respecting.
        assert bounds[0][0] <= r.best_x[0] <= bounds[0][1]
        assert bounds[1][0] <= r.best_x[1] <= bounds[1][1]
        assert r.extras["n_local_minima"] >= 1
        # Loose (within-basin) comparison against the Brady-Livescu stored
        # α ≈ [-0.7733, 0.1624].  A 0.5 L∞ tolerance keeps this a
        # containment check, not an identity check (matching the 43.9d
        # convention).
        published = np.array([-0.7733323791884821, 0.1623961700641681])
        assert np.max(np.abs(r.best_x - published)) < 0.5


class TestStaged:
    """Tests for :func:`run_staged_optimize` (plan 43.6)."""

    def test_staged_rejects_shallow_validator(self):
        with pytest.raises(ValueError, match="validator_max_layer"):
            run_staged_optimize(
                scheme="E4",
                kernel="tension",
                report_field="layer3.max_stab_eig",
                bounds=[(0.5, 20.0)],
                inner_gate=3,
                inner_max_layer=3,
                validator_max_layer=2,
            )

    def test_staged_rejects_inner_shallower_than_gate(self):
        with pytest.raises(ValueError, match="inner_max_layer"):
            run_staged_optimize(
                scheme="E4",
                kernel="tension",
                report_field="layer3.max_stab_eig",
                bounds=[(0.5, 20.0)],
                inner_gate=3,
                inner_max_layer=2,
                validator_max_layer=3,
            )

    def test_staged_rejects_zero_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            run_staged_optimize(
                scheme="E4",
                kernel="tension",
                report_field="layer3.max_stab_eig",
                bounds=[(0.5, 20.0)],
                top_k=0,
            )

    @pytest.mark.slow
    def test_staged_tension_e4_convergence(self):
        # Inner/validator at the same layer so the "improves on or ties"
        # invariant is a pure re-ranking check: validator best <= inner best
        # at the same x is tautological here, but the staged pipeline must
        # still deliver a finite feasible winner, bound-respecting, and
        # populate both best_params and best_report.  (The earlier
        # specific-σ acceptance was dropped per 43.3c.)
        bounds = [(0.5, 20.0)]
        r = run_staged_optimize(
            scheme="E4",
            kernel="tension",
            report_field="layer3.max_stab_eig",
            bounds=bounds,
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=3,
            top_k=3,
            method="Nelder-Mead",
            n_restarts=3,
            seed=0,
            max_evals=40,
        )
        assert r.method == "staged"
        assert r.converged
        assert np.isfinite(r.best_objective)
        assert bounds[0][0] <= r.best_x[0] <= bounds[0][1]
        assert r.best_params == {"sigma": float(r.best_x[0])}
        # Validator-picked winner is at least as good as the inner-stage
        # best (both measured at the same L3 field, since they share the
        # same max_layer here).
        assert r.best_objective <= r.extras["inner_best_objective"] + 1e-9
        # Validator ranking is populated and sorted ascending.
        ranking = r.extras["validator_ranking"]
        assert len(ranking) >= 1
        ranking_f = [fv for (_x, fv) in ranking]
        assert ranking_f == sorted(ranking_f)
        # Serialized report has at least the gate layers.
        assert "layer3" in r.best_report

    def test_staged_validator_reorders(self, monkeypatch):
        # Deterministic synthetic re-order test (plan 43.6d): stub both
        # ``multi_start_optimize`` (to return a canned inner history) and
        # ``brady2d_stability_score`` (to give validator-depth rankings that
        # disagree with the inner ranking).  This replaces the previous
        # tension-E4 L6 integration test whose outcome was data-dependent and
        # whose assertion only checked that ``stage`` was populated with one
        # of two possible values — a regression that silently made the
        # validator mirror the inner winner would have passed.
        #
        # Design:
        #   - inner ranks A=2.0 best on ``layer3.max_stab_eig`` (lowest value),
        #     then B=8.0, then C=5.0.
        #   - validator ranks B=8.0 best on ``layer6.transient_growth_bound``
        #     (quadratic well centered at 8.0).
        #   - With ``top_k=3`` the validator sees all three; its winner is B,
        #     distinct from the inner's winner A, so ``stage`` must be
        #     ``"validated"`` and ``best_x`` must be B.
        import stencil_gen.optimizer as opt_mod

        A = np.array([2.0])
        B = np.array([8.0])
        C = np.array([5.0])

        canned_history = [
            (A.copy(), -0.5),
            (B.copy(), -0.3),
            (C.copy(), -0.1),
            (np.array([9.0]), 0.2),
        ]
        canned_inner = OptimizeResult(
            best_params={"sigma": 2.0},
            best_x=A.copy(),
            best_objective=-0.5,
            best_report={},
            method="Nelder-Mead",
            converged=True,
            n_evals=4,
            compute_time=0.0,
            history=canned_history,
            extras={
                "inner_method": "Nelder-Mead",
                "n_restarts": 4,
                "seed": 0,
                "n_feasible_restarts": 4,
            },
        )

        def fake_multi_start(f, bounds, **kwargs):
            return canned_inner

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit=True):
            # Inner is bypassed via fake_multi_start; the validator stage is
            # the only caller that reaches here in this test.
            sigma = float(params["sigma"])
            tgb = (sigma - 8.0) ** 2
            return StabilityReport(
                layer1={"boundary_gv_err": 1e-4},
                layer3={"max_stab_eig": -0.3},
                layer6={
                    "transient_growth_bound": tgb,
                    "spectral_abscissa": -0.1,
                    "kreiss_constant": 1.0,
                },
                failed_layer=None,
                overall_verdict="pass",
            )

        monkeypatch.setattr(opt_mod, "multi_start_optimize", fake_multi_start)
        monkeypatch.setattr(opt_mod, "brady2d_stability_score", fake_score)

        r = run_staged_optimize(
            scheme="E4",
            kernel="tension",
            report_field="layer6.transient_growth_bound",
            bounds=[(0.5, 20.0)],
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=6,
            top_k=3,
        )

        assert r.method == "staged"
        # Validator reordered: winner is B, not the inner's A.
        assert r.extras["stage"] == "validated"
        np.testing.assert_allclose(r.best_x, B)
        assert not np.allclose(r.best_x, canned_inner.best_x)
        # Validator's transient_growth_bound at σ=8 is exactly 0.
        assert r.best_objective == pytest.approx(0.0)
        # Inner fallback field was used (report_field is L6-only).
        assert r.extras["inner_field"] == "layer3.max_stab_eig"
        assert r.extras["validator_max_layer"] == 6
        # Inner diagnostics preserved in extras.
        np.testing.assert_allclose(r.extras["inner_best_x"], A)
        assert r.extras["inner_best_objective"] == pytest.approx(-0.5)
        # Validator ranking sorted ascending; B is first, C second, A last.
        ranking = r.extras["validator_ranking"]
        assert len(ranking) == 3
        ranking_f = [fv for (_x, fv) in ranking]
        assert ranking_f == sorted(ranking_f)
        np.testing.assert_allclose(ranking[0][0], B)
        # best_report carries the L6 payload from the validator run.
        assert "layer6" in r.best_report

    def test_staged_validator_all_blowups(self, monkeypatch):
        # Every validator re-run raises: the staged pipeline must return the
        # inner-stage result wrapped with ``method="staged"``,
        # ``stage="inner"``, ``converged=False`` (the validator did not
        # confirm), and the fallback ``extras`` must carry the
        # ``inner_best_objective`` / ``inner_best_x`` keys that the
        # success-path extras populates — otherwise downstream callers and
        # tests that read those keys ``KeyError`` on the fallback branch.
        import stencil_gen.optimizer as opt_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit=True):
            if max_layer <= 3:
                # Inner stage: feasible, L3 max_stab_eig varies with sigma so
                # multi_start_optimize has a real objective to descend on.
                sigma = float(params["sigma"])
                return StabilityReport(
                    layer1={"boundary_gv_err": 1e-4},
                    layer3={"max_stab_eig": -0.1 + 1e-3 * (sigma - 3.0) ** 2},
                    failed_layer=None,
                    overall_verdict="pass",
                )
            # Validator stage (max_layer=6): blow up every time.
            raise RuntimeError("validator blew up")

        monkeypatch.setattr(opt_mod, "brady2d_stability_score", fake_score)

        r = run_staged_optimize(
            scheme="E4",
            kernel="tension",
            report_field="layer6.transient_growth_bound",
            bounds=[(0.5, 20.0)],
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=6,
            top_k=3,
            method="Nelder-Mead",
            n_restarts=2,
            seed=0,
            max_evals=20,
        )

        assert r.method == "staged"
        assert r.extras["stage"] == "inner"
        assert r.converged is False
        # Fallback extras parity with the success path: both keys must be
        # present and mirror the inner result.
        assert "inner_best_objective" in r.extras
        assert "inner_best_x" in r.extras
        # Fallback builds the result via ``replace(inner_result, method="staged",
        # converged=False, ...)`` without touching ``best_objective`` / ``best_x``,
        # so the exposed fields must match the ``inner_*`` extras.
        assert r.extras["inner_best_objective"] == pytest.approx(r.best_objective)
        assert np.allclose(r.extras["inner_best_x"], r.best_x)
        assert r.extras["inner_best_x"].shape == r.best_x.shape
        assert np.isfinite(r.extras["inner_best_objective"])
        assert r.extras["inner_best_x"].shape == (1,)
        # Validator ranking entries are all infeasible.
        ranking = r.extras["validator_ranking"]
        assert len(ranking) >= 1
        assert all(not np.isfinite(fv) for (_x, fv) in ranking)


class TestOptimizeCLI:
    """Smoke tests for ``sweeps.optimize`` (plan 43.7c).

    One subprocess test verifies the real ``python -m sweeps.optimize`` entry
    point end-to-end; the error-path tests call ``main`` in-process to keep
    the suite fast (parser.error raises SystemExit, which is what "exits
    non-zero" means in a subprocess).
    """

    @pytest.mark.slow
    def test_cli_tension_nelder_mead(self):
        """A tiny tension-E4 Nelder-Mead run completes and prints a summary.

        Marked slow: subprocess pays the SymPy cold-start tax (~5-7 min on
        first invocation in a fresh environment).
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        stencil_gen_dir = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["SYMPY_CACHE_SIZE"] = env.get("SYMPY_CACHE_SIZE", "50000")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "sweeps.optimize",
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--gate-layer", "3",
                "--max-layer", "3",
                "--bounds", "0.5", "20",
                "--method", "Nelder-Mead",
                "--n-restarts", "1",
                "--max-evals", "10",
                "--seed", "0",
            ],
            cwd=str(stencil_gen_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, (
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        assert "best_objective" in proc.stdout
        assert "best_params" in proc.stdout

    def test_cli_rejects_bad_objective(self):
        """Unknown objective prefix cannot infer max_layer → SystemExit.

        ``bogus.field`` has neither a ``layerN.`` prefix nor an entry in
        ``_FIELD_LAYER_ALIAS``, so ``make_objective`` raises ``ValueError`` at
        construction (before any evaluation), which the CLI surfaces as
        ``parser.error`` → ``SystemExit(2)``.
        """
        from sweeps.optimize import main

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--scheme", "E4",
                    "--kernel", "tension",
                    "--objective", "bogus.field",
                    "--bounds", "0.5", "20",
                    "--method", "Nelder-Mead",
                    "--n-restarts", "1",
                    "--max-evals", "4",
                ]
            )
        # argparse.error exits with code 2.
        assert exc_info.value.code != 0

    def test_cli_rejects_kernel_bounds_dim_mismatch(self):
        """``--kernel classical --bounds 0.5 20`` (1D bounds for 2D kernel) errors."""
        from sweeps.optimize import main

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--scheme", "E4",
                    "--kernel", "classical",
                    "--objective", "layer3.max_stab_eig",
                    "--bounds", "0.5", "20",
                    "--method", "Nelder-Mead",
                    "--n-restarts", "1",
                    "--max-evals", "4",
                ]
            )
        assert exc_info.value.code != 0

    def test_cli_update_known_values_additive_and_drops_history(
        self, monkeypatch
    ):
        """``--update-known-values`` writes under ``brady2d_optima`` without
        touching unrelated keys, and omits ``history`` from the persisted form.

        Uses a monkey-patched ``_run_method`` so the test never enters the real
        SymPy pipeline — the goal is to pin the persistence contract in
        ``sweeps/optimize.py`` (plan item 43.8a), not to re-exercise the
        optimizers.
        """
        from sweeps import optimize as optimize_mod

        store: dict = {
            "brady2d_calibration": {"E4": {"tension": [1.0, 2.0]}},
            "brady2d_sweep": {"E4": {"tension": {"sigma": 3.0}}},
        }

        def fake_load() -> dict:
            # Shallow copy so the CLI's setdefault mutations go through
            # ``fake_save`` rather than silently aliasing ``store``.
            import copy
            return copy.deepcopy(store)

        def fake_save(data: dict) -> None:
            store.clear()
            store.update(data)

        monkeypatch.setattr(optimize_mod, "load_known_values", fake_load)
        monkeypatch.setattr(optimize_mod, "save_known_values", fake_save)

        canned = OptimizeResult(
            best_params={"sigma": 3.1},
            best_x=np.array([3.1]),
            best_objective=-1.5,
            best_report={"layer3": {"max_stab_eig": -1.5}},
            method="Nelder-Mead",
            converged=True,
            n_evals=17,
            compute_time=0.1,
            history=[
                (np.array([3.0]), -1.4),
                (np.array([3.1]), -1.5),
            ],
            extras={"n_restarts": 1},
        )
        monkeypatch.setattr(
            optimize_mod, "_run_method", lambda args, bounds: canned
        )

        rc = optimize_mod.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--bounds", "0.5", "20",
                "--method", "Nelder-Mead",
                "--n-restarts", "1",
                "--max-evals", "4",
                "--update-known-values",
            ]
        )
        assert rc == 0

        # Persisted under brady2d_optima[scheme][kernel][objective].
        opt = store["brady2d_optima"]["E4"]["tension"]["layer3.max_stab_eig"]
        assert opt["best_objective"] == pytest.approx(-1.5)
        assert opt["best_params"] == {"sigma": 3.1}
        assert opt["converged"] is True
        assert opt["n_evals"] == 17
        assert opt["method"] == "Nelder-Mead"
        assert opt["bounds"] == [[0.5, 20.0]]
        assert opt["best_x"] == [pytest.approx(3.1)]
        # history is intentionally omitted from the persisted form.
        assert "history" not in opt
        # Plan 43.8c: gate_layer and inferred max_layer round-trip (no
        # --max-layer was passed for this call, so max_layer is the inferred
        # layer3 from the "layer3.max_stab_eig" prefix).  Non-staged method,
        # so validator_max_layer must be absent.
        assert opt["gate_layer"] == 3
        assert opt["max_layer"] == 3
        assert "validator_max_layer" not in opt
        # Existing top-level keys are untouched.
        assert store["brady2d_calibration"] == {"E4": {"tension": [1.0, 2.0]}}
        assert store["brady2d_sweep"] == {"E4": {"tension": {"sigma": 3.0}}}

        # A second CLI call at a different objective must coexist with the
        # first under the same scheme/kernel bucket (additive behaviour).
        second = replace(canned, best_objective=-0.75, best_x=np.array([4.0]),
                         best_params={"sigma": 4.0})
        monkeypatch.setattr(
            optimize_mod, "_run_method", lambda args, bounds: second
        )
        rc2 = optimize_mod.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer6.transient_growth_bound",
                "--max-layer", "6",
                "--bounds", "0.5", "20",
                "--method", "Nelder-Mead",
                "--n-restarts", "1",
                "--max-evals", "4",
                "--update-known-values",
            ]
        )
        assert rc2 == 0
        kernel_bucket = store["brady2d_optima"]["E4"]["tension"]
        assert set(kernel_bucket.keys()) == {
            "layer3.max_stab_eig",
            "layer6.transient_growth_bound",
        }
        assert kernel_bucket["layer3.max_stab_eig"]["best_objective"] == \
            pytest.approx(-1.5)
        assert kernel_bucket["layer6.transient_growth_bound"][
            "best_objective"
        ] == pytest.approx(-0.75)
        # Plan 43.8c: explicit --max-layer 6 on the second call must round-trip
        # (no inference fallback).  Still non-staged, so no validator field.
        second_opt = kernel_bucket["layer6.transient_growth_bound"]
        assert second_opt["gate_layer"] == 3
        assert second_opt["max_layer"] == 6
        assert "validator_max_layer" not in second_opt

        # Plan 43.8c: a staged call must round-trip gate_layer + max_layer
        # (inner depth) + validator_max_layer.  Use a fresh objective bucket so
        # this case is distinct from the two above.
        staged_canned = replace(
            canned,
            method="staged",
            best_objective=-2.25,
            best_x=np.array([5.0]),
            best_params={"sigma": 5.0},
        )
        monkeypatch.setattr(
            optimize_mod, "_run_method", lambda args, bounds: staged_canned
        )
        rc3 = optimize_mod.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer6.kreiss_constant",
                "--bounds", "0.5", "20",
                "--method", "staged",
                "--n-restarts", "1",
                "--max-evals", "4",
                "--validator-max-layer", "7",
                "--update-known-values",
            ]
        )
        assert rc3 == 0
        staged_opt = store["brady2d_optima"]["E4"]["tension"][
            "layer6.kreiss_constant"
        ]
        # No --max-layer passed and method is staged, so the inner-depth
        # default of 3 is persisted; validator_max_layer is the explicit 7.
        assert staged_opt["method"] == "staged"
        assert staged_opt["gate_layer"] == 3
        assert staged_opt["max_layer"] == 3
        assert staged_opt["validator_max_layer"] == 7
