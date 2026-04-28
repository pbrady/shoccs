"""Unit tests for :mod:`sweeps._bo_io` and :mod:`sweeps.bo`.

Currently covers only the int-key restoration fix from plan item 47.4c.1.
The full :class:`TestBOCLI` (argparse / dispatch) and the rest of
:class:`TestBOIO` (filename, sorting, complex round-trip, ...) are
deliverables for plan item 47.4d, which adds them to this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from stencil_gen.bo import BOEval, BOResult, make_multi_fidelity_objective

from sweeps._bo_io import (
    _INT_KEYED_TOP_LEVEL,
    load_bo_run,
    save_bo_run,
)


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


class TestBOIO:
    """Plan 47.4c / 47.4c.1: per-run JSON persistence + int-key restoration."""

    def test_load_restores_int_keys(self, tmp_path):
        """Plan 47.4c.1: the four whitelisted dict[int,...] fields come back as int."""
        history = (_make_bo_eval(fidelity=1), _make_bo_eval(fidelity=7, value=0.42))
        result = _make_bo_result(history)
        path = save_bo_run(result, directory=tmp_path)
        loaded = load_bo_run(path)
        for name in _INT_KEYED_TOP_LEVEL:
            assert name in loaded, f"missing field {name} in loaded payload"
            for key in loaded[name].keys():
                assert isinstance(key, int), (
                    f"loaded {name} has non-int key {key!r} ({type(key).__name__})"
                )
        # Concrete spot-check against the source values.
        assert loaded["report_fields_by_layer"][7] == "layer7.max_spectral_abscissa"
        assert loaded["cost_model"][1] == pytest.approx(0.076)
        assert loaded["n_evals_per_fidelity"][3] == 3

    def test_roundtrip_preserves_eval_history(self, tmp_path):
        """Save + load: every BOEval field round-trips, int keys preserved."""
        history = (
            _make_bo_eval(fidelity=1, value=0.10),
            _make_bo_eval(fidelity=3, value=0.05),
            _make_bo_eval(fidelity=7, value=0.42),
        )
        result = _make_bo_result(history)
        path = save_bo_run(result, directory=tmp_path)
        loaded = load_bo_run(path)
        assert len(loaded["eval_history"]) == len(history)
        for src, dst in zip(history, loaded["eval_history"]):
            assert dst["fidelity"] == src.fidelity
            assert dst["value"] == pytest.approx(src.value)
            assert dst["wall_time"] == pytest.approx(src.wall_time)
            assert dst["x"] == pytest.approx(src.x.tolist())
            assert dst["params"] == src.params
            assert dst["report"] == src.report
        # Plan 47.4c.1 strengthening: int-keyed top-level fields equal the source dict.
        assert loaded["report_fields_by_layer"] == result.report_fields_by_layer
        assert loaded["cost_model"] == result.cost_model
        assert loaded["n_evals_per_fidelity"] == result.n_evals_per_fidelity
        assert loaded["wall_time_per_fidelity"] == result.wall_time_per_fidelity

    def test_make_objective_accepts_loaded_report_fields(self, tmp_path):
        """Plan 47.4c.1: loaded report_fields_by_layer flows into the factory.

        Without int-key restoration, ``make_multi_fidelity_objective`` raises
        ``TypeError`` at field-vs-layer validation (``int > str``).
        """
        result = _make_bo_result()
        path = save_bo_run(result, directory=tmp_path)
        loaded = load_bo_run(path)
        # Must not raise.
        objective = make_multi_fidelity_objective(
            loaded["scheme"],
            loaded["kernel"],
            loaded["report_fields_by_layer"],
        )
        assert callable(objective)
