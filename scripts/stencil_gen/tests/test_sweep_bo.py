"""Unit tests for :mod:`sweeps._bo_io` and :mod:`sweeps.bo`.

Covers:

- :class:`TestBOIO` — int-key restoration (plan 47.4c.1), file creation,
  filename-includes-seed, sorted iteration, and complex round-tripping
  through ``extras`` (plan 47.4d).
- :class:`TestBOCLI` — argparse surface (minimal accepting invocation,
  budget-mutex enforcement, cheap > HF rejection), dispatch through
  ``python -m sweeps bo --help``, and the ``--baseline staged`` stub
  acceptance (plan 47.4d).

The :class:`TestBOCLI` tests stub :func:`run_mfbo` via ``monkeypatch`` so
no botorch / brady2d pipeline is entered — keeps the tests in the fast
suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stencil_gen.bo import BOEval, BOResult, make_multi_fidelity_objective

from sweeps import bo as bo_cli
from sweeps._bo_io import (
    _INT_KEYED_TOP_LEVEL,
    iter_bo_runs,
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

    def test_save_bo_run_creates_file(self, tmp_path):
        """``save_bo_run`` writes a JSON file at the documented filename."""
        result = _make_bo_result()
        path = save_bo_run(result, directory=tmp_path)
        assert path.exists()
        assert path.is_file()
        # Filename schema: {scheme}_{kernel}_{mangled_HF_field}_{seed}.json.
        # HF field is layer7.max_spectral_abscissa → layer7_max_spectral_abscissa.
        assert path.name == "E4_classical_layer7_max_spectral_abscissa_1.json"
        assert path.parent == tmp_path
        # Payload is valid JSON with the documented top-level keys.
        payload = json.loads(path.read_text())
        for required in ("best_x", "best_objective", "scheme", "kernel",
                         "fidelity_levels", "eval_history"):
            assert required in payload, f"missing top-level key {required!r}"

    def test_serializer_handles_complex(self, tmp_path):
        """``_BOEncoder`` serialises ``complex`` values as ``[real, imag]``.

        Any layer that surfaces a :class:`KreissResult` (``layer2`` if it
        lands in fidelity_layers — see plan 46.2b) carries a ``witness_s``
        complex through ``extras``.  Without the encoder branch
        ``json.dump`` would raise ``TypeError``.
        """
        result = _make_bo_result()
        # Inject a complex in extras and rebuild a frozen dataclass with it.
        # ``BOResult`` is frozen, so we must build a fresh instance.
        from dataclasses import replace

        result_with_complex = replace(
            result,
            extras={"witness_s": 1.5 + 2.5j, "n_sentinel_filtered": 0},
        )
        path = save_bo_run(result_with_complex, directory=tmp_path)
        payload = json.loads(path.read_text())
        # Complex should round-trip as a 2-element list of floats.
        assert payload["extras"]["witness_s"] == [1.5, 2.5]

    def test_filename_includes_seed(self, tmp_path):
        """Same (scheme, kernel, objective) but distinct seeds → distinct files."""
        from dataclasses import replace

        base = _make_bo_result()
        r1 = replace(base, seed=1)
        r2 = replace(base, seed=2)
        p1 = save_bo_run(r1, directory=tmp_path)
        p2 = save_bo_run(r2, directory=tmp_path)
        assert p1 != p2
        assert p1.name.endswith("_1.json")
        assert p2.name.endswith("_2.json")
        # Both files exist and are independent.
        assert p1.exists() and p2.exists()
        assert json.loads(p1.read_text())["seed"] == 1
        assert json.loads(p2.read_text())["seed"] == 2

    def test_iter_bo_runs_sorted(self, tmp_path):
        """``iter_bo_runs`` yields ``*.json`` paths in sorted order."""
        from dataclasses import replace

        base = _make_bo_result()
        # Save in deliberately non-sorted order to exercise the sort.
        save_bo_run(replace(base, seed=3), directory=tmp_path)
        save_bo_run(replace(base, seed=1), directory=tmp_path)
        save_bo_run(replace(base, seed=2), directory=tmp_path)
        # Drop a non-JSON file to confirm it is ignored.
        (tmp_path / "ignore.txt").write_text("not json\n")

        seen = list(iter_bo_runs(tmp_path))
        assert all(p.suffix == ".json" for p in seen)
        assert seen == sorted(seen)
        assert [p.name for p in seen] == [
            "E4_classical_layer7_max_spectral_abscissa_1.json",
            "E4_classical_layer7_max_spectral_abscissa_2.json",
            "E4_classical_layer7_max_spectral_abscissa_3.json",
        ]


# ---------------------------------------------------------------------------
# TestBOCLI — sweeps.bo argparse + dispatch (plan 47.4d)
# ---------------------------------------------------------------------------


class TestBOCLI:
    """Argparse surface and dispatch wiring for ``sweeps bo``."""

    def test_argparse_minimal_invocation(self, monkeypatch, capsys):
        """Minimal accepting invocation routes (parsed args) into run_mfbo."""
        calls: list[dict] = []

        def _fake_run_mfbo(**kwargs):
            calls.append(kwargs)
            # Return a stub BOResult shaped to match the (cheap=1, HF=3) call.
            stub = _make_bo_result()
            from dataclasses import replace
            return replace(
                stub,
                fidelity_levels=(1, 3),
                hf_level=3,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                },
                cost_model={1: 0.076, 3: 0.038},
                n_evals_per_fidelity={1: 5, 3: 5},
                wall_time_per_fidelity={1: 0.4, 3: 0.2},
                bounds=((0.5, 20.0),),
                kernel="tension",
            )

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--cheap-fidelities", "1",
                "--bounds", "0.5", "20",
                "--budget-evals", "10",
                "--seed", "1",
            ]
        )
        assert rc == 0
        assert len(calls) == 1
        call = calls[0]
        assert call["scheme"] == "E4"
        assert call["kernel"] == "tension"
        assert call["seed"] == 1
        assert call["budget_evals"] == 10
        assert call["budget_seconds"] is None
        # report_fields_by_layer was assembled from --objective + --cheap-fidelities.
        assert call["report_fields_by_layer"] == {
            1: "layer1.boundary_gv_err",
            3: "layer3.max_stab_eig",
        }
        # 1D bounds for the tension kernel.
        assert list(call["bounds"]) == [(0.5, 20.0)]
        out = capsys.readouterr().out
        assert "BoTorch-qMFKG" in out
        assert "layer3.max_stab_eig" in out

    def test_argparse_rejects_no_budget(self):
        """Neither --budget-evals nor --budget-seconds → SystemExit (mutex required)."""
        with pytest.raises(SystemExit) as exc_info:
            bo_cli.main(
                [
                    "--scheme", "E4",
                    "--kernel", "tension",
                    "--objective", "layer3.max_stab_eig",
                    "--cheap-fidelities", "1",
                    "--bounds", "0.5", "20",
                ]
            )
        assert exc_info.value.code != 0

    def test_argparse_rejects_both_budgets(self):
        """Both --budget-evals and --budget-seconds → SystemExit (mutex)."""
        with pytest.raises(SystemExit) as exc_info:
            bo_cli.main(
                [
                    "--scheme", "E4",
                    "--kernel", "tension",
                    "--objective", "layer3.max_stab_eig",
                    "--cheap-fidelities", "1",
                    "--bounds", "0.5", "20",
                    "--budget-evals", "10",
                    "--budget-seconds", "5.0",
                ]
            )
        assert exc_info.value.code != 0

    def test_argparse_rejects_bad_field_layer(self, monkeypatch):
        """A cheap fidelity ≥ HF layer → parser.error (cheap > HF)."""
        # If the parser passes the bad spec through, run_mfbo would be
        # called.  Sentinel here detects that case and surfaces it.
        def _unexpected(**kwargs):
            raise AssertionError(
                "run_mfbo should not be invoked when cheap-fidelities >= HF"
            )

        monkeypatch.setattr(bo_cli, "run_mfbo", _unexpected)

        with pytest.raises(SystemExit) as exc_info:
            bo_cli.main(
                [
                    "--scheme", "E4",
                    "--kernel", "classical",
                    "--objective", "layer7.max_spectral_abscissa",
                    "--cheap-fidelities", "8",  # > HF inferred from layer7.*
                    "--budget-evals", "10",
                ]
            )
        assert exc_info.value.code != 0

    def test_dispatch_via_main(self):
        """``python -m sweeps bo --help`` exits 0 and lists the BO flags."""
        stencil_gen_dir = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env.setdefault("SYMPY_CACHE_SIZE", "50000")
        proc = subprocess.run(
            [sys.executable, "-m", "sweeps", "bo", "--help"],
            cwd=str(stencil_gen_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        for needle in (
            "--objective",
            "--cheap-fidelities",
            "--budget-evals",
            "--budget-seconds",
            "--baseline",
            "--validate-with-cpp",
            "--persist",
        ):
            assert needle in proc.stdout, (
                f"missing {needle!r} in `python -m sweeps bo --help` output"
            )

    def test_baseline_staged_invokes_run_staged_optimize(self, monkeypatch, capsys):
        """Plan 47.5b: ``--baseline staged`` actually runs ``run_staged_optimize``.

        Both ``run_mfbo`` and ``run_staged_optimize`` are monkeypatched so no
        cascade is invoked.  The test pins (a) ``run_staged_optimize`` is called,
        (b) the same seed flows into both methods (fairness), (c) the
        ``best_objective`` HF objective field flows through verbatim, and (d) the
        baseline record lands under ``result.extras["baseline"]`` and in the
        side-by-side print.

        Per the plan-46 lesson cited in 47.5a, ``--baseline staged`` must NOT
        emit the legacy "deferred" message; that path is removed in 47.5b.
        """
        from dataclasses import replace

        bo_calls: list[dict] = []
        staged_calls: list[dict] = []

        def _fake_run_mfbo(**kwargs):
            bo_calls.append(kwargs)
            stub = _make_bo_result()
            return replace(
                stub,
                fidelity_levels=(1, 3),
                hf_level=3,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                },
                cost_model={1: 0.076, 3: 0.038},
                n_evals_per_fidelity={1: 5, 3: 5},
                wall_time_per_fidelity={1: 0.4, 3: 0.2},
                bounds=((0.5, 20.0),),
                kernel="tension",
                best_x=np.array([10.0]),
                best_params={"sigma": 10.0},
                best_objective=-1.0e-3,
                extras={"n_sentinel_filtered": 0},
            )

        def _fake_run_staged_optimize(**kwargs):
            staged_calls.append(kwargs)
            from stencil_gen.optimizer import OptimizeResult as _OR

            return _OR(
                best_params={"sigma": 11.0},
                best_x=np.array([11.0]),
                best_objective=-2.0e-3,
                best_report={"failed_layer": None},
                method="staged",
                converged=True,
                n_evals=42,
                compute_time=1.5,
                history=[],
                extras={
                    "stage": "validated",
                    "validator_ranking": [(np.array([11.0]), -2.0e-3), (np.array([12.0]), -1.5e-3)],
                },
            )

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)
        monkeypatch.setattr(bo_cli, "run_staged_optimize", _fake_run_staged_optimize)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--cheap-fidelities", "1",
                "--bounds", "0.5", "20",
                "--budget-evals", "10",
                "--seed", "7",
                "--baseline", "staged",
            ]
        )
        assert rc == 0
        assert len(bo_calls) == 1
        assert len(staged_calls) == 1
        # Both runs share the seed.
        assert bo_calls[0]["seed"] == 7
        assert staged_calls[0]["seed"] == 7
        # Staged reuses the HF objective field that MF-BO targeted.
        assert staged_calls[0]["report_field"] == "layer3.max_stab_eig"
        # Staged reuses the same bounds.
        assert staged_calls[0]["bounds"] == [(0.5, 20.0)]
        # Plan-body literal: n_restarts=10 for the baseline.
        assert staged_calls[0]["n_restarts"] == 10
        # Plan 47.5b.1.1 (fix branch 1): the BO baseline must match
        # ``python -m sweeps optimize --method staged ...``'s CLI-resolved
        # defaults (the user-facing reference) rather than
        # ``run_staged_optimize``'s function-level defaults.  CLI form is
        # ``inner_max_layer=3``, ``inner_gate=max(3-1,0)=2``.  HF=L3 here ⇒
        # ``validator_max_layer=3`` (``max(hf_level, 3)`` fairness fix).
        assert staged_calls[0]["inner_gate"] == 2
        assert staged_calls[0]["inner_max_layer"] == 3
        assert staged_calls[0]["validator_max_layer"] == 3
        out = capsys.readouterr().out
        # New behaviour: side-by-side comparison appears.
        assert "comparison (side-by-side)" in out
        assert "staged" in out
        # Old stub message must NOT appear post-47.5b.
        assert "deferred" not in out

    def test_baseline_staged_uses_canonical_inner_gate_at_hf_l7(
        self, monkeypatch, capsys
    ):
        """Plan 47.5b.1.1: with HF=L7, the BO baseline matches the
        ``sweeps optimize --method staged`` CLI defaults (``inner_gate=2``,
        ``inner_max_layer=3``) and lifts ``validator_max_layer`` to 7
        (fairness fix tracking MF-BO's HF target).

        This is the second half of the 47.5b.1.1 contract: fairness-fix
        raises validator depth; the CLI-default inner-stage gate is
        passed explicitly so the baseline is what users actually invoke.
        """
        from dataclasses import replace

        bo_calls: list[dict] = []
        staged_calls: list[dict] = []

        def _fake_run_mfbo(**kwargs):
            bo_calls.append(kwargs)
            stub = _make_bo_result()
            return replace(
                stub,
                fidelity_levels=(1, 3, 5, 6, 7),
                hf_level=7,
                report_fields_by_layer={
                    1: "layer1.boundary_gv_err",
                    3: "layer3.max_stab_eig",
                    5: "layer_bl42.max_spectral_abscissa",
                    6: "layer6.transient_growth_bound",
                    7: "layer7.max_spectral_abscissa",
                },
                cost_model={1: 0.076, 3: 0.038, 5: 0.486, 6: 0.846, 7: 1.434},
                n_evals_per_fidelity={1: 5, 3: 3, 5: 1, 6: 1, 7: 2},
                wall_time_per_fidelity={1: 0.4, 3: 0.1, 5: 0.5, 6: 0.8, 7: 2.9},
                bounds=((-2.0, 2.0), (0.05, 2.0)),
                kernel="classical",
                best_x=np.array([-0.7733, 0.1624]),
                best_params={"alpha": [-0.7733, 0.1624]},
                best_objective=-1.0e-3,
                extras={"n_sentinel_filtered": 0},
            )

        def _fake_run_staged_optimize(**kwargs):
            staged_calls.append(kwargs)
            from stencil_gen.optimizer import OptimizeResult as _OR

            return _OR(
                best_params={"alpha": [-0.7733, 0.1624]},
                best_x=np.array([-0.7733, 0.1624]),
                best_objective=-2.0e-3,
                best_report={"failed_layer": None},
                method="staged",
                converged=True,
                n_evals=42,
                compute_time=1.5,
                history=[],
                extras={
                    "stage": "validated",
                    "validator_ranking": [
                        (np.array([-0.7733, 0.1624]), -2.0e-3),
                    ],
                },
            )

        monkeypatch.setattr(bo_cli, "run_mfbo", _fake_run_mfbo)
        monkeypatch.setattr(bo_cli, "run_staged_optimize", _fake_run_staged_optimize)

        rc = bo_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "classical",
                "--objective", "layer7.max_spectral_abscissa",
                "--cheap-fidelities", "1", "3", "5", "6",
                "--bounds", "-2", "2", "0.05", "2",
                "--budget-evals", "12",
                "--seed", "1",
                "--baseline", "staged",
            ]
        )
        assert rc == 0
        assert len(staged_calls) == 1
        # Fairness fix: validator depth tracks MF-BO's HF target.
        assert staged_calls[0]["validator_max_layer"] == 7
        # CLI-default inner-stage gate (plan 47.5b.1.1, fix branch 1):
        # ``inner_max_layer = min(3, validator_max_layer) = 3``,
        # ``inner_gate = max(inner_max_layer - 1, 0) = 2``.  Same values
        # for HF=L3 and HF=L7 because ``min(3, ...)`` saturates at 3.
        assert staged_calls[0]["inner_gate"] == 2
        assert staged_calls[0]["inner_max_layer"] == 3
        capsys.readouterr()
