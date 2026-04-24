"""Unit tests for :mod:`sweeps.pareto` (plan 45.3c).

Covers:

- :func:`sweeps.pareto._mangle_objectives` — filename mangling round-trip.
- :func:`sweeps.pareto.main` argparse surface — minimal accepting invocation,
  single-objective rejection, odd-bound-parity rejection.
- ``python -m sweeps pareto --help`` subprocess smoke — confirms the dispatch
  wiring from ``sweeps/__main__.py`` survived the registration of the new
  subcommand.

``TestParetoCLI::test_argparse_accepts_minimal_invocation`` stubs
:func:`sweeps.pareto.run_nsga2` via ``monkeypatch`` so no pymoo / brady2d
pipeline is entered — keeps the test in the fast suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stencil_gen.pareto import ParetoPoint, ParetoResult

from sweeps import pareto as pareto_cli


# ---------------------------------------------------------------------------
# _mangle_objectives
# ---------------------------------------------------------------------------


class TestMangleObjectives:
    def test_roundtrip_legible(self):
        fields = ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"]
        assert (
            pareto_cli._mangle_objectives(fields)
            == "layer1_boundary_gv_err__layer_bl42_max_spectral_abscissa"
        )

    def test_order_preserved(self):
        a = ["layer1.boundary_gv_err", "layer3.max_stab_eig"]
        b = ["layer3.max_stab_eig", "layer1.boundary_gv_err"]
        assert pareto_cli._mangle_objectives(a) != pareto_cli._mangle_objectives(b)


# ---------------------------------------------------------------------------
# main(argv) argparse / dispatch
# ---------------------------------------------------------------------------


def _stub_result(
    *,
    objective_fields: tuple[str, ...],
    scheme: str = "E4",
    kernel: str = "classical",
    bounds: tuple[tuple[float, float], ...] = ((-2.0, 2.0), (0.05, 2.0)),
    pop_size: int = 6,
    n_gen: int = 2,
    seed: int = 1,
) -> ParetoResult:
    """Build a minimal, deterministic ParetoResult for CLI stubbing."""
    n_obj = len(objective_fields)
    n_var = len(bounds)
    pt = ParetoPoint(
        x=np.zeros(n_var, dtype=float),
        params={"alpha": [0.0] * n_var} if kernel == "classical" else {},
        objectives=np.arange(n_obj, dtype=float) + 1.0,
        report={},
    )
    return ParetoResult(
        front=(pt,),
        objective_fields=objective_fields,
        scheme=scheme,
        kernel=kernel,
        bounds=bounds,
        method="NSGA-II",
        pop_size=pop_size,
        n_gen=n_gen,
        n_evals=pop_size * n_gen,
        seed=seed,
        compute_time=0.001,
        hv_trace=(0.1, 0.2),
        ref_point=tuple(1.0 for _ in range(n_obj)),
        extras={"n_sentinel_filtered": 0, "hv_n_nds": (1, 1)},
    )


class TestParetoCLI:
    def test_argparse_accepts_minimal_invocation(self, monkeypatch, capsys):
        """Minimal classical/E4 invocation with mocked run_nsga2 exits 0."""
        calls: list[dict] = []

        def _fake_run_nsga2(**kwargs):
            calls.append(kwargs)
            return _stub_result(
                objective_fields=tuple(kwargs["report_fields"]),
                scheme=kwargs["scheme"],
                kernel=kwargs["kernel"],
                bounds=tuple(kwargs["bounds"]),
                pop_size=kwargs["pop_size"],
                n_gen=kwargs["n_gen"],
                seed=kwargs["seed"],
            )

        monkeypatch.setattr(pareto_cli, "run_nsga2", _fake_run_nsga2)

        rc = pareto_cli.main(
            [
                "--scheme", "E4",
                "--kernel", "classical",
                "--objectives",
                "layer1.boundary_gv_err",
                "layer3.max_stab_eig",
                "--pop-size", "6",
                "--n-gen", "2",
                "--seed", "1",
            ]
        )
        assert rc == 0
        assert len(calls) == 1
        call = calls[0]
        assert call["scheme"] == "E4"
        assert call["kernel"] == "classical"
        assert list(call["report_fields"]) == [
            "layer1.boundary_gv_err",
            "layer3.max_stab_eig",
        ]
        assert call["pop_size"] == 6
        assert call["n_gen"] == 2
        assert call["seed"] == 1
        # Default bounds for (E4, classical) are 2D, matching the classical kernel.
        assert len(call["bounds"]) == 2
        out = capsys.readouterr().out
        assert "NSGA-II" in out
        assert "layer1.boundary_gv_err" in out

    def test_argparse_rejects_single_objective(self):
        """A single --objectives value exits non-zero (min 2 enforced)."""
        with pytest.raises(SystemExit) as exc_info:
            pareto_cli.main(
                [
                    "--scheme", "E4",
                    "--kernel", "classical",
                    "--objectives", "layer3.max_stab_eig",
                    "--pop-size", "6",
                    "--n-gen", "2",
                ]
            )
        assert exc_info.value.code != 0

    def test_argparse_rejects_bad_bounds_parity(self):
        """An odd number of --bounds values exits non-zero (pairs required)."""
        with pytest.raises(SystemExit) as exc_info:
            pareto_cli.main(
                [
                    "--scheme", "E4",
                    "--kernel", "classical",
                    "--objectives",
                    "layer1.boundary_gv_err",
                    "layer3.max_stab_eig",
                    "--bounds", "-2.0", "2.0", "0.05",
                    "--pop-size", "6",
                    "--n-gen", "2",
                ]
            )
        assert exc_info.value.code != 0

    def test_dispatch_registered(self):
        """`python -m sweeps pareto --help` is routed through __main__."""
        stencil_gen_dir = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env.setdefault("SYMPY_CACHE_SIZE", "50000")
        proc = subprocess.run(
            [sys.executable, "-m", "sweeps", "pareto", "--help"],
            cwd=str(stencil_gen_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, (
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        # The top-level dispatch parser prints its own --help for the
        # subparser; spot-check a handful of options.
        for needle in ("--objectives", "--pop-size", "--n-gen", "--seed"):
            assert needle in proc.stdout, (
                f"missing {needle!r} in `python -m sweeps pareto --help` output"
            )
