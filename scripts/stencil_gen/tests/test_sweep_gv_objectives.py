"""Unit tests for sweeps.gv_objectives helper wrappers.

These tests exercise the thin scalar helpers used by sweep scripts.  They
should all run well under 2 seconds in aggregate so they stay in the default
(non-slow) suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stencil_gen.phs import build_diff_matrix_rbf

from sweeps import _common as sweeps_common
from sweeps import tension_sweep
from sweeps.gv_objectives import (
    boundary_gv_error_max,
    cutcell_gv_min_C,
    gv_score_from_matrix,
    interior_cutoff_fraction,
    interior_gv_error_max,
)

KNOWN_VALUES_PATH = Path(__file__).parent.parent / "sweeps" / "known_values.json"


def _load_known():
    with open(KNOWN_VALUES_PATH) as f:
        return json.load(f)


def test_interior_gv_error_max_e2_positive_finite():
    val = interior_gv_error_max(p=1, nu=1, n_xi=100)
    assert np.isfinite(val)
    assert val > 0.0


def test_interior_cutoff_fraction_improves_with_order():
    f2 = interior_cutoff_fraction(p=1, nu=1, n_xi=200)
    f4 = interior_cutoff_fraction(p=2, nu=1, n_xi=200)
    assert f4 > f2  # higher-order scheme resolves more of the spectrum


def test_interior_cutoff_fraction_in_unit_interval():
    frac = interior_cutoff_fraction(p=2, nu=1, n_xi=100)
    assert 0.0 < frac <= 1.0 + 1e-12


def test_boundary_gv_error_max_e2_tension_at_known_sigma():
    kv = _load_known()
    sigma = kv["E2_1"]["tension"]["sigma"]
    val = boundary_gv_error_max(
        p=1, q=1, nextra=1, nu=1, sigma=sigma, kernel="tension", n_xi=100,
    )
    assert np.isfinite(val)
    assert val > 0.0


def test_boundary_gv_error_max_e4_tension_at_known_sigma():
    kv = _load_known()
    sigma = kv["E4_1"]["tension"]["sigma"]
    val = boundary_gv_error_max(
        p=2, q=3, nextra=0, nu=1, sigma=sigma, kernel="tension", n_xi=100,
    )
    assert np.isfinite(val)
    assert val > 0.0


def test_cutcell_gv_min_C_e2_returns_finite_tuple():
    """cutcell_gv_min_C on E2_1 returns (finite float, bool) at small psi grid."""
    from stencil_gen.temo import E2_1

    result = cutcell_gv_min_C(
        E2_1,
        psi_values=np.linspace(0.05, 0.95, 5),
        alpha_values={},
        n_xi=100,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    min_C, has_sign_reversal = result
    assert np.isfinite(min_C)
    assert isinstance(has_sign_reversal, bool)


def test_gv_score_from_matrix_matches_boundary_helper():
    """gv_score_from_matrix on a real D should agree with boundary_gv_error_max."""
    kv = _load_known()
    sigma = kv["E2_1"]["tension"]["sigma"]
    D = build_diff_matrix_rbf(
        40, p=1, q=1, epsilon=sigma, kernel="tension", nu=1, nextra=1,
    )
    from_matrix = gv_score_from_matrix(D, n_xi=100)
    from_helper = boundary_gv_error_max(
        p=1, q=1, nextra=1, nu=1, sigma=sigma, kernel="tension", n_xi=100,
    )
    assert np.isclose(from_matrix["max_gv_error"], from_helper, rtol=1e-12)
    assert 0.0 < from_matrix["min_cutoff_xi"] <= np.pi + 1e-12


def test_gv_score_from_matrix_small_hardcoded():
    """Deterministic smoke: tiny hand-built matrix produces finite results."""
    # 5-point matrix: first two rows boundary-like (start at column 0),
    # remaining rows interior-shifted so they are ignored by the scanner.
    D = np.zeros((5, 5))
    # Row 0: forward difference from column 0
    D[0, 0] = -1.0
    D[0, 1] = 1.0
    # Row 1: still starts at column 0 (boundary block)
    D[1, 0] = -0.5
    D[1, 1] = 0.0
    D[1, 2] = 0.5
    # Rows 2..4: centered interior (leftmost nonzero > 0 → scanner stops)
    for i in range(2, 4):
        D[i, i - 1] = -0.5
        D[i, i + 1] = 0.5
    score = gv_score_from_matrix(D, n_xi=50)
    assert np.isfinite(score["max_gv_error"])
    assert np.isfinite(score["min_cutoff_xi"])
    assert score["max_gv_error"] > 0.0


def _seed_kv(path: Path) -> dict:
    seed = {
        "E2_1": {
            "params": {"p": 1, "q": 1, "nextra": 1, "nu": 1},
            "tension": {
                "sigma": 6.0,
                "stable_at": [20, 40, 80],
                "gv_error": 1.234,
                "preexisting_extra_key": "survive",
            },
            "tension_gv": {
                "sigma": 5.5,
                "gv_error": 1.234,
                "stable_at": [20, 40],
            },
        }
    }
    with open(path, "w") as f:
        json.dump(seed, f, indent=2)
    return seed


def test_tension_sweep_main_merges_known_values(tmp_path, monkeypatch, capsys):
    """Regression for 40.2d: --update-known-values must merge, not overwrite.

    A non-GV --update-known-values invocation must preserve any pre-existing
    keys that an earlier --include-gv run wrote (gv_error on tension, the
    full tension_gv entry).  A subsequent --include-gv invocation must
    refresh those keys without dropping unrelated keys (preexisting_extra_key).
    """
    kv_path = tmp_path / "known_values.json"
    monkeypatch.setattr(sweeps_common, "KNOWN_VALUES_PATH", kv_path)
    _seed_kv(kv_path)

    rc = tension_sweep.main([
        "--scheme", "E2",
        "--n-sigma", "5",
        "--n-values", "20",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        after_non_gv = json.load(f)
    tension = after_non_gv["E2_1"]["tension"]
    assert tension["gv_error"] == 1.234
    assert tension["preexisting_extra_key"] == "survive"
    assert "sigma" in tension and "stable_at" in tension
    assert after_non_gv["E2_1"]["tension_gv"] == {
        "sigma": 5.5,
        "gv_error": 1.234,
        "stable_at": [20, 40],
    }

    rc = tension_sweep.main([
        "--scheme", "E2",
        "--n-sigma", "5",
        "--n-values", "20",
        "--include-gv",
        "--update-known-values",
    ])
    assert rc == 0
    capsys.readouterr()

    with open(kv_path) as f:
        after_gv = json.load(f)
    tension = after_gv["E2_1"]["tension"]
    assert tension["preexisting_extra_key"] == "survive"
    assert "sigma" in tension
    assert "stable_at" in tension
    assert np.isfinite(tension["gv_error"])
    tension_gv = after_gv["E2_1"]["tension_gv"]
    assert set(tension_gv) == {"sigma", "gv_error", "stable_at"}
    assert np.isfinite(tension_gv["sigma"])
    assert np.isfinite(tension_gv["gv_error"])
    assert isinstance(tension_gv["stable_at"], list)
