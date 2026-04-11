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

from sweeps.gv_objectives import (
    boundary_gv_error_max,
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
