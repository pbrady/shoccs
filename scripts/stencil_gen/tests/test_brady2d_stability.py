"""Tests for stencil_gen.brady2d_stability — layered stability scoring."""

import json
from pathlib import Path

import numpy as np
import pytest

from stencil_gen.brady2d_stability import (
    L1_TOL,
    STABILITY_TOL,
    StabilityReport,
    layer1_interior_boundary_gv,
    layer3_1d_eigenvalue,
)


KNOWN_VALUES_PATH = Path(__file__).parent.parent / "sweeps" / "known_values.json"


def _load_known_values():
    with open(KNOWN_VALUES_PATH) as f:
        return json.load(f)


class TestLayer1:
    """Layer 1: interior + boundary group velocity error."""

    def test_layer1_classical_e4_passes(self):
        """Classical E4 with known-good alpha produces boundary_gv_err < L1_TOL."""
        # Known-good alpha values from E4u_1.t.cpp
        alpha = [-0.7733323791884821, 0.1623961700641681]
        result = layer1_interior_boundary_gv(
            "E4", "classical", {"alpha": alpha},
        )
        assert result["boundary_gv_err"] < L1_TOL, (
            f"boundary_gv_err={result['boundary_gv_err']:.6f} >= {L1_TOL}"
        )
        assert result["interior_gv_err_x"] < L1_TOL
        assert result["interior_gv_err_y"] < L1_TOL
        assert 0 < result["cutoff_fraction"] < 1

    def test_layer1_tension_e4_passes(self):
        """Tension E4 at sigma=3.0 passes L1."""
        result = layer1_interior_boundary_gv(
            "E4", "tension", {"sigma": 3.0},
        )
        assert result["boundary_gv_err"] < L1_TOL, (
            f"boundary_gv_err={result['boundary_gv_err']:.6f} >= {L1_TOL}"
        )
        assert result["interior_gv_err_x"] < L1_TOL
        assert result["interior_gv_err_y"] < L1_TOL
        assert 0 < result["cutoff_fraction"] < 1

    def test_layer1_gaussian_e4_known_unstable_still_passes_at_this_layer(self):
        """Gaussian eps=0.1 (known_unstable) passes L1.

        Confirms that L1 is necessary but not sufficient — this scheme is
        eigenvalue-unstable (fails at L2 or L3) but has acceptable low-frequency
        dispersion.
        """
        kv = _load_known_values()
        unstable = kv["E4_1"]["known_unstable"][0]
        assert unstable["kernel"] == "gaussian"
        eps = unstable["epsilon"]

        result = layer1_interior_boundary_gv(
            "E4", "gaussian", {"epsilon": eps},
        )
        assert result["boundary_gv_err"] < L1_TOL, (
            f"known-unstable Gaussian eps={eps} boundary_gv_err="
            f"{result['boundary_gv_err']:.6f} should pass L1"
        )
        assert result["interior_gv_err_x"] < L1_TOL
        assert result["interior_gv_err_y"] < L1_TOL

    def test_layer1_return_keys(self):
        """Layer 1 returns all expected keys."""
        result = layer1_interior_boundary_gv(
            "E4", "tension", {"sigma": 3.0},
        )
        expected_keys = {
            "interior_gv_err_x",
            "interior_gv_err_y",
            "boundary_gv_err",
            "cutoff_fraction",
        }
        assert set(result.keys()) == expected_keys

    def test_layer1_interior_symmetry(self):
        """interior_gv_err_x == interior_gv_err_y on Cartesian grid."""
        result = layer1_interior_boundary_gv(
            "E4", "tension", {"sigma": 3.0},
        )
        assert result["interior_gv_err_x"] == result["interior_gv_err_y"]


class TestStabilityReport:
    """Basic tests for the StabilityReport dataclass."""

    def test_default_values(self):
        report = StabilityReport()
        assert report.layer1 is None
        assert report.failed_layer is None
        assert report.overall_verdict == "unknown"
        assert report.compute_time == 0.0

    def test_with_layer1(self):
        result = {"interior_gv_err_x": 0.01, "boundary_gv_err": 0.02}
        report = StabilityReport(layer1=result, overall_verdict="pass")
        assert report.layer1 == result
        assert report.overall_verdict == "pass"


class TestLayer3:
    """Layer 3: 1D eigenvalue stability check at multiple grid sizes."""

    def test_classical_e4_stable(self):
        """Classical E4 with known-good alpha is stable at all grid sizes."""
        alpha = [-0.7733323791884821, 0.1623961700641681]
        result = layer3_1d_eigenvalue("E4", "classical", {"alpha": alpha})
        for n, se in result["eigenvalues"].items():
            assert se <= STABILITY_TOL, (
                f"Classical E4 unstable at n={n}: max Re(lambda)={se:.6e}"
            )
        assert result["max_stab_eig"] <= STABILITY_TOL

    def test_tension_e4_sigma_3_stable(self):
        """Tension E4 at sigma=3.0 is stable at all grid sizes."""
        result = layer3_1d_eigenvalue("E4", "tension", {"sigma": 3.0})
        for n, se in result["eigenvalues"].items():
            assert se <= STABILITY_TOL, (
                f"Tension E4 sigma=3.0 unstable at n={n}: max Re(lambda)={se:.6e}"
            )
        assert result["max_stab_eig"] <= STABILITY_TOL

    def test_gaussian_e4_eps_01_unstable(self):
        """Gaussian E4 eps=0.1 (known_unstable) fails L3."""
        kv = _load_known_values()
        unstable = kv["E4_1"]["known_unstable"][0]
        assert unstable["kernel"] == "gaussian"
        eps = unstable["epsilon"]

        result = layer3_1d_eigenvalue(
            "E4", "gaussian", {"epsilon": eps},
        )
        assert result["max_stab_eig"] > STABILITY_TOL, (
            f"Expected Gaussian eps={eps} to be unstable, got "
            f"max_stab_eig={result['max_stab_eig']:.6e}"
        )

    def test_return_keys(self):
        """Layer 3 returns expected keys."""
        result = layer3_1d_eigenvalue("E4", "tension", {"sigma": 3.0})
        assert "eigenvalues" in result
        assert "max_stab_eig" in result
        assert set(result["eigenvalues"].keys()) == {20, 40, 80}

    def test_custom_n_values(self):
        """Custom n_values are respected."""
        result = layer3_1d_eigenvalue(
            "E4", "tension", {"sigma": 3.0}, n_values=(15, 30),
        )
        assert set(result["eigenvalues"].keys()) == {15, 30}
