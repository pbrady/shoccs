"""Tests for stencil_gen.brady2d_stability — layered stability scoring."""

import json
from pathlib import Path

import numpy as np
import pytest

from stencil_gen.brady2d_stability import (
    L1_TOL,
    L4_TOL,
    L5_TOL,
    L7_TOL,
    L7_TRANSIENT_GROWTH_TOL,
    STABILITY_TOL,
    StabilityReport,
    build_sparse_2d_operator,
    layer1_interior_boundary_gv,
    layer3_1d_eigenvalue,
    layer4_local_gv_2d,
    layer5_anisotropy,
    layer7_sparse_2d_eigenvalue,
    layer7_with_non_normality,
)
from stencil_gen.gks_kreiss import KreissResult
from stencil_gen.non_normality import NonNormalityReport
from stencil_gen.group_velocity import local_group_velocity_2d_varying


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
    """Tests for the StabilityReport dataclass."""

    def test_default_values(self):
        report = StabilityReport()
        assert report.layer1 is None
        assert report.layer2 is None
        assert report.layer3 is None
        assert report.layer4 is None
        assert report.layer5 is None
        assert report.layer6 is None
        assert report.layer7 is None
        assert report.non_normality is None
        assert report.kreiss is None
        assert report.failed_layer is None
        assert report.failed_reason == ""
        assert report.overall_verdict == "unknown"
        assert report.compute_time == 0.0

    def test_empty_factory(self):
        report = StabilityReport.empty()
        assert report.layer1 is None
        assert report.layer7 is None
        assert report.overall_verdict == "unknown"
        assert report.compute_time == 0.0

    def test_with_layer1(self):
        result = {"interior_gv_err_x": 0.01, "boundary_gv_err": 0.02}
        report = StabilityReport(layer1=result, overall_verdict="pass")
        assert report.layer1 == result
        assert report.overall_verdict == "pass"

    def test_with_kreiss_result(self):
        kr = KreissResult(is_stable=True, compute_time=0.5)
        report = StabilityReport(layer2=kr, kreiss=kr, overall_verdict="pass")
        assert report.layer2 is kr
        assert report.kreiss is kr
        assert report.layer2.is_stable is True

    def test_with_non_normality(self):
        nn = NonNormalityReport(
            spectral_abscissa=-1.0,
            numerical_abscissa=5.0,
            henrici_departure=0.1,
            eigenvector_condition=10.0,
            pseudospectral_abscissae={1e-2: -0.5},
            kreiss_constant=3.0,
            transient_growth_bound=3.0 * np.e,
            n=100,
            compute_time=1.0,
        )
        report = StabilityReport(non_normality=nn, overall_verdict="pass")
        assert report.non_normality is nn
        assert report.non_normality.kreiss_constant == 3.0

    def test_failed_report(self):
        report = StabilityReport(
            layer1={"boundary_gv_err": 0.01},
            layer3={"max_stab_eig": 0.5},
            overall_verdict="fail",
            failed_layer=3,
            failed_reason="max_stab_eig=0.5 > STABILITY_TOL",
        )
        assert report.overall_verdict == "fail"
        assert report.failed_layer == 3
        assert "max_stab_eig" in report.failed_reason

    def test_str_minimal(self):
        """__str__ works on a minimal (empty) report."""
        report = StabilityReport.empty()
        s = str(report)
        assert "Brady-Livescu 2D Stability Report" in s
        assert "UNKNOWN" in s

    def test_str_with_layers(self):
        """__str__ shows layer summaries when populated."""
        report = StabilityReport(
            layer1={"boundary_gv_err": 0.01, "interior_gv_err_x": 0.001},
            layer3={"max_stab_eig": -1e-14},
            layer4={"max_local_gv_error": 0.003},
            layer5={"max_aligned_error": 0.02},
            layer7={"max_spectral_abscissa": -0.001},
            overall_verdict="pass",
            compute_time=2.5,
        )
        s = str(report)
        assert "L1" in s
        assert "L3" in s
        assert "L4" in s
        assert "L5" in s
        assert "L7" in s
        assert "PASS" in s
        assert "2.50s" in s

    def test_str_failed_report(self):
        """__str__ shows failure info when a layer fails."""
        report = StabilityReport(
            layer1={"boundary_gv_err": 0.01},
            overall_verdict="fail",
            failed_layer=3,
            failed_reason="max_stab_eig > STABILITY_TOL",
            compute_time=0.1,
        )
        s = str(report)
        assert "FAIL" in s
        assert "layer 3" in s


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


class TestLayer4:
    """Layer 4: per-point local GV error on the Brady-Livescu 2D field."""

    def test_classical_e4_passes(self):
        """Classical E4 passes L4 on the BL coefficient field."""
        result = layer4_local_gv_2d("E4", "classical", {})
        assert result["max_local_gv_error"] <= L4_TOL, (
            f"E4 local GV error {result['max_local_gv_error']:.4f} exceeds L4_TOL={L4_TOL}"
        )

    def test_e2_has_larger_error_than_e4(self):
        """E2 has larger local GV error than E4 (lower-order → worse dispersion)."""
        r2 = layer4_local_gv_2d("E2", "tension", {"sigma": 0.0}, N=21)
        r4 = layer4_local_gv_2d("E4", "tension", {"sigma": 3.0}, N=21)
        assert r2["max_local_gv_error"] > r4["max_local_gv_error"], (
            "E2 should have larger local GV error than E4"
        )

    def test_return_keys(self):
        """Layer 4 returns expected keys."""
        result = layer4_local_gv_2d("E4", "tension", {"sigma": 3.0}, N=11)
        assert "max_local_gv_error" in result
        assert "worst_point" in result
        assert "worst_xi" in result
        assert isinstance(result["worst_point"], tuple)
        assert len(result["worst_point"]) == 2

    def test_worst_xi_in_range(self):
        """worst_xi should be in (0, pi]."""
        result = layer4_local_gv_2d("E4", "tension", {"sigma": 3.0}, N=11)
        assert 0.0 < result["worst_xi"] <= np.pi

    def test_synthetic_bad_stencil_fails(self):
        """A synthetic stencil with poor GV exceeds L4_TOL.

        Weights [0.5, -0.5] with offsets [-1, 0] give C(xi) = -0.5*cos(xi),
        which deviates massively from the exact GV of 1.  With c_x == 1
        everywhere the scaled error is |C(xi) - 1| > 1 for all xi, far
        exceeding L4_TOL = 0.1.
        """
        bad_weights = np.array([0.5, -0.5])
        bad_offsets = np.array([-1.0, 0.0])
        bad_stencil = (bad_weights, bad_offsets)

        N = 11
        c_x = np.ones((N, N))
        c_y = np.zeros((N, N))
        xi_array = np.linspace(0.01, np.pi, 50)

        result = local_group_velocity_2d_varying(
            bad_stencil, bad_stencil, c_x, c_y, xi_array,
        )
        err_x = np.abs(result["gv_error_x_field"])
        max_err = float(np.max(err_x))
        assert max_err > L4_TOL, (
            f"Synthetic bad stencil max GV error {max_err:.4f} "
            f"should exceed L4_TOL={L4_TOL}"
        )


class TestLayer5:
    """Layer 5: 2D anisotropy over the Brady-Livescu coefficient field."""

    def test_classical_e4_passes(self):
        """Classical E4 passes L5 on the BL coefficient field."""
        result = layer5_anisotropy("E4", "classical", {})
        assert result["max_aligned_error"] <= L5_TOL, (
            f"E4 anisotropy error {result['max_aligned_error']:.6f} exceeds L5_TOL={L5_TOL}"
        )

    def test_tension_e4_passes(self):
        """Tension E4 at sigma=3.0 passes L5."""
        result = layer5_anisotropy("E4", "tension", {"sigma": 3.0})
        assert result["max_aligned_error"] <= L5_TOL, (
            f"Tension E4 anisotropy error {result['max_aligned_error']:.6f} exceeds L5_TOL={L5_TOL}"
        )

    def test_e2_has_larger_anisotropy_than_e4(self):
        """E2 has larger anisotropy error than E4 (lower-order → more anisotropic)."""
        r2 = layer5_anisotropy("E2", "tension", {"sigma": 0.0}, N=21)
        r4 = layer5_anisotropy("E4", "tension", {"sigma": 3.0}, N=21)
        assert r2["max_aligned_error"] > r4["max_aligned_error"], (
            "E2 should have larger anisotropy error than E4"
        )

    def test_return_keys(self):
        """Layer 5 returns expected keys."""
        result = layer5_anisotropy("E4", "tension", {"sigma": 3.0}, N=11)
        expected_keys = {"max_aligned_error", "worst_point", "worst_theta"}
        assert set(result.keys()) == expected_keys
        assert isinstance(result["worst_point"], tuple)
        assert len(result["worst_point"]) == 2

    def test_worst_point_in_range(self):
        """worst_point indices should be within the grid dimensions."""
        N = 21
        result = layer5_anisotropy("E4", "tension", {"sigma": 3.0}, N=N)
        i, j = result["worst_point"]
        assert 0 <= i < N
        assert 0 <= j < N

    def test_worst_theta_in_range(self):
        """worst_theta should be in a reasonable angular range."""
        result = layer5_anisotropy("E4", "tension", {"sigma": 3.0}, N=11)
        # BL field has angles roughly in (0, pi/4), so worst_theta
        # should be in the first quadrant.
        assert 0.0 < result["worst_theta"] < np.pi / 2

    def test_large_xi_mag_exceeds_threshold(self):
        """At a large wavenumber (80% of cutoff), E2 anisotropy exceeds L5_TOL.

        The default layer5_anisotropy uses 20% of the cutoff, where E2 just
        barely passes (0.048 vs L5_TOL=0.05).  At 80% of the cutoff the
        anisotropy error is much larger, providing the failure-side test for
        the L5 threshold.
        """
        from stencil_gen.benchmarks.brady_livescu_2d import make_coefficient_field
        from stencil_gen.group_velocity import (
            anisotropy_over_coefficient_field,
            interior_group_velocity,
        )

        N = 21
        _, _, c_x, c_y = make_coefficient_field(N)
        theta_array = np.linspace(0.01, np.pi / 2 - 0.01, 200)

        # E2 interior cutoff at p=1
        profile = interior_group_velocity(1, 1, np.linspace(0.01, np.pi, 200))
        xi_mag = profile.cutoff_xi * 0.8  # 80% of cutoff instead of 20%

        result = anisotropy_over_coefficient_field(
            "E2", c_x, c_y, theta_array, xi_mag,
        )
        assert result["max_aligned_error"] > L5_TOL, (
            f"E2 at 80% cutoff xi_mag={xi_mag:.3f}: "
            f"max_aligned_error={result['max_aligned_error']:.6f} "
            f"should exceed L5_TOL={L5_TOL}"
        )


class TestBuildSparse2D:
    """Tests for build_sparse_2d_operator."""

    def test_shape_n11(self):
        """At N=11, the reduced operator has shape (100, 100)."""
        L_red, keep_idx = build_sparse_2d_operator(
            "E4", "tension", {"sigma": 3.0}, N=11,
        )
        assert L_red.shape == (100, 100), f"Expected (100, 100), got {L_red.shape}"
        assert len(keep_idx) == 100

    def test_shape_n21(self):
        """At N=21, the reduced operator has shape (400, 400)."""
        L_red, keep_idx = build_sparse_2d_operator(
            "E4", "tension", {"sigma": 3.0}, N=21,
        )
        assert L_red.shape == (400, 400), f"Expected (400, 400), got {L_red.shape}"
        assert len(keep_idx) == 400

    def test_keep_idx_excludes_inflow(self):
        """keep_idx excludes all DOFs with i=0 or j=0."""
        N = 11
        _, keep_idx = build_sparse_2d_operator(
            "E4", "tension", {"sigma": 3.0}, N=N,
        )
        ii = keep_idx % N
        jj = keep_idx // N
        assert np.all(ii > 0), "keep_idx should exclude i=0 (x inflow)"
        assert np.all(jj > 0), "keep_idx should exclude j=0 (y inflow)"

    def test_sparse_format(self):
        """Output is CSR sparse matrix."""
        import scipy.sparse as sp

        L_red, _ = build_sparse_2d_operator(
            "E4", "tension", {"sigma": 3.0}, N=11,
        )
        assert sp.issparse(L_red), "L_red should be sparse"
        assert L_red.format == "csr", f"Expected CSR, got {L_red.format}"

    def test_classical_e4_builds(self):
        """Classical E4 kernel builds without error."""
        alpha = [-0.7733323791884821, 0.1623961700641681]
        L_red, keep_idx = build_sparse_2d_operator(
            "E4", "classical", {"alpha": alpha}, N=11,
        )
        assert L_red.shape == (100, 100)

    def test_eigenvalues_finite(self):
        """Eigenvalues of the reduced operator are all finite."""
        L_red, _ = build_sparse_2d_operator(
            "E4", "tension", {"sigma": 3.0}, N=11,
        )
        eigs = np.linalg.eigvals(L_red.toarray())
        assert np.all(np.isfinite(eigs)), "All eigenvalues should be finite"


class TestLayer7:
    """Layer 7: sparse 2D Arnoldi eigenvalue check on the BL operator."""

    def test_tension_e4_stable(self):
        """Tension E4 at sigma=3.0 is stable at small grid sizes."""
        result = layer7_sparse_2d_eigenvalue(
            "E4", "tension", {"sigma": 3.0}, n_values=(21, 31),
        )
        for n, max_re in result["eigenvalues"].items():
            assert max_re <= L7_TOL, (
                f"Tension E4 sigma=3.0 unstable at N={n}: "
                f"max Re(lambda)={max_re:.6e}"
            )
        assert result["max_spectral_abscissa"] <= L7_TOL

    def test_classical_e4_stable(self):
        """Classical E4 with known-good alpha is stable at N=21."""
        alpha = [-0.7733323791884821, 0.1623961700641681]
        result = layer7_sparse_2d_eigenvalue(
            "E4", "classical", {"alpha": alpha}, n_values=(21,),
        )
        assert result["max_spectral_abscissa"] <= L7_TOL, (
            f"Classical E4 unstable: "
            f"max_spectral_abscissa={result['max_spectral_abscissa']:.6e}"
        )

    def test_gaussian_e4_eps_01_unstable(self):
        """Gaussian E4 eps=0.1 (known_unstable) fails L7."""
        kv = _load_known_values()
        unstable = kv["E4_1"]["known_unstable"][0]
        assert unstable["kernel"] == "gaussian"
        eps = unstable["epsilon"]

        result = layer7_sparse_2d_eigenvalue(
            "E4", "gaussian", {"epsilon": eps}, n_values=(21,),
        )
        assert result["max_spectral_abscissa"] > L7_TOL, (
            f"Expected Gaussian eps={eps} to be unstable, got "
            f"max_spectral_abscissa={result['max_spectral_abscissa']:.6e}"
        )

    def test_return_keys(self):
        """Layer 7 returns expected keys."""
        result = layer7_sparse_2d_eigenvalue(
            "E4", "tension", {"sigma": 3.0}, n_values=(21,),
        )
        assert "eigenvalues" in result
        assert "max_spectral_abscissa" in result
        assert 21 in result["eigenvalues"]

    def test_custom_n_values(self):
        """Custom n_values are respected."""
        result = layer7_sparse_2d_eigenvalue(
            "E4", "tension", {"sigma": 3.0}, n_values=(11, 21),
        )
        assert set(result["eigenvalues"].keys()) == {11, 21}


class TestLayer7WithNonNormality:
    """Layer 7 + L6: non-normality diagnostics on the full 2D BL operator."""

    @pytest.mark.slow
    def test_classical_e4_passes(self):
        """Classical E4 with known-good alpha passes the combined L7+L6 check.

        Both the spectral abscissa and transient growth bound must be within
        tolerances for a scheme that is known to be long-time stable in the
        Brady-Livescu benchmark.
        """
        alpha = [-0.7733323791884821, 0.1623961700641681]
        report = layer7_with_non_normality(
            "E4", "classical", {"alpha": alpha}, N=21,
        )
        assert report.spectral_abscissa <= L7_TOL, (
            f"Classical E4 spectral_abscissa={report.spectral_abscissa:.6e} "
            f"exceeds L7_TOL={L7_TOL}"
        )
        assert report.transient_growth_bound <= L7_TRANSIENT_GROWTH_TOL, (
            f"Classical E4 transient_growth_bound={report.transient_growth_bound:.2f} "
            f"exceeds L7_TRANSIENT_GROWTH_TOL={L7_TRANSIENT_GROWTH_TOL}"
        )
        # Sanity: all fields populated and finite
        assert np.isfinite(report.numerical_abscissa)
        assert np.isfinite(report.henrici_departure)
        assert np.isfinite(report.kreiss_constant)
        assert report.n == 20 * 20  # (21-1)^2 = 400

    @pytest.mark.slow
    def test_gaussian_e4_eps_01_fails(self):
        """Gaussian E4 eps=0.1 (known_unstable) fails the combined check.

        This scheme has a positive spectral abscissa in the 2D BL operator,
        so it must fail at least on the spectral_abscissa criterion.
        """
        kv = _load_known_values()
        unstable = kv["E4_1"]["known_unstable"][0]
        assert unstable["kernel"] == "gaussian"
        eps = unstable["epsilon"]

        report = layer7_with_non_normality(
            "E4", "gaussian", {"epsilon": eps}, N=21,
        )
        # The known-unstable scheme must fail on at least one criterion
        fails_spectral = report.spectral_abscissa > L7_TOL
        fails_transient = report.transient_growth_bound > L7_TRANSIENT_GROWTH_TOL
        assert fails_spectral or fails_transient, (
            f"Gaussian eps={eps} should fail: "
            f"spectral_abscissa={report.spectral_abscissa:.6e}, "
            f"transient_growth_bound={report.transient_growth_bound:.2f}"
        )

    @pytest.mark.slow
    def test_report_fields_populated(self):
        """All NonNormalityReport fields are populated for a BL-sized operator."""
        report = layer7_with_non_normality(
            "E4", "tension", {"sigma": 3.0}, N=21,
        )
        assert np.isfinite(report.spectral_abscissa)
        assert np.isfinite(report.numerical_abscissa)
        assert np.isfinite(report.henrici_departure)
        assert report.henrici_departure >= 0.0
        assert np.isfinite(report.kreiss_constant)
        assert report.kreiss_constant >= 0.0
        assert np.isfinite(report.transient_growth_bound)
        assert report.compute_time > 0.0
        assert isinstance(report.pseudospectral_abscissae, dict)
        assert len(report.pseudospectral_abscissae) > 0
