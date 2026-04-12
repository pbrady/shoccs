"""Tests for stencil_gen.non_normality module."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from stencil_gen.non_normality import (
    spectral_abscissa_sparse,
    numerical_abscissa_sparse,
    henrici_departure,
    eigenvector_condition,
    _sigma_field,
)


# ---------------------------------------------------------------------------
# TestSpectralAbscissa (41.8b)
# ---------------------------------------------------------------------------


class TestSpectralAbscissa:
    """Tests for spectral_abscissa_sparse."""

    def test_diagonal_negative(self):
        """Diagonal -diag(1..50) has spectral abscissa ≈ -1."""
        diag_vals = -np.arange(1, 51, dtype=float)
        L = sp.diags(diag_vals, format="csr")
        max_re, evals = spectral_abscissa_sparse(L, k=10)
        assert max_re == pytest.approx(-1.0, abs=1e-10)
        # All eigenvalues should have negative real part
        assert np.all(evals.real < 0)

    def test_random_sparse_returns_finite(self):
        """Random sparse matrix returns a finite spectral abscissa."""
        rng = np.random.default_rng(42)
        n = 100
        # Random sparse with density ~5%
        data = rng.standard_normal(500)
        rows = rng.integers(0, n, 500)
        cols = rng.integers(0, n, 500)
        L = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        max_re, evals = spectral_abscissa_sparse(L, k=10)
        assert np.isfinite(max_re)
        assert len(evals) > 0

    def test_dense_fallback_small_n(self):
        """Dense fallback path is exercised at N=20."""
        rng = np.random.default_rng(123)
        n = 20
        A = rng.standard_normal((n, n))
        # Make it stable: shift eigenvalues to left half-plane
        A = A - 3.0 * np.eye(n)

        # Pass as dense ndarray — should trigger dense fallback since n <= 900
        # and n <= k+1 (k=20 by default, so 20 <= 21)
        max_re, evals = spectral_abscissa_sparse(A, k=20)
        assert np.isfinite(max_re)
        # Should have all n eigenvalues from dense path
        assert len(evals) == n

        # Verify against direct numpy eigvals
        expected = np.max(np.linalg.eigvals(A).real)
        assert max_re == pytest.approx(expected, abs=1e-10)

    def test_identity_spectral_abscissa(self):
        """Identity matrix has spectral abscissa = 1."""
        L = sp.eye(30, format="csr")
        max_re, evals = spectral_abscissa_sparse(L, k=5)
        assert max_re == pytest.approx(1.0, abs=1e-10)

    def test_negative_identity(self):
        """-I has spectral abscissa = -1."""
        L = -sp.eye(30, format="csr")
        max_re, evals = spectral_abscissa_sparse(L, k=5)
        assert max_re == pytest.approx(-1.0, abs=1e-10)

    def test_known_eigenvalues_tridiagonal(self):
        """Tridiagonal matrix with known eigenvalues."""
        # Symmetric tridiagonal: -2 on diagonal, 1 on off-diagonals
        # Eigenvalues: -2 + 2*cos(k*pi/(n+1)) for k=1..n
        n = 50
        diag_main = -2.0 * np.ones(n)
        diag_off = np.ones(n - 1)
        L = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csr")

        max_re, evals = spectral_abscissa_sparse(L, k=10)

        # Largest eigenvalue is at k=1: -2 + 2*cos(pi/(n+1))
        expected_max = -2.0 + 2.0 * np.cos(np.pi / (n + 1))
        assert max_re == pytest.approx(expected_max, abs=1e-8)


# ---------------------------------------------------------------------------
# TestNormMetrics (41.8c)
# ---------------------------------------------------------------------------


class TestNormMetrics:
    """Tests for numerical_abscissa_sparse, henrici_departure, eigenvector_condition."""

    def test_diagonal_numerical_abscissa(self):
        """Diagonal -diag(1..50): numerical abscissa = spectral abscissa = -1."""
        diag_vals = -np.arange(1, 51, dtype=float)
        L = sp.diags(diag_vals, format="csr")
        na = numerical_abscissa_sparse(L)
        # For a real diagonal (hence symmetric) matrix, numerical abscissa
        # equals spectral abscissa: max eigenvalue of H = max eigenvalue of L.
        assert na == pytest.approx(-1.0, abs=1e-10)

    def test_diagonal_henrici_zero(self):
        """Diagonal matrix is normal: Henrici departure = 0."""
        diag_vals = -np.arange(1, 51, dtype=float)
        L = sp.diags(diag_vals, format="csr")
        h = henrici_departure(L)
        assert h == pytest.approx(0.0, abs=1e-12)

    def test_diagonal_eigenvector_condition_one(self):
        """Diagonal matrix has eigenvector condition number ≈ 1."""
        diag_vals = -np.arange(1, 51, dtype=float)
        L = sp.diags(diag_vals, format="csr")
        cond_v = eigenvector_condition(L)
        # Eigenvectors of a diagonal matrix are the identity columns,
        # so V = permutation of I, cond(V) = 1.
        assert cond_v == pytest.approx(1.0, abs=1e-8)

    def test_numerical_abscissa_dense_input(self):
        """numerical_abscissa_sparse works with dense ndarray input."""
        A = np.diag([-3.0, -2.0, -1.0])
        na = numerical_abscissa_sparse(A)
        assert na == pytest.approx(-1.0, abs=1e-10)

    def test_henrici_dense_input(self):
        """henrici_departure works with dense ndarray input."""
        A = np.diag([1.0, 2.0, 3.0])
        h = henrici_departure(A)
        assert h == pytest.approx(0.0, abs=1e-12)

    def test_eigenvector_condition_large_returns_nan(self):
        """eigenvector_condition returns NaN when N exceeds threshold."""
        L = sp.eye(1000, format="csr")
        cond_v = eigenvector_condition(L, small_dense_threshold=500)
        assert np.isnan(cond_v)

    def test_non_normal_matrix_positive_henrici(self):
        """A non-normal matrix (upper triangular shift) has Henrici > 0."""
        n = 30
        # Upper-shift matrix: L[i, i+1] = 1
        L = sp.diags([np.ones(n - 1)], [1], shape=(n, n), format="csr")
        h = henrici_departure(L)
        assert h > 0.0

    def test_non_normal_matrix_large_eigenvector_condition(self):
        """A non-normal matrix has cond(V) >> 1."""
        n = 30
        # Jordan-like: -I + nilpotent shift
        A = -np.eye(n) + n * np.diag(np.ones(n - 1), 1)
        cond_v = eigenvector_condition(A)
        assert cond_v > 10.0

    def test_numerical_abscissa_ge_spectral_abscissa(self):
        """Numerical abscissa >= spectral abscissa (fundamental inequality)."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((30, 30)) - 3.0 * np.eye(30)
        na = numerical_abscissa_sparse(A)
        sa, _ = spectral_abscissa_sparse(A)
        assert na >= sa - 1e-9

    def test_zero_matrix_henrici(self):
        """Zero matrix: henrici_departure returns 0 (guard against div-by-zero)."""
        L = sp.csr_matrix((10, 10))
        h = henrici_departure(L)
        assert h == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# TestSigmaField (41.8d)
# ---------------------------------------------------------------------------


class TestSigmaField:
    """Tests for _sigma_field: sigma_min(sI - L) over a complex grid."""

    def test_diagonal_matches_brute_force(self):
        """On a small diagonal matrix, compare _sigma_field to dense SVD."""
        diag_vals = np.array([-3.0, -2.0, -1.0, 0.5, 1.5])
        L = sp.diags(diag_vals, format="csr")
        n = L.shape[0]

        # Build a small grid
        re_vals = np.linspace(-1, 3, 5)
        im_vals = np.linspace(-2, 2, 5)
        s_grid = re_vals[:, None] + 1j * im_vals[None, :]

        result = _sigma_field(L, s_grid)
        assert result.shape == s_grid.shape

        # Brute-force reference
        for i in range(s_grid.shape[0]):
            for j in range(s_grid.shape[1]):
                s = s_grid[i, j]
                M = s * np.eye(n) - np.diag(diag_vals)
                sv_ref = np.linalg.svd(M, compute_uv=False)[-1]
                assert result[i, j] == pytest.approx(sv_ref, abs=1e-10)

    def test_dense_input(self):
        """_sigma_field works with a dense ndarray as L."""
        A = np.diag([-2.0, -1.0, 0.0])
        s_grid = np.array([0.0 + 0j, 1.0 + 0j, 0.0 + 1j])

        result = _sigma_field(A, s_grid)
        assert result.shape == s_grid.shape

        # Check each point via brute-force
        for idx, s in enumerate(s_grid):
            M = s * np.eye(3) - A
            sv_ref = np.linalg.svd(M, compute_uv=False)[-1]
            assert result[idx] == pytest.approx(sv_ref, abs=1e-10)

    def test_at_eigenvalue_sigma_min_near_zero(self):
        """sigma_min(sI - L) ≈ 0 when s is an eigenvalue of L."""
        diag_vals = np.array([-3.0, -1.0, 2.0])
        L = sp.diags(diag_vals, format="csr")

        # Evaluate exactly at each eigenvalue
        s_grid = np.array([-3.0 + 0j, -1.0 + 0j, 2.0 + 0j])
        result = _sigma_field(L, s_grid)

        for val in result:
            assert val == pytest.approx(0.0, abs=1e-10)

    def test_identity_sigma_min(self):
        """For L = I, sigma_min(sI - I) = |s - 1| (all singular values equal)."""
        n = 10
        L = sp.eye(n, format="csr")
        s_grid = np.array([0.5 + 0j, 1.0 + 0j, 2.0 + 1j, -1.0 + 0.5j])

        result = _sigma_field(L, s_grid)
        for idx, s in enumerate(s_grid):
            expected = abs(s - 1.0)
            assert result[idx] == pytest.approx(expected, abs=1e-10)

    def test_2d_grid_shape_preserved(self):
        """Output shape matches a 2D input grid shape."""
        L = sp.diags([-1.0, -2.0, -3.0], format="csr")
        s_grid = np.zeros((4, 7), dtype=complex)
        s_grid.real = np.linspace(-1, 1, 4)[:, None]
        s_grid.imag = np.linspace(-3, 3, 7)[None, :]

        result = _sigma_field(L, s_grid)
        assert result.shape == (4, 7)

    def test_sparse_medium_size(self):
        """Sparse path exercised for N > 200 (if we make L sparse and large)."""
        n = 250
        # Stable tridiagonal
        diag_main = -2.0 * np.ones(n)
        diag_off = np.ones(n - 1)
        L = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csc")

        # Small grid — just a few points to keep test fast
        s_grid = np.array([0.0 + 0j, 0.0 + 1j, -1.0 + 0j])
        result = _sigma_field(L, s_grid)

        # Compare to dense
        L_dense = L.toarray()
        for idx, s in enumerate(s_grid):
            M = s * np.eye(n) - L_dense
            sv_ref = np.linalg.svd(M, compute_uv=False)[-1]
            assert result[idx] == pytest.approx(sv_ref, abs=1e-6)
