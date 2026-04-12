"""Tests for stencil_gen.non_normality module."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from stencil_gen.non_normality import spectral_abscissa_sparse


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
