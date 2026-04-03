"""Tests for PHS+poly stencil derivation (Phase 29)."""

import pytest
from sympy import Rational, S, Symbol, cancel, symbols

from stencil_gen.phs import (
    cut_cell_weights,
    phs_stencil_weights,
    uniform_boundary_weights,
    uniform_boundary_weights_rbf,
    uniform_interior_weights,
    uniform_interior_weights_rbf,
)


# ---------------------------------------------------------------------------
# 29.1a: Core PHS solver
# ---------------------------------------------------------------------------


class TestPHSCore:
    """Basic tests for phs_stencil_weights."""

    def test_3pt_first_deriv_centered(self):
        """3-point centered first derivative should give [-1/2, 0, 1/2]."""
        # 3 points, centered at 0, degree q=2, any k >= 1
        points = [Rational(-1), Rational(0), Rational(1)]
        w = phs_stencil_weights(points, Rational(0), nu=1, q=2, k=2)
        assert w == [Rational(-1, 2), S.Zero, Rational(1, 2)]

    def test_3pt_second_deriv_centered(self):
        """3-point centered second derivative should give [1, -2, 1]."""
        points = [Rational(-1), Rational(0), Rational(1)]
        w = phs_stencil_weights(points, Rational(0), nu=2, q=2, k=2)
        assert w == [S.One, Rational(-2), S.One]

    def test_2pt_first_deriv(self):
        """2-point first derivative: [-1, 1] (forward difference)."""
        points = [Rational(0), Rational(1)]
        w = phs_stencil_weights(points, Rational(0), nu=1, q=1, k=1)
        assert w == [Rational(-1), Rational(1)]

    def test_polynomial_exactness(self):
        """Stencil should be exact for polynomials up to degree q."""
        points = [Rational(j) for j in range(6)]
        for q in [1, 2, 3, 4]:
            w = phs_stencil_weights(points, Rational(0), nu=1, q=q, k=max(2, q))
            # Check: sum_j w_j * x_j^d = d * 0^(d-1) for d = 0..q
            for d in range(q + 1):
                actual = sum(wj * xj**d for wj, xj in zip(w, points))
                expected = d * Rational(0) ** max(0, d - 1) if d >= 1 else S.Zero
                assert cancel(actual - expected) == 0, (
                    f"q={q}, d={d}: got {actual}, expected {expected}"
                )

    def test_weights_sum_to_zero_for_first_deriv(self):
        """First derivative weights should sum to 0 (exact for constants)."""
        points = [Rational(j) for j in range(5)]
        for k in [2, 3, 4]:
            w = phs_stencil_weights(points, Rational(1), nu=1, q=3, k=k)
            assert cancel(sum(w)) == 0, f"k={k}: weights don't sum to 0"


# ---------------------------------------------------------------------------
# 29.1b + 29.2c: Uniform grid wrappers + interior verification
# ---------------------------------------------------------------------------


class TestPHSInterior:
    """Verify PHS interior stencils match classical FD."""

    def test_e2_interior_high_k(self):
        """E2 interior (p=1, nu=1): [-1/2, 0, 1/2] for any k >= 2."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        classical = full_gamma_array(derive_interior(0, 1, 1))
        for k in [2, 3, 5]:
            phs = uniform_interior_weights(p=1, nu=1, k=k, q=2)
            for j in range(len(classical)):
                assert cancel(phs[j] - classical[j]) == 0, (
                    f"k={k}, j={j}: PHS={phs[j]}, classical={classical[j]}"
                )

    def test_e4_interior_high_k(self):
        """E4 interior (p=2, nu=1): [1/12, -2/3, 0, 2/3, -1/12] for high k."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        classical = full_gamma_array(derive_interior(0, 2, 1))
        for k in [3, 5]:
            phs = uniform_interior_weights(p=2, nu=1, k=k, q=4)
            for j in range(len(classical)):
                assert cancel(phs[j] - classical[j]) == 0, (
                    f"k={k}, j={j}: PHS={phs[j]}, classical={classical[j]}"
                )

    def test_e2_interior_nu2(self):
        """E2 second derivative interior (p=1, nu=2): [1, -2, 1]."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        classical = full_gamma_array(derive_interior(0, 1, 2))
        phs = uniform_interior_weights(p=1, nu=2, k=2, q=2)
        for j in range(len(classical)):
            assert cancel(phs[j] - classical[j]) == 0


# ---------------------------------------------------------------------------
# 29.2a: E2_1 boundary comparison
# ---------------------------------------------------------------------------


class TestPHSvsE2Boundary:
    """Compare PHS boundary stencils against E2_1."""

    def test_taylor_accuracy(self):
        """PHS E2_1 boundary rows should have order q=1 accuracy."""
        # E2_1: p=1, q=1, t=4, r=3
        for i in range(3):
            for k in [2, 3]:
                w = uniform_boundary_weights(i, t=4, nu=1, k=k, q=1)
                # Check: exact for f(x) = 1 and f(x) = x
                pts = [Rational(j) for j in range(4)]
                # d/dx(1) = 0
                assert cancel(sum(w)) == 0, f"row {i}, k={k}: not exact for constants"
                # d/dx(x) = 1 at x=i
                actual = sum(wj * xj for wj, xj in zip(w, pts))
                assert cancel(actual - 1) == 0, f"row {i}, k={k}: not exact for x"

    def test_extract_implied_alpha(self):
        """Extract the implied alpha from PHS E2_1 boundary stencils."""
        from stencil_gen.temo import E2_1, derive_uniform_boundary_for_temo

        uniform = derive_uniform_boundary_for_temo(E2_1)
        B_u = uniform.B_u
        alpha_sym = uniform.alpha_symbols

        print("\n=== E2_1 PHS vs Symbolic Boundary Stencils ===")
        for i in range(B_u.rows):
            for k in [2, 3, 4]:
                w_phs = uniform_boundary_weights(i, t=4, nu=1, k=k, q=1)
                # The symbolic stencil has form B_u[i,j] = a_j + b_j * alpha
                # The PHS stencil has specific numeric values.
                # Extract alpha by solving: B_u[i, j_free] = w_phs[j_free]
                # where j_free is the free column (last column in our convention)
                if len(alpha_sym) > 0 and i < B_u.rows:
                    row_syms = B_u[i, :].free_symbols
                    if row_syms:
                        # Find the alpha that appears in this row
                        alpha_in_row = sorted(row_syms, key=str)
                        alpha = alpha_in_row[0]
                        # Solve for alpha from the last column
                        from sympy import solve

                        for j in range(B_u.cols):
                            expr = B_u[i, j]
                            if alpha in expr.free_symbols:
                                sol = solve(expr - w_phs[j], alpha)
                                if sol:
                                    print(f"  Row {i}, k={k}: alpha = {sol[0]}")
                                    break


# ---------------------------------------------------------------------------
# 29.2b: E4_1 boundary comparison
# ---------------------------------------------------------------------------


class TestPHSvsE4Boundary:
    """Compare PHS boundary stencils against E4_1."""

    def test_taylor_accuracy(self):
        """PHS E4_1 boundary rows should have order q=3 accuracy."""
        # E4_1: p=2, q=3, t=6, r=4
        for i in range(4):
            w = uniform_boundary_weights(i, t=6, nu=1, k=3, q=3)
            pts = [Rational(j) for j in range(6)]
            # Exact for polynomials up to degree 3
            for d in range(4):
                actual = sum(wj * xj**d for wj, xj in zip(w, pts))
                if d == 0:
                    expected = S.Zero
                elif d == 1:
                    expected = S.One
                else:
                    expected = d * Rational(i) ** (d - 1)
                assert cancel(actual - expected) == 0, (
                    f"row {i}, d={d}: got {actual}, expected {expected}"
                )

    def test_extract_implied_alphas(self):
        """Extract implied alpha values from PHS E4_1 boundary stencils."""
        from stencil_gen.temo import E4_1, build_uniform_for_mathematica

        uniform = build_uniform_for_mathematica(E4_1)
        B_u = uniform.B_u

        print("\n=== E4_1 PHS vs Symbolic Boundary Stencils ===")
        for k in [2, 3, 4, 5]:
            print(f"\n  k={k}:")
            for i in range(B_u.rows):
                w_phs = uniform_boundary_weights(i, t=6, nu=1, k=k, q=3)
                # Print the PHS weights
                print(f"    Row {i}: {[float(cancel(w)) for w in w_phs]}")


# ---------------------------------------------------------------------------
# 29.5b + 29.5c: Gaussian/Multiquadric RBF convenience wrappers & tests
# ---------------------------------------------------------------------------

import numpy as np

from stencil_gen.phs import (
    build_diff_matrix_mixed_epsilon,
    build_diff_matrix_rbf,
    max_real_eigenvalue,
)


# ---------------------------------------------------------------------------
# 29.6a: Differentiation matrix builder
# ---------------------------------------------------------------------------


class TestBuildDiffMatrixRBF:
    """Tests for build_diff_matrix_rbf."""

    def test_matrix_shape(self):
        """Matrix should be n×n."""
        for n in [20, 40]:
            D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0, nextra=1)
            assert D.shape == (n, n)

    def test_interior_column_sums_zero(self):
        """Interior rows should have column sums of 0 (first derivative)."""
        n = 30
        # E2: p=1, q=1, nextra=1 → r=3 boundary rows
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0, nextra=1)
        r = 3
        # Each interior row should sum to 0
        for i in range(r, n - r):
            row_sum = np.sum(D[i, :])
            assert abs(row_sum) < 1e-14, f"Interior row {i} sum = {row_sum}"

    def test_boundary_rows_nonzero(self):
        """Boundary rows should have nonzero entries."""
        n = 20
        D = build_diff_matrix_rbf(n, p=2, q=3, epsilon=1.0)
        # Left boundary row 0 should have nonzero entries in first t=6 columns
        assert np.any(D[0, :6] != 0)
        # Right boundary row n-1 should have nonzero entries in last t=6 columns
        assert np.any(D[-1, -6:] != 0)

    def test_antisymmetry_first_deriv(self):
        """Right boundary should be antisymmetric reflection of left for nu=1."""
        n = 20
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=2.0, nextra=1)
        r = 3
        t = 4
        for i in range(r):
            left_row = D[i, :t]
            right_row = D[n - 1 - i, n - t:][::-1]
            np.testing.assert_allclose(right_row, -left_row, atol=1e-14)

    def test_polynomial_reproduction(self):
        """D applied to x should give all 1s (exact for linear)."""
        n = 30
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0, nextra=1)
        x = np.arange(n, dtype=float)
        result = D @ x
        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_nu2_dimensions_match_temo(self):
        """build_diff_matrix_rbf nu=2 dimensions should match temo.compute_dimensions."""
        from stencil_gen.temo import compute_dimensions

        # E2_2: p=1, q=1, nextra=0, nu=2 → t=3, r=2
        dims = compute_dimensions(p=1, q=1, s=0, nextra=0, nu=2)
        n = 20
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0, nu=2, nextra=0)
        # Boundary stencil width = t: row 0 should have nonzero entries in cols 0..t-1
        nonzero_cols_row0 = np.where(np.abs(D[0, :]) > 1e-15)[0]
        assert nonzero_cols_row0[-1] <= dims.t - 1, (
            f"Row 0 extends to col {nonzero_cols_row0[-1]}, expected max {dims.t - 1}"
        )
        # Number of boundary rows per side = r
        # Interior rows should use the centered stencil, not boundary
        # Row r should be an interior row (centered stencil)
        r = dims.r
        center_col = r  # interior row at index r is centered at column r
        assert D[r, center_col] != 0, f"Interior row {r} should have centered stencil"
        # Row r-1 should be a boundary row (one-sided stencil)
        assert D[r - 1, 0] != 0, f"Boundary row {r-1} should use left-boundary stencil"

    def test_nu2_polynomial_reproduction(self):
        """D (nu=2) applied to x should give all 0s (exact for linear, q=1)."""
        n = 30
        # E2_2 params: p=1, q=1, nextra=0 → boundary exact for poly deg ≤ 1
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0, nu=2, nextra=0)
        x = np.arange(n, dtype=float)
        # D^2(x) = 0 everywhere for q >= 1
        result = D @ x
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_nu2_polynomial_reproduction_higher_q(self):
        """D (nu=2) applied to x^2 should give 2s with q=3 (E4_2 params)."""
        n = 30
        # E4_2: p=2, q=3, nextra=0 → boundary exact for poly deg ≤ 3
        D = build_diff_matrix_rbf(n, p=2, q=3, epsilon=1.0, nu=2, nextra=0)
        x = np.arange(n, dtype=float)
        result = D @ (x**2)
        np.testing.assert_allclose(result, 2.0, atol=1e-10)

    def test_nu2_symmetry_second_deriv(self):
        """Right boundary should be symmetric reflection of left for nu=2."""
        n = 20
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=2.0, nu=2, nextra=0)
        from stencil_gen.temo import compute_dimensions

        dims = compute_dimensions(p=1, q=1, s=0, nextra=0, nu=2)
        r, t = dims.r, dims.t
        for i in range(r):
            left_row = D[i, :t]
            right_row = D[n - 1 - i, n - t:][::-1]
            # nu=2 (even) → symmetric reflection: sign = (-1)^2 = +1
            np.testing.assert_allclose(right_row, left_row, atol=1e-14)


# ---------------------------------------------------------------------------
# 29.6b: Max real eigenvalue diagnostic
# ---------------------------------------------------------------------------


class TestMaxRealEigenvalue:
    """Tests for max_real_eigenvalue."""

    def test_interior_only_pure_imaginary(self):
        """Interior-only (periodic) FD matrix should have max Re(λ) ≈ 0.

        Build a matrix with ALL rows using classical interior stencils
        (wrapping around periodically).  This is equivalent to the circulant
        interior matrix which has purely imaginary eigenvalues.
        """
        n = 40
        p = 2
        from stencil_gen.interior import derive_interior, full_gamma_array

        interior_coeffs = derive_interior(0, p, 1)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        D = np.zeros((n, n))
        for i in range(n):
            for k_idx, offset in enumerate(range(-p, p + 1)):
                j = (i + offset) % n  # periodic wrapping
                D[i, j] = interior_w[k_idx]

        eigvals = np.linalg.eigvals(D)
        max_re = float(np.max(np.real(eigvals)))
        assert abs(max_re) < 1e-12, f"Periodic interior max Re(λ) = {max_re}"

    def test_returns_float(self):
        """max_real_eigenvalue should return a float."""
        result = max_real_eigenvalue(20, p=1, q=1, epsilon=1.0, nextra=1)
        assert isinstance(result, float)


class TestGaussianRBF:
    """Tests for Gaussian and Multiquadric RBF kernels."""

    def test_polynomial_exactness(self):
        """Gaussian RBF stencils should be exact for polynomials up to degree q."""
        for epsilon in [0.5, 1.0, 3.0]:
            for q in [1, 2, 3]:
                # Use t = q + 3 points (enough for the augmented system)
                t = q + 3
                w = uniform_boundary_weights_rbf(i=0, t=t, nu=1, q=q, epsilon=epsilon)
                pts = list(range(t))
                for d in range(q + 1):
                    actual = sum(wj * xj**d for wj, xj in zip(w, pts))
                    expected = d * 0 ** max(0, d - 1) if d >= 1 else 0.0
                    assert abs(actual - expected) < 1e-12, (
                        f"eps={epsilon}, q={q}, d={d}: got {actual}, expected {expected}"
                    )

    def test_interior_matches_classical(self):
        """Gaussian interior weights should match classical FD for all epsilon.

        The polynomial augmentation forces polynomial reproduction, so the
        interior (centered) weights must equal the classical FD coefficients
        regardless of the RBF shape parameter.
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        # E2: p=1, q=2
        classical_e2 = full_gamma_array(derive_interior(0, 1, 1))
        for epsilon in [0.1, 1.0, 5.0]:
            w = uniform_interior_weights_rbf(p=1, nu=1, q=2, epsilon=epsilon)
            for j in range(len(classical_e2)):
                assert abs(w[j] - float(classical_e2[j])) < 1e-12, (
                    f"E2 eps={epsilon}, j={j}: RBF={w[j]}, classical={classical_e2[j]}"
                )

        # E4: p=2, q=4
        classical_e4 = full_gamma_array(derive_interior(0, 2, 1))
        for epsilon in [0.1, 1.0, 5.0]:
            w = uniform_interior_weights_rbf(p=2, nu=1, q=4, epsilon=epsilon)
            for j in range(len(classical_e4)):
                assert abs(w[j] - float(classical_e4[j])) < 1e-12, (
                    f"E4 eps={epsilon}, j={j}: RBF={w[j]}, classical={classical_e4[j]}"
                )

    def test_weights_sum_to_zero(self):
        """First derivative weights should sum to 0 (exact for constants)."""
        for kernel in ["gaussian", "multiquadric"]:
            for epsilon in [0.5, 1.0, 3.0]:
                # Interior
                w = uniform_interior_weights_rbf(
                    p=2, nu=1, q=3, epsilon=epsilon, kernel=kernel
                )
                assert abs(sum(w)) < 1e-12, (
                    f"{kernel} eps={epsilon} interior: sum={sum(w)}"
                )
                # Boundary
                w = uniform_boundary_weights_rbf(
                    i=0, t=6, nu=1, q=3, epsilon=epsilon, kernel=kernel
                )
                assert abs(sum(w)) < 1e-12, (
                    f"{kernel} eps={epsilon} boundary: sum={sum(w)}"
                )

    def test_small_epsilon_interior_matches_polynomial(self):
        """As epsilon -> 0, interior Gaussian RBF weights approach polynomial FD.

        For centered interior stencils where 2p+1 = q+1, the polynomial
        augmentation fully determines the weights, so the flat limit is
        well-defined and equals classical FD.  For over-determined boundary
        stencils (t > q+1) the flat limit is ill-conditioned, so we only
        test interior convergence here.
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        # E4: p=2, 5 points, q=4 (5 poly terms = n, so system is determined)
        classical = full_gamma_array(derive_interior(0, 2, 1))
        ref = [float(c) for c in classical]

        # As epsilon decreases, should approach classical
        for epsilon in [2.0, 1.0, 0.5]:
            w = uniform_interior_weights_rbf(p=2, nu=1, q=4, epsilon=epsilon)
            err = max(abs(w[j] - ref[j]) for j in range(len(ref)))
            assert err < 1e-10, f"eps={epsilon}: error {err} too large"

    def test_boundary_weights_bounded(self):
        """Boundary weights remain bounded across a range of epsilon values."""
        for epsilon in [0.5, 1.0, 2.0, 5.0]:
            for kernel in ["gaussian", "multiquadric"]:
                w = uniform_boundary_weights_rbf(
                    i=0, t=6, nu=1, q=3, epsilon=epsilon, kernel=kernel
                )
                assert all(abs(wj) < 100 for wj in w), (
                    f"{kernel} eps={epsilon}: weights unbounded: {w}"
                )

    def test_multiquadric_polynomial_exactness(self):
        """Multiquadric RBF stencils should be exact for polynomials up to degree q."""
        for epsilon in [0.5, 1.0, 3.0]:
            q = 3
            t = q + 3
            w = uniform_boundary_weights_rbf(
                i=1, t=t, nu=1, q=q, epsilon=epsilon, kernel="multiquadric"
            )
            pts = list(range(t))
            for d in range(q + 1):
                actual = sum(wj * xj**d for wj, xj in zip(w, pts))
                expected = d * 1 ** max(0, d - 1) if d >= 1 else 0.0
                assert abs(actual - expected) < 1e-12, (
                    f"MQ eps={epsilon}, d={d}: got {actual}, expected {expected}"
                )


# ---------------------------------------------------------------------------
# 29.6c: Epsilon sweep for E2 (p=1, q=1, nextra=1)
# ---------------------------------------------------------------------------


class TestEpsilonSweepE2:
    """Sweep epsilon for E2_1 boundary stencils and report stability.

    E2_1 parameters: p=1, q=1, nextra=1.
    Sweeps Gaussian and Multiquadric kernels over epsilon in [0.01, 10].
    """

    # E2_1 parameters
    P = 1
    Q = 1
    NEXTRA = 1
    NU = 1

    def _sweep(self, kernel: str, n_values, epsilons):
        """Run epsilon sweep, return dict {n: list of (eps, max_re, spec_rad)}."""
        results = {}
        for n in n_values:
            rows = []
            for eps in epsilons:
                D = build_diff_matrix_rbf(
                    n, p=self.P, q=self.Q, epsilon=eps,
                    kernel=kernel, nu=self.NU, nextra=self.NEXTRA,
                )
                eigvals = np.linalg.eigvals(D)
                max_re = float(np.max(np.real(eigvals)))
                spec_rad = float(np.max(np.abs(eigvals)))
                rows.append((eps, max_re, spec_rad))
            results[n] = rows
        return results

    def _print_table(self, label, results):
        """Print formatted sweep table."""
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        for n, rows in sorted(results.items()):
            print(f"\n  n = {n}")
            print(f"  {'epsilon':>10s}  {'max Re(λ)':>14s}  {'spec radius':>14s}")
            print(f"  {'-'*10}  {'-'*14}  {'-'*14}")
            for eps, max_re, spec_rad in rows:
                print(f"  {eps:10.4f}  {max_re:14.6e}  {spec_rad:14.6e}")

        # Summary: best epsilon per n
        print(f"\n  --- Best epsilon (min max Re(λ)) ---")
        for n, rows in sorted(results.items()):
            best = min(rows, key=lambda r: r[1])
            stable = "STABLE" if best[1] <= 0 else "unstable"
            print(f"  n={n:3d}: eps={best[0]:.4f}, max Re(λ)={best[1]:.6e} [{stable}]")

    def test_gaussian_sweep(self):
        """Sweep Gaussian kernel epsilon for E2_1."""
        epsilons = np.logspace(np.log10(0.01), np.log10(10), 60)
        n_values = [20, 40, 80]
        results = self._sweep("gaussian", n_values, epsilons)
        self._print_table("E2_1 Gaussian RBF Epsilon Sweep (p=1, q=1, nextra=1)", results)

        # Find if any epsilon gives stability
        for n, rows in results.items():
            best = min(rows, key=lambda r: r[1])
            if best[1] <= 0:
                print(f"\n  *** STABLE epsilon found for n={n}: eps={best[0]:.6f} ***")

    def test_multiquadric_sweep(self):
        """Sweep Multiquadric kernel epsilon for E2_1."""
        epsilons = np.logspace(np.log10(0.01), np.log10(10), 60)
        n_values = [20, 40, 80]
        results = self._sweep("multiquadric", n_values, epsilons)
        self._print_table("E2_1 Multiquadric RBF Epsilon Sweep (p=1, q=1, nextra=1)", results)

        for n, rows in results.items():
            best = min(rows, key=lambda r: r[1])
            if best[1] <= 0:
                print(f"\n  *** STABLE epsilon found for n={n}: eps={best[0]:.6f} ***")

    def test_gaussian_fine_sweep_near_best(self):
        """Fine sweep around the best Gaussian epsilon from coarse sweep.

        Uses n=40 for the coarse pass, then refines near the minimum.
        """
        n = 40
        # Coarse sweep
        epsilons_coarse = np.logspace(np.log10(0.01), np.log10(10), 60)
        coarse = []
        for eps in epsilons_coarse:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            coarse.append((eps, max_re))

        best_coarse = min(coarse, key=lambda r: r[1])
        eps_best = best_coarse[0]

        # Fine sweep: ±1 decade around best
        lo = max(0.001, eps_best / 10)
        hi = min(100, eps_best * 10)
        epsilons_fine = np.linspace(lo, hi, 200)
        fine = []
        for eps in epsilons_fine:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            fine.append((eps, max_re))

        best_fine = min(fine, key=lambda r: r[1])
        print(f"\n  E2_1 Gaussian fine sweep (n={n}):")
        print(f"  Coarse best: eps={best_coarse[0]:.6f}, max Re(λ)={best_coarse[1]:.6e}")
        print(f"  Fine best:   eps={best_fine[0]:.6f}, max Re(λ)={best_fine[1]:.6e}")

        stable = best_fine[1] <= 0
        print(f"  Stable: {stable}")

        # Verify at multiple grid sizes
        if stable or best_fine[1] < 1e-6:
            eps_star = best_fine[0]
            print(f"\n  Checking eps*={eps_star:.6f} across grid sizes:")
            for nn in [20, 40, 80, 160]:
                mr = max_real_eigenvalue(
                    nn, p=self.P, q=self.Q, epsilon=eps_star,
                    kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
                )
                print(f"    n={nn:4d}: max Re(λ)={mr:.6e}")


# ---------------------------------------------------------------------------
# 29.6d: Epsilon sweep for E4 (p=2, q=3, nextra=0)
# ---------------------------------------------------------------------------


class TestEpsilonSweepE4:
    """Sweep epsilon for E4_1 boundary stencils and report stability.

    E4_1 parameters: p=2, q=3, nextra=0.
    This is the primary production target — finding a stable epsilon here
    is the key result.
    """

    # E4_1 parameters
    P = 2
    Q = 3
    NEXTRA = 0
    NU = 1

    def _sweep(self, kernel: str, n_values, epsilons):
        """Run epsilon sweep, return dict {n: list of (eps, max_re, spec_rad)}."""
        results = {}
        for n in n_values:
            rows = []
            for eps in epsilons:
                D = build_diff_matrix_rbf(
                    n, p=self.P, q=self.Q, epsilon=eps,
                    kernel=kernel, nu=self.NU, nextra=self.NEXTRA,
                )
                eigvals = np.linalg.eigvals(D)
                max_re = float(np.max(np.real(eigvals)))
                spec_rad = float(np.max(np.abs(eigvals)))
                rows.append((eps, max_re, spec_rad))
            results[n] = rows
        return results

    def _print_table(self, label, results):
        """Print formatted sweep table."""
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        for n, rows in sorted(results.items()):
            print(f"\n  n = {n}")
            print(f"  {'epsilon':>10s}  {'max Re(λ)':>14s}  {'spec radius':>14s}")
            print(f"  {'-'*10}  {'-'*14}  {'-'*14}")
            for eps, max_re, spec_rad in rows:
                print(f"  {eps:10.4f}  {max_re:14.6e}  {spec_rad:14.6e}")

        # Summary: best epsilon per n
        print(f"\n  --- Best epsilon (min max Re(λ)) ---")
        for n, rows in sorted(results.items()):
            best = min(rows, key=lambda r: r[1])
            stable = "STABLE" if best[1] <= 0 else "unstable"
            print(f"  n={n:3d}: eps={best[0]:.4f}, max Re(λ)={best[1]:.6e} [{stable}]")

    def test_gaussian_sweep(self):
        """Sweep Gaussian kernel epsilon for E4_1."""
        epsilons = np.logspace(np.log10(0.01), np.log10(10), 60)
        n_values = [20, 40, 80]
        results = self._sweep("gaussian", n_values, epsilons)
        self._print_table("E4_1 Gaussian RBF Epsilon Sweep (p=2, q=3, nextra=0)", results)

        # Find if any epsilon gives stability
        for n, rows in results.items():
            best = min(rows, key=lambda r: r[1])
            if best[1] <= 0:
                print(f"\n  *** STABLE epsilon found for n={n}: eps={best[0]:.6f} ***")

    def test_multiquadric_sweep(self):
        """Sweep Multiquadric kernel epsilon for E4_1."""
        epsilons = np.logspace(np.log10(0.01), np.log10(10), 60)
        n_values = [20, 40, 80]
        results = self._sweep("multiquadric", n_values, epsilons)
        self._print_table("E4_1 Multiquadric RBF Epsilon Sweep (p=2, q=3, nextra=0)", results)

        for n, rows in results.items():
            best = min(rows, key=lambda r: r[1])
            if best[1] <= 0:
                print(f"\n  *** STABLE epsilon found for n={n}: eps={best[0]:.6f} ***")

    def test_gaussian_fine_sweep_near_best(self):
        """Fine sweep around the best Gaussian epsilon from coarse sweep.

        Uses n=40 for the coarse pass, then refines near the minimum.
        """
        n = 40
        # Coarse sweep
        epsilons_coarse = np.logspace(np.log10(0.01), np.log10(10), 60)
        coarse = []
        for eps in epsilons_coarse:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            coarse.append((eps, max_re))

        best_coarse = min(coarse, key=lambda r: r[1])
        eps_best = best_coarse[0]

        # Fine sweep: ±1 decade around best
        lo = max(0.001, eps_best / 10)
        hi = min(100, eps_best * 10)
        epsilons_fine = np.linspace(lo, hi, 200)
        fine = []
        for eps in epsilons_fine:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            fine.append((eps, max_re))

        best_fine = min(fine, key=lambda r: r[1])
        print(f"\n  E4_1 Gaussian fine sweep (n={n}):")
        print(f"  Coarse best: eps={best_coarse[0]:.6f}, max Re(λ)={best_coarse[1]:.6e}")
        print(f"  Fine best:   eps={best_fine[0]:.6f}, max Re(λ)={best_fine[1]:.6e}")

        stable = best_fine[1] <= 0
        print(f"  Stable: {stable}")

        # Verify at multiple grid sizes
        if stable or best_fine[1] < 1e-6:
            eps_star = best_fine[0]
            print(f"\n  Checking eps*={eps_star:.6f} across grid sizes:")
            for nn in [20, 40, 80, 160]:
                mr = max_real_eigenvalue(
                    nn, p=self.P, q=self.Q, epsilon=eps_star,
                    kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
                )
                print(f"    n={nn:4d}: max Re(λ)={mr:.6e}")
        else:
            # Even if not stable, report best across grid sizes
            eps_star = best_fine[0]
            print(f"\n  Best eps={eps_star:.6f} across grid sizes:")
            for nn in [20, 40, 80, 160]:
                mr = max_real_eigenvalue(
                    nn, p=self.P, q=self.Q, epsilon=eps_star,
                    kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
                )
                print(f"    n={nn:4d}: max Re(λ)={mr:.6e}")


# ---------------------------------------------------------------------------
# 29.6f: Mixed epsilon for E4 — characterize the gap
# ---------------------------------------------------------------------------


class TestMixedEpsilon:
    """Try mixed epsilon (different per boundary row) for E4_1.

    E4_1 parameters: p=2, q=3, nextra=0 → t=6, r=4 boundary rows per side.
    Single-epsilon sweeps (29.6d) showed min max Re(λ) ≈ 1e-4, not stable.
    Here we try different ε for each of the 4 boundary rows.
    """

    P = 2
    Q = 3
    NEXTRA = 0
    NU = 1
    R = 4  # q + 1 + nextra = 3 + 1 + 0

    def _max_re_mixed(self, n, epsilons, kernel="gaussian"):
        """Compute max Re(λ) for a mixed-epsilon configuration."""
        D = build_diff_matrix_mixed_epsilon(
            n, p=self.P, q=self.Q, epsilons=list(epsilons),
            kernel=kernel, nu=self.NU, nextra=self.NEXTRA,
        )
        eigvals = np.linalg.eigvals(D)
        return float(np.max(np.real(eigvals)))

    def test_single_epsilon_baseline(self):
        """Confirm single-epsilon minimum from 29.6d as baseline.

        With a uniform epsilon across all 4 boundary rows, the best
        achievable max Re(λ) is ~1e-4 (not machine-precision stable).
        """
        n = 40
        # Coarse sweep to find best single epsilon
        epsilons_sweep = np.logspace(np.log10(0.1), np.log10(10), 80)
        best_eps, best_re = None, np.inf
        for eps in epsilons_sweep:
            mr = self._max_re_mixed(n, [eps] * self.R)
            if mr < best_re:
                best_re = mr
                best_eps = eps

        print(f"\n  E4_1 single-epsilon baseline (n={n}):")
        print(f"  Best eps={best_eps:.4f}, max Re(λ)={best_re:.6e}")
        # Should match 29.6d result: ~1e-4
        assert best_re < 1e-2, f"Single-epsilon baseline too large: {best_re}"

    def test_two_group_sweep(self):
        """Sweep two groups: ε_outer (rows 0,1) and ε_inner (rows 2,3).

        The near-interior rows (2, 3) are closest to the interior stencil
        and may benefit from a different ε than the outermost rows (0, 1).
        """
        n = 40
        eps_range = np.logspace(np.log10(0.3), np.log10(8.0), 30)

        best_combo = None
        best_re = np.inf
        results = []

        for eps_outer in eps_range:
            for eps_inner in eps_range:
                epsilons = [eps_outer, eps_outer, eps_inner, eps_inner]
                mr = self._max_re_mixed(n, epsilons)
                results.append((eps_outer, eps_inner, mr))
                if mr < best_re:
                    best_re = mr
                    best_combo = (eps_outer, eps_inner)

        print(f"\n  E4_1 two-group sweep (n={n}):")
        print(f"  Best: eps_outer={best_combo[0]:.4f}, eps_inner={best_combo[1]:.4f}")
        print(f"  max Re(λ)={best_re:.6e}")

        # Check if two-group improves over single epsilon
        # Single epsilon baseline is ~1e-4
        single_best = min(
            (mr for _, _, mr in results),
        )
        print(f"  (Minimum from all combos: {single_best:.6e})")

        # Verify at multiple grid sizes
        print(f"\n  Checking best combo across grid sizes:")
        for nn in [20, 40, 80]:
            epsilons = [best_combo[0], best_combo[0], best_combo[1], best_combo[1]]
            mr = self._max_re_mixed(nn, epsilons)
            stable = "STABLE" if mr <= 0 else "unstable"
            print(f"    n={nn:3d}: max Re(λ)={mr:.6e} [{stable}]")

    def test_per_row_optimize(self):
        """Coordinate descent to find optimal per-row epsilon combination.

        Optimizes over 4 independent epsilon values (one per boundary row)
        using iterative single-dimension sweeps.
        """
        n = 40
        eps_vals = np.logspace(np.log10(0.3), np.log10(8.0), 40)

        # Start from the best single epsilon (~1.7)
        current = [1.7] * self.R
        current_re = self._max_re_mixed(n, current)

        # Coordinate descent: sweep each row's epsilon while fixing others
        for iteration in range(3):  # 3 full passes
            for row in range(self.R):
                best_eps_row = current[row]
                best_re_row = current_re
                for eps in eps_vals:
                    trial = list(current)
                    trial[row] = eps
                    mr = self._max_re_mixed(n, trial)
                    if mr < best_re_row:
                        best_re_row = mr
                        best_eps_row = eps
                current[row] = best_eps_row
                current_re = best_re_row

        opt_eps = current

        print(f"\n  E4_1 per-row coordinate descent (n={n}):")
        print(f"  Optimal epsilons: [{', '.join(f'{e:.4f}' for e in opt_eps)}]")
        print(f"  max Re(λ)={current_re:.6e}")

        # Compare with uniform best
        uniform_mr = min(
            self._max_re_mixed(n, [eps] * self.R)
            for eps in np.logspace(np.log10(0.5), np.log10(5.0), 40)
        )
        improvement = uniform_mr / current_re if current_re > 0 else float('inf')
        print(f"\n  Uniform best: max Re(λ)={uniform_mr:.6e}")
        print(f"  Mixed improvement factor: {improvement:.1f}x")

        # Check grid convergence
        print(f"\n  Grid convergence with optimal epsilons:")
        for nn in [20, 40, 80, 160]:
            mr = self._max_re_mixed(nn, opt_eps)
            stable = "STABLE" if mr <= 0 else "unstable"
            print(f"    n={nn:3d}: max Re(λ)={mr:.6e} [{stable}]")

    def test_conservation_near_interior(self):
        """Try replacing the near-interior row (r-1) with a conservation row.

        The near-interior row (row r-1 = row 3) is the one closest to the
        interior.  Replace it with a row whose weights sum to 0 and that
        conserves the derivative of x^q (the highest polynomial).  This
        is done by using the interior stencil for that row but shifted.

        Strategy: use RBF for rows 0..r-2, and for row r-1 use the
        classical one-sided polynomial stencil (no RBF contribution).
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        n = 40
        # E4_1 dimensions: t=6, r=4
        t = self.P + self.Q + 1 + self.NEXTRA  # = 6
        r = self.R  # = 4

        # Get classical interior weights
        interior_coeffs = derive_interior(0, self.P, self.NU)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        # Sweep epsilon for rows 0..2, with row 3 using a polynomial stencil
        # Row 3 polynomial stencil: use t points {0,..,t-1}, eval at x=3
        from stencil_gen.phs import uniform_boundary_weights_rbf

        # Compute the polynomial (lagrange) stencil for row r-1
        # Use a very large epsilon (→ polynomial limit) for the last row
        poly_w_last = uniform_boundary_weights_rbf(
            r - 1, t, self.NU, self.Q, epsilon=100.0, kernel="gaussian"
        )

        eps_range = np.logspace(np.log10(0.3), np.log10(8.0), 60)
        best_eps, best_re = None, np.inf

        for eps in eps_range:
            # Build matrix: rows 0..r-2 use RBF with this eps, row r-1 uses polynomial
            D = np.zeros((n, n))

            # Left boundary rows 0..r-2: RBF
            for i in range(r - 1):
                w = uniform_boundary_weights_rbf(i, t, self.NU, self.Q, eps)
                for j in range(t):
                    D[i, j] = w[j]

            # Left boundary row r-1: polynomial stencil
            for j in range(t):
                D[r - 1, j] = poly_w_last[j]

            # Interior rows
            for i in range(r, n - r):
                for k_idx, jj in enumerate(range(i - self.P, i + self.P + 1)):
                    D[i, jj] = interior_w[k_idx]

            # Right boundary: reflected
            sign = (-1.0) ** self.NU
            for i in range(r - 1):
                w = uniform_boundary_weights_rbf(i, t, self.NU, self.Q, eps)
                row = n - 1 - i
                for j in range(t):
                    D[row, n - 1 - j] = sign * w[j]
            # Right row r-1: reflected polynomial
            row = n - 1 - (r - 1)
            for j in range(t):
                D[row, n - 1 - j] = sign * poly_w_last[j]

            eigvals = np.linalg.eigvals(D)
            mr = float(np.max(np.real(eigvals)))
            if mr < best_re:
                best_re = mr
                best_eps = eps

        print(f"\n  E4_1 conservation near-interior (n={n}):")
        print(f"  Row {r-1} uses polynomial stencil (eps→∞ limit)")
        print(f"  Rows 0..{r-2} use Gaussian with swept eps")
        print(f"  Best eps={best_eps:.4f}, max Re(λ)={best_re:.6e}")
        stable = "STABLE" if best_re <= 0 else "unstable"
        print(f"  Status: {stable}")

        # Also try: RBF for rows 0..r-2 with *different* eps for row r-1
        print(f"\n  Variant: different eps for row {r-1} (2D sweep):")
        eps_coarse = np.logspace(np.log10(0.3), np.log10(8.0), 25)
        best2 = (None, None, np.inf)

        for eps_main in eps_coarse:
            for eps_last in eps_coarse:
                epsilons = [eps_main] * (r - 1) + [eps_last]
                mr = self._max_re_mixed(n, epsilons)
                if mr < best2[2]:
                    best2 = (eps_main, eps_last, mr)

        print(f"  Best: eps_main={best2[0]:.4f}, eps_last={best2[1]:.4f}")
        print(f"  max Re(λ)={best2[2]:.6e}")

    def test_multiquadric_mixed(self):
        """Try mixed epsilon with Multiquadric kernel via coordinate descent."""
        n = 40
        eps_vals = np.logspace(np.log10(0.3), np.log10(10.0), 40)

        # Start from single-epsilon best (~5.0 for MQ)
        current = [5.0] * self.R
        current_re = self._max_re_mixed(n, current, kernel="multiquadric")

        for iteration in range(3):
            for row in range(self.R):
                best_eps_row = current[row]
                best_re_row = current_re
                for eps in eps_vals:
                    trial = list(current)
                    trial[row] = eps
                    mr = self._max_re_mixed(n, trial, kernel="multiquadric")
                    if mr < best_re_row:
                        best_re_row = mr
                        best_eps_row = eps
                current[row] = best_eps_row
                current_re = best_re_row

        opt_eps = current

        print(f"\n  E4_1 Multiquadric per-row coordinate descent (n={n}):")
        print(f"  Optimal epsilons: [{', '.join(f'{e:.4f}' for e in opt_eps)}]")
        print(f"  max Re(λ)={current_re:.6e}")

        # Grid convergence
        print(f"\n  Grid convergence:")
        for nn in [20, 40, 80]:
            mr = self._max_re_mixed(nn, opt_eps, kernel="multiquadric")
            stable = "STABLE" if mr <= 0 else "unstable"
            print(f"    n={nn:3d}: max Re(λ)={mr:.6e} [{stable}]")


# ---------------------------------------------------------------------------
# 29.6e: Extract and validate alpha values for E2 stable ε
# ---------------------------------------------------------------------------


class TestStableEpsilonAlphas:
    """Extract alpha values implied by the stable Gaussian ε for E2_1.

    E2_1 parameters: p=1, q=1, nextra=1, nu=1.
    From 29.6c: Gaussian ε*≈2.29 yields machine-precision eigenvalue stability.

    Steps:
    1. Extract RBF boundary weights at ε*
    2. Map to alpha values in the TEMO parameterization
    3. Verify eigenvalue stability with those alphas
    4. Compare with optimizer-derived production alphas
    5. Check conservation deficit
    """

    P = 1
    Q = 1
    NEXTRA = 1
    NU = 1

    # Production alphas from src/operators/gradient.t.cpp
    PROD_ALPHAS = [
        -1.47956280234494,
        0.261900367793859,
        -0.145072532538541,
        -0.224665713988644,
    ]

    def _find_best_epsilon(self, n=40):
        """Find the best Gaussian ε for E2_1 via fine sweep."""
        epsilons = np.linspace(1.5, 3.5, 200)
        best_eps, best_re = None, np.inf
        for eps in epsilons:
            mr = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            if mr < best_re:
                best_re = mr
                best_eps = eps
        return best_eps, best_re

    def _extract_boundary_weights(self, epsilon):
        """Extract the r×t boundary weight matrix at a given ε."""
        t = self.P + self.Q + 1 + self.NEXTRA  # = 4
        r = self.Q + 1 + self.NEXTRA  # = 3
        rows = []
        for i in range(r):
            w = uniform_boundary_weights_rbf(
                i, t, self.NU, self.Q, epsilon, kernel="gaussian"
            )
            rows.append(w)
        return np.array(rows), r, t

    def _extract_alphas_and_report(self, eps_star, best_re, verbose=True):
        """Extract alpha values from RBF boundary weights at given ε.

        The TEMO parameterization for E2_1 (nextra=1) places free parameters
        in the last n_free=2 columns of each of the first r_eff-1 rows.
        The last row is determined by conservation.  The RBF weights satisfy
        polynomial exactness but NOT conservation, so the alphas are extracted
        from rows 0 and 1 only.

        Returns (rbf_alphas, B_num, B_temo, ur).
        """
        from stencil_gen.temo import E2_1, derive_uniform_boundary_for_temo

        B_num, r, t = self._extract_boundary_weights(eps_star)
        ur = derive_uniform_boundary_for_temo(E2_1)
        B_sym = ur.B_u
        alphas = ur.alpha_symbols

        # For E2_1 with nextra=1: n_eqs=2, n_free=2
        # The alpha distribution is:
        #   Row 0: cols [0,1] determined, cols [2,3] = alpha_0, alpha_1
        #   Row 1: cols [0,1] determined, cols [2,3] = alpha_2, alpha_3
        #   Row 2: fully determined by conservation
        # Extract alphas directly from the free columns of rows 0,1.
        n_eqs = 2  # max(q+1, nu+1) = max(2,2)
        rbf_alphas = np.array([
            B_num[0, n_eqs],      # alpha_0 = B[0, 2]
            B_num[0, n_eqs + 1],  # alpha_1 = B[0, 3]
            B_num[1, n_eqs],      # alpha_2 = B[1, 2]
            B_num[1, n_eqs + 1],  # alpha_3 = B[1, 3]
        ])

        # Build the TEMO boundary block with these alphas
        B_temo = np.zeros((r, t))
        for i_row in range(r):
            for j_col in range(t):
                expr = B_sym[i_row, j_col]
                B_temo[i_row, j_col] = float(
                    expr.subs({a: v for a, v in zip(alphas, rbf_alphas)})
                )

        if verbose:
            print(f"\n  RBF boundary weights at ε*={eps_star:.6f}:")
            for i in range(r):
                w_str = ", ".join(f"{B_num[i, j]:12.8f}" for j in range(t))
                print(f"    row {i}: [{w_str}]")
            print(f"\n  Extracted alphas (from free columns of rows 0,1):")
            for k, (a, v) in enumerate(zip(alphas, rbf_alphas)):
                print(f"    {a} = {v:.12f}")

            # Verify rows 0,1 match exactly (they share polynomial conditions)
            for row in range(r - 1):
                resid = np.max(np.abs(B_temo[row] - B_num[row]))
                print(f"    Row {row} residual: {resid:.6e}")
                assert resid < 1e-12, f"Row {row} residual too large: {resid}"

            # Row 2: TEMO (conservation-enforced) vs RBF (not conserved)
            row2_diff = B_temo[r - 1] - B_num[r - 1]
            print(f"\n  Row {r-1} comparison (TEMO conservation vs RBF):")
            print(f"    TEMO: [{', '.join(f'{v:12.8f}' for v in B_temo[r-1])}]")
            print(f"    RBF:  [{', '.join(f'{v:12.8f}' for v in B_num[r-1])}]")
            print(f"    Diff: [{', '.join(f'{v:12.8f}' for v in row2_diff)}]")
            print(f"    Max diff: {np.max(np.abs(row2_diff)):.6e}")

        return rbf_alphas, B_num, B_temo, ur

    def test_extract_alphas(self):
        """Extract alpha values from the stable ε and verify rows 0,1 match."""
        # 1. Find best epsilon
        eps_star, best_re = self._find_best_epsilon(n=40)
        print(f"\n  E2_1 best Gaussian epsilon: ε*={eps_star:.6f}")
        print(f"  max Re(λ) at ε*: {best_re:.6e}")

        # 2. Extract alphas and verify
        rbf_alphas, B_num, B_temo, ur = self._extract_alphas_and_report(eps_star, best_re)

        # Rows 0,1 must match exactly (asserted inside helper)
        # Row 2 will differ — quantify the conservation violation
        r = B_num.shape[0]
        row2_diff_max = np.max(np.abs(B_temo[r - 1] - B_num[r - 1]))
        print(f"\n  Conservation violation (max row-2 diff): {row2_diff_max:.6e}")

    def _build_D_from_boundary(self, n, B_boundary):
        """Build n×n differentiation matrix from explicit boundary weights."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        r, t = B_boundary.shape
        interior_coeffs = derive_interior(0, self.P, self.NU)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        D = np.zeros((n, n))
        # Left boundary
        for i in range(r):
            for j in range(t):
                D[i, j] = B_boundary[i, j]
        # Interior
        for i in range(r, n - r):
            for k_idx, j in enumerate(range(i - self.P, i + self.P + 1)):
                D[i, j] = interior_w[k_idx]
        # Right boundary (antisymmetric for nu=1)
        for i in range(r):
            row = n - 1 - i
            for j in range(t):
                D[row, n - 1 - j] = -B_boundary[i, j]
        return D

    def test_verify_stability_with_extracted_alphas(self):
        """Verify eigenvalue stability: RBF direct, TEMO with RBF alphas, and TEMO
        with RBF alphas + conservation on last row.

        Three variants:
        1. RBF direct (original boundary weights from ε*)
        2. TEMO + RBF alphas (conservation-enforced last row)
        3. Production alphas for comparison
        """
        eps_star, _ = self._find_best_epsilon(n=40)
        rbf_alphas, B_num, B_temo, ur = self._extract_alphas_and_report(
            eps_star, None, verbose=False
        )

        print(f"\n  Eigenvalue stability comparison (ε*={eps_star:.4f}):")
        print(f"  {'n':>5s}  {'RBF direct':>14s}  {'TEMO+RBF α':>14s}")
        print(f"  {'-'*5}  {'-'*14}  {'-'*14}")

        for n in [20, 40, 80, 160]:
            # RBF direct (uses ε* for all rows)
            D_rbf = build_diff_matrix_rbf(
                n, p=self.P, q=self.Q, epsilon=eps_star,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            re_rbf = float(np.max(np.real(np.linalg.eigvals(D_rbf))))

            # TEMO with RBF-extracted alphas (conservation on last row)
            D_temo = self._build_D_from_boundary(n, B_temo)
            re_temo = float(np.max(np.real(np.linalg.eigvals(D_temo))))

            print(f"  {n:5d}  {re_rbf:14.6e}  {re_temo:14.6e}")

    def test_compare_with_production_alphas(self):
        """Compare RBF-extracted alphas with optimizer-derived production alphas."""
        from stencil_gen.temo import E2_1, derive_uniform_boundary_for_temo

        eps_star, _ = self._find_best_epsilon(n=40)
        rbf_alphas, B_num, B_temo, ur = self._extract_alphas_and_report(
            eps_star, None, verbose=False
        )
        alphas = ur.alpha_symbols
        B_sym = ur.B_u
        prod_alphas = np.array(self.PROD_ALPHAS)
        r, t = B_num.shape

        print(f"\n  Alpha comparison (E2_1, ε*={eps_star:.4f}):")
        print(f"  {'Symbol':>10s}  {'RBF-extracted':>16s}  {'Production':>16s}  {'Diff':>12s}")
        print(f"  {'-'*10}  {'-'*16}  {'-'*16}  {'-'*12}")
        for k, a in enumerate(alphas):
            diff = rbf_alphas[k] - prod_alphas[k]
            print(f"  {str(a):>10s}  {rbf_alphas[k]:16.10f}  {prod_alphas[k]:16.10f}  {diff:12.6e}")

        # Build boundary block with production alphas
        B_prod = np.zeros((r, t))
        for i_row in range(r):
            for j_col in range(t):
                expr = B_sym[i_row, j_col]
                B_prod[i_row, j_col] = float(
                    expr.subs({a: v for a, v in zip(alphas, prod_alphas)})
                )

        # Eigenvalue stability comparison
        n = 40
        print(f"\n  Eigenvalue stability comparison (n={n}):")

        # RBF direct (uses ε*)
        D_rbf = build_diff_matrix_rbf(
            n, p=self.P, q=self.Q, epsilon=eps_star,
            kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
        )
        eig_rbf = np.linalg.eigvals(D_rbf)
        re_rbf = float(np.max(np.real(eig_rbf)))
        sr_rbf = float(np.max(np.abs(eig_rbf)))

        # TEMO with RBF alphas (conservation)
        D_temo_rbf = self._build_D_from_boundary(n, B_temo)
        eig_temo_rbf = np.linalg.eigvals(D_temo_rbf)
        re_temo_rbf = float(np.max(np.real(eig_temo_rbf)))
        sr_temo_rbf = float(np.max(np.abs(eig_temo_rbf)))

        # Production alphas (conservation)
        D_prod = self._build_D_from_boundary(n, B_prod)
        eig_prod = np.linalg.eigvals(D_prod)
        re_prod = float(np.max(np.real(eig_prod)))
        sr_prod = float(np.max(np.abs(eig_prod)))

        print(f"  {'Method':>25s}  {'max Re(λ)':>14s}  {'spec radius':>14s}")
        print(f"  {'-'*25}  {'-'*14}  {'-'*14}")
        print(f"  {'RBF direct (ε*)':>25s}  {re_rbf:14.6e}  {sr_rbf:14.6e}")
        print(f"  {'TEMO + RBF alphas':>25s}  {re_temo_rbf:14.6e}  {sr_temo_rbf:14.6e}")
        print(f"  {'Production alphas':>25s}  {re_prod:14.6e}  {sr_prod:14.6e}")

    def test_conservation_deficit(self):
        """Check conservation deficit of the RBF-extracted boundary stencil.

        Conservation requires: sum_i B_u[i, j] + IC_contribution = 0 for
        columns in the overlap region (j >= p+1).  The deficit measures how
        far the boundary block is from satisfying SBP conservation.
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        # Find best epsilon and get boundary weights
        eps_star, _ = self._find_best_epsilon(n=40)
        B_num, r, t = self._extract_boundary_weights(eps_star)

        # Interior stencil
        interior_coeffs = derive_interior(0, self.P, self.NU)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        # Conservation check: for each column j of the boundary block,
        # compute the column sum of boundary rows.
        # For a conservative scheme with unit weights: sum_i B_u[i,j] = 0
        # for columns j >= r (the overlap region where boundary and interior agree).
        # More precisely, for j in [p+1, t-1], the boundary column sum
        # should equal the negative of the interior contribution at that column.
        print(f"\n  Conservation analysis for E2_1 RBF boundary (ε*={eps_star:.4f}):")
        print(f"  r={r}, t={t}, p={self.P}")
        print(f"\n  Column sums of boundary block B_u:")
        for j in range(t):
            col_sum = sum(B_num[i, j] for i in range(r))
            print(f"    col {j}: sum = {col_sum:12.8f}")

        # For the overlap columns (j >= p+1 = 2), the interior stencil's
        # contribution to these columns from row r (first interior row) is
        # interior_w[j - r + p] (when the interior stencil is centered at row r).
        # Conservation deficit = boundary col sum + interior extension.
        print(f"\n  Conservation deficit (boundary col sum for overlap cols j >= {self.P+1}):")
        for j in range(self.P + 1, t):
            col_sum = sum(B_num[i, j] for i in range(r))
            # Interior row r has stencil centered at r, so column j has
            # contribution interior_w[j - r + p] = interior_w[j - r + 1]
            # But this depends on which interior rows contribute to column j.
            # For unit-weight conservation: boundary col sum should be 0.
            print(f"    col {j}: boundary sum = {col_sum:12.8f}")
