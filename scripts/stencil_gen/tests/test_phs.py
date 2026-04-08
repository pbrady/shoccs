"""Tests for PHS+poly stencil derivation (Phase 29)."""

import pytest
from sympy import Rational, S, Symbol, cancel, symbols

from stencil_gen.phs import (
    _tension_kernel_eval,
    _tension_kernel_deriv,
    build_diff_matrix_rbf_penalty,
    cut_cell_weights,
    phs_stencil_weights,
    uniform_boundary_weights,
    uniform_boundary_weights_rbf,
    uniform_boundary_weights_tension,
    uniform_interior_weights,
    uniform_interior_weights_rbf,
    uniform_interior_weights_tension,
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
    stability_eigenvalue,
    stability_eigenvalue_from_matrix,
)

# Floating-point eigenvalue solvers return tiny positive real parts (~1e-14)
# for genuinely stable operators.  Use this threshold to distinguish true
# instability from numerical noise.
STABILITY_TOL = 1e-10


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
# 30.2a: Diff matrix builder with tension kernel
# ---------------------------------------------------------------------------


class TestBuildDiffMatrixTension:
    """Tests for build_diff_matrix_rbf with kernel='tension'."""

    def test_matrix_shape(self):
        """Tension diff matrix should be n×n."""
        for n in [20, 40]:
            D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=2.0,
                                      kernel="tension", nextra=1)
            assert D.shape == (n, n)

    def test_interior_column_sums_zero(self):
        """Interior rows should sum to 0 (first derivative)."""
        n = 30
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=3.0,
                                  kernel="tension", nextra=1)
        r = 3  # q + 1 + nextra = 1 + 1 + 1
        for i in range(r, n - r):
            row_sum = np.sum(D[i, :])
            assert abs(row_sum) < 1e-14, f"Interior row {i} sum = {row_sum}"

    def test_sigma_zero_matches_phs(self):
        """At σ=0, tension diff matrix should match PHS k=2."""
        n = 20
        D_tension = build_diff_matrix_rbf(n, p=1, q=1, epsilon=0.0,
                                          kernel="tension", nextra=1)
        D_phs = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0,
                                      kernel="gaussian", nextra=1)
        # Build reference PHS k=2 matrix manually
        from stencil_gen.interior import derive_interior, full_gamma_array

        r = 3  # q + 1 + nextra
        t = 4  # p + q + 1 + nextra
        D_ref = np.zeros((n, n))
        for i in range(r):
            w = [float(x) for x in phs_stencil_weights(
                [Rational(j) for j in range(t)], Rational(i), 1, 1, k=2)]
            for j in range(t):
                D_ref[i, j] = w[j]
        interior_w = [float(c) for c in full_gamma_array(derive_interior(0, 1, 1))]
        for i in range(r, n - r):
            for k_idx, j in enumerate(range(i - 1, i + 2)):
                D_ref[i, j] = interior_w[k_idx]
        for i in range(r):
            w = [float(x) for x in phs_stencil_weights(
                [Rational(j) for j in range(t)], Rational(i), 1, 1, k=2)]
            row = n - 1 - i
            for j in range(t):
                D_ref[row, n - 1 - j] = -w[j]

        np.testing.assert_allclose(D_tension, D_ref, atol=1e-13,
                                   err_msg="Tension at σ=0 matrix ≠ PHS k=2 matrix")

    def test_polynomial_reproduction(self):
        """D applied to x should give all 1s (exact for linear)."""
        n = 30
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=5.0,
                                  kernel="tension", nextra=1)
        x = np.arange(n, dtype=float)
        result = D @ x
        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_antisymmetry_first_deriv(self):
        """Right boundary should be antisymmetric reflection of left for nu=1."""
        n = 20
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=3.0,
                                  kernel="tension", nextra=1)
        r = 3
        t = 4
        for i in range(r):
            left_row = D[i, :t]
            right_row = D[n - 1 - i, n - t:][::-1]
            np.testing.assert_allclose(right_row, -left_row, atol=1e-14)

    def test_nu2_polynomial_reproduction(self):
        """D (nu=2) applied to x should give all 0s (exact for linear)."""
        n = 30
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=3.0,
                                  kernel="tension", nu=2, nextra=0)
        x = np.arange(n, dtype=float)
        result = D @ x
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_eigenvalue_finite(self):
        """max_real_eigenvalue should return a finite float for tension kernel."""
        result = max_real_eigenvalue(20, p=1, q=1, epsilon=2.0,
                                    kernel="tension", nextra=1)
        assert np.isfinite(result), f"Non-finite max Re(λ) = {result}"

    def test_mixed_epsilon_tension(self):
        """build_diff_matrix_mixed_epsilon should work with tension kernel."""
        n = 20
        r = 3  # q + 1 + nextra for p=1, q=1, nextra=1
        sigmas = [1.0, 2.0, 3.0]
        D = build_diff_matrix_mixed_epsilon(n, p=1, q=1, epsilons=sigmas,
                                            kernel="tension", nextra=1)
        assert D.shape == (n, n)
        # Polynomial reproduction: D @ x = 1
        x = np.arange(n, dtype=float)
        result = D @ x
        np.testing.assert_allclose(result, 1.0, atol=1e-12)


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
# 30.1d: Tension spline kernel tests
# ---------------------------------------------------------------------------


class TestTensionSpline:
    """Tests for the tension spline kernel φ(r;σ) = σ|r| - 1 + exp(-σ|r|)."""

    def test_sigma_zero_matches_phs_k2(self):
        """At very small σ, tension boundary weights ≈ PHS k=2 weights."""
        import numpy as np

        # E2 boundary: p=1, q=1, t=3, row i=0
        phs_w = uniform_boundary_weights(0, 3, nu=1, k=2, q=1)
        phs_w_float = [float(w) for w in phs_w]

        # Tension with very small sigma should approach PHS k=2
        tension_w = uniform_boundary_weights_tension(0, 3, nu=1, q=1, sigma=1e-6)

        np.testing.assert_allclose(tension_w, phs_w_float, atol=1e-6,
                                   err_msg="Tension at σ≈0 should match PHS k=2")

    def test_polynomial_exactness(self):
        """Tension stencil should be exact for polynomials up to degree q."""
        import numpy as np

        for q in [1, 2, 3]:
            t = q + 3  # enough points
            sigma = 2.0
            for i in range(min(2, t)):
                w = uniform_boundary_weights_tension(i, t, nu=1, q=q, sigma=sigma)
                pts = list(range(t))
                for d in range(q + 1):
                    # sum_j w_j * x_j^d should equal d * i^(d-1) for d >= 1, 0 for d=0
                    actual = sum(wj * xj**d for wj, xj in zip(w, pts))
                    expected = d * i ** max(0, d - 1) if d >= 1 else 0.0
                    np.testing.assert_allclose(
                        actual, expected, atol=1e-10,
                        err_msg=f"q={q}, i={i}, d={d}: poly exactness failed"
                    )

    def test_weights_sum_to_zero(self):
        """First derivative weights should sum to 0 (exact for constants)."""
        import numpy as np

        for sigma in [0.5, 2.0, 10.0]:
            w = uniform_boundary_weights_tension(0, 4, nu=1, q=1, sigma=sigma)
            np.testing.assert_allclose(
                sum(w), 0.0, atol=1e-12,
                err_msg=f"σ={sigma}: weights don't sum to 0"
            )

    def test_kernel_symmetry(self):
        """φ(r;σ) = φ(-r;σ) — kernel is an even function."""
        for sigma in [0.1, 1.0, 5.0, 20.0]:
            for r in [0.5, 1.0, 2.5, 7.0]:
                val_pos = _tension_kernel_eval(r, sigma)
                val_neg = _tension_kernel_eval(-r, sigma)
                assert abs(val_pos - val_neg) < 1e-14, (
                    f"σ={sigma}, r={r}: φ(r)={val_pos} ≠ φ(-r)={val_neg}"
                )

    def test_interior_matches_classical(self):
        """Interior tension weights match classical FD for all σ."""
        import numpy as np
        from stencil_gen.interior import derive_interior, full_gamma_array

        classical = [float(c) for c in full_gamma_array(derive_interior(0, 1, 1))]

        for sigma in [0.01, 1.0, 5.0, 20.0]:
            tension_w = uniform_interior_weights_tension(p=1, nu=1, q=2, sigma=sigma)
            np.testing.assert_allclose(
                tension_w, classical, atol=1e-10,
                err_msg=f"σ={sigma}: interior weights differ from classical"
            )

    def test_numerical_stability_large_sigma(self):
        """No overflow for σ up to 50 on unit grid."""
        import numpy as np

        for sigma in [10.0, 25.0, 50.0]:
            w = uniform_boundary_weights_tension(0, 4, nu=1, q=1, sigma=sigma)
            assert all(np.isfinite(w)), (
                f"σ={sigma}: non-finite weights {w}"
            )

    def test_kernel_positive_for_nonzero_r(self):
        """φ(r;σ) > 0 for r ≠ 0 and σ > 0."""
        for sigma in [0.1, 1.0, 5.0, 20.0]:
            for r in [0.1, 0.5, 1.0, 3.0, 10.0]:
                val = _tension_kernel_eval(r, sigma)
                assert val > 0, f"σ={sigma}, r={r}: φ={val} should be positive"

    def test_kernel_zero_at_origin(self):
        """φ(0;σ) = 0 for all σ."""
        for sigma in [0.0, 0.1, 1.0, 10.0]:
            assert _tension_kernel_eval(0.0, sigma) == 0.0

    def test_d1_antisymmetric(self):
        """D¹φ is an odd function: D¹φ(-r) = -D¹φ(r)."""
        for sigma in [0.5, 2.0, 10.0]:
            for r in [0.5, 1.0, 3.0]:
                dp = _tension_kernel_deriv(r, 1, sigma)
                dm = _tension_kernel_deriv(-r, 1, sigma)
                assert abs(dp + dm) < 1e-13, (
                    f"σ={sigma}, r={r}: D¹φ(r)+D¹φ(-r) = {dp+dm}"
                )

    def test_d2_symmetric(self):
        """D²φ is an even function: D²φ(-r) = D²φ(r)."""
        for sigma in [0.5, 2.0, 10.0]:
            for r in [0.5, 1.0, 3.0]:
                dp = _tension_kernel_deriv(r, 2, sigma)
                dm = _tension_kernel_deriv(-r, 2, sigma)
                assert abs(dp - dm) < 1e-13, (
                    f"σ={sigma}, r={r}: D²φ(r)-D²φ(-r) = {dp-dm}"
                )

    def test_taylor_matches_direct(self):
        """Taylor branch (z<2) matches direct evaluation at the boundary z≈2."""
        import numpy as np

        # Test at z = 1.99 (Taylor) vs z = 2.01 (direct) — should be close
        sigma = 2.0
        r = 1.0  # z = sigma*r = 2.0, right at boundary
        # Evaluate slightly on each side
        r_lo = 0.99  # z = 1.98, Taylor path
        r_hi = 1.01  # z = 2.02, direct path
        phi_lo = _tension_kernel_eval(r_lo, sigma)
        phi_hi = _tension_kernel_eval(r_hi, sigma)
        # They should be close (continuous function)
        expected_diff = abs(phi_hi - phi_lo)
        assert expected_diff < 0.1, (
            f"φ discontinuous near Taylor/direct boundary: {phi_lo} vs {phi_hi}"
        )

        # Also check derivative continuity
        for nu in [1, 2]:
            d_lo = _tension_kernel_deriv(r_lo, nu, sigma)
            d_hi = _tension_kernel_deriv(r_hi, nu, sigma)
            assert abs(d_hi - d_lo) < 0.2, (
                f"D{nu}φ discontinuous near boundary: {d_lo} vs {d_hi}"
            )

    def test_sigma_exactly_zero_dispatches_to_phs(self):
        """At σ=0.0, tension wrappers must not crash and must match PHS k=2."""
        import numpy as np

        # Boundary weights: E2 layout (p=1, q=1, t=3, row i=0)
        phs_w = uniform_boundary_weights(0, 3, nu=1, k=2, q=1)
        phs_w_float = [float(w) for w in phs_w]

        tension_w = uniform_boundary_weights_tension(0, 3, nu=1, q=1, sigma=0.0)
        np.testing.assert_allclose(
            tension_w, phs_w_float, atol=1e-14,
            err_msg="Tension at σ=0 should exactly match PHS k=2",
        )

        # Interior weights: p=1, q=1
        phs_int = uniform_interior_weights(1, nu=1, k=2, q=1)
        phs_int_float = [float(w) for w in phs_int]

        tension_int = uniform_interior_weights_tension(1, nu=1, q=1, sigma=0.0)
        np.testing.assert_allclose(
            tension_int, phs_int_float, atol=1e-14,
            err_msg="Interior tension at σ=0 should exactly match PHS k=2",
        )

    def test_nu2_polynomial_exactness(self):
        """Second-derivative weights reproduce D² x^d exactly for d ≤ q."""
        import numpy as np

        for q in [2, 3]:
            t = q + 4  # enough points for a well-determined system
            sigma = 3.0
            for i in range(t):
                w = uniform_boundary_weights_tension(i, t, nu=2, q=q, sigma=sigma)
                pts = np.arange(t, dtype=float)
                for d in range(q + 1):
                    # D² x^d at x=i should be d*(d-1)*i^(d-2) for d>=2, else 0
                    got = sum(wj * xj**d for wj, xj in zip(w, pts))
                    if d >= 2:
                        expected = d * (d - 1) * float(i) ** (d - 2)
                    else:
                        expected = 0.0
                    assert abs(got - expected) < 1e-10, (
                        f"nu=2 poly exactness failed: q={q}, i={i}, d={d}, "
                        f"got={got}, expected={expected}"
                    )


# ---------------------------------------------------------------------------
# 30.3a: Soft conservation penalty
# ---------------------------------------------------------------------------


class TestConservationPenalty:
    """Tests for build_diff_matrix_rbf_penalty (Phase 30.3a).

    Verifies that the penalty-augmented RBF-FD system:
    1. At γ=0, recovers the standard RBF weights exactly.
    2. As γ→∞, approaches conservation-enforced weights (zero column sums).
    3. Preserves polynomial exactness at all γ values.
    """

    # E2_1 parameters
    E2_P, E2_Q, E2_NEXTRA, E2_NU = 1, 1, 1, 1
    # E4_1 parameters
    E4_P, E4_Q, E4_NEXTRA, E4_NU = 2, 3, 0, 1

    def _conservation_deficit(self, D):
        """Max absolute column sum of D."""
        return float(np.max(np.abs(np.sum(D, axis=0))))

    def _polynomial_reproduction_error(self, D, n, nu, q):
        """Max error in D applied to polynomials x^d for d=0..q.

        The differentiation matrix D is built for unit-spacing grid
        {0, 1, ..., n-1}, so f_j = j^d and (Df)_i should equal
        d!/(d-nu)! * i^{d-nu}.
        """
        x = np.arange(n, dtype=float)
        max_err = 0.0
        for d in range(q + 1):
            f = x**d
            Df = D @ f
            if d >= nu:
                coeff = 1.0
                for j in range(nu):
                    coeff *= (d - j)
                exact = coeff * x ** (d - nu)
            else:
                exact = np.zeros(n)
            err = np.max(np.abs(Df - exact))
            max_err = max(max_err, err)
        return max_err

    def test_gamma_zero_matches_standard_e2(self):
        """γ=0 penalty matrix is identical to standard RBF matrix (E2)."""
        n, sigma = 40, 6.0
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU

        D_std = build_diff_matrix_rbf(n, p, q, sigma, "tension", nu, nextra)
        D_pen = build_diff_matrix_rbf_penalty(
            n, p, q, sigma, "tension", nu, nextra, gamma=0.0
        )

        np.testing.assert_allclose(D_pen, D_std, atol=1e-15)

    def test_gamma_zero_matches_standard_e4(self):
        """γ=0 penalty matrix is identical to standard RBF matrix (E4)."""
        n, sigma = 40, 37.0
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU

        D_std = build_diff_matrix_rbf(n, p, q, sigma, "tension", nu, nextra)
        D_pen = build_diff_matrix_rbf_penalty(
            n, p, q, sigma, "tension", nu, nextra, gamma=0.0
        )

        np.testing.assert_allclose(D_pen, D_std, atol=1e-15)

    def test_conservation_improves_with_gamma_e2(self):
        """Conservation deficit decreases as γ increases (E2).

        Full conservation is NOT achievable while maintaining polynomial
        exactness: the null space of P has dimension t-(q+1) per row,
        but all rows share the same null space, so the effective column-sum
        freedom is only t-(q+1) dimensions vs t conservation equations.
        The penalty reduces the deficit to a fundamental limit set by the
        polynomial-exactness / conservation trade-off.
        """
        n, sigma = 40, 6.0
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU

        gammas = [0, 1, 10, 100, 1000, 1e6]
        deficits = []
        for g in gammas:
            D = build_diff_matrix_rbf_penalty(
                n, p, q, sigma, "tension", nu, nextra, gamma=g
            )
            deficits.append(self._conservation_deficit(D))

        print("\n  E2 conservation deficit vs γ:")
        for g, d in zip(gammas, deficits):
            print(f"    γ={g:>10.0f}  deficit={d:.6e}")

        # Deficit should decrease overall
        assert deficits[-1] < deficits[0], (
            f"Large γ ({deficits[-1]:.6e}) should reduce deficit vs γ=0 ({deficits[0]:.6e})"
        )
        # At large γ, deficit converges to a fundamental limit (rank-limited)
        assert abs(deficits[-1] - deficits[-2]) / deficits[-2] < 0.01, (
            "Deficit should converge at large γ"
        )

    def test_conservation_improves_with_gamma_e4(self):
        """Conservation deficit decreases as γ increases (E4)."""
        n, sigma = 40, 37.0
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU

        gammas = [0, 1, 10, 100, 1000, 1e6]
        deficits = []
        for g in gammas:
            D = build_diff_matrix_rbf_penalty(
                n, p, q, sigma, "tension", nu, nextra, gamma=g
            )
            deficits.append(self._conservation_deficit(D))

        print("\n  E4 conservation deficit vs γ:")
        for g, d in zip(gammas, deficits):
            print(f"    γ={g:>10.0f}  deficit={d:.6e}")

        assert deficits[-1] < deficits[0], (
            f"Large γ ({deficits[-1]:.6e}) should reduce deficit vs γ=0 ({deficits[0]:.6e})"
        )
        # Converges at large γ
        assert abs(deficits[-1] - deficits[-2]) / deficits[-2] < 0.01, (
            "Deficit should converge at large γ"
        )

    def test_polynomial_exactness_preserved_e2(self):
        """Polynomial exactness is maintained at all γ values (E2)."""
        n, sigma = 40, 6.0
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU

        for g in [0, 10, 1000, 1e6]:
            D = build_diff_matrix_rbf_penalty(
                n, p, q, sigma, "tension", nu, nextra, gamma=g
            )
            err = self._polynomial_reproduction_error(D, n, nu, q)
            assert err < 1e-8, (
                f"Polynomial exactness lost at γ={g}: error={err:.6e}"
            )

    def test_polynomial_exactness_preserved_e4(self):
        """Polynomial exactness is maintained at all γ values (E4)."""
        n, sigma = 40, 37.0
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU

        for g in [0, 10, 1000, 1e6]:
            D = build_diff_matrix_rbf_penalty(
                n, p, q, sigma, "tension", nu, nextra, gamma=g
            )
            err = self._polynomial_reproduction_error(D, n, nu, q)
            assert err < 1e-8, (
                f"Polynomial exactness lost at γ={g}: error={err:.6e}"
            )



# ---------------------------------------------------------------------------
# 30.4b: Modified wavenumber analysis for tension spline stencils
# ---------------------------------------------------------------------------


class TestModifiedWavenumber:
    """Modified wavenumber analysis for boundary vs interior stencils (Phase 30.4b).

    For a stencil w_j applied at node i_eval using nodes {j_0, ..., j_{t-1}},
    the modified wavenumber is:

        κ*(ξ) = Σ_j w_j · exp(i·(j - i_eval)·ξ)

    For D¹: exact κ* = iξ  →  Re(κ*)=0, Im(κ*)=ξ.
      - Re(κ*) < 0 is dissipative (good for stability)
      - Re(κ*) > 0 is amplifying (bad — causes eigenvalue instability)

    We check:
    1. Interior stencil Re(κ*) = 0 (centered, antisymmetric → pure imaginary)
    2. At optimal tension σ*, boundary Re(κ*_bdy) ≤ 0 for all ξ (E2)
    3. For E4, boundary Re(κ*_bdy) may have small positive region (unstable)
    4. Compare boundary vs interior dispersion Im(κ*)
    """

    # E2_1 parameters
    E2_P, E2_Q, E2_NEXTRA, E2_NU = 1, 1, 1, 1
    # E4_1 parameters
    E4_P, E4_Q, E4_NEXTRA, E4_NU = 2, 3, 0, 1

    N_XI = 500  # wavenumber resolution

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _modified_wavenumber(weights, i_eval, node_indices, xi_array):
        """Compute modified wavenumber κ*(ξ) for given stencil weights.

        Parameters
        ----------
        weights : array-like
            Stencil coefficients w_j.
        i_eval : int
            Grid index where derivative is evaluated.
        node_indices : array-like of int
            Grid indices used by the stencil (e.g. [0,1,...,t-1] for boundary).
        xi_array : np.ndarray
            Wavenumber values ξ ∈ [0, π].

        Returns
        -------
        np.ndarray (complex)
            κ*(ξ) = Σ_j w_j exp(i (j - i_eval) ξ)
        """
        w = np.asarray(weights, dtype=complex)
        offsets = np.asarray(node_indices) - i_eval  # j - i_eval
        # κ*(ξ) = Σ w_j exp(i·offset_j·ξ)  (vectorized over ξ)
        # shape: (len(xi), len(offsets))
        phase = np.exp(1j * np.outer(xi_array, offsets))
        return phase @ w  # shape (len(xi),)

    def _interior_mod_wavenumber(self, p, nu, xi_array):
        """Compute modified wavenumber for the classical interior stencil."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        coeffs = derive_interior(0, p, nu)
        w = [float(c) for c in full_gamma_array(coeffs)]
        nodes = list(range(-p, p + 1))
        return self._modified_wavenumber(w, 0, nodes, xi_array)

    def _boundary_mod_wavenumbers(self, p, q, nextra, nu, sigma, kernel="tension"):
        """Compute modified wavenumber for all boundary rows.

        Returns dict: {row_index: κ*(ξ)} for rows 0..r-1.
        """
        t = p + q + 1 + nextra  # boundary stencil width (nu=1)
        r = q + 1 + nextra       # number of boundary rows

        xi = np.linspace(0, np.pi, self.N_XI)
        nodes = list(range(t))
        result = {}
        for i in range(r):
            w = uniform_boundary_weights_rbf(i, t, nu, q, sigma, kernel=kernel)
            result[i] = self._modified_wavenumber(w, i, nodes, xi)
        return result

    def _find_best_sigma(self, n, p, q, nu, nextra):
        """Coarse + fine sweep for best tension σ using corrected stability metric."""
        sigmas_coarse = np.concatenate([[0.0], np.linspace(1.0, 55.0, 100)])
        best_sigma, best_se = None, np.inf
        for s in sigmas_coarse:
            se = stability_eigenvalue(n, p, q, s, "tension", nu, nextra)
            if se < best_se:
                best_se = se
                best_sigma = s

        lo = max(0.0, best_sigma - 5.0)
        hi = min(60.0, best_sigma + 5.0)
        for s in np.linspace(lo, hi, 200):
            se = stability_eigenvalue(n, p, q, s, "tension", nu, nextra)
            if se < best_se:
                best_se = se
                best_sigma = s

        return best_sigma, best_se

    # ---------------------------------------------- interior sanity check

    def test_interior_pure_imaginary(self):
        """Interior centered D¹ stencil has Re(κ*)=0 (antisymmetric weights)."""
        xi = np.linspace(0, np.pi, self.N_XI)
        for p in [1, 2]:
            kappa = self._interior_mod_wavenumber(p, nu=1, xi_array=xi)
            max_real = float(np.max(np.abs(np.real(kappa))))
            assert max_real < 1e-14, (
                f"Interior p={p} Re(κ*) should be 0, got max |Re|={max_real:.2e}"
            )

    # ---------------------------------------------- E2 boundary analysis

    def test_e2_boundary_at_optimal_sigma(self):
        """Modified wavenumber profile of E2 boundary rows at optimal tension σ*.

        Key finding: the full operator is stable under corrected metric.
        Individual boundary stencils can have small positive Re(κ*), but
        stability is a global property of the coupled operator, not a
        per-stencil property.
        """
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU
        sigma_star, best_se = self._find_best_sigma(40, p, q, nu, nextra)

        xi = np.linspace(0, np.pi, self.N_XI)
        bdy_kappas = self._boundary_mod_wavenumbers(p, q, nextra, nu, sigma_star)

        r = q + 1 + nextra
        print(f"\n  E2 Modified Wavenumber Analysis at σ*={sigma_star:.3f}")
        print(f"  (stab_eig = {best_se:.2e})")
        print(f"  {'row':>4s}  {'max Re(κ*)':>14s}  {'min Re(κ*)':>14s}"
              f"  {'max |Im(κ*)-ξ|':>16s}")
        print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*16}")

        kappa_int = self._interior_mod_wavenumber(p, nu, xi)

        max_re_all = -np.inf
        for i in range(r):
            kappa = bdy_kappas[i]
            max_re = float(np.max(np.real(kappa)))
            min_re = float(np.min(np.real(kappa)))
            disp_err = float(np.max(np.abs(np.imag(kappa) - np.imag(kappa_int))))
            print(f"  {i:4d}  {max_re:14.6e}  {min_re:14.6e}  {disp_err:16.6e}")
            max_re_all = max(max_re_all, max_re)

        # Row 0 (boundary point itself) should be dissipative
        assert float(np.max(np.real(bdy_kappas[0]))) < STABILITY_TOL, (
            f"E2 boundary row 0 should be dissipative at σ*={sigma_star:.3f}"
        )

        # Per-stencil amplification is bounded (O(0.1-0.3), much less than 1).
        # At the corrected-metric optimal σ*, per-stencil amplification can be
        # larger than under the old metric because the optimal σ* is different.
        assert max_re_all < 0.5, (
            f"E2 boundary max Re(κ*) too large at σ*={sigma_star:.3f}: {max_re_all:.6e}"
        )

        # Full matrix is stable under corrected metric
        assert best_se < 0, (
            f"E2 full matrix should be stable at σ*: stab_eig = {best_se:.6e}"
        )

    def test_e2_boundary_amplifying_at_sigma_zero(self):
        """At σ=0 (PHS k=2), E2 boundary rows have Re(κ*) > 0 (some amplifying).

        Per-stencil amplification (Re(κ*) > 0 for some wavenumbers) is a local
        property that does NOT imply full-operator instability.  PHS k=2 IS
        stable under the corrected full-matrix test, but individual boundary
        stencils can still have amplifying modes at certain wavenumbers.
        """
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU
        # Use σ → 0 (dispatches to PHS k=2)
        sigma_zero = 1e-15

        bdy_kappas = self._boundary_mod_wavenumbers(p, q, nextra, nu, sigma_zero)

        r = q + 1 + nextra
        any_amplifying = False
        for i in range(r):
            kappa = bdy_kappas[i]
            max_re = float(np.max(np.real(kappa)))
            if max_re > STABILITY_TOL:
                any_amplifying = True

        assert any_amplifying, (
            "At σ=0 (PHS k=2), at least one E2 boundary row should be amplifying"
        )

    # ---------------------------------------------- E4 boundary analysis

    def test_e4_boundary_at_optimal_sigma(self):
        """Modified wavenumber profile of E4 boundary rows at optimal tension σ*.

        Under corrected metric, E4 IS stable (full operator has stab_eig < 0).
        Per-stencil modified wavenumber can still show positive Re(κ*) regions
        at some wavenumbers — this is a local property that does NOT imply
        full-operator instability.  Per-stencil analysis overpredicts
        instability vs the coupled operator.
        """
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU
        sigma_star, best_se = self._find_best_sigma(40, p, q, nu, nextra)

        xi = np.linspace(0, np.pi, self.N_XI)
        bdy_kappas = self._boundary_mod_wavenumbers(p, q, nextra, nu, sigma_star)

        r = q + 1 + nextra
        kappa_int = self._interior_mod_wavenumber(p, nu, xi)

        print(f"\n  E4 Modified Wavenumber Analysis at σ*={sigma_star:.3f}")
        print(f"  (stab_eig = {best_se:.2e})")
        print(f"  {'row':>4s}  {'max Re(κ*)':>14s}  {'min Re(κ*)':>14s}"
              f"  {'max |Im(κ*)-ξ|':>16s}")
        print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*16}")

        overall_max_re = -np.inf
        for i in range(r):
            kappa = bdy_kappas[i]
            max_re = float(np.max(np.real(kappa)))
            min_re = float(np.min(np.real(kappa)))
            disp_err = float(np.max(np.abs(np.imag(kappa) - np.imag(kappa_int))))
            print(f"  {i:4d}  {max_re:14.6e}  {min_re:14.6e}  {disp_err:16.6e}")
            overall_max_re = max(overall_max_re, max_re)

        # Per-stencil amplification is bounded (O(0.1), much less than 1)
        assert overall_max_re < 0.5, (
            f"E4 boundary max Re(κ*) too large at σ*={sigma_star:.3f}: {overall_max_re:.6e}"
        )

        # Full operator is stable under corrected metric, even though
        # individual boundary stencils may have per-stencil amplification
        assert best_se < 0, (
            f"E4 full matrix should be stable at σ*: stab_eig = {best_se:.6e}"
        )

    def test_e4_phs_boundary_vs_tension_per_stencil(self):
        """Compare per-stencil max Re(κ*) between PHS k=2 and tension σ=3.0.

        Under corrected metric, PHS k=2 (σ=0) is the full-operator optimal.
        Tension at σ>0 can reduce per-stencil amplification, but this is a
        local property that doesn't affect full-operator stability.
        Both configurations are stable under the correct full-matrix test.
        """
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU

        # PHS k=2 boundary max Re
        bdy_phs = self._boundary_mod_wavenumbers(p, q, nextra, nu, 1e-15)
        r = q + 1 + nextra
        phs_max_re = max(
            float(np.max(np.real(bdy_phs[i]))) for i in range(r)
        )

        # Tension σ=3.0 (production value) boundary max Re
        sigma_prod = 3.0
        bdy_tension = self._boundary_mod_wavenumbers(p, q, nextra, nu, sigma_prod)
        tension_max_re = max(
            float(np.max(np.real(bdy_tension[i]))) for i in range(r)
        )

        print(f"\n  E4 max Re(κ*) across boundary rows:")
        print(f"    PHS k=2 (σ=0):      {phs_max_re:.6e}")
        print(f"    Tension σ={sigma_prod:.1f}:    {tension_max_re:.6e}")

        # Both should have bounded per-stencil amplification
        assert phs_max_re < 0.5, (
            f"PHS k=2 per-stencil amplification too large: {phs_max_re:.6e}"
        )
        assert tension_max_re < 0.5, (
            f"Tension σ={sigma_prod} per-stencil amplification too large: "
            f"{tension_max_re:.6e}"
        )

    # ---------------------------------------------- dispersion comparison

    def test_dispersion_comparison(self):
        """Compare boundary vs interior dispersion for E2 and E4.

        Interior Im(κ*) approximates ξ to order 2p.  Boundary rows may have
        different dispersion.  We verify boundary dispersion error is bounded
        and report the comparison.
        """
        xi = np.linspace(0, np.pi, self.N_XI)

        configs = [
            ("E2", self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU),
            ("E4", self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU),
        ]

        print(f"\n  {'='*80}")
        print(f"  Dispersion Comparison: boundary vs interior Im(κ*)")
        print(f"  {'='*80}")

        for label, p, q, nextra, nu in configs:
            sigma_star, _ = self._find_best_sigma(40, p, q, nu, nextra)
            kappa_int = self._interior_mod_wavenumber(p, nu, xi)

            bdy_kappas = self._boundary_mod_wavenumbers(
                p, q, nextra, nu, sigma_star,
            )
            r = q + 1 + nextra

            print(f"\n  {label} — σ*={sigma_star:.3f}")
            print(f"  {'row':>4s}  {'max |ΔIm|':>14s}  {'mean |ΔIm|':>14s}")
            print(f"  {'-'*4}  {'-'*14}  {'-'*14}")

            # Interior dispersion error vs exact (iξ → Im = ξ)
            int_disp_err = float(np.max(np.abs(np.imag(kappa_int) - xi)))
            print(f"  {'int':>4s}  {int_disp_err:14.6e}  "
                  f"{float(np.mean(np.abs(np.imag(kappa_int) - xi))):14.6e}")

            for i in range(r):
                kappa = bdy_kappas[i]
                disp_vs_exact = np.abs(np.imag(kappa) - xi)
                max_err = float(np.max(disp_vs_exact))
                mean_err = float(np.mean(disp_vs_exact))
                print(f"  {i:4d}  {max_err:14.6e}  {mean_err:14.6e}")

                # Boundary dispersion should be finite (not blow up)
                assert max_err < 10.0, (
                    f"{label} row {i} dispersion error too large: {max_err:.2e}"
                )




# ---------------------------------------------------------------------------
# 32.1c: Validate corrected stability infrastructure
# ---------------------------------------------------------------------------


class TestStabilityInfrastructure:
    """Validate stability_eigenvalue with correct BC and sign convention.

    The corrected test removes the inflow row/column (Dirichlet at left)
    and checks eigenvalues of -D (the semi-discrete advection operator).
    """

    def test_production_e4_tension_stable(self):
        """E4 with tension spline at σ=3.0 is stable under correct test.

        With the corrected stability check (inflow-Dirichlet BC, eigenvalues
        of -D), the E4 tension spline configuration is stable at all grid
        sizes.
        """
        for n in [20, 40, 80, 160]:
            se = stability_eigenvalue(n, p=2, q=3, epsilon=3.0,
                                      kernel="tension", nu=1, nextra=0)
            assert se < STABILITY_TOL, (
                f"E4 tension σ=3.0, n={n}: expected stable, "
                f"got stability_eigenvalue={se:.6e}"
            )

    def test_interior_only_neutrally_stable(self):
        """Periodic interior-only matrix should be neutrally stable.

        A pure circulant interior matrix has purely imaginary eigenvalues,
        so eigenvalues of -D also have zero real parts (neutrally stable).
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        n = 40
        p = 2
        interior_coeffs = derive_interior(0, p, 1)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        D = np.zeros((n, n))
        for i in range(n):
            for k_idx, offset in enumerate(range(-p, p + 1)):
                j = (i + offset) % n
                D[i, j] = interior_w[k_idx]

        se = stability_eigenvalue_from_matrix(D)
        assert abs(se) < 1e-12, (
            f"Periodic interior stability_eigenvalue should be ≈0, got {se:.6e}"
        )

    def test_unstable_detected(self):
        """A known-unstable configuration should have positive stability_eigenvalue.

        E4 (p=2) with Gaussian RBF at epsilon=0.1 produces an unstable
        advection operator, confirming the test can detect instability.
        """
        se = stability_eigenvalue(20, p=2, q=3, epsilon=0.1,
                                  kernel="gaussian", nu=1, nextra=0)
        assert se > STABILITY_TOL, (
            f"Expected unstable configuration, got stability_eigenvalue={se:.6e}"
        )

    def test_stability_eigenvalue_from_matrix_consistent(self):
        """stability_eigenvalue and stability_eigenvalue_from_matrix agree."""
        n, p, q, eps = 20, 1, 2, 1.0
        D = build_diff_matrix_rbf(n, p, q, eps, kernel="gaussian", nu=1, nextra=0)
        se_direct = stability_eigenvalue(n, p, q, eps, kernel="gaussian",
                                         nu=1, nextra=0)
        se_from_mat = stability_eigenvalue_from_matrix(D)
        assert abs(se_direct - se_from_mat) < 1e-14, (
            f"Mismatch: direct={se_direct:.6e}, from_matrix={se_from_mat:.6e}"
        )

    def test_e2_stable(self):
        """E2 (p=1) with standard parameters should be stable.

        E2 is known to be unconditionally stable with reasonable boundary
        closures.
        """
        for n in [20, 40, 80]:
            se = stability_eigenvalue(n, p=1, q=2, epsilon=1.0,
                                      kernel="gaussian", nu=1, nextra=0)
            assert se < STABILITY_TOL, (
                f"E2 n={n}: expected stable, got stability_eigenvalue={se:.6e}"
            )


# ---------------------------------------------------------------------------
# 32.2a: E2 corrected sweep (PHS/Gaussian/MQ) with stability_eigenvalue
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCorrectedSweepE2:
    """Re-run Phase 29 E2 sweeps with the corrected stability metric.

    Uses stability_eigenvalue (inflow-Dirichlet BC, eigenvalues of -D)
    instead of the raw max Re(eig(D)) used in Phase 29.

    E2_1 parameters: p=1, q=1, nextra=1.
    """

    # E2_1 parameters
    P = 1
    Q = 1
    NEXTRA = 1
    NU = 1

    def _sweep_stability(self, kernel: str, n_values, epsilons):
        """Run epsilon sweep using stability_eigenvalue.

        Returns dict {n: list of (eps, se)} where se = max Re(eig(-D_bc)).
        """
        results = {}
        for n in n_values:
            rows = []
            for eps in epsilons:
                se = stability_eigenvalue(
                    n, p=self.P, q=self.Q, epsilon=eps,
                    kernel=kernel, nu=self.NU, nextra=self.NEXTRA,
                )
                rows.append((eps, se))
            results[n] = rows
        return results

    def _print_table(self, label, results):
        """Print formatted sweep table with stability classification."""
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        for n, rows in sorted(results.items()):
            print(f"\n  n = {n}")
            print(f"  {'epsilon':>10s}  {'stab_eig':>14s}  {'status':>10s}")
            print(f"  {'-'*10}  {'-'*14}  {'-'*10}")
            for eps, se in rows:
                status = "STABLE" if se < STABILITY_TOL else "unstable"
                print(f"  {eps:10.4f}  {se:14.6e}  {status:>10s}")

        # Summary: best epsilon per n (most negative stability eigenvalue = most stable)
        print(f"\n  --- Best epsilon (min stability eigenvalue) ---")
        for n, rows in sorted(results.items()):
            best = min(rows, key=lambda r: r[1])
            stable = "STABLE" if best[1] < STABILITY_TOL else "unstable"
            print(f"  n={n:3d}: eps={best[0]:.4f}, stab_eig={best[1]:.6e} [{stable}]")

    def test_gaussian_sweep(self):
        """Sweep Gaussian kernel epsilon for E2_1 with corrected stability."""
        epsilons = np.logspace(np.log10(0.01), np.log10(10), 60)
        n_values = [20, 40, 80]
        results = self._sweep_stability("gaussian", n_values, epsilons)
        self._print_table(
            "E2_1 Gaussian — Corrected Stability (p=1, q=1, nextra=1)", results
        )

        # Report stable epsilon ranges
        for n, rows in results.items():
            stable_eps = [eps for eps, se in rows if se < STABILITY_TOL]
            if stable_eps:
                print(f"\n  n={n}: {len(stable_eps)}/{len(rows)} stable, "
                      f"eps range [{min(stable_eps):.4f}, {max(stable_eps):.4f}]")
            else:
                best = min(rows, key=lambda r: r[1])
                print(f"\n  n={n}: no stable epsilon found, "
                      f"best stab_eig={best[1]:.6e} at eps={best[0]:.4f}")

        # Assert at least 55/60 epsilons stable at each grid size (57/60 observed)
        for n, rows in results.items():
            n_stable = sum(1 for _, se in rows if se < STABILITY_TOL)
            assert n_stable >= 55, (
                f"n={n}: expected >=55/60 stable Gaussian epsilons, got {n_stable}"
            )

    def test_multiquadric_sweep(self):
        """Sweep Multiquadric kernel epsilon for E2_1 with corrected stability."""
        epsilons = np.logspace(np.log10(0.01), np.log10(10), 60)
        n_values = [20, 40, 80]
        results = self._sweep_stability("multiquadric", n_values, epsilons)
        self._print_table(
            "E2_1 Multiquadric — Corrected Stability (p=1, q=1, nextra=1)", results
        )

        for n, rows in results.items():
            stable_eps = [eps for eps, se in rows if se < STABILITY_TOL]
            if stable_eps:
                print(f"\n  n={n}: {len(stable_eps)}/{len(rows)} stable, "
                      f"eps range [{min(stable_eps):.4f}, {max(stable_eps):.4f}]")
            else:
                best = min(rows, key=lambda r: r[1])
                print(f"\n  n={n}: no stable epsilon found, "
                      f"best stab_eig={best[1]:.6e} at eps={best[0]:.4f}")

        # Assert all 60 epsilons stable at each grid size (60/60 observed)
        for n, rows in results.items():
            n_stable = sum(1 for _, se in rows if se < STABILITY_TOL)
            assert n_stable == 60, (
                f"n={n}: expected 60/60 stable MQ epsilons, got {n_stable}"
            )

    def test_phs_k2_baseline(self):
        """PHS k=2 baseline for E2_1 with corrected stability.

        PHS k=2 is equivalent to tension at σ=0.  Build the matrix via the
        symbolic PHS weights and check with stability_eigenvalue_from_matrix.
        """
        from sympy import cancel as sym_cancel

        from stencil_gen.interior import derive_interior, full_gamma_array
        from stencil_gen.phs import uniform_boundary_weights

        p, q, nextra, nu = self.P, self.Q, self.NEXTRA, self.NU
        t = p + q + 1 + nextra  # boundary stencil width
        r = q + 1 + nextra       # number of boundary rows per side

        interior_coeffs = derive_interior(0, p, nu)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        print(f"\n  E2_1 PHS k=2 Corrected Stability (p={p}, q={q}, nextra={nextra})")
        print(f"  {'n':>6s}  {'stab_eig':>14s}  {'status':>10s}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*10}")

        for n in [20, 40, 80]:
            D = np.zeros((n, n))

            # Left boundary rows
            for i in range(r):
                w_sym = uniform_boundary_weights(i, t, nu, 2, q)
                w = [float(sym_cancel(c)) for c in w_sym]
                for j in range(t):
                    D[i, j] = w[j]

            # Interior rows
            for i in range(r, n - r):
                for k_idx, j in enumerate(range(i - p, i + p + 1)):
                    D[i, j] = interior_w[k_idx]

            # Right boundary (antisymmetric for nu=1)
            sign = (-1.0) ** nu
            for i in range(r):
                w_sym = uniform_boundary_weights(i, t, nu, 2, q)
                w = [float(sym_cancel(c)) for c in w_sym]
                row = n - 1 - i
                for j in range(t):
                    D[row, n - 1 - j] = sign * w[j]

            se = stability_eigenvalue_from_matrix(D)
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"  {n:6d}  {se:14.6e}  {status:>10s}")
            assert se < STABILITY_TOL, (
                f"E2 PHS k=2 should be stable at n={n}, got stab_eig={se:.6e}"
            )

    def test_gaussian_fine_sweep(self):
        """Fine sweep around best Gaussian epsilon with corrected stability.

        Uses n=40 for coarse pass, then refines around minimum.
        """
        n = 40
        # Coarse sweep
        epsilons_coarse = np.logspace(np.log10(0.01), np.log10(10), 60)
        coarse = []
        for eps in epsilons_coarse:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            coarse.append((eps, se))

        best_coarse = min(coarse, key=lambda r: r[1])
        eps_best = best_coarse[0]

        # Fine sweep: ±1 decade around best
        lo = max(0.001, eps_best / 10)
        hi = min(100, eps_best * 10)
        epsilons_fine = np.linspace(lo, hi, 200)
        fine = []
        for eps in epsilons_fine:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            fine.append((eps, se))

        best_fine = min(fine, key=lambda r: r[1])
        stable = best_fine[1] < STABILITY_TOL
        print(f"\n  E2_1 Gaussian corrected fine sweep (n={n}):")
        print(f"  Coarse best: eps={best_coarse[0]:.6f}, stab_eig={best_coarse[1]:.6e}")
        print(f"  Fine best:   eps={best_fine[0]:.6f}, stab_eig={best_fine[1]:.6e}")
        print(f"  Stable: {stable}")

        # Verify best eps* is stable at multiple grid sizes
        eps_star = best_fine[0]
        print(f"\n  Checking eps*={eps_star:.6f} across grid sizes:")
        for nn in [20, 40, 80, 160]:
            se = stability_eigenvalue(
                nn, p=self.P, q=self.Q, epsilon=eps_star,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"    n={nn:4d}: stab_eig={se:.6e} [{status}]")
            assert se < STABILITY_TOL, (
                f"E2 Gaussian best eps*={eps_star:.6f} should be stable at n={nn}, "
                f"got stab_eig={se:.6e}"
            )


# ---------------------------------------------------------------------------
# 32.2b: E4 corrected sweep (PHS/Gaussian/MQ)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCorrectedSweepE4:
    """Re-run Phase 29 E4 sweeps with the corrected stability metric.

    Uses stability_eigenvalue (inflow-Dirichlet BC, eigenvalues of -D)
    instead of the raw max Re(eig(D)) used in Phase 29.

    E4_1 parameters: p=2, q=3, nextra=0.

    Key question: is the O(1e-5) instability "floor" from Phase 29 still
    present, or was it entirely an artifact of the wrong BC/sign convention?
    """

    # E4_1 parameters
    P = 2
    Q = 3
    NEXTRA = 0
    NU = 1

    def _sweep_stability(self, kernel: str, n_values, epsilons):
        """Run epsilon sweep using stability_eigenvalue.

        Returns dict {n: list of (eps, se)} where se = max Re(eig(-D_bc)).
        """
        results = {}
        for n in n_values:
            rows = []
            for eps in epsilons:
                se = stability_eigenvalue(
                    n, p=self.P, q=self.Q, epsilon=eps,
                    kernel=kernel, nu=self.NU, nextra=self.NEXTRA,
                )
                rows.append((eps, se))
            results[n] = rows
        return results

    def _print_table(self, label, results):
        """Print formatted sweep table with stability classification."""
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        for n, rows in sorted(results.items()):
            print(f"\n  n = {n}")
            print(f"  {'epsilon':>10s}  {'stab_eig':>14s}  {'status':>10s}")
            print(f"  {'-'*10}  {'-'*14}  {'-'*10}")
            for eps, se in rows:
                status = "STABLE" if se < STABILITY_TOL else "unstable"
                print(f"  {eps:10.4f}  {se:14.6e}  {status:>10s}")

        # Summary: best epsilon per n
        print(f"\n  --- Best epsilon (min stability eigenvalue) ---")
        for n, rows in sorted(results.items()):
            best = min(rows, key=lambda r: r[1])
            stable = "STABLE" if best[1] < STABILITY_TOL else "unstable"
            print(f"  n={n:3d}: eps={best[0]:.4f}, stab_eig={best[1]:.6e} [{stable}]")

    def test_gaussian_sweep(self):
        """Sweep Gaussian kernel epsilon for E4_1 with corrected stability."""
        epsilons = np.logspace(np.log10(0.01), np.log10(10), 60)
        n_values = [20, 40, 80]
        results = self._sweep_stability("gaussian", n_values, epsilons)
        self._print_table(
            "E4_1 Gaussian — Corrected Stability (p=2, q=3, nextra=0)", results
        )

        for n, rows in results.items():
            stable_eps = [eps for eps, se in rows if se < STABILITY_TOL]
            n_stable = len(stable_eps)
            if stable_eps:
                print(f"\n  n={n}: {n_stable}/{len(rows)} stable, "
                      f"eps range [{min(stable_eps):.4f}, {max(stable_eps):.4f}]")
            else:
                best = min(rows, key=lambda r: r[1])
                print(f"\n  n={n}: no stable epsilon found, "
                      f"best stab_eig={best[1]:.6e} at eps={best[0]:.4f}")

        # Gaussian E4 has a narrow stable band (8/60 observed at each n).
        # Assert at least 5/60 stable — confirms genuine stability exists.
        for n, rows in results.items():
            n_stable = sum(1 for _, se in rows if se < STABILITY_TOL)
            assert n_stable >= 5, (
                f"n={n}: expected >=5/60 stable Gaussian epsilons, got {n_stable}"
            )
        # Best epsilon should be genuinely stable (negative)
        for n, rows in results.items():
            best_se = min(se for _, se in rows)
            assert best_se < -1e-6, (
                f"n={n}: best Gaussian stab_eig={best_se:.6e} should be clearly negative"
            )

    def test_multiquadric_sweep(self):
        """Sweep Multiquadric kernel epsilon for E4_1 with corrected stability."""
        epsilons = np.logspace(np.log10(0.01), np.log10(10), 60)
        n_values = [20, 40, 80]
        results = self._sweep_stability("multiquadric", n_values, epsilons)
        self._print_table(
            "E4_1 Multiquadric — Corrected Stability (p=2, q=3, nextra=0)", results
        )

        for n, rows in results.items():
            stable_eps = [eps for eps, se in rows if se < STABILITY_TOL]
            n_stable = len(stable_eps)
            if stable_eps:
                print(f"\n  n={n}: {n_stable}/{len(rows)} stable, "
                      f"eps range [{min(stable_eps):.4f}, {max(stable_eps):.4f}]")
            else:
                best = min(rows, key=lambda r: r[1])
                print(f"\n  n={n}: no stable epsilon found, "
                      f"best stab_eig={best[1]:.6e} at eps={best[0]:.4f}")

        # MQ E4 has broad stable region (21-24/60 observed). Assert >=15/60.
        for n, rows in results.items():
            n_stable = sum(1 for _, se in rows if se < STABILITY_TOL)
            assert n_stable >= 15, (
                f"n={n}: expected >=15/60 stable MQ epsilons, got {n_stable}"
            )

    def test_phs_k2_baseline(self):
        """PHS k=2 baseline for E4_1 with corrected stability.

        Build matrix from symbolic PHS weights and check with
        stability_eigenvalue_from_matrix.
        """
        from sympy import cancel as sym_cancel

        from stencil_gen.interior import derive_interior, full_gamma_array
        from stencil_gen.phs import uniform_boundary_weights

        p, q, nextra, nu = self.P, self.Q, self.NEXTRA, self.NU
        t = p + q + 1 + nextra  # boundary stencil width
        r = q + 1 + nextra       # number of boundary rows per side

        interior_coeffs = derive_interior(0, p, nu)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        print(f"\n  E4_1 PHS k=2 Corrected Stability (p={p}, q={q}, nextra={nextra})")
        print(f"  {'n':>6s}  {'stab_eig':>14s}  {'status':>10s}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*10}")

        for n in [20, 40, 80]:
            D = np.zeros((n, n))

            # Left boundary rows
            for i in range(r):
                w_sym = uniform_boundary_weights(i, t, nu, 2, q)
                w = [float(sym_cancel(c)) for c in w_sym]
                for j in range(t):
                    D[i, j] = w[j]

            # Interior rows
            for i in range(r, n - r):
                for k_idx, j in enumerate(range(i - p, i + p + 1)):
                    D[i, j] = interior_w[k_idx]

            # Right boundary (antisymmetric for nu=1)
            sign = (-1.0) ** nu
            for i in range(r):
                w_sym = uniform_boundary_weights(i, t, nu, 2, q)
                w = [float(sym_cancel(c)) for c in w_sym]
                row = n - 1 - i
                for j in range(t):
                    D[row, n - 1 - j] = sign * w[j]

            se = stability_eigenvalue_from_matrix(D)
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"  {n:6d}  {se:14.6e}  {status:>10s}")
            assert se < STABILITY_TOL, (
                f"E4 PHS k=2 should be stable at n={n}, got stab_eig={se:.6e}"
            )

    def test_gaussian_fine_sweep(self):
        """Fine sweep around best Gaussian epsilon with corrected stability.

        Uses n=40 for coarse pass, then refines around minimum.
        """
        n = 40
        # Coarse sweep
        epsilons_coarse = np.logspace(np.log10(0.01), np.log10(10), 60)
        coarse = []
        for eps in epsilons_coarse:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            coarse.append((eps, se))

        best_coarse = min(coarse, key=lambda r: r[1])
        eps_best = best_coarse[0]

        # Fine sweep: ±1 decade around best
        lo = max(0.001, eps_best / 10)
        hi = min(100, eps_best * 10)
        epsilons_fine = np.linspace(lo, hi, 200)
        fine = []
        for eps in epsilons_fine:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            fine.append((eps, se))

        best_fine = min(fine, key=lambda r: r[1])
        stable = best_fine[1] < STABILITY_TOL
        print(f"\n  E4_1 Gaussian corrected fine sweep (n={n}):")
        print(f"  Coarse best: eps={best_coarse[0]:.6f}, stab_eig={best_coarse[1]:.6e}")
        print(f"  Fine best:   eps={best_fine[0]:.6f}, stab_eig={best_fine[1]:.6e}")
        print(f"  Stable: {stable}")

        assert stable, (
            f"E4 Gaussian fine sweep best eps={best_fine[0]:.6f} should be stable, "
            f"got stab_eig={best_fine[1]:.6e}"
        )

        # Verify best eps* is stable at multiple grid sizes
        eps_star = best_fine[0]
        print(f"\n  Checking eps*={eps_star:.6f} across grid sizes:")
        for nn in [20, 40, 80, 160]:
            se = stability_eigenvalue(
                nn, p=self.P, q=self.Q, epsilon=eps_star,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"    n={nn:4d}: stab_eig={se:.6e} [{status}]")
            assert se < STABILITY_TOL, (
                f"E4 Gaussian best eps*={eps_star:.6f} should be stable at n={nn}, "
                f"got stab_eig={se:.6e}"
            )


# ---------------------------------------------------------------------------
# 32.3a: E2 corrected tension sweep
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCorrectedTensionE2:
    """Re-run Phase 30 E2 tension sweeps with the corrected stability metric.

    Uses stability_eigenvalue (inflow-Dirichlet BC, eigenvalues of -D)
    instead of the raw max Re(eig(D)) used in Phase 30.

    E2_1 parameters: p=1, q=1, nextra=1.
    Sweeps tension parameter σ over [0, 20].
    """

    # E2_1 parameters
    P = 1
    Q = 1
    NEXTRA = 1
    NU = 1

    def _sweep_stability(self, n_values, sigmas):
        """Run sigma sweep using stability_eigenvalue.

        Returns dict {n: list of (sigma, se)} where se = max Re(eig(-D_bc)).
        """
        results = {}
        for n in n_values:
            rows = []
            for sigma in sigmas:
                se = stability_eigenvalue(
                    n, p=self.P, q=self.Q, epsilon=sigma,
                    kernel="tension", nu=self.NU, nextra=self.NEXTRA,
                )
                rows.append((sigma, se))
            results[n] = rows
        return results

    def _print_table(self, label, results):
        """Print formatted sweep table with stability classification."""
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        for n, rows in sorted(results.items()):
            print(f"\n  n = {n}")
            print(f"  {'sigma':>10s}  {'stab_eig':>14s}  {'status':>10s}")
            print(f"  {'-'*10}  {'-'*14}  {'-'*10}")
            for sigma, se in rows:
                status = "STABLE" if se < STABILITY_TOL else "unstable"
                print(f"  {sigma:10.4f}  {se:14.6e}  {status:>10s}")

        # Summary: best sigma per n
        print(f"\n  --- Best sigma (min stability eigenvalue) ---")
        for n, rows in sorted(results.items()):
            best = min(rows, key=lambda r: r[1])
            stable = "STABLE" if best[1] < STABILITY_TOL else "unstable"
            print(f"  n={n:3d}: σ={best[0]:.4f}, stab_eig={best[1]:.6e} [{stable}]")

    def test_tension_coarse_sweep(self):
        """Coarse sweep of σ over [0, 20] for E2_1 with corrected stability."""
        sigmas = np.concatenate([[0.0], np.logspace(np.log10(0.01), np.log10(20), 60)])
        n_values = [20, 40, 80]
        results = self._sweep_stability(n_values, sigmas)
        self._print_table(
            "E2_1 Tension — Corrected Stability (p=1, q=1, nextra=1)", results
        )

        # Report stable sigma ranges
        for n, rows in results.items():
            stable_sigmas = [s for s, se in rows if se < STABILITY_TOL]
            n_stable = len(stable_sigmas)
            if stable_sigmas:
                print(f"\n  n={n}: {n_stable}/{len(rows)} stable, "
                      f"σ range [{min(stable_sigmas):.4f}, {max(stable_sigmas):.4f}]")
            else:
                best = min(rows, key=lambda r: r[1])
                print(f"\n  n={n}: no stable σ found, "
                      f"best stab_eig={best[1]:.6e} at σ={best[0]:.4f}")

        # E2 was universally stable under corrected test (Phase 32.2a).
        # Tension (being a generalization of PHS k=2) should also be stable
        # across essentially the entire σ range.
        for n, rows in results.items():
            n_stable = sum(1 for _, se in rows if se < STABILITY_TOL)
            assert n_stable >= 55, (
                f"n={n}: expected >=55/61 stable tension sigmas, got {n_stable}"
            )

    def test_tension_fine_sweep(self):
        """Fine sweep around best σ from coarse sweep with corrected stability.

        Uses n=40 for the coarse pass, then refines around minimum.
        """
        n = 40
        # Coarse sweep (include σ=0)
        sigmas_coarse = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(20), 60)]
        )
        coarse = []
        for sigma in sigmas_coarse:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            coarse.append((sigma, se))

        best_coarse = min(coarse, key=lambda r: r[1])
        sigma_best = best_coarse[0]

        # Fine sweep: ±factor around best (or [0, 2] if best is near 0)
        if sigma_best < 0.1:
            lo, hi = 0.0, 2.0
        else:
            lo = max(0.0, sigma_best / 5)
            hi = sigma_best * 5
        sigmas_fine = np.linspace(lo, hi, 200)
        fine = []
        for sigma in sigmas_fine:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            fine.append((sigma, se))

        best_fine = min(fine, key=lambda r: r[1])
        stable = best_fine[1] < STABILITY_TOL
        print(f"\n  E2_1 Tension corrected fine sweep (n={n}):")
        print(f"  Coarse best: σ={best_coarse[0]:.6f}, stab_eig={best_coarse[1]:.6e}")
        print(f"  Fine best:   σ={best_fine[0]:.6f}, stab_eig={best_fine[1]:.6e}")
        print(f"  Stable: {stable}")

        assert stable, (
            f"E2 tension fine sweep best σ={best_fine[0]:.6f} should be stable, "
            f"got stab_eig={best_fine[1]:.6e}"
        )

        # Verify best σ* is stable at multiple grid sizes
        sigma_star = best_fine[0]
        print(f"\n  Checking σ*={sigma_star:.6f} across grid sizes:")
        for nn in [20, 40, 80, 160]:
            se = stability_eigenvalue(
                nn, p=self.P, q=self.Q, epsilon=sigma_star,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"    n={nn:4d}: stab_eig={se:.6e} [{status}]")
            assert se < STABILITY_TOL, (
                f"E2 tension σ*={sigma_star:.6f} should be stable at n={nn}, "
                f"got stab_eig={se:.6e}"
            )

    def test_compare_with_gaussian(self):
        """Compare tension best σ with Gaussian best ε for E2_1.

        Both should be comfortably stable under the corrected test.
        """
        n = 40
        # Tension sweep
        sigmas = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(20), 100)]
        )
        tension_results = []
        for sigma in sigmas:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            tension_results.append((sigma, se))

        # Gaussian sweep (same range for comparison)
        epsilons = np.logspace(np.log10(0.01), np.log10(20), 100)
        gaussian_results = []
        for eps in epsilons:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            gaussian_results.append((eps, se))

        best_tension = min(tension_results, key=lambda r: r[1])
        best_gaussian = min(gaussian_results, key=lambda r: r[1])

        print(f"\n  E2_1 Corrected Comparison (n={n}):")
        print(f"  {'Method':>15s}  {'param':>10s}  {'stab_eig':>14s}  {'status':>10s}")
        print(f"  {'-'*15}  {'-'*10}  {'-'*14}  {'-'*10}")

        t_stable = "STABLE" if best_tension[1] < STABILITY_TOL else "unstable"
        g_stable = "STABLE" if best_gaussian[1] < STABILITY_TOL else "unstable"
        print(f"  {'Tension':>15s}  {best_tension[0]:10.4f}  {best_tension[1]:14.6e}  {t_stable:>10s}")
        print(f"  {'Gaussian':>15s}  {best_gaussian[0]:10.4f}  {best_gaussian[1]:14.6e}  {g_stable:>10s}")

        # PHS k=2 (σ=0) for reference
        phs_se = tension_results[0][1]  # σ=0 entry
        phs_stable = "STABLE" if phs_se < STABILITY_TOL else "unstable"
        print(f"  {'PHS k=2 (σ=0)':>15s}  {'0.0':>10s}  {phs_se:14.6e}  {phs_stable:>10s}")

        # Both tension and Gaussian should be stable for E2 under corrected test
        assert best_tension[1] < STABILITY_TOL, (
            f"E2 tension best not stable: stab_eig = {best_tension[1]:.6e}"
        )
        assert best_gaussian[1] < STABILITY_TOL, (
            f"E2 Gaussian best not stable: stab_eig = {best_gaussian[1]:.6e}"
        )


# ---------------------------------------------------------------------------
# 32.3b: E4 corrected tension sweep
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCorrectedTensionE4:
    """Re-run Phase 30 E4 tension sweeps with the corrected stability metric.

    Uses stability_eigenvalue (inflow-Dirichlet BC, eigenvalues of -D)
    instead of the raw max Re(eig(D)) used in Phase 30.

    E4_1 parameters: p=2, q=3, nextra=0.
    Sweeps tension parameter σ over [0, 20].

    Key question: with the correct BC and sign convention, is E4 tension
    stable across a broad σ range (like E2), or does it have a narrow
    stable band (like E4 Gaussian)?
    """

    # E4_1 parameters
    P = 2
    Q = 3
    NEXTRA = 0
    NU = 1

    def _sweep_stability(self, n_values, sigmas):
        """Run sigma sweep using stability_eigenvalue.

        Returns dict {n: list of (sigma, se)} where se = max Re(eig(-D_bc)).
        """
        results = {}
        for n in n_values:
            rows = []
            for sigma in sigmas:
                se = stability_eigenvalue(
                    n, p=self.P, q=self.Q, epsilon=sigma,
                    kernel="tension", nu=self.NU, nextra=self.NEXTRA,
                )
                rows.append((sigma, se))
            results[n] = rows
        return results

    def _print_table(self, label, results):
        """Print formatted sweep table with stability classification."""
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        for n, rows in sorted(results.items()):
            print(f"\n  n = {n}")
            print(f"  {'sigma':>10s}  {'stab_eig':>14s}  {'status':>10s}")
            print(f"  {'-'*10}  {'-'*14}  {'-'*10}")
            for sigma, se in rows:
                status = "STABLE" if se < STABILITY_TOL else "unstable"
                print(f"  {sigma:10.4f}  {se:14.6e}  {status:>10s}")

        # Summary: best sigma per n
        print(f"\n  --- Best sigma (min stability eigenvalue) ---")
        for n, rows in sorted(results.items()):
            best = min(rows, key=lambda r: r[1])
            stable = "STABLE" if best[1] < STABILITY_TOL else "unstable"
            print(f"  n={n:3d}: σ={best[0]:.4f}, stab_eig={best[1]:.6e} [{stable}]")

    def test_tension_coarse_sweep(self):
        """Coarse sweep of σ over [0, 20] for E4_1 with corrected stability."""
        sigmas = np.concatenate([[0.0], np.logspace(np.log10(0.01), np.log10(20), 60)])
        n_values = [20, 40, 80]
        results = self._sweep_stability(n_values, sigmas)
        self._print_table(
            "E4_1 Tension — Corrected Stability (p=2, q=3, nextra=0)", results
        )

        # Report stable sigma ranges
        for n, rows in results.items():
            stable_sigmas = [s for s, se in rows if se < STABILITY_TOL]
            n_stable = len(stable_sigmas)
            if stable_sigmas:
                print(f"\n  n={n}: {n_stable}/{len(rows)} stable, "
                      f"σ range [{min(stable_sigmas):.4f}, {max(stable_sigmas):.4f}]")
            else:
                best = min(rows, key=lambda r: r[1])
                print(f"\n  n={n}: no stable σ found, "
                      f"best stab_eig={best[1]:.6e} at σ={best[0]:.4f}")

        # PHS k=2 (σ=0) should be stable (confirmed in TestCorrectedSweepE4)
        for n, rows in results.items():
            sigma0 = [r for r in rows if r[0] == 0.0][0]
            assert sigma0[1] < STABILITY_TOL, (
                f"n={n}: PHS k=2 (σ=0) should be stable, got stab_eig={sigma0[1]:.6e}"
            )

        # Best sigma should be clearly stable (negative eigenvalue)
        for n, rows in results.items():
            best_se = min(se for _, se in rows)
            assert best_se < -1e-6, (
                f"n={n}: best tension stab_eig={best_se:.6e} should be clearly negative"
            )

        # Count assertions: E4 tension has broad stability across σ range
        for n, rows in results.items():
            n_stable = sum(1 for _, se in rows if se < STABILITY_TOL)
            if n <= 40:
                assert n_stable >= 55, (
                    f"n={n}: expected >=55/61 stable tension sigmas, got {n_stable}"
                )
            else:
                # n=80 has a narrow unstable band near σ≈1.0-1.4
                assert n_stable >= 50, (
                    f"n={n}: expected >=50/61 stable tension sigmas, got {n_stable}"
                )

    def test_tension_fine_sweep(self):
        """Fine sweep around best σ from coarse sweep with corrected stability.

        Uses n=40 for the coarse pass, then refines around minimum.
        """
        n = 40
        # Coarse sweep (include σ=0)
        sigmas_coarse = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(20), 60)]
        )
        coarse = []
        for sigma in sigmas_coarse:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            coarse.append((sigma, se))

        best_coarse = min(coarse, key=lambda r: r[1])
        sigma_best = best_coarse[0]

        # Fine sweep: ±factor around best (or [0, 2] if best is near 0)
        if sigma_best < 0.1:
            lo, hi = 0.0, 2.0
        else:
            lo = max(0.0, sigma_best / 5)
            hi = sigma_best * 5
        sigmas_fine = np.linspace(lo, hi, 200)
        fine = []
        for sigma in sigmas_fine:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            fine.append((sigma, se))

        best_fine = min(fine, key=lambda r: r[1])
        stable = best_fine[1] < STABILITY_TOL
        print(f"\n  E4_1 Tension corrected fine sweep (n={n}):")
        print(f"  Coarse best: σ={best_coarse[0]:.6f}, stab_eig={best_coarse[1]:.6e}")
        print(f"  Fine best:   σ={best_fine[0]:.6f}, stab_eig={best_fine[1]:.6e}")
        print(f"  Stable: {stable}")

        assert stable, (
            f"E4 tension fine sweep best σ={best_fine[0]:.6f} should be stable, "
            f"got stab_eig={best_fine[1]:.6e}"
        )

        # Verify best σ* is stable at multiple grid sizes
        sigma_star = best_fine[0]
        print(f"\n  Checking σ*={sigma_star:.6f} across grid sizes:")
        for nn in [20, 40, 80, 160]:
            se = stability_eigenvalue(
                nn, p=self.P, q=self.Q, epsilon=sigma_star,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"    n={nn:4d}: stab_eig={se:.6e} [{status}]")
            assert se < STABILITY_TOL, (
                f"E4 tension σ*={sigma_star:.6f} should be stable at n={nn}, "
                f"got stab_eig={se:.6e}"
            )

    def test_compare_with_gaussian(self):
        """Compare tension best σ with Gaussian best ε for E4_1.

        Both use the corrected stability metric. Tension starts from PHS k=2
        (which is known-stable for E4), so it should maintain a broader stable
        region than Gaussian.
        """
        n = 40
        # Tension sweep
        sigmas = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(20), 100)]
        )
        tension_results = []
        for sigma in sigmas:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            tension_results.append((sigma, se))

        # Gaussian sweep (same range for comparison)
        epsilons = np.logspace(np.log10(0.01), np.log10(20), 100)
        gaussian_results = []
        for eps in epsilons:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            gaussian_results.append((eps, se))

        best_tension = min(tension_results, key=lambda r: r[1])
        best_gaussian = min(gaussian_results, key=lambda r: r[1])

        # Count stable entries
        n_stable_tension = sum(1 for _, se in tension_results if se < STABILITY_TOL)
        n_stable_gaussian = sum(1 for _, se in gaussian_results if se < STABILITY_TOL)

        print(f"\n  E4_1 Corrected Comparison (n={n}):")
        print(f"  {'Method':>15s}  {'param':>10s}  {'stab_eig':>14s}  {'stable_count':>14s}  {'status':>10s}")
        print(f"  {'-'*15}  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*10}")

        t_stable = "STABLE" if best_tension[1] < STABILITY_TOL else "unstable"
        g_stable = "STABLE" if best_gaussian[1] < STABILITY_TOL else "unstable"
        print(f"  {'Tension':>15s}  {best_tension[0]:10.4f}  {best_tension[1]:14.6e}  {n_stable_tension:>14d}  {t_stable:>10s}")
        print(f"  {'Gaussian':>15s}  {best_gaussian[0]:10.4f}  {best_gaussian[1]:14.6e}  {n_stable_gaussian:>14d}  {g_stable:>10s}")

        # PHS k=2 (σ=0) for reference
        phs_se = tension_results[0][1]  # σ=0 entry
        phs_stable = "STABLE" if phs_se < STABILITY_TOL else "unstable"
        print(f"  {'PHS k=2 (σ=0)':>15s}  {'0.0':>10s}  {phs_se:14.6e}  {'—':>14s}  {phs_stable:>10s}")

        # Both should have at least some stable configurations
        assert best_tension[1] < STABILITY_TOL, (
            f"E4 tension best not stable: stab_eig = {best_tension[1]:.6e}"
        )
        assert best_gaussian[1] < STABILITY_TOL, (
            f"E4 Gaussian best not stable: stab_eig = {best_gaussian[1]:.6e}"
        )


# ---------------------------------------------------------------------------
# 32.3c: E4 tension + conservation penalty with corrected stability test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCorrectedTensionPenaltyE4:
    """Re-run Phase 30.3c joint (σ, γ) sweep for E4 with corrected stability.

    Uses stability_eigenvalue_from_matrix (inflow-Dirichlet BC, eigenvalues
    of -D) instead of the raw max Re(eig(D)) used in Phase 30.

    E4_1 parameters: p=2, q=3, nextra=0.
    Sweeps tension parameter σ and conservation penalty γ jointly.

    Key question: with the correct stability metric, does the penalty (γ > 0)
    improve on tension-only (γ = 0)?  Phase 32.3b showed tension-only is
    already broadly stable — does the penalty help or hurt?
    """

    P, Q, NEXTRA, NU = 2, 3, 0, 1

    def _eval_point(self, n, sigma, gamma):
        """Evaluate (σ, γ) point with corrected stability metric.

        Returns (stab_eig, deficit) where stab_eig = max Re(eig(-D_bc)).
        """
        D = build_diff_matrix_rbf_penalty(
            n, self.P, self.Q, sigma, "tension", self.NU, self.NEXTRA,
            gamma=gamma,
        )
        se = stability_eigenvalue_from_matrix(D)
        deficit = float(np.max(np.abs(np.sum(D, axis=0))))
        return se, deficit

    def test_joint_sweep_coarse(self):
        """Coarse 2D sweep over σ × γ for E4_1 with corrected stability.

        Phase 30.3c used σ ∈ [5, 55] (the old optimal range under wrong metric).
        Phase 32.3b found E4 tension stable across [0, 20] at n=20,40.
        Use σ ∈ [0, 20] (matching 32.3b) with γ ∈ [0, 100].
        """
        n = 40
        sigmas = np.concatenate([[0.0], np.logspace(np.log10(0.01), np.log10(20), 24)])
        gammas = np.concatenate([[0.0], np.logspace(-1, 2, 24)])  # 0 + log[0.1..100]

        best_se = float("inf")
        best_sigma = None
        best_gamma = None
        best_deficit = None

        # Track γ=0 baseline
        baseline_se = float("inf")

        # Track best among γ > 0
        best_se_gamma_pos = float("inf")

        for sigma in sigmas:
            for gamma in gammas:
                se, deficit = self._eval_point(n, sigma, gamma)

                if se < best_se:
                    best_se = se
                    best_sigma = sigma
                    best_gamma = gamma
                    best_deficit = deficit

                if gamma == 0.0 and se < baseline_se:
                    baseline_se = se

                if gamma > 0.0 and se < best_se_gamma_pos:
                    best_se_gamma_pos = se

        print(f"\n  E4_1 Corrected Joint (σ, γ) Sweep (n={n})")
        print(f"  Grid: {len(sigmas)} σ × {len(gammas)} γ = "
              f"{len(sigmas) * len(gammas)} points")

        print(f"\n  Best (σ, γ) point (lowest stab_eig):")
        print(f"    σ*={best_sigma:.4f}, γ*={best_gamma:.4f}")
        print(f"    stab_eig={best_se:.6e}")
        print(f"    deficit={best_deficit:.6e}")

        print(f"\n  Baseline (γ=0) best stab_eig: {baseline_se:.6e}")
        print(f"  Best γ>0 stab_eig: {best_se_gamma_pos:.6e}")

        stable = best_se < STABILITY_TOL
        baseline_stable = baseline_se < STABILITY_TOL
        print(f"\n  Overall best stable: {stable}")
        print(f"  Baseline (γ=0) stable: {baseline_stable}")

        # γ=0 baseline should be stable (confirmed in 32.3b)
        assert baseline_se < STABILITY_TOL, (
            f"E4 tension γ=0 baseline should be stable, got stab_eig={baseline_se:.6e}"
        )

        # Overall best should also be stable
        assert best_se < STABILITY_TOL, (
            f"E4 tension+penalty best should be stable, got stab_eig={best_se:.6e}"
        )

    def test_penalty_effect_at_optimal_sigma(self):
        """Check how γ affects corrected stability at σ* near PHS k=2 (σ=0).

        Phase 32.3b found σ=0 (PHS k=2) is optimal for E4 at n=40.
        Sweep γ at σ=0 to check whether penalty helps or hurts.
        """
        n = 40
        sigma_star = 0.0  # PHS k=2, optimal from 32.3b
        gammas = np.concatenate([[0.0], np.logspace(-2, 3, 50)])

        print(f"\n  E4_1 Corrected Penalty Effect at σ*={sigma_star} (n={n})")
        print(f"  {'γ':>10s}  {'stab_eig':>14s}  {'deficit':>14s}  {'status':>10s}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*10}")

        baseline_se = None
        best_se = float("inf")
        best_gamma = None

        for gamma in gammas:
            se, deficit = self._eval_point(n, sigma_star, gamma)
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            # Print representative subset
            if (gamma == 0.0 or gamma < 0.02
                    or abs(gamma - 1.0) < 0.3 or abs(gamma - 10.0) < 2.0
                    or abs(gamma - 100.0) < 20.0 or gamma > 500.0):
                print(f"  {gamma:10.4f}  {se:14.6e}  {deficit:14.6e}  {status:>10s}")

            if gamma == 0.0:
                baseline_se = se
            if se < best_se:
                best_se = se
                best_gamma = gamma

        print(f"\n  Baseline (γ=0): stab_eig={baseline_se:.6e}")
        print(f"  Best γ={best_gamma:.4f}: stab_eig={best_se:.6e}")

        # Baseline should be stable (PHS k=2 confirmed in 32.3b)
        assert baseline_se < STABILITY_TOL, (
            f"PHS k=2 should be stable, got stab_eig={baseline_se:.6e}"
        )

        # Penalty should not destroy stability (best should still be stable)
        assert best_se < STABILITY_TOL, (
            f"Best (σ=0, γ={best_gamma}) should be stable, got stab_eig={best_se:.6e}"
        )

    def test_grid_independence(self):
        """Verify corrected (σ*, γ*) stability across grid sizes.

        Pick the best (σ, γ) from coarse sweep at n=40 and check n=20,40,80.
        """
        n_opt = 40
        sigmas = np.concatenate([[0.0], np.logspace(np.log10(0.01), np.log10(20), 24)])
        gammas = np.concatenate([[0.0], np.logspace(-1, 2, 24)])

        best_se = float("inf")
        best_sigma = None
        best_gamma = None

        for sigma in sigmas:
            for gamma in gammas:
                se, _ = self._eval_point(n_opt, sigma, gamma)
                if se < best_se:
                    best_se = se
                    best_sigma = sigma
                    best_gamma = gamma

        print(f"\n  E4_1 Corrected Grid Independence (σ*={best_sigma:.4f}, γ*={best_gamma:.4f})")
        print(f"  Optimized at n={n_opt}, stab_eig={best_se:.6e}")

        print(f"\n  {'n':>6s}  {'stab_eig':>14s}  {'deficit':>14s}  {'status':>10s}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*10}")

        for nn in [20, 40, 80]:
            se, deficit = self._eval_point(nn, best_sigma, best_gamma)
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"  {nn:6d}  {se:14.6e}  {deficit:14.6e}  {status:>10s}")

            # Should be stable at all tested grid sizes.
            # The optimal (σ≈0, γ≈0) = PHS k=2 is solidly stable at n=80
            # (stab_eig≈-5.9e-4).  Narrow unstable bands only affect σ≈1.0-1.4.
            assert se < STABILITY_TOL, (
                f"n={nn}: (σ*={best_sigma:.4f}, γ*={best_gamma:.4f}) should be stable, "
                f"got stab_eig={se:.6e}"
            )


# ---------------------------------------------------------------------------
# 32.4a: Nextra sweep with correct stability test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCorrectedFootprint:
    """Re-run Phase 31 nextra × σ sweep for E4 with corrected stability.

    Uses stability_eigenvalue (inflow-Dirichlet BC, eigenvalues of -D)
    instead of the raw max Re(eig(D)) used in Phase 31.

    Phase 32.3b confirmed E4 at nextra=0 is already broadly stable with PHS k=2
    (σ=0).  This test verifies that finding and checks how nextra affects stability.

    E4 dimensions per nextra:
      nx=0: t=6, r=4, extra DOF/row=2, total=8
      nx=1: t=7, r=5, extra DOF/row=3, total=15
      nx=2: t=8, r=6, extra DOF/row=4, total=24
      nx=3: t=9, r=7, extra DOF/row=5, total=35
    """

    P = 2
    Q = 3
    NU = 1

    def _sweep_stability(self, n, nextra, sigmas):
        """Sweep sigma for given nextra, return list of (sigma, se).

        se = max Re(eig(-D_bc)) via stability_eigenvalue.
        """
        rows = []
        for sigma in sigmas:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=nextra,
            )
            rows.append((sigma, se))
        return rows

    def test_nextra_comparison(self):
        """Compare nextra=0 vs nextra=1 for E4 tension with corrected stability.

        Phase 31.2a found nextra=1 helped under the wrong metric.
        Phase 32.3b found E4 nextra=0 is already stable.
        Confirm: nextra=0 is stable, and nextra=1 is also stable.
        """
        n = 40
        sigmas = np.concatenate([
            [0.0],
            np.logspace(np.log10(0.01), np.log10(50), 80),
        ])

        results_nx0 = self._sweep_stability(n, nextra=0, sigmas=sigmas)
        results_nx1 = self._sweep_stability(n, nextra=1, sigmas=sigmas)

        best_nx0 = min(results_nx0, key=lambda r: r[1])
        best_nx1 = min(results_nx1, key=lambda r: r[1])

        n_stable_nx0 = sum(1 for _, se in results_nx0 if se < STABILITY_TOL)
        n_stable_nx1 = sum(1 for _, se in results_nx1 if se < STABILITY_TOL)

        print(f"\n{'='*72}")
        print(f"  Phase 32.4a: E4 Tension — nextra=0 vs nextra=1 (corrected, n={n})")
        print(f"{'='*72}")
        print(f"\n  nextra=0: t={self.P + self.Q + 1}, r={self.Q + 1}")
        print(f"  nextra=1: t={self.P + self.Q + 2}, r={self.Q + 2}")
        print(f"\n  {'sigma':>10s}  {'nx=0 stab_eig':>16s}  {'nx=1 stab_eig':>16s}")
        print(f"  {'-'*10}  {'-'*16}  {'-'*16}")
        for (s0, se0), (s1, se1) in zip(results_nx0, results_nx1):
            m0 = " *" if se0 < STABILITY_TOL else ""
            m1 = " *" if se1 < STABILITY_TOL else ""
            print(f"  {s0:10.4f}  {se0:14.6e}{m0}  {se1:14.6e}{m1}")

        print(f"\n  --- Summary ---")
        print(f"  nextra=0: best σ={best_nx0[0]:.4f}, stab_eig={best_nx0[1]:.6e}, "
              f"{n_stable_nx0}/{len(sigmas)} stable")
        print(f"  nextra=1: best σ={best_nx1[0]:.4f}, stab_eig={best_nx1[1]:.6e}, "
              f"{n_stable_nx1}/{len(sigmas)} stable")

        # nextra=0 is already stable (confirmed in 32.3b)
        assert best_nx0[1] < STABILITY_TOL, (
            f"E4 nextra=0 should be stable, got stab_eig={best_nx0[1]:.6e}"
        )

        # nextra=1 should also be stable
        assert best_nx1[1] < STABILITY_TOL, (
            f"E4 nextra=1 should be stable, got stab_eig={best_nx1[1]:.6e}"
        )

        # Both should have broad stability (most sigmas stable)
        assert n_stable_nx0 >= 50, (
            f"nextra=0: expected >=50/{len(sigmas)} stable, got {n_stable_nx0}"
        )

    def test_nextra_sweep_e4_tension(self):
        """Full nextra × σ sweep for E4 tension with corrected stability.

        Redo Phase 31.2b with stability_eigenvalue.
        Sweep nextra ∈ {0, 1, 2, 3} × σ ∈ [0, 50] at n=40.
        """
        n = 40
        nextra_values = [0, 1, 2, 3]
        sigmas = np.concatenate([
            [0.0],
            np.logspace(np.log10(0.01), np.log10(50), 100),
        ])

        best_per_nx = {}
        all_results = {}
        stable_counts = {}

        for nx in nextra_values:
            r = self.Q + 1 + nx
            if n < 2 * r:
                print(f"  nextra={nx}: grid too small (n={n} < 2*r={2*r}), skipping")
                continue

            rows = self._sweep_stability(n, nextra=nx, sigmas=sigmas)
            all_results[nx] = rows
            best = min(rows, key=lambda r: r[1])
            best_per_nx[nx] = best
            stable_counts[nx] = sum(1 for _, se in rows if se < STABILITY_TOL)

        # Print results
        print(f"\n{'='*80}")
        print(f"  Phase 32.4a: E4 Tension — Full nextra × σ Sweep (corrected, n={n})")
        print(f"{'='*80}")

        header = f"  {'sigma':>10s}"
        for nx in nextra_values:
            if nx in all_results:
                header += f"  {'nx=' + str(nx):>16s}"
        print(header)
        divider = f"  {'-'*10}"
        for nx in nextra_values:
            if nx in all_results:
                divider += f"  {'-'*16}"
        print(divider)

        # Print every 5th row
        for idx in range(0, len(sigmas), 5):
            line = f"  {sigmas[idx]:10.4f}"
            for nx in nextra_values:
                if nx in all_results:
                    _, se = all_results[nx][idx]
                    marker = " *" if se < STABILITY_TOL else ""
                    line += f"  {se:14.6e}{marker}"
            print(line)

        # Summary table
        print(f"\n  {'='*70}")
        print(f"  Summary: Corrected stability per nextra")
        print(f"  {'='*70}")
        print(f"  {'nextra':>6s}  {'t':>3s}  {'r':>3s}  "
              f"{'extra DOF':>9s}  {'best σ':>10s}  {'stab_eig':>16s}  "
              f"{'stable':>8s}  {'status':>10s}")
        print(f"  {'-'*6}  {'-'*3}  {'-'*3}  "
              f"{'-'*9}  {'-'*10}  {'-'*16}  "
              f"{'-'*8}  {'-'*10}")

        for nx in nextra_values:
            if nx not in best_per_nx:
                continue
            best_sigma, best_se = best_per_nx[nx]
            t = self.P + self.Q + 1 + nx
            r = self.Q + 1 + nx
            extra_dof = r * (self.P + nx)
            status = "STABLE" if best_se < STABILITY_TOL else "unstable"
            sc = stable_counts[nx]
            total = len(all_results[nx])
            print(f"  {nx:6d}  {t:3d}  {r:3d}  "
                  f"{extra_dof:9d}  {best_sigma:10.4f}  {best_se:16.6e}  "
                  f"{sc:>3d}/{total:<3d}  {status:>10s}")

        # Key assertion: nextra=0 is already stable
        assert 0 in best_per_nx, "nextra=0 must be present"
        assert best_per_nx[0][1] < STABILITY_TOL, (
            f"E4 nextra=0 should be stable, got stab_eig={best_per_nx[0][1]:.6e}"
        )

        # All tested nextra values should have at least some stable region
        for nx in nextra_values:
            if nx in best_per_nx:
                assert best_per_nx[nx][1] < STABILITY_TOL, (
                    f"E4 nextra={nx} should have a stable σ, "
                    f"got best stab_eig={best_per_nx[nx][1]:.6e}"
                )

        # Count assertions: nextra=0 is broadest, nextra=1 is narrowest
        min_stable = {0: 90, 1: 50, 2: 70, 3: 90}
        for nx, threshold in min_stable.items():
            if nx in stable_counts:
                assert stable_counts[nx] >= threshold, (
                    f"nextra={nx}: expected >={threshold}/{len(sigmas)} stable, "
                    f"got {stable_counts[nx]}"
                )

    def test_nextra0_grid_independence(self):
        """Verify E4 nextra=0 PHS k=2 stability across grid sizes.

        Confirms the 32.3b result: PHS k=2 (σ=0, nextra=0) is stable at all
        grid sizes.  This is the key result — nextra>0 is not needed for stability.
        """
        sigma = 0.0  # PHS k=2
        print(f"\n  E4 nextra=0, σ={sigma} (PHS k=2) — Grid Independence")
        print(f"  {'n':>6s}  {'stab_eig':>14s}  {'status':>10s}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*10}")

        for nn in [20, 40, 80, 160]:
            se = stability_eigenvalue(
                nn, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=0,
            )
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"  {nn:6d}  {se:14.6e}  {status:>10s}")

            assert se < STABILITY_TOL, (
                f"n={nn}: E4 PHS k=2 (nextra=0) should be stable, "
                f"got stab_eig={se:.6e}"
            )


# ---------------------------------------------------------------------------
# 32.5a: Comprehensive comparison table with corrected stability metric
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCorrectedComparison:
    """Comprehensive comparison of all approaches with corrected stability.

    Phase 32.5a: Side-by-side comparison using stability_eigenvalue
    (inflow-Dirichlet BC, eigenvalues of -D) instead of the raw
    max Re(eig(D)) used in prior Phases 29-31.

    Methods compared:
    1. PHS k=2 (σ=0 baseline)
    2. Gaussian ε* (best from corrected sweeps)
    3. Tension σ* (best from corrected sweeps)
    4. Tension + conservation penalty (σ*, γ*)

    Metrics: stability_eigenvalue, spectral radius, CFL with RK4,
    conservation deficit.

    Production E4u_1 was separately validated as stable in 32.1c
    (E4 tension σ=3.0 stable at n=20,40,80,160).
    """

    # E2_1 parameters
    E2_P, E2_Q, E2_NEXTRA, E2_NU = 1, 1, 1, 1
    # E4_1 parameters
    E4_P, E4_Q, E4_NEXTRA, E4_NU = 2, 3, 0, 1

    RK4_IMAG_LIMIT = 2.828

    # ------------------------------------------------------------------ helpers

    def _corrected_metrics(self, D):
        """Compute (stab_eig, spectral_radius, cfl_rk4, conservation_deficit).

        stab_eig = max Re(eig(-D_bc)) where D_bc = D[1:, 1:] (inflow removed).
        Spectral radius and CFL use the full D matrix.
        """
        stab_eig = stability_eigenvalue_from_matrix(D)
        full_eigs = np.linalg.eigvals(D)
        spec_rad = float(np.max(np.abs(full_eigs)))
        cfl = self.RK4_IMAG_LIMIT / spec_rad if spec_rad > 0 else float("inf")
        deficit = float(np.max(np.abs(np.sum(D, axis=0))))
        return stab_eig, spec_rad, cfl, deficit

    def _find_best_epsilon_corrected(self, n, p, q, nu, nextra, kernel="gaussian"):
        """Coarse + fine sweep for best ε using corrected stability."""
        eps_coarse = np.logspace(np.log10(0.01), np.log10(15), 80)
        best_eps, best_se = None, float("inf")
        for e in eps_coarse:
            se = stability_eigenvalue(n, p, q, e, kernel, nu, nextra)
            if se < best_se:
                best_se = se
                best_eps = e

        lo = max(0.001, best_eps / 3)
        hi = min(50.0, best_eps * 3)
        for e in np.linspace(lo, hi, 200):
            se = stability_eigenvalue(n, p, q, e, kernel, nu, nextra)
            if se < best_se:
                best_se = se
                best_eps = e

        return best_eps, best_se

    def _find_best_sigma_corrected(self, n, p, q, nu, nextra):
        """Coarse + fine sweep for best tension σ using corrected stability."""
        sigmas_coarse = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(20), 100)]
        )
        best_sigma, best_se = None, float("inf")
        for s in sigmas_coarse:
            se = stability_eigenvalue(n, p, q, s, "tension", nu, nextra)
            if se < best_se:
                best_se = se
                best_sigma = s

        # Fine sweep around best
        if best_sigma > 0.5:
            lo = max(0.0, best_sigma * 0.5)
            hi = min(30.0, best_sigma * 2.0)
        else:
            lo = 0.0
            hi = 2.0
        for s in np.linspace(lo, hi, 200):
            se = stability_eigenvalue(n, p, q, s, "tension", nu, nextra)
            if se < best_se:
                best_se = se
                best_sigma = s

        return best_sigma, best_se

    def _find_best_sigma_gamma_corrected(self, n, p, q, nu, nextra, sigma_hint):
        """2D sweep over (σ, γ) with corrected stability metric."""
        sigmas = np.concatenate(
            [[0.0], np.linspace(max(0.01, sigma_hint - 5), sigma_hint + 10, 25)]
        )
        gammas = np.concatenate([[0.0], np.logspace(-2, 2, 25)])

        best_sigma, best_gamma, best_se = None, None, float("inf")
        for s in sigmas:
            for g in gammas:
                D = build_diff_matrix_rbf_penalty(
                    n, p, q, s, "tension", nu, nextra, gamma=g,
                )
                se = stability_eigenvalue_from_matrix(D)
                if se < best_se:
                    best_se = se
                    best_sigma = s
                    best_gamma = g

        return best_sigma, best_gamma, best_se

    def _print_table(self, title, results):
        """Print comparison table with corrected metrics."""
        print(f"\n  {'=' * 100}")
        print(f"  {title}")
        print(f"  {'=' * 100}")
        hdr = (f"  {'Method':>30s}  {'stab_eig':>14s}  {'|λ|_max':>14s}"
               f"  {'CFL(RK4)':>10s}  {'cons deficit':>14s}  {'status':>10s}")
        print(hdr)
        print(f"  {'-' * 30}  {'-' * 14}  {'-' * 14}"
              f"  {'-' * 10}  {'-' * 14}  {'-' * 10}")
        for name, se, sr, cfl, cd in results:
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"  {name:>30s}  {se:14.6e}  {sr:14.6e}"
                  f"  {cfl:10.4f}  {cd:14.6e}  {status:>10s}")

    # --------------------------------------------------------- E2 comparison

    def test_e2_comparison(self):
        """Comprehensive E2_1 comparison with corrected stability metric.

        Finds optimal parameters at n=40, then evaluates at n=20,40,80.
        E2 should be universally stable under the corrected test (Phase 32.2a).
        """
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU
        n_opt = 40

        # Find optimal parameters at n=40
        eps_g, _ = self._find_best_epsilon_corrected(
            n_opt, p, q, nu, nextra, "gaussian"
        )
        sigma_t, _ = self._find_best_sigma_corrected(n_opt, p, q, nu, nextra)
        sg, gg, _ = self._find_best_sigma_gamma_corrected(
            n_opt, p, q, nu, nextra, sigma_t
        )

        print(f"\n  E2_1 optimal params (found at n={n_opt}):")
        print(f"    Gaussian ε*={eps_g:.4f}")
        print(f"    Tension σ*={sigma_t:.4f}")
        print(f"    Tension+penalty σ*={sg:.4f}, γ*={gg:.4f}")

        for n in [20, 40, 80]:
            results = []

            # 1. PHS k=2 (σ→0)
            D_phs = build_diff_matrix_rbf(n, p, q, 1e-15, "tension", nu, nextra)
            results.append(("PHS k=2 (σ=0)", *self._corrected_metrics(D_phs)))

            # 2. Gaussian ε*
            D_gauss = build_diff_matrix_rbf(n, p, q, eps_g, "gaussian", nu, nextra)
            results.append((f"Gaussian ε*={eps_g:.3f}", *self._corrected_metrics(D_gauss)))

            # 3. Tension σ*
            D_tension = build_diff_matrix_rbf(n, p, q, sigma_t, "tension", nu, nextra)
            results.append((f"Tension σ*={sigma_t:.2f}", *self._corrected_metrics(D_tension)))

            # 4. Tension + penalty
            D_pen = build_diff_matrix_rbf_penalty(
                n, p, q, sg, "tension", nu, nextra, gamma=gg,
            )
            results.append((f"Tension σ={sg:.2f} γ={gg:.2f}", *self._corrected_metrics(D_pen)))

            self._print_table(f"E2_1 Corrected Comparison (n={n})", results)

            # E2 is universally stable — assert all methods at every grid size
            for name, se, _, _, _ in results:
                assert se < STABILITY_TOL, (
                    f"E2 {name} should be stable at n={n}, "
                    f"got stab_eig={se:.6e}"
                )

    # --------------------------------------------------------- E4 comparison

    def test_e4_comparison(self):
        """Comprehensive E4_1 comparison with corrected stability metric.

        Finds optimal parameters at n=40, then evaluates at n=20,40,80.
        PHS k=2 and tension should be stable; Gaussian may have narrower
        stable band (Phase 32.2b).
        """
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU
        n_opt = 40

        # Find optimal parameters at n=40
        eps_g, _ = self._find_best_epsilon_corrected(
            n_opt, p, q, nu, nextra, "gaussian"
        )
        sigma_t, _ = self._find_best_sigma_corrected(n_opt, p, q, nu, nextra)
        sg, gg, _ = self._find_best_sigma_gamma_corrected(
            n_opt, p, q, nu, nextra, sigma_t
        )

        print(f"\n  E4_1 optimal params (found at n={n_opt}):")
        print(f"    Gaussian ε*={eps_g:.4f}")
        print(f"    Tension σ*={sigma_t:.4f}")
        print(f"    Tension+penalty σ*={sg:.4f}, γ*={gg:.4f}")

        for n in [20, 40, 80]:
            results = []

            # 1. PHS k=2 (σ→0)
            D_phs = build_diff_matrix_rbf(n, p, q, 1e-15, "tension", nu, nextra)
            results.append(("PHS k=2 (σ=0)", *self._corrected_metrics(D_phs)))

            # 2. Gaussian ε*
            D_gauss = build_diff_matrix_rbf(n, p, q, eps_g, "gaussian", nu, nextra)
            results.append((f"Gaussian ε*={eps_g:.3f}", *self._corrected_metrics(D_gauss)))

            # 3. Tension σ*
            D_tension = build_diff_matrix_rbf(n, p, q, sigma_t, "tension", nu, nextra)
            results.append((f"Tension σ*={sigma_t:.2f}", *self._corrected_metrics(D_tension)))

            # 4. Tension + penalty
            D_pen = build_diff_matrix_rbf_penalty(
                n, p, q, sg, "tension", nu, nextra, gamma=gg,
            )
            results.append((f"Tension σ={sg:.2f} γ={gg:.2f}", *self._corrected_metrics(D_pen)))

            self._print_table(f"E4_1 Corrected Comparison (n={n})", results)

            # All 4 methods at optimal params should be stable at every grid size
            for name, se, sr, cfl, cd in results:
                assert se < STABILITY_TOL, (
                    f"E4 {name} should be stable at n={n}, "
                    f"got stab_eig={se:.6e}"
                )
                assert cfl > 0.01, f"E4 {name} n={n}: CFL too small ({cfl:.6f})"

    # --------------------------------------------------------- Grid convergence

    def test_grid_convergence_summary(self):
        """Grid convergence of stability_eigenvalue for best methods.

        Tracks how stability eigenvalue scales with grid refinement.
        PHS k=2 (σ=0) at E2 and E4 across n=20,40,80,160.
        """
        configs = [
            ("E2_1", self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU),
            ("E4_1", self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU),
        ]

        print(f"\n  {'=' * 72}")
        print(f"  Grid Convergence — PHS k=2 (σ=0) — Corrected Stability")
        print(f"  {'=' * 72}")

        for label, p, q, nextra, nu in configs:
            print(f"\n  {label} (p={p}, q={q}, nextra={nextra})")
            print(f"  {'n':>6s}  {'stab_eig':>14s}  {'|λ|_max':>14s}"
                  f"  {'CFL(RK4)':>10s}  {'status':>10s}")
            print(f"  {'-' * 6}  {'-' * 14}  {'-' * 14}"
                  f"  {'-' * 10}  {'-' * 10}")

            prev_se = None
            for n in [20, 40, 80, 160]:
                D = build_diff_matrix_rbf(n, p, q, 1e-15, "tension", nu, nextra)
                se, sr, cfl, _ = self._corrected_metrics(D)
                status = "STABLE" if se < STABILITY_TOL else "unstable"
                ratio = f" (×{prev_se / se:.1f})" if prev_se is not None and se != 0 else ""
                print(f"  {n:6d}  {se:14.6e}  {sr:14.6e}"
                      f"  {cfl:10.4f}  {status:>10s}{ratio}")
                prev_se = se

                # Both E2 and E4 PHS k=2 should be stable at all grid sizes
                assert se < STABILITY_TOL, (
                    f"{label} PHS k=2 n={n}: expected stable, "
                    f"got stab_eig={se:.6e}"
                )


# ---------------------------------------------------------------------------
# Regression test helpers — load known-good values from sweeps/known_values.json
# ---------------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path

_KNOWN_VALUES_FILE = (
    _Path(__file__).resolve().parent.parent / "sweeps" / "known_values.json"
)


def _load_known_values() -> dict:
    with open(_KNOWN_VALUES_FILE) as f:
        return _json.load(f)


_KNOWN = _load_known_values()


# ---------------------------------------------------------------------------
# 37.3a: Fast regression tests for E2 stability (replaces swept classes)
# ---------------------------------------------------------------------------


class TestRegressionE2Stability:
    """Fast regression spot-checks for E2 stability with known-good parameters.

    Values loaded from sweeps/known_values.json (E2_1 entry).
    """

    _kv = _KNOWN["E2_1"]
    P = _kv["params"]["p"]
    Q = _kv["params"]["q"]
    NEXTRA = _kv["params"]["nextra"]
    NU = _kv["params"]["nu"]

    def test_e2_tension_optimal_sigma(self):
        """E2_1 tension at known σ is stable."""
        sigma = _KNOWN["E2_1"]["tension"]["sigma"]
        se = stability_eigenvalue(
            40, p=self.P, q=self.Q, epsilon=sigma,
            kernel="tension", nu=self.NU, nextra=self.NEXTRA,
        )
        assert se < STABILITY_TOL, (
            f"E2_1 tension σ={sigma} n=40: expected stable, got {se:.6e}"
        )

    def test_e2_gaussian_optimal_epsilon(self):
        """E2_1 Gaussian at known ε is stable."""
        eps = _KNOWN["E2_1"]["gaussian"]["epsilon"]
        se = stability_eigenvalue(
            40, p=self.P, q=self.Q, epsilon=eps,
            kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
        )
        assert se < STABILITY_TOL, (
            f"E2_1 Gaussian ε={eps} n=40: expected stable, got {se:.6e}"
        )

    def test_e2_stable_at_multiple_grid_sizes(self):
        """E2_1 tension and PHS k=2 are stable at known grid sizes."""
        kv = _KNOWN["E2_1"]
        sigma = kv["tension"]["sigma"]
        for s, label, grid_key in [
            (0.0, "PHS k=2", "phs_k2"),
            (sigma, f"tension σ={sigma}", "tension"),
        ]:
            for n in kv[grid_key]["stable_at"]:
                se = stability_eigenvalue(
                    n, p=self.P, q=self.Q, epsilon=s,
                    kernel="tension", nu=self.NU, nextra=self.NEXTRA,
                )
                assert se < STABILITY_TOL, (
                    f"E2_1 {label} n={n}: expected stable, got {se:.6e}"
                )


# ---------------------------------------------------------------------------
# 37.3b: Fast regression tests for E4 stability (replaces swept classes)
# ---------------------------------------------------------------------------


class TestRegressionE4Stability:
    """Fast regression spot-checks for E4 stability with known-good parameters.

    Values loaded from sweeps/known_values.json (E4_1 entry).
    """

    _kv = _KNOWN["E4_1"]
    P = _kv["params"]["p"]
    Q = _kv["params"]["q"]
    NEXTRA = _kv["params"]["nextra"]
    NU = _kv["params"]["nu"]

    def test_e4_tension_known_sigma(self):
        """E4_1 tension at known σ is stable."""
        sigma = _KNOWN["E4_1"]["tension"]["sigma"]
        se = stability_eigenvalue(
            40, p=self.P, q=self.Q, epsilon=sigma,
            kernel="tension", nu=self.NU, nextra=self.NEXTRA,
        )
        assert se < STABILITY_TOL, (
            f"E4_1 tension σ={sigma} n=40: expected stable, got {se:.6e}"
        )

    def test_e4_gaussian_known_epsilon(self):
        """E4_1 Gaussian at known ε is stable."""
        eps = _KNOWN["E4_1"]["gaussian"]["epsilon"]
        se = stability_eigenvalue(
            40, p=self.P, q=self.Q, epsilon=eps,
            kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
        )
        assert se < STABILITY_TOL, (
            f"E4_1 Gaussian ε={eps} n=40: expected stable, got {se:.6e}"
        )

    def test_e4_multiquadric_known_epsilon(self):
        """E4_1 multiquadric at known ε is stable."""
        eps = _KNOWN["E4_1"]["multiquadric"]["epsilon"]
        se = stability_eigenvalue(
            40, p=self.P, q=self.Q, epsilon=eps,
            kernel="multiquadric", nu=self.NU, nextra=self.NEXTRA,
        )
        assert se < STABILITY_TOL, (
            f"E4_1 multiquadric ε={eps} n=40: expected stable, got {se:.6e}"
        )

    def test_e4_stable_at_multiple_grid_sizes(self):
        """E4_1 tension and PHS k=2 are stable at known grid sizes."""
        kv = _KNOWN["E4_1"]
        sigma = kv["tension"]["sigma"]
        for s, label, grid_key in [
            (0.0, "PHS k=2", "phs_k2"),
            (sigma, f"tension σ={sigma}", "tension"),
        ]:
            for n in kv[grid_key]["stable_at"]:
                se = stability_eigenvalue(
                    n, p=self.P, q=self.Q, epsilon=s,
                    kernel="tension", nu=self.NU, nextra=self.NEXTRA,
                )
                assert se < STABILITY_TOL, (
                    f"E4_1 {label} n={n}: expected stable, got {se:.6e}"
                )

    def test_e4_unstable_detected(self):
        """Known-unstable configurations should be detected."""
        for entry in _KNOWN["E4_1"]["known_unstable"]:
            se = stability_eigenvalue(
                entry["n"], p=self.P, q=self.Q, epsilon=entry["epsilon"],
                kernel=entry["kernel"], nu=self.NU, nextra=self.NEXTRA,
            )
            assert se > STABILITY_TOL, (
                f"E4_1 {entry['kernel']} ε={entry['epsilon']} n={entry['n']}: "
                f"expected unstable, got {se:.6e}"
            )


# ---------------------------------------------------------------------------
# 37.3c: Fast regression tests for footprint/nextra (replaces swept classes)
# ---------------------------------------------------------------------------


class TestRegressionFootprint:
    """Fast regression spot-checks for E4 nextra stability.

    Values loaded from sweeps/known_values.json (footprint entry).
    Uses E4_1 base parameters (p=2, q=3, nu=1).
    """

    _e4 = _KNOWN["E4_1"]["params"]
    P = _e4["p"]
    Q = _e4["q"]
    NU = _e4["nu"]
    _fp = _KNOWN["footprint"]

    def test_nextra0_phs_k2_grid_independence(self):
        """E4 nextra=0, PHS k=2 is stable at known grid sizes."""
        entry = self._fp["E4_nextra0_phs"]
        for n in entry["stable_at"]:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=0.0,
                kernel="tension", nu=self.NU, nextra=entry["nextra"],
            )
            assert se < STABILITY_TOL, (
                f"E4 PHS k=2 nextra=0 n={n}: expected stable, got {se:.6e}"
            )

    def test_nextra0_tension(self):
        """E4 nextra=0, tension at known σ is stable."""
        entry = self._fp["E4_nextra0_tension_3"]
        sigma = entry["sigma"]
        for n in entry["stable_at"]:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=entry["nextra"],
            )
            assert se < STABILITY_TOL, (
                f"E4 nextra=0 tension σ={sigma} n={n}: expected stable, got {se:.6e}"
            )

    def test_nextra1_has_stable_sigma(self):
        """E4 nextra=1 has a stable PHS k=2."""
        entry = self._fp["E4_nextra1_phs"]
        for n in entry["stable_at"]:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=0.0,
                kernel="tension", nu=self.NU, nextra=entry["nextra"],
            )
            assert se < STABILITY_TOL, (
                f"E4 nextra=1 PHS k=2 n={n}: expected stable, got {se:.6e}"
            )

    def test_nextra2_has_stable_sigma(self):
        """E4 nextra=2 has a stable PHS k=2."""
        entry = self._fp["E4_nextra2_phs"]
        for n in entry["stable_at"]:
            se = stability_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=0.0,
                kernel="tension", nu=self.NU, nextra=entry["nextra"],
            )
            assert se < STABILITY_TOL, (
                f"E4 nextra=2 PHS k=2 n={n}: expected stable, got {se:.6e}"
            )


# ---------------------------------------------------------------------------
# 37.3d: Fast regression tests for comparison table (replaces swept class)
# ---------------------------------------------------------------------------


class TestRegressionComparison:
    """Fast regression spot-checks for the comprehensive comparison.

    Values loaded from sweeps/known_values.json.
    Tests all methods for both E2_1 and E4_1 at known-good parameters.
    """

    @staticmethod
    def _configs_for(scheme_key):
        """Build (label, kernel, eps) configs from known values."""
        kv = _KNOWN[scheme_key]
        configs = [("PHS k=2", "tension", 0.0)]
        for method in ("gaussian", "tension", "multiquadric"):
            if method in kv:
                param_key = "sigma" if method == "tension" else "epsilon"
                configs.append((method, method, kv[method][param_key]))
        return configs

    def test_e2_all_methods_stable(self):
        """E2_1: all methods stable at n=40."""
        kv = _KNOWN["E2_1"]
        p = kv["params"]
        for label, kernel, eps in self._configs_for("E2_1"):
            se = stability_eigenvalue(
                40, p=p["p"], q=p["q"], epsilon=eps,
                kernel=kernel, nu=p["nu"], nextra=p["nextra"],
            )
            assert se < STABILITY_TOL, (
                f"E2_1 {label} n=40: expected stable, got {se:.6e}"
            )

    def test_e4_all_methods_stable(self):
        """E4_1: all methods stable at n=40."""
        kv = _KNOWN["E4_1"]
        p = kv["params"]
        for label, kernel, eps in self._configs_for("E4_1"):
            se = stability_eigenvalue(
                40, p=p["p"], q=p["q"], epsilon=eps,
                kernel=kernel, nu=p["nu"], nextra=p["nextra"],
            )
            assert se < STABILITY_TOL, (
                f"E4_1 {label} n=40: expected stable, got {se:.6e}"
            )

    def test_phs_k2_grid_convergence(self):
        """PHS k=2 is stable at known grid sizes for both E2 and E4."""
        for scheme_key in ("E2_1", "E4_1"):
            kv = _KNOWN[scheme_key]
            p = kv["params"]
            for n in kv["phs_k2"]["stable_at"]:
                se = stability_eigenvalue(
                    n, p=p["p"], q=p["q"], epsilon=0.0,
                    kernel="tension", nu=p["nu"], nextra=p["nextra"],
                )
                assert se < STABILITY_TOL, (
                    f"{scheme_key} PHS k=2 n={n}: expected stable, got {se:.6e}"
                )
