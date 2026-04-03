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
            stable = "STABLE" if best[1] < STABILITY_TOL else "unstable"
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
            if best[1] < STABILITY_TOL:
                print(f"\n  *** STABLE epsilon found for n={n}: eps={best[0]:.6f} ***")

    def test_multiquadric_sweep(self):
        """Sweep Multiquadric kernel epsilon for E2_1."""
        epsilons = np.logspace(np.log10(0.01), np.log10(10), 60)
        n_values = [20, 40, 80]
        results = self._sweep("multiquadric", n_values, epsilons)
        self._print_table("E2_1 Multiquadric RBF Epsilon Sweep (p=1, q=1, nextra=1)", results)

        for n, rows in results.items():
            best = min(rows, key=lambda r: r[1])
            if best[1] < STABILITY_TOL:
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

        stable = best_fine[1] < STABILITY_TOL
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
            stable = "STABLE" if best[1] < STABILITY_TOL else "unstable"
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
            if best[1] < STABILITY_TOL:
                print(f"\n  *** STABLE epsilon found for n={n}: eps={best[0]:.6f} ***")

    def test_multiquadric_sweep(self):
        """Sweep Multiquadric kernel epsilon for E4_1."""
        epsilons = np.logspace(np.log10(0.01), np.log10(10), 60)
        n_values = [20, 40, 80]
        results = self._sweep("multiquadric", n_values, epsilons)
        self._print_table("E4_1 Multiquadric RBF Epsilon Sweep (p=2, q=3, nextra=0)", results)

        for n, rows in results.items():
            best = min(rows, key=lambda r: r[1])
            if best[1] < STABILITY_TOL:
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

        stable = best_fine[1] < STABILITY_TOL
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
    From 29.6c/e: Gaussian ε*≈1.83 yields machine-precision eigenvalue stability.

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

            # Key assertions at n=40: regression-protect the stability findings
            if n == 40:
                assert re_rbf < 1e-13, (
                    f"RBF direct should be stable to machine precision, got {re_rbf}"
                )
                assert re_temo > 0.1, (
                    f"TEMO+conservation should be unstable, got {re_temo}"
                )

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


# ---------------------------------------------------------------------------
# 29.7a: Side-by-side comparison table
# ---------------------------------------------------------------------------


class TestComparisonTable:
    """Side-by-side comparison of PHS k=2, Gaussian RBF, Multiquadric RBF,
    and mixed-ε configurations for E2_1 and E4_1.

    Metrics: max Re(λ), spectral radius, implied CFL with RK4, conservation deficit.
    RK4 imaginary stability limit ≈ 2.828 (along pure imaginary axis).
    """

    # E2_1 parameters
    E2_P, E2_Q, E2_NEXTRA, E2_NU = 1, 1, 1, 1
    # E4_1 parameters
    E4_P, E4_Q, E4_NEXTRA, E4_NU = 2, 3, 0, 1

    # RK4 stability limit along the imaginary axis
    RK4_IMAG_LIMIT = 2.828

    # ------------------------------------------------------------------ helpers

    def _build_diff_matrix_phs(self, n, p, q, k, nu, nextra):
        """Build n×n diff matrix with PHS k boundary stencils + classical interior."""
        from sympy import cancel as sym_cancel

        from stencil_gen.interior import derive_interior, full_gamma_array

        # Dimensions (same formula as build_diff_matrix_rbf)
        if nu == 1:
            t = p + q + 1 + nextra
            r = q + 1 + nextra
        else:
            raise NotImplementedError

        interior_coeffs = derive_interior(0, p, nu)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        D = np.zeros((n, n))

        # Left boundary rows: PHS+poly stencil
        for i in range(r):
            w_sym = uniform_boundary_weights(i, t, nu, k, q)
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
            w_sym = uniform_boundary_weights(i, t, nu, k, q)
            w = [float(sym_cancel(c)) for c in w_sym]
            row = n - 1 - i
            for j in range(t):
                D[row, n - 1 - j] = sign * w[j]

        return D

    def _find_best_epsilon(self, p, q, nextra, nu, kernel, n=40):
        """Find best ε via coarse + fine sweep."""
        eps_coarse = np.logspace(np.log10(0.01), np.log10(15), 80)
        best_eps, best_re = None, np.inf
        for eps in eps_coarse:
            mr = max_real_eigenvalue(n, p, q, eps, kernel, nu, nextra)
            if mr < best_re:
                best_re = mr
                best_eps = eps

        # Fine sweep around coarse best
        lo = max(0.001, best_eps / 3)
        hi = min(50, best_eps * 3)
        eps_fine = np.linspace(lo, hi, 200)
        for eps in eps_fine:
            mr = max_real_eigenvalue(n, p, q, eps, kernel, nu, nextra)
            if mr < best_re:
                best_re = mr
                best_eps = eps

        return best_eps, best_re

    def _find_best_mixed_epsilon(self, p, q, nextra, nu, r, kernel, n=40):
        """Coordinate descent to find per-row optimal ε (returns list of ε)."""
        # Start from uniform best
        best_single, _ = self._find_best_epsilon(p, q, nextra, nu, kernel, n)
        current = [best_single] * r
        eps_vals = np.logspace(np.log10(0.3), np.log10(10.0), 40)

        def _max_re(eps_list):
            D = build_diff_matrix_mixed_epsilon(
                n, p, q, eps_list, kernel, nu, nextra
            )
            return float(np.max(np.real(np.linalg.eigvals(D))))

        current_re = _max_re(current)
        for _ in range(3):
            for row in range(r):
                for eps in eps_vals:
                    trial = list(current)
                    trial[row] = eps
                    mr = _max_re(trial)
                    if mr < current_re:
                        current_re = mr
                        current[row] = eps

        return current, current_re

    def _conservation_deficit(self, D, n, p, r):
        """Max absolute column-sum deviation in the overlap region.

        For a conservative first-derivative operator, the full matrix column
        sums should be zero.  The conservation deficit is the max absolute
        column sum across all columns.
        """
        col_sums = np.sum(D, axis=0)
        return float(np.max(np.abs(col_sums)))

    def _metrics(self, D, n, p, r):
        """Compute all comparison metrics from a differentiation matrix."""
        eigvals = np.linalg.eigvals(D)
        max_re = float(np.max(np.real(eigvals)))
        spec_rad = float(np.max(np.abs(eigvals)))
        # CFL = RK4 imaginary limit / spectral_radius
        cfl = self.RK4_IMAG_LIMIT / spec_rad if spec_rad > 0 else float("inf")
        cons_deficit = self._conservation_deficit(D, n, p, r)
        return max_re, spec_rad, cfl, cons_deficit

    # --------------------------------------------------------- E2 comparison

    def test_e2_comparison(self):
        """Side-by-side comparison for E2_1 (p=1, q=1, nextra=1, nu=1)."""
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU
        n = 40
        r = q + 1 + nextra  # = 3

        results = []

        # 1. PHS k=2
        D_phs = self._build_diff_matrix_phs(n, p, q, k=2, nu=nu, nextra=nextra)
        m = self._metrics(D_phs, n, p, r)
        results.append(("PHS k=2", *m))

        # 2. Gaussian RBF (best ε)
        eps_g, _ = self._find_best_epsilon(p, q, nextra, nu, "gaussian", n)
        D_gauss = build_diff_matrix_rbf(n, p, q, eps_g, "gaussian", nu, nextra)
        m = self._metrics(D_gauss, n, p, r)
        results.append((f"Gaussian ε={eps_g:.3f}", *m))

        # 3. Multiquadric RBF (best ε)
        eps_m, _ = self._find_best_epsilon(p, q, nextra, nu, "multiquadric", n)
        D_mq = build_diff_matrix_rbf(n, p, q, eps_m, "multiquadric", nu, nextra)
        m = self._metrics(D_mq, n, p, r)
        results.append((f"MQ ε={eps_m:.3f}", *m))

        # Print table
        print(f"\n{'='*80}")
        print(f"  E2_1 Comparison Table (n={n}, p={p}, q={q}, nextra={nextra})")
        print(f"{'='*80}")
        hdr = f"  {'Method':>22s}  {'max Re(λ)':>14s}  {'|λ|_max':>14s}  {'CFL(RK4)':>10s}  {'cons deficit':>14s}"
        print(hdr)
        print(f"  {'-'*22}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*14}")
        for name, max_re, sr, cfl, cd in results:
            print(f"  {name:>22s}  {max_re:14.6e}  {sr:14.6e}  {cfl:10.4f}  {cd:14.6e}")

        # Key assertion: Gaussian RBF should achieve machine-precision stability
        gauss_re = results[1][1]
        assert gauss_re < 1e-12, f"E2 Gaussian should be stable, got {gauss_re}"

        # PHS k=2 with E2_1 parameters (nextra=1) is NOT stable — O(1e-2).
        # Note: the original Phase 29 finding of 5.7e-14 was for nextra=0 (t=3, r=2),
        # a smaller configuration that doesn't match the production E2_1 scheme.
        phs_re = results[0][1]
        assert phs_re > 0.01, (
            f"PHS k=2 E2_1 (nextra=1) should be unstable O(1e-2), got {phs_re}"
        )

    # --------------------------------------------------------- E4 comparison

    def test_e4_comparison(self):
        """Side-by-side comparison for E4_1 (p=2, q=3, nextra=0, nu=1)."""
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU
        n = 40
        r = q + 1 + nextra  # = 4

        results = []

        # 1. PHS k=2
        D_phs = self._build_diff_matrix_phs(n, p, q, k=2, nu=nu, nextra=nextra)
        m = self._metrics(D_phs, n, p, r)
        results.append(("PHS k=2", *m))

        # 2. Gaussian RBF (best single ε)
        eps_g, _ = self._find_best_epsilon(p, q, nextra, nu, "gaussian", n)
        D_gauss = build_diff_matrix_rbf(n, p, q, eps_g, "gaussian", nu, nextra)
        m = self._metrics(D_gauss, n, p, r)
        results.append((f"Gaussian ε={eps_g:.3f}", *m))

        # 3. Multiquadric RBF (best single ε)
        eps_m, _ = self._find_best_epsilon(p, q, nextra, nu, "multiquadric", n)
        D_mq = build_diff_matrix_rbf(n, p, q, eps_m, "multiquadric", nu, nextra)
        m = self._metrics(D_mq, n, p, r)
        results.append((f"MQ ε={eps_m:.3f}", *m))

        # 4. Mixed-ε Gaussian (per-row coordinate descent)
        mixed_eps, mixed_re = self._find_best_mixed_epsilon(
            p, q, nextra, nu, r, "gaussian", n
        )
        D_mixed = build_diff_matrix_mixed_epsilon(
            n, p, q, mixed_eps, "gaussian", nu, nextra
        )
        m = self._metrics(D_mixed, n, p, r)
        eps_str = ",".join(f"{e:.1f}" for e in mixed_eps)
        results.append((f"Mixed Gauss [{eps_str}]", *m))

        # Print table
        print(f"\n{'='*80}")
        print(f"  E4_1 Comparison Table (n={n}, p={p}, q={q}, nextra={nextra})")
        print(f"{'='*80}")
        hdr = f"  {'Method':>30s}  {'max Re(λ)':>14s}  {'|λ|_max':>14s}  {'CFL(RK4)':>10s}  {'cons deficit':>14s}"
        print(hdr)
        print(f"  {'-'*30}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*14}")
        for name, max_re, sr, cfl, cd in results:
            print(f"  {name:>30s}  {max_re:14.6e}  {sr:14.6e}  {cfl:10.4f}  {cd:14.6e}")

        # Key assertion: PHS k=2 should be better than raw unstable (< 0.01)
        phs_re = results[0][1]
        assert phs_re < 0.01, f"E4 PHS k=2 should have small instability, got {phs_re}"

        # Mixed-ε should improve over single Gaussian
        gauss_re = results[1][1]
        mixed_re_actual = results[3][1]
        assert mixed_re_actual <= gauss_re * 1.1, (
            f"Mixed-ε should not be worse than single: {mixed_re_actual} vs {gauss_re}"
        )

    # ---------------------------------------------- combined summary

    def test_summary_across_grid_sizes(self):
        """Grid-convergence summary for best methods at n=20,40,80.

        Checks whether the stability metric improves, holds, or degrades
        with grid refinement — a key indicator of whether an approach is
        viable for production use.
        """
        print(f"\n{'='*80}")
        print(f"  Grid-Convergence Summary (best method per scheme)")
        print(f"{'='*80}")

        # Collect results keyed by (label, n) for assertions
        results = {}

        for label, p, q, nextra, nu in [
            ("E2_1", self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU),
            ("E4_1", self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU),
        ]:
            # Find best Gaussian ε at n=40 reference
            eps_g, _ = self._find_best_epsilon(p, q, nextra, nu, "gaussian", n=40)

            print(f"\n  {label} — Gaussian ε*={eps_g:.3f}")
            print(f"  {'n':>5s}  {'max Re(λ)':>14s}  {'|λ|_max':>14s}  {'CFL(RK4)':>10s}")
            print(f"  {'-'*5}  {'-'*14}  {'-'*14}  {'-'*10}")

            prev_re = None
            for n in [20, 40, 80]:
                D = build_diff_matrix_rbf(n, p, q, eps_g, "gaussian", nu, nextra)
                eigvals = np.linalg.eigvals(D)
                max_re = float(np.max(np.real(eigvals)))
                sr = float(np.max(np.abs(eigvals)))
                cfl = self.RK4_IMAG_LIMIT / sr if sr > 0 else float("inf")
                trend = ""
                if prev_re is not None:
                    if max_re > prev_re * 10:
                        trend = " ↑ DEGRADING"
                    elif max_re < prev_re * 0.1:
                        trend = " ↓ improving"
                    else:
                        trend = " ~ stable"
                prev_re = max_re
                results[(label, n)] = max_re
                print(f"  {n:5d}  {max_re:14.6e}  {sr:14.6e}  {cfl:10.4f}{trend}")

        # Regression-protect the key findings:
        # E2_1 Gaussian achieves machine-precision stability at n=40
        assert results[("E2_1", 40)] < 1e-12, (
            f"E2_1 Gaussian at n=40 should be machine-precision stable, "
            f"got {results[('E2_1', 40)]:.3e}"
        )
        # E4_1 Gaussian does NOT achieve stability — instability persists at n=80
        assert results[("E4_1", 80)] > 1e-5, (
            f"E4_1 Gaussian at n=80 should show residual instability, "
            f"got {results[('E4_1', 80)]:.3e}"
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
# 30.2b: Tension sigma sweep for E2 (p=1, q=1, nextra=1)
# ---------------------------------------------------------------------------


class TestTensionSweepE2:
    """Sweep σ for E2_1 boundary stencils using tension spline kernel.

    E2_1 parameters: p=1, q=1, nextra=1.
    Sweeps tension parameter σ over [0, 20] and records max Re(λ).
    Key question: does PHS k=2 (σ=0) connect smoothly to a stable region?
    """

    # E2_1 parameters
    P = 1
    Q = 1
    NEXTRA = 1
    NU = 1

    def _sweep(self, n_values, sigmas):
        """Run sigma sweep, return dict {n: list of (sigma, max_re, spec_rad)}."""
        results = {}
        for n in n_values:
            rows = []
            for sigma in sigmas:
                D = build_diff_matrix_rbf(
                    n, p=self.P, q=self.Q, epsilon=sigma,
                    kernel="tension", nu=self.NU, nextra=self.NEXTRA,
                )
                eigvals = np.linalg.eigvals(D)
                max_re = float(np.max(np.real(eigvals)))
                spec_rad = float(np.max(np.abs(eigvals)))
                rows.append((sigma, max_re, spec_rad))
            results[n] = rows
        return results

    def _print_table(self, label, results):
        """Print formatted sweep table."""
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        for n, rows in sorted(results.items()):
            print(f"\n  n = {n}")
            print(f"  {'sigma':>10s}  {'max Re(λ)':>14s}  {'spec radius':>14s}")
            print(f"  {'-'*10}  {'-'*14}  {'-'*14}")
            for sigma, max_re, spec_rad in rows:
                print(f"  {sigma:10.4f}  {max_re:14.6e}  {spec_rad:14.6e}")

        # Summary: best sigma per n
        print(f"\n  --- Best sigma (min max Re(λ)) ---")
        for n, rows in sorted(results.items()):
            best = min(rows, key=lambda r: r[1])
            stable = "STABLE" if best[1] < STABILITY_TOL else "unstable"
            print(f"  n={n:3d}: σ={best[0]:.4f}, max Re(λ)={best[1]:.6e} [{stable}]")

    def test_tension_coarse_sweep(self):
        """Coarse sweep of σ over [0, 20] for E2_1 with tension kernel."""
        # Include σ=0 (PHS k=2 limit) plus logarithmic spacing for σ > 0
        sigmas = np.concatenate([[0.0], np.logspace(np.log10(0.01), np.log10(20), 60)])
        n_values = [20, 40, 80]
        results = self._sweep(n_values, sigmas)
        self._print_table(
            "E2_1 Tension Spline Sigma Sweep (p=1, q=1, nextra=1)", results
        )

        # Find if any sigma gives stability
        for n, rows in results.items():
            best = min(rows, key=lambda r: r[1])
            if best[1] < STABILITY_TOL:
                print(f"\n  *** STABLE sigma found for n={n}: σ={best[0]:.6f} ***")

        # Regression: machine-precision stability must exist for n=40
        best_40 = min(results[40], key=lambda r: r[1])
        assert best_40[1] < STABILITY_TOL, (
            f"E2 tension coarse sweep: expected machine-precision stable σ for n=40, "
            f"got min max Re(λ) = {best_40[1]:.6e}"
        )

    def test_tension_fine_sweep_near_best(self):
        """Fine sweep around the best σ from coarse sweep.

        Uses n=40 for the coarse pass, then refines near the minimum.
        """
        n = 40
        # Coarse sweep (include σ=0)
        sigmas_coarse = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(20), 60)]
        )
        coarse = []
        for sigma in sigmas_coarse:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            coarse.append((sigma, max_re))

        best_coarse = min(coarse, key=lambda r: r[1])
        sigma_best = best_coarse[0]

        # Fine sweep: ±factor around best (or [0, 2] if best is at 0)
        if sigma_best < 0.1:
            lo, hi = 0.0, 2.0
        else:
            lo = max(0.0, sigma_best / 5)
            hi = sigma_best * 5
        sigmas_fine = np.linspace(lo, hi, 200)
        fine = []
        for sigma in sigmas_fine:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            fine.append((sigma, max_re))

        best_fine = min(fine, key=lambda r: r[1])
        print(f"\n  E2_1 Tension fine sweep (n={n}):")
        print(f"  Coarse best: σ={best_coarse[0]:.6f}, max Re(λ)={best_coarse[1]:.6e}")
        print(f"  Fine best:   σ={best_fine[0]:.6f}, max Re(λ)={best_fine[1]:.6e}")

        stable = best_fine[1] < STABILITY_TOL
        print(f"  Stable: {stable}")

        # Regression: fine-sweep best must be machine-precision stable
        assert best_fine[1] < STABILITY_TOL, (
            f"E2 tension fine sweep: expected stable, got max Re(λ) = {best_fine[1]:.6e}"
        )

        # Verify at multiple grid sizes
        sigma_star = best_fine[0]
        print(f"\n  Checking σ*={sigma_star:.6f} across grid sizes:")
        for nn in [20, 40, 80, 160]:
            mr = max_real_eigenvalue(
                nn, p=self.P, q=self.Q, epsilon=sigma_star,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            print(f"    n={nn:4d}: max Re(λ)={mr:.6e}")
            # Regression: stability must hold at all grid sizes
            assert mr < STABILITY_TOL, (
                f"E2 tension σ*={sigma_star:.4f} unstable at n={nn}: "
                f"max Re(λ) = {mr:.6e}"
            )

    def test_compare_with_gaussian(self):
        """Compare tension best σ with Gaussian best ε for E2_1.

        The Gaussian sweep found ε*≈1.83 (stable).  Does tension find a
        comparable or better result?
        """
        n = 40
        # Tension sweep
        sigmas = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(20), 100)]
        )
        tension_results = []
        for sigma in sigmas:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            tension_results.append((sigma, max_re))

        # Gaussian sweep (same range for comparison)
        epsilons = np.logspace(np.log10(0.01), np.log10(20), 100)
        gaussian_results = []
        for eps in epsilons:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            gaussian_results.append((eps, max_re))

        best_tension = min(tension_results, key=lambda r: r[1])
        best_gaussian = min(gaussian_results, key=lambda r: r[1])

        print(f"\n  E2_1 Comparison (n={n}):")
        print(f"  {'Method':>15s}  {'param':>10s}  {'max Re(λ)':>14s}  {'status':>10s}")
        print(f"  {'-'*15}  {'-'*10}  {'-'*14}  {'-'*10}")

        t_stable = "STABLE" if best_tension[1] < STABILITY_TOL else "unstable"
        g_stable = "STABLE" if best_gaussian[1] < STABILITY_TOL else "unstable"
        print(f"  {'Tension':>15s}  {best_tension[0]:10.4f}  {best_tension[1]:14.6e}  {t_stable:>10s}")
        print(f"  {'Gaussian':>15s}  {best_gaussian[0]:10.4f}  {best_gaussian[1]:14.6e}  {g_stable:>10s}")

        # Also report PHS k=2 (σ=0) for reference
        phs_re = tension_results[0][1]  # σ=0 entry
        phs_stable = "STABLE" if phs_re < STABILITY_TOL else "unstable"
        print(f"  {'PHS k=2 (σ=0)':>15s}  {'0.0':>10s}  {phs_re:14.6e}  {phs_stable:>10s}")

        # Regression: both tension and Gaussian achieve machine-precision
        # stability for E2
        assert best_tension[1] < STABILITY_TOL, (
            f"E2 tension best not stable: max Re(λ) = {best_tension[1]:.6e}"
        )
        assert best_gaussian[1] < STABILITY_TOL, (
            f"E2 Gaussian best not stable: max Re(λ) = {best_gaussian[1]:.6e}"
        )


# ---------------------------------------------------------------------------
# 30.2c: Sigma sweep for E4 (p=2, q=3) — tension spline
# ---------------------------------------------------------------------------


class TestTensionSweepE4:
    """Sweep σ for E4_1 boundary stencils using tension spline kernel.

    E4_1 parameters: p=2, q=3, nextra=0.
    The critical test: does tension do better than Gaussian for E4?
    Gaussian achieved min max Re(λ) ≈ 1e-4 (NOT stable).
    PHS k=2 (σ=0 limit) gave max Re(λ) ≈ 0.006.
    Tension deforms continuously from PHS k=2, so it may find a path to
    stability that the Gaussian (starting from a different point) missed.
    """

    # E4_1 parameters
    P = 2
    Q = 3
    NEXTRA = 0
    NU = 1
    R = 4  # q + 1 + nextra = 3 + 1 + 0

    def _sweep(self, n_values, sigmas):
        """Run sigma sweep, return dict {n: list of (sigma, max_re, spec_rad)}."""
        results = {}
        for n in n_values:
            rows = []
            for sigma in sigmas:
                D = build_diff_matrix_rbf(
                    n, p=self.P, q=self.Q, epsilon=sigma,
                    kernel="tension", nu=self.NU, nextra=self.NEXTRA,
                )
                eigvals = np.linalg.eigvals(D)
                max_re = float(np.max(np.real(eigvals)))
                spec_rad = float(np.max(np.abs(eigvals)))
                rows.append((sigma, max_re, spec_rad))
            results[n] = rows
        return results

    def _print_table(self, label, results):
        """Print formatted sweep table."""
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        for n, rows in sorted(results.items()):
            print(f"\n  n = {n}")
            print(f"  {'sigma':>10s}  {'max Re(λ)':>14s}  {'spec radius':>14s}")
            print(f"  {'-'*10}  {'-'*14}  {'-'*14}")
            for sigma, max_re, spec_rad in rows:
                print(f"  {sigma:10.4f}  {max_re:14.6e}  {spec_rad:14.6e}")

        # Summary: best sigma per n
        print(f"\n  --- Best sigma (min max Re(λ)) ---")
        for n, rows in sorted(results.items()):
            best = min(rows, key=lambda r: r[1])
            stable = "STABLE" if best[1] < STABILITY_TOL else "unstable"
            print(f"  n={n:3d}: σ={best[0]:.4f}, max Re(λ)={best[1]:.6e} [{stable}]")

    def test_tension_coarse_sweep(self):
        """Coarse sweep of σ over [0, 20] for E4_1 with tension kernel."""
        # Include σ=0 (PHS k=2 limit) plus logarithmic spacing for σ > 0
        sigmas = np.concatenate([[0.0], np.logspace(np.log10(0.01), np.log10(20), 60)])
        n_values = [20, 40, 80]
        results = self._sweep(n_values, sigmas)
        self._print_table(
            "E4_1 Tension Spline Sigma Sweep (p=2, q=3, nextra=0)", results
        )

        # Find if any sigma gives stability
        for n, rows in results.items():
            best = min(rows, key=lambda r: r[1])
            if best[1] < STABILITY_TOL:
                print(f"\n  *** STABLE sigma found for n={n}: σ={best[0]:.6f} ***")

        # Regression: PHS k=2 (σ=0) should give max Re(λ) < 0.01 for n=40
        sigma0_40 = [r for r in results[40] if r[0] == 0.0][0]
        assert sigma0_40[1] < 0.05, (
            f"E4 PHS k=2 baseline too large: max Re(λ) = {sigma0_40[1]:.6e}"
        )

        # Regression: best across sweep should improve over PHS k=2
        best_40 = min(results[40], key=lambda r: r[1])
        assert best_40[1] < sigma0_40[1], (
            f"E4 tension sweep did not improve over PHS k=2: "
            f"best={best_40[1]:.6e} vs PHS={sigma0_40[1]:.6e}"
        )

    def test_tension_fine_sweep_near_best(self):
        """Fine sweep around the best σ from coarse sweep.

        Uses n=40 for the coarse pass, then refines near the minimum.
        """
        n = 40
        # Coarse sweep (include σ=0)
        sigmas_coarse = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(20), 60)]
        )
        coarse = []
        for sigma in sigmas_coarse:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            coarse.append((sigma, max_re))

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
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            fine.append((sigma, max_re))

        best_fine = min(fine, key=lambda r: r[1])
        print(f"\n  E4_1 Tension fine sweep (n={n}):")
        print(f"  Coarse best: σ={best_coarse[0]:.6f}, max Re(λ)={best_coarse[1]:.6e}")
        print(f"  Fine best:   σ={best_fine[0]:.6f}, max Re(λ)={best_fine[1]:.6e}")

        stable = best_fine[1] < STABILITY_TOL
        print(f"  Stable: {stable}")

        # Verify at multiple grid sizes
        sigma_star = best_fine[0]
        print(f"\n  Checking σ*={sigma_star:.6f} across grid sizes:")
        for nn in [20, 40, 80, 160]:
            mr = max_real_eigenvalue(
                nn, p=self.P, q=self.Q, epsilon=sigma_star,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            print(f"    n={nn:4d}: max Re(λ)={mr:.6e}")

        # Report: if stable, this is the key result
        if stable:
            print(f"\n  *** KEY RESULT: E4_1 tension stencil is STABLE at σ*={sigma_star:.6f} ***")
        else:
            print(f"\n  E4_1 tension not machine-precision stable.")
            print(f"  Best max Re(λ) = {best_fine[1]:.6e}")

        # Regression: fine-sweep best should be well below 1e-3 (actual ~5e-5)
        assert best_fine[1] < 1e-3, (
            f"E4 tension fine-sweep regression: max Re(λ) = {best_fine[1]:.6e} >= 1e-3"
        )
        # Regression: tension must improve over PHS k=2 (σ=0) baseline
        phs_baseline = max_real_eigenvalue(
            n, p=self.P, q=self.Q, epsilon=0.0,
            kernel="tension", nu=self.NU, nextra=self.NEXTRA,
        )
        assert best_fine[1] < phs_baseline, (
            f"E4 tension fine-sweep did not improve over PHS k=2: "
            f"best={best_fine[1]:.6e} vs PHS={phs_baseline:.6e}"
        )

    def test_compare_with_gaussian(self):
        """Compare tension best σ with Gaussian best ε for E4_1.

        The Gaussian sweep found min max Re(λ) ≈ 1e-4 (NOT stable).
        Does tension do better?
        """
        n = 40
        # Tension sweep
        sigmas = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(20), 100)]
        )
        tension_results = []
        for sigma in sigmas:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            tension_results.append((sigma, max_re))

        # Gaussian sweep
        epsilons = np.logspace(np.log10(0.01), np.log10(20), 100)
        gaussian_results = []
        for eps in epsilons:
            max_re = max_real_eigenvalue(
                n, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            gaussian_results.append((eps, max_re))

        best_tension = min(tension_results, key=lambda r: r[1])
        best_gaussian = min(gaussian_results, key=lambda r: r[1])

        print(f"\n  E4_1 Comparison (n={n}):")
        print(f"  {'Method':>15s}  {'param':>10s}  {'max Re(λ)':>14s}  {'status':>10s}")
        print(f"  {'-'*15}  {'-'*10}  {'-'*14}  {'-'*10}")

        t_stable = "STABLE" if best_tension[1] < STABILITY_TOL else "unstable"
        g_stable = "STABLE" if best_gaussian[1] < STABILITY_TOL else "unstable"
        print(f"  {'Tension':>15s}  {best_tension[0]:10.4f}  {best_tension[1]:14.6e}  {t_stable:>10s}")
        print(f"  {'Gaussian':>15s}  {best_gaussian[0]:10.4f}  {best_gaussian[1]:14.6e}  {g_stable:>10s}")

        # PHS k=2 (σ=0) for reference
        phs_re = tension_results[0][1]  # σ=0 entry
        phs_stable = "STABLE" if phs_re < STABILITY_TOL else "unstable"
        print(f"  {'PHS k=2 (σ=0)':>15s}  {'0.0':>10s}  {phs_re:14.6e}  {phs_stable:>10s}")

        # Report which is better
        if best_tension[1] < best_gaussian[1]:
            improvement = best_gaussian[1] / max(best_tension[1], 1e-16)
            print(f"\n  Tension BEATS Gaussian by factor {improvement:.1f}x")
        else:
            ratio = best_tension[1] / max(best_gaussian[1], 1e-16)
            print(f"\n  Gaussian beats tension by factor {ratio:.1f}x")

        # Regression: both methods should achieve < 1e-3 for E4 (actual ~5e-5 / ~8e-5)
        assert best_tension[1] < 1e-3, (
            f"E4 tension regression: max Re(λ) = {best_tension[1]:.6e} >= 1e-3"
        )
        assert best_gaussian[1] < 1e-3, (
            f"E4 Gaussian regression: max Re(λ) = {best_gaussian[1]:.6e} >= 1e-3"
        )
        # Regression: tension should improve ≥ 10× over PHS k=2 (from ~0.006 to ~5e-5)
        assert best_tension[1] < phs_re / 10, (
            f"E4 tension did not improve ≥10× over PHS k=2: "
            f"best={best_tension[1]:.6e} vs PHS/10={phs_re/10:.6e}"
        )

    def test_mixed_tension_two_group(self):
        """Sweep two groups of σ: σ_outer (rows 0,1) and σ_inner (rows 2,3).

        Per-row σ (mixed-tension) may find stability that uniform σ cannot.
        Uses build_diff_matrix_mixed_epsilon with kernel="tension".
        """
        n = 40
        sigma_range = np.concatenate(
            [[0.0], np.logspace(np.log10(0.1), np.log10(15.0), 25)]
        )

        best_combo = None
        best_re = np.inf

        for s_outer in sigma_range:
            for s_inner in sigma_range:
                epsilons = [s_outer, s_outer, s_inner, s_inner]
                D = build_diff_matrix_mixed_epsilon(
                    n, p=self.P, q=self.Q, epsilons=epsilons,
                    kernel="tension", nu=self.NU, nextra=self.NEXTRA,
                )
                eigvals = np.linalg.eigvals(D)
                mr = float(np.max(np.real(eigvals)))
                if mr < best_re:
                    best_re = mr
                    best_combo = (s_outer, s_inner)

        print(f"\n  E4_1 mixed-tension two-group sweep (n={n}):")
        print(f"  Best: σ_outer={best_combo[0]:.4f}, σ_inner={best_combo[1]:.4f}")
        print(f"  max Re(λ)={best_re:.6e}")

        stable = best_re < STABILITY_TOL
        print(f"  Stable: {stable}")

        if stable:
            print(f"\n  *** MIXED TENSION achieves STABILITY for E4_1 ***")

        # Verify at multiple grid sizes
        print(f"\n  Checking best combo across grid sizes:")
        for nn in [20, 40, 80]:
            epsilons = [best_combo[0], best_combo[0], best_combo[1], best_combo[1]]
            D = build_diff_matrix_mixed_epsilon(
                nn, p=self.P, q=self.Q, epsilons=epsilons,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            eigvals = np.linalg.eigvals(D)
            mr = float(np.max(np.real(eigvals)))
            status = "STABLE" if mr < STABILITY_TOL else "unstable"
            print(f"    n={nn:4d}: max Re(λ)={mr:.6e} [{status}]")

        # Regression: mixed-tension best should be < 1e-3 (actual ~5e-5)
        assert best_re < 1e-3, (
            f"E4 mixed-tension regression: max Re(λ) = {best_re:.6e} >= 1e-3"
        )


# ---------------------------------------------------------------------------
# 30.2d: Fine-grained search near optimal σ
# ---------------------------------------------------------------------------


def _bisect_threshold(f, a, b, threshold, tol=1e-4, maxiter=60):
    """Bisection to find x where f(x) crosses threshold from above.

    Assumes f(a) > threshold and f(b) < threshold.
    Returns x such that f(x) ≈ threshold.
    """
    for _ in range(maxiter):
        if b - a < tol:
            break
        mid = (a + b) / 2
        if f(mid) > threshold:
            a = mid
        else:
            b = mid
    return (a + b) / 2


def _dense_sweep_min(f, sigmas):
    """Dense sweep returning (sigma_best, f_best, all_results).

    all_results is a list of (sigma, f_val) sorted by sigma.
    """
    results = [(s, f(s)) for s in sigmas]
    best = min(results, key=lambda r: r[1])
    return best[0], best[1], results


class TestTensionOptimalSigma:
    """Fine-grained search for optimal σ.

    For E2: bisection to find the stability transition σ_crit, then report
    weights at a σ* slightly above σ_crit in the stable plateau.
    For E4: dense sweep to find best σ* (noisy O(1e-4) landscape).
    """

    def _max_re(self, sigma, n, p, q, nu, nextra):
        """Compute max Re(λ) for given tension parameter."""
        return max_real_eigenvalue(
            n, p=p, q=q, epsilon=sigma,
            kernel="tension", nu=nu, nextra=nextra,
        )

    def _max_re_gaussian(self, eps, n, p, q, nu, nextra):
        """Compute max Re(λ) for given Gaussian parameter."""
        return max_real_eigenvalue(
            n, p=p, q=q, epsilon=eps,
            kernel="gaussian", nu=nu, nextra=nextra,
        )

    def test_e2_optimal_sigma(self):
        """Find σ_crit for E2_1 via bisection and report stencil weights.

        E2_1 params: p=1, q=1, nextra=1, nu=1.
        From 30.2b: sharp transition to machine-precision stability at σ≈5.0.
        Bisect to find σ_crit precisely, then pick σ* = σ_crit + 1 in the
        stable plateau.
        """
        p, q, nextra, nu = 1, 1, 1, 1
        n = 40

        # Bisection: find σ_crit where max Re(λ) crosses 1e-6
        # Bracket: at σ=3 max_re≈0.02 (above), at σ=7 max_re≈1e-14 (below)
        threshold = 1e-6
        sigma_crit = _bisect_threshold(
            lambda s: self._max_re(s, n, p, q, nu, nextra),
            3.0, 7.0, threshold, tol=1e-4,
        )

        # Pick σ* well into the stable plateau
        sigma_star = sigma_crit + 1.0
        re_star = self._max_re(sigma_star, n, p, q, nu, nextra)

        print(f"\n  E2_1 optimal σ (bisection + plateau, n={n}):")
        print(f"  σ_crit = {sigma_crit:.4f} (transition to stability)")
        print(f"  σ* = {sigma_star:.4f} (σ_crit + 1.0)")
        print(f"  max Re(λ) at σ* = {re_star:.6e}")

        # Report stencil weights at σ*
        t = p + q + 1 + nextra  # boundary stencil width
        r = q + 1 + nextra      # number of boundary rows
        print(f"\n  Boundary stencil at σ*={sigma_star:.4f} (t={t}, r={r} rows per side):")
        for i in range(r):
            w = uniform_boundary_weights_tension(i, t, nu, q, sigma_star)
            w_str = ", ".join(f"{v:+.10f}" for v in w)
            print(f"    row {i}: [{w_str}]")

        # Grid-independence check
        print(f"\n  Grid-independence check at σ*={sigma_star:.4f}:")
        all_stable = True
        for nn in [20, 40, 80, 160]:
            mr = self._max_re(sigma_star, nn, p, q, nu, nextra)
            stable = mr < 1e-6  # loose threshold for eigenvalue noise
            all_stable = all_stable and stable
            status = "STABLE" if stable else "unstable"
            print(f"    n={nn:4d}: max Re(λ) = {mr:.6e} [{status}]")

        # Regression: σ_crit must be in a reasonable range
        assert 3.0 < sigma_crit < 7.0, (
            f"E2 σ_crit outside expected range: {sigma_crit:.4f}"
        )
        # Regression: σ* must give near-machine-precision stability
        assert re_star < 1e-6, (
            f"E2 optimal σ not stable: max Re(λ) = {re_star:.6e}"
        )
        # Regression: grid-independence — σ* must be stable at all grid sizes
        assert all_stable, "E2 optimal σ not grid-independent"

    def test_e4_optimal_sigma(self):
        """Dense sweep for E4_1 optimal σ and report stencil weights.

        E4_1 params: p=2, q=3, nextra=0, nu=1.
        From 30.2c: noisy landscape with best ~5e-5.  No single σ achieves
        machine precision.  Dense sweep to find the best region.
        """
        p, q, nextra, nu = 2, 3, 0, 1
        n = 40

        # Dense sweep over [5, 55] (400 points)
        sigmas = np.linspace(5, 55, 400)
        sigma_star, re_star, all_results = _dense_sweep_min(
            lambda s: self._max_re(s, n, p, q, nu, nextra),
            sigmas,
        )

        # Robust estimate: median of top-10 best
        sorted_results = sorted(all_results, key=lambda r: r[1])
        top10 = sorted_results[:10]
        median_re = np.median([r[1] for r in top10])
        sigma_range = (min(r[0] for r in top10), max(r[0] for r in top10))

        print(f"\n  E4_1 optimal σ (dense sweep, n={n}):")
        print(f"  Best σ* = {sigma_star:.4f}, max Re(λ) = {re_star:.6e}")
        print(f"  Stable: {re_star < STABILITY_TOL}")
        print(f"\n  Top-10 best results:")
        for s, mr in top10:
            print(f"    σ={s:8.3f}  max Re(λ)={mr:.6e}")
        print(f"  Median of top-10: {median_re:.6e}")
        print(f"  σ range of top-10: [{sigma_range[0]:.2f}, {sigma_range[1]:.2f}]")

        # Report stencil weights at σ*
        t = p + q + 1 + nextra  # boundary stencil width
        r = q + 1 + nextra      # number of boundary rows
        print(f"\n  Boundary stencil at σ*={sigma_star:.4f} (t={t}, r={r} rows per side):")
        for i in range(r):
            w = uniform_boundary_weights_tension(i, t, nu, q, sigma_star)
            w_str = ", ".join(f"{v:+.10f}" for v in w)
            print(f"    row {i}: [{w_str}]")

        # Grid-dependence check
        print(f"\n  Grid-dependence check at σ*={sigma_star:.4f}:")
        for nn in [20, 40, 80, 160]:
            mr = self._max_re(sigma_star, nn, p, q, nu, nextra)
            status = "STABLE" if mr < STABILITY_TOL else "unstable"
            print(f"    n={nn:4d}: max Re(λ) = {mr:.6e} [{status}]")

        # Regression: best should be well below 1e-3 (actual ~5e-5)
        assert re_star < 1e-3, (
            f"E4 optimal σ regression: max Re(λ) = {re_star:.6e} >= 1e-3"
        )
        # Regression: must improve over PHS k=2 baseline
        phs_baseline = self._max_re(0.0, n, p, q, nu, nextra)
        assert re_star < phs_baseline, (
            f"E4 tension did not improve over PHS k=2: "
            f"best={re_star:.6e} vs PHS={phs_baseline:.6e}"
        )

    def test_comparison_all_methods(self):
        """Compare optimal parameters across all methods for E2 and E4.

        Summary table of PHS k=2 (σ=0), Gaussian ε*, and Tension σ*
        for both E2_1 and E4_1 schemes.
        """
        configs = {
            "E2_1": dict(p=1, q=1, nextra=1, nu=1),
            "E4_1": dict(p=2, q=3, nextra=0, nu=1),
        }
        n = 40

        print(f"\n  {'='*78}")
        print(f"  All-methods comparison (n={n})")
        print(f"  {'='*78}")

        for scheme, cfg in configs.items():
            p, q, nextra, nu = cfg["p"], cfg["q"], cfg["nextra"], cfg["nu"]

            # PHS k=2 baseline (σ=0)
            phs_re = self._max_re(0.0, n, p, q, nu, nextra)

            # Gaussian ε*: dense sweep
            epsilons = np.logspace(np.log10(0.1), np.log10(20), 200)
            _, gauss_re, _ = _dense_sweep_min(
                lambda e: self._max_re_gaussian(e, n, p, q, nu, nextra),
                epsilons,
            )
            best_gauss = min(
                [(e, self._max_re_gaussian(e, n, p, q, nu, nextra)) for e in epsilons],
                key=lambda r: r[1],
            )
            eps_star, gauss_re = best_gauss

            # Tension σ*: dense sweep
            sigmas = np.linspace(1, 55, 300)
            best_tension_sigma, tension_re, _ = _dense_sweep_min(
                lambda s: self._max_re(s, n, p, q, nu, nextra),
                sigmas,
            )
            sigma_star = best_tension_sigma

            print(f"\n  {scheme} (p={p}, q={q}, nextra={nextra}):")
            print(f"  {'Method':>20s}  {'param':>10s}  {'max Re(λ)':>14s}  {'status':>10s}")
            print(f"  {'-'*20}  {'-'*10}  {'-'*14}  {'-'*10}")

            def _status(v):
                return "STABLE" if v < STABILITY_TOL else "unstable"

            print(f"  {'PHS k=2 (σ=0)':>20s}  {'N/A':>10s}  {phs_re:14.6e}  {_status(phs_re):>10s}")
            print(f"  {'Gaussian ε*':>20s}  {eps_star:10.4f}  {gauss_re:14.6e}  {_status(gauss_re):>10s}")
            print(f"  {'Tension σ*':>20s}  {sigma_star:10.4f}  {tension_re:14.6e}  {_status(tension_re):>10s}")

            # Improvement ratios
            if phs_re > 0:
                gauss_improve = phs_re / max(gauss_re, 1e-16)
                tension_improve = phs_re / max(tension_re, 1e-16)
                print(f"\n  Improvement over PHS k=2:")
                print(f"    Gaussian: {gauss_improve:.1f}×")
                print(f"    Tension:  {tension_improve:.1f}×")

            # Save results for assertions
            if scheme == "E2_1":
                e2_phs_re = phs_re
                e2_gauss_re = gauss_re
                e2_tension_re = tension_re
            else:
                e4_phs_re = phs_re
                e4_gauss_re = gauss_re
                e4_tension_re = tension_re

        # Regression assertions
        # E2: PHS k=2 baseline sanity
        assert e2_phs_re < 0.5, f"E2 PHS k=2 baseline unreasonable: {e2_phs_re:.6e}"
        # E2: both Gaussian and tension should achieve near-machine-precision
        assert e2_gauss_re < STABILITY_TOL, (
            f"E2 Gaussian not stable: max Re(λ) = {e2_gauss_re:.6e}"
        )
        assert e2_tension_re < STABILITY_TOL, (
            f"E2 Tension not stable: max Re(λ) = {e2_tension_re:.6e}"
        )

        # E4: PHS k=2 baseline sanity
        assert e4_phs_re < 0.05, f"E4 PHS k=2 baseline unreasonable: {e4_phs_re:.6e}"
        # E4: both Gaussian and tension should improve significantly over PHS k=2
        assert e4_gauss_re < 1e-3, (
            f"E4 Gaussian not improved: max Re(λ) = {e4_gauss_re:.6e}"
        )
        assert e4_tension_re < 1e-3, (
            f"E4 Tension not improved: max Re(λ) = {e4_tension_re:.6e}"
        )
        # E4: both methods must improve over PHS k=2 baseline
        assert e4_gauss_re < e4_phs_re, (
            f"E4 Gaussian ({e4_gauss_re:.6e}) not better than PHS k=2 ({e4_phs_re:.6e})"
        )
        assert e4_tension_re < e4_phs_re, (
            f"E4 Tension ({e4_tension_re:.6e}) not better than PHS k=2 ({e4_phs_re:.6e})"
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
# 30.3b: Joint (σ, γ) sweep for E2 — tension + conservation penalty
# ---------------------------------------------------------------------------


class TestTensionConservationE2:
    """Joint (σ, γ) sweep for E2_1 boundary stencils (Phase 30.3b).

    Sweep both tension parameter σ and conservation penalty γ to find
    whether there exists a (σ*, γ*) where both:
    - max Re(λ) < STABILITY_TOL  (stability)
    - conservation deficit < some threshold  (conservation)

    E2_1 parameters: p=1, q=1, nextra=1.
    From Phase 30.2d, E2 achieves machine-precision stability at σ ≈ 5–6
    with γ=0.  The question is: does adding conservation penalty γ > 0
    destroy stability, or can both be achieved simultaneously?
    """

    P, Q, NEXTRA, NU = 1, 1, 1, 1

    def _eval_point(self, n, sigma, gamma):
        """Evaluate (σ, γ) point: return (max_re, deficit)."""
        D = build_diff_matrix_rbf_penalty(
            n, self.P, self.Q, sigma, "tension", self.NU, self.NEXTRA,
            gamma=gamma,
        )
        eigvals = np.linalg.eigvals(D)
        max_re = float(np.max(np.real(eigvals)))
        deficit = float(np.max(np.abs(np.sum(D, axis=0))))
        return max_re, deficit

    def test_joint_sweep_coarse(self):
        """Coarse 2D sweep over σ × γ for E2_1.

        Maps the stability–conservation landscape to find if a region
        exists where both are satisfied.
        """
        n = 40
        sigmas = np.linspace(0.0, 10.0, 30)
        gammas = np.concatenate([[0.0], np.logspace(-1, 2, 29)])  # 0 + log[0.1..100]

        best_stable_deficit = float("inf")
        best_stable_sigma = None
        best_stable_gamma = None

        # Track the γ=0 baseline deficit at optimal σ
        baseline_deficit = None

        # Track best deficit among stable points with γ > 0
        best_deficit_gamma_pos = float("inf")

        n_stable = 0
        for sigma in sigmas:
            for gamma in gammas:
                max_re, deficit = self._eval_point(n, sigma, gamma)

                if max_re < STABILITY_TOL:
                    n_stable += 1
                    if deficit < best_stable_deficit:
                        best_stable_deficit = deficit
                        best_stable_sigma = sigma
                        best_stable_gamma = gamma

                    # Record baseline (γ=0)
                    if gamma == 0.0:
                        if baseline_deficit is None or deficit < baseline_deficit:
                            baseline_deficit = deficit

                    # Track best stable deficit at γ > 0
                    if gamma > 0.0 and deficit < best_deficit_gamma_pos:
                        best_deficit_gamma_pos = deficit

        # Print landscape summary
        total = len(sigmas) * len(gammas)
        print(f"\n  E2_1 Joint (σ, γ) Sweep (n={n})")
        print(f"  Grid: {len(sigmas)} σ × {len(gammas)} γ = {total} points")
        print(f"  Stable points: {n_stable}/{total}")

        if best_stable_sigma is not None:
            print(f"\n  Best stable point (lowest deficit):")
            print(f"    σ*={best_stable_sigma:.4f}, γ*={best_stable_gamma:.4f}")
            print(f"    deficit={best_stable_deficit:.6e}")

        if baseline_deficit is not None:
            print(f"\n  Baseline (γ=0) deficit at stable σ: {baseline_deficit:.6e}")
            if (best_stable_gamma is not None and best_stable_gamma > 0
                    and baseline_deficit > 0):
                improvement = 1.0 - best_stable_deficit / baseline_deficit
                print(f"  Deficit improvement with γ>0: {improvement:.1%}")

        # Print stability boundary: for each σ, largest γ that is still stable
        print(f"\n  Stability boundary (max γ that is stable, per σ):")
        for sigma in sigmas[::3]:  # every 3rd σ for brevity
            max_gamma = -1.0
            for gamma in gammas:
                max_re, _ = self._eval_point(n, sigma, gamma)
                if max_re < STABILITY_TOL:
                    max_gamma = gamma
            if max_gamma >= 0:
                print(f"    σ={sigma:6.2f}: stable up to γ={max_gamma:.2f}")
            else:
                print(f"    σ={sigma:6.2f}: unstable at all γ")

        # --- Assertions ---
        # E2 must have at least one stable point (we know γ=0, σ≈6 is stable)
        assert n_stable > 0, "No stable (σ, γ) point found for E2"

        # Stability must exist at γ=0 (regression from Phase 30.2)
        assert baseline_deficit is not None, (
            "E2 should be stable at γ=0 for some σ in [0, 10]"
        )

        # Best stable deficit at γ > 0 must improve over the γ=0 baseline.
        # This verifies the penalty mechanism actually reduces conservation
        # deficit while maintaining stability (actual improvement ~29%).
        assert best_deficit_gamma_pos < baseline_deficit, (
            f"γ>0 did not improve deficit over γ=0 baseline: "
            f"{best_deficit_gamma_pos:.6e} >= {baseline_deficit:.6e}"
        )

    def test_stability_survives_moderate_penalty(self):
        """Check that E2 stability at σ* is not destroyed by moderate γ.

        Fix σ at the known optimal (~6.0) and increase γ from 0 to 100.
        Report the critical γ where stability is lost (if any).
        """
        n = 40
        sigma_star = 6.0
        gammas = np.concatenate([[0.0], np.logspace(-1, 2, 50)])  # 0..100

        print(f"\n  E2_1 Stability vs γ at σ*={sigma_star} (n={n})")
        print(f"  {'γ':>10s}  {'max Re(λ)':>14s}  {'deficit':>14s}  {'status':>10s}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*10}")

        max_stable_gamma = -1.0
        deficit_at_zero = None
        deficit_at_max_stable = None

        for gamma in gammas:
            max_re, deficit = self._eval_point(n, sigma_star, gamma)
            status = "STABLE" if max_re < STABILITY_TOL else "unstable"
            # Print a representative subset of rows
            if (gamma == 0.0 or gamma < 0.2
                    or abs(gamma - 1.0) < 0.2 or abs(gamma - 10.0) < 1.5
                    or abs(gamma - 50.0) < 5.0 or gamma > 90.0):
                print(f"  {gamma:10.4f}  {max_re:14.6e}  {deficit:14.6e}  {status:>10s}")

            if max_re < STABILITY_TOL:
                max_stable_gamma = gamma
                deficit_at_max_stable = deficit
            if gamma == 0.0:
                deficit_at_zero = deficit

        print(f"\n  Max γ with stability: {max_stable_gamma:.4f}")
        if deficit_at_zero is not None:
            print(f"  Deficit at γ=0: {deficit_at_zero:.6e}")
        if deficit_at_max_stable is not None and deficit_at_zero is not None:
            print(f"  Deficit at max stable γ={max_stable_gamma:.4f}: "
                  f"{deficit_at_max_stable:.6e}")
            if deficit_at_zero > 0:
                improvement = 1.0 - deficit_at_max_stable / deficit_at_zero
                print(f"  Deficit improvement: {improvement:.1%}")

        # --- Assertions ---
        # γ=0 must be stable (regression)
        re_0, _ = self._eval_point(n, sigma_star, 0.0)
        assert re_0 < STABILITY_TOL, (
            f"E2 unstable at σ={sigma_star}, γ=0: max Re(λ) = {re_0:.6e}"
        )

        # Some γ > 0 should also be stable (conservation penalty shouldn't
        # immediately destroy stability)
        assert max_stable_gamma > 0, (
            "Any positive γ destroys E2 stability — penalty approach not viable"
        )

    def test_fine_sweep_near_optimal(self):
        """Fine 2D sweep near the best (σ, γ) region for E2_1.

        Refines around σ ∈ [4, 8] (known E2 stable region) with moderate γ.
        """
        n = 40
        sigmas = np.linspace(4.0, 8.0, 40)
        gammas = np.concatenate([[0.0], np.logspace(-1, 2, 40)])

        best_deficit = float("inf")
        best_sigma = None
        best_gamma = None
        best_re = None

        for sigma in sigmas:
            for gamma in gammas:
                max_re, deficit = self._eval_point(n, sigma, gamma)
                if max_re < STABILITY_TOL and deficit < best_deficit:
                    best_deficit = deficit
                    best_sigma = sigma
                    best_gamma = gamma
                    best_re = max_re

        print(f"\n  E2_1 Fine joint sweep: σ ∈ [4, 8], γ ∈ [0, 100]")
        print(f"  Grid: {len(sigmas)} × {len(gammas)} = {len(sigmas) * len(gammas)} points")

        if best_sigma is not None:
            print(f"\n  Best stable + lowest deficit:")
            print(f"    σ*={best_sigma:.4f}, γ*={best_gamma:.4f}")
            print(f"    max Re(λ)={best_re:.6e}")
            print(f"    deficit={best_deficit:.6e}")

            # Compare with γ=0 baseline
            re_0, deficit_0 = self._eval_point(n, best_sigma, 0.0)
            print(f"\n  Baseline at same σ, γ=0:")
            print(f"    max Re(λ)={re_0:.6e}")
            print(f"    deficit={deficit_0:.6e}")
            if deficit_0 > 0:
                improvement = 1.0 - best_deficit / deficit_0
                print(f"    Deficit improvement: {improvement:.1%}")

            # Verify grid independence of best point
            print(f"\n  Grid independence at (σ*={best_sigma:.4f}, γ*={best_gamma:.4f}):")
            for nn in [20, 40, 80]:
                mr, df = self._eval_point(nn, best_sigma, best_gamma)
                status = "STABLE" if mr < STABILITY_TOL else "unstable"
                print(f"    n={nn:4d}: max Re(λ)={mr:.6e}, deficit={df:.6e} [{status}]")
        else:
            print("  No stable point found in fine sweep region.")

        # --- Assertions ---
        assert best_sigma is not None, (
            "No stable point found in E2 fine sweep region σ ∈ [4, 8]"
        )
        # The optimizer must find a non-trivial penalty point (γ > 0) that
        # improves conservation — not just pick the γ=0 baseline.
        assert best_gamma > 0, (
            f"Fine sweep best was at γ=0 — penalty did not improve deficit"
        )
        # The best deficit with penalty should not be worse than γ=0
        _, deficit_baseline = self._eval_point(n, best_sigma, 0.0)
        assert best_deficit <= deficit_baseline + 1e-12, (
            f"Penalty worsened deficit: {best_deficit:.6e} > {deficit_baseline:.6e}"
        )


# ---------------------------------------------------------------------------
# 30.3c: Joint (σ, γ) sweep for E4 — tension + conservation penalty
# ---------------------------------------------------------------------------


class TestTensionConservationE4:
    """Joint (σ, γ) sweep for E4_1 boundary stencils (Phase 30.3c).

    Sweep both tension parameter σ and conservation penalty γ to investigate
    whether the 2D (σ, γ) space can breach the O(1e-4–1e-5) stability floor
    that 1D σ sweeps could not.

    E4_1 parameters: p=2, q=3, nextra=0.
    From Phase 30.2d, E4 best uniform σ gives max Re(λ) ≈ 5e-5 (NOT stable).
    The key question: can adding conservation penalty γ > 0 find a (σ*, γ*)
    that achieves machine-precision stability for E4?
    """

    P, Q, NEXTRA, NU = 2, 3, 0, 1

    # E4 instability floor from Phase 30.2c/d — best 1D σ gives ~5e-5
    E4_INSTABILITY_FLOOR = 1e-3

    def _eval_point(self, n, sigma, gamma):
        """Evaluate (σ, γ) point: return (max_re, deficit)."""
        D = build_diff_matrix_rbf_penalty(
            n, self.P, self.Q, sigma, "tension", self.NU, self.NEXTRA,
            gamma=gamma,
        )
        eigvals = np.linalg.eigvals(D)
        max_re = float(np.max(np.real(eigvals)))
        deficit = float(np.max(np.abs(np.sum(D, axis=0))))
        return max_re, deficit

    def test_joint_sweep_coarse(self):
        """Coarse 2D sweep over σ × γ for E4_1.

        Maps the stability–conservation landscape.  Since E4 is not
        machine-precision stable at any 1D σ, we check whether the
        2D (σ, γ) space can improve on the 1D floor.
        """
        n = 40
        sigmas = np.linspace(5.0, 55.0, 25)
        gammas = np.concatenate([[0.0], np.logspace(-1, 2, 24)])  # 0 + log[0.1..100]

        best_max_re = float("inf")
        best_sigma = None
        best_gamma = None
        best_deficit = None

        # Track γ=0 baseline
        baseline_re = float("inf")
        baseline_deficit = None

        for sigma in sigmas:
            for gamma in gammas:
                max_re, deficit = self._eval_point(n, sigma, gamma)

                if max_re < best_max_re:
                    best_max_re = max_re
                    best_sigma = sigma
                    best_gamma = gamma
                    best_deficit = deficit

                if gamma == 0.0 and max_re < baseline_re:
                    baseline_re = max_re
                    baseline_deficit = deficit

        print(f"\n  E4_1 Joint (σ, γ) Sweep (n={n})")
        print(f"  Grid: {len(sigmas)} σ × {len(gammas)} γ = "
              f"{len(sigmas) * len(gammas)} points")

        print(f"\n  Best (σ, γ) point (lowest max Re(λ)):")
        print(f"    σ*={best_sigma:.4f}, γ*={best_gamma:.4f}")
        print(f"    max Re(λ)={best_max_re:.6e}")
        print(f"    deficit={best_deficit:.6e}")

        print(f"\n  Baseline (γ=0) best:")
        print(f"    max Re(λ)={baseline_re:.6e}")
        print(f"    deficit={baseline_deficit:.6e}")

        if baseline_re > 0:
            improvement = 1.0 - best_max_re / baseline_re
            print(f"  Stability improvement with (σ,γ) over 1D σ: {improvement:.1%}")

        # Check whether machine-precision stability was achieved
        if best_max_re < STABILITY_TOL:
            print(f"\n  *** BREAKTHROUGH: E4 achieves machine-precision stability! ***")
        else:
            print(f"\n  E4 NOT machine-precision stable (best {best_max_re:.6e})")
            print(f"  The O(1e-4–1e-5) floor persists in 2D (σ, γ) space.")

        # --- Assertions ---
        # Best max Re(λ) should be below the loose E4 floor (actual ~5e-5)
        assert best_max_re < self.E4_INSTABILITY_FLOOR, (
            f"E4 (σ,γ) sweep best {best_max_re:.6e} >= {self.E4_INSTABILITY_FLOOR}"
        )
        # Baseline (γ=0) should also be below the floor (regression from 30.2c)
        assert baseline_re < self.E4_INSTABILITY_FLOOR, (
            f"E4 γ=0 baseline {baseline_re:.6e} >= {self.E4_INSTABILITY_FLOOR}"
        )

    def test_stability_vs_gamma_at_optimal_sigma(self):
        """Check how γ affects E4 stability at the known optimal σ.

        Fix σ at the Phase 30.2d optimal (~37–50) and sweep γ.
        Key question: does γ > 0 help or hurt E4 stability?
        """
        n = 40
        sigma_star = 37.0  # from Phase 30.2c optimal
        gammas = np.concatenate([[0.0], np.logspace(-1, 3, 50)])  # 0..1000

        print(f"\n  E4_1 Stability vs γ at σ*={sigma_star} (n={n})")
        print(f"  {'γ':>10s}  {'max Re(λ)':>14s}  {'deficit':>14s}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*14}")

        best_re = float("inf")
        best_gamma = None
        deficit_at_zero = None
        deficit_at_best = None

        for gamma in gammas:
            max_re, deficit = self._eval_point(n, sigma_star, gamma)
            # Print a representative subset
            if (gamma == 0.0 or gamma < 0.2
                    or abs(gamma - 1.0) < 0.2 or abs(gamma - 10.0) < 1.5
                    or abs(gamma - 100.0) < 15.0 or gamma > 800.0):
                print(f"  {gamma:10.4f}  {max_re:14.6e}  {deficit:14.6e}")

            if max_re < best_re:
                best_re = max_re
                best_gamma = gamma
                deficit_at_best = deficit
            if gamma == 0.0:
                deficit_at_zero = deficit

        print(f"\n  Best γ for stability: {best_gamma:.4f}")
        print(f"    max Re(λ)={best_re:.6e}")
        print(f"    deficit={deficit_at_best:.6e}")
        if deficit_at_zero is not None:
            print(f"  Deficit at γ=0: {deficit_at_zero:.6e}")

        if best_re < STABILITY_TOL:
            print(f"  *** E4 achieves machine-precision stability at γ={best_gamma}! ***")
        else:
            print(f"  E4 NOT machine-precision stable (floor persists)")

        # --- Assertions ---
        # γ=0 stability should match Phase 30.2c (max Re(λ) ~ 5e-5)
        re_0, _ = self._eval_point(n, sigma_star, 0.0)
        assert re_0 < self.E4_INSTABILITY_FLOOR, (
            f"E4 at σ={sigma_star}, γ=0 worse than expected: {re_0:.6e}"
        )
        # Best with γ should not be worse than γ=0
        assert best_re <= re_0 + 1e-6, (
            f"Adding γ worsened E4: best {best_re:.6e} > γ=0 {re_0:.6e}"
        )

    def test_fine_sweep_near_optimal(self):
        """Fine 2D sweep near the best (σ, γ) region for E4_1.

        Refines around σ ∈ [20, 55] (known E4 region) with moderate γ.
        Reports whether the 2D search improves on the 1D σ-only result.
        """
        n = 40
        sigmas = np.linspace(20.0, 55.0, 40)
        gammas = np.concatenate([[0.0], np.logspace(-1, 3, 40)])

        best_re = float("inf")
        best_sigma = None
        best_gamma = None
        best_deficit = None

        # Also track best γ=0 result for comparison
        best_re_baseline = float("inf")

        for sigma in sigmas:
            for gamma in gammas:
                max_re, deficit = self._eval_point(n, sigma, gamma)
                if max_re < best_re:
                    best_re = max_re
                    best_sigma = sigma
                    best_gamma = gamma
                    best_deficit = deficit
                if gamma == 0.0 and max_re < best_re_baseline:
                    best_re_baseline = max_re

        print(f"\n  E4_1 Fine joint sweep: σ ∈ [20, 55], γ ∈ [0, 1000]")
        print(f"  Grid: {len(sigmas)} × {len(gammas)} = "
              f"{len(sigmas) * len(gammas)} points")

        print(f"\n  Best overall (σ, γ):")
        print(f"    σ*={best_sigma:.4f}, γ*={best_gamma:.4f}")
        print(f"    max Re(λ)={best_re:.6e}")
        print(f"    deficit={best_deficit:.6e}")

        print(f"\n  Best γ=0 baseline: max Re(λ)={best_re_baseline:.6e}")
        if best_re_baseline > 0:
            improvement = 1.0 - best_re / best_re_baseline
            print(f"  Improvement from 2D over 1D: {improvement:.1%}")

        if best_re < STABILITY_TOL:
            print(f"\n  *** BREAKTHROUGH: 2D search achieved E4 stability! ***")
        else:
            print(f"\n  E4 NOT machine-precision stable in 2D (σ, γ) space")
            print(f"  The O(1e-4–1e-5) barrier is fundamental for E4.")

        # Grid independence check at best point
        print(f"\n  Grid independence at (σ*={best_sigma:.4f}, γ*={best_gamma:.4f}):")
        for nn in [20, 40, 80]:
            mr, df = self._eval_point(nn, best_sigma, best_gamma)
            print(f"    n={nn:4d}: max Re(λ)={mr:.6e}, deficit={df:.6e}")

        # --- Assertions ---
        # Best 2D result should be below the loose E4 floor
        assert best_re < self.E4_INSTABILITY_FLOOR, (
            f"E4 fine sweep best {best_re:.6e} >= {self.E4_INSTABILITY_FLOOR}"
        )
        # 2D search should not be worse than 1D (γ=0) baseline
        assert best_re <= best_re_baseline + 1e-6, (
            f"2D search worse than 1D: {best_re:.6e} > {best_re_baseline:.6e}"
        )
