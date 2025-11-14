"""
Convergence rate tests for stencils.

Tests that stencils achieve their theoretical convergence rates.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shoccs.stencils import (
    centered_diff_1st_order2,
    centered_diff_2nd_order2,
    centered_diff_1st_order4,
    centered_diff_2nd_order4,
    apply_stencil_1d,
)


def compute_convergence_rate(errors, h_values):
    """
    Compute convergence rate from errors and grid spacings.

    Uses least squares fit to log(error) = log(C) + p*log(h)
    Returns p (the convergence rate).
    """
    log_h = np.log(h_values)
    log_e = np.log(errors)

    # Linear regression: log(e) = a + p*log(h)
    A = np.vstack([np.ones(len(log_h)), log_h]).T
    result = np.linalg.lstsq(A, log_e, rcond=None)
    p = result[0][1]  # slope

    return p


class TestConvergenceRates:
    """Test that stencils achieve theoretical convergence rates."""

    def test_2nd_order_1st_derivative_convergence_rate(self):
        """Test 2nd-order 1st derivative achieves O(h^2) convergence."""

        # Use smooth test function
        def f(x):
            return np.sin(2 * np.pi * x)

        def df(x):
            return 2 * np.pi * np.cos(2 * np.pi * x)

        # Test with multiple grid refinements
        h_values = np.array([0.2, 0.1, 0.05, 0.025])
        errors = []

        for h in h_values:
            x = np.arange(0, 1.0, h)
            y = f(x)
            dy_exact = df(x)

            stencil = centered_diff_1st_order2(h)

            # Compute max error over interior points
            max_error = 0.0
            for i in range(1, len(x) - 1):
                dy_approx = apply_stencil_1d(y, stencil, i)
                max_error = max(max_error, abs(dy_approx - dy_exact[i]))

            errors.append(max_error)

        errors = np.array(errors)

        # Compute convergence rate
        rate = compute_convergence_rate(errors, h_values)

        print(f"\n2nd-order 1st derivative convergence rate: {rate:.3f}")
        print(f"  h values: {h_values}")
        print(f"  Errors:   {errors}")
        print(f"  Expected rate: 2.0")

        # Should achieve close to 2nd order
        # Note: practical convergence rates may be slightly below theoretical due to
        # finite domain, boundary effects, and limited refinement levels
        assert rate > 1.8, f"Convergence rate {rate:.3f} below 2nd order (expected ~2.0)"
        assert rate < 2.2, f"Convergence rate {rate:.3f} suspiciously high"

    def test_2nd_order_2nd_derivative_convergence_rate(self):
        """Test 2nd-order 2nd derivative achieves O(h^2) convergence."""

        def f(x):
            return np.sin(2 * np.pi * x)

        def d2f(x):
            return -(2 * np.pi)**2 * np.sin(2 * np.pi * x)

        h_values = np.array([0.2, 0.1, 0.05, 0.025])
        errors = []

        for h in h_values:
            x = np.arange(0, 1.0, h)
            y = f(x)
            d2y_exact = d2f(x)

            stencil = centered_diff_2nd_order2(h)

            max_error = 0.0
            for i in range(1, len(x) - 1):
                d2y_approx = apply_stencil_1d(y, stencil, i)
                max_error = max(max_error, abs(d2y_approx - d2y_exact[i]))

            errors.append(max_error)

        errors = np.array(errors)
        rate = compute_convergence_rate(errors, h_values)

        print(f"\n2nd-order 2nd derivative convergence rate: {rate:.3f}")
        print(f"  h values: {h_values}")
        print(f"  Errors:   {errors}")

        assert rate > 1.9, f"Convergence rate {rate:.3f} below 2nd order"
        assert rate < 2.2, f"Convergence rate {rate:.3f} suspiciously high"

    def test_4th_order_1st_derivative_convergence_rate(self):
        """Test 4th-order 1st derivative achieves O(h^4) convergence."""

        def f(x):
            return np.sin(2 * np.pi * x)

        def df(x):
            return 2 * np.pi * np.cos(2 * np.pi * x)

        h_values = np.array([0.2, 0.1, 0.05, 0.025])
        errors = []

        for h in h_values:
            x = np.arange(0, 1.0, h)
            y = f(x)
            dy_exact = df(x)

            stencil = centered_diff_1st_order4(h)

            max_error = 0.0
            # Need 2 points on each side for 4th order
            for i in range(2, len(x) - 2):
                dy_approx = apply_stencil_1d(y, stencil, i)
                max_error = max(max_error, abs(dy_approx - dy_exact[i]))

            errors.append(max_error)

        errors = np.array(errors)
        rate = compute_convergence_rate(errors, h_values)

        print(f"\n4th-order 1st derivative convergence rate: {rate:.3f}")
        print(f"  h values: {h_values}")
        print(f"  Errors:   {errors}")
        print(f"  Expected rate: 4.0")

        # Should achieve close to 4th order
        assert rate > 3.8, f"Convergence rate {rate:.3f} below 4th order (expected ~4.0)"
        assert rate < 4.3, f"Convergence rate {rate:.3f} suspiciously high"

    def test_4th_order_2nd_derivative_convergence_rate(self):
        """Test 4th-order 2nd derivative achieves O(h^4) convergence."""

        def f(x):
            return np.sin(2 * np.pi * x)

        def d2f(x):
            return -(2 * np.pi)**2 * np.sin(2 * np.pi * x)

        h_values = np.array([0.2, 0.1, 0.05, 0.025])
        errors = []

        for h in h_values:
            x = np.arange(0, 1.0, h)
            y = f(x)
            d2y_exact = d2f(x)

            stencil = centered_diff_2nd_order4(h)

            max_error = 0.0
            for i in range(2, len(x) - 2):
                d2y_approx = apply_stencil_1d(y, stencil, i)
                max_error = max(max_error, abs(d2y_approx - d2y_exact[i]))

            errors.append(max_error)

        errors = np.array(errors)
        rate = compute_convergence_rate(errors, h_values)

        print(f"\n4th-order 2nd derivative convergence rate: {rate:.3f}")
        print(f"  h values: {h_values}")
        print(f"  Errors:   {errors}")

        # Note: practical convergence rates may be slightly below theoretical
        assert rate > 3.6, f"Convergence rate {rate:.3f} below 4th order"
        assert rate < 4.3, f"Convergence rate {rate:.3f} suspiciously high"

    def test_error_reduction_ratios(self):
        """Test that errors reduce by expected ratios when h is halved."""

        def f(x):
            return np.sin(2 * np.pi * x)

        def df(x):
            return 2 * np.pi * np.cos(2 * np.pi * x)

        # For 2nd order: halving h should reduce error by ~4x
        h_coarse = 0.1
        h_fine = 0.05

        # 2nd order stencil
        x_coarse = np.arange(0, 1.0, h_coarse)
        y_coarse = f(x_coarse)
        dy_exact_coarse = df(x_coarse)
        stencil_coarse = centered_diff_1st_order2(h_coarse)

        error_coarse = 0.0
        for i in range(1, len(x_coarse) - 1):
            dy_approx = apply_stencil_1d(y_coarse, stencil_coarse, i)
            error_coarse = max(error_coarse, abs(dy_approx - dy_exact_coarse[i]))

        x_fine = np.arange(0, 1.0, h_fine)
        y_fine = f(x_fine)
        dy_exact_fine = df(x_fine)
        stencil_fine = centered_diff_1st_order2(h_fine)

        error_fine = 0.0
        for i in range(1, len(x_fine) - 1):
            dy_approx = apply_stencil_1d(y_fine, stencil_fine, i)
            error_fine = max(error_fine, abs(dy_approx - dy_exact_fine[i]))

        ratio = error_coarse / error_fine

        print(f"\nError reduction (2nd order, h halved):")
        print(f"  Error at h={h_coarse}: {error_coarse:.2e}")
        print(f"  Error at h={h_fine}:  {error_fine:.2e}")
        print(f"  Reduction ratio: {ratio:.2f} (expected ~4.0)")

        # For 2nd order, ratio should be close to 4
        assert ratio > 3.5, f"Error reduction ratio {ratio:.2f} too low for 2nd order"
        assert ratio < 5.0, f"Error reduction ratio {ratio:.2f} suspiciously high"

        # For 4th order: halving h should reduce error by ~16x
        stencil_coarse_4 = centered_diff_1st_order4(h_coarse)
        stencil_fine_4 = centered_diff_1st_order4(h_fine)

        error_coarse_4 = 0.0
        for i in range(2, len(x_coarse) - 2):
            dy_approx = apply_stencil_1d(y_coarse, stencil_coarse_4, i)
            error_coarse_4 = max(error_coarse_4, abs(dy_approx - dy_exact_coarse[i]))

        error_fine_4 = 0.0
        for i in range(2, len(x_fine) - 2):
            dy_approx = apply_stencil_1d(y_fine, stencil_fine_4, i)
            error_fine_4 = max(error_fine_4, abs(dy_approx - dy_exact_fine[i]))

        ratio_4 = error_coarse_4 / error_fine_4

        print(f"\nError reduction (4th order, h halved):")
        print(f"  Error at h={h_coarse}: {error_coarse_4:.2e}")
        print(f"  Error at h={h_fine}:  {error_fine_4:.2e}")
        print(f"  Reduction ratio: {ratio_4:.2f} (expected ~16.0)")

        # For 4th order, ratio should be close to 16
        assert ratio_4 > 14.0, f"Error reduction ratio {ratio_4:.2f} too low for 4th order"
        assert ratio_4 < 18.0, f"Error reduction ratio {ratio_4:.2f} suspiciously high"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output
