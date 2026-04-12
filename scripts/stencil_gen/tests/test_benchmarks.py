"""Tests for stencil_gen.benchmarks modules."""

import math

import numpy as np
import pytest

from stencil_gen.benchmarks.brady_livescu_2d import (
    L_DOMAIN,
    PSI_OFFSET,
    c_x,
    c_y,
    exact_solution,
    inflow_bc_x,
    inflow_bc_y,
    initial_condition,
    make_coefficient_field,
    psi,
)


class TestBradyLivescu2D:
    """Tests for the Brady & Livescu 2019 §4.3 reference problem."""

    def test_constants(self):
        assert L_DOMAIN == pytest.approx(math.sqrt(2.0))
        assert PSI_OFFSET == pytest.approx(0.25)

    def test_exact_solution_initial_time(self):
        """exact_solution(x, y, 0) == initial_condition(x, y) at sample points."""
        rng = np.random.default_rng(42)
        xs = rng.uniform(0.0, L_DOMAIN, 5)
        ys = rng.uniform(0.0, L_DOMAIN, 5)
        for x_val, y_val in zip(xs, ys):
            assert exact_solution(x_val, y_val, 0.0) == pytest.approx(
                initial_condition(x_val, y_val)
            )

    def test_exact_solution_satisfies_pde(self):
        """Central-difference approximation of u_t + c_x*u_x + c_y*u_y ~ 0."""
        rng = np.random.default_rng(123)
        h = 1e-6
        for _ in range(5):
            x_val = rng.uniform(0.1, L_DOMAIN - 0.1)
            y_val = rng.uniform(0.1, L_DOMAIN - 0.1)
            t_val = rng.uniform(0.1, 10.0)

            # u_t via central difference
            u_t = (
                exact_solution(x_val, y_val, t_val + h)
                - exact_solution(x_val, y_val, t_val - h)
            ) / (2.0 * h)

            # u_x via central difference
            u_x = (
                exact_solution(x_val + h, y_val, t_val)
                - exact_solution(x_val - h, y_val, t_val)
            ) / (2.0 * h)

            # u_y via central difference
            u_y = (
                exact_solution(x_val, y_val + h, t_val)
                - exact_solution(x_val, y_val - h, t_val)
            ) / (2.0 * h)

            residual = u_t + c_x(x_val, y_val) * u_x + c_y(x_val, y_val) * u_y
            assert abs(residual) < 1e-6, (
                f"PDE residual {residual} at ({x_val}, {y_val}, {t_val})"
            )

    def test_coefficient_field_shape(self):
        """make_coefficient_field(31) returns (31, 31) arrays with unit speed."""
        x, y, cx, cy = make_coefficient_field(31)
        assert x.shape == (31, 31)
        assert y.shape == (31, 31)
        assert cx.shape == (31, 31)
        assert cy.shape == (31, 31)
        speed = np.sqrt(cx**2 + cy**2)
        np.testing.assert_allclose(speed, 1.0, atol=1e-14)

    def test_inflow_bc_matches_exact_at_edges(self):
        """Inflow BCs match exact solution at the domain edges."""
        rng = np.random.default_rng(77)
        ys = rng.uniform(0.0, L_DOMAIN, 5)
        ts = rng.uniform(0.0, 100.0, 5)
        for y_val, t_val in zip(ys, ts):
            assert inflow_bc_x(y_val, t_val) == pytest.approx(
                exact_solution(0.0, y_val, t_val)
            )

        xs = rng.uniform(0.0, L_DOMAIN, 5)
        ts = rng.uniform(0.0, 100.0, 5)
        for x_val, t_val in zip(xs, ts):
            assert inflow_bc_y(x_val, t_val) == pytest.approx(
                exact_solution(x_val, 0.0, t_val)
            )
