"""Shared fixtures for stencil_gen tests."""

import pytest


def run_pipeline(p, nu=1, s=0):
    """Run derive_boundary + conservation, return full pipeline results."""
    from stencil_gen.boundary import derive_boundary
    from stencil_gen.conservation import build_conservation_system, solve_conservation

    result = derive_boundary(p=p, nu=nu, s=s)
    equations, w_syms, last_free = build_conservation_system(
        result.r, result.t, p, result.rows, result.interior_coeffs
    )
    solution_dict, updated_rows = solve_conservation(
        equations, w_syms, last_free, result.all_free_params, result.rows
    )
    return updated_rows, solution_dict, w_syms, result


@pytest.fixture(scope="module")
def e4u_pipeline():
    """Run E4u pipeline once per module."""
    return run_pipeline(p=2)


@pytest.fixture(scope="module")
def e6u_pipeline():
    """Run E6u pipeline once per module."""
    return run_pipeline(p=3)


@pytest.fixture(scope="module")
def e8u_pipeline():
    """Run E8u pipeline once per module."""
    return run_pipeline(p=4)
