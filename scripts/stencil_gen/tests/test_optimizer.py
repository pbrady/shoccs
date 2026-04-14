"""Tests for :mod:`stencil_gen.optimizer` (plan 43)."""

from __future__ import annotations

import numpy as np
import pytest

from stencil_gen.optimizer import (
    DEFAULT_BOUNDS,
    params_from_vector,
    vector_from_params,
)


class TestParamsVector:
    @pytest.mark.parametrize(
        ("kernel", "x", "expected"),
        [
            ("tension", [3.0], {"sigma": 3.0}),
            ("gaussian", [1.5], {"epsilon": 1.5}),
            ("multiquadric", [0.7], {"epsilon": 0.7}),
            ("tension-penalty", [2.0, 10.0], {"sigma": 2.0, "gamma": 10.0}),
            ("mixed-epsilon", [1.0, 2.0, 3.0, 4.0], {"epsilons": [1.0, 2.0, 3.0, 4.0]}),
            ("classical", [-0.5, 197.0 / 288.0], {"alpha": [-0.5, 197.0 / 288.0]}),
        ],
    )
    def test_params_from_vector(self, kernel, x, expected):
        assert params_from_vector(kernel, np.asarray(x)) == expected

    @pytest.mark.parametrize(
        ("kernel", "params", "expected_x"),
        [
            ("tension", {"sigma": 3.0}, [3.0]),
            ("gaussian", {"epsilon": 1.5}, [1.5]),
            ("multiquadric", {"epsilon": 0.7}, [0.7]),
            ("tension-penalty", {"sigma": 2.0, "gamma": 10.0}, [2.0, 10.0]),
            ("mixed-epsilon", {"epsilons": [1.0, 2.0, 3.0, 4.0]}, [1.0, 2.0, 3.0, 4.0]),
            ("classical", {"alpha": [-0.5, 1.0]}, [-0.5, 1.0]),
        ],
    )
    def test_vector_from_params(self, kernel, params, expected_x):
        v = vector_from_params(kernel, params)
        assert v.dtype == np.float64
        np.testing.assert_array_equal(v, np.asarray(expected_x, dtype=float))

    @pytest.mark.parametrize(
        ("kernel", "params"),
        [
            ("tension", {"sigma": 3.0}),
            ("gaussian", {"epsilon": 1.5}),
            ("multiquadric", {"epsilon": 0.7}),
            ("tension-penalty", {"sigma": 2.0, "gamma": 10.0}),
            ("mixed-epsilon", {"epsilons": [1.0, 2.0, 3.0, 4.0]}),
            ("classical", {"alpha": [-0.5, 1.0]}),
        ],
    )
    def test_roundtrip_params_vector_params(self, kernel, params):
        v = vector_from_params(kernel, params)
        assert params_from_vector(kernel, v) == params

    @pytest.mark.parametrize(
        ("kernel", "x"),
        [
            ("tension", [1.0, 2.0]),
            ("gaussian", [1.0, 2.0]),
            ("multiquadric", []),
            ("tension-penalty", [1.0]),
            ("classical", [1.0]),
            ("mixed-epsilon", []),
        ],
    )
    def test_params_from_vector_wrong_dim(self, kernel, x):
        with pytest.raises(ValueError):
            params_from_vector(kernel, np.asarray(x, dtype=float))

    def test_params_from_vector_unknown_kernel(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            params_from_vector("nope", np.array([1.0]))

    def test_vector_from_params_unknown_kernel(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            vector_from_params("nope", {})

    def test_classical_requires_length_two_alpha(self):
        with pytest.raises(ValueError):
            vector_from_params("classical", {"alpha": [0.1]})

    def test_default_bounds_shapes_match_roundtrip(self):
        # For each entry in DEFAULT_BOUNDS, build a midpoint vector and round-trip
        # through params_from_vector / vector_from_params.
        for (scheme, kernel), bounds in DEFAULT_BOUNDS.items():
            mid = np.array([0.5 * (lo + hi) for (lo, hi) in bounds], dtype=float)
            params = params_from_vector(kernel, mid)
            v = vector_from_params(kernel, params)
            assert v.shape == mid.shape, f"{scheme}/{kernel}: shape mismatch"
            np.testing.assert_array_equal(v, mid)
