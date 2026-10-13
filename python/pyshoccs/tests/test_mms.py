"""The gaussian manufactured solution against finite differences."""

import numpy as np
import pytest

from pyshoccs.mms import Gaussian, HoistedGaussian


def make(dims):
    if dims == 2:
        return Gaussian(2, [[0.4, 0.5], [1.5, 1.6]], [[0.5, 0.2], [0.3, 0.8]], [2.0, -1.2], [0.5, 0.8])
    return Gaussian(3, [[0.4, 0.5, 0.3], [1.5, 1.6, 0.9]], [[0.5, 0.2, 0.7], [0.3, 0.8, 0.4]],
                    [2.0, -1.2], [0.5, 0.8])


def points(dims, n=40, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 2.0, size=(n, 3))
    if dims == 2:
        p[:, 2] = 0.0
    return p


@pytest.mark.parametrize("dims", [2, 3])
def test_ddt_matches_time_difference(dims):
    ms, p, t = make(dims), points(dims), 0.7
    e = 1e-4
    fd = (ms.value(t + e, p) - ms.value(t - e, p)) / (2 * e)
    assert np.allclose(ms.ddt(t, p), fd, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("dims", [2, 3])
def test_gradient_matches_space_difference(dims):
    ms, p, t = make(dims), points(dims), 0.7
    e = 1e-5
    g = ms.gradient(t, p)
    for a in range(dims):
        pp, pm = p.copy(), p.copy()
        pp[:, a] += e
        pm[:, a] -= e
        fd = (ms.value(t, pp) - ms.value(t, pm)) / (2 * e)
        assert np.allclose(g[:, a], fd, rtol=1e-6, atol=1e-8)
    for a in range(dims, 3):
        assert np.all(g[:, a] == 0.0)


@pytest.mark.parametrize("dims", [2, 3])
def test_laplacian_matches_second_differences(dims):
    ms, p, t = make(dims), points(dims), 0.7
    v0 = ms.value(t, p)

    def second_differences(e):
        lap = np.zeros(p.shape[0])
        for a in range(dims):
            pp, pm = p.copy(), p.copy()
            pp[:, a] += e
            pm[:, a] -= e
            lap += (ms.value(t, pp) - 2 * v0 + ms.value(t, pm)) / e**2
        return lap

    # Richardson extrapolation removes the O(e^2) truncation term
    e = 2e-3
    lap = (4 * second_differences(e / 2) - second_differences(e)) / 3
    assert np.allclose(ms.laplacian(t, p), lap, rtol=1e-7, atol=1e-7)


def test_expanded_cpp_form_2d_agrees():
    """The C++ polynomial form of the 2D laplacian (gauss2d.cpp) to rounding."""
    ms, p, t = make(2), points(2), 0.3
    x, y = p[:, 0], p[:, 1]
    ref = np.zeros(p.shape[0])
    for i in range(ms.n_terms):
        cx, cy = ms.center[i, :2]
        vx, vy = ms.variance[i, :2]
        A, f = ms.amplitude[i], ms.frequency[i]
        ref += (np.exp(-0.5 * (x - cx) ** 2 * vx**-2 + -0.5 * (y - cy) ** 2 * vy**-2) * A * np.cos(t * f)
                * vx**-4 * vy**-4
                * (((y**2 * vx**4 + (-2 * y * cy * vx**4 + (cy**2 * vx**4 + (x**2 * vy**4
                    + (-2 * x * cx * vy**4 + cx**2 * vy**4))))) - vx**2 * vy**4) - vx**4 * vy**2))
    lap = ms.laplacian(t, p)
    assert np.max(np.abs(lap - ref)) <= 1e-13 * np.max(np.abs(ref))


def test_hoisted_matches_direct():
    ms, p = make(3), points(3)
    k = 1 / 30
    h = HoistedGaussian(ms, p, k)
    for t in (0.0, 0.37, 2.1):
        assert np.allclose(h.value(t), ms.value(t, p), rtol=1e-14, atol=1e-15)
        assert np.allclose(h.source(t), ms.source(t, p, k), rtol=1e-13, atol=1e-14)


def test_from_meta_pads_2d_parameters():
    meta = {"dims": 2, "mms": {"type": "gaussian", "center": [[1, 1]], "variance": [[1, 1]],
                                "amplitude": [2], "frequency": [0.5]}}
    ms = Gaussian.from_meta(meta)
    assert ms.center.shape == (1, 3) and ms.variance[0, 2] == 1.0
    v = ms.value(0.0, np.array([[1.0, 1.0, 0.0]]))
    assert v[0] == pytest.approx(2.0)
