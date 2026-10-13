"""The npy loader, the layout helpers and the exact operator on the synthetic export."""

import numpy as np
import pytest

from pyshoccs.tables import ExportFormatError, Layout, expect, load_npy, load_tables


def test_load_npy_roundtrip(tmp_path):
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    b = np.linspace(0, 1, 5)
    c = np.array([1, 0, 1], dtype=np.uint8)
    for name, arr in (("a", a), ("b", b), ("c", c)):
        np.save(tmp_path / f"{name}.npy", arr)
    assert np.array_equal(load_npy(tmp_path / "a.npy"), a)
    assert load_npy(tmp_path / "a.npy").flags.c_contiguous
    assert np.array_equal(load_npy(tmp_path / "b.npy"), b)
    assert load_npy(tmp_path / "c.npy").dtype == np.uint8
    with pytest.raises(ExportFormatError, match="missing"):
        load_npy(tmp_path / "nope.npy")


def test_load_npy_fortran_order_becomes_c(tmp_path):
    a = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    np.save(tmp_path / "f.npy", a)
    out = load_npy(tmp_path / "f.npy")
    assert out.flags.c_contiguous and np.array_equal(out, a)


def test_expect_reports_file_dtype_and_shape():
    with pytest.raises(ExportFormatError, match="A0_val: dtype int64"):
        expect(np.zeros(3, np.int64), "A0_val", np.float64)
    with pytest.raises(ExportFormatError, match=r"shape \(3,\)"):
        expect(np.zeros(3), "R0_position", np.float64, shape=(None, 3))
    assert expect(np.zeros((2, 3), np.int32), "ok", np.int64, shape=(None, 3)).shape == (2, 3)


def test_layout_segments():
    L = Layout((3, 2, 1), 6, (2, 1, 0), {"D": 0, "Rx": 6, "Ry": 8, "Rz": 9}, 9)
    assert L.strides == (2, 1, 1)
    assert L.active_dims == (0, 1)
    assert list(L.segment_of([0, 5, 6, 7, 8])) == [0, 0, 1, 1, 2]
    assert L.describe_row(5) == "V[5] = D[5] (i,j,k)=(2,1,0)"
    assert L.describe_row(8) == "V[8] = Ry[0]"
    assert L.segment_slices["Rx"] == slice(6, 8)


def test_synthetic_export_loads(synthetic_tables):
    t = synthetic_tables
    assert t.layout.extents == (7, 5, 1)
    assert t.layout.n_V == 35 + 3
    assert t.R[0].count == 3 and t.R[1].count == 0
    assert t.A[0].nnz > 0 and t.A[1].nnz > 0 and t.A[2].nnz == 0
    assert t.locations.shape == (38, 3)
    assert t.locations[35, 1] == pytest.approx(t.y[1])
    assert "rand_V" in t.oracle and t.stats_u1 is not None and len(t.stats_u1) == 11
    assert not t.has_neumann


def test_to_scipy_matches_exact_apply(synthetic_tables):
    t = synthetic_tables
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, t.layout.n_V)
    A = t.to_scipy()
    assert A.shape == (38, 38)
    assert np.allclose(A @ u, t.exact_apply(u), rtol=1e-14, atol=1e-13)
    for d in (0, 1):
        assert np.allclose(t.to_scipy(dirs=(d,)) @ u, t.exact_apply(u, dirs=(d,)), rtol=1e-14, atol=1e-13)


def test_loader_rejects_bad_offsets(tmp_path, synthetic_dir):
    import json
    import shutil

    bad = tmp_path / "bad"
    shutil.copytree(synthetic_dir, bad)
    meta = json.loads((bad / "meta.json").read_text())
    meta["offsets"]["Rx"] = 34
    (bad / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ExportFormatError, match="offsets"):
        load_tables(bad)
