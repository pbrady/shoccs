"""The five parity checks: plumbing on the synthetic export, the gate on the real one."""

import numpy as np
import pytest

from pyshoccs import parity
from pyshoccs.heat import HeatSystem
from pyshoccs.structure import derive_structure


# -- synthetic export: every check runs and passes by construction ------------


def test_synthetic_run_all(synthetic_dir, tmp_path):
    report = parity.run_all(synthetic_dir, tmp_path / "parity.json", verbose=False)
    names = {c["name"]: c for c in report["checks"]}
    for key in ("1 operator", "2 rhs identity (rhs(rand_V)", "2 rhs identity (rhs(u0_V)",
                "3 monomial", "4 step", "4b stats", "5 structural"):
        hit = [c for n, c in names.items() if n.startswith(key)]
        assert hit, key
        assert hit[0]["passed"], hit[0]
    assert report["all_passed"]
    assert (tmp_path / "parity.json").is_file()
    # the one-sided three-point closures fail the k = 3 monomial, as a real closure would
    assert "k=3" in " ".join(names["3 monomial oracle (R rows, k <= 2)"]["notes"])
    assert report["oracle_summary"]["R:k=3"]["n_fail"] == 2


def test_synthetic_row_classes(synthetic_tables):
    c = parity.classify_rows(synthetic_tables)
    # solid points on a Dirichlet face count as Dirichlet (update_boundary writes them)
    assert c.counts["solid_D"] == 3 and c.counts["R"] == 2 and c.counts["R_dirichlet"] == 1
    assert c.counts["dirichlet_D"] == 20 and c.counts["clamp_D"] == 0
    assert c.counts["fluid_D"] == 12  # i in 1..4, j in 1..3


def test_synthetic_structured_backend_steps_like_scipy(synthetic_tables):
    t = synthetic_tables
    a = HeatSystem(t, backend="scipy")
    b = HeatSystem(t, backend="structured")
    u0 = t.oracle["u0_V"]
    ua = a.rk4_step(u0, a.t0, a.dt).copy()
    ub = b.rk4_step(u0, b.t0, b.dt).copy()
    assert np.allclose(ua, ub, rtol=1e-13, atol=1e-14)
    assert np.allclose(ua, t.oracle["u1_V"], rtol=1e-13, atol=1e-14)
    assert a.stats(ua, a.t0 + a.dt) == pytest.approx(t.stats_u1, rel=1e-12)


def test_synthetic_mms_vectors(synthetic_tables):
    t = synthetic_tables
    s = HeatSystem(t)
    assert np.allclose(s.solution(s.t0), t.oracle["sol_t0_V"], rtol=1e-14, atol=0)
    assert np.allclose(s.source(s.t0 + s.dt / 2), t.oracle["src_t1_V"], rtol=1e-14, atol=0)


# -- real export (skipped without one) -----------------------------------------


@pytest.fixture(scope="module")
def real(tables):
    classes = parity.classify_rows(tables)
    structure = derive_structure(tables)
    system = HeatSystem(tables, backend="scipy", structure=structure)
    return tables, classes, structure, system


def test_mms_against_export(real):
    tables, _, _, system = real
    for name, t, fn in (("sol_t0_V", system.t0, system.solution),
                        ("sol_t1_V", system.t0 + system.dt / 2, system.solution),
                        ("src_t0_V", system.t0, system.source),
                        ("src_t1_V", system.t0 + system.dt / 2, system.source)):
        if name not in tables.oracle:
            pytest.skip(f"{name} absent")
        ref = tables.oracle[name]
        got = fn(t)
        diff = np.abs(got - ref)
        scale = np.abs(ref).max()
        w = int(np.argmax(diff))
        assert diff.max() <= 1e-13 * scale, (
            f"{name}: max |diff| {diff.max():.3e} vs scale {scale:.3e} at "
            f"{tables.layout.describe_row(w)} (python {got[w]!r}, cpp {ref[w]!r})")


def test_check_operator(real):
    tables, classes, _, _ = real
    r = parity.check_operator(tables, classes)
    assert r.passed, r.message()


def test_check_rhs(real):
    tables, classes, _, system = real
    for r in parity.check_rhs(tables, system, classes):
        assert r.passed, r.message()


def test_check_oracle(real):
    tables, classes, _, _ = real
    r, orc = parity.check_oracle(tables, classes)
    assert r.passed, r.message() + "\n" + orc.format()


def test_check_step(real):
    tables, classes, _, system = real
    for r in parity.check_step(tables, system, classes):
        assert r.passed, r.message()


def test_check_structure(real):
    tables, classes, structure, _ = real
    r = parity.check_structure(tables, structure, classes)
    assert r.passed, r.message()


def test_structured_backend_rhs(real):
    tables, classes, structure, scipy_system = real
    s = HeatSystem(tables, backend="structured", structure=structure)
    u = tables.oracle["rand_V"]
    a = scipy_system.rhs(u, s.t0).copy()
    b = s.rhs(u, s.t0).copy()
    diff = np.abs(a - b)
    w = int(np.argmax(diff))
    assert diff.max() <= 1e-13 * np.abs(a).max(), (
        f"structured vs scipy rhs: max |diff| {diff.max():.3e} at {tables.layout.describe_row(w)} "
        f"class {classes.name(w)}")


def test_exporter_cross_check(real):
    tables, _, structure, _ = real
    from pyshoccs.structure import compare_with_export

    rep = compare_with_export(tables, structure)
    if not rep["role"].get("present"):
        pytest.skip("exporter wrote no role.npy")
    assert rep["role"]["match"], rep["role"]
    if rep["exc"].get("present") and "match" in rep["exc"]:
        assert rep["exc"]["match"], rep["exc"]
