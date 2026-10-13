"""Shared fixtures: the real export (skipped when absent) and a synthetic one."""

from __future__ import annotations

import pytest

from pyshoccs.tables import default_export_dir, load_tables

from synthetic import write_synthetic_export


@pytest.fixture(scope="session")
def export_dir():
    p = default_export_dir()
    if p is None:
        pytest.skip("no export directory (set PYSHOCCS_EXPORT_DIR or create .scratch/export/heat51)")
    return p


@pytest.fixture(scope="session")
def tables(export_dir):
    return load_tables(export_dir)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_export") / "synth7x5"
    write_synthetic_export(root)
    return root


@pytest.fixture(scope="session")
def synthetic_tables(synthetic_dir):
    return load_tables(synthetic_dir)
