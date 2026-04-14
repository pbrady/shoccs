"""Tests for scripts/stencil_gen/stencil_gen/cpp_bridge.py.

Plan 42.2a scope: make_brady2d_lua marker substitution and scheme-table
emission. The subprocess-driven run_cpp_brady2d tests come in 42.2b/42.2c.
"""

from __future__ import annotations

import re

import pytest

from stencil_gen.cpp_bridge import (
    BRADY_LIVESCU_TEMPLATE,
    LUA_TEMPLATE_DIR,
    REPO_ROOT,
    SHOCCS_BINARY,
    BridgeResult,
    make_brady2d_lua,
)


class TestBridgePaths:
    def test_repo_root_contains_expected_markers(self):
        assert (REPO_ROOT / "CMakeLists.txt").exists()
        assert (REPO_ROOT / "lua-configs").is_dir()

    def test_lua_template_dir_matches_repo_root(self):
        assert LUA_TEMPLATE_DIR == REPO_ROOT / "lua-configs"

    def test_brady_livescu_template_exists(self):
        assert BRADY_LIVESCU_TEMPLATE.exists()
        text = BRADY_LIVESCU_TEMPLATE.read_text()
        for marker in ("--{{N}}--", "--{{T_FINAL}}--", "--{{SCHEME_TABLE}}--"):
            assert marker in text, f"template missing marker {marker!r}"

    def test_shoccs_binary_path_matches_plan(self):
        assert SHOCCS_BINARY == REPO_ROOT / "build" / "src" / "app" / "shoccs"


class TestMakeBrady2DLua:
    CLASSICAL_ALPHA = [-0.7733323791884821, 0.1623961700641681]

    def _render_classical(self, **overrides) -> str:
        kwargs = dict(
            scheme_type="E4u",
            params={"alpha": self.CLASSICAL_ALPHA},
            N=31,
            t_final=10.0,
        )
        kwargs.update(overrides)
        return make_brady2d_lua(**kwargs)

    def test_all_markers_replaced(self):
        rendered = self._render_classical()
        for marker in ("--{{N}}--", "--{{T_FINAL}}--", "--{{SCHEME_TABLE}}--"):
            assert marker not in rendered, f"marker {marker!r} survived substitution"

    def test_n_substituted_as_integer(self):
        rendered = self._render_classical(N=41)
        # index_extents = {41, 41}
        assert re.search(r"index_extents\s*=\s*\{41,\s*41\}", rendered)

    def test_t_final_substituted(self):
        rendered = self._render_classical(t_final=5.5)
        assert re.search(r"max_time\s*=\s*5\.5", rendered)

    def test_scheme_table_contains_type_and_alpha(self):
        rendered = self._render_classical()
        # Expect scheme = { order = 1, type = "E4u", alpha = {-0.77..., 0.16...} }
        assert re.search(r'type\s*=\s*"E4u"', rendered)
        for val in self.CLASSICAL_ALPHA:
            assert repr(val) in rendered, f"alpha coefficient {val!r} missing"
        assert re.search(r"alpha\s*=\s*\{", rendered)

    def test_scheme_table_with_sigma(self):
        rendered = make_brady2d_lua(
            scheme_type="tension_E4u",
            params={"sigma": 3.0},
            N=31,
            t_final=10.0,
        )
        assert re.search(r'type\s*=\s*"tension_E4u"', rendered)
        assert re.search(r"sigma\s*=\s*3\.0", rendered)
        assert "alpha" not in rendered.split("scheme")[1].split("system")[0]

    def test_scheme_table_with_epsilon(self):
        rendered = make_brady2d_lua(
            scheme_type="gaussian_E4u",
            params={"epsilon": 0.9},
            N=31,
            t_final=10.0,
        )
        assert re.search(r'type\s*=\s*"gaussian_E4u"', rendered)
        assert re.search(r"epsilon\s*=\s*0\.9", rendered)

    def test_rendered_output_is_lua_like(self):
        """Basic sanity: every Lua brace must be balanced after substitution."""
        rendered = self._render_classical()
        assert rendered.count("{") == rendered.count("}")
        assert "simulation" in rendered
        assert "mesh" in rendered
        assert "system" in rendered

    def test_int_like_float_for_t_final(self):
        # 10.0 should appear with a decimal point so Lua parses it as number
        rendered = self._render_classical(t_final=10.0)
        assert re.search(r"max_time\s*=\s*10\.0", rendered)


class TestBridgeResultDefaults:
    def test_default_construction(self):
        r = BridgeResult(final_linf=0.0)
        assert r.stable is False
        assert r.wall_time_s == 0.0
        assert r.exit_code == 0
        assert r.stderr == ""
        assert r.linf_trace.shape == (0,)
        assert r.t_trace.shape == (0,)

    def test_explicit_fields(self):
        import numpy as np

        r = BridgeResult(
            final_linf=0.123,
            linf_trace=np.array([0.0, 0.1, 0.123]),
            t_trace=np.array([0.0, 0.5, 1.0]),
            stable=True,
            wall_time_s=4.2,
            exit_code=0,
            stderr="",
        )
        assert r.stable is True
        assert r.final_linf == pytest.approx(0.123)
        assert r.linf_trace.shape == (3,)
