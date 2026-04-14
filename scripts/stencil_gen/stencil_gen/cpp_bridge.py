"""Python → C++ bridge for the Brady-Livescu 2D stability validator.

Builds Lua configs from a template and drives the compiled shoccs binary so
plan 41's analytical stability stack can validate survivors end-to-end in the
real solver. Plan 42.2a scope: path constants, BridgeResult, make_brady2d_lua.
The run_cpp_brady2d driver is added in 42.2b.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT: Path = Path(__file__).resolve().parents[3]
LUA_TEMPLATE_DIR: Path = REPO_ROOT / "lua-configs"
BRADY_LIVESCU_TEMPLATE: Path = LUA_TEMPLATE_DIR / "brady_livescu_4_3.lua"
SHOCCS_BINARY: Path = REPO_ROOT / "build" / "src" / "app" / "shoccs"


@dataclass
class BridgeResult:
    final_linf: float
    linf_trace: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_trace: np.ndarray = field(default_factory=lambda: np.empty(0))
    stable: bool = False
    wall_time_s: float = 0.0
    exit_code: int = 0
    stderr: str = ""


def _format_lua_number(x: float) -> str:
    """Format a Python float as a Lua-parseable numeric literal.

    Uses repr() to preserve full double precision, which matters for the alpha
    coefficients that must match known_values.json exactly.
    """
    return repr(float(x))


def _format_lua_array(values: list[float]) -> str:
    return "{" + ", ".join(_format_lua_number(v) for v in values) + "}"


def _scheme_table_for(scheme_type: str, params: dict[str, Any]) -> str:
    """Emit a Lua scheme sub-table for the requested type and parameters.

    Classical schemes pass `alpha` as an array. Spline families (added in
    42.7) pass a scalar `sigma` or `epsilon`. The shape of params dictates
    which parameter is emitted.
    """
    entries: list[str] = ["order = 1", f'type = "{scheme_type}"']
    if "alpha" in params:
        entries.append(f"alpha = {_format_lua_array(list(params['alpha']))}")
    if "sigma" in params:
        entries.append(f"sigma = {_format_lua_number(params['sigma'])}")
    if "epsilon" in params:
        entries.append(f"epsilon = {_format_lua_number(params['epsilon'])}")
    return "{ " + ", ".join(entries) + " }"


def make_brady2d_lua(
    scheme_type: str,
    params: dict[str, Any],
    *,
    N: int,
    t_final: float,
    template: Path = BRADY_LIVESCU_TEMPLATE,
) -> str:
    """Render the Brady-Livescu 2D Lua config with the supplied parameters.

    Substitutes the three explicit markers --{{N}}--, --{{T_FINAL}}--, and
    --{{SCHEME_TABLE}}-- with their runtime values. No regex, no Lua AST
    parsing — just str.replace().
    """
    src = template.read_text()
    src = src.replace("--{{N}}--", str(int(N)))
    src = src.replace("--{{T_FINAL}}--", _format_lua_number(t_final))
    src = src.replace("--{{SCHEME_TABLE}}--", _scheme_table_for(scheme_type, params))
    return src
