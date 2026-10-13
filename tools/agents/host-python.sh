#!/bin/sh
# Run a Python script from this repository with the stencil_gen uv environment on the
# HOST, outside the Bash sandbox, so that Metal, PyTorch MPS and MLX are reachable
# (the macOS sandbox hides the GPU: torch.backends.mps.is_available() is False and
# mlx.core fails to load a Metal device inside it).
#
# Only scripts under scripts/ or docs/python-dsl-survey/experiments/ are accepted, so
# allow-listing this wrapper does not amount to allow-listing arbitrary unsandboxed code.
#
#   tools/agents/host-python.sh [--with PKG]... <script.py> [script args...]
#
# Example (from the repository root):
#   tools/agents/host-python.sh --with mlx docs/python-dsl-survey/experiments/host_primitives_bench.py --cases 51
set -eu

here=$(cd "$(dirname "$0")/../.." && pwd)
withs=""

while [ $# -gt 0 ]; do
    case "$1" in
        --with) withs="$withs --with $2"; shift 2 ;;
        -*) echo "host-python: unknown option $1" >&2; exit 64 ;;
        *) break ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "usage: $0 [--with PKG]... <script.py> [args...]" >&2
    exit 64
fi

script=$1
shift
case "$script" in
    /*) abs=$script ;;
    *) abs=$here/$script ;;
esac
if [ ! -f "$abs" ]; then
    echo "host-python: no such script: $script" >&2
    exit 66
fi
abs=$(cd "$(dirname "$abs")" && pwd)/$(basename "$abs")

case "$abs" in
    "$here"/scripts/*|"$here"/docs/python-dsl-survey/experiments/*) ;;
    *)
        echo "host-python: refusing $script (only scripts/ and docs/python-dsl-survey/experiments/ are allowed)" >&2
        exit 65
        ;;
esac

cd "$here/scripts/stencil_gen"
# shellcheck disable=SC2086
exec uv run $withs python "$abs" "$@"
