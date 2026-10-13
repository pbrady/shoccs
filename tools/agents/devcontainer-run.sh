#!/bin/sh
# Run one command inside the pinned SHOCCS devcontainer image with this repository
# mounted at /workspace. Nothing else is mounted and no other flags are forwarded.
#
# Purpose: the C++ dependency stack (Kokkos, fmt, spdlog, sol2, lapackpp, ...) exists
# only in the image's Spack view, so agent sessions need Docker to build and benchmark
# the solver. Allow-listing this script (and excluding it from the Bash sandbox, which
# denies the Docker socket) lets that happen without granting arbitrary `docker run`.
#
#   tools/agents/devcontainer-run.sh [--image IMAGE] [--cpus N] [--print] -- <command...>
#
# Examples (from the repository root):
#   tools/agents/devcontainer-run.sh -- cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release
#   tools/agents/devcontainer-run.sh -- cmake --build build-linux --target bench_rhs_cutcell
#   tools/agents/devcontainer-run.sh -- build-linux/benchmarks/bench_rhs_cutcell --benchmark_min_time=0.3s
#
# SHOCCS_DEVCONTAINER_IMAGE overrides the default image tag.
set -eu

here=$(cd "$(dirname "$0")/../.." && pwd)
image=${SHOCCS_DEVCONTAINER_IMAGE:-shoccs-devcontainer:latest}
cpus=""
print=0

while [ $# -gt 0 ]; do
    case "$1" in
        --image) image=$2; shift 2 ;;
        --cpus) cpus="--cpus=$2"; shift 2 ;;
        --print) print=1; shift ;;
        --) shift; break ;;
        -*) echo "devcontainer-run: unknown option $1" >&2; exit 64 ;;
        *) break ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "usage: $0 [--image IMAGE] [--cpus N] [--print] -- <command...>" >&2
    exit 64
fi

if [ "$print" -eq 1 ]; then
    echo docker run --rm $cpus -v "$here:/workspace" -w /workspace "$image" "$@"
    exit 0
fi

# shellcheck disable=SC2086
exec docker run --rm $cpus -v "$here:/workspace" -w /workspace "$image" "$@"
