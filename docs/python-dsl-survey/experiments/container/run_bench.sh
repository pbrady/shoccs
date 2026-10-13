#!/bin/bash
# Run the micro-benchmarks built in .scratch/build-linux inside the devcontainer.
set -u
cd /workspace
B=.scratch/build-linux/benchmarks
OUT=.scratch/bench-linux
mkdir -p "$OUT"
echo "nproc=$(nproc)"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
ldd .scratch/build-linux/src/app/shoccs | grep -iE 'omp|kokkos' || true
for b in bench_expr bench_selection bench_block bench_derivative bench_rhs; do
  echo "=== $b ==="
  "$B/$b" --benchmark_min_time=0.3s --benchmark_out="$OUT/$b.json" --benchmark_out_format=json --benchmark_format=console 2>&1 | tail -n 60
done
echo "done"
