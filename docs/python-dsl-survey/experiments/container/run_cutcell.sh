#!/bin/bash
# Build and run the cut-cell RHS benchmark inside the devcontainer at 1 and 8 threads.
set -u
cd /workspace
cmake --build .scratch/build-linux --target bench_rhs_cutcell > .scratch/cutcell-build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "BUILD FAILED rc=$rc"; tail -n 80 .scratch/cutcell-build.log; exit 1
fi
mkdir -p .scratch/bench-linux
for T in 1 8; do
  echo "=== threads=$T ==="
  if [ "$T" = "1" ]; then
    OMP_NUM_THREADS=1 OMP_PROC_BIND=false .scratch/build-linux/benchmarks/bench_rhs_cutcell \
      --benchmark_min_time=0.3s --benchmark_out=.scratch/bench-linux/bench_rhs_cutcell_t1.json \
      --benchmark_out_format=json 2>&1 | grep -E '^(BM_|---|Bench|terminate|what)'
  else
    OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads .scratch/build-linux/benchmarks/bench_rhs_cutcell \
      --benchmark_min_time=0.3s --benchmark_out=.scratch/bench-linux/bench_rhs_cutcell_t8.json \
      --benchmark_out_format=json 2>&1 | grep -E '^(BM_|---|Bench|terminate|what)'
  fi
done
echo "done"
