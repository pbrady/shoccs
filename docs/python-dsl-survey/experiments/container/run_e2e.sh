#!/bin/bash
# End-to-end step timings of the current C++ solver inside the devcontainer.
# Each case runs with io disabled; logs/system.csv carries a Wall_ms column per step.
set -u
cd /workspace
EXE=/workspace/.scratch/build-linux/src/app/shoccs
ROOT=/workspace/.scratch/e2e
mkdir -p "$ROOT"

run_case () {
  local name=$1 threads=$2 cfg=$3 script=$4
  local dir="$ROOT/${name}_t${threads}"
  rm -rf "$dir"; mkdir -p "$dir"; cd "$dir"
  echo "=== $name threads=$threads ==="
  if [ "$threads" = "1" ]; then
    OMP_NUM_THREADS=1 OMP_PROC_BIND=false "$EXE" "$cfg" --script "$script" > run.out 2>&1
  else
    OMP_NUM_THREADS=$threads OMP_PROC_BIND=spread OMP_PLACES=threads "$EXE" "$cfg" --script "$script" > run.out 2>&1
  fi
  echo "exit=$?"
  tail -n 3 run.out
  cd /workspace
}

HEAT=/workspace/heat.lua
WAVE=/workspace/scalar_wave.lua
H51='simulation.io = nil'
H201='simulation.io = nil; simulation.mesh.index_extents = {201, 201}; simulation.step_controller.max_time = 0.03'
H401='simulation.io = nil; simulation.mesh.index_extents = {401, 401}; simulation.step_controller.max_time = 0.006'
H3D='simulation.io = nil; simulation.mesh.index_extents = {65, 65, 65}; simulation.mesh.domain_bounds = {2, 2, 2}; simulation.domain_boundaries.zmin = "dirichlet"; simulation.domain_boundaries.zmax = "dirichlet"; simulation.shapes[1].center = {16/17, 25/22, 1}; simulation.manufactured_solution[1].center = {1, 1, 1}; simulation.manufactured_solution[1].variance = {1, 1, 1}; simulation.step_controller.max_time = 0.04'
W71='simulation.io = nil; simulation.step_controller.max_step = 60'

for T in 1 8; do
  run_case heat51 $T "$HEAT" "$H51"
  run_case heat201 $T "$HEAT" "$H201"
  run_case heat401 $T "$HEAT" "$H401"
  run_case heat3d65 $T "$HEAT" "$H3D"
  run_case wave71 $T "$WAVE" "$W71"
done
echo "all done"
