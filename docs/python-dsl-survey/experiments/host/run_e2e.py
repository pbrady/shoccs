"""Native end-to-end step timings: the container's run_e2e.sh cases, on the host build.

Each case runs the shoccs app with io disabled in its own directory so that
logs/system.csv (Wall_ms per step) lands there. Thread count via
OMP_NUM_THREADS; macOS libomp has no affinity, so no OMP_PROC_BIND.
"""
import os
import pathlib
import subprocess
import sys

W = pathlib.Path("/Users/ptb/projects/shoccs/.claude/worktrees/python-dsl-survey")
EXE = W / "build-host" / "src" / "app" / "shoccs"
ROOT = W / ".scratch" / "e2e-host"
HEAT = W / "heat.lua"
WAVE = W / "scalar_wave.lua"

H51 = "simulation.io = nil"
H201 = "simulation.io = nil; simulation.mesh.index_extents = {201, 201}; simulation.step_controller.max_time = 0.03"
H401 = "simulation.io = nil; simulation.mesh.index_extents = {401, 401}; simulation.step_controller.max_time = 0.006"
H3D = (
    "simulation.io = nil; simulation.mesh.index_extents = {65, 65, 65}; "
    "simulation.mesh.domain_bounds = {2, 2, 2}; "
    'simulation.domain_boundaries.zmin = "dirichlet"; simulation.domain_boundaries.zmax = "dirichlet"; '
    "simulation.shapes[1].center = {16/17, 25/22, 1}; "
    "simulation.manufactured_solution[1].center = {1, 1, 1}; "
    "simulation.manufactured_solution[1].variance = {1, 1, 1}; "
    "simulation.step_controller.max_time = 0.04"
)
W71 = "simulation.io = nil; simulation.step_controller.max_step = 60"

CASES = [
    ("heat51", HEAT, H51),
    ("heat201", HEAT, H201),
    ("heat401", HEAT, H401),
    ("heat3d65", HEAT, H3D),
    ("wave71", WAVE, W71),
]

# (suffix, threads, extra env)
VARIANTS = [
    ("t1", 1, {}),
    ("t4", 4, {}),
    ("t4spin", 4, {"KMP_BLOCKTIME": "infinite", "OMP_WAIT_POLICY": "active"}),
]

only = set(sys.argv[1:])
for suffix, threads, extra in VARIANTS:
    for name, cfg, script in CASES:
        case = f"{name}_{suffix}"
        if only and case not in only:
            continue
        d = ROOT / case
        d.mkdir(parents=True, exist_ok=True)
        log = d / "logs" / "system.csv"
        if log.exists():
            log.unlink()
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), **extra)
        with open(d / "run.out", "w") as out:
            rc = subprocess.call([str(EXE), str(cfg), "--script", script], cwd=d, env=env, stdout=out, stderr=subprocess.STDOUT)
        print(f"{case}: exit={rc} steps_logged={'yes' if log.exists() else 'no'}", flush=True)
