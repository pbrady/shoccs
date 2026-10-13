"""Run bench/bench_heat.py over cases x thread settings x runs, one process each.

Usage: python run_matrix.py [backends] [cases] [settings] [runs]
  backends  comma list or alias for bench_heat.py (default cpu)
  cases     comma list of export directory names (default heat51,heat201,heat3d65)
  settings  comma list of 1t,4t,4t_spin (default all three)
  runs      comma list of run numbers (default 1,2)

The loops are run outermost, then setting, then case, so the two runs of a
cell are separated in time. Each process writes
.scratch/phase1/py_arm/<case>_<setting>_run<n>.json and a .log beside it.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/ptb/projects/shoccs/.claude/worktrees/python-dsl-survey")
PKG = ROOT / "python" / "pyshoccs"
OUT = ROOT / ".scratch" / "phase1" / "py_arm"
CASES = {"heat51": (100, 10), "heat201": (60, 10), "heat3d65": (20, 5)}  # reps, warm
SETTINGS = {
    "1t": (1, {}),
    "4t": (4, {}),
    "4t_spin": (4, {"KMP_BLOCKTIME": "infinite", "OMP_WAIT_POLICY": "active"}),
    # the MLX and GPU backends, run outside the sandbox with one host thread
    "metal": (1, {}),
}


def main(argv):
    backends = argv[1] if len(argv) > 1 else "cpu"
    cases = argv[2].split(",") if len(argv) > 2 else list(CASES)
    settings = argv[3].split(",") if len(argv) > 3 else list(SETTINGS)
    runs = [int(r) for r in argv[4].split(",")] if len(argv) > 4 else [1, 2]
    OUT.mkdir(parents=True, exist_ok=True)
    for run in runs:
        for setting in settings:
            threads, extra = SETTINGS[setting]
            for case in cases:
                reps, warm = CASES[case]
                env = dict(os.environ)
                env.update(extra)
                out = OUT / f"{case}_{setting}_run{run}.json"
                log = OUT / f"{case}_{setting}_run{run}.log"
                cmd = ["uv", "run", "python", "bench/bench_heat.py",
                       "--export", str(ROOT / ".scratch" / "export" / case),
                       "--reps", str(reps), "--warm", str(warm), "--threads", str(threads),
                       "--backends", backends, "--out", str(out)]
                t = time.time()
                with open(log, "w") as lf:
                    r = subprocess.run(cmd, cwd=PKG, env=env, stdout=lf, stderr=subprocess.STDOUT)
                print(f"{case:9s} {setting:8s} run{run} rc={r.returncode} {time.time() - t:6.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
