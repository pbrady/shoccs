"""Render RESULTS.md of the Python array arm from the bench_heat.py JSONs in this directory.

Files: <case>_<setting>_run<n>.json (sandboxed CPU matrix, settings 1t, 4t,
4t_spin) and <case>_metal_run<n>.json (the unsandboxed MLX CPU fp64 and GPU
fp32 runs). For every (case, setting, backend, metric) the lower of the run
medians is kept. Narrative paragraphs live in NARRATIVE below and are written
after the numbers were read.

Usage: python summarize.py [dir] > RESULTS.md
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE

CASES = ["heat51", "heat201", "heat3d65"]
CASE_LABEL = {"heat51": "heat.lua 51x51", "heat201": "201x201", "heat3d65": "65^3 sphere"}
SETTINGS = ["1t", "4t", "4t_spin", "metal"]
SETTING_LABEL = {"1t": "1 thread", "4t": "4 threads", "4t_spin": "4 threads spin", "metal": "Metal (unsandboxed)"}
BACKENDS = ["scipy", "structured", "torch-eager", "torch-compile", "mlx", "torch-mps", "mlx-gpu"]
BACKEND_LABEL = {
    "scipy": "scipy CSR (fp64)",
    "structured": "structured NumPy (fp64)",
    "torch-eager": "torch eager CPU (fp64)",
    "torch-compile": "torch.compile CPU (fp64)",
    "mlx": "MLX CPU compiled (fp64 arrays, cos/sin in fp32)",
    "torch-mps": "torch MPS eager (fp32)",
    "mlx-gpu": "MLX GPU compiled (fp32)",
}
METRICS = ["operator", "rhs", "source", "boundary", "stats", "step", "step_nostats", "axpy"]
# the C++ setting each Python setting is compared with
BAR_SETTING = {"1t": "1t", "4t": "4t", "4t_spin": "4t_spin"}


def load():
    """data[case][setting][backend] -> merged record (lower median per metric, status, errors)."""
    data = defaultdict(lambda: defaultdict(dict))
    meta = {}
    runs_seen = defaultdict(set)
    pat = re.compile(r"^(heat51|heat201|heat3d65)_(1t|4t|4t_spin|metal)_run(\d+)\.json$")
    for p in sorted(DIR.glob("*.json")):
        m = pat.match(p.name)
        if not m:
            continue
        case, setting, run = m.group(1), m.group(2), int(m.group(3))
        with open(p) as f:
            d = json.load(f)
        meta.setdefault(case, d)
        runs_seen[(case, setting)].add(run)
        for b, rec in d["results"].items():
            slot = data[case][setting].setdefault(b, {"status": rec["status"], "timings": {}, "runs": {},
                                                      "reasons": set(), "notes": set()})
            if rec["status"] != "ok":
                slot["reasons"].add(rec.get("reason", ""))
                if slot["status"] == "ok":
                    pass
                continue
            slot["status"] = "ok"
            slot["notes"].add(rec.get("note", ""))
            slot["runs"][run] = rec
            for k, v in rec["timings"].items():
                cur = slot["timings"].get(k)
                if cur is None or v["ms_median"] < cur["ms_median"]:
                    slot["timings"][k] = dict(v, run=run)
            for k in ("rel_err_u1_vs_cpp", "rel_err_u1_vs_numpy", "rel_err_un_vs_numpy",
                      "abs_err_un_vs_numpy", "n_steps", "stats_max_rel_err_vs_cpp",
                      "stats_index_match", "first_step_s", "device", "dtype"):
                if k in rec:
                    slot.setdefault(k, rec[k])
                    if k in ("rel_err_u1_vs_cpp", "rel_err_un_vs_numpy", "abs_err_un_vs_numpy"):
                        slot[k] = max(slot[k], rec[k])
            if "split_ms" in rec:
                slot.setdefault("splits", {})[run] = rec["split_ms"]
    return data, meta, runs_seen


def fmt_ms(v):
    if v is None:
        return ""
    if v < 0.01:
        return f"{v:.4f}"
    if v < 1:
        return f"{v:.3f}"
    return f"{v:.2f}"


def fmt_x(v):
    if v is None:
        return ""
    if v < 0.1:
        return f"{v:.3f}x"
    if v < 10:
        return f"{v:.2f}x"
    return f"{v:.0f}x"


def fmt_e(v):
    return "" if v is None else f"{v:.1e}"


def ms(rec, key):
    t = rec["timings"].get(key)
    return None if t is None else t["ms_median"]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def per_case_tables(data, meta):
    parts = []
    for case in CASES:
        if case not in meta:
            continue
        d = meta[case]
        bar = d["cpp_bar"]
        n_D = d["n_D"]
        parts.append(f"### {case} ({CASE_LABEL[case]}, n_D {n_D}, n_V {d['n_V']}, "
                     f"{d['structure_counts']['exc_rows']} exception rows)")
        parts.append("")
        parts.append("Native C++ bar (Part IV E): RHS graph submit "
                     + " / ".join(f"{bar['rhs_submit_us'][s]:.0f} us {s}" for s in ("1t", "4t", "4t_spin"))
                     + "; shipped step "
                     + " / ".join(f"{bar['step_ms'][s]:.3g} ms {s}" for s in ("1t", "4t", "4t_spin")) + ".")
        parts.append("")
        headers = ["setting", "backend", "operator ms", "RHS ms", "step ms", "RHS / C++ RHS submit",
                   "step / 4 submits", "step / C++ shipped step", "ns per D point per step",
                   "u1 vs C++", "drift 10 steps vs fp64 NumPy"]
        rows = []
        for setting in SETTINGS:
            for b in BACKENDS:
                rec = data[case].get(setting, {}).get(b)
                if rec is None or rec["status"] != "ok":
                    continue
                st = ms(rec, "step")
                rhs = ms(rec, "rhs")
                if setting == "metal":
                    # MLX CPU is single-stream; GPU rows are compared with the 1t and 4t spin bars
                    bar_s = ["1t", "4t_spin"]
                else:
                    bar_s = [BAR_SETTING[setting]]
                r_rhs = " / ".join(fmt_x(rhs * 1e3 / bar["rhs_submit_us"][s]) for s in bar_s)
                r_4 = " / ".join(fmt_x(st * 1e3 / (4 * bar["rhs_submit_us"][s])) for s in bar_s)
                r_st = " / ".join(fmt_x(st / bar["step_ms"][s]) for s in bar_s)
                label = SETTING_LABEL[setting] if setting != "metal" else "Metal, 1 host thread"
                if len(bar_s) > 1:
                    label += " (vs 1t / 4t spin)"
                rows.append([label, BACKEND_LABEL[b], fmt_ms(ms(rec, "operator")), fmt_ms(rhs), fmt_ms(st),
                             r_rhs, r_4, r_st, f"{st * 1e6 / n_D:.0f}",
                             fmt_e(rec.get("rel_err_u1_vs_cpp")), fmt_e(rec.get("rel_err_un_vs_numpy"))])
        parts.append(table(headers, rows))
        parts.append("")
    return "\n".join(parts)


def best(data, case, setting, key, backends=None):
    bestv, bestb = None, None
    for b in (backends or BACKENDS):
        rec = data[case].get(setting, {}).get(b)
        if rec is None or rec["status"] != "ok":
            continue
        v = ms(rec, key)
        if v is not None and (bestv is None or v < bestv):
            bestv, bestb = v, b
    return bestv, bestb


def gate_table(data, meta):
    rows = []
    checks = [
        ("heat51", "1t", "1t", 1.0, "51x51, 1 thread, at most 1.0x"),
        ("heat51", "4t_spin", "4t_spin", 1.0, "51x51, 4 threads spin, at most 1.0x"),
        ("heat51", "4t", "4t", 1.0, "51x51, 4 threads default, at most 1.0x"),
        ("heat3d65", "4t", "4t", 0.5, "65^3, 4 threads default, at most 0.5x"),
        ("heat3d65", "1t", "1t", 0.5, "65^3, 1 thread, at most 0.5x"),
        ("heat3d65", "4t_spin", "4t_spin", 0.5, "65^3, 4 threads spin, at most 0.5x"),
    ]
    verdicts = {}
    for case, setting, bar_s, thr, label in checks:
        if case not in meta:
            continue
        bar_us = meta[case]["cpp_bar"]["rhs_submit_us"][bar_s]
        v, b = best(data, case, setting, "rhs")
        if v is None:
            continue
        ratio = v * 1e3 / bar_us
        met = ratio <= thr
        verdicts[(case, setting)] = (ratio, b, met)
        cells = []
        for bb in ("scipy", "structured", "torch-eager", "torch-compile"):
            rec = data[case].get(setting, {}).get(bb)
            if rec is None or rec["status"] != "ok":
                cells.append("")
            else:
                r = ms(rec, "rhs") * 1e3 / bar_us
                cells.append(f"{ms(rec, 'rhs') * 1e3:.1f} us ({fmt_x(r)})")
        rows.append([label, f"{bar_us:.0f} us"] + cells + [f"{BACKEND_LABEL[b]} {fmt_x(ratio)}", "yes" if met else "no"])
    headers = ["threshold", "bar", "scipy CSR", "structured NumPy", "torch eager", "torch.compile", "best", "met"]
    return table(headers, rows), verdicts


def split_table(data, meta, case, setting, backend="structured"):
    rec = data[case].get(setting, {}).get(backend)
    if rec is None or rec["status"] != "ok":
        return f"(no {backend} record for {case} {setting})"
    bar = meta[case]["cpp_bar"]
    s = BAR_SETTING[setting]
    T = rec["timings"]
    g = lambda k: T[k]["ms_median"]  # noqa: E731
    op, rhs, src, bc, st, step, step_ns, axpy = (g(k) for k in METRICS)
    glue = rhs - op - src
    py = {
        "operator (4x)": 4 * op, "source (4x)": 4 * src, "RHS glue: scale, scatter-add, zero fill (4x)": 4 * glue,
        "boundary update (4x)": 4 * bc, "stats (1x)": st, "stage axpy and fills (1x)": axpy,
    }
    py_sum = sum(py.values())
    cpp = {
        "operator (4x)": 4 * bar["rhs_submit_us"][s] / 1e3, "source (4x)": 4 * bar["fill_source_us"][s] / 1e3,
        "RHS glue: scale, scatter-add, zero fill (4x)": None, "boundary update (4x)": 4 * bar["update_boundary_us"][s] / 1e3,
        "stats (1x)": bar["stats_us"][s] / 1e3, "stage axpy and fills (1x)": None,
    }
    cpp_sum = sum(v for v in cpp.values() if v is not None)
    rows = []
    for k in py:
        c = cpp[k]
        rows.append([k, fmt_ms(py[k]), f"{100 * py[k] / step:.0f}%", fmt_ms(c) if c is not None else "(in the step remainder)",
                     fmt_x(py[k] / c) if c else ""])
    rows.append(["sum of parts", fmt_ms(py_sum), f"{100 * py_sum / step:.0f}%", fmt_ms(cpp_sum) + " (RHS, source, boundary, stats)", fmt_x(py_sum / cpp_sum)])
    rows.append(["measured step (with stats)", fmt_ms(step), "100%", fmt_ms(bar["step_ms"][s]) + " (shipped step)", fmt_x(step / bar["step_ms"][s])])
    rows.append(["remainder (Python call overhead, ufunc dispatch)", fmt_ms(step - py_sum), f"{100 * (step - py_sum) / step:.0f}%",
                 fmt_ms(bar["step_ms"][s] - cpp_sum) + " (slot ops, graph launch, logging)", ""])
    rows.append(["step without stats", fmt_ms(step_ns), f"{100 * step_ns / step:.0f}%", "", ""])
    headers = ["component", f"Python {backend} {SETTING_LABEL[setting]} ms", "share", f"C++ {s} ms (Part IV E)", "Python / C++"]
    return table(headers, rows)


def failures(data):
    rows = []
    for case in CASES:
        for setting in SETTINGS:
            for b, rec in data[case].get(setting, {}).items():
                if rec["status"] != "ok":
                    for reason in sorted(rec["reasons"]):
                        rows.append([case, SETTING_LABEL[setting], b, rec["status"], reason[:160]])
    return table(["case", "setting", "backend", "status", "reason"], rows) if rows else "none"


def machine_lines(meta, runs_seen):
    d = next(iter(meta.values()))
    m = d["machine"]
    lines = [f"Python {m['python']}, numpy {m['numpy']}, scipy {m['scipy']}, torch {m.get('torch')}, mlx {m.get('mlx')}; "
             f"{m['platform']}."]
    cells = sorted(runs_seen.items())
    lines.append("Runs per cell: " + ", ".join(f"{c}/{s}: {sorted(r)}" for (c, s), r in cells) + ".")
    return "\n".join(lines)


def best_path_lines(data, meta):
    lines = []
    for case in CASES:
        if case not in meta:
            continue
        n_D = meta[case]["n_D"]
        for setting in ("1t", "4t", "4t_spin"):
            v, b = best(data, case, setting, "step", ["scipy", "structured", "torch-eager", "torch-compile"])
            if v is None:
                continue
            bar = meta[case]["cpp_bar"]["step_ms"][BAR_SETTING[setting]]
            lines.append(f"- {case} {SETTING_LABEL[setting]}: best step {fmt_ms(v)} ms ({BACKEND_LABEL[b]}), "
                         f"{v * 1e6 / n_D:.0f} ns per D point per step; C++ shipped step {bar} ms, "
                         f"{bar * 1e6 / n_D:.0f} ns per D point ({fmt_x(v / bar)}).")
    return "\n".join(lines)


NARRATIVE = {
    "intro": """Measured 2026-10-13 on the owner's Apple M4 (hw.ncpu 10, 4 performance cores, 16 GiB), macOS 26.2, the machine of Part IV E, against the native C++ bar of `docs/python-dsl-survey/05-measurements.md` Part IV E. Code: `python/pyshoccs/bench/bench_heat.py` on the real exports `.scratch/export/heat51`, `heat201` and `heat3d65` (the exports the parity protocol passed on, byte-fresh against the exporter). Protocol: per backend and metric the median of 100 timed calls at 51x51, 60 at 201x201 and 20 at 65^3 after 10 (5 at 65^3) warm-ups; every cell was run twice in separate processes (run 1 of every cell first, then run 2) and the lower of the two medians is kept, metric by metric. The four CPU fp64 backends (scipy CSR, structured NumPy, torch eager, torch.compile) ran inside the Bash sandbox at three thread settings: 1 thread (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), 4 threads (libomp defaults) and 4 threads spin (`KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active`). The MLX CPU backend and the two fp32 GPU backends (torch MPS eager, MLX GPU compiled) need a Metal device, which the sandbox hides, so they ran in one unsandboxed call (`run_matrix.py`, setting `metal`) with one host thread. Load average at the start of every process was 1.0 to 1.7 and nothing else was benchmarking.

A step here is one RK4 step from `u0_V` with the export's `dt`, including the three stage boundary updates, the final `update_boundary(t0 + dt)` and the stats of the new state (what `simulation_cycle` does once per step), so it is comparable with the shipped C++ step. The RHS is `k A u + src(t)` with the Dirichlet rows zeroed, applied to `rand_V`. The correctness columns are the stepped state against the exporter's `u1_V` (relative to max|u1|) and the state after 10 chained steps against the fp64 SciPy path. Raw data: `<case>_<setting>_run<n>.json` and `.log` in this directory; `summarize.py` renders this file.""",
    "tables": """`RHS / C++ RHS submit` and `step / C++ shipped step` use the bar at the same thread setting; `step / 4 submits` divides the step by four RHS graph submits, the discretisation-only bar. The Metal rows carry two ratios, against the 1-thread and the 4-thread-spin bars, because MLX exposes no CPU thread count and the GPU rows have no matching CPU setting. `ns per D point per step` divides the step by n_D (grid points, solid ones included), as the Part IV E table counts D points. The thread setting does not move the NumPy backends (scipy's CSR matvec and NumPy's ufuncs are single-threaded; the BLAS thread count only touches the two-row dgemv of the hoisted source), so their 4-thread rows repeat the 1-thread numbers to within noise. torch at 4 threads with sleeping libomp threads shows the same wake-up pathology as the Kokkos 4t bar at 51x51 (torch.compile RHS 100 us against 24 us at 1 thread and 20 us with spinning threads); at 65^3 four threads halve the torch.compile RHS (1.44 ms to 0.745 ms default, 0.667 ms spin).""",
    "gate": """Thresholds from `docs/PYTHON_DSL_SURVEY.md` Section 4.2: the Python RHS at most 1.0x of the native RHS graph submit at 51x51 and at most 0.5x at 65^3 with the sphere, at the thread setting fixed in phase 1(a): 1 thread for the shipped 2D case (bar 21.0 us) and 4 threads with libomp defaults for 65^3 (bar 961 us). The other settings are shown for orientation. The gate text names the fixed-up C++ RHS graph of phase 1(b) as its bar; that graph does not exist yet, so the shipped graph submit stands in for it. Changes (1), (2) and (4) can only make the C++ graph faster, so a "met" below is provisional against a bar that will move down, while a "not met" stands.

Verdict: met at 51x51 (scipy CSR 15.9 us, 0.76x; it is the only backend under 1.0x at 1 thread, torch.compile is 1.13x and structured NumPy 1.41x); not met at 65^3 as the gate is defined (best 4-thread-default RHS torch.compile 745 us, 0.78x of 961 us; scipy CSR 1.10x). At 65^3 the 0.5x threshold is reached only against the other two bars: scipy CSR at 1 thread sits exactly on it (1032 us, 0.50x of 2065 us; the second run gave 1078 us, 0.52x, so this cell is inside run-to-run noise and is not a pass) and torch.compile with spinning threads is under it (667 us, 0.37x of 1810 us). The C++ fused-stage arm (`.scratch/phase1/cpp_arm/RESULTS.md`) is the other reference point: its pencil one-launch RHS is 2.81 us at 51x51 1t and 114 us at 65^3 4t, so the best Python RHS is 5.7x and 6.5x slower than the compact-table C++ kernel at the gate settings (3.0x at 65^3 1t: 1032 against 340 us), and the best Python step is 4.4x slower than the fused pencil step at 51x51 1t (89 against 20.1 us), 2.3x at 51x51 4t spin (61 against 26.7 us), 4.2x at 65^3 4t (3.74 against 0.894 ms) and 2.2x at 65^3 1t (6.09 against 2.80 ms).""",
    "split": """The structured NumPy backend is the one whose step can be taken apart on the same buffers: operator (4x), source (4x, the hoisted manufactured solution: one small dgemv per call), RHS glue (4x: the scale by k, the scatter-add of the source onto the fluid and R rows and the zero fill of the Dirichlet rows), boundary update (4x), stats (1x) and the stage arithmetic measured on its own (`stage_axpy` in `bench_heat.py`: the two fills, the copy and the seven multiply or add passes of `rk4_step_generic`). The C++ column is Part IV E's decomposition (4 x submit, 4 x fill_source, 4 x update_boundary, stats) with its remainder (slot ops, graph launch, logging). The Python remainder is a few percent at 201x201 and 65^3 and within noise at 51x51 (the parts are timed separately, so cache state differs). Reading the 65^3 column: the operator is 57 percent of the step and the RHS glue 19 percent, so the two passes that a fused kernel removes (the separate scale and the scatter of the source) cost as much as the stats and the stage arithmetic together; the source and boundary evaluations, which dominate the shipped C++ step (fill_source is 55 percent of it at 1 thread), are 1 to 2 percent here because the Gaussian's spatial factors are hoisted. The scipy CSR backend has no such split (one matvec over V), which is why its RHS is the cheapest fp64 one at every size at 1 thread: a single pass over 1.5 to 3.4 million stored entries against the structured form's five passes over the padded D block plus the exception CSR.""",
    "gpu": """Single precision on the Apple GPU, same structured tables, run outside the sandbox (the Metal rows of the tables above). MLX GPU compiled: step 0.292 ms at 51x51 (112 ns per D point; 1.10x of the 4-thread-spin shipped step, 0.34x of the 1-thread one), 0.441 ms at 201x201 (11 ns per point, 0.20x of 2.17 ms) and 2.09 ms at 65^3 (8 ns per point, 0.12x of 17.49 ms and 0.036x of 58.4 ms): the fastest step measured at 65^3 in either precision, 0.59x of the best CPU fp64 step (torch.compile 3.54 ms at 4 threads spin). Its RHS at 65^3 is 667 us, 0.69x of the 4-thread-default bar, 0.37x of 4t spin and 0.32x of 1t; at 51x51 the step costs little more than one RHS (0.29 against 0.25 ms) because a fixed per-eval floor of about 0.2 ms dominates. torch MPS eager is launch-bound: 1.19 / 1.46 / 6.98 ms per step at the three sizes, no better than the CPU paths. Precision: the stepped state differs from the C++ fp64 `u1_V` by 6e-8 to 1.1e-7 (relative to max|u1|) and after 10 chained steps from the fp64 NumPy run by 7.6e-7 (51x51), 2.1e-6 (201x201) and 6.4e-6 (65^3) for MLX GPU and 6.4e-7 / 2.3e-6 / 6.9e-6 for torch MPS, where every fp64 CPU path is at 1e-15 to 1e-14. The stats of each step are computed on the host from two device-to-host copies of the source-point values, as the C++ step computes them each step.""",
    "failures": """No backend failed on any case. Inside the Bash sandbox MLX and torch MPS are unavailable (`import mlx.core` raises `[metal::load_device] No Metal device available`, even for the CPU stream; `torch.backends.mps.is_available()` is False), so the script records them as skipped there (`smoke_heat51.json`) and the Metal cells were run outside it. MLX's CPU `cos`, `sin` and `exp` on float64 arrays are evaluated in single precision (`mlx_f64_probe.py`: relative error 7e-8 to 1.2e-7, every other op of the step exact), which is why the MLX CPU row agrees with `u1_V` only to 1.8e-8 although its arrays are float64; only the MMS time factors go through those functions here. The MLX CPU stream is also slow on this workload (28 ms per step at 65^3, 4.7x the scipy step), so it was not pursued further. torch.compile's first step costs 0.5 s per process with a warm inductor cache (2.2 to 2.5 s when the 4-thread variant compiles afresh); MLX compile 0.8 to 6 s in the first process of a case.""",
    "reproduce": """```
cd python/pyshoccs && uv sync --extra bench
python3 -I .scratch/phase1/py_arm/run_matrix.py cpu                                   # sandboxed: 3 cases x (1t, 4t, 4t_spin) x 2 runs
python3 -I .scratch/phase1/py_arm/run_matrix.py mlx,gpu heat51,heat201,heat3d65 metal 1,2   # outside the sandbox
uv run --project python/pyshoccs python .scratch/phase1/py_arm/mlx_f64_probe.py       # outside the sandbox
python3 -I .scratch/phase1/py_arm/summarize.py > .scratch/phase1/py_arm/RESULTS.md
```""",
}


def main():
    data, meta, runs_seen = load()
    if not meta:
        print("no bench JSONs found", file=sys.stderr)
        return 1
    gate, verdicts = gate_table(data, meta)
    out = []
    out.append("# Phase 1, Python array arm: pyshoccs on the real exports against the native C++ bar")
    out.append("")
    out.append(NARRATIVE.get("intro", ""))
    out.append("")
    out.append(machine_lines(meta, runs_seen))
    out.append("")
    out.append("## Per case: ms per RHS and per step, every backend and thread setting")
    out.append("")
    out.append(NARRATIVE.get("tables", ""))
    out.append("")
    out.append(per_case_tables(data, meta))
    out.append("## Gate (2): RHS against the native RHS graph submit")
    out.append("")
    out.append(NARRATIVE.get("gate", ""))
    out.append("")
    out.append(gate)
    out.append("")
    out.append("## Best Python step, ns per point")
    out.append("")
    out.append(best_path_lines(data, meta))
    out.append("")
    out.append("## Step decomposition of the structured NumPy backend beside the C++ decomposition")
    out.append("")
    out.append(NARRATIVE.get("split", ""))
    out.append("")
    for case in CASES:
        for setting in ("1t", "4t"):
            if case in meta and data[case].get(setting, {}).get("structured", {}).get("status") == "ok":
                out.append(f"### {case}, {SETTING_LABEL[setting]}")
                out.append("")
                out.append(split_table(data, meta, case, setting))
                out.append("")
    out.append("## GPU (fp32) rows")
    out.append("")
    out.append(NARRATIVE.get("gpu", ""))
    out.append("")
    out.append("## Backend failures and skips")
    out.append("")
    out.append(failures(data))
    out.append("")
    out.append(NARRATIVE.get("failures", ""))
    out.append("")
    out.append("## Reproduce")
    out.append("")
    out.append(NARRATIVE.get("reproduce", ""))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
