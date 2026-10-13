"""Tables for RESULTS.md from the Google Benchmark JSON files in this directory."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ["2/51", "2/71", "2/201", "2/401", "3/65"]
SETTINGS = ["t1", "t4", "t4_spin"]
LABEL = {"t1": "1 thread", "t4": "4 threads", "t4_spin": "4 threads spin", "t10": "10 threads"}
RHS = [
    ("graph (in process)", "BM_graph_rhs"),
    ("two-launch", "BM_structured_rhs_two"),
    ("two-launch MDRange", "BM_structured_rhs_two_md"),
    ("one-launch", "BM_structured_rhs_one"),
    ("pencil two-launch", "BM_structured_rhs_pencil"),
    ("pencil one-launch", "BM_structured_rhs_pencil_one"),
]
STEP = [
    ("graph step", "BM_graph_step"),
    ("fused two-launch", "BM_fused_step_two"),
    ("fused one-launch", "BM_fused_step_one"),
    ("fused pencil two-launch", "BM_fused_step_pencil"),
    ("fused pencil one-launch", "BM_fused_step_pencil_one"),
]


def load(name):
    path = HERE / name
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for b in data["benchmarks"]:
        if b.get("run_type") == "aggregate" and b.get("aggregate_name") != "median":
            continue
        out[b.get("run_name", b["name"])] = b
    return out


def us(b):
    return b["real_time"] * {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}[b["time_unit"]]


def fmt(x):
    if x is None:
        return "-"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def ratio(a, b):
    if a is None or b is None or b == 0:
        return "-"
    return f"{a / b:.2f}x"


def case_label(c):
    return c.replace("2/", "2D ").replace("3/", "3D ") + ("^3" if c.startswith("3") else "^2")


def main():
    tables = {s: load(f"bench_tables_{s}.json") for s in SETTINGS}
    tables["t4_rep3"] = load("bench_tables_t4_rep3.json")
    cut = {s: load(f"bench_rhs_cutcell_{s}.json") for s in SETTINGS}
    triad = {s: load(f"bench_tables_{s}.json") for s in SETTINGS}
    triad["t10"] = load("bench_tables_t10_triad.json")

    def get(src, bm, c):
        b = src.get(f"{bm}/{c}")
        return us(b) if b else None

    for s in SETTINGS + ["t4_rep3"]:
        src = tables[s]
        if not src:
            continue
        cut_src = cut["t4"] if s == "t4_rep3" else cut[s]
        title = LABEL.get(s, "4 threads, median of 3 repetitions")
        print(f"\n### {title}: RHS (us)\n")
        print("| case | graph (bench_rhs_cutcell) | " + " | ".join(l for l, _ in RHS) + " |")
        print("|---|" + "---|" * (len(RHS) + 1))
        for c in CASES:
            row = [case_label(c), fmt(get(cut_src, "BM_heat_cut_rhs", c))]
            row += [fmt(get(src, bm, c)) for _, bm in RHS]
            print("| " + " | ".join(row) + " |")

        print(f"\n### {title}: RK4 step (us)\n")
        print("| case | " + " | ".join(l for l, _ in STEP) + " |")
        print("|---|" + "---|" * len(STEP))
        for c in CASES:
            row = [case_label(c)] + [fmt(get(src, bm, c)) for _, bm in STEP]
            print("| " + " | ".join(row) + " |")

        print(f"\n### {title}: ratios against the standalone graph submit\n")
        print("| case | best structured RHS | pencil one-launch RHS | flat one-launch RHS | "
              "graph step / 4 submits | best fused step / graph step | best fused step / 4 submits |")
        print("|---|---|---|---|---|---|---|")
        for c in CASES:
            g = get(cut_src, "BM_heat_cut_rhs", c)
            rhs = {l: get(src, bm, c) for l, bm in RHS[1:]}
            best_rhs = min((v, l) for l, v in rhs.items() if v is not None)
            gs = get(src, "BM_graph_step", c)
            fused = {l: get(src, bm, c) for l, bm in STEP[1:]}
            best_f = min((v, l) for l, v in fused.items() if v is not None)
            print(f"| {case_label(c)} | {ratio(best_rhs[0], g)} ({best_rhs[1]}) | "
                  f"{ratio(rhs['pencil one-launch'], g)} | {ratio(rhs['one-launch'], g)} | "
                  f"{ratio(gs, 4 * g if g else None)} | {ratio(best_f[0], gs)} ({best_f[1]}) | "
                  f"{ratio(best_f[0], 4 * g if g else None)} |")

    print("\n### STREAM triad, a = b + s c, three buffers of 16.8M doubles (134 MB each)\n")
    print("| threads | median of 10 (ms) | GiB/s (median iteration) | GiB/s (best iteration) |")
    print("|---|---|---|---|")
    for s in ["t1", "t4", "t4_spin", "t10"]:
        b = triad[s].get("BM_stream_triad/16777216/manual_time")
        if b:
            print(f"| {LABEL[s]} | {b['real_time']:.2f} | {b['GiBps_median']:.1f} | "
                  f"{b['GiBps_best']:.1f} |")


if __name__ == "__main__":
    sys.exit(main())
