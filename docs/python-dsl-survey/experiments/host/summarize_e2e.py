"""Summarize per-step wall times from logs/system.csv under each case directory.

Usage: python3 -I summarize.py <root> > summary.json
"""
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
out = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    f = d / "logs" / "system.csv"
    if not f.exists():
        continue
    with open(f, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        continue
    key = next((k for k in rows[0] if k.strip() == "Wall_ms"), None)
    if key is None:
        continue
    w = [float(r[key]) for r in rows if r[key].strip()]
    w = [x for x in w if x > 0]
    if not w:
        continue
    s = sorted(w)
    out.append(
        {
            "case": d.name,
            "steps": len(w),
            "median_ms": round(statistics.median(w), 4),
            "mean_ms": round(statistics.mean(w), 4),
            "min_ms": round(s[0], 4),
            "p90_ms": round(s[int(0.9 * (len(s) - 1))], 4),
        }
    )
json.dump(out, sys.stdout, indent=1)
print()
