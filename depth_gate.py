# Depth check on top of the VLM answers: a YES only counts if the ZED depth shows something close at BOTH image edges.
#
#   python depth_gate.py                                  # development: scan D on farmB_1..9, print the table
#   python depth_gate.py --D 3.0 --eps farmB_10 ... --csv probe_results_farmC.csv --out gated_test.csv   # apply one D
#
# For each frame of each episode: gate = (left_m < D and right_m < D), left/right = median depth of the outer 15 %
# of the image (farm2_depth.py). A missing depth value counts as "not close".
#   vlm+gate   : answer YES only if the VLM said YES and the gate is true; then the frozen step6 rule (closed_loop_sim.py)
#   depth only : answer YES whenever the gate is true (no VLM) -> shows whether the language model adds anything
# Writes gated answer files and runs closed_loop_sim.py at 1.9 s per answer.

import argparse
import csv
import json
import math
import re
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--csv", nargs="+", default=["probe_results_farmB.csv", "probe_results_farmB2.csv"])
p.add_argument("--eps", nargs="+", default=[f"farmB_{i}" for i in range(1, 10)])
p.add_argument("--depth", default="farm2/04_13D_Jackal/depth.csv")
p.add_argument("--D", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
p.add_argument("--latency", default="1.9")
p.add_argument("--out", default=None, help="with a single D: keep the gated answers in this file")
a = p.parse_args()

dep = {}
for r in csv.DictReader(open(a.depth)):
    f = lambda x: float(x) if x not in ("", "nan") else math.nan
    dep[round(float(r["t"]), 1)] = (f(r["left_m"]), f(r["right_m"]))

ans = []
for path in a.csv:
    for r in csv.DictReader(open(path)):
        if r["model"] == "qwen3-vl:4b" and r["episode"] in a.eps:
            ans.append(r)
src = {}
for ep in a.eps:
    for g in json.load(open(f"dataset/{ep}/gps.json")):
        src[(ep, g["t"])] = g["src_t"]


def run(rows, tag):
    path = f"gated_{tag}.csv" if a.out is None else a.out
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "episode", "time", "answer", "truth", "raw"])
        w.writeheader()
        w.writerows(rows)
    out = subprocess.run([sys.executable, "closed_loop_sim.py", "--csv", path, "--eps", *a.eps, "--latency", a.latency,
                          "--policies", "step6"], capture_output=True, text=True).stdout
    m = re.search(r"\n\d+\.\d+\s+(\d+)/(\d+) \[([^\]]+)\]( ns\d+)?", out)
    return f"{m.group(1)}/{m.group(2)} [{m.group(3)}]{m.group(4) or ''}"


print(f"step6 at {a.latency} s per answer on {', '.join(a.eps)}   (successes, [mean error m], ns = no stop)\n")
print(f"{'D (m)':>6}  {'VLM + depth gate':>24}  {'depth only':>24}   early YES vlm+gate / depth only")
base = run([dict(r, raw="") for r in ans], "none")
print(f"{'-':>6}  {base:>24}  {'(VLM only, no gate)':>24}")
for D in a.D:
    g_rows, d_rows, e1, e2 = [], [], 0, 0
    for r in ans:
        st = src.get((r["episode"], round(float(r["time"]), 1)))
        lm, rm = dep.get(round(st, 1), (math.nan, math.nan)) if st is not None else (math.nan, math.nan)
        gate = lm < D and rm < D
        yes_g = r["answer"] == "YES" and gate
        g_rows.append(dict(r, answer="YES" if yes_g else "NO", raw=""))
        d_rows.append(dict(r, answer="YES" if gate else "NO", raw=""))
        if r["truth"] == "not arrived":
            e1 += yes_g
            e2 += gate
    print(f"{D:>6.1f}  {run(g_rows, f'vlm_D{D}'):>24}  {run(d_rows, f'depth_D{D}'):>24}   {e1:>5} / {e2}")
