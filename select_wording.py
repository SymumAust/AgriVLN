# Wording selection for one model (plan_wording_8b.txt): step6 successes on the 9 development clips per candidate.
#
#   python select_wording.py --model qwen3-vl:8b \
#       --A probe_results_8b.csv probe_results_w8b_A.csv --B probe_results_w8b_B.csv --C probe_results_w8b_C.csv --D probe_results_w8b_D.csv
import argparse
import csv
import re
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
for c in "ABCD":
    p.add_argument(f"--{c}", nargs="+", required=True)
p.add_argument("--latency", default="1.9")
a = p.parse_args()
DEV = ["farm_6", "farm_7", "farm_8"] + [f"farmB_{i}" for i in range(1, 7)]
LEN = {"A": 55, "B": 114, "C": 131, "D": 106}
rows = []
for name in "ABCD":
    files = getattr(a, name)
    sel = [r for f in files for r in csv.DictReader(open(f)) if r["model"] == a.model and r["episode"] in DEV]
    have = sorted({r["episode"] for r in sel})
    if len(have) < len(DEV):
        print(f"{name}: missing clips {sorted(set(DEV) - set(have))}")
    pre = [r for r in sel if r["truth"] == "not arrived"]
    early = sum(r["answer"] == "YES" for r in pre)
    blanks = sum(r["answer"] not in ("YES", "NO") for r in sel)
    out = subprocess.run([sys.executable, "closed_loop_sim.py", "--csv", *files, "--model", a.model, "--eps", *DEV,
                          "--latency", a.latency, "--policies", "step6"], capture_output=True, text=True).stdout
    m = re.search(r"\n\d+\.\d+\s+(\d+)/(\d+) \[([^\]]+)\]", out)
    rows.append((name, int(m.group(1)), int(m.group(2)), m.group(3), early, len(pre), blanks, len(sel)))
rows.sort(key=lambda r: (-r[1], r[4], LEN[r[0]]))
print(f"\n{a.model}\n{'cand':<5}{'step6':>8}{'mean err':>10}{'early YES':>14}{'blanks':>10}")
for name, ok, n, mean, early, npre, bl, nall in rows:
    print(f"{name:<5}{ok:>4}/{n:<3}{mean:>10}{early:>8}/{npre:<5}{bl:>6}/{nall}")
best = rows[0][0]
if best != "A" and rows[0][1] <= next(r[1] for r in rows if r[0] == "A"):
    best = "A"
print(f"\nchosen wording for {a.model}: {best}")
