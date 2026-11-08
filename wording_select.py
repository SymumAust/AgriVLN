# Apply the selection rule of plan_wording.txt to the development runs.
#
#   python wording_select.py
#
# For each candidate: closed-loop step6 at 1.9 s per answer on the 9 development clips (closed_loop_sim.py),
# plus early YES (frames more than 2 s before the stop answered YES). Prints the ranking and the winner.

import csv
import re
import subprocess
import sys

DEV = ["farm_6", "farm_7", "farm_8"] + [f"farmB_{i}" for i in range(1, 7)]
CAND = {"A": ["probe_results_farm2.csv", "probe_results_farmB.csv"],
        "B": ["probe_results_wordB.csv"], "C": ["probe_results_wordC.csv"], "D": ["probe_results_wordD.csv"]}
LEN = {"A": 55, "B": 114, "C": 131, "D": 106}      # characters of the wording, for the last tie-break

rows = []
for name, files in CAND.items():
    try:
        early = total = 0
        for f in files:
            for r in csv.DictReader(open(f)):
                if r["model"] == "qwen3-vl:4b" and r["episode"] in DEV and r["truth"] == "not arrived":
                    total += 1
                    early += r["answer"] == "YES"
    except FileNotFoundError:
        print(f"{name}: results not found ({', '.join(files)}), skipped")
        continue
    out = subprocess.run([sys.executable, "closed_loop_sim.py", "--csv", *files, "--eps", *DEV, "--latency", "1.9",
                          "--policies", "step6"], capture_output=True, text=True).stdout
    m = re.search(r"\n1\.9\s+(\d+)/(\d+) \[([^\]]+)\]", out)
    ok, n = int(m.group(1)), int(m.group(2))
    rows.append((name, ok, n, m.group(3), early, total))

rows.sort(key=lambda r: (-r[1], r[4], LEN[r[0]]))
print(f"\n{'cand':<5}{'step6 @1.9 s':>14}{'mean err':>10}{'early YES':>14}")
for name, ok, n, mean, early, total in rows:
    print(f"{name:<5}{ok:>10}/{n:<3}{mean:>10}{early:>8}/{total:<5}")
best = rows[0][0] if rows else "A"
if rows and best != "A" and rows[0][1] <= next(r[1] for r in rows if r[0] == "A"):
    best = "A"   # a candidate must beat A on successes to replace it
print(f"\nchosen wording: {best}")
