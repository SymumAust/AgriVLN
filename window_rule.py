# "Last 10 votes" STOP rule, evaluated on the per-frame answers saved by probe_arrival.py.
#
# Rule (fixed in advance): keep only the LAST 10 answers (= last 2 s at 0.2 s per frame).
#   STOP at the first frame where at least K of those 10 answers are YES.
#   Main rule: K = 6 (majority). K = 8 and K = 10 are shown as checks.
# Old answers are forgotten, so an early mistake cannot pile up, and an early "not yet"
# cannot cancel a later "arrived".
#
# Because the video does not depend on the robot's actions, this is exact offline:
# no model is run again.
#
#   python window_rule.py                    # every model / episode in probe_results.csv
#   python window_rule.py --window 10 --k 6

import argparse
import csv
import json
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("--window", type=int, default=10)
p.add_argument("--k", type=int, nargs="+", default=[6, 8, 10])
p.add_argument("--csv", default="probe_results.csv")
a = p.parse_args()

answers = defaultdict(dict)
for r in csv.DictReader(open(a.csv)):
    answers[(r["model"], r["episode"])][float(r["time"])] = r["answer"] == "YES"   # last run wins

print(f"{'model':<16} {'episode':<9} {'label':>6} " + " ".join(f"{'>=' + str(k) + '/' + str(a.window):>16}" for k in a.k))
for (model, ep), ans in sorted(answers.items()):
    labels = json.load(open(f"dataset/{ep}/label.json"))
    stop_t = next(l["time_range"][0] for l in labels if l["action"] == "[STOP]")
    length = json.load(open(f"dataset/{ep}/info.json"))["length"]
    ts = sorted(ans)
    cells = []
    for k in a.k:
        stop = None
        for i in range(a.window - 1, len(ts)):
            if sum(ans[t] for t in ts[i - a.window + 1:i + 1]) >= k:
                stop = ts[i]
                break
        if stop is None:
            cells.append("no stop")
        else:
            ne = max(0.0, (stop_t - stop) / stop_t * length)
            cells.append(f"{stop:.1f}s {ne:.2f}m{' OK' if ne < 2.0 else ''}")
    print(f"{model:<16} {ep:<9} {stop_t:>6.1f} " + " ".join(f"{c:>16}" for c in cells))
print("\nNE = (label stop - our stop) / label stop x path length, as in evaluate.py. OK = NE < 2 m (success).")
