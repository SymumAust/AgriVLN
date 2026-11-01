# Score the arrival-question answers on the farm episodes with REAL distances (RTK GPS), not time.
#
#   python farm_score.py --csv probe_results_farm.csv
#
# Same stop rule as window_rule.py: stop at the first frame where >= k of the last 10 answers are YES.
# Stop error = |distance driven at our stop - distance driven at the labelled stop|, from gps.json.
# Shown with a sign: "-" = stopped too early, "+" = too late. Success = error < 2 m. No stop = failure.

import argparse
import csv
import json
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("--csv", default="probe_results_farm.csv")
p.add_argument("--window", type=int, default=10)
p.add_argument("--k", type=int, nargs="+", default=[6, 8, 10])
a = p.parse_args()

ans = defaultdict(dict)
for r in csv.DictReader(open(a.csv)):
    if r["episode"].startswith("farm_"):
        ans[(r["model"], r["episode"])][round(float(r["time"]), 1)] = r["answer"] == "YES"

print(f"{'model':<16} {'episode':<8} " + " ".join(f"{'>=' + str(k) + '/' + str(a.window):>16}" for k in a.k))
succ = defaultdict(int)
for (model, ep), d in sorted(ans.items()):
    gps = {g["t"]: g["dist_m"] for g in json.load(open(f"dataset/{ep}/gps.json"))}
    lab = json.load(open(f"dataset/{ep}/label.json"))
    stop_t = next(l["time_range"][0] for l in lab if l["action"] == "[STOP]")
    ts = sorted(d)
    cells = []
    for k in a.k:
        hist, st = [], None
        for t in ts:
            hist.append(d[t])
            if len(hist) >= a.window and sum(hist[-a.window:]) >= k:
                st = t
                break
        if st is None:
            cells.append("no stop")
        else:
            err = gps[st] - gps[stop_t]
            ok = abs(err) < 2
            succ[(model, k)] += ok
            cells.append(f"{st:.1f}s {err:+.1f}m{' OK' if ok else ''}")
    print(f"{model:<16} {ep:<8} " + " ".join(f"{c:>16}" for c in cells))
n_ep = len({ep for _, ep in ans})
print("\nsuccesses: " + ", ".join(f"{m} >={k}/10: {v}/{n_ep}" for (m, k), v in sorted(succ.items())))
print("error = GPS distance between our stop and the labelled stop; - early, + late; OK = within 2 m")
