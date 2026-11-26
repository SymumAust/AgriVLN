# cam_summary.py - one table for the camera-mounting runs (python cam_summary.py run_cam_h*)
import csv, sys, os
from collections import defaultdict
WANT = {"person_stand": "stop", "person_lie": "stop", "person_moving": "stop", "ditch": "stop", "rock": "detour", "drum": "detour"}
print(f"{'camera':14s} {'ditch final gap (m)':26s} {'collisions':>10s} {'first slow before (m): person / lying / rock / ditch':>54s}"
      f" {'false stops none|grass':>23s} {'right decision':>15s}")
for d in sorted(sys.argv[1:], key=lambda x: (float(x.split('_t')[1]), float(x.split('_h')[1].split('_t')[0]))):
    T = list(csv.DictReader(open(os.path.join(d, "trials.csv"))))
    F = list(csv.DictReader(open(os.path.join(d, "frames.csv"))))
    first = defaultdict(list); seen = set()
    for r in F:
        key = (r["scenario"], r["trial"])
        if key in seen or r["acted_label"] == "go" or not r["gap_m"]: continue
        seen.add(key); first[r["scenario"]].append(float(r["gap_m"]))
    mean = lambda L: f"{sum(L) / len(L):4.2f}" if L else "  - "
    ditch = [r["final_gap_m"] for r in T if r["scenario"] == "ditch"]
    col = sum(int(r["collided"]) for r in T)
    fs = sum(1 for r in T if r["scenario"] in ("none", "grass") and r["decision"])
    fs_none = sum(1 for r in T if r["scenario"] == "none" and r["decision"])
    right = sum(1 for r in T if r["scenario"] in WANT and r["decision"] == WANT[r["scenario"]] and r["collided"] == "0")
    n_obs = sum(1 for r in T if r["scenario"] in WANT)
    cam = f"h {T[0]['cam_h']} t {T[0]['tilt']}"
    print(f"{cam:14s} {', '.join(ditch):26s} {col:10d} {mean(first['person_stand']):>36s} / {mean(first['person_lie'])} / {mean(first['rock'])} / {mean(first['ditch'])}"
          f" {fs_none:>14d} | {fs - fs_none:<6d} {right:>8d}/{n_obs}")
