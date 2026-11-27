# cam_summary.py - one table for camera-mounting runs   (python cam_summary.py run_cam_h*   or   run_cam4_*)
import csv, sys, os, re
from collections import defaultdict
WANT = {"person_stand": "stop", "person_lie": "stop", "person_moving": "stop", "ditch": "stop", "rock": "detour", "drum": "detour"}
def key(d):
    m = re.search(r"h([\d.]+)_t([\d.]+)", d)
    return (d.split("_h")[0], float(m.group(2)), float(m.group(1)))
print(f"{'run':24s} {'ditch final gap (m)':24s} {'coll.':>5s} {'first slow (m) person/lying/rock/ditch':>40s}"
      f" {'false stop none|grass':>22s} {'rock+drum detoured past':>24s} {'right':>7s}")
for d in sorted(sys.argv[1:], key=key):
    T = list(csv.DictReader(open(os.path.join(d, "trials.csv"))))
    F = list(csv.DictReader(open(os.path.join(d, "frames.csv"))))
    first, seen = defaultdict(list), set()
    for r in F:
        k = (r["scenario"], r["trial"])
        if k in seen or r["acted_label"] == "go" or not r["gap_m"]: continue
        seen.add(k); first[r["scenario"]].append(float(r["gap_m"]))
    mean = lambda L: f"{sum(L) / len(L):4.2f}" if L else "  - "
    ditch = ", ".join(r["final_gap_m"] for r in T if r["scenario"] == "ditch")
    col = sum(int(r["collided"]) for r in T)
    fs_none = sum(1 for r in T if r["scenario"] == "none" and r["decision"])
    fs_grass = sum(1 for r in T if r["scenario"] == "grass" and r["decision"])
    past = sum(1 for r in T if r["scenario"] in ("rock", "drum") and r.get("finished") == "detoured past")
    right = sum(1 for r in T if r["scenario"] in WANT and r["decision"] == WANT[r["scenario"]] and r["collided"] == "0")
    n_obs = sum(1 for r in T if r["scenario"] in WANT)
    name = os.path.basename(d.rstrip("/")).replace("run_cam4_", "").replace("run_cam_", "")
    fl = f"{mean(first['person_stand'])} / {mean(first['person_lie'])} / {mean(first['rock'])} / {mean(first['ditch'])}"
    print(f"{name:24s} {ditch:24s} {col:5d} {fl:>40s} {fs_none:>12d} | {fs_grass:<8d} {past:>16d}/6 {right:>8d}/{n_obs}")
