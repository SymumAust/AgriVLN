# Find the U-turns in a CitrusFarm drive from its GPS ground truth, and list which camera files (zed_*.bag)
# cover the moments just after each turn (the robot entering the next tree row).
# Nothing is looked at except the GPS track, so no model and no image influences the choice.
#
#   python farm2_select.py --gt ~/datasets/citrusfarm/04_13D_Jackal/gt.csv --seq 04_13D_Jackal
#
# Output: a table of turns (time, place), the zed bags to download (each ~4 GB, ~29 s of video),
#         and farm2_<seq>_track.png (the path from above, turns marked).
# Bag start times come from the file names (local time, UTC-7 in July 2023 California).

import argparse
import csv
import math
from datetime import datetime, timedelta, timezone

import numpy as np

ZED = {
    "04_13D_Jackal": ["14-25-34", "14-26-03", "14-26-31", "14-27-00", "14-27-29", "14-27-58", "14-28-27", "14-28-55",
                      "14-29-24", "14-29-53", "14-30-22", "14-30-51", "14-31-19", "14-31-48", "14-32-17", "14-32-46",
                      "14-33-15", "14-33-43", "14-34-12", "14-34-41", "14-35-10", "14-35-39", "14-36-07", "14-36-36",
                      "14-37-05", "14-37-34", "14-38-03"],
    "05_13D_Jackal": ["17-05-13", "17-05-42", "17-06-11", "17-06-39", "17-07-08", "17-07-37", "17-08-06", "17-08-35",
                      "17-09-03", "17-09-32", "17-10-01", "17-10-30", "17-10-59", "17-11-27", "17-11-56", "17-12-25",
                      "17-12-54", "17-13-23", "17-13-51", "17-14-20", "17-14-49", "17-15-18", "17-15-47", "17-16-15",
                      "17-16-44", "17-17-13", "17-17-42"],
}
DATE = {"04_13D_Jackal": "2023-07-18", "05_13D_Jackal": "2023-07-16"}
PDT = timezone(timedelta(hours=-7))

p = argparse.ArgumentParser()
p.add_argument("--gt", required=True)
p.add_argument("--seq", required=True, choices=list(ZED))
p.add_argument("--before", type=float, default=16.0, help="seconds of video needed before the end of a turn")
p.add_argument("--after", type=float, default=10.0, help="seconds of video needed after the end of a turn")
a = p.parse_args()

rows = [r for r in csv.reader(open(a.gt)) if r and not r[0].startswith("#")]
T = np.array([float(r[0]) for r in rows]); X = np.array([float(r[1]) for r in rows]); Y = np.array([float(r[2]) for r in rows])
o = np.argsort(T); T, X, Y = T[o], X[o], Y[o]
gap = np.diff(T)
print(f"{len(T)} GPS points, {T[-1] - T[0]:.0f} s, median step {np.median(gap):.2f} s, largest gap {gap.max():.1f} s")

# resample every 0.5 m of travel -> heading per metre of path, independent of speed and stops
d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
s = np.arange(0, d[-1], 0.5)
xs, ys, ts = np.interp(s, d, X), np.interp(s, d, Y), np.interp(s, d, T)
hd = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
k = 5
hd = np.convolve(hd, np.ones(2 * k + 1) / (2 * k + 1), mode="same")      # smooth over ~5 m
print(f"path length {d[-1]:.0f} m")

# a U-turn: heading changes by > 150 deg within 25 m of path
turns, i, W = [], 0, 50                                                   # 50 samples x 0.5 m = 25 m
while i < len(s) - W:
    if abs(hd[i + W] - hd[i]) > math.radians(150):
        j0 = i
        while j0 + W < len(s) and abs(hd[j0 + W] - hd[j0]) > math.radians(150):
            j0 += 1
        seg = slice(i, j0 + W)
        rate = np.abs(np.gradient(hd[seg]))
        busy = np.where(rate > math.radians(2.0))[0]                     # > 2 deg per 0.5 m = turning
        start, end = i + busy[0], i + busy[-1]
        turns.append((start, end, math.degrees(hd[end] - hd[start])))
        i = j0 + W
    else:
        i += 1

starts = [datetime.strptime(f"{DATE[a.seq]} {h}", "%Y-%m-%d %H-%M-%S").replace(tzinfo=PDT).timestamp() for h in ZED[a.seq]]
names = [f"zed_{DATE[a.seq]}-{h}_{n}.bag" for n, h in enumerate(ZED[a.seq])]
print(f"camera files: {len(names)}, first starts {starts[0] - T[0]:+.1f} s from the first GPS point\n")


def bags_for(t0, t1):
    out = []
    for n, st in enumerate(starts):
        en = starts[n + 1] if n + 1 < len(starts) else st + 29
        if st < t1 and en > t0:
            out.append(names[n])
    return out


print(f"{'turn':>4} {'start s':>8} {'end s':>8} {'turned':>7} {'at (x, y) m':>18}   camera files needed")
need = []
for n, (st, en, deg) in enumerate(turns, 1):
    te = ts[en]
    b = bags_for(te - a.before, te + a.after)
    need += [x for x in b if x not in need]
    print(f"{n:>4} {ts[st] - T[0]:>8.1f} {te - T[0]:>8.1f} {deg:>6.0f}° ({xs[en]:7.1f},{ys[en]:7.1f})   " + ", ".join(b))
print(f"\n{len(turns)} U-turns -> {len(need)} camera files (~{len(need) * 3.9:.0f} GB):")
for x in need:
    print("  ", x)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(xs, ys, color="#1E4E79", lw=1.2)
ax.plot(xs[0], ys[0], "o", color="#2E7D32", ms=9, label="start")
for n, (st, en, deg) in enumerate(turns, 1):
    ax.plot(xs[st:en], ys[st:en], color="#E09A2B", lw=3)
    ax.plot(xs[en], ys[en], "v", color="#C0392B", ms=9)
    ax.annotate(f"turn {n}\n{ts[en] - T[0]:.0f} s", (xs[en], ys[en]), fontsize=8)
ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend()
ax.set_title(f"{a.seq}: {d[-1]:.0f} m, {len(turns)} U-turns (orange), turn ends (red)")
fig.tight_layout(); fig.savefig(f"farm2_{a.seq}_track.png", dpi=130)
print(f"\nsaved farm2_{a.seq}_track.png")
