# Build test episodes farmB_1 .. farmB_6 from the extracted 04_13D_Jackal frames (farm2_extract.py).
# Stop label = the GPS row-entrance time (crossing x=-4 m westward / x=-51 m eastward), rounded to 0.2 s,
# as written in plan_closed_loop.txt; each one was checked by eye on its strip before any model ran.
# Clip: 12 s before the stop + 4 s after. gps.json holds the distance driven at every frame (for closed_loop_sim / farm_score).
#
#   python farm2_episodes.py                 # all episodes whose frames exist
#   python farm2_episodes.py --ids 7 8 9     # only the wording test set

import argparse
import csv
import json
import os
import shutil

INSTR = "Drive into the next tree row and stop when trees are close on both the left and right sides of you."
ENTRANCES = [159.4, 227.1, 290.5, 453.6, 510.7, 570.4, 346.2, 400.2, 639.2]   # farmB_1..6 (closed-loop test), farmB_7..9 (wording test)
ap = argparse.ArgumentParser()
ap.add_argument("--ids", type=int, nargs="+", default=None, help="only these ids (default: all whose frames exist)")
ap.add_argument("--src", default="farm2/04_13D_Jackal", help="folder made by farm2_extract.py")
ap.add_argument("--prefix", default="farmB", help="episode names <prefix>_<n>")
ap.add_argument("--entrances", type=float, nargs="+", default=None, help="entrance times (s); default: the 04_13D list")
ARGS = ap.parse_args()
SRC = ARGS.src
if ARGS.entrances:
    ENTRANCES = ARGS.entrances

rows = {int(r["k"]): r for r in csv.DictReader(open(f"{SRC}/frames.csv"))}
for n, te in enumerate(ENTRANCES, 1):
    if ARGS.ids and n not in ARGS.ids:
        continue
    ks = round(te * 5)
    k0, k1 = ks - 60, ks + 20
    missing = [k for k in range(k0, k1 + 1) if k not in rows]
    if len(missing) > 4 or any(not any(k + d in rows for d in (-2, -1, 1, 2)) for k in missing):
        print(f"{ARGS.prefix}_{n} (entrance {te} s): {len(missing)} frames missing (first {missing[0] / 5:.1f} s): download its camera files first")
        continue
    filled = {}
    for k in missing:                    # a single dropped camera frame: use the nearest frame (recorded in info.json)
        near = min((k + d for d in (-1, 1, -2, 2) if k + d in rows), key=lambda x: abs(x - k))
        filled[k] = near
    out = f"dataset/{ARGS.prefix}_{n}"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(f"{out}/frames")
    d0 = float(rows[k0]["dist_m"]) if k0 in rows else float(rows[filled[k0]]["dist_m"])
    gps = []
    for j, k in enumerate(range(k0, k1 + 1)):
        a, b = divmod(j * 2, 10)
        kk = filled.get(k, k)
        shutil.copy(f"{SRC}/frames/{rows[kk]['file']}", f"{out}/frames/frame_{a}'{b}.jpg")
        if k in rows:
            dist, x, y = float(rows[k]["dist_m"]), float(rows[k]["x_m"]), float(rows[k]["y_m"])
        else:                            # position of a filled frame: halfway between its neighbours
            nb = [rows[q] for q in (k - 1, k + 1) if q in rows] or [rows[kk]]
            dist, x, y = (sum(float(r[c]) for r in nb) / len(nb) for c in ("dist_m", "x_m", "y_m"))
        gps.append({"t": round(j * 0.2, 1), "src_t": round(k / 5, 1), "dist_m": round(dist - d0, 3), "x_m": x, "y_m": y})
    stop_t, end_t = 12.0, round((k1 - k0) * 0.2 + 0.2, 1)
    json.dump({"instruction": INSTR, "length": round(float(rows[filled.get(ks, ks)]["dist_m"]) - d0, 2), "orientation": "landscape",
               "frame_size": [1280, 720], "source": f"CitrusFarm {os.path.basename(SRC)}, {k0 / 5:.1f}-{k1 / 5:.1f} s, entrance {te} s",
               "split": "farm-closed-loop-test",
               "filled_frames": [round(k / 5, 1) for k in sorted(filled)]}, open(f"{out}/info.json", "w"), indent=2)
    json.dump([{"time_range": [0.0, stop_t], "action": "[FORWARD]"}, {"time_range": [stop_t, end_t], "action": "[STOP]"}],
              open(f"{out}/label.json", "w"), indent=2)
    json.dump(gps, open(f"{out}/gps.json", "w"))
    print(f"{out}: {k1 - k0 + 1} frames ({len(filled)} filled), stop at 12.0 s (drive time {ks / 5:.1f} s), {gps[60]['dist_m']:.1f} m to the stop")
