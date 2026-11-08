# Build test episodes farmB_1 .. farmB_6 from the extracted 04_13D_Jackal frames (farm2_extract.py).
# Stop label = the GPS row-entrance time (crossing x=-4 m westward / x=-51 m eastward), rounded to 0.2 s,
# as written in plan_closed_loop.txt; each one was checked by eye on its strip before any model ran.
# Clip: 12 s before the stop + 4 s after. gps.json holds the distance driven at every frame (for closed_loop_sim / farm_score).
#
#   python farm2_episodes.py

import csv
import json
import os
import shutil

SRC = "farm2/04_13D_Jackal"
INSTR = "Drive into the next tree row and stop when trees are close on both the left and right sides of you."
ENTRANCES = [159.4, 227.1, 290.5, 453.6, 510.7, 570.4]

rows = {int(r["k"]): r for r in csv.DictReader(open(f"{SRC}/frames.csv"))}
for n, te in enumerate(ENTRANCES, 1):
    ks = round(te * 5)
    k0, k1 = ks - 60, ks + 20
    missing = [k for k in range(k0, k1 + 1) if k not in rows]
    if missing:
        print(f"farmB_{n} (entrance {te} s): {len(missing)} frames missing (first {missing[0] / 5:.1f} s): download its camera files first")
        continue
    out = f"dataset/farmB_{n}"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(f"{out}/frames")
    d0 = float(rows[k0]["dist_m"])
    gps = []
    for j, k in enumerate(range(k0, k1 + 1)):
        a, b = divmod(j * 2, 10)
        shutil.copy(f"{SRC}/frames/{rows[k]['file']}", f"{out}/frames/frame_{a}'{b}.jpg")
        gps.append({"t": round(j * 0.2, 1), "src_t": round(k / 5, 1), "dist_m": round(float(rows[k]["dist_m"]) - d0, 3),
                    "x_m": float(rows[k]["x_m"]), "y_m": float(rows[k]["y_m"])})
    stop_t, end_t = 12.0, round((k1 - k0) * 0.2 + 0.2, 1)
    json.dump({"instruction": INSTR, "length": round(float(rows[ks]["dist_m"]) - d0, 2), "orientation": "landscape",
               "frame_size": [1280, 720], "source": f"CitrusFarm 04_13D_Jackal, {k0 / 5:.1f}-{k1 / 5:.1f} s, entrance {te} s",
               "split": "farm-closed-loop-test"}, open(f"{out}/info.json", "w"), indent=2)
    json.dump([{"time_range": [0.0, stop_t], "action": "[FORWARD]"}, {"time_range": [stop_t, end_t], "action": "[STOP]"}],
              open(f"{out}/label.json", "w"), indent=2)
    json.dump(gps, open(f"{out}/gps.json", "w"))
    print(f"{out}: {k1 - k0 + 1} frames, stop at 12.0 s (drive time {ks / 5:.1f} s), {gps[60]['dist_m']:.1f} m to the stop")
