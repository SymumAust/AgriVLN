# Find the labelled STOP time with a fixed rule, not by eye:
#   STOP starts at the first frame after which the camera stays (almost) still until the end.
# "Still" = the change between frames, averaged over 1 s (5 frames, against hand shake), is below
#           40 % of the typical change while walking.
#
#   python find_stop.py --ep garden_3            # show the motion and the suggested stop
#   python find_stop.py --ep garden_3 --write    # also write dataset/garden_3/label.json

import argparse
import glob
import json
import re

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--ep", required=True)
p.add_argument("--still", type=float, default=0.40, help="still = motion below this share of walking motion")
p.add_argument("--write", action="store_true")
a = p.parse_args()


def tsec(f):
    m = re.search(r"frame_(\d+)'(\d)\.jpg$", f)
    return int(m.group(1)) + int(m.group(2)) / 10


frames = sorted(glob.glob(f"dataset/{a.ep}/frames/frame_*.jpg"), key=tsec)
imgs = [np.asarray(Image.open(f).convert("L").resize((160, 90)), dtype=np.float32) for f in frames]
times = [tsec(f) for f in frames]
motion = [0.0] + [float(np.mean(np.abs(imgs[i] - imgs[i - 1]))) for i in range(1, len(imgs))]
walking = float(np.median(motion[1:max(2, len(motion) // 2)]))   # typical motion in the first half
limit = a.still * walking
smooth = [float(np.mean(motion[max(1, i - 2):i + 3])) for i in range(len(motion))]   # 1 s average

stop_i = len(motion) - 1
for i in range(len(motion) - 1, 0, -1):          # walk back from the end while the camera is still
    if smooth[i] < limit:
        stop_i = i
    else:
        break
stop_t = times[stop_i]
end_t = round(times[-1] + 0.2, 1)

print(f"typical walking motion {walking:.1f}, 'still' below {limit:.1f}")
print("   time  motion  1s-avg")
for t, m, sm in zip(times, motion, smooth):
    if t >= times[-1] - 6:
        bar = "#" * int(sm / max(smooth) * 40)
        print(f"  {t:5.1f} s  {m:5.1f}  {sm:5.1f}  {bar}{'   <- STOP starts' if t == stop_t else ''}")
print(f"\nSuggested label: FORWARD 0.0-{stop_t} s, STOP {stop_t}-{end_t} s  (still for {end_t - stop_t:.1f} s)")
if end_t - stop_t < 1.0:
    print("[WARNING] still for less than 1 s: the camera may not have been held still at the end.")

if a.write:
    label = [{"time_range": [0.0, stop_t], "action": "[FORWARD]"},
             {"time_range": [stop_t, end_t], "action": "[STOP]"}]
    json.dump(label, open(f"dataset/{a.ep}/label.json", "w"), indent=2)
    print(f"written dataset/{a.ep}/label.json")
