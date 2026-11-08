# Depth near the left and right image edges, for every 5 Hz frame, from the ZED stereo depth in the same bags.
#
#   python farm2_depth.py --gt farm2/gt_04_13D_Jackal.csv --out farm2/04_13D_Jackal ~/datasets/citrusfarm/04_13D_Jackal/zed_*.bag
#
# Output: <out>/depth.csv with, per frame (same k / t grid as frames.csv):
#   left_m, right_m  : median depth (m) of the valid pixels in the left / right edge band
#                      (outer 15 % of the width, rows 25-75 % of the height)
#   centre_m         : median depth of the centre band (middle 30 % of width, same rows)
#   left_valid, right_valid : share of pixels in the band with a valid depth (0-1)
#   left_conf, right_conf   : median ZED confidence value in the band (ZED: lower = more confident)
# and <out>/depth/d_<k>.png, a small colour depth picture per frame (near = yellow, far = dark), to check by eye.

import argparse
import csv
import os

import numpy as np
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

p = argparse.ArgumentParser()
p.add_argument("bags", nargs="+")
p.add_argument("--gt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--edge", type=float, default=0.15)
p.add_argument("--max_m", type=float, default=20.0, help="depth beyond this is treated as invalid")
a = p.parse_args()

ts = get_typestore(Stores.ROS1_NOETIC)
rows = [r for r in csv.reader(open(a.gt)) if r and not r[0].startswith("#")]
T0 = min(float(r[0]) for r in rows)
os.makedirs(f"{a.out}/depth", exist_ok=True)


def to_array(msg):
    enc = msg.encoding.lower()
    buf = bytes(msg.data)
    if enc == "32fc1":
        d = np.frombuffer(buf, np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width]
    elif enc in ("16uc1", "mono16"):
        d = np.frombuffer(buf, np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width].astype(np.float32) / 1000.0
    elif enc in ("8uc1", "mono8"):
        d = np.frombuffer(buf, np.uint8).reshape(msg.height, msg.step)[:, :msg.width].astype(np.float32)
    else:
        raise ValueError(f"unknown encoding {msg.encoding}")
    return d


def band(d, x0, x1):
    h, w = d.shape
    return d[int(0.25 * h):int(0.75 * h), int(x0 * w):int(x1 * w)]


def stats(b, valid_max):
    ok = np.isfinite(b) & (b > 0.1) & (b < valid_max)
    return (float(np.median(b[ok])) if ok.any() else float("nan")), float(ok.mean())


e = a.edge
out_rows, done = [], set()
for bag in sorted(a.bags):
    with Reader(bag) as r:
        cd = [c for c in r.connections if c.topic.endswith("depth/depth_registered")]
        cc = [c for c in r.connections if c.topic.endswith("confidence/confidence_map")]
        pend_d, pend_c, n, enc = {}, {}, 0, "-"
        for conn, stamp, raw in r.messages(connections=cd + cc):
            msg = ts.deserialize_ros1(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - T0
            k = round(t * 5)
            if abs(t - k / 5) > 0.06 or t < 0 or k in done:
                continue
            (pend_d if conn in cd else pend_c).setdefault(k, to_array(msg))
            if conn in cd:
                enc = msg.encoding
            for kk in [x for x in pend_d if x in pend_c or x < k - 10]:   # pair depth with its confidence map, keep memory small
                d, c = pend_d.pop(kk), pend_c.pop(kk, None)
                lm, lv = stats(band(d, 0, e), a.max_m)
                rm, rv = stats(band(d, 1 - e, 1), a.max_m)
                cm, _ = stats(band(d, 0.35, 0.65), a.max_m)
                lc = float(np.nanmedian(band(c, 0, e))) if c is not None else float("nan")
                rc = float(np.nanmedian(band(c, 1 - e, 1))) if c is not None else float("nan")
                out_rows.append([kk, round(kk / 5, 1), round(lm, 2), round(rm, 2), round(cm, 2), round(lv, 2), round(rv, 2), round(lc, 1), round(rc, 1)])
                v = np.clip(np.nan_to_num(d, nan=a.max_m, posinf=a.max_m), 0, 10) / 10.0
                rgb = (np.stack([1 - v, 1 - v * 0.8, v * 0.4], -1) * 255).astype(np.uint8)      # near = yellow, far = dark blue
                Image.fromarray(rgb).resize((320, 180)).save(f"{a.out}/depth/d_{kk:06d}.png")
                done.add(kk)
                n += 1
            for kk in [x for x in pend_c if x < k - 10]:
                pend_c.pop(kk)
        print(f"{os.path.basename(bag)}: {n} depth frames, encoding {enc}")

out_rows.sort()
with open(f"{a.out}/depth.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["k", "t", "left_m", "right_m", "centre_m", "left_valid", "right_valid", "left_conf", "right_conf"])
    w.writerows(out_rows)
print(f"{len(out_rows)} frames -> {a.out}/depth.csv and {a.out}/depth/")
