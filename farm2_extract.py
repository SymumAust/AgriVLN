# Extract the front camera (ZED left) from CitrusFarm ROS1 bags at 5 frames per second, with the GPS position
# (ground truth gt.csv) of every frame. Works on just the bags you downloaded; no ROS installation needed.
#
#   python farm2_extract.py --list ~/datasets/citrusfarm/04_13D_Jackal/zed_2023-07-18-14-27-58_5.bag
#   python farm2_extract.py --gt farm2/gt_04_13D_Jackal.csv --out farm2/04_13D_Jackal ~/datasets/citrusfarm/04_13D_Jackal/zed_*.bag
#
# Output in --out:  frames/f_<time x5>.jpg (1280x720), frames.csv (t from the start of the drive, x, y, dist_m),
#                   and a strip image around every entrance time in --entrances (for choosing the stop frame by eye).
# Needs: pip install rosbags pillow numpy

import argparse
import csv
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

p = argparse.ArgumentParser()
p.add_argument("bags", nargs="+")
p.add_argument("--list", action="store_true", help="only list the topics in the bags")
p.add_argument("--topic", default=None, help="default: the first topic with 'left' and 'image' in its name, rectified colour preferred")
p.add_argument("--gt", default=None)
p.add_argument("--out", default=None)
p.add_argument("--entrances", type=float, nargs="*", default=[159.4, 227.1, 290.5, 453.6, 510.7, 570.4],
               help="seconds from the start of the drive (from the GPS analysis) to make strips around")
a = p.parse_args()

ts = get_typestore(Stores.ROS1_NOETIC)
bags = sorted(a.bags)

if a.list:
    with Reader(bags[0]) as r:
        print(f"{bags[0]}: {r.duration / 1e9:.1f} s, {r.message_count} messages")
        for c in r.connections:
            print(f"  {c.topic:<55} {c.msgtype:<35} {c.msgcount:>6}")
    sys.exit()

rows = [r for r in csv.reader(open(a.gt)) if r and not r[0].startswith("#")]
GT = np.array([[float(x) for x in r[:3]] for r in rows])
GT = GT[np.argsort(GT[:, 0])]
T0 = GT[0, 0]
gt_t, gx, gy = GT[:, 0] - T0, GT[:, 1], GT[:, 2]
gd = np.concatenate([[0], np.cumsum(np.hypot(np.diff(gx), np.diff(gy)))])
os.makedirs(f"{a.out}/frames", exist_ok=True)


def to_rgb(msg):
    if msg.__msgtype__.endswith("CompressedImage"):
        import io
        return Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    ch = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3, "mono8": 1}[enc]
    img = buf.reshape(msg.height, msg.step)[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    if enc.startswith("bgr"):
        img = img[:, :, [2, 1, 0]]
    elif ch == 4:
        img = img[:, :, :3]
    return Image.fromarray(img if ch > 1 else img[:, :, 0]).convert("RGB")


kept = {}
for bag in bags:
    with Reader(bag) as r:
        topic = a.topic
        if topic is None:
            cands = [c.topic for c in r.connections if "left" in c.topic and "image" in c.topic and "depth" not in c.topic]
            cands.sort(key=lambda t: (("rect_color" not in t), ("compressed" in t), t))
            topic = cands[0]
        conns = [c for c in r.connections if c.topic == topic]
        n = 0
        for conn, stamp, raw in r.messages(connections=conns):
            msg = ts.deserialize_ros1(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - T0
            k = round(t * 5)                              # 5 Hz grid: frame k at k/5 s
            if k in kept or abs(t - k / 5) > 0.06 or t < 0:
                continue
            to_rgb(msg).resize((1280, 720)).save(f"{a.out}/frames/f_{k:06d}.jpg", quality=90)
            kept[k] = t
            n += 1
        print(f"{os.path.basename(bag)}: topic {topic}, kept {n} frames")

with open(f"{a.out}/frames.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["k", "t", "file", "x_m", "y_m", "dist_m"])
    for k in sorted(kept):
        t = k / 5
        w.writerow([k, round(t, 1), f"f_{k:06d}.jpg", round(float(np.interp(t, gt_t, gx)), 3),
                    round(float(np.interp(t, gt_t, gy)), 3), round(float(np.interp(t, gt_t, gd)), 3)])
print(f"{len(kept)} frames in {a.out}/frames, positions in {a.out}/frames.csv")

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
except Exception:
    font = ImageFont.load_default()
for te in a.entrances:
    ks = [k for k in range(round((te - 8) * 5), round((te + 6) * 5) + 1, 2) if k in kept]
    if not ks:
        continue
    TW, TH = 320, 180
    strip = Image.new("RGB", (TW * 7, TH * ((len(ks) + 6) // 7)), "white")
    d = ImageDraw.Draw(strip)
    for j, k in enumerate(ks):
        im = Image.open(f"{a.out}/frames/f_{k:06d}.jpg").resize((TW - 4, TH - 4))
        x0, y0 = (j % 7) * TW, (j // 7) * TH
        strip.paste(im, (x0 + 2, y0 + 2))
        d.rectangle([x0 + 2, y0 + 2, x0 + 150, y0 + 30], fill="black")
        d.text((x0 + 6, y0 + 4), f"{k / 5:.1f}s", fill="yellow", font=font)
    strip.save(f"{a.out}/entrance_{te:.0f}.jpg", quality=85)
    print(f"strip {a.out}/entrance_{te:.0f}.jpg ({ks[0] / 5:.1f}-{ks[-1] / 5:.1f} s, every 0.4 s)")
