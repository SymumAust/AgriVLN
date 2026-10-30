# Is the camera still moving FORWARD? Measured from the images, not from brightness change.
#
# Walking forward makes the picture EXPAND (things grow from the centre outward).
# Hand shake only shifts or tilts the picture, it does not make it grow.
# So for every pair of frames we match image features (ORB) and fit a similarity transform
# (shift + rotation + scale). The scale above 1 is the expansion = forward motion.
#
#   python forward_motion.py --ep garden_3          # table + plot forward_motion_garden_3.png
#
# "forward"   = expansion per frame, in % (about 0 when you stand still, even with shake)
# "shake"     = sideways shift per frame, in % of image width (hand shake / turning)
# The stop suggestion = first time after which the 1 s average expansion stays below a quarter
# of the typical expansion while walking (first half of the video).

import argparse
import glob
import re

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--ep", required=True)
p.add_argument("--share", type=float, default=0.25)
a = p.parse_args()


def tsec(f):
    m = re.search(r"frame_(\d+)'(\d)\.jpg$", f)
    return int(m.group(1)) + int(m.group(2)) / 10


frames = sorted(glob.glob(f"dataset/{a.ep}/frames/frame_*.jpg"), key=tsec)
times = [tsec(f) for f in frames]
orb = cv2.ORB_create(1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
prev = None
fwd, shake = [0.0], [0.0]
for f in frames:
    g = cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (640, 360))
    kp, des = orb.detectAndCompute(g, None)
    if prev is not None:
        kp0, des0 = prev
        s, dx = np.nan, np.nan
        if des is not None and des0 is not None and len(kp) > 20 and len(kp0) > 20:
            m = bf.match(des0, des)
            if len(m) >= 12:
                p0 = np.float32([kp0[x.queryIdx].pt for x in m])
                p1 = np.float32([kp[x.trainIdx].pt for x in m])
                M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None and inl is not None and inl.sum() >= 10:
                    s = float(np.hypot(M[0, 0], M[1, 0]))          # scale of the similarity
                    dx = float(np.hypot(M[0, 2], M[1, 2])) / 640 * 100
        fwd.append((s - 1) * 100 if s == s else np.nan)
        shake.append(dx)
    prev = (kp, des)

fwd = np.array(fwd)
f_fill = np.where(np.isnan(fwd), 0.0, np.clip(fwd, 0, None))           # only expansion counts
smooth = np.array([np.mean(f_fill[max(1, i - 2):i + 3]) for i in range(len(f_fill))])
walking = float(np.median(smooth[1:len(smooth) // 2]))
limit = a.share * walking
stop_i = len(smooth) - 1
for i in range(len(smooth) - 1, 0, -1):
    if smooth[i] < limit:
        stop_i = i
    else:
        break

print(f"typical forward expansion while walking {walking:.2f} %/frame, 'stopped' below {limit:.2f}")
print("   time   forward%  1s-avg  shake%")
for t, fv, sm, sh in zip(times, fwd, smooth, shake):
    if t >= times[-1] - 6:
        bar = "#" * int(min(sm / max(smooth.max(), 1e-6), 1) * 40)
        print(f"  {t:5.1f} s  {fv:7.2f}  {sm:6.2f}  {sh:6.2f}  {bar}{'   <- stopped from here' if t == times[stop_i] else ''}")
still = round(times[-1] + 0.2 - times[stop_i], 1)
print(f"\nForward motion stops at {times[stop_i]} s (no forward motion for the last {still} s)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(times, smooth, color="#2E6DA4", lw=2, label="forward motion (1 s average)")
    ax.axhline(limit, color="#888", ls=":", lw=1, label="'stopped' level")
    ax.axvline(times[stop_i], color="#2E7D32", lw=2, label=f"stops at {times[stop_i]} s")
    ax.set_xlabel("time (s)"); ax.set_ylabel("expansion %/frame"); ax.legend(frameon=False, fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(f"forward_motion_{a.ep}.png", dpi=130)
    print(f"saved forward_motion_{a.ep}.png")
except Exception as e:
    print("plot skipped:", e)
