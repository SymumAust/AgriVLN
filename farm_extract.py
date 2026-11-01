# Step 2 of the farm-episode pipeline (CitrusFarm sample, or any MCAP with foxglove messages).
# Extracts the front colour camera at 5 frames per second, the GPS position of every frame, and
# contact sheets of thumbnails so you can browse the drive and pick targets.
#
#   python farm_extract.py ~/datasets/citrusfarm_sample/data/06_14B_Jackal.fo.mcap
#
# Output folder farm/<recording name>/
#   frames/f_00000.jpg ...    every 2nd camera frame (10 Hz -> 5 Hz), full 1280x720
#   frames.csv                idx, time (s), file, lat, lon, x_m, y_m (east/north from start), dist_m (driven so far)
#   sheets/sheet_00.jpg ...   60 thumbnails per sheet (12 s of driving), labelled with idx and time
#   trajectory.png            the path seen from above, with a time label every 20 s
#   info.txt                  what was found (first GPS message, ground-truth fields, the /sequence text)
#
# Needs: pip install mcap mcap-protobuf-support imageio-ffmpeg pillow numpy matplotlib

import argparse
import bisect
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument("mcap")
p.add_argument("--camera", default="/zed-left")
p.add_argument("--gps", default="/gnss")
p.add_argument("--every", type=int, default=2, help="keep every n-th frame (10 Hz camera, 2 -> 5 Hz)")
p.add_argument("--out", default=None)
a = p.parse_args()

name = os.path.basename(a.mcap).split(".")[0]
out = a.out or f"farm/{name}"
os.makedirs(f"{out}/frames", exist_ok=True)
os.makedirs(f"{out}/sheets", exist_ok=True)
info = open(f"{out}/info.txt", "w")


def log(*s):
    print(*s)
    print(*s, file=info)


# ------------------------------------------------------------------ read camera, GPS, extras
fac = DecoderFactory()
decoders = {}
video_path = f"{out}/camera_stream.bin"
vid_t, fmt = [], None
gps = []                       # (t, lat, lon)
gt_example, seq_text = None, None
t0 = None
with open(a.mcap, "rb") as f, open(video_path, "wb") as vf:
    r = make_reader(f)
    for schema, channel, msg in r.iter_messages(topics=[a.camera, a.gps, "/ground-truth.plot", "/sequence"]):
        t = msg.log_time / 1e9
        t0 = t if t0 is None else min(t0, t)
        if channel.message_encoding == "json":
            if gt_example is None:
                gt_example = json.loads(msg.data)
            continue
        key = channel.id
        if key not in decoders:
            decoders[key] = fac.decoder_for(channel.message_encoding, schema)
        m = decoders[key](msg.data)
        if channel.topic == a.camera:
            fmt = fmt or getattr(m, "format", "h264")
            vf.write(m.data)
            vid_t.append(t)
        elif channel.topic == a.gps:
            gps.append((t, m.latitude, m.longitude))
        elif channel.topic == "/sequence":
            seq_text = str(m)[:2000]

log(f"camera {a.camera}: {len(vid_t)} messages, format '{fmt}'")
log(f"GPS {a.gps}: {len(gps)} fixes")
log(f"/ground-truth.plot example: {json.dumps(gt_example)[:400] if gt_example else 'none'}")
log(f"/sequence text: {seq_text}")
if not vid_t or not gps:
    sys.exit("[ERROR] camera or GPS topic empty: check the topic names with farm_inspect.py")

# ------------------------------------------------------------------ decode video with ffmpeg
try:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FF = "ffmpeg"
tmp = f"{out}/_all"
os.makedirs(tmp, exist_ok=True)
demux = {"h264": "h264", "h265": "hevc", "hevc": "hevc"}.get(str(fmt).lower(), "h264")
# the stream has no timestamps of its own, so give every frame one (10 Hz); try three ways, keep the first that works
tries = [
    ["-fflags", "+genpts", "-r", "10", "-f", demux, "-i", video_path, "-q:v", "3"],
    ["-f", demux, "-i", video_path, "-vf", "setpts=N/(10*TB)", "-fps_mode", "passthrough", "-q:v", "3"],
    ["-f", demux, "-i", video_path, "-vsync", "0", "-q:v", "3"],
]
dec = []
for k, opts in enumerate(tries, 1):
    for fdel in glob.glob(f"{tmp}/*.jpg"):
        os.remove(fdel)
    cmd = [FF, "-v", "error", "-y"] + opts + [f"{tmp}/%06d.jpg"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dec = sorted(glob.glob(f"{tmp}/*.jpg"))
    log(f"decoding try {k}: exit {res.returncode}, {len(dec)} frames" + (f"  ({res.stderr.strip().splitlines()[-1]})" if res.returncode else ""))
    if res.returncode == 0 and len(dec) >= 0.9 * len(vid_t):
        break
if len(dec) < 0.9 * len(vid_t):
    sys.exit("[ERROR] video could not be decoded; send me the lines above")
log(f"decoded {len(dec)} frames from {len(vid_t)} messages")
n = min(len(dec), len(vid_t))
if abs(len(dec) - len(vid_t)) > 2:
    log("[WARNING] frame count differs from message count; times may be shifted slightly")
offset = len(vid_t) - n          # if a few leading frames could not be decoded, align to the end

# ------------------------------------------------------------------ GPS -> local metres
g = np.array(gps)
lat0, lon0 = g[0, 1], g[0, 2]
R_E = 6371000.0
gx = np.radians(g[:, 2] - lon0) * R_E * np.cos(np.radians(lat0))
gy = np.radians(g[:, 1] - lat0) * R_E
gt = g[:, 0]


def at(t):
    return float(np.interp(t, gt, gx)), float(np.interp(t, gt, gy)), float(np.interp(t, gt, g[:, 1])), float(np.interp(t, gt, g[:, 2]))


rows, dist, prev = [], 0.0, None
for k, i in enumerate(range(0, n, a.every)):
    t = vid_t[i + offset]
    x, y, la, lo = at(t)
    if prev is not None:
        dist += float(np.hypot(x - prev[0], y - prev[1]))
    prev = (x, y)
    fn = f"f_{k:05d}.jpg"
    os.replace(dec[i], f"{out}/frames/{fn}")
    rows.append(dict(idx=k, time=round(t - t0, 2), file=fn, lat=f"{la:.8f}", lon=f"{lo:.8f}", x_m=round(x, 3), y_m=round(y, 3), dist_m=round(dist, 2)))
for fdel in glob.glob(f"{tmp}/*.jpg"):
    os.remove(fdel)
os.rmdir(tmp)
os.remove(video_path)
with open(f"{out}/frames.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
log(f"kept {len(rows)} frames at {10 / a.every:.0f} Hz, path driven {dist:.1f} m, saved {out}/frames.csv")

# ------------------------------------------------------------------ contact sheets
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
except Exception:
    font = ImageFont.load_default()
TW, TH, COLS, PER = 320, 180, 10, 60
for s in range(0, len(rows), PER):
    chunk = rows[s:s + PER]
    sheet = Image.new("RGB", (COLS * TW, ((len(chunk) - 1) // COLS + 1) * TH), "white")
    d = ImageDraw.Draw(sheet)
    for j, rw in enumerate(chunk):
        im = Image.open(f"{out}/frames/{rw['file']}").convert("RGB").resize((TW - 4, TH - 4))
        x0, y0 = (j % COLS) * TW, (j // COLS) * TH
        sheet.paste(im, (x0 + 2, y0 + 2))
        d.rectangle([x0 + 2, y0 + 2, x0 + 150, y0 + 28], fill="black")
        d.text((x0 + 6, y0 + 4), f"#{rw['idx']}  {rw['time']:.1f}s", fill="yellow", font=font)
    sheet.save(f"{out}/sheets/sheet_{s // PER:02d}.jpg", quality=80)
log(f"contact sheets: {out}/sheets/ ({(len(rows) - 1) // PER + 1} sheets, {PER} frames = {PER / (10 / a.every):.0f} s each)")

# ------------------------------------------------------------------ trajectory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
xs = [r_["x_m"] for r_ in rows]; ys = [r_["y_m"] for r_ in rows]
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(xs, ys, color="#1E4E79", lw=1.5)
ax.plot(xs[0], ys[0], "o", color="#2E7D32", ms=9, label="start")
ax.plot(xs[-1], ys[-1], "s", color="#C0392B", ms=9, label="end")
step = int(20 * 10 / a.every)
for rw in rows[::step]:
    ax.annotate(f"{rw['time']:.0f}s #{rw['idx']}", (rw["x_m"], rw["y_m"]), fontsize=8, color="#595959")
ax.set_aspect("equal"); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)"); ax.legend()
ax.set_title(f"{name}: {dist:.0f} m driven")
fig.tight_layout(); fig.savefig(f"{out}/trajectory.png", dpi=130)
log(f"trajectory: {out}/trajectory.png")
info.close()
