# Independent perception for the STOP decision: WHAT is in front of the robot, and HOW FAR.
#
#   python perception.py --ep garden_1 --target "round stone basin with a bamboo spout"
#   python perception.py --ep garden_2 --target "tall rectangular concrete planter"
#   python perception.py --ep garden_1 --frames 5        # quick test on the first 5 frames
#
# For every frame in dataset/<ep>/frames it runs
#   - Grounding DINO (open-vocabulary detector): is the target in the image, and where?
#     -> score 0..1 (semantic evidence, a first U_s) and a bounding box
#   - Depth Anything V2 (metric, indoor): distance in metres for every pixel
#     -> median distance inside the target's box (a first U_d input)
# and saves dataset/<ep>/perception.json plus a plot perception_<ep>.png.
# Nothing here uses the VLM, so it is independent of AgriVLN's own claims.

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from PIL import Image
from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                          AutoModelForZeroShotObjectDetection, AutoProcessor)

DET_MODEL = "IDEA-Research/grounding-dino-tiny"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

parser = argparse.ArgumentParser()
parser.add_argument("--ep", required=True, help="episode folder name, e.g. garden_1")
parser.add_argument("--target", help="short name of the final target (saved into info.json)")
parser.add_argument("--frames", type=int, default=0, help="only the first N frames (quick test)")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

ep_dir = f"dataset/{args.ep}"
info_path = f"{ep_dir}/info.json"
info = json.load(open(info_path))
if args.target:
   info["target"] = args.target
   json.dump(info, open(info_path, "w"), indent=2)
target = info.get("target")
if not target:
   raise SystemExit(f'[ERROR] No target. Run with --target "<name of the final target>" first.')


def frame_time(path):
   m = re.search(r"frame_(\d+)'(\d)\.jpg$", path)
   return int(m.group(1)) + int(m.group(2)) / 10, f"{m.group(1)}'{m.group(2)}"


frames = sorted(glob.glob(f"{ep_dir}/frames/frame_*.jpg"), key=lambda p: frame_time(p)[0])
if args.frames:
   frames = frames[: args.frames]
print(f"[INFO] {args.ep}: {len(frames)} frames, target = '{target}', device = {args.device}")

dev = args.device
det_proc = AutoProcessor.from_pretrained(DET_MODEL)
det_model = AutoModelForZeroShotObjectDetection.from_pretrained(DET_MODEL).to(dev).eval()
dep_proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
dep_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(dev).eval()
text = target.lower().strip().rstrip(".") + "."


def detect(img):
   inputs = det_proc(images=img, text=text, return_tensors="pt").to(dev)
   with torch.no_grad():
      out = det_model(**inputs)
   kw = dict(target_sizes=[img.size[::-1]], text_threshold=0.1)
   try:
      res = det_proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=0.1, **kw)[0]
   except TypeError:   # older transformers versions call it box_threshold
      res = det_proc.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=0.1, **kw)[0]
   if len(res["scores"]) == 0:
      return 0.0, None
   i = int(torch.argmax(res["scores"]))
   return float(res["scores"][i]), [round(float(v), 1) for v in res["boxes"][i]]


def depth_map(img):
   inputs = dep_proc(images=img, return_tensors="pt").to(dev)
   with torch.no_grad():
      d = dep_model(**inputs).predicted_depth          # metres
   d = torch.nn.functional.interpolate(d.unsqueeze(1), size=img.size[::-1], mode="bicubic",
                                       align_corners=False)[0, 0]
   return d.float().cpu().numpy()


records = []
for k, path in enumerate(frames):
   t, tname = frame_time(path)
   img = Image.open(path).convert("RGB")
   score, box = detect(img)
   d = depth_map(img)
   h, w = d.shape
   center = float(np.median(d[int(0.3 * h):int(0.7 * h), int(0.35 * w):int(0.65 * w)]))
   box_depth = None
   if box:
      x0, y0, x1, y1 = box
      cx0, cx1 = int(x0 + 0.25 * (x1 - x0)), int(x1 - 0.25 * (x1 - x0))   # central half of the box
      cy0, cy1 = int(y0 + 0.25 * (y1 - y0)), int(y1 - 0.25 * (y1 - y0))
      patch = d[max(cy0, 0):max(cy1, cy0 + 1), max(cx0, 0):max(cx1, cx0 + 1)]
      box_depth = round(float(np.median(patch)), 2) if patch.size else None
   rec = {"time": tname, "t": t, "score": round(score, 3), "box": box,
          "box_depth": box_depth, "center_depth": round(center, 2)}
   records.append(rec)
   print(f"  {tname:>5}  target score {rec['score']:.2f}   box depth {str(box_depth):>5} m   centre depth {rec['center_depth']:.2f} m")

json.dump({"target": target, "det_model": DET_MODEL, "depth_model": DEPTH_MODEL, "frames": records},
          open(f"{ep_dir}/perception.json", "w"), indent=1)
print(f"[INFO] saved {ep_dir}/perception.json")

# ---- plot: score and distance over time, with the labelled stop time ----
try:
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   labels = json.load(open(f"{ep_dir}/label.json"))
   stop_t = next((l["time_range"][0] for l in labels if l["action"] == "[STOP]"), None)
   ts = [r["t"] for r in records]
   fig, ax1 = plt.subplots(figsize=(10, 4))
   ax1.plot(ts, [r["score"] for r in records], color="#1E4E79", label="target score (detector)")
   ax1.axhline(0.35, color="#1E4E79", ls=":", lw=1)
   ax1.set_ylabel("target score", color="#1E4E79"); ax1.set_ylim(0, 1); ax1.set_xlabel("time (s)")
   ax2 = ax1.twinx()
   ax2.plot(ts, [r["box_depth"] if r["box_depth"] is not None else np.nan for r in records],
            color="#C0392B", label="distance to target box (m)")
   ax2.axhline(2.0, color="#C0392B", ls=":", lw=1)
   ax2.set_ylabel("distance to target (m)", color="#C0392B"); ax2.set_ylim(0, 8)
   if stop_t is not None:
      ax1.axvline(stop_t, color="#2E7D32", lw=2, label=f"real stop ({stop_t} s)")
   fig.legend(loc="upper right", fontsize=8)
   ax1.set_title(f"{args.ep}: is '{target}' in view, and how far?")
   fig.tight_layout()
   fig.savefig(f"perception_{args.ep}.png", dpi=130)
   print(f"[INFO] saved perception_{args.ep}.png")
except Exception as e:
   print(f"[WARNING] plot skipped: {e}")
