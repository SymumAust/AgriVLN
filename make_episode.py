# Convert a video into an AgriVLN/A2A-style episode folder.
import argparse, glob, json, os, shutil, subprocess

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
p.add_argument("--place", required=True)        # e.g. garden
p.add_argument("--id", required=True, type=int) # e.g. 1
p.add_argument("--instruction", required=True)
p.add_argument("--length", required=True, type=float)  # path length in metres
a = p.parse_args()

out = f"dataset/{a.place}_{a.id}"
frames, tmp = f"{out}/frames", f"{out}/_tmp"
os.makedirs(frames, exist_ok=True); os.makedirs(tmp, exist_ok=True)

# 5 frames per second, same as AgriVLN (FPS = 5.0); long side 1280 px (portrait videos stay portrait)
subprocess.run(["ffmpeg", "-loglevel", "error", "-i", a.video,
                "-vf", "fps=5,scale='if(gt(iw,ih),1280,-2)':'if(gt(iw,ih),-2,1280)'", "-q:v", "2", f"{tmp}/%06d.jpg"], check=True)

files = sorted(glob.glob(f"{tmp}/*.jpg"))
for i, f in enumerate(files):
    t_a, t_b = divmod(i * 2, 10)               # frame i is at i*0.2 s
    os.replace(f, f"{frames}/frame_{t_a}'{t_b}.jpg")
shutil.rmtree(tmp)

from PIL import Image
w, h = Image.open(sorted(glob.glob(f"{frames}/*.jpg"))[0]).size
json.dump({"instruction": a.instruction, "length": a.length,
           "orientation": "portrait" if h > w else "landscape", "frame_size": [w, h]},
          open(f"{out}/info.json", "w"), indent=2)

dur = len(files) / 5
if not os.path.exists(f"{out}/label.json"):
    json.dump([{"time_range": [0.0, round(dur - 1, 1)], "action": "[FORWARD]"},
               {"time_range": [round(dur - 1, 1), round(dur, 1)], "action": "[STOP]"}],
              open(f"{out}/label.json", "w"), indent=2)

print(f"Created {out}: {len(files)} frames, {dur:.1f} s. Now edit {out}/label.json")
