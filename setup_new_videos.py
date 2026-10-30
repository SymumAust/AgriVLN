# Set up new episodes from phone videos in ~/Downloads.
#
# Step 1, see which videos you have (newest first, with duration and orientation):
#   python setup_new_videos.py --list
#
# Step 2, build the episodes (id=file, pot or planter):
#   python setup_new_videos.py 5=VID2026...mp4:pot 6=VID2026...mp4:pot 7=VID2026...mp4:pot 8=VID2026...mp4:planter
#
# For each episode: copies the video to videos/video<id>.mp4, runs make_episode.py, runs forward_motion.py,
# sets length = walking time x 1.2 m/s, and prints the end of the motion table so the label can be chosen.

import glob
import json
import os
import re
import shutil
import subprocess
import sys

DL = os.path.expanduser("~/Downloads")
INSTR = {
    "pot": "Walk forward and stop when the terracotta plant pot with the small palm tree is right in front of you.",
    "planter": "Walk forward and stop when the tall rectangular concrete planter is right in front of you.",
}
SPEED = 1.2


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation:format=duration", "-of", "json", path],
                         capture_output=True, text=True).stdout
    d = json.loads(out or "{}")
    s = (d.get("streams") or [{}])[0]
    w, h = s.get("width", 0), s.get("height", 0)
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        rot = abs(int(sd.get("rotation", 0)))
    if rot in (90, 270):
        w, h = h, w
    return float(d.get("format", {}).get("duration", 0)), ("portrait" if h > w else "landscape")


if len(sys.argv) < 2 or sys.argv[1] == "--list":
    files = sorted(glob.glob(f"{DL}/*.mp4"), key=os.path.getmtime, reverse=True)[:8]
    print(f"{'file':<34} {'seconds':>8}  orientation")
    for f in files:
        dur, ori = probe(f)
        print(f"{os.path.basename(f):<34} {dur:8.1f}  {ori}")
    sys.exit()

for arg in sys.argv[1:]:
    m = re.fullmatch(r"(\d+)=(.+\.mp4):(pot|planter)", arg)
    if not m:
        sys.exit(f"[ERROR] could not read '{arg}'. Use id=FILE.mp4:pot or id=FILE.mp4:planter")
    i, f, target = int(m.group(1)), m.group(2), m.group(3)
    ep = f"garden_{i}"
    src = f"{DL}/{f}"
    if not os.path.exists(src):
        sys.exit(f"[ERROR] {src} not found. Run: python setup_new_videos.py --list")
    shutil.copy(src, f"videos/video{i}.mp4")
    out = subprocess.run([sys.executable, "make_episode.py", "--video", f"videos/video{i}.mp4", "--place", "garden",
                          "--id", str(i), "--instruction", INSTR[target], "--length", "10"],
                         capture_output=True, text=True)
    fm = subprocess.run([sys.executable, "forward_motion.py", "--ep", ep], capture_output=True, text=True).stdout
    x = float(re.search(r"Forward motion stops at ([\d.]+) s", fm).group(1))
    p = f"dataset/{ep}/info.json"
    d = json.load(open(p))
    d["length"] = round(x * SPEED, 1)
    d["length_note"] = f"walking time x {SPEED} m/s"
    json.dump(d, open(p, "w"), indent=2)
    print(f"\n===== {ep}: {f} ({target}, {d['orientation']})")
    print(out.stdout.strip() or out.stderr.strip())
    print(f"forward motion stops at {x} s  ->  length {d['length']} m")
    print("\n".join(fm.splitlines()[-34:]))
