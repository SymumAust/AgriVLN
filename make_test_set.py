# Build the TEST episodes garden_9 .. garden_12 from ~/Downloads/garden9.mp4 .. garden12.mp4
#
#   python make_test_set.py
#
# Instructions were written from each video's final frame BEFORE any model was run (see plan_test_9to12.txt).
# Length is fixed at 18 m for every video (same rule as garden_1..8).
# Prints the motion table of the last seconds (forward motion + shake) so the STOP label and any
# turn at the end can be set; the label itself is written in the next step.

import json
import os
import re
import shutil
import subprocess
import sys

DL = os.path.expanduser("~/Downloads")
TEST = {
    9:  "Walk forward, then turn right and stop when the blue metal pole is right in front of you.",
    10: "Walk forward, then turn right and stop when the nearest yellow-and-black striped bollard is right in front of you.",
    11: "Walk forward, bearing slightly to the right, and stop when the orange traffic cone is right in front of you.",
    12: "Walk forward, then turn right and stop when the brown wooden planter box with black corners is right in front of you.",
}
TARGET = {9: "blue metal pole", 10: "yellow and black striped bollard",
          11: "orange traffic cone", 12: "brown wooden planter box with black corners"}

for i, instr in TEST.items():
    src = f"{DL}/garden{i}.mp4"
    if not os.path.exists(src):
        sys.exit(f"[ERROR] {src} not found")
    shutil.copy(src, f"videos/video{i}.mp4")
    out = subprocess.run([sys.executable, "make_episode.py", "--video", f"videos/video{i}.mp4", "--place", "garden",
                          "--id", str(i), "--instruction", instr, "--length", "18"], capture_output=True, text=True)
    p = f"dataset/garden_{i}/info.json"
    d = json.load(open(p))
    d["target"] = TARGET[i]
    d["length_note"] = "fixed 18 m for all videos"
    d["split"] = "test"
    json.dump(d, open(p, "w"), indent=2)
    fm = subprocess.run([sys.executable, "forward_motion.py", "--ep", f"garden_{i}"], capture_output=True, text=True).stdout
    print(f"\n===== garden_{i} ({d['orientation']})\n{out.stdout.strip() or out.stderr.strip()}")
    print("\n".join(fm.splitlines()[-40:]))
