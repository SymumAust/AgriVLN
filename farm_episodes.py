# Step 3 of the farm pipeline: cut episodes from the CitrusFarm drive, in the same format as dataset/garden_*.
#
#   python farm_episodes.py                 # builds dataset/farm_1 .. farm_5 and farm_preview.jpg
#
# Targets were chosen from the contact sheets BEFORE any model was run on this data, and the instructions
# below are fixed now (pre-registered, like garden_9..12). Stop frame = the frame where the condition
# becomes true; please check farm_preview.jpg (the stop frame is the green one) and change STOP if needed.
#
# Each episode: 20 s before the stop frame + 4 s after (the robot keeps driving; this lets late stops be measured).
# Extra file gps.json: distance driven (m, from RTK GPS) for every frame, so farm_score.py can measure the
# stop error in real metres instead of from time.

import csv
import json
import os
import shutil

from PIL import Image, ImageDraw, ImageFont

SRC = "farm/06_14B_Jackal"
BEFORE, AFTER = 100, 20          # frames (5 per second)
TARGETS = [  # (episode id, stop frame #, instruction)
    (1, 339, "Drive along the tree row and stop when the trees on both sides are behind you."),
    (2, 708, "Drive along the tree row and stop when the trees on both sides are behind you."),
    (3, 1082, "Drive along the tree row and stop when the trees on both sides are behind you."),
    (4, 1468, "Drive along the tree row and stop when the trees on both sides are behind you."),
    (5, 760, "Drive forward and stop when the white concrete block is right beside you on the left."),
]

rows = list(csv.DictReader(open(f"{SRC}/frames.csv")))
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
except Exception:
    font = ImageFont.load_default()
strips = []
for eid, stop, instr in TARGETS:
    a, b = max(0, stop - BEFORE), min(len(rows) - 1, stop + AFTER)
    out = f"dataset/farm_{eid}"
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(f"{out}/frames")
    d0 = float(rows[a]["dist_m"])
    gps = []
    for k, i in enumerate(range(a, b + 1)):
        t_a, t_b = divmod(k * 2, 10)                  # frame k is at k*0.2 s, same naming as make_episode.py
        shutil.copy(f"{SRC}/frames/{rows[i]['file']}", f"{out}/frames/frame_{t_a}'{t_b}.jpg")
        gps.append({"t": round(k * 0.2, 1), "src_frame": i, "dist_m": round(float(rows[i]["dist_m"]) - d0, 3),
                    "x_m": float(rows[i]["x_m"]), "y_m": float(rows[i]["y_m"])})
    stop_t = round((stop - a) * 0.2, 1)
    end_t = round((b - a) * 0.2 + 0.2, 1)
    length = round(float(rows[stop]["dist_m"]) - d0, 2)
    json.dump({"instruction": instr, "length": length, "orientation": "landscape", "frame_size": [1280, 720],
               "source": f"CitrusFarm {SRC.split('/')[-1]}, frames {a}-{b}, stop frame {stop}",
               "split": "farm-test", "length_note": "RTK GPS distance from the first frame to the stop frame"},
              open(f"{out}/info.json", "w"), indent=2)
    json.dump([{"time_range": [0.0, stop_t], "action": "[FORWARD]"},
               {"time_range": [stop_t, end_t], "action": "[STOP]"}], open(f"{out}/label.json", "w"), indent=2)
    json.dump(gps, open(f"{out}/gps.json", "w"))
    print(f"{out}: {b - a + 1} frames ({end_t:.1f} s), stop at {stop_t} s (frame #{stop}), {length:.1f} m driven to the stop")

    # preview strip: 3 s before .. 2 s after the stop, one frame every 0.4 s
    tiles = []
    for i in range(stop - 15, stop + 11, 2):
        if 0 <= i < len(rows):
            im = Image.open(f"{SRC}/frames/{rows[i]['file']}").convert("RGB").resize((320, 180))
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 120, 32], fill="#2E7D32" if i == stop else "black")
            dr.text((5, 3), f"#{i}", fill="white", font=font)
            tiles.append(im)
    strip = Image.new("RGB", (320 * len(tiles), 180 + 40), "white")
    ImageDraw.Draw(strip).text((8, 6), f"farm_{eid}: {instr}", fill="black", font=font)
    for j, im in enumerate(tiles):
        strip.paste(im, (j * 320, 40))
    strips.append(strip)

W = max(s.width for s in strips)
prev = Image.new("RGB", (W, sum(s.height for s in strips) + 10 * len(strips)), "white")
y = 0
for s in strips:
    prev.paste(s, (0, y)); y += s.height + 10
prev = prev.resize((prev.width // 2, prev.height // 2))
prev.save("farm_preview.jpg", quality=85)
print("saved farm_preview.jpg (green = stop frame). Check it before running any model.")
