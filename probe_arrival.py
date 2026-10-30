# Ask a vision model ONE question on every frame, outside the navigation loop:
#   "Is the instruction's arrival condition true in this image right now?"
# This tests the model itself (does it see arrival only at the real stop?), with nothing else changed.
#
#   python probe_arrival.py --models qwen2.5vl:7b qwen3-vl:8b --eps garden_1 garden_2
#
# Ground truth from label.json:  arrived = frame time >= labelled STOP time,
#                                 not arrived = more than 2 s before it (the last 2 s are left out).
# Output: probe_results.csv (every answer) and a summary table:
#   false YES   = share of "not arrived" frames answered YES   (the early-STOP problem; lower is better)
#   true YES    = share of "arrived" frames answered YES        (higher is better)
#   first 3xYES = first time with 3 YES answers in a row        (when a 3-vote rule would stop)

import argparse
import csv
import glob
import json
import os
import re
import time

import ollama

from agrivln.STL_template import split_instruction

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["qwen2.5vl:7b"])
parser.add_argument("--eps", nargs="+", default=["garden_1", "garden_2"])
parser.add_argument("--every", type=int, default=1, help="use every n-th frame (2 = half the calls)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mode", choices=["yesno", "choice"], default="yesno",
                    help="yesno: 'is this true? YES/NO' (v2).  choice: 'A = still need to move forward, B = there now' (v3)")
parser.add_argument("--out", default=None, help="default: probe_results_v2.csv (yesno) / probe_results_v3.csv (choice)")
args = parser.parse_args()
args.out = args.out or ("probe_results_v2.csv" if args.mode == "yesno" else "probe_results_v3.csv")


def tsec(path):
    m = re.search(r"frame_(\d+)'(\d)\.jpg$", path)
    return int(m.group(1)) + int(m.group(2)) / 10


def ask_choice(model, image, condition):
    """v3: forced choice instead of yes/no. Models tend to agree with a yes/no statement once the
    object is clearly visible; here they must choose between 'not yet, keep moving' and 'there now'."""
    q = ("This image comes from the front camera of a small ground robot. The camera is where the robot is.\n"
         f"Target position from the instruction: \"{condition}\"\n"
         "Which is true right now?\n"
         "A) Not yet: the robot would still need to move forward to be in that position "
         "(for example, the object is still ahead of the camera, or only partly matches the description).\n"
         "B) Yes: the robot is in that position now, exactly as described.\n"
         "Answer with one letter: A or B.")
    r = ollama.chat(model=model, messages=[{"role": "user", "content": q, "images": [image]}],
                    options={"temperature": 0, "seed": args.seed})
    text = r["message"]["content"]
    m = re.search(r"\b([AB])\b", text)
    a = {"A": "NO", "B": "YES"}[m.group(1)] if m else "?"
    return a, text.strip().replace("\n", " ")[:200]


def ask(model, image, condition):
    if args.mode == "choice":
        return ask_choice(model, image, condition)
    q = ("This image comes from the front camera of a small ground robot. The camera is where the robot is.\n"
         f"Is the following true in this image right now: \"{condition}\"?\n"
         "Seeing the object from a distance is NOT enough; it must be true exactly as written.\n"
         "Answer with one word: YES or NO.")
    r = ollama.chat(model=model, messages=[{"role": "user", "content": q, "images": [image]}],
                    options={"temperature": 0, "seed": args.seed})
    text = r["message"]["content"]
    m = re.findall(r"\b(yes|no)\b", text.lower())
    return (m[-1].upper() if m else "?"), text.strip().replace("\n", " ")[:200]


rows, summary = [], []
for model in args.models:
    for ep in args.eps:
        info = json.load(open(f"dataset/{ep}/info.json"))
        labels = json.load(open(f"dataset/{ep}/label.json"))
        stop_t = next(l["time_range"][0] for l in labels if l["action"] == "[STOP]")
        _, _, arrival = split_instruction(info["instruction"])
        # v2 bug fix: "the robot" made models answer NO because no robot is visible in the image
        condition = re.sub(r"\byour\b", "the camera's", re.sub(r"\byou\b", "the camera", arrival))
        frames = sorted(glob.glob(f"dataset/{ep}/frames/frame_*.jpg"), key=tsec)[::args.every]
        print(f"\n### {model} / {ep}: {len(frames)} frames, condition: \"{condition}\"")
        t0, answers = time.time(), []
        for f in frames:
            t = tsec(f)
            a, raw = ask(model, f, condition)
            truth = "arrived" if t >= stop_t else ("not arrived" if t < stop_t - 2.0 else "last 2 s")
            answers.append((t, a, truth))
            rows.append({"model": model, "episode": ep, "time": t, "answer": a, "truth": truth, "raw": raw})
            print(f"  {t:5.1f} s  {a:<3}  ({truth})")
        neg = [a for t, a, g in answers if g == "not arrived"]
        pos = [a for t, a, g in answers if g == "arrived"]
        streak, first3 = 0, None
        for t, a, g in answers:
            streak = streak + 1 if a == "YES" else 0
            if streak == 3 and first3 is None:
                first3 = t
        first_yes = next((t for t, a, g in answers if a == "YES"), None)
        summary.append((model, ep, stop_t, neg.count("YES") / max(len(neg), 1), pos.count("YES") / max(len(pos), 1),
                        first_yes, first3, sum(a == "?" for _, a, _ in answers), (time.time() - t0) / 60))

new = not os.path.exists(args.out)
with open(args.out, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["model", "episode", "time", "answer", "truth", "raw"])
    if new:
        w.writeheader()
    w.writerows(rows)

print(f"\n{'model':<16} {'episode':<9} {'real stop':>9} {'false YES':>9} {'true YES':>8} {'first YES':>9} {'first 3xYES':>11} {'unclear':>7} {'min':>5}")
for m, ep, st, fy, ty, f1, f3, unk, mins in summary:
    f = lambda v: "-" if v is None else f"{v:.1f}"
    print(f"{m:<16} {ep:<9} {st:>9.1f} {fy:>9.0%} {ty:>8.0%} {f(f1):>9} {f(f3):>11} {unk:>7} {mins:>5.1f}")
print(f"\nSaved every answer to {args.out}")
