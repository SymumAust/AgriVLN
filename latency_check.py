# How long does one arrival answer really take on this PC?  (input for closed_loop_sim.py)
#
#   python latency_check.py                      # 20 frames of garden_4 and farm_6, three models
#   python latency_check.py --models qwen3-vl:4b --n 40
#
# Same question as probe_arrival.py, temperature 0, seed 42. The first call per model loads it onto the GPU
# and is reported separately (warm-up). Prints median and 90th-percentile seconds per answer.

import argparse
import json
import re
import statistics
import time

import ollama

from agrivln.STL_template import split_instruction

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["qwen3-vl:4b", "qwen2.5vl:7b", "qwen3-vl:4b-instruct"])
p.add_argument("--eps", nargs="+", default=["garden_4", "farm_6"])
p.add_argument("--n", type=int, default=20, help="frames per episode, spread over the clip")
a = p.parse_args()


def question(ep):
    arrival = split_instruction(json.load(open(f"dataset/{ep}/info.json"))["instruction"])[2]
    cond = re.sub(r"\byour\b", "the camera's", re.sub(r"\byou\b", "the camera", arrival))
    return ("This image comes from the front camera of a small ground robot. The camera is where the robot is.\n"
            f"Is the following true in this image right now: \"{cond}\"?\n"
            "Seeing the object from a distance is NOT enough; it must be true exactly as written.\n"
            "Answer with one word: YES or NO.")


rows = []
for model in a.models:
    secs, warm = [], None
    for ep in a.eps:
        lab = json.load(open(f"dataset/{ep}/label.json"))
        end = lab[-1]["time_range"][1] - 0.2
        q = question(ep)
        for i in range(a.n):
            t = round(round(end * i / max(a.n - 1, 1) / 0.2) * 0.2, 1)
            img = f"dataset/{ep}/frames/frame_{int(t)}'{int(round((t - int(t)) * 10))}.jpg"
            t0 = time.time()
            ollama.chat(model=model, messages=[{"role": "user", "content": q, "images": [img]}],
                        options={"temperature": 0, "seed": 42})
            dt = time.time() - t0
            if warm is None:
                warm = dt
            else:
                secs.append(dt)
    secs.sort()
    p90 = secs[int(0.9 * (len(secs) - 1))]
    rows.append((model, warm, statistics.median(secs), p90, max(secs)))
    print(f"{model}: warm-up {warm:.1f} s, median {statistics.median(secs):.2f} s, 90% {p90:.2f} s, max {max(secs):.1f} s")

print(f"\n{'model':<24}{'median s':>10}{'90% s':>10}{'max s':>10}")
for m, w, med, p90, mx in rows:
    print(f"{m:<24}{med:>10.2f}{p90:>10.2f}{mx:>10.1f}")
print("\nUse the median (and the 90% value) as --latency in closed_loop_sim.py.")
