# Why does qwen3-vl:4b sometimes return an empty answer?
# Re-asks the exact frames that came back blank in probe_results_v2.csv (same question, temperature 0, seed 42)
# and prints everything Ollama returns, with thinking on (default) and off.
#
#   python blank_check.py                 # up to 8 blank frames from the garden runs
#   python blank_check.py --n 19          # all of them

import argparse
import csv
import json
import re

import ollama

from agrivln.STL_template import split_instruction

p = argparse.ArgumentParser()
p.add_argument("--model", default="qwen3-vl:4b")
p.add_argument("--csv", default="probe_results_v2.csv")
p.add_argument("--n", type=int, default=8)
a = p.parse_args()

blank = [(r["episode"], float(r["time"])) for r in csv.DictReader(open(a.csv))
         if r["model"] == a.model and r["answer"] == "?"][:a.n]
print(f"{len(blank)} blank frames to re-ask\n")


def question(ep):
    info = json.load(open(f"dataset/{ep}/info.json"))
    arrival = split_instruction(info["instruction"])[2]
    cond = re.sub(r"\byour\b", "the camera's", re.sub(r"\byou\b", "the camera", arrival))
    return ("This image comes from the front camera of a small ground robot. The camera is where the robot is.\n"
            f"Is the following true in this image right now: \"{cond}\"?\n"
            "Seeing the object from a distance is NOT enough; it must be true exactly as written.\n"
            "Answer with one word: YES or NO.")


def show(tag, r):
    m = r["message"]
    content = getattr(m, "content", None) or (m["content"] if isinstance(m, dict) else "") or ""
    thinking = getattr(m, "thinking", None) or (m.get("thinking") if isinstance(m, dict) else "") or ""
    print(f"   {tag:<12} content={content.strip()[:60]!r:<30} thinking={len(thinking):>5} chars  "
          f"done_reason={getattr(r, 'done_reason', None)}  tokens={getattr(r, 'eval_count', None)}")
    if thinking:
        print(f"   {'':<12} thinking ends: ...{thinking.strip()[-120:]!r}")


for ep, t in blank:
    img = f"dataset/{ep}/frames/frame_{int(t)}'{int(round((t - int(t)) * 10))}.jpg"
    q = question(ep)
    print(f"{ep} {t:.1f} s")
    msgs = [{"role": "user", "content": q, "images": [img]}]
    show("default", ollama.chat(model=a.model, messages=msgs, options={"temperature": 0, "seed": 42}))
    try:
        show("think=False", ollama.chat(model=a.model, messages=msgs, think=False, options={"temperature": 0, "seed": 42}))
    except TypeError:
        print("   think=False not supported by this ollama package: run  pip install -U 'ollama>=0.5'")
    except Exception as e:
        print(f"   think=False error: {e}")
