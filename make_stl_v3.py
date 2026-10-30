# Build the TEMPLATE subtask lists (agrivln/STL_template.py) once; they stay frozen.
#
#   python make_stl_v3.py --place garden --ids 1 2
#
# Writes runs/STL3/<place>_<id>/STL.json and parts.json (movement / target / arrival found).
# run_experiment.py --stl v3 copies these, so every condition uses the same list.

import argparse
import json
import os

from agrivln.STL_template import build_stl
from agrivln.STL_v2 import check

parser = argparse.ArgumentParser()
parser.add_argument("--place", default="garden")
parser.add_argument("--ids", type=int, nargs="+", default=[1, 2])
parser.add_argument("--model", default="qwen2.5vl:7b", help="only used if the text rules fail")
args = parser.parse_args()

for id in args.ids:
    ep = f"{args.place}_{id}"
    out_dir = f"runs/STL3/{ep}"
    os.makedirs(out_dir, exist_ok=True)
    instruction = json.load(open(f"dataset/{ep}/info.json"))["instruction"]
    stl, parts = build_stl(instruction, args.model)
    json.dump(stl, open(f"{out_dir}/STL.json", "w"), indent=3, ensure_ascii=False)
    json.dump(parts, open(f"{out_dir}/parts.json", "w"), indent=3, ensure_ascii=False)
    print(f"\n===== {ep} =====\ninstruction: {instruction}")
    print(f"found by {parts['source']}:  target = '{parts['target']}'")
    for s in stl:
        print(f"{s['step']}. {s['subtask']}\n   start: {s['start_condition']}\n   end:   {s['end_condition']}")
    problems = check(stl, instruction)
    print("check:", "OK" if not problems else "\n  - " + "\n  - ".join(problems))
