# Generate the "approach, then confirm" subtask lists ONCE, look at them, then they stay frozen.
#
#   python make_stl_v2.py --place garden --ids 1 2            # generate (or just show + check the frozen ones)
#   python make_stl_v2.py --place garden --ids 1 2 --force    # regenerate; the old list is kept as STL_prev.json
#
# Writes runs/STL2/<place>_<id>/STL.json (+ STL2_raw.txt with the model's reasoning).
# run_experiment.py --stl v2 then copies these files, so every condition uses the same list.

import argparse
import json
import os
import shutil

from agrivln.STL_v2 import check, generate_stl_v2

parser = argparse.ArgumentParser()
parser.add_argument("--place", default="garden")
parser.add_argument("--ids", type=int, nargs="+", default=[1, 2])
parser.add_argument("--model", default="qwen2.5vl:7b")
parser.add_argument("--force", action="store_true", help="overwrite an existing list")
args = parser.parse_args()

for id in args.ids:
    ep = f"{args.place}_{id}"
    out_dir = f"runs/STL2/{ep}"
    out = f"{out_dir}/STL.json"
    instruction = json.load(open(f"dataset/{ep}/info.json"))["instruction"]
    if os.path.exists(out) and not args.force:
        print(f"[INFO] {out} exists (frozen). Use --force to regenerate.")
    else:
        if os.path.exists(out):
            shutil.copy(out, f"{out_dir}/STL_prev.json")
            print(f"[INFO] previous list kept as {out_dir}/STL_prev.json")
        stl, raw = generate_stl_v2(args.model, instruction)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(stl, open(out, "w"), indent=3, ensure_ascii=False)
        open(f"{out_dir}/STL2_raw.txt", "w").write(raw)
    stl = json.load(open(out))
    print(f"\n===== {ep} =====\ninstruction: {instruction}")
    for s in stl:
        print(f"{s['step']}. {s['subtask']}\n   start: {s['start_condition']}\n   end:   {s['end_condition']}")
    problems = check(stl, instruction)
    print("check:", "OK" if not problems else "\n  - " + "\n  - ".join(problems))
