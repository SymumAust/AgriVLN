# Fair comparison: original AgriVLN vs qubit tracker variants
#
#   python run_experiment.py --place garden --ids 1 2                 # all conditions, temperature 0
#   python run_experiment.py --place garden --ids 2 --conditions base q6gate
#   python run_experiment.py --place garden --ids 1 2 --temperature 0.7 --repeats 3
#   python run_experiment.py --place garden --ids 1 2 --conditions base q6last c6last --stl v2
#        (--stl v2: use the frozen "approach, then confirm" lists from make_stl_v2.py)
#
# Every condition uses the SAME subtask list (copied from runs/AgriVLN/<ep>/STL.json)
# and the SAME model settings (temperature, seed), so only the decision logic differs.
# Results: runs/X4-<condition>-T<temp>-r<repeat>/<ep>/   and   results.csv

import argparse
import csv
import json
import math
import os
import re
import shutil
import time

import ollama

CONDITIONS = {
   #  name        (description,                                       delta,         strict, gate,        classical)
   "base":     ("original AgriVLN (no tracker)",                     None,          False,  None,        False),
   "q3":       ("tracker pi/3",                                      math.pi / 3,   False,  None,        False),
   "q6":       ("tracker pi/6",                                      math.pi / 6,   False,  None,        False),
   "q3gate":   ("tracker pi/3 + strict done + all-done gate",        math.pi / 3,   True,   "all_done",  False),
   "q6gate":   ("tracker pi/6 + strict done + all-done gate",        math.pi / 6,   True,   "all_done",  False),
   "q3last":   ("tracker pi/3 + strict done + last-subtask gate",    math.pi / 3,   True,   "last_done", False),
   "q6last":   ("tracker pi/6 + strict done + last-subtask gate",    math.pi / 6,   True,   "last_done", False),
   "q8last":   ("tracker pi/8 + strict done + last-subtask gate",    math.pi / 8,   True,   "last_done", False),
   "q12last":  ("tracker pi/12 + strict done + last-subtask gate",   math.pi / 12,  True,   "last_done", False),
   "c6last":   ("CLASSICAL counter (same as q6last, no qubit)",      math.pi / 6,   True,   "last_done", True),
   "e6last":   ("ENTANGLED qubits pi/6 (order) + last-subtask gate", math.pi / 6,   True,   "last_done", False),
   "ec6last":  ("CLASSICAL chain (same as e6last, no circuit)",      math.pi / 6,   True,   "last_done", True),
   "depth":    ("original states + perception gate (detector + depth)", math.pi,   False,  "percept",   False),
   "q6depth":  ("tracker pi/6 last-subtask gate + perception gate",  math.pi / 6,   True,   "last+percept", False),
   "check":    ("original states + arrival check (k/k YES)",         math.pi,       False,  "check",     False),
   "q6check":  ("tracker pi/6 + arrival check (k/k YES)",            math.pi / 6,   False,  "check",     False),
}
ENTANGLED = {"e6last", "ec6last"}

parser = argparse.ArgumentParser()
parser.add_argument("--place", default="garden")
parser.add_argument("--ids", type=int, nargs="+", default=[1, 2])
parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--model", default="qwen2.5vl:7b")
parser.add_argument("--score_min", type=float, default=0.35, help="perception gate: detector score")
parser.add_argument("--depth_max", type=float, default=2.0, help="perception gate: max distance (m)")
parser.add_argument("--stl", choices=["orig", "v2", "v3"], default="orig",
                    help="orig: runs/AgriVLN/<ep>/STL.json   v2: runs/STL2 (make_stl_v2.py)   v3: runs/STL3 (make_stl_v3.py, template)")
parser.add_argument("--check_k", type=int, default=3, help="arrival check: YES needed out of k")
args = parser.parse_args()

# --- same sampling settings for every call (original code and quantum code alike) ---
_original_chat = ollama.chat
SAMPLING = {"temperature": args.temperature, "seed": args.seed}


def _chat(*a, **k):
   opts = dict(SAMPLING)
   opts.update(k.pop("options", {}) or {})   # explicit options (arrival check) win
   return _original_chat(*a, options=opts, **k)


ollama.chat = _chat

from home_agrivln import check_label_format          # noqa: E402  (after the patch)
from agrivln.STL import generate_initial_stl_state   # noqa: E402
from agrivln.decide import decide                    # noqa: E402
from agrivln.decide_quantum import decide_quantum    # noqa: E402
from agrivln.evaluate import evaluate                # noqa: E402
from compare_switching import summarise              # noqa: E402

rows = []
for r in range(args.repeats):
   SAMPLING["seed"] = args.seed + r
   for cond in args.conditions:
      desc, delta, strict, gate, classical = CONDITIONS[cond]
      exp = {"orig": f"X4-{cond}", "v2": f"X5-{cond}-STL2", "v3": f"X6-{cond}-STL3"}[args.stl] + f"-T{args.temperature:g}-r{r + 1}"
      if args.model != "qwen2.5vl:7b":                 # other model: own folder, nothing overwritten
         exp += "-" + re.sub(r"[^A-Za-z0-9.]+", "_", args.model)
      for id in args.ids:
         ep = f"{args.place}_{id}"
         run_dir = f"runs/{exp}/{ep}"
         base_stl = {"orig": f"runs/AgriVLN/{ep}/STL.json", "v2": f"runs/STL2/{ep}/STL.json", "v3": f"runs/STL3/{ep}/STL.json"}[args.stl]
         if not os.path.exists(base_stl):
            print(f"[ERROR] {base_stl} not found: run " + {"orig": "home_agrivln.py", "v2": "make_stl_v2.py", "v3": "make_stl_v3.py"}[args.stl] + " first.")
            continue
         if not check_label_format(f"dataset/{ep}/label.json"):
            continue
         shutil.rmtree(run_dir, ignore_errors=True)
         os.makedirs(run_dir)
         shutil.copy(base_stl, f"{run_dir}/STL.json")
         generate_initial_stl_state(f"{run_dir}/STL.json", f"{run_dir}/log.json")

         print(f"\n######## {exp} / {ep}: {desc}  (temperature {args.temperature}, seed {SAMPLING['seed']})")
         t0 = time.time()
         if delta is None:
            decide(args.model, exp, args.place, id, 'False')
         else:
            decide_quantum(args.model, exp, args.place, id, 'False', delta=delta, strict=strict, gate=gate, check_k=args.check_k, classical=classical,
                           score_min=args.score_min, depth_max=args.depth_max, entangled=cond in ENTANGLED)
         evaluate(exp, args.place, id)
         row = summarise(run_dir)
         row.update(condition=cond if args.stl == "orig" else f"{cond}-{args.stl.replace('v', 'STL')}", description=desc, episode=ep, repeat=r + 1, model=args.model,
                    temperature=args.temperature, seed=SAMPLING["seed"], minutes=round((time.time() - t0) / 60, 1))
         rows.append(row)

if rows:
   keys = ["condition", "model", "description", "episode", "repeat", "temperature", "seed", "switches_kept",
           "backwards_kept", "switches_proposed", "first_stop_wanted", "first_stop_executed",
           "stops_blocked", "type", "SR", "NE", "minutes"]
   out_csv = "results.csv"
   if os.path.exists(out_csv) and "model" not in open(out_csv).readline():
      out_csv = "results_models.csv"            # old file has no model column: keep it untouched
   new = not os.path.exists(out_csv)
   with open(out_csv, "a", newline="") as f:
      w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
      if new:
         w.writeheader()
      w.writerows(rows)
   print(f"\nSaved to {out_csv}. Summary:\n")
   from compare_switching import print_table
   print_table(rows)
