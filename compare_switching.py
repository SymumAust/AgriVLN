# Summarise AgriVLN / Q-AgriVLN runs
#
#   python compare_switching.py garden_1 garden_2        # original vs runs/Q-AgriVLN
#   python compare_switching.py --all                    # every runs/X-* experiment
#
# switches = pending<->doing changes before the first STOP the VLM wanted (as on slide 25).

import glob
import json
import os
import sys


def _states(path):
   return [[s["state"] for s in e["subtask_list"]] for e in json.load(open(path))]


def _switches(seq, n):
   sw = back = 0
   for t in range(1, min(n, len(seq) - 1) + 1):
      for a, b in zip(seq[t - 1], seq[t]):
         if {a, b} == {"pending", "doing"}:
            sw += 1
            back += a == "doing"
   return sw, back


def _t(s):
   a, b = s.split("'")
   return int(a) + int(b) / 10


def summarise(run_dir):
   pred = json.load(open(f"{run_dir}/predict.json"))
   gate_path = f"{run_dir}/gate_log.json"
   gates = json.load(open(gate_path)) if os.path.exists(gate_path) else []
   wanted = [g["time"] for g in gates] or [p["time"] for p in pred if p["action"] == "[STOP]"]
   executed = [p["time"] for p in pred if p["action"] == "[STOP]"]
   first_wanted = wanted[0] if wanted else None
   # number of log transitions before the first wanted STOP
   n = ([p["time"] for p in pred].index(first_wanted) if first_wanted else len(pred)) - 1
   kept = _switches(_states(f"{run_dir}/log.json"), n)
   raw_path = f"{run_dir}/raw_log.json"
   proposed = _switches(_states(raw_path), n) if os.path.exists(raw_path) else kept
   ev = json.load(open(f"{run_dir}/evaluate.json")) if os.path.exists(f"{run_dir}/evaluate.json") else {}
   return {
      "run": run_dir,
      "switches_kept": kept[0], "backwards_kept": kept[1], "switches_proposed": proposed[0],
      "first_stop_wanted": _t(first_wanted) if first_wanted else None,
      "first_stop_executed": _t(executed[0]) if executed else None,
      "stops_blocked": sum(1 for g in gates if g.get("blocked")),
      "percept": [f"{g['time']}:{g['percept']['score']}/{g['percept']['box_depth']}m" for g in gates if "percept" in g],
      "check_yes": [f"{g['time']}:{''.join(a[0] for a in g['answers'])}" for g in gates if "answers" in g],
      "type": ev.get("type"), "SR": ev.get("SR"), "NE": ev.get("NE"),
      "label_stop": ev.get("label_stop_time"),
   }


def print_table(rows):
   print(f"{'run':<44} {'sw kept':>7} {'back':>5} {'sw prop':>7} {'STOP want':>9} {'STOP done':>9} "
         f"{'blocked':>7} {'type':>11} {'SR':>3} {'NE':>7}")
   for r in rows:
      f = lambda v: "-" if v is None else v
      print(f"{r['run']:<44} {r['switches_kept']:>7} {r['backwards_kept']:>5} {r['switches_proposed']:>7} "
            f"{f(r['first_stop_wanted']):>9} {f(r['first_stop_executed']):>9} {r['stops_blocked']:>7} "
            f"{f(r['type']):>11} {f(r['SR']):>3} {f(r['NE']):>7}")
      if r.get("percept"):
         print(f"      perception at STOP requests (score/distance): {'  '.join(r['percept'])}")
      if r.get("check_yes"):
         print(f"      arrival check (Y/N per call): {'  '.join(r['check_yes'])}")
   print("\nNote: AgriVLN's evaluate.py only looks for STOP from 2.0 s on, and scores 'no_stop' as NE = 0;"
         "\n      read 'STOP done' and 'type' together with NE.")


if __name__ == "__main__":
   if "--all" in sys.argv:
      dirs = sorted(d for d in glob.glob("runs/X*-*/*") if os.path.exists(f"{d}/predict.json"))
   else:
      eps = [a for a in sys.argv[1:]] or ["garden_1", "garden_2"]
      dirs = [d for ep in eps for d in (f"runs/AgriVLN/{ep}", f"runs/Q-AgriVLN/{ep}") if os.path.exists(f"{d}/predict.json")]
   print_table([summarise(d) for d in dirs])
