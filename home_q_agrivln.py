# Q-AgriVLN - AgriVLN with a qubit-based subtask state tracker
#
# Example:
#   python home_q_agrivln.py --place garden --id_range 1 2 --reuse_stl
# Results go to runs/Q-AgriVLN/... (the original runs/AgriVLN/... is never touched).

import argparse
import math
import os
import shutil
import sys
import time

from home_agrivln import check_label_format
from agrivln.STL import STL, generate_initial_stl_state
from agrivln.decide_quantum import decide_quantum
from agrivln.evaluate import evaluate


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--place", type=str, required=True)
   parser.add_argument("--id_range", type=int, nargs='+', required=True)
   parser.add_argument("-t", "--if_token", type=str, default='False')
   parser.add_argument("--delta_div", type=float, default=3.0,
                       help="rotation per step = pi / delta_div (1 = original behaviour, 3 = pi/3)")
   parser.add_argument("--strict", action="store_true", help="'done' also needs evidence")
   parser.add_argument("--gate", choices=["all_done", "last_done", "check"], default=None, help="STOP gate")
   parser.add_argument("--reuse_stl", action="store_true",
                       help="use the subtask list from runs/AgriVLN/<ep> so only the tracker differs")
   args = parser.parse_args()

   ids = args.id_range
   if len(ids) == 2:
      ids = list(range(ids[0], ids[1] + 1))

   exp = 'Q-AgriVLN'
   LLM = 'qwen2.5vl:7b'
   VLM = 'qwen2.5vl:7b'
   delta = math.pi / args.delta_div

   print(f'[INFO] Experiment: {exp}   LLM: {LLM}   VLM: {VLM}   delta: pi/{args.delta_div:g}')

   for id in ids:
      ep = f"{args.place}_{id}"
      dir_path = f"runs/{exp}/{ep}"
      shutil.rmtree(dir_path, ignore_errors=True)        # start each run clean
      os.makedirs(dir_path, exist_ok=True)

      if not check_label_format(f"dataset/{ep}/label.json"):
         print(f'[ERROR] {ep} label format is wrong.')
         continue
      print(f'--- {ep} starts ---')

      base_stl = f"runs/AgriVLN/{ep}/STL.json"
      if args.reuse_stl and os.path.exists(base_stl):
         shutil.copy(base_stl, f"{dir_path}/STL.json")
         generate_initial_stl_state(f"{dir_path}/STL.json", f"{dir_path}/log.json")
         print(f'[INFO] Reused subtask list from {base_stl}')
      else:
         STL(LLM, exp, args.place, id)
      time.sleep(0.1)

      decide_quantum(VLM, exp, args.place, id, args.if_token, delta=delta, strict=args.strict, gate=args.gate)
      time.sleep(0.1)

      evaluate(exp, args.place, id)
      print(f'--- {ep} ends ---')
