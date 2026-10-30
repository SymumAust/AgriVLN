# AgriVLN - Decision Making Module with a qubit-based subtask state tracker (v3)
#
# Same loop as agrivln/decide.py, plus:
#   - QubitTracker filters the VLM's subtask-state claims before they are saved to
#     log.json (which is what the VLM sees at the next step). Only the subtask the VLM
#     talked about is updated; a claim whose "from" state does not match the kept state
#     is ignored, exactly like the original update_subtask_state().
#   - strict=True : "done" also needs evidence; pending -> done jumps are not accepted.
#   - gate:
#        None        : STOP is executed as the VLM says
#        "all_done"  : STOP only if the tracker says every subtask is done
#        "last_done" : STOP only if the tracker says the FINAL subtask is done
#        "check"     : STOP only if an independent arrival check says YES k times out of k
#        "percept"   : STOP only if the detector sees the target (score >= score_min) AND the
#                      depth model puts it within depth_max metres (needs perception.py first)
#        "last+percept" : both "last_done" and "percept"
# Extra files in runs/<exp>/<ep>/:
#   raw_log.json    the state the original rules would have produced
#   qubit_log.json  P(doing) and P(done) of every subtask at each step
#   gate_log.json   every STOP the VLM wanted, the arrival-check answers, blocked or not

import ollama
import os
import json
import math
import re

from agrivln.for_json import get_stop_start_time, clean_format_stl_state, append_action
from agrivln.decide import (get_system_prompt, get_user_prompt, extract, extract_state,
                            append_stl_state, merge_stl_and_state)
from agrivln.quantum_tracker import QubitTracker, BACKEND
from agrivln.entangled_tracker import EntangledTracker


def parse_claim(state_text):
   """Return (step_number, from_state, to_state) from the VLM's <state> line, or Nones."""
   if not state_text:
      return None, None, None
   number, old, new = extract_state(state_text)
   if number is not None:
      return number, old, new
   m = re.search(r"NO\.(\d+)\s+keeps state of (\w+)", state_text)
   if m:
      return int(m.group(1)), m.group(2), m.group(2)
   return None, None, None


def arrival_check(my_model, instruction, image_path, k=3, temperature=0.7, seed=1000):
   """Ask the VLM k times, with sampling, whether the robot has reached the final target."""
   question = (
      "You are checking an agricultural robot's front camera image.\n"
      f"The robot was told: \"{instruction}\"\n"
      "Question: Has the robot ARRIVED at the final target of this instruction right now? "
      "Arrived means the final target object is clearly visible AND very close, where the "
      "instruction says to stop. If the target is not visible, partly hidden, blurry, far away, "
      "or you only see something that looks similar, answer NO.\n"
      "Answer with one word: YES or NO.")
   answers = []
   for i in range(k):
      r = ollama.chat(model=my_model,
                      messages=[{'role': 'user', 'content': question, 'images': [image_path]}],
                      options={'temperature': temperature, 'seed': seed + i})
      m = re.search(r"\b(yes|no)\b", r['message']['content'].lower())
      answers.append(m.group(1).upper() if m else "?")
   return answers


def decide_quantum(my_model, exp, place, id, if_token, delta=math.pi / 3, strict=False,
                   gate=None, check_k=3, classical=False,
                   score_min=0.35, depth_max=2.0, entangled=False):

   if gate is True:          # backwards compatibility with v2 (--gate)
      gate = "all_done"

   t_a = 0
   t_b = 0
   t_b_interval = 2
   FPS = 5.0
   safe_redundancy = 2

   label_path = f"dataset/{place}_{id}/label.json"
   with open(label_path, "r") as f:
      labels = json.load(f)
   with open(f"dataset/{place}_{id}/info.json") as f:
      instruction = json.load(f)["instruction"]

   percept = {}
   if gate in ("percept", "last+percept"):
      ppath = f"dataset/{place}_{id}/perception.json"
      if not os.path.exists(ppath):
         raise SystemExit(f"[ERROR] {ppath} not found: run perception.py --ep {place}_{id} first.")
      percept = {r["time"]: r for r in json.load(open(ppath))["frames"]}
      print(f"[INFO] Perception gate: score >= {score_min}, distance <= {depth_max} m")

   stop_start_time = get_stop_start_time(labels)
   print("[INFO] Label STOP begins at: ", stop_start_time)
   max_time = float(stop_start_time) + (1 / FPS) * safe_redundancy
   print(f'[INFO] Max time step: {max_time}')
   print(f'[INFO] Qubit tracker: delta = {delta:.3f} rad, strict = {strict}, gate = {gate}, backend = {"classical" if classical else BACKEND}, entangled = {entangled}')

   run_dir = f"runs/{exp}/{place}_{id}"
   STL_path, log_path = f"{run_dir}/STL.json", f"{run_dir}/log.json"
   raw_path, qubit_path = f"{run_dir}/raw_log.json", f"{run_dir}/qubit_log.json"
   gate_path, predict_path = f"{run_dir}/gate_log.json", f"{run_dir}/predict.json"

   with open(log_path) as f:
      initial = json.load(f)
   with open(raw_path, "w") as f:
      json.dump(initial, f)
   for p in (qubit_path, gate_path):
      with open(p, "w") as f:
         json.dump([], f)

   tracker = None
   stop_quantity = 0

   while t_a < max_time:

      t = f'{t_a}\'{t_b}'

      STL = merge_stl_and_state(STL_path, log_path, current_time=t)
      if tracker is None:
         tracker = (EntangledTracker(len(STL), delta, classical) if entangled
                    else QubitTracker(len(STL), delta, strict, classical))
      steps = [s["step"] for s in STL]

      image_path = f"dataset/{place}_{id}/frames/frame_{t}.jpg"
      messages = [
         {'role': 'system', 'content': get_system_prompt()},
         {'role': 'user', 'content': get_user_prompt(STL), 'images': [image_path]}
      ]
      response = ollama.chat(model=my_model, messages=messages)
      message = response['message']['content']

      if if_token == 'True':
         token_path = f"{run_dir}/token.json"
         data = json.load(open(token_path)) if os.path.exists(token_path) else []
         data.append({"place": place, "time": t,
                      "token_prompt": response.prompt_eval_count,
                      "token_completion": response.eval_count})
         with open(token_path, "w") as token_f:
            json.dump(data, token_f, indent=4)

      result = extract(message)
      action, thought, state = result['action'], result['thought'], result['state']

      # 1) the VLM's claim about ONE subtask
      number, old_state, new_state = parse_claim(state)
      if state is None:
         print('[WARNING] State = None.')
      index = steps.index(number) if number in steps else None
      before = [s["state"] for s in STL]
      raw = list(before)
      if index is not None and before[index] == old_state:      # same rule as the original code
         raw[index] = new_state
         said = new_state
      else:
         if index is not None:
            print(f'[WARNING] Claim ignored: subtask {number} is "{before[index]}", VLM said "{old_state}".')
         index, said = None, None

      # 2) the qubit tracker decides what is kept
      kept, p_doing, p_done = tracker.update(index, said)
      if kept != raw:
         print(f'[QUBIT] original rule -> {raw}, kept {kept}  P(doing)={p_doing}  P(done)={p_done}')

      # 3) STOP gate
      executed = action
      if action == '[STOP]' and gate:
         entry = {"time": t, "wanted": action}
         if gate == "all_done":
            allowed = tracker.all_done()
         elif gate == "last_done":
            allowed = kept[-1] == "done"
         elif gate in ("percept", "last+percept"):
            r = percept.get(t, {})
            seen = r.get("score", 0.0) >= score_min
            near = r.get("box_depth") is not None and r["box_depth"] <= depth_max
            allowed = seen and near and (gate == "percept" or kept[-1] == "done")
            entry["percept"] = {"score": r.get("score"), "box_depth": r.get("box_depth"), "last_done": kept[-1] == "done"}
            print(f'[PERCEPT] {t}: target score {r.get("score")}, distance {r.get("box_depth")} m, '
                  f'final subtask {kept[-1]} -> {"allow" if allowed else "block"}')
         else:
            answers = arrival_check(my_model, instruction, image_path, k=check_k)
            allowed = answers.count("YES") == check_k
            entry["answers"] = answers
            print(f'[CHECK] {t}: arrived? {answers}')
         if not allowed:
            executed = '[FORWARD]'
            print(f'[GATE] STOP blocked at {t}')
         entry.update(executed=executed, blocked=not allowed, kept=kept)
         append_stl_state(gate_path, entry)
      elif action == '[STOP]':
         append_stl_state(gate_path, {"time": t, "wanted": action, "executed": action,
                                      "blocked": False, "kept": kept})

      append_action(predict_path, t, executed, thought, state)
      print(f'{place}_{id}, {t_a}.{t_b}, {executed}' + ('  (VLM wanted [STOP])' if executed != action else ''))

      t_b += t_b_interval
      if t_b == 10:
         t_b = 0
         t_a += 1
      next_t = f"{t_a}'{t_b}"

      append_stl_state(raw_path, {"time": next_t, "subtask_list":
                                  [{"step": s, "state": o} for s, o in zip(steps, raw)]})
      append_stl_state(log_path, {"time": next_t, "subtask_list":
                                  [{"step": s, "state": k} for s, k in zip(steps, kept)]})
      append_stl_state(qubit_path, {"time": next_t, "p_doing": p_doing, "p_done": p_done})
      clean_format_stl_state(log_path)
      clean_format_stl_state(raw_path)

      if executed == '[STOP]':
         stop_quantity += 1
      if stop_quantity >= 3:
         break
