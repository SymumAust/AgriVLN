# OFFLINE replay: feed the VLM claims recorded in a finished run through different trackers
# and see when the final-subtask gate would open.
#
#   python replay_trackers.py runs/X4-q6last-T0-r1/garden_1 runs/X4-q6last-T0-r1/garden_2
#
# Caution: in a real run the kept states are shown to the VLM at the next step, so a different
# tracker would change the VLM's later answers. The replay is exact only until the first step where
# the new tracker keeps a different state from the recorded run ("diverges at").

import json
import math
import sys

from agrivln.decide_quantum import parse_claim
from agrivln.entangled_tracker import EntangledTracker
from agrivln.quantum_tracker import QubitTracker


def tsec(t):
    a, b = t.split("'")
    return int(a) + int(b) / 10


def replay(run_dir, make):
    pred = json.load(open(f"{run_dir}/predict.json"))
    recorded = json.load(open(f"{run_dir}/log.json"))
    rec_states = {e["time"]: [s["state"] for s in e["subtask_list"]] for e in recorded}
    stl = json.load(open(f"{run_dir}/STL.json"))
    steps = [s["step"] for s in stl]
    tr = make(len(stl))
    kept = ["pending"] * len(stl)
    diverge, stop_at, requests = None, None, []
    for p in pred:
        t = p["time"]
        if diverge is None and rec_states.get(t) and rec_states[t] != kept:
            diverge = t
        number, old, new = parse_claim(p.get("state"))
        index = steps.index(number) if number in steps else None
        said = new if index is not None and kept[index] == old else None
        if said is None:
            index = None
        kept, ps, pd = tr.update(index, said)
        if t in STOP_WANTED[run_dir]:
            requests.append(t)
            if stop_at is None and kept[-1] == "done":
                stop_at = t
    return stop_at, diverge, requests, ps, pd


def ne(run_dir, t):
    if t is None:
        return "no stop"
    ev = json.load(open(f"{run_dir}/evaluate.json"))
    L, T = ev["path_length"], ev["label_stop_time"]
    return f"{max(0.0, (T - tsec(t)) / T * L):.2f} m"


STOP_WANTED = {}
if __name__ == "__main__":
    d = math.pi / 6
    trackers = {
        "independent qubits (as run)": lambda n: QubitTracker(n, d, strict=True),
        "entangled qubits (order)": lambda n: EntangledTracker(n, d),
        "classical chain (control)": lambda n: EntangledTracker(n, d, classical=True),
    }
    for run_dir in sys.argv[1:]:
        STOP_WANTED[run_dir] = [g["time"] for g in json.load(open(f"{run_dir}/gate_log.json"))]
        print(f"\n=== {run_dir}  (VLM STOP requests: {', '.join(STOP_WANTED[run_dir])})")
        for name, make in trackers.items():
            stop_at, diverge, req, ps, pd = replay(run_dir, make)
            print(f"  {name:<30} STOP at {stop_at or '-':>5}  NE {ne(run_dir, stop_at):>8}   "
                  f"kept states differ from the recorded run from: {diverge or 'never'}")
            print(f"  {'':<30} final beliefs  P(started)={ps}  P(done)={pd}")
