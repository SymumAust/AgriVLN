# Timing-aware replay: what happens to the stop decision when each answer takes as long as it really takes?
#
#   python closed_loop_sim.py                                  # default sweep, qwen3-vl:4b, garden + farm_6..8
#   python closed_loop_sim.py --latency 0.2 1 2 4 6            # seconds per answer
#   python closed_loop_sim.py --model qwen2.5vl:7b --latency 1.1
#
# Simulation (one judge, asking again as soon as the previous answer arrives):
#   - the robot moves along the recorded path; "video time" tau advances at the robot's speed factor
#     (1 = recorded speed, 0.25 = a quarter of it). Its position at tau comes from GPS (farm) or from
#     path length x tau / labelled stop time (garden, constant speed).
#   - an answer asked at tau is the saved answer for the frame at tau; it arrives L seconds later,
#     during which the robot keeps moving. Decisions only use answers that have arrived.
#   - stop error = position when the STOP is executed - position of the labelled stop (signed, metres).
#     If the robot passes the end of the clip without stopping: failure (overshoot >= what is shown).
#
# Policies (all use the same saved YES/NO answers; nothing new is asked):
#   vote10   : stop at 10 YES in a row (current rule), full speed
#   win2s    : stop when every answer received in the last 2 s of travel is YES (at least 2 answers), full speed
#   slow10   : after 3 YES in a row drive at --slow x speed; stop at 10 YES in a row
#   slow2    : after 3 YES in a row drive at --slow x speed; stop at 2 more YES in a row (5 in total)
#   step10 / step4 : MOVE-STOP-ASK. The robot drives a step, stands still while the model answers, then decides.
#                    Steps of --far m; after the first YES, steps of --near m; stop at 10 (or 4) YES in a row.
#                    The error no longer depends on how slow the model is, only the time to arrive does.
# Exploratory: choose a policy here on the data we have, then test it on new, pre-registered episodes.

import argparse
import csv
import glob
import json
import math
import re

p = argparse.ArgumentParser()
p.add_argument("--model", default="qwen3-vl:4b")
p.add_argument("--csv", nargs="+", default=["probe_results_v2.csv", "probe_results_farm2.csv"])
p.add_argument("--eps", nargs="+", default=[f"garden_{i}" for i in range(1, 13)] + ["farm_6", "farm_7", "farm_8"])
p.add_argument("--latency", type=float, nargs="+", default=[0.2, 0.7, 1.1, 2.5, 4.0, 6.0])
p.add_argument("--slow", type=float, default=0.25)
p.add_argument("--policies", nargs="+", default=["vote10", "win2s", "slow2", "step10", "step4"])
p.add_argument("--far", type=float, default=1.0, help="move-stop-ask: step length before the first YES (m)")
p.add_argument("--near", type=float, default=0.25, help="move-stop-ask: step length after the first YES (m)")
a = p.parse_args()

ans = {}
for path in a.csv:
    try:
        for r in csv.DictReader(open(path)):
            if r["model"] == a.model:
                ans.setdefault(r["episode"], {})[round(float(r["time"]), 1)] = r["answer"] == "YES"
    except FileNotFoundError:
        pass


def episode(ep):
    lab = json.load(open(f"dataset/{ep}/label.json"))
    stop = next(l["time_range"][0] for l in lab if l["action"] == "[STOP]")
    frames = glob.glob(f"dataset/{ep}/frames/frame_*.jpg")
    end = max(int(m.group(1)) + int(m.group(2)) / 10 for m in (re.search(r"frame_(\d+)'(\d)\.jpg$", f) for f in frames))
    try:
        g = sorted((x["t"], x["dist_m"]) for x in json.load(open(f"dataset/{ep}/gps.json")))

        def pos(tau):
            for (t0, d0), (t1, d1) in zip(g, g[1:]):
                if tau <= t1:
                    return d0 + (d1 - d0) * (tau - t0) / (t1 - t0) if t1 > t0 else d0
            return g[-1][1]
    except FileNotFoundError:
        L = json.load(open(f"dataset/{ep}/info.json"))["length"]

        def pos(tau):
            return L * tau / stop
    return stop, end, pos


def simulate(ep, L, policy, dt=0.05):
    stop, end, pos = episode(ep)
    a_ep = ans[ep]
    times = sorted(a_ep)
    frame = lambda tau: a_ep[min(times, key=lambda t: abs(t - tau))]
    if policy.startswith("step"):
        k = int(policy[4:])
        tau, t, run, step = 0.0, 0.0, 0, a.far
        while tau <= end:
            yes = frame(tau)                     # robot stands still while the model answers
            t += L
            run = run + 1 if yes else 0
            if run >= k:
                return pos(tau) - pos(stop), t
            if yes:
                step = a.near
            target = pos(tau) + step             # drive the next step at recorded speed
            while tau <= end and pos(tau) < target:
                tau += dt
                t += dt
        return None, t
    t, tau, speed = 0.0, 0.0, 1.0
    pending = (t + L, frame(tau), tau)          # (arrival time, answer, travel time it refers to)
    got = []                                    # list of (tau_at_arrival, answer)
    run = 0
    slow_mode = False
    while tau <= end:
        t += dt
        tau += speed * dt
        if t >= pending[0] - 1e-9:
            yes = pending[1]
            got.append((tau, yes))
            run = run + 1 if yes else 0
            if policy == "vote10" and run >= 10:
                break
            if policy == "win2s":
                recent = [y for tt, y in got if tt >= tau - 2.0]
                if len(recent) >= 2 and all(recent):
                    break
            if policy in ("slow10", "slow2"):
                if run >= 3 and not slow_mode:
                    slow_mode, speed = True, a.slow
                if slow_mode and run >= (10 if policy == "slow10" else 5):
                    break
            pending = (t + L, frame(tau), tau)
    else:
        return None, t
    return pos(tau) - pos(stop), t


print(f"model {a.model}; answers per video from {', '.join(a.csv)}; slow factor {a.slow}")
print("cells: successes (|error| < 2 m) / videos   [mean error of the stops, m; + late, - early]\n")
print(f"{'latency s/answer':<17}" + "".join(f"{pol:>22}" for pol in a.policies))
eps = [e for e in a.eps if e in ans]
for L in a.latency:
    cells, times_used = [], {}
    for pol in a.policies:
        ok, errs, nostop = 0, [], 0
        for ep in eps:
            err, _ = simulate(ep, L, pol)
            if err is None:
                nostop += 1
            else:
                errs.append(err)
                ok += abs(err) < 2
        mean = f"{sum(errs) / len(errs):+.1f}" if errs else "-"
        cells.append(f"{ok}/{len(eps)} [{mean}]" + (f" ns{nostop}" if nostop else ""))
        times_used.setdefault(pol, []).extend(simulate(ep, L, pol)[1] for ep in eps)
    print(f"{L:<17.1f}" + "".join(f"{c:>22}" for c in cells))
    print(f"{'  time per video':<17}" + "".join(f"{sum(v) / len(v):>21.0f}s" for v in times_used.values()))
print("\nns = no stop before the end of the clip (passed the target). Per-video detail: --eps farm_6 --latency 4")
if len(eps) <= 3:
    for ep in eps:
        for L in a.latency:
            print(ep, L, {pol: (lambda r: (None if r[0] is None else round(r[0], 2), round(r[1], 1)))(simulate(ep, L, pol)) for pol in a.policies})
