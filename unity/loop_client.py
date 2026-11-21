# loop_client.py  -  the Python half of the closed loop with Unity (OrchardWorld.cs).
#
#   python loop_client.py --policy depth_drive                 # depth rule, decide while driving
#   python loop_client.py --policy depth_step6                 # depth rule, move-stop-ask
#   python loop_client.py --policy vlm_step6 --model qwen3-vl:4b          # VLM decides, robot stands while it thinks
#   python loop_client.py --policy vlm_drive --model qwen3-vl:4b          # VLM decides, robot KEEPS DRIVING while it thinks
#   python loop_client.py --policy depth_drive --trials 5 --save_images
#   python loop_client.py --policy vlm_drive --start_min -22 --start_max -18   # start far back in the open headland
#
# Every loop step:  Python sends (v, w, dt)  ->  Unity moves the robot, renders the camera, measures edge depth
#                   ->  Unity answers with pose + depth (+ JPEG)  ->  Python decides  ->  next step.
# Stop error = camera z where the robot comes to rest - entrance z (the farm labelling rule, computed in Unity).
# Success = |error| < 2 m. Every check is written to <out>/trialNN.csv; images to <out>/trialNN/ with --save_images.

import argparse
import csv
import io
import json
import math
import os
import re
import socket
import struct
import time

p = argparse.ArgumentParser()
p.add_argument("--policy", default="depth_drive", choices=["depth_drive", "depth_step6", "vlm_step6", "vlm_drive"])
p.add_argument("--model", default="qwen3-vl:4b")
p.add_argument("--wording", default="A", choices=["A", "C"])
p.add_argument("--trials", type=int, default=3)
p.add_argument("--speed", type=float, default=1.2)
p.add_argument("--D", type=float, default=3.5, help="depth rule threshold (m), frozen from the farm data")
p.add_argument("--band", default="full", choices=["full", "hi"],
               help="depth rule rows: full = rows 25-75 %% (frozen farm rule), hi = rows 25-50 %% only (above the horizon, v2 candidate)")
p.add_argument("--K", type=int, default=3, help="drive policies: YES in a row before stopping")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--save_images", action="store_true")
p.add_argument("--out", default="unity_runs")
p.add_argument("--port", type=int, default=5555)
p.add_argument("--start_min", type=float, default=-10.0, help="robot start z is drawn from [start_min, start_max] (entrance is at about z = -2.5)")
p.add_argument("--start_max", type=float, default=-7.0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

WORDING = {"A": "trees are close on both the left and right sides of the camera",
           "C": "tree branches fill both the left edge and the right edge of the camera's view, and a long narrow path between tree rows goes straight ahead"}
QUESTION = ("This image comes from the front camera of a small ground robot. The camera is where the robot is.\n"
            "Is the following true in this image right now: \"{c}\"?\n"
            "Seeing the object from a distance is NOT enough; it must be true exactly as written.\n"
            "Answer with one word: YES or NO.")


class Unity:
    """Lock-step connection: every call moves the robot (or not) and returns what it sees."""

    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=60)

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Unity closed the connection")
            buf += chunk
        return buf

    def call(self, **req):
        self.s.sendall((json.dumps(req) + "\n").encode())
        n = struct.unpack(">i", self._recv(4))[0]
        state = json.loads(self._recv(n))
        m = struct.unpack(">i", self._recv(4))[0]
        img = self._recv(m) if m else None
        return state, img

    def reset(self, x0, z0, yaw):
        return self.call(cmd="reset", x0=x0, z0=z0, yaw=yaw, image=False)

    def step(self, v, w, dt, image=False):
        return self.call(cmd="step", v=v, w=w, dt=dt, image=image)


def drive_for(u, st, seconds, v, image=False):
    """Drive for `seconds` of simulated time in small 0.05 s steps, re-computing the steering each step
    (the robot's low-level controller keeps it in the lane while the VLM is thinking)."""
    left = seconds
    img = None
    while left > 1e-6:
        h = min(0.05, left)
        left -= h
        st, img = u.step(v, steer(st), h, image=(image and left <= 1e-6))
    return st, img


def steer(st):
    """keep to the lane centre (x = 0) and straight (yaw = 0); w > 0 turns left"""
    return max(-1.0, min(1.0, 1.5 * st["x"] + 2.0 * st["yaw"]))


def depth_yes(st):
    l, r = (st["left_m"], st["right_m"]) if a.band == "full" else (st.get("left_hi"), st.get("right_hi"))
    return l is not None and r is not None and l < a.D and r < a.D, f"{l}/{r}"


def vlm_yes(img):
    import ollama
    t0 = time.time()
    r = ollama.chat(model=a.model, messages=[{"role": "user", "content": QUESTION.format(c=WORDING[a.wording]), "images": [img]}],
                    options={"temperature": 0, "seed": 42})
    lat = time.time() - t0
    m = re.findall(r"\b(yes|no)\b", (r["message"]["content"] or "").lower())
    ans = m[-1].upper() if m else "?"
    return ans == "YES", ans, lat


def run_trial(u, k, rng):
    x0, yaw0, z0 = rng.uniform(-0.3, 0.3), rng.uniform(-0.08, 0.08), rng.uniform(a.start_min, a.start_max)
    st, _ = u.reset(x0, z0, yaw0)
    entr = st["entrance_z"]
    fh = open(os.path.join(a.out, f"trial{k:02d}.csv"), "w", newline="")
    log = csv.writer(fh); log.writerow(["t", "cam_z", "rel_to_entrance", "answer", "yes", "latency_s", "left_m", "right_m"])
    imgdir = os.path.join(a.out, f"trial{k:02d}")
    if a.save_images:
        os.makedirs(imgdir, exist_ok=True)
    need_img = a.policy.startswith("vlm") or a.save_images
    streak, first_yes, wait = 0, False, 0.0
    stopped = False

    def look(st, img):
        nonlocal wait
        if a.policy.startswith("depth"):
            yes, txt = depth_yes(st); lat = 0.0
        else:
            yes, txt, lat = vlm_yes(img)
        if a.save_images and img:
            open(os.path.join(imgdir, f"t{st['t']:07.2f}_{txt.replace('/', '-')}.jpg"), "wb").write(img)
        log.writerow([st["t"], round(st["cam_z"], 3), round(st["cam_z"] - entr, 3), txt, int(yes), round(lat, 2), st["left_m"], st["right_m"]])
        return yes, lat

    while st["cam_z"] - entr < 12.0 and st["t"] < 120:
        if a.policy.endswith("drive"):
            st, img = drive_for(u, st, 0.2, a.speed, image=need_img)          # drive 0.2 s, then look
            yes, lat = look(st, img)
            if lat > 0:                                                        # the robot keeps driving while the VLM thinks
                st, _ = drive_for(u, st, lat, a.speed)
            streak = streak + 1 if yes else 0
            if streak >= a.K:
                stopped = True
                break
        else:                                                                  # move-stop-ask
            target = st["cam_z"] + (0.25 if first_yes else 1.0)
            while st["cam_z"] < target:
                st, _ = u.step(min(a.speed, 0.4 + 2 * (target - st["cam_z"])), steer(st), 0.05)
            st, img = u.step(0.0, 0.0, 0.0, image=need_img)                    # stand still and look
            yes, lat = look(st, img)
            wait += lat
            if lat > 0:
                st, _ = u.step(0.0, 0.0, lat)                                  # time passes, the robot stands
            first_yes = first_yes or yes
            streak = streak + 1 if yes else 0
            if streak >= 6:
                stopped = True
                break
    if stopped and a.policy.endswith("drive"):
        st, _ = u.step(0.0, 0.0, 0.3)                                          # brake (kinematic robot stops at once)
    fh.close()
    err = st["cam_z"] - entr
    return dict(trial=k, stopped=int(stopped), error_m=round(err, 2) if stopped else "", sim_time_s=round(st["t"], 1),
                entrance_z=round(entr, 2), lookalike=st["lookalike"])


import random
rng = random.Random(a.seed)
u = Unity(a.port)
rows = []
for k in range(a.trials):
    r = run_trial(u, k, rng)
    rows.append(r)
    print(f"trial {k}: " + (f"stopped at {r['error_m']:+.2f} m from the entrance" if r["stopped"] else "did not stop")
          + f"  ({r['sim_time_s']} s simulated)")
with open(os.path.join(a.out, "trials.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = sum(1 for r in rows if r["stopped"] and abs(r["error_m"]) < 2)
print(f"\n{a.policy}" + (f" ({a.model}, wording {a.wording})" if a.policy.startswith("vlm") else f" (D = {a.D} m, band {a.band})")
      + f": {ok}/{len(rows)} within 2 m  -> {a.out}/trials.csv")
