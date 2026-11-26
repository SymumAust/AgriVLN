# field_client.py  -  closed loop with FieldWorld.cs (open field + one obstacle): go / slow / stop / detour
#
#   python field_client.py --policy oracle --scenarios all --trials 3 --out run_field_oracle
#   python field_client.py --policy drive  --scenarios all --trials 1 --out run_field_drive --save_images
#
# Labels come from the GROUND TRUTH that Unity sends (where every object really is), using the zone rule from the
# professor's plan (ISO 18497-2 idea, formula = our proposal):
#     d_hazard = t_r * v + v^2 / (2 a) + d_m          d_warn = d_hazard + t_w * v          v = speed at this moment
#   nearest non-passable object in the robot's corridor (robot half width + margin each side), distance from the front bumper d:
#     d <= d_hazard : STOP  (people, moving people, ditches: always)   or DETOUR (static object AND free side space)
#     d <= d_warn   : SLOW
#     otherwise     : GO
#   side space = free strip between the object and the crop beds, at least robot width + 2 margins (no crop damage).
#
# Policies:
#   drive   constant speed, never reacts. Only records what the correct label would be at every frame (frame ratio).
#   oracle  acts on the ground-truth label, but with a reaction delay t_r and braking at a (like a perfect sensor).
#           Checks that the zone rule itself is sound: it should stop about d_m before every obstacle and never on grass.
#   depth_v3 decides from the depth grid alone (depth_v3.py), with the same reaction delay and braking as the oracle;
#           the ground-truth label is still logged next to it.
# detour is recorded as the decision and the robot stops there (driving the detour comes later).

import argparse, csv, json, os, socket, struct, time
import numpy as np

SCENARIOS = ["none", "person_stand", "person_lie", "person_moving", "rock", "drum", "ditch", "grass"]
ALWAYS_STOP = ("person", "person_lying", "person_moving", "ditch")

p = argparse.ArgumentParser()
p.add_argument("--policy", default="oracle", choices=["oracle", "drive", "depth_v3"])
p.add_argument("--scenarios", default="all", help="comma list or 'all': " + ",".join(SCENARIOS))
p.add_argument("--trials", type=int, default=3)
p.add_argument("--obs_z", type=float, default=30.0)
p.add_argument("--obs_x", type=float, default=0.0)
p.add_argument("--start_min", type=float, default=18.0, help="start: front bumper this many m before the obstacle (random in range)")
p.add_argument("--start_max", type=float, default=22.0)
p.add_argument("--v", type=float, default=1.0, help="working speed (m/s)")
p.add_argument("--v_slow", type=float, default=0.5)
p.add_argument("--dt", type=float, default=0.1, help="control period (s)")
p.add_argument("--t_r", type=float, default=0.3, help="reaction time (s)")
p.add_argument("--a", type=float, default=1.0, help="braking / speed change (m/s^2)")
p.add_argument("--d_m", type=float, default=0.5, help="margin (m)")
p.add_argument("--t_w", type=float, default=3.0, help="warning time (s)")
p.add_argument("--half_w", type=float, default=0.215, help="robot half width (m)")
p.add_argument("--margin", type=float, default=0.15, help="side margin each side (m)")
p.add_argument("--zone_speed", default="current", choices=["current", "nominal"],
               help="current = zones from the speed at this moment (as in the plan); nominal = always from --v")
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--save_images", action="store_true")
p.add_argument("--no_depth", action="store_true", help="do not ask for / save the depth grid")
p.add_argument("--out", default="run_field")
p.add_argument("--port", type=int, default=5555)
a = p.parse_args()
if a.policy == "depth_v3":
    from depth_v3 import DepthV3
    V3 = DepthV3(half_w=a.half_w, margin=a.margin)
scen = SCENARIOS if a.scenarios == "all" else a.scenarios.split(",")
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)


class Unity:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=120)
        self.f = self.s.makefile("rb")

    def _read(self, n):
        b = self.f.read(n)
        if len(b) != n: raise ConnectionError("Unity closed the connection")
        return b

    def _call(self, d):
        self.s.sendall((json.dumps(d) + "\n").encode())
        head = json.loads(self._read(struct.unpack(">i", self._read(4))[0]))
        n = struct.unpack(">i", self._read(4))[0]; jpg = self._read(n) if n else None
        n = struct.unpack(">i", self._read(4))[0]
        dep = np.frombuffer(self._read(n), "<f4").reshape(head["depth_h"], head["depth_w"]) if n else None
        return head, jpg, dep

    def reset(self, z0, scenario, obs_z, obs_x, image=False, depth=True):
        return self._call(dict(cmd="reset", x0=0.0, z0=z0, yaw=0.0, scenario=scenario, obs_z=obs_z, obs_x=obs_x, image=image, depth=depth))

    def step(self, v, dt, image=False, depth=True):
        return self._call(dict(cmd="step", v=v, w=0.0, dt=dt, image=image, depth=depth))


def zones(v):
    v = a.v if a.zone_speed == "nominal" else v
    dh = a.t_r * v + v * v / (2 * a.a) + a.d_m
    return dh, dh + a.t_w * v


def ground_truth(st, v):
    """Correct label at this moment + the distance (front bumper -> nearest blocking object) + which object."""
    cs, sn = np.cos(st["yaw"]), np.sin(st["yaw"])
    front = st["front_offset"]
    corr = a.half_w + a.margin
    best = None
    for o in st["objects"]:
        if o["pass"]:
            continue
        dx, dz = o["x"] - st["x"], o["z"] - st["z"]
        fwd = dz * cs + dx * sn                      # ahead of the robot centre
        lat = dx * cs - dz * sn                      # to the right
        if abs(lat) - o["hw"] >= corr:               # not in the robot's corridor
            continue
        d = fwd - o["hl"] - front                    # front bumper to the object's near face
        if fwd + o["hl"] < front:                    # already passed
            continue
        if best is None or d < best[0]:
            best = (d, o, lat)
    dh, dw = zones(v)
    if best is None:
        return "go", float("nan"), None, dh, dw
    d, o, lat = best
    if d <= dh:
        if o["kind"] in ALWAYS_STOP:
            lab = "stop"
        else:
            need = 2 * (a.half_w + a.margin)
            left_gap = (lat - o["hw"]) + st["crop_offset"]
            right_gap = st["crop_offset"] - (lat + o["hw"])
            lab = "detour" if max(left_gap, right_gap) >= need else "stop"
    elif d <= dw:
        lab = "slow"
    else:
        lab = "go"
    return lab, d, o["kind"], dh, dw


u = Unity(a.port)
fr_f = open(os.path.join(a.out, "frames.csv"), "w", newline="")
fr = csv.writer(fr_f)
fr.writerow(["scenario", "trial", "t", "z_front", "v", "gap_m", "kind", "d_hazard", "d_warn", "gt_label", "acted_label"])
rows = []
for sc in scen:
    for k in range(a.trials):
        start = rng.uniform(a.start_min, a.start_max)
        st, jpg, dep = u.reset(a.obs_z - start - 0.215, sc, a.obs_z, a.obs_x, image=a.save_images, depth=not a.no_depth)
        v, t, vt = 0.0, 0.0, a.v                       # current speed, time, target speed
        delay = max(0, int(round(a.t_r / a.dt)))
        queue, frames, depths, labs, gaps = [], 0, [], [], []
        decision, collided, min_gap, switches, prev = None, False, float("inf"), 0, None
        idir = os.path.join(a.out, f"{sc}_{k:02d}")
        if a.save_images: os.makedirs(idir, exist_ok=True)
        v = a.v                                          # the robot is already moving at working speed
        while t < 120.0:
            lab, gap, kind, dh, dw = ground_truth(st, v)
            if not np.isnan(gap):
                min_gap = min(min_gap, gap)
                if gap < 0: collided = True
            if prev is not None and lab != prev: switches += 1
            prev = lab
            mine = V3.decide(dep, dh, dw)[0] if a.policy == "depth_v3" else lab
            queue.append(mine)
            acted = queue[-1 - delay] if len(queue) > delay else "go"
            fr.writerow([sc, k, f"{t:.2f}", f"{st['z'] + st['front_offset']:.3f}", f"{v:.3f}", "" if np.isnan(gap) else f"{gap:.3f}",
                         kind or "", f"{dh:.3f}", f"{dw:.3f}", lab, acted if a.policy != "drive" else "go"])
            labs.append(lab); gaps.append(gap)
            if dep is not None: depths.append(dep.astype(np.float16))
            if jpg is not None: open(os.path.join(idir, f"t{t:06.2f}_{lab}.jpg"), "wb").write(jpg)
            frames += 1
            if a.policy != "drive":
                if acted in ("stop", "detour") and decision is None: decision = acted
                vt = 0.0 if decision else (a.v_slow if acted == "slow" else a.v)
            if decision and v <= 1e-6: break
            if collided and a.policy != "drive": break
            if st["z"] + st["front_offset"] > a.obs_z + 4.0: break          # passed the obstacle area
            dv = np.clip(vt - v, -a.a * a.dt, a.a * a.dt)
            v_next = max(0.0, v + dv)
            st, jpg, dep = u.step(0.5 * (v + v_next), a.dt, image=a.save_images, depth=not a.no_depth)
            v = v_next; t += a.dt
        final_gap = gaps[-1]
        if a.policy != "drive":
            outcome = "collision" if collided else (f"{decision} at {final_gap:.2f} m" if decision else "drove through")
        else:
            outcome = "drove through (no reaction)"
        cnt = {L: labs.count(L) for L in ("go", "slow", "stop", "detour")}
        rows.append(dict(scenario=sc, trial=k, start_m=round(start, 2), decision=decision or "", final_gap_m="" if np.isnan(final_gap) else round(final_gap, 3),
                         min_gap_m="" if min_gap == float("inf") else round(min_gap, 3), collided=int(collided), label_switches=switches,
                         frames=frames, **{f"n_{L}": n for L, n in cnt.items()}, time_s=round(t, 2)))
        if depths:
            np.savez_compressed(os.path.join(a.out, f"{sc}_{k:02d}_depth.npz"), depth=np.stack(depths), label=np.array(labs), gap=np.array(gaps, dtype=np.float32))
        print(f"{sc:14s} trial {k}: {outcome:24s}  min gap {rows[-1]['min_gap_m']!s:>6} m  frames {frames:4d}  "
              f"go {cnt['go']:3d} slow {cnt['slow']:3d} stop {cnt['stop']:3d} detour {cnt['detour']:3d}  label switches {switches}", flush=True)
fr_f.close()
with open(os.path.join(a.out, "trials.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

tot = {L: sum(r[f"n_{L}"] for r in rows) for L in ("go", "slow", "stop", "detour")}
n = sum(tot.values())
print(f"\nframe ratio over all {n} frames: " + "  ".join(f"{L} {100 * c / n:.0f} %" for L, c in tot.items()) +
      f"   -> normal (go) {100 * tot['go'] / n:.0f} % / intervention {100 * (n - tot['go']) / n:.0f} %")
if a.policy != "drive":
    bad = [r for r in rows if r["collided"] or (r["scenario"] in ("none", "grass") and r["decision"])]
    print(f"{a.policy}: {len(rows) - len(bad)}/{len(rows)} trials safe and without false stop (no collision, no stop on none/grass)  -> {a.out}/trials.csv")
    want = {"person_stand": "stop", "person_lie": "stop", "person_moving": "stop", "ditch": "stop", "rock": "detour", "drum": "detour"}
    ok = [r for r in rows if r["scenario"] in want and r["decision"] == want[r["scenario"]] and not r["collided"]]
    n_obs = sum(r["scenario"] in want for r in rows)
    if n_obs: print(f"{a.policy}: right decision (stop for people/ditch, detour for rock/drum) in {len(ok)}/{n_obs} obstacle trials")
