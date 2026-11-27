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
#   depth_v4 like depth_v3 but with a local memory map (depth_v4.py): early detour, obstacles in the blind zone remembered.
#   fusion  depth v3 says WHERE (distance, hole/obstacle, side space), the VLM says WHAT (vlm_class.py), asked again and
#           again while driving. Lock-step: the answer is used only after its real answer time has passed in simulation.
#           Uncertainty -> SLOW: no answer yet, no YES-able word, VLM and depth disagree, or "grass" that depth sees as tall.
#           PERSON/ANIMAL/HOLE -> stop (never detour); OBJECT/VEHICLE -> detour decided EARLY (in the warning zone) if the
#           side is free; GRASS (and depth: lower than --h_grass) -> drive through.
# detour is recorded as the decision and the robot stops there (driving the detour comes later).

import argparse, csv, json, os, socket, struct, time
import numpy as np
try:
    from vlm_class import ask as ask_vlm
except ImportError:
    ask_vlm = None

SCENARIOS = ["none", "person_stand", "person_lie", "person_moving", "rock", "drum", "ditch", "grass"]
ALWAYS_STOP = ("person", "person_lying", "person_moving", "ditch")

p = argparse.ArgumentParser()
p.add_argument("--policy", default="oracle", choices=["oracle", "drive", "depth_v3", "depth_v4", "fusion"])
p.add_argument("--model", default="qwen3-vl:4b", help="fusion: VLM model")
p.add_argument("--wording", default="", help="fusion: W1 or W2 (chosen by vlm_select.py)")
p.add_argument("--h_grass", type=float, default=0.6, help="fusion: grass is only passable if depth says it is lower than this (m)")
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
p.add_argument("--watch", action="store_true", help="live view: real-time pacing + decision panel and status light in Unity")
p.add_argument("--drive_detour", action="store_true", help="when the decision is DETOUR, really drive around the obstacle "
               "(path planned from the obstacle's true position) instead of stopping")
p.add_argument("--cam_h", type=float, default=0.0, help="camera height (m); 0 = keep what Unity has (Inspector)")
p.add_argument("--tilt", type=float, default=0.0, help="camera tilt (deg, + = down); used only with --cam_h")
p.add_argument("--save_images", action="store_true")
p.add_argument("--no_depth", action="store_true", help="do not ask for / save the depth grid")
p.add_argument("--out", default="run_field")
p.add_argument("--port", type=int, default=5555)
a = p.parse_args()
if a.policy == "fusion" and not a.wording:
    p.error("--policy fusion needs --wording (W1 or W2, from vlm_select.py)")
if a.policy in ("depth_v3", "depth_v4", "fusion"):
    from depth_v3 import DepthV3
    from depth_v4 import DepthV4
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

    def reset(self, z0, scenario, obs_z, obs_x, image=False, depth=True, cam_h=0.0, tilt=0.0):
        return self._call(dict(cmd="reset", x0=0.0, z0=z0, yaw=0.0, scenario=scenario, obs_z=obs_z, obs_x=obs_x, image=image, depth=depth,
                               cam_h=cam_h, tilt=tilt))

    def step(self, v, dt, image=False, depth=True, w=0.0, hud=None, label=None):
        d = dict(cmd="step", v=v, w=w, dt=dt, image=image, depth=depth)
        if hud is not None: d["hud"] = hud
        if label is not None: d["label"] = label
        return self._call(d)


def nearest_blocking(st):
    """the nearest non-passable object ahead (dict) and its lateral offset, or (None, 0)"""
    best = (None, 0.0, np.inf)
    for o in st["objects"]:
        if o["pass"] or o["kind"] == "ditch": continue
        dz = o["z"] - st["z"]
        if dz + o["hl"] < 0: continue
        if dz < best[2]: best = (o, o["x"] - st["x"], dz)
    return best[0], best[1]


def plan_detour(st):
    """detour path around the nearest obstacle: (x_ref function, end z) or None if there is no room before the crops"""
    o, lat = nearest_blocking(st)
    if o is None: return None
    hw = a.half_w + a.margin
    sides = []
    for sgn in (-1, 1):
        x_t = o["x"] + sgn * (o["hw"] + hw)
        if abs(x_t) + hw <= st["crop_offset"]: sides.append((abs(x_t), x_t))
    if not sides: return None
    x_t = min(sides)[1]                                   # the side that needs the smaller swerve
    z0, x0 = st["z"], st["x"]
    z_in = max(o["z"] - o["hl"] - 0.215 - 0.3, z0 + 0.8)
    z_hold = o["z"] + o["hl"] + 0.215 + 0.5
    z_back = z_hold + 2.5
    def x_ref(z):
        if z < z_in: return x0 + (x_t - x0) * np.clip((z - z0) / (z_in - z0), 0, 1)
        if z < z_hold: return x_t
        if z < z_back: return x_t * (1 - (z - z_hold) / (z_back - z_hold))
        return 0.0
    return x_ref, z_back + 1.0


def touching(st):
    """true footprint check (robot box vs object boxes) and crop check, used while driving a detour"""
    for o in st["objects"]:
        if o["pass"] or o["kind"] == "ditch": continue
        if abs(o["x"] - st["x"]) < o["hw"] + a.half_w and abs(o["z"] - st["z"]) < o["hl"] + 0.215: return "collision"
    if abs(st["x"]) + a.half_w > st["crop_offset"]: return "crop damage"
    return None


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


def fuse(dep, v, cls, waiting):
    """fusion decision: (label, reason)"""
    from vlm_class import STOP_CLASSES, DETOUR_CLASSES, PASS_CLASSES
    d, kind, z, X, H = V3.nearest(dep)
    dh, dw = zones(v)
    if kind is None or d > dw:
        if cls in STOP_CLASSES + DETOUR_CLASSES: return "slow", "vlm_sees_more"     # disagreement -> slow
        return "go", "clear"
    if kind == "hole": return ("stop" if d <= dh else "slow"), "depth_hole"
    win = (z > 0) & (np.abs(X) < V3.corr) & (H > V3.h_obs) & (z + V3.cam_front <= d + 1.0)
    hmax = float(H[win].max()) if win.any() else 0.0
    if cls in PASS_CLASSES and hmax <= a.h_grass: return "go", "grass_pass"
    if cls in ("PERSON", "ANIMAL", "HOLE"): return ("stop" if d <= dh else "slow"), cls.lower()
    if cls in DETOUR_CLASSES:
        if V3.side_space(d, z, X, H) >= V3.need: return "detour", "early_detour"
        return ("stop" if d <= dh else "slow"), "no_side_space"
    return ("stop" if d <= dh else "slow"), ("waiting" if waiting and cls is None else "uncertain")


u = Unity(a.port)
fr_f = open(os.path.join(a.out, "frames.csv"), "w", newline="")
fr = csv.writer(fr_f)
fr.writerow(["scenario", "trial", "t", "z_front", "v", "gap_m", "kind", "d_hazard", "d_warn", "gt_label", "acted_label", "vlm_class", "waiting", "reason"])
vl_f = open(os.path.join(a.out, "vlm_asks.csv"), "w", newline="") if a.policy == "fusion" else None
vl = csv.writer(vl_f) if vl_f else None
if vl: vl.writerow(["scenario", "trial", "t_sent", "gap_m", "answer", "latency_s"])
rows = []
for sc in scen:
    for k in range(a.trials):
        start = rng.uniform(a.start_min, a.start_max)
        st, jpg, dep = u.reset(a.obs_z - start - 0.215, sc, a.obs_z, a.obs_x, image=a.save_images, depth=not a.no_depth,
                               cam_h=a.cam_h, tilt=a.tilt)
        if a.policy in ("depth_v3", "depth_v4", "fusion"):          # the decider uses the camera Unity really has
            V3 = (DepthV4 if a.policy == "depth_v4" else DepthV3)(cam_height=st["cam_height"], tilt_deg=st["tilt_deg"], hfov_deg=st["hfov_deg"], img_w=st["img_w"], img_h=st["img_h"],
                         grid_w=st["depth_w"], grid_h=st["depth_h"], half_w=a.half_w, margin=a.margin)
        v, t, vt = 0.0, 0.0, a.v                       # current speed, time, target speed
        delay = max(0, int(round(a.t_r / a.dt)))
        queue, frames, depths, labs, gaps = [], 0, [], [], []
        decision, collided, min_gap, switches, prev = None, False, float("inf"), 0, None
        cls, pending, reason, n_ask, dec_gap = None, None, "", 0, float("nan")
        idir = os.path.join(a.out, f"{sc}_{k:02d}")
        if a.save_images: os.makedirs(idir, exist_ok=True)
        v = a.v                                          # the robot is already moving at working speed
        detour_plan, w_cmd, finished, t_wall = None, 0.0, None, time.time()
        while t < 120.0:
            lab, gap, kind, dh, dw = ground_truth(st, v)
            if not np.isnan(gap):
                min_gap = min(min_gap, gap)
                if gap < 0: collided = True
            if prev is not None and lab != prev: switches += 1
            prev = lab
            if a.policy == "fusion":
                if pending is not None and t >= pending[1] - 1e-9:
                    cls, pending = pending[0], None
                if pending is None and decision is None:
                    _, img, _ = u.step(0.0, 0.0, image=True, depth=False,           # picture of this moment (no time passes)
                                       hud=(f"{a.policy}   scene {sc}   trial {k}\nVLM thinking ... (simulated time paused)\n"
                                            f"last answer: {cls or '-'}") if a.watch else None)
                    ans, L = ask_vlm(a.model, a.wording, img)
                    pending = (ans, t + L); n_ask += 1
                    vl.writerow([sc, k, f"{t:.2f}", "" if np.isnan(gap) else f"{gap:.3f}", ans, f"{L:.2f}"]); vl_f.flush()
                    if a.save_images: open(os.path.join(idir, f"ask_t{t:06.2f}_{ans}.jpg"), "wb").write(img)
                mine, reason = fuse(dep, v, cls, pending is not None)
            elif a.policy == "depth_v3":
                mine = V3.decide(dep, dh, dw)[0]
            elif a.policy == "depth_v4":
                pose = (st["x"], st["z"], st["yaw"])         # Unity pose = perfect odometry
                V3.update(dep, pose)
                mine = V3.decide_pose(pose, dh, dw)[0]
            else:
                mine = lab
            queue.append(mine)
            acted = queue[-1 - delay] if len(queue) > delay else "go"
            fr.writerow([sc, k, f"{t:.2f}", f"{st['z'] + st['front_offset']:.3f}", f"{v:.3f}", "" if np.isnan(gap) else f"{gap:.3f}",
                         kind or "", f"{dh:.3f}", f"{dw:.3f}", lab, acted if a.policy != "drive" else "go",
                         cls or "", int(pending is not None), reason])
            labs.append(lab); gaps.append(gap)
            if dep is not None: depths.append(dep.astype(np.float16))
            if jpg is not None: open(os.path.join(idir, f"t{t:06.2f}_{lab}.jpg"), "wb").write(jpg)
            frames += 1
            if a.policy != "drive":
                if acted in ("stop", "detour") and decision is None:
                    decision, dec_gap = acted, gap
                    if decision == "detour" and a.drive_detour:
                        detour_plan = plan_detour(st)
                        if detour_plan is None: decision = "stop"          # no room before the crops: stop instead
                if detour_plan is not None:
                    vt = a.v_slow
                    look = 1.0
                    psi_des = np.arctan2(detour_plan[0](st["z"] + look) - st["x"], look)
                    w_cmd = float(np.clip(-2.5 * (psi_des - st["yaw"]), -1.2, 1.2))   # Unity: w > 0 turns left, yaw > 0 = right
                    hit = touching(st)
                    if hit: finished = hit; collided = collided or hit == "collision"
                    elif st["z"] > detour_plan[1]: finished = "detoured past"
                else:
                    vt = 0.0 if decision else (a.v_slow if acted == "slow" else a.v)
            if finished: break
            if decision and detour_plan is None and v <= 1e-6: break
            if collided and a.policy != "drive": break
            if detour_plan is None and st["z"] + st["front_offset"] > a.obs_z + 4.0: break          # passed the obstacle area
            dv = np.clip(vt - v, -a.a * a.dt, a.a * a.dt)
            v_next = max(0.0, v + dv)
            hud = label = None
            if a.watch:
                label = "detour" if detour_plan is not None else (acted if a.policy != "drive" else lab)
                g = "none ahead" if np.isnan(gap) else f"{gap:.1f} m"
                lines = [f"{a.policy}   scene {sc}   trial {k}", f"speed {v:.1f} m/s    obstacle {g}"]
                if a.policy == "fusion": lines.append(f"VLM says: {cls or '-'}   ({reason})")
                if a.policy in ("depth_v3", "depth_v4"): lines.append(f"depth decider: {mine}")
                lines.append(f"correct label now: {lab}")
                hud = "\n".join(lines)
            st, jpg, dep = u.step(0.5 * (v + v_next), a.dt, image=a.save_images, depth=not a.no_depth, w=w_cmd if detour_plan is not None else 0.0,
                                  hud=hud, label=label)
            v = v_next; t += a.dt
            if a.watch:
                time.sleep(max(0.0, a.dt - (time.time() - t_wall))); t_wall = time.time()
        final_gap = gaps[-1]
        if a.policy != "drive":
            outcome = "collision" if collided else (f"{decision} at {final_gap:.2f} m" if decision else "drove through")
            if finished in ("detoured past", "crop damage"): outcome = finished
            if a.watch: time.sleep(1.5)
            if a.policy == "fusion": outcome += f" ({n_ask} asks)"
        else:
            outcome = "drove through (no reaction)"
        cnt = {L: labs.count(L) for L in ("go", "slow", "stop", "detour")}
        rows.append(dict(scenario=sc, trial=k, cam_h=st["cam_height"], tilt=st["tilt_deg"], start_m=round(start, 2), decision=decision or "",
                         decision_gap_m="" if np.isnan(dec_gap) else round(dec_gap, 3), vlm_asks=n_ask, finished=finished or "", final_gap_m="" if np.isnan(final_gap) else round(final_gap, 3),
                         min_gap_m="" if min_gap == float("inf") else round(min_gap, 3), collided=int(collided), label_switches=switches,
                         frames=frames, **{f"n_{L}": n for L, n in cnt.items()}, time_s=round(t, 2)))
        if depths:
            np.savez_compressed(os.path.join(a.out, f"{sc}_{k:02d}_depth.npz"), depth=np.stack(depths), label=np.array(labs), gap=np.array(gaps, dtype=np.float32))
        print(f"{sc:14s} trial {k}: {outcome:34s}  min gap {rows[-1]['min_gap_m']!s:>6} m  frames {frames:4d}  "
              f"go {cnt['go']:3d} slow {cnt['slow']:3d} stop {cnt['stop']:3d} detour {cnt['detour']:3d}  label switches {switches}", flush=True)
fr_f.close()
if vl_f: vl_f.close()
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
