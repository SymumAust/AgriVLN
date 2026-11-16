# Live closed-loop test of the stop decision in a simulated orchard (PyChrono 10).
# The robot drives from the headland into a tree row and must stop when "trees are close on both sides".
# Unlike closed_loop_sim.py (replay of recorded frames), here the robot really moves, the soil makes it slip,
# and the depth is measured from where the robot actually is.
#
#   python row_stop_sim.py --trials 20                        # depth rule from ray casting (no GPU needed)
#   python row_stop_sim.py --trials 20 --policy step6         # move-stop-ask with the depth rule
#   python row_stop_sim.py --trials 20 --depth camera         # depth from the simulated depth camera (GPU, OptiX)
#   python row_stop_sim.py --trials 10 --decider vlm --policy step6 --model qwen3-vl:4b     # the VLM decides, live (GPU + Ollama)
#   python row_stop_sim.py --trials 10 --scenario lookalike   # a wide tree corridor (like the turning lane) before the row
#
# Depth rule (frozen on the farm data, plan_depth.txt): left / right = median depth of the outer 15 % of the image
# width, rows 25-75 %; YES if both < D (3.5 m). Ray casting reproduces this with a grid of rays through those pixels
# (z-depth along the camera axis, like the ZED; trees and ground are hit, nothing beyond 20 m).
# Policies:  drive = decide while driving, stop after K YES in a row at 5 Hz (depth is instant, so this is possible)
#            step6 = move-stop-ask as on the farm: 1 m steps, 0.25 m after the first YES, stop at 6 YES in a row
# Row entrance (label) = the same rule we used to label the farm clips by eye: the first camera position at which a canopy
# of the two rows around the lane reaches BOTH the left and the right edge of the image (outermost pixel column, horizon
# row), computed from the geometry. Success = |camera x at rest - entrance| < 2 m.
# The orchard has 8 parallel hedge rows (so the edges of the image also see the neighbouring row ends, as on the farm).
# Each trial randomises: canopy radius 1.5-1.9 m, clear lane between hedges 3.2-4.0 m (row spacing = lane + 2 radii),
# tree spacing 1.2-1.6 radii (canopies overlap into a hedge), start offset, heading, soil.
# Deciders: depth = the frozen farm depth rule;  vlm = the simulated camera image goes to a VLM in Ollama with the same
#   question as probe_arrival.py (wording A or C); blank answers count as NO. With the drive policy the robot keeps
#   driving while the VLM thinks (the simulation advances by the measured answer time); with step6 it stands still.
# Scenarios: row = open headland, then the row;  lookalike = before the open headland the robot drives through a wide
#   corridor between two short hedges (faces 3.5-5 m from the robot, like the turning lane that fooled the VLM on the farm).
# Output: trials.csv (one line per trial), checks_trialNN.csv (every check), and a summary.

import argparse
import csv
import math
import os
import time

import numpy as np
import pychrono as chrono
import pychrono.vehicle as veh

p = argparse.ArgumentParser()
p.add_argument("--trials", type=int, default=10)
p.add_argument("--policy", default="drive", choices=["drive", "step6"])
p.add_argument("--depth", default="ray", choices=["ray", "camera"])
p.add_argument("--D", type=float, default=3.5)
p.add_argument("--K", type=int, default=3, help="drive policy: YES in a row (5 Hz) before stopping")
p.add_argument("--speed", type=float, default=1.2)
p.add_argument("--soils", nargs="+", default=["firm", "soft", "mud"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="row_stop_out")
p.add_argument("--save_images", action="store_true", help="camera mode: save RGB frames of each trial")
p.add_argument("--decider", default="depth", choices=["depth", "vlm"])
p.add_argument("--model", default="qwen3-vl:4b")
p.add_argument("--wording", default="A", choices=["A", "C"])
p.add_argument("--scenario", default="row", choices=["row", "lookalike"])
p.add_argument("--fake_vlm", action="store_true", help="testing only: a fake VLM that answers with the ray depth rule after 2 s")
p.add_argument("--log_depth", action="store_true", help="write every check (and, in camera mode, the ray-cast values at the same moment and the depth images) to depth_trialNN.csv / npy")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

SOILS = {"firm": (5301e3, 102e3, 0.793, 1.3e3, 31.1, 1.2e-2, 4e8, 3e4),
         "soft": (0.2e6, 0, 1.1, 0, 30.0, 1.0e-2, 4e7, 3e4),
         "mud":  (0.1e6, 0, 1.1, 0.5e3, 18.0, 3.0e-2, 2e7, 3e4)}
R_W, W_W, B, L = 0.098, 0.04, 0.37, 0.262
CH = (0.43, 0.30, 0.15)
CAM_X, CAM_Z, HFOV, W_IMG, H_IMG = 0.2, 0.09, 1.78, 640, 360   # ZED2i as on the CitrusFarm Jackal: ~0.28 m above ground, 102 deg (calibration page)
DT, RATE = 2e-3, 5.0
WORDING = {"A": "trees are close on both the left and right sides of the camera",
           "C": "tree branches fill both the left edge and the right edge of the camera's view, and a long narrow path between tree rows goes straight ahead"}
QUESTION = ("This image comes from the front camera of a small ground robot. The camera is where the robot is.\n"
            "Is the following true in this image right now: \"{c}\"?\n"
            "Seeing the object from a distance is NOT enough; it must be true exactly as written.\n"
            "Answer with one word: YES or NO.")              # identical to probe_arrival.py (default mode)
SETTLE = 2.0                       # s: let the robot settle on the soil before any check


def ray_grid():
    """Directions (camera frame: x forward, y left, z up) through the two edge bands of the image, built exactly like
    Chrono's camera: dir = forward + d.x * h * right + d.y * h * up, h = hFOV / pi * 2, d.y scaled by height / width."""
    h = HFOV / math.pi * 2
    us = np.linspace(0.0, 0.15, 6, endpoint=False) + 0.0125         # fraction of image width from the left edge
    vs = np.linspace(0.25, 0.75, 8)
    out = {}
    for side, ufr in (("left", us), ("right", 1 - us)):
        d = []
        for u in ufr:
            for v in vs:
                dx = 2 * u - 1
                dy = (1 - 2 * v) * H_IMG / W_IMG
                d.append((1.0, -dx * h, dy * h))   # x = 1, so the ray parameter equals the z-depth
        out[side] = d
    return out


RAYS = ray_grid()


def leaf_material(rng):
    m = chrono.ChVisualMaterial()
    m.SetDiffuseColor(chrono.ChColor(0.10, rng.uniform(0.28, 0.40), 0.08))
    m.SetKdTexture(chrono.GetChronoDataFile("sensor/textures/green_grass.jpg"))
    m.SetTextureScale(4, 4); m.SetRoughness(0.9)
    return m


def run_trial(k, rng):
    soil = a.soils[k % len(a.soils)]
    lane = rng.uniform(3.2, 4.0); crown_r = rng.uniform(1.5, 1.9); row_gap = lane + 2 * crown_r   # clear driving lane between hedges
    tree_gap = crown_r * rng.uniform(1.2, 1.6)                                                     # canopies overlap along the row
    y_off = rng.uniform(-0.3, 0.3); yaw0 = rng.uniform(-0.08, 0.08); x0 = rng.uniform(-10.0, -7.0)
    look_half = rng.uniform(3.5, 5.0)                       # lookalike corridor: canopy faces this far from the path
    if a.scenario == "lookalike":
        x0 = rng.uniform(-20.0, -18.0)

    sys_ = chrono.ChSystemSMC()
    sys_.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    sys_.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    yc = row_gap / 2
    q0 = chrono.QuatFromAngleZ(yaw0)
    chassis = chrono.ChBodyEasyBox(*CH, 1000, True, False)
    chassis.SetMass(13.0); chassis.SetInertiaXX(chrono.ChVector3d(0.1, 0.2, 0.25))
    chassis.SetPos(chrono.ChVector3d(x0, yc + y_off, R_W + 0.09)); chassis.SetRot(q0)
    sys_.Add(chassis)
    wmat = chrono.ChContactMaterialSMC(); wmat.SetFriction(0.8)
    motors, funcs = [], []
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        w = chrono.ChBodyEasyCylinder(chrono.ChAxis_Y, R_W, W_W, 1000, True, True, wmat)
        w.SetMass(1.0)
        off = q0.Rotate(chrono.ChVector3d(sx * L / 2, sy * B / 2, 0))
        w.SetPos(chrono.ChVector3d(x0, yc + y_off, R_W + 0.03) + off); w.SetRot(q0)
        sys_.Add(w)
        m = chrono.ChLinkMotorRotationSpeed(); f = chrono.ChFunctionConst(0.0); m.SetSpeedFunction(f)
        m.Initialize(w, chassis, chrono.ChFramed(w.GetPos(), q0 * chrono.QuatFromAngleX(-math.pi / 2)))
        sys_.Add(m); motors.append(m); funcs.append(f)

    terrain = veh.SCMTerrain(sys_)
    terrain.SetReferenceFrame(chrono.ChCoordsysd(chrono.ChVector3d(-5.0, yc, 0), chrono.QUNIT))
    # the rest of the field: a flat, visual-only ground plane just below the soil surface
    terrain.SetSoilParameters(*SOILS[soil])
    terrain.SetPlotType(veh.SCMTerrain.PLOT_NONE, 0, 1)
    terrain.SetColor(chrono.ChColor(0.55, 0.45, 0.33))
    terrain.Initialize(42.0, 6.0, 0.05)        # soft soil only around the driving lane (keeps GPU memory small)
    terrain.AddActiveDomain(chassis, chrono.ChVector3d(0, 0, 0), chrono.ChVector3d(1.0, 0.8, 0.6))

    tmat = chrono.ChContactMaterialSMC()
    canopies = []
    ground_mat = chrono.ChVisualMaterial(); ground_mat.SetDiffuseColor(chrono.ChColor(0.55, 0.45, 0.33))
    ground_mat.SetRoughness(1.0)
    terrain.SetMeshWireframe(False)
    field = chrono.ChBodyEasyBox(80.0, 12 * row_gap, 0.02, 1000, True, False)
    field.SetPos(chrono.ChVector3d(0, yc, -0.02)); field.SetFixed(True)
    field.GetVisualShape(0).SetMaterial(0, ground_mat); sys_.Add(field)
    terrain.GetMesh().SetMaterial(0, ground_mat) if terrain.GetMesh().GetNumMaterials() else terrain.GetMesh().AddMaterial(ground_mat)
    for yrow in [kk * row_gap for kk in range(-3, 5)]:        # 8 parallel rows: the target gap is between rows 0 and 1
        for x in np.arange(0.0, 15.01, tree_gap):
            r = crown_r * rng.uniform(0.92, 1.0)
            trunk = chrono.ChBodyEasyCylinder(chrono.ChAxis_Z, 0.08, 0.9, 500, True, True, tmat)
            trunk.SetPos(chrono.ChVector3d(float(x), yrow, 0.45)); trunk.SetFixed(True)
            trunk.GetVisualShape(0).SetColor(chrono.ChColor(0.35, 0.25, 0.15)); sys_.Add(trunk)
            # canopy = sphere (the sensor renderer draws spheres, boxes, cylinders and meshes, not ellipsoids)
            crown = chrono.ChBodyEasySphere(r, 50, True, True, tmat)
            crown.SetPos(chrono.ChVector3d(float(x), yrow, 0.85 * r)); crown.SetFixed(True)
            canopies.append((float(x), yrow, r))
            crown.GetVisualShape(0).SetMaterial(0, leaf_material(rng))
            sys_.Add(crown)

    if a.scenario == "lookalike":                              # short hedges on both sides, x = -14 .. -8 (not part of the label)
        for side in (1, -1):
            for x in np.arange(-14.0, -7.99, crown_r * 1.3):
                r = crown_r * rng.uniform(0.92, 1.0)
                cr = chrono.ChBodyEasySphere(r, 50, True, True, tmat)
                cr.SetPos(chrono.ChVector3d(float(x), yc + side * (look_half + r), 0.85 * r)); cr.SetFixed(True)
                cr.GetVisualShape(0).SetMaterial(0, leaf_material(rng)); sys_.Add(cr)

    # entrance label (farm rule): both outermost image columns at the horizon hit a canopy of rows 0 / 1
    hfac = HFOV / math.pi * 2
    cz = R_W + 0.09 + CAM_Z - 0.03
    trees = [(float(x), yr, r_) for (x, yr, r_) in canopies if yr in (0.0, row_gap)]
    def edge_hits(xc, side):
        dy = hfac if side == "left" else -hfac
        for (tx, ty, r_) in trees:
            if (side == "left") != (ty > yc):
                continue
            zc = 0.85 * r_
            if abs(cz - zc) >= r_:
                continue
            rs = math.sqrt(r_ ** 2 - (cz - zc) ** 2)            # canopy slice radius at camera height
            # distance from the canopy centre to the ray (xc, yc) + t (1, dy), t > 0
            vx, vy = tx - xc, ty - yc
            t = (vx + vy * dy) / (1 + dy * dy)
            if t > 0 and math.hypot(vx - t, vy - t * dy) <= rs:
                return True
        return False
    x_entr = next(xc for xc in np.arange(-15.0, 15.0, 0.02) if edge_hits(xc, "left") and edge_hits(xc, "right"))

    dep = mgr = cam = None
    need_rgb = a.save_images or (a.decider == "vlm" and not a.fake_vlm)
    if (a.depth == "camera" and a.decider == "depth") or need_rgb:
        import pychrono.sensor as sens
        mgr = sens.ChSensorManager(sys_)
        mgr.scene.AddDirectionalLight(chrono.ChColor(1, 1, 1), 50 * chrono.CH_DEG_TO_RAD, 30 * chrono.CH_DEG_TO_RAD)
        mgr.scene.SetAmbientLight(chrono.ChVector3f(0.3, 0.3, 0.3))
        bg = sens.Background(); bg.mode = sens.BackgroundMode_SOLID_COLOR; bg.color_zenith = chrono.ChVector3f(0.55, 0.7, 0.9)
        mgr.scene.SetBackground(bg)
        pose = chrono.ChFramed(chrono.ChVector3d(CAM_X, 0, CAM_Z), chrono.QUNIT)
        if a.depth == "camera" and a.decider == "depth":
            dep = sens.ChDepthCamera(chassis, RATE, pose, W_IMG, H_IMG, HFOV); dep.SetMaxDepth(20.0); mgr.AddSensor(dep)
        if need_rgb:
            cam = sens.ChCameraSensor(chassis, RATE, pose, W_IMG, H_IMG, HFOV)
            cam.PushFilter(sens.ChFilterRGBA8Access())
            if a.save_images:
                cam.PushFilter(sens.ChFilterSave(os.path.join(a.out, f"trial{k:02d}") + "/"))
            mgr.AddSensor(cam)

    def cam_pose():
        rot = chassis.GetRot()
        return chassis.GetPos() + rot.Rotate(chrono.ChVector3d(CAM_X, 0, CAM_Z)), rot

    def depth_ray():
        o, rot = cam_pose()
        res = []
        for side in ("left", "right"):
            ds = []
            for d in RAYS[side]:
                dw = rot.Rotate(chrono.ChVector3d(*d))            # world direction (not unit; x-component = 1 in camera frame)
                n = math.sqrt(dw.x ** 2 + dw.y ** 2 + dw.z ** 2)
                best = 20.0
                if dw.z < -1e-6:                                    # ground plane z = 0
                    best = min(best, -o.z / dw.z)                   # parameter along d = z-depth (camera x = 1 per unit)
                r = chrono.ChRayhitResult()
                end = o + dw * (20.0)
                if sys_.GetCollisionSystem().RayHit(o, end, r) and r.hit:
                    hp = r.abs_hitPoint
                    best = min(best, ((hp.x - o.x) * dw.x + (hp.y - o.y) * dw.y + (hp.z - o.z) * dw.z) / (n * n))
                if best < 20.0:                                     # beyond 20 m / sky = invalid, as on the farm
                    ds.append(best)
            res.append(float(np.median(ds)) if ds else float("nan"))
        return res

    last_stamp = [-1.0]
    zfac = None

    def depth_cam(min_t=None):
        """Chrono's depth camera returns the distance along each ray; the ZED (and our farm rule) uses z-depth."""
        nonlocal zfac
        buf = dep.GetMostRecentDepthBuffer()
        if not buf.HasData() or buf.TimeStamp <= last_stamp[0] or buf.TimeStamp < (SETTLE if min_t is None else min_t):
            return None                                             # no new frame since the last check
        last_stamp[0] = buf.TimeStamp
        d = np.asarray(buf.GetDepthData(), dtype=np.float64).reshape(buf.Height, buf.Width)
        if zfac is None:
            hh, ww = d.shape; h = HFOV / math.pi * 2
            dx = (np.arange(ww) + 0.5) / ww * 2 - 1; dy = ((np.arange(hh) + 0.5) / hh * 2 - 1) * hh / ww
            zfac = 1.0 / np.sqrt(1 + (dx[None, :] * h) ** 2 + (dy[:, None] * h) ** 2)
        if a.log_depth:
            np.save(os.path.join(a.out, f"depth_trial{k:02d}", f"d_{buf.TimeStamp:07.2f}.npy"), d.astype(np.float16))
        d = np.where(d >= 19.99, np.nan, d) * zfac
        h, w = d.shape; band = d[int(0.25 * h):int(0.75 * h)]; e = int(0.15 * w)
        out = []
        for b in (band[:, :e], band[:, w - e:]):
            ok = b[np.isfinite(b) & (b > 0.1) & (b < 20)]
            out.append(float(np.median(ok)) if ok.size else float("nan"))
        return out

    last_rgb = [-1.0]
    qdir = os.path.join(a.out, f"queries_trial{k:02d}")
    if a.decider == "vlm" and a.save_images:
        os.makedirs(qdir, exist_ok=True)

    def vlm_check(min_t):
        """Newest camera frame taken at or after min_t -> VLM answer. Returns (yes, latency_s, text) or None."""
        if a.fake_vlm:
            lm, rm = depth_ray(); time.sleep(0)
            return (lm < a.D and rm < a.D), 2.0, "fake"
        buf = cam.GetMostRecentRGBA8Buffer()
        if not buf.HasData() or buf.TimeStamp <= last_rgb[0] or buf.TimeStamp < min_t:
            return None
        last_rgb[0] = buf.TimeStamp
        img = np.asarray(buf.GetRGBA8Data(), dtype=np.uint8).reshape(buf.Height, buf.Width, 4)[::-1, :, :3]   # bottom-up -> top-down
        from PIL import Image
        import io
        import ollama
        bio = io.BytesIO(); Image.fromarray(img).save(bio, format="JPEG", quality=92)
        if a.save_images:
            Image.fromarray(img).save(os.path.join(qdir, f"q_{buf.TimeStamp:07.2f}.jpg"), quality=92)
        t0 = time.time()
        r = ollama.chat(model=a.model, messages=[{"role": "user", "content": QUESTION.format(c=WORDING[a.wording]), "images": [bio.getvalue()]}],
                        options={"temperature": 0, "seed": 42})
        lat = time.time() - t0
        import re
        txt = (r["message"]["content"] or "")
        m = re.findall(r"\b(yes|no)\b", txt.lower())
        return (bool(m) and m[-1] == "yes"), lat, (m[-1].upper() if m else "?")

    def check(min_t):
        """One stop check. Returns (yes, latency_s, answer_text) or None if no new frame yet."""
        if a.decider == "vlm":
            return vlm_check(min_t)
        got = depth_ray() if a.depth == "ray" else depth_cam(min_t=min_t)
        if got is None:
            return None
        return (got[0] < a.D and got[1] < a.D), 0.0, f"{got[0]:.2f}/{got[1]:.2f}"

    def advance(seconds, driving):
        """Let simulated time pass (the robot keeps its current wheel commands), e.g. while the VLM is thinking."""
        t_stop = sys_.GetChTime() + seconds
        while sys_.GetChTime() < t_stop:
            if driving:
                set_wheels(a.speed, steer())
            if mgr is not None:
                mgr.Update()
            sys_.DoStepDynamics(DT)

    def set_wheels(v, wz):
        wl, wr = (v - wz * B / 2) / R_W, (v + wz * B / 2) / R_W
        for i, f in enumerate(funcs):
            f.SetConstant(wl if i < 2 else wr)

    # simple heading controller to drive straight along the row centre (y = yc)
    def steer():
        pos = chassis.GetPos(); yaw = chassis.GetRot().GetCardanAnglesZYX().z
        return -1.5 * (pos.y - yc) - 2.0 * yaw

    t_end = 60.0
    streak, first_yes_x, stopped = 0, None, False
    target_x = None                     # step6: next stop position (camera x)
    phase = "settle"; wait_until = 1.0
    next_eval = 1.0
    n_asks = 0
    halt_still = None
    clog_fh = open(os.path.join(a.out, f"checks_trial{k:02d}.csv"), "w", newline="")
    clog = csv.writer(clog_fh); clog.writerow(["t_photo", "x_rel_entrance", "answer", "yes", "latency_s"])
    wait_total = 0.0
    dlog = None
    if a.log_depth:
        dfh = open(os.path.join(a.out, f"depth_trial{k:02d}.csv"), "w", newline="")
        dlog = csv.writer(dfh); dlog.writerow(["t", "x_rel_entrance", "left_m", "right_m", "ray_left_m", "ray_right_m", "yes"])
        os.makedirs(os.path.join(a.out, f"depth_trial{k:02d}"), exist_ok=True)
    while sys_.GetChTime() < t_end:
        t = sys_.GetChTime()
        o, _ = cam_pose()
        if a.policy == "drive":
            if phase == "settle" and t >= SETTLE:
                phase = "go"; next_eval = t
            if phase == "go":
                set_wheels(a.speed, steer())
                if t >= next_eval:
                    got = check(SETTLE)
                    if got is None:                                  # camera: no new frame yet, try next step
                        next_eval = t + DT
                    else:
                        yes, lat, txt = got
                        n_asks += 1
                        clog.writerow([round(t, 2), round(o.x - x_entr, 2), txt, int(yes), round(lat, 2)])
                        if lat > 0:                                  # the robot keeps driving while the model thinks
                            advance(lat, driving=True)
                        next_eval = sys_.GetChTime() + (1.0 / RATE if lat == 0 else 0.0)
                        streak = streak + 1 if yes else 0
                        if streak >= a.K:
                            phase = "brake"; set_wheels(0, 0); brake_t = sys_.GetChTime()
            elif phase == "brake":
                set_wheels(0, 0)
                v = chassis.GetPosDt()
                if math.hypot(v.x, v.y) < 0.02 and t - brake_t > 0.5:
                    stopped = True; break
        else:   # step6
            if phase == "settle" and t >= SETTLE:
                phase, target_x = "move", o.x + 1.0
            if phase == "move":
                set_wheels(min(a.speed, 0.6 + 2 * max(target_x - o.x, 0)), steer())
                if o.x >= target_x:
                    phase = "halt"; set_wheels(0, 0); halt_t = t
            elif phase == "halt":
                set_wheels(0, 0)
                v = chassis.GetPosDt()
                got = None
                if math.hypot(v.x, v.y) < 0.02:
                    if halt_still is None:
                        halt_still = t
                    got = check(halt_still + 1e-3)                    # camera: a frame taken while standing
                if got is not None:
                    halt_still = None
                    yes, lat, txt = got
                    wait_total += lat                                 # standing still: thinking time costs time, not metres
                    clog.writerow([round(t, 2), round(o.x - x_entr, 2), txt, int(yes), round(lat, 2)])
                    n_asks += 1
                    streak = streak + 1 if yes else 0
                    if yes and first_yes_x is None:
                        first_yes_x = o.x
                    if streak >= 6:
                        stopped = True; break
                    phase = "move"; target_x = o.x + (0.25 if first_yes_x is not None else 1.0)
        if o.x - x_entr > 12.0:                                    # far past the entrance: give up
            break
        if mgr is not None:
            mgr.Update()
        sys_.DoStepDynamics(DT)
    clog_fh.close()
    o, _ = cam_pose()
    return dict(trial=k, soil=soil, row_gap=round(row_gap, 2), tree_gap=round(tree_gap, 2), crown_r=round(crown_r, 2),
                y_off=round(y_off, 2), yaw0=round(yaw0, 3), stopped=int(stopped), err_m=round(o.x - x_entr, 2) if stopped else "", entrance_x=round(x_entr, 2),
                sim_s=round(sys_.GetChTime(), 1), mission_s=round(sys_.GetChTime() + wait_total, 1), asks=n_asks,
                scenario=a.scenario, decider=a.decider + (f":{a.model}:{a.wording}" if a.decider == 'vlm' else ''), policy=a.policy)


rng = np.random.default_rng(a.seed)
rows = []
t0 = time.time()
for k in range(a.trials):
    r = run_trial(k, rng)
    rows.append(r)
    print(f"trial {k:2d} {r['soil']:<5} rows {r['row_gap']:.1f} m apart: " +
          (f"stopped at {r['err_m']:+.2f} m" if r["stopped"] else "did not stop") + f"  ({r['mission_s']} s, {r['asks']} checks)")
with open(os.path.join(a.out, "trials.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if r["stopped"] and abs(r["err_m"]) < 2]
errs = [r["err_m"] for r in rows if r["stopped"]]
early = sum(1 for r in rows if r["stopped"] and r["err_m"] <= -2)
print(f"\nscenario {a.scenario}, decider {a.decider}" + (f" ({a.model}, wording {a.wording})" if a.decider == "vlm" else f" ({a.depth}, D = {a.D} m)")
      + f", policy {a.policy}: {len(ok)}/{len(rows)} within 2 m, {early} early"
      + (f", mean error {np.mean(errs):+.2f} m" if errs else "") + f", {sum(1 - r['stopped'] for r in rows)} no stop"
      + f", mean mission time {np.mean([r['mission_s'] for r in rows]):.0f} s"
      + f"   [{time.time() - t0:.0f} s wall]")
