# depth_v5.py  -  depth v4 (memory) + two safety layers that need no VLM   (step C, layers 1 and 2)
#
# Layer 1 - MOTION: the nearest object near the path (current frame, |X| < 1.565 m, up to d_warn + 3 m ahead) is tracked
#   by the world x of its points (lateral position). A straight-line fit over the last 1 s (at least 6 frames) gives its
#   sideways speed. Faster than 0.6 m/s -> "moving" (remembered for 3 s). A moving object is never detoured around.
#   (Only sideways motion is measured; an object coming straight at the robot is not covered here.)
# Layer 2 - POSSIBLE PERSON: the map remembers the highest point seen in every cell. If the nearest object is 1.2-2.2 m
#   tall and at most 0.9 m wide, it may be a person -> never detoured around.
# In both cases DETOUR becomes STOP (inside d_hazard) or SLOW (inside d_warn). Everything else is v4.
# Thresholds were set on the step-1 drive recordings (practice data): sideways speed of static objects 0.00 m/s,
# tall grass up to 0.42 m/s, the walking person 0.98 m/s. A lying person (0.4 m high, 1.7 m wide) is NOT caught by
# either layer - that needs the class (VLM, layer 3).

import numpy as np
from depth_v4 import DepthV4, OBST


class DepthV5(DepthV4):
    def __init__(self, v_move=0.6, window=1.0, min_frames=6, hold=3.0, near_x=1.565, h_person=(1.2, 2.2), w_person=0.9, **kw):
        super().__init__(**kw)
        self.v_move, self.window, self.min_frames, self.hold = v_move, window, min_frames, hold
        self.near_x, self.h_person, self.w_person = near_x, h_person, w_person
        self.hmax, self.track, self.moving_until = {}, [], -1.0
        self.last = {}

    def reset(self):
        super().reset(); self.hmax, self.track, self.moving_until = {}, [], -1.0

    def observe(self, depth, pose, t, d_warn):
        """update the map, the per-cell height memory and the motion track"""
        self.update(depth, pose)
        F, X, H = self.points(depth)
        ok = (F > 0) & (F < self.max_range) & (H > self.h_obs)
        wx, wz = self._world(F[ok], X[ok], pose)
        for kx, kz, h in zip(np.floor(wx / self.cell).astype(int), np.floor(wz / self.cell).astype(int), H[ok]):
            k = (kx, kz)
            if self.hmax.get(k, 0.0) < h: self.hmax[k] = h
        for k in [k for k in self.hmax if self.map.get(k) != OBST]: del self.hmax[k]   # cleared or forgotten cells
        x, z, yaw = pose
        s, c = np.sin(yaw), np.cos(yaw)
        Fr = (wz - z) * c + (wx - x) * s - self.front
        Xr = (wx - x) * c - (wz - z) * s
        sel = (np.abs(Xr) < self.near_x) & (Fr >= 0) & (Fr <= d_warn + 3.0)
        if sel.sum() >= 3:
            near = sel & (Fr <= Fr[sel].min() + 1.0)
            self.track.append((t, float(wx[near].mean())))
        self.track = [p for p in self.track if p[0] >= t - self.window - 1e-9]
        speed = 0.0
        if len(self.track) >= self.min_frames:
            tt, xx = np.array(self.track).T
            if tt.max() - tt.min() > 0.3: speed = abs(np.polyfit(tt, xx, 1)[0])
        if speed > self.v_move: self.moving_until = t + self.hold
        self.last = {"speed": speed, "moving": t <= self.moving_until}

    def nearest_object_shape(self, pose):
        """(height, width) of the nearest remembered obstacle near the path"""
        Fr, Xr, code = self._robot_frame(pose)
        if len(code) == 0: return 0.0, 0.0
        keys = list(self.map.keys())
        sel = (code == OBST) & (np.abs(Xr) < self.near_x) & (Fr >= -0.05)
        if not sel.any(): return 0.0, 0.0
        near = sel & (Fr <= Fr[sel].min() + 1.0)
        idx = np.where(near)[0]
        h = max(self.hmax.get(keys[i], 0.0) for i in idx)
        w = float(Xr[near].max() - Xr[near].min() + self.cell)
        return h, w

    def decide_v5(self, pose, d_hazard, d_warn):
        lab, d, kind = self.decide_pose(pose, d_hazard, d_warn)
        h, w = self.nearest_object_shape(pose)
        person_like = self.h_person[0] <= h <= self.h_person[1] and w <= self.w_person
        self.last.update({"height": h, "width": w, "person_like": person_like})
        why = ""
        if lab == "detour" and (self.last.get("moving") or person_like):
            why = "moving" if self.last.get("moving") else "person-shaped"
            lab = "stop" if d <= d_hazard else "slow"
        self.last["why"] = why
        return lab, d, kind
