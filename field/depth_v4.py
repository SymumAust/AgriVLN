# depth_v4.py  -  depth decider WITH MEMORY: a small local map that remembers what the camera has seen
#
# What changes from v3 (depth_v3.py), and why (step 4 lessons):
#   1. MEMORY. Every frame, depth cells are put into a 0.1 m grid in world coordinates, using the robot pose
#      (here Unity's pose = perfect odometry; on the real robot: wheel odometry + IMU / RTK).
#        obstacle cell : something sticks up (H > 0.15 m)
#        hole cell     : the ray went below the ground (H < -0.05 m)
#        free cell     : ground seen (|H| <= 0.05 m)
#      A cell seen as ground later clears an old mark (an object that moved away, or a hole mark that was too careful).
#      Within one frame obstacle and hole marks win over ground. What is no longer in view keeps its last mark.
#      Decisions use the map, so an obstacle or ditch that slips into the camera's near blind zone is not forgotten.
#   2. EARLY DETOUR. As soon as an obstacle is in the warning zone, the free strip beside it is measured in the map
#      (where the side was seen from further away). Free strip >= robot width + 2 margins (0.73 m) -> DETOUR now.
#   3. CAREFUL HOLE DISTANCE. A hole is marked at the ground point of the next closer ray (the near edge lies between
#      that ray and the one that went into the hole), not where the diving ray would have met the ground (past the edge).
#      Far away the rays are sparse, so this is early; it gets tighter as the robot comes closer.
# Same zones as before: hole/obstacle within d_hazard -> stop (or detour, see 2); within d_warn -> slow; else go.
# What v4 still cannot know: WHAT the object is (person vs drum, grass vs rock). That is step C.

import numpy as np
from depth_v3 import DepthV3

FREE, OBST, HOLE = 0, 1, 2


class DepthV4(DepthV3):
    def __init__(self, cell=0.1, keep=15.0, cam_ahead_of_centre=0.2, front_offset=0.215, **kw):
        super().__init__(**kw)
        self.cell, self.keep, self.cam_ahead, self.front = cell, keep, cam_ahead_of_centre, front_offset
        H, W = self.ndy.shape[0], self.ndx.shape[1]
        # ground point of every ray (where it would meet flat ground), in the camera's forward / lateral frame
        fg = np.broadcast_to(self.f_ground, (H, W))
        self.Fg = np.where(np.isfinite(fg), fg, np.nan)
        self.Xg = self.Fg / self.f_per_z * self.ndx
        self.map = {}

    def reset(self):
        self.map = {}

    def _world(self, F, X, pose):
        x, z, yaw = pose
        s, c = np.sin(yaw), np.cos(yaw)                       # Unity: yaw > 0 = turned right; forward = (sin, cos)
        cx, cz = x + self.cam_ahead * s, z + self.cam_ahead * c
        return cx + F * s + X * c, cz + F * c - X * s

    def update(self, depth, pose):
        F, X, H = self.points(depth)
        ok = (F > 0) & (F < self.max_range)
        free = ok & (np.abs(H) <= self.h_hole)
        obst = ok & (H > self.h_obs)
        hole = ok & (H < -self.h_hole)
        # hole -> the ground point of the next closer ray in the same column (row below), else its own ground point
        hj, hi = np.where(hole)
        jn = np.minimum(hj + 1, self.Fg.shape[0] - 1)
        hF = np.where(np.isfinite(self.Fg[jn, hi]), self.Fg[jn, hi], self.Fg[hj, hi])
        hX = np.where(np.isfinite(self.Fg[jn, hi]), self.Xg[jn, hi], self.Xg[hj, hi])
        good = np.isfinite(hF)
        m = self.map
        for code, (fx, fz) in ((FREE, self._world(F[free], X[free], pose)),
                               (OBST, self._world(F[obst], X[obst], pose)),
                               (HOLE, self._world(hF[good], hX[good], pose))):
            keys = zip(np.floor(fx / self.cell).astype(int), np.floor(fz / self.cell).astype(int))
            for k in keys:
                m[k] = code                                    # later in the same frame wins: free < obstacle < hole
        # forget what is far away
        x, z, _ = pose
        r2 = self.keep ** 2
        for k in [k for k in m if ((k[0] + .5) * self.cell - x) ** 2 + ((k[1] + .5) * self.cell - z) ** 2 > r2]:
            del m[k]

    def _robot_frame(self, pose):
        if not self.map: return np.zeros(0), np.zeros(0), np.zeros(0, int)
        x, z, yaw = pose
        k = np.array(list(self.map.keys()), float)
        code = np.array(list(self.map.values()), int)
        wx, wz = (k[:, 0] + .5) * self.cell, (k[:, 1] + .5) * self.cell
        s, c = np.sin(yaw), np.cos(yaw)
        Fr = (wz - z) * c + (wx - x) * s - self.front         # ahead of the front bumper
        Xr = (wx - x) * c - (wz - z) * s                      # to the right
        return Fr, Xr, code

    def side_space(self, d, Fr, Xr, code):
        win = (Fr >= d - 0.2) & (Fr <= d + 1.0)
        edges = np.arange(-3.0, 3.0001, self.cell)
        nb = len(edges) - 1
        occ, fre = np.zeros(nb, bool), np.zeros(nb, bool)
        idx = np.clip(np.digitize(Xr[win], edges) - 1, 0, nb - 1)
        occ[idx[code[win] != FREE]] = True
        fre[idx[code[win] == FREE]] = True
        fre &= ~occ
        centres = (edges[:-1] + edges[1:]) / 2
        inner = np.where(occ & (np.abs(centres) < self.corr + self.cell / 2))[0]
        if len(inner) == 0: return 0.0
        lo, hi = inner.min(), inner.max()
        while lo > 0 and occ[lo - 1]: lo -= 1
        while hi < nb - 1 and occ[hi + 1]: hi += 1
        best = 0
        for step, start in ((-1, lo - 1), (1, hi + 1)):
            n, i = 0, start
            while 0 <= i < nb and fre[i]: n += 1; i += step
            best = max(best, n)
        return best * self.cell

    def decide_pose(self, pose, d_hazard, d_warn):
        Fr, Xr, code = self._robot_frame(pose)
        inc = (np.abs(Xr) < self.corr + self.cell / 2) & (Fr >= -0.05)
        d_o = np.sort(Fr[inc & (code == OBST)])
        d_h = np.sort(Fr[inc & (code == HOLE)])
        d_o = d_o[self.min_cells - 1] if len(d_o) >= self.min_cells else np.inf
        d_h = d_h[0] if len(d_h) else np.inf
        d = min(d_o, d_h)
        if not np.isfinite(d) or d > d_warn:
            return "go", d, None
        if d_h <= d_o:
            return ("stop" if d <= d_hazard else "slow"), d, "hole"
        if self.side_space(d_o, Fr, Xr, code) >= self.need:
            return "detour", d, "obstacle"                    # early: decided as soon as it is in the warning zone
        return ("stop" if d <= d_hazard else "slow"), d, "obstacle"
