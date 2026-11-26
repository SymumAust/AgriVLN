# depth_v3.py  -  go / slow / stop / detour from the depth grid alone (no ground truth, no VLM)
#
# Idea (traversability with a flat-ground model, the simplest version that a real stereo camera can also run):
#   every depth cell -> a 3D point: forward z, lateral X (right +), height above the ground H (camera height + up offset)
#   corridor = |X| < robot half width + margin (the strip the robot body will sweep)
#   OBSTACLE cell: in the corridor and H > h_obs (default 0.15 m): something sticks up out of the ground
#   HOLE cell:     in the corridor and H < -h_hole (default 0.05 m): the ray went BELOW the ground = ditch / step down;
#                  its distance is where the ray would have met flat ground (= just past the hole's near edge)
#   nearest obstacle/hole distance d (from the front bumper), zones as in field_client.py:
#     d <= d_hazard: hole -> STOP; obstacle -> DETOUR if a free strip >= robot width + 2 margins is SEEN beside it, else STOP
#     d <= d_warn  : SLOW          otherwise GO
# What depth cannot know: WHAT the object is. A person and a drum look the same (both: obstacle), and tall grass is also
# "something sticking up". Those are the cases where step 3 (VLM + uncertainty) has to help.

import numpy as np

class DepthV3:
    def __init__(self, cam_height=0.28, hfov_deg=102.0, img_w=640, img_h=360, grid_w=64, grid_h=36, tilt_deg=0.0,
                 cam_ahead_of_front=-0.015, half_w=0.215, margin=0.15, h_obs=0.15, h_hole=0.05, min_cells=2, max_range=20.0):
        th = np.tan(np.radians(hfov_deg / 2)); tv = th * img_h / img_w
        self.ndx = ((2 * (np.arange(grid_w) + .5) / grid_w - 1) * th)[None, :]
        self.ndy = ((1 - 2 * (np.arange(grid_h) + .5) / grid_h) * tv)[:, None]
        if abs(tilt_deg) > 1e-6:
            raise NotImplementedError("tilt: rotate the points first (camera tilt is 0 in FieldWorld for now)")
        self.h, self.th, self.cam_front = cam_height, th, cam_ahead_of_front
        self.corr, self.need = half_w + margin, 2 * (half_w + margin)
        self.h_obs, self.h_hole, self.min_cells, self.max_range = h_obs, h_hole, min_cells, max_range

    def points(self, depth):
        z = depth.astype(float)
        X = z * self.ndx
        H = self.h + z * self.ndy
        return z, X, H

    def nearest(self, depth):
        """(distance from the front bumper, 'obstacle'|'hole'|None, z, X, H)"""
        z, X, H = self.points(depth)
        ok = (z > 0) & (z < self.max_range)
        corr = ok & (np.abs(X) < self.corr)
        obs = corr & (H > self.h_obs)
        hole = corr & (H < -self.h_hole)
        best = (np.inf, None)
        if obs.sum() >= self.min_cells:
            best = (np.sort(z[obs])[self.min_cells - 1] + self.cam_front, "obstacle")
        if hole.sum() >= self.min_cells:
            z_ground = (self.h / np.maximum(-self.ndy, 1e-9)) * np.ones_like(z)        # where flat ground would have been
            dh = np.sort(z_ground[hole])[self.min_cells - 1] + self.cam_front
            if dh < best[0]: best = (dh, "hole")
        return best[0], best[1], z, X, H

    def side_space(self, d, z, X, H):
        """largest free strip SEEN beside the obstacle at distance d (m). Out of view counts as not free."""
        win = (z > 0) & (H > self.h_obs) & (z + self.cam_front >= d - 0.2) & (z + self.cam_front <= d + 1.0)
        edges = np.arange(-3.0, 3.0001, 0.05)
        occ = np.zeros(len(edges) - 1, bool)
        if win.any():
            idx = np.clip(np.digitize(X[win], edges) - 1, 0, len(occ) - 1); occ[idx] = True
        centres = (edges[:-1] + edges[1:]) / 2
        seen = np.abs(centres) <= max(d - self.cam_front, 0.05) * self.th                   # inside the field of view at that distance
        blocked = occ | ~seen
        inner = np.where(occ & (np.abs(centres) < self.corr))[0]
        if len(inner) == 0: return 0.0
        lo, hi = inner.min(), inner.max()
        # grow the obstacle to its connected occupied bins, then measure free bins outward
        while lo > 0 and occ[lo - 1]: lo -= 1
        while hi < len(occ) - 1 and occ[hi + 1]: hi += 1
        left = 0
        i = lo - 1
        while i >= 0 and not blocked[i]: left += 1; i -= 1
        right = 0
        i = hi + 1
        while i < len(occ) and not blocked[i]: right += 1; i += 1
        return 0.05 * max(left, right)

    def decide(self, depth, d_hazard, d_warn):
        d, kind, z, X, H = self.nearest(depth)
        if kind is None or d > d_warn:
            return "go", d, kind
        if d > d_hazard:
            return "slow", d, kind
        if kind == "hole":
            return "stop", d, kind
        return ("detour" if self.side_space(d, z, X, H) >= self.need else "stop"), d, kind
