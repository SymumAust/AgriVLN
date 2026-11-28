# depth_v6.py  -  depth v5 + "grass-like -> crawl through" (instead of a false stop)
#
# Grass-like cue (set on the step-1 drive recordings = practice data): while the nearest obstacle in the robot corridor is
# 1.0 - d_warn ahead, take the depth spread of its points in this frame (90th - 10th percentile of their distance).
# A solid thing has its points on one surface (rock 0.15-0.18 m, drum, people <= 0.14 m as 10-frame medians); a patch of
# grass has blades at many distances (0.35-0.46 m). Rule: median spread of the last 10 such frames >= 0.26 m AND the object
# is at most 0.5 m high AND not moving AND not person-shaped -> "grass-like".
# Then the robot does not stop: it CRAWLS through at --v_crawl (label: slow). The state is kept while the obstacle is close,
# and dropped when nothing is in the corridor ahead for 1 s.
# Known risk, tested on purpose: something low hidden inside the grass (scene grass_person) looks the same to depth.

import numpy as np
from depth_v5 import DepthV5


class DepthV6(DepthV5):
    def __init__(self, spread_min=0.26, h_max=0.5, n_frames=10, clear_time=1.0, **kw):
        super().__init__(**kw)
        self.spread_min, self.h_max, self.n_frames, self.clear_time = spread_min, h_max, n_frames, clear_time
        self.spreads, self.grass, self.last_seen = [], False, -1.0

    def reset(self):
        super().reset(); self.spreads, self.grass, self.last_seen = [], False, -1.0

    def observe(self, depth, pose, t, d_warn):
        super().observe(depth, pose, t, d_warn)
        F, X, H = self.points(depth)
        ok = (F > 0) & (F < self.max_range)
        obs = ok & (np.abs(X) < self.corr) & (H > self.h_obs)
        if obs.sum() >= 2:
            self.last_seen = t
            d = F[obs].min() + self.cam_front
            if 1.0 <= d <= d_warn:
                self.spreads.append((float(np.percentile(F[obs], 90) - np.percentile(F[obs], 10)), float(H[obs].max())))
                self.spreads = self.spreads[-self.n_frames:]
        if t - self.last_seen > self.clear_time:                 # nothing in the corridor for a while: forget
            self.spreads, self.grass = [], False
        if len(self.spreads) >= self.n_frames and not self.grass:
            sp = np.median([s for s, _ in self.spreads]); hm = max(h for _, h in self.spreads)
            if sp >= self.spread_min and hm <= self.h_max and not self.last.get("moving"):
                self.grass = True
        self.last["spread"] = float(np.median([s for s, _ in self.spreads])) if self.spreads else 0.0

    def decide_v6(self, pose, d_hazard, d_warn):
        lab, d, kind = self.decide_v5(pose, d_hazard, d_warn)
        crawl = False
        if self.grass and kind == "obstacle" and not self.last.get("moving") and not self.last.get("person_like"):
            lab, crawl = "slow", True                             # crawl through instead of stop / detour
            self.last["why"] = "grass-like: crawl"
        self.last["crawl"] = crawl
        return lab, d, kind
