# eval_v3.py  -  depth v3 on recorded drive runs, frame by frame against the ground-truth labels
#   python eval_v3.py run_field_drive
import sys, glob, os, numpy as np
from depth_v3 import DepthV3
d_hazard, d_warn = 0.3 * 1.0 + 1.0 / 2 + 0.5, 0.3 * 1.0 + 1.0 / 2 + 0.5 + 3.0      # drive runs: constant 1 m/s
v3 = DepthV3()
LAB = ["go", "slow", "stop", "detour"]
tot = np.zeros((4, 4), int)
print(f"zones: d_hazard {d_hazard:.2f} m, d_warn {d_warn:.2f} m\n")
print(f"{'scenario':14s} agree  first non-go (gt / v3)   first stop|detour (gt / v3)          false alarms")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_depth.npz"))):
    sc = os.path.basename(f).replace("_00_depth.npz", "")
    r = np.load(f); D, gt, gap = r["depth"], r["label"], r["gap"]
    pred, dist = [], []
    for k in range(len(D)):
        lab, d, kind = v3.decide(D[k], d_hazard, d_warn); pred.append(lab); dist.append(d)
    pred = np.array(pred)
    for g, p_ in zip(gt, pred): tot[LAB.index(g), LAB.index(p_)] += 1
    agree = (pred == gt).mean()
    def first(arr, cond):
        i = np.where(cond(arr))[0]; return i[0] if len(i) else None
    # distance of the true obstacle at a frame: gap (nan when nothing blocks, e.g. none/grass)
    t = np.arange(len(D)) * 0.1
    def at(i, arr):
        if i is None: return "never"
        g = gap[i]
        return f"{'' if np.isnan(g) else f'{g:5.2f} m '}t={t[i]:5.1f}s {arr[i]}"
    g1, p1 = first(gt, lambda a: a != "go"), first(pred, lambda a: a != "go")
    g2, p2 = first(gt, lambda a: np.isin(a, ["stop", "detour"])), first(pred, lambda a: np.isin(a, ["stop", "detour"]))
    fa = int(((gt == "go") & (pred != "go")).sum())
    print(f"{sc:14s} {agree:5.0%}  {at(g1, gt):>18s} / {at(p1, pred):<22s} {at(g2, gt):>20s} / {at(p2, pred):<24s} {fa:4d}")
print("\nconfusion (rows = ground truth, columns = v3):")
print("          " + "".join(f"{l:>8s}" for l in LAB))
for i, l in enumerate(LAB): print(f"{l:>8s}  " + "".join(f"{n:8d}" for n in tot[i]))
print(f"frame agreement over all scenarios: {np.trace(tot) / tot.sum():.1%}")
