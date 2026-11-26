# vlm_select.py  -  choose the class wording (W1 or W2) on the step-1 drive images (practice set), before the closed loop
#   python vlm_select.py run_field_drive --model qwen3-vl:4b
# Frames: every 6th frame with the obstacle 0.5 - 8 m ahead of the front bumper (for none/grass: the same stretch of path).
# Expected class: none NOTHING, grass GRASS, person_* PERSON, rock/drum OBJECT, ditch HOLE.
# Rule (fixed in plan_field_step3.txt): most correct frames in total; tie -> more correct PERSON frames; then the shorter wording.
import argparse, csv, os, sys
from collections import defaultdict
from vlm_class import WORDINGS, ask

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--model", default="qwen3-vl:4b")
p.add_argument("--every", type=int, default=6)
p.add_argument("--obs_z", type=float, default=30.0)
a = p.parse_args()
EXPECT = {"none": "NOTHING", "grass": "GRASS", "person_stand": "PERSON", "person_lie": "PERSON", "person_moving": "PERSON",
          "rock": "OBJECT", "drum": "OBJECT", "ditch": "HOLE"}

frames = []
with open(os.path.join(a.run_dir, "frames.csv")) as fh:
    rows = list(csv.DictReader(fh))
by_sc = defaultdict(list)
for r in rows:
    ahead = a.obs_z - float(r["z_front"])
    if 0.5 <= ahead <= 8.0: by_sc[r["scenario"]].append((r, ahead))
for sc, lst in by_sc.items():
    for r, ahead in lst[::a.every]:
        img = os.path.join(a.run_dir, f"{sc}_00", f"t{float(r['t']):06.2f}_{r['gt_label']}.jpg")
        if os.path.exists(img): frames.append((sc, ahead, img))
print(f"{len(frames)} practice frames from {len(by_sc)} scenarios", flush=True)

out = open(os.path.join(a.run_dir, "vlm_select.csv"), "w", newline="")
w = csv.writer(out); w.writerow(["wording", "scenario", "ahead_m", "expected", "answer", "latency_s"])
score = {}
for wd in WORDINGS:
    ok, ok_person, lat, unk = defaultdict(int), 0, [], 0
    n = defaultdict(int)
    for sc, ahead, img in frames:
        ans, L = ask(a.model, wd, open(img, "rb").read())
        exp = EXPECT[sc]
        w.writerow([wd, sc, f"{ahead:.2f}", exp, ans, f"{L:.2f}"]); out.flush()
        n[sc] += 1; ok[sc] += ans == exp; lat.append(L); unk += ans == "?"
        if exp == "PERSON" and ans == "PERSON": ok_person += 1
    tot = sum(ok.values())
    score[wd] = (tot, ok_person, -len(WORDINGS[wd]))
    lat.sort()
    print(f"\n{wd}: {tot}/{len(frames)} correct, PERSON {ok_person}, no answer {unk}, median {lat[len(lat) // 2]:.1f} s")
    for sc in EXPECT:
        if n[sc]: print(f"   {sc:14s} {ok[sc]:3d}/{n[sc]:<3d} (expected {EXPECT[sc]})")
out.close()
best = max(score, key=lambda k: score[k])
print(f"\nchosen wording: {best}   (details: {a.run_dir}/vlm_select.csv)")
