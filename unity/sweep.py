# sweep.py  -  "where does each decider start saying YES?"  (Unity, no stopping)
#
#   python sweep.py --model qwen3-vl:4b --wording A --out sweep_4b_A
#
# The robot drives straight down the lane centre from 20 m BEFORE the entrance to 4 m AFTER it,
# stands still every 1 m, takes one camera image and asks:
#   (1) the VLM (same prompt and wording as loop_client.py)   and   (2) the depth rule (D = 3.5 m).
# Nothing stops the robot. Output: <out>/sweep.csv  (rel_to_entrance, vlm answer, latency, left_m, right_m, depth answer)
# and every image in <out>/ named by distance and answers.  This is an open-loop probe INSIDE the simulator:
# it measures judgement only (where YES starts), not timing.

import argparse, csv, os, re, time
import sys
sys.argv_saved = sys.argv
p = argparse.ArgumentParser()
p.add_argument("--model", default="qwen3-vl:4b")
p.add_argument("--wording", default="A", choices=["A", "C"])
p.add_argument("--D", type=float, default=3.5)
p.add_argument("--from_m", type=float, default=-20.0)
p.add_argument("--to_m", type=float, default=4.0)
p.add_argument("--every", type=float, default=1.0)
p.add_argument("--out", default="sweep_out")
p.add_argument("--port", type=int, default=5555)
p.add_argument("--no_vlm", action="store_true", help="depth only (fast)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

# reuse the connection class and the prompt from loop_client.py without running its experiment
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "loop_client.py")).read()
ns = {}
exec(src.split("class Unity:")[0].split("p = argparse.ArgumentParser()")[0], ns)         # imports
exec("WORDING" + src.split("WORDING", 1)[1].split("class Unity:")[0], ns)                # WORDING, QUESTION
exec("class Unity:" + src.split("class Unity:")[1].split("def drive_for")[0], ns)        # Unity
WORDING, QUESTION, Unity = ns["WORDING"], ns["QUESTION"], ns["Unity"]

if not a.no_vlm:
    import ollama


def ask(img):
    t0 = time.time()
    r = ollama.chat(model=a.model, messages=[{"role": "user", "content": QUESTION.format(c=WORDING[a.wording]), "images": [img]}],
                    options={"temperature": 0, "seed": 42})
    lat = time.time() - t0
    m = re.findall(r"\b(yes|no)\b", (r["message"]["content"] or "").lower())
    return (m[-1].upper() if m else "?"), lat


u = Unity(a.port)
st, _ = u.reset(0.0, -30.0, 0.0)
entr = st["entrance_z"]
cam_off = st["cam_z"] - st["z"]
rows = []
d = a.from_m
while d <= a.to_m + 1e-6:
    st, img = u.reset(0.0, entr + d - cam_off, 0.0)          # place the robot exactly; reset returns no image
    st, img = u.step(0.0, 0.0, 0.0, image=True)
    rel = st["cam_z"] - entr
    ans, lat = ("-", 0.0) if a.no_vlm else ask(img)
    l, r = st["left_m"], st["right_m"]
    dep = "YES" if (l is not None and r is not None and l < a.D and r < a.D) else "NO"
    lh, rh = st.get("left_hi"), st.get("right_hi")
    dhi = "YES" if (lh is not None and rh is not None and lh < a.D and rh < a.D) else "NO"
    rows.append(dict(rel_to_entrance=round(rel, 2), vlm=ans, latency_s=round(lat, 2), left_m=l, right_m=r, depth=dep,
                     left_hi=lh, right_hi=rh, depth_hi=dhi))
    open(os.path.join(a.out, f"d{rel:+06.1f}_vlm{ans}_depth{dep}.jpg"), "wb").write(img)
    print(f"{rel:+6.1f} m   VLM {ans:3s} ({lat:4.1f} s)   depth {dep:3s} (L {l}, R {r})   depth_hi {dhi:3s} (L {lh}, R {rh})", flush=True)
    d += a.every

with open(os.path.join(a.out, "sweep.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
fy = lambda k: next((r["rel_to_entrance"] for r in rows if r[k] == "YES"), None)
print(f"\nfirst VLM YES at {fy('vlm')} m, first depth YES at {fy('depth')} m, first depth_hi YES at {fy('depth_hi')} m   (0 = entrance; negative = before)")
print(f"VLM YES before the entrance: {sum(r['vlm']=='YES' and r['rel_to_entrance']<-2 for r in rows)} of "
      f"{sum(r['rel_to_entrance']<-2 for r in rows)} positions more than 2 m before it  -> {a.out}/sweep.csv")
