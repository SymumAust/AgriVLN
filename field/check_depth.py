# check_depth.py  -  is Unity's depth grid what flat-ground geometry predicts?  (run after the "drive" run)
#   python check_depth.py run_field_drive/none_00_depth.npz
import sys, numpy as np
d = np.load(sys.argv[1])["depth"][0].astype(float)          # first frame, rows x cols, row 0 = top
H, W = d.shape
h, hfov = 0.28, 102.0                                         # camera height and field of view in FieldWorld (tilt 0)
tv = np.tan(np.radians(hfov / 2)) * 360 / 640
print(" row   Unity (centre col)   flat ground expects")
for r in (19, 22, 25, 28, 31, 35):
    ndc = 1 - 2 * (r + 0.5) / H
    exp = h / (-ndc * tv) if ndc < 0 else 0.0
    print(f"{r:4d}   {d[r, W // 2]:10.3f} m        {exp:10.3f} m")
print("If the two columns match (within about 0.01 m), Unity's depth grid and our geometry agree.")
