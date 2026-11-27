#!/bin/bash
# cam_sweep_v4.sh - depth v3 and depth v4 (memory), 5 camera mountings, 8 scenes x 3 trials, detours really driven
set -e
for pol in depth_v3 depth_v4; do
  for cfg in "0.28 0" "0.6 0" "1.0 0" "0.6 15" "1.0 15"; do
    set -- $cfg
    echo "=== $pol  camera height $1 m, tilt $2 deg ==="
    python field_client.py --policy $pol --drive_detour --scenarios all --trials 3 --cam_h $1 --tilt $2 --out run_cam4_${pol}_h$1_t$2 | tail -2
  done
done
python cam_summary.py run_cam4_*
