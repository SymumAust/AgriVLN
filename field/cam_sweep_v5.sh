#!/bin/bash
# cam_sweep_v5.sh - step 6: depth v5 (memory + motion + person shape), 5 camera mountings, detours driven
set -e
for cfg in "0.28 0" "0.6 0" "1.0 0" "0.6 15" "1.0 15"; do
  set -- $cfg
  echo "=== depth_v5  camera height $1 m, tilt $2 deg ==="
  python field_client.py --policy depth_v5 --drive_detour --scenarios all --trials 3 --cam_h $1 --tilt $2 --out run_cam6_depth_v5_h$1_t$2 | tail -2
done
python cam_summary.py run_cam6_*
