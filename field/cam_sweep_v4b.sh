#!/bin/bash
# cam_sweep_v4b.sh - second run of step 5: depth v4 only, contact-based collision while driving a detour
set -e
for cfg in "0.28 0" "0.6 0" "1.0 0" "0.6 15" "1.0 15"; do
  set -- $cfg
  echo "=== depth_v4  camera height $1 m, tilt $2 deg ==="
  python field_client.py --policy depth_v4 --drive_detour --scenarios all --trials 3 --cam_h $1 --tilt $2 --out run_cam5_depth_v4_h$1_t$2 | tail -2
done
python cam_summary.py run_cam5_*
