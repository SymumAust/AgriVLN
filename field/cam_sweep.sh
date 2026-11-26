#!/bin/bash
# cam_sweep.sh - depth v3 with 5 camera mountings, all 8 scenarios x 3 trials each (Unity Field scene must be in Play)
set -e
for cfg in "0.28 0" "0.6 0" "1.0 0" "0.6 15" "1.0 15"; do
  set -- $cfg
  echo "=== camera height $1 m, tilt $2 deg ==="
  python field_client.py --policy depth_v3 --scenarios all --trials 3 --cam_h $1 --tilt $2 --out run_cam_h$1_t$2 | tail -2
done
python cam_summary.py run_cam_h*
