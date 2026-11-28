#!/bin/bash
# cam_sweep_v6.sh - step 7: depth v6 (grass-like -> crawl), 5 camera mountings, 9 scenes incl. grass_person
set -e
for cfg in "0.28 0" "0.6 0" "1.0 0" "0.6 15" "1.0 15"; do
  set -- $cfg
  echo "=== depth_v6  camera height $1 m, tilt $2 deg ==="
  python field_client.py --policy depth_v6 --drive_detour --scenarios all --trials 3 --cam_h $1 --tilt $2 --out run_cam7_depth_v6_h$1_t$2 | tail -2
done
echo "=== reference: depth_v5 on grass and grass_person, camera 0.28 m ==="
python field_client.py --policy depth_v5 --drive_detour --scenarios grass,grass_person --trials 3 --cam_h 0.28 --tilt 0 --out run_cam7_depth_v5_h0.28_t0 | tail -2
python cam_summary.py run_cam7_*
