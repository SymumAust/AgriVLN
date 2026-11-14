#!/bin/bash
# Step 1 of plan_wording_8b.txt: wording candidates for both 8B models on the 9 development clips.
# Run from the repo folder with the agrivln environment active:   bash run_wording_8b.sh
set -e
DEV="farm_6 farm_7 farm_8 farmB_1 farmB_2 farmB_3 farmB_4 farmB_5 farmB_6"
B="you are inside a narrow straight lane between two rows of trees, with tree trunks close on both the left and right"
C="tree branches fill both the left edge and the right edge of your view, and a long narrow path between tree rows goes straight ahead"
D="you are between two tree rows that are close on both sides, not in a wide open area at the end of the rows"
for M in qwen3-vl:8b-instruct qwen3-vl:8b; do
  T=$( [ "$M" = "qwen3-vl:8b" ] && echo w8b || echo w8bi )
  python3 probe_arrival.py --models $M --eps farm_6 farm_7 farm_8 --out probe_results_${T}_A.csv     # A on farmB_1-6 exists
  python3 probe_arrival.py --models $M --eps $DEV --condition "$B" --out probe_results_${T}_B.csv
  python3 probe_arrival.py --models $M --eps $DEV --condition "$C" --out probe_results_${T}_C.csv
  python3 probe_arrival.py --models $M --eps $DEV --condition "$D" --out probe_results_${T}_D.csv
  ollama stop $M
done
echo "STEP 1 DONE"
