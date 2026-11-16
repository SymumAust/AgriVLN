#!/bin/bash
# plan_sim.txt: E1 (CPU), E2 + E3 (GPU, Ollama). Run from ~/projects/sim with the chrono environment (+ pip install ollama pillow).
set -e
S=2026
for sc in row lookalike; do for pol in drive step6; do
  python row_stop_sim.py --seed $S --trials 30 --scenario $sc --policy $pol --out E1_${sc}_${pol}            | tail -1
done; done
for sc in row lookalike; do for pol in drive step6; do
  python row_stop_sim.py --seed $S --trials 10 --scenario $sc --policy $pol --depth camera --out E2_${sc}_${pol} | tail -1
done; done
for m in "qwen3-vl:4b A" "qwen3-vl:8b-instruct C"; do set -- $m
  for sc in row lookalike; do for pol in step6 drive; do
    python row_stop_sim.py --seed $S --trials 10 --scenario $sc --policy $pol --decider vlm --model $1 --wording $2 \
        --out E3_${1//[:.]/_}_${sc}_${pol} | tail -1
  done; done
  ollama stop $1
done
echo "ALL DONE"
