#!/bin/bash
# Checkpoint-sweep eval of the DR seeds across the train mix + held-out -01
# (2026-08-03). Reports rooms-visited (primary), keys (retain), vs the random
# floor, per state. CPU only. Usage: run_dr_eval.sh dr11 dr12
set -uo pipefail
cd "$(dirname "$0")"
for R in "$@"; do
  echo ""
  echo "################# EVAL runs/alttp_$R #################"
  docker run --rm --user 1000:1000 -v /home/timothy/projects/VGA:/vga -w /vga/train/rl \
    thor-rl:cu130 python3 dr_eval.py --run "runs/alttp_$R" --episodes 8 --ckpts 5
done
echo ""
echo "=== dr eval done ==="
