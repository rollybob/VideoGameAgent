#!/bin/bash
# Domain-randomization + exploration run (2026-08-03). The first test of whether
# the RECIPE generalizes (not the -04 policy, which is a proven overfit): ONE
# policy trained over the mix {-02,-03,-04}, a different room sampled per episode,
# with the new dense exploration breadcrumb (explore, shift 5 / scale 2) and the
# per-state enemy_dmg gate (rewarded only in -04). Standing config otherwise:
# lr-decay (now default-on), gamma 0.999, ent_coef 0.01, target_kl off, KEY 150.
#
# PRE-REGISTERED metric (eval AFTER, checkpoint-sweep, never final.zip alone):
#   PRIMARY (explore/generalize): mean rooms-visited/episode across the mix beats
#     the random floor (~1.0-1.9) -- the policy should TRAVERSE rooms, not farm one.
#   RETAIN: in -04 it still collects keys (>0, beats random's 6). A DR generalist
#     will not match the 12-key -04 specialist, but must keep competency.
#   GENERALIZE: zero-shot rooms-visited / keys on the held-out room -01.
#   KILL: if it neither explores more than random NOR retains -04 competency, the
#     approach needs rethinking (do not just add seeds).
# 2 seeds first (proof of concept, ~1.6h parallel as two 1-seed jobs); scale to 4
# only if it learns. Env validated offline by measure_dr before this launch.
set -uo pipefail
cd "$(dirname "$0")"
STATES=alttp_human-02,alttp_human-03,alttp_human-04
for SEED in "$@"; do
  RUN=runs/alttp_dr$SEED
  echo "=== launching dr$SEED (mix $STATES) -> $RUN  $(date -Is) ==="
  docker run --rm --runtime nvidia --user 1000:1000 \
    -v /home/timothy/projects/VGA:/vga -w /vga/train/rl \
    -e VGA_ALTTP_STATES=$STATES thor-rl:cu130 \
    python3 train_ppo.py --env alttp --total-timesteps 1000000 \
    --lr-decay --target-kl 0 --ent-coef 0.01 --gamma 0.999 --seed "$SEED" \
    --checkpoint-dir "$RUN"
  echo "=== dr$SEED finished rc=$? $(date -Is) ==="
done
