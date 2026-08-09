#!/usr/bin/env bash
# Rung 3: joint {room4 + room2} chain distill -- faithful r2c recipe + round3_room2.
# HYPOTHESIS: adding a 2nd key-room's chain demos makes the COLLECT-after-kill step
#   room-general. Rung-2 (room4-only) distilled to only 20% collect|kill in held-out
#   room2 vs 75% in room4 -- KILL transfers, COLLECT does not. This adds room2's collect.
# METRIC (pre-registered, 2026-08-07):
#   in-sample (measurable now): rung3 room2 collect|kill UP from ~20% toward room4's
#       level (>=50% target); room4 collect|kill NOT regressed.
#   far-guard: rung-1 far-seek collect preserved >=85% (do not regress rung 1).
#   held-out (LATER, needs a 3rd-dungeon key room from Tim's capture): the true
#       generalization test -- collect|kill on a room in NEITHER training set.
set -u
cd /home/timothy/projects/VGA
DOCKER="docker run --rm --runtime nvidia --user 1000:1000 \
  -v /home/timothy/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130"
H2='/vga/train/perception/keybank_holdout2/states/keydrop-*.state'
H1='/vga/train/perception/keybank_holdout/states/keydrop-*.state'
mkdir -p /home/timothy/projects/VGA/train/rl/runs/distill_r3

echo "=== [1/3] joint distill from base (round1+round2+round2far+round3_room2) ==="
$DOCKER python3 -u distill_bc.py \
  --data distill_data/round1 distill_data/round2 distill_data/round2far distill_data/round3_room2 \
  --ckpt runs/alttp_s1v2_scratch04/ppo_alttp_600000_steps.zip \
  --out runs/distill_r3/s1_joint_r3.zip || { echo "TRAIN FAILED"; exit 5; }

echo "=== [2/3] PRIMARY: in-sample chain-complete, all policies x {room4,room2,room0} ==="
$DOCKER python3 -u transfer_chain_eval.py || { echo "TRANSFER EVAL FAILED"; exit 6; }

echo "=== [3/3] far-guard: rung-1 far-seek preserved (arbiter s1 on holdout banks) ==="
for SEED in 0 100 200 300 400 500; do
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r3/s1_joint_r3.zip --states "$H2" \
    --out "distill_r3_h2_s${SEED}.json" || { echo "GUARD FAILED"; exit 7; }
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r3/s1_joint_r3.zip --states "$H1" \
    --out "distill_r3_h1_s${SEED}.json" || { echo "GUARD FAILED"; exit 7; }
done
echo "=== DONE rung3 distill+eval (analyze transfer table + far-guard JSONs) ==="
