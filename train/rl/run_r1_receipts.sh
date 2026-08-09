#!/bin/bash
# Post-SUCCESS receipts: (1) composability -- arbiter arm ON TOP of the distilled
# ckpt (does S2 still add?); (2) reflex retention -- standard long episodes.
set -u
cd /home/timothy/projects/VGA
DOCKER="docker run --rm --runtime nvidia --network host --user 1000:1000 \
  -v /home/timothy/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130"
H2='/vga/train/perception/keybank_holdout2/states/keydrop-*.state'
H1='/vga/train/perception/keybank_holdout/states/keydrop-*.state'
curl -s -m 15 http://127.0.0.1:8077/health >/dev/null || { echo "VLM DOWN"; exit 3; }
for SEED in 0 100 200; do
  $DOCKER python3 -u arbiter_v1.py --arm arbiter --seed "$SEED" \
    --ckpt runs/distill_r1/s1_distilled.zip --states "$H2" \
    --out "r1_arb_on_distilled_h2_s${SEED}.json" || { echo "COMPOSE H2 FAILED"; exit 4; }
  $DOCKER python3 -u arbiter_v1.py --arm arbiter --seed "$SEED" \
    --ckpt runs/distill_r1/s1_distilled.zip --states "$H1" \
    --out "r1_arb_on_distilled_h1_s${SEED}.json" || { echo "COMPOSE H1 FAILED"; exit 4; }
done
echo "=== reflex retention (standard episodes) ==="
$DOCKER python3 -u eval_ppo_md.py \
  --models base=runs/alttp_s1v2_scratch04/ppo_alttp_600000_steps.zip \
           distilled=runs/distill_r1/s1_distilled.zip \
  --states alttp_human-04 --episodes 6 --horizon 15000 \
  || { echo "RETENTION EVAL FAILED"; exit 5; }
echo "=== DONE ==="
