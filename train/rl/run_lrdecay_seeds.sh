#!/bin/bash
# LR-decay stability test (2026-08-02): the one principled untried lever.
#
# Hypothesis: best!=final is fixed-lr PPO never SETTLING. clip_fraction sat at
# 0.3-0.5 (healthy 0.1-0.2) the whole run and the crashes were LOW-KL cumulative
# drift (so target_kl missed them, and ent_coef only changed the symptom). Linear
# lr anneal to 0 shrinks the step size at the END so the policy settles into
# wherever it is instead of jittering out of a good region.
#
# SHORT TEST: 2 seeds (11, 14) -- the WORST e-run oscillators (e11 final +15,
# e14 final +82, both FAILED final.zip despite great peaks). If lr-decay flips
# both finals to beat random with keys, it works; scale to 4. Everything else =
# standing config (KEY 150, ent 0.01, target_kl 0, gamma 0.999, enemy_dmg 13).
# Two parallel jobs (1 seed each) -> ~1.6h.
#
# PRE-REGISTERED: (1) final.zip beats random (+115) WITH keys for BOTH seeds
# (e-runs: 0/2 for these); (2) clip_fraction drops as lr anneals and late-training
# reward stops oscillating (the mechanism). DO NOT resume these (lr schedule
# re-bases on resume).
set -uo pipefail
cd "$(dirname "$0")"
STATE=/vga/train/rl/states/alttp_human-04.state
for SEED in "$@"; do
  RUN=runs/alttp_lr$SEED
  echo "=== launching lr$SEED -> $RUN  $(date -Is) ==="
  docker run --rm --runtime nvidia --user 1000:1000 \
    -v /home/timothy/projects/VGA:/vga -w /vga/train/rl \
    -e VGA_ALTTP_STATE=$STATE thor-rl:cu130 \
    python3 train_ppo.py --env alttp --total-timesteps 1000000 \
    --lr-decay --target-kl 0 --ent-coef 0.01 --gamma 0.999 --seed "$SEED" \
    --checkpoint-dir "$RUN"
  echo "=== lr$SEED finished rc=$? $(date -Is) ==="
done
