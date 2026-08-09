#!/bin/bash
# Stability experiment (2026-08-02): e-config + a MILD target_kl to fix the
# convergence instability the engagement result left open.
#
# The e-runs (enemy_dmg, target_kl 0) reached good policies on 4/4 seeds but did
# NOT converge -- best != final, runs oscillated, and finals often near-collapsed
# (e11 +15/1key, e14 +82/1key). The destructive updates were the cause (KL spikes
# to 0.2-1.3). target_kl=0.08 caps those (SB3 early-stops the epoch at 1.5x =
# 0.12) while still allowing the healthy ~0.05-0.09 updates the successes used --
# deliberately looser than a6's 0.03, which OVER-damped (realised KL 0.008).
# Now SAFE to apply because avoidance is no longer the attractor enemy_dmg
# dethroned, so damping updates cannot resurrect the collapse it caused before.
#
# Also carries Tim's config decisions (inert for the stability question in -04):
# KEY_SCALE 150, rupees 2/pt collection-only. Same seeds 11-14 as g/e.
#
# PRE-REGISTERED metric (stability is intra-run, so judged on FINAL.zip, not the
# sweep): does final.zip beat random WITH keys? e-runs' finals were 2/4 (e11/e14
# near-collapsed). Target: >= 3/4 finals hold, AND the checkpoint-sweep stays 4/4
# (target_kl must not break REACHING a good policy while stabilising it).
set -uo pipefail
cd "$(dirname "$0")"
STATE=/vga/train/rl/states/alttp_human-04.state
for SEED in "$@"; do
  RUN=runs/alttp_s$SEED
  echo "=== launching s$SEED -> $RUN  $(date -Is) ==="
  docker run --rm --runtime nvidia --user 1000:1000 \
    -v /home/timothy/projects/VGA:/vga -w /vga/train/rl \
    -e VGA_ALTTP_STATE=$STATE thor-rl:cu130 \
    python3 train_ppo.py --env alttp --total-timesteps 1000000 \
    --target-kl 0.08 --gamma 0.999 --ent-coef 0.01 --seed "$SEED" \
    --checkpoint-dir "$RUN"
  echo "=== s$SEED finished rc=$? $(date -Is) ==="
done
