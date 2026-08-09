#!/bin/bash
# Entropy-collapse fix (2026-08-02): raise ent_coef, drop target_kl.
#
# The stability diagnosis: the e/s runs decline from their peak via ENTROPY
# COLLAPSE (at each crash approx_kl is LOW 0.015-0.07 while entropy_loss falls to
# ~0 -- the policy goes deterministic), NOT destructive KL updates. target_kl
# (part 7) targeted the wrong mechanism and over-damped. The right lever is a
# higher entropy bonus: ent_coef 0.01 -> 0.025 resists the premature determinism.
# target_kl OFF (it hurt). KEY 150 standing config, enemy_dmg 13, gamma 0.999.
# Same seeds 11-14 as g/e/s.
#
# PRE-REGISTERED:
#  MECHANISM (the direct test): entropy_loss stays elevated -- no collapse toward
#    0 at the peaks, unlike e/s where it fell to -0.1..-0.4.
#  OUTCOME (stability): final.zip beats random WITH keys for >= 3/4 (e-runs 2/4,
#    s-runs 1/4), AND the checkpoint sweep stays >= 4/4 (must not break reaching a
#    good policy).
#  GUARD: too much entropy stops the policy sharpening -- watch that peaks do NOT
#    drop vs the e-runs (would mean 0.025 over-explores; back off toward 0.015).
set -uo pipefail
cd "$(dirname "$0")"
STATE=/vga/train/rl/states/alttp_human-04.state
for SEED in "$@"; do
  RUN=runs/alttp_h$SEED
  echo "=== launching h$SEED -> $RUN  $(date -Is) ==="
  docker run --rm --runtime nvidia --user 1000:1000 \
    -v /home/timothy/projects/VGA:/vga -w /vga/train/rl \
    -e VGA_ALTTP_STATE=$STATE thor-rl:cu130 \
    python3 train_ppo.py --env alttp --total-timesteps 1000000 \
    --target-kl 0 --ent-coef 0.025 --gamma 0.999 --seed "$SEED" \
    --checkpoint-dir "$RUN"
  echo "=== h$SEED finished rc=$? $(date -Is) ==="
done
