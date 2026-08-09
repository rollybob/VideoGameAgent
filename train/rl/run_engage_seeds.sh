#!/bin/bash
# Engagement experiment (2026-08-02): g11-g14 config + the new enemy_dmg reward.
#
# CONTROLLED: identical to g11-g14 in every way (gamma 0.999, ent_coef 0.01,
# target_kl 0, KEY_SCALE 120, state -04) EXCEPT the enemy_dmg channel is now on
# (IWRAM 0x03253, ENEMY_DMG_SCALE=13 derived by measure_engage.py). Same SEEDS as
# g11-g14, so it is a seed-by-seed comparison: g11/g13 FAILED (found keys early,
# collapsed to avoidance), g12/g14 succeeded. HYPOTHESIS: the dense, early,
# unfarmable engagement reward keeps key-seeking from collapsing, flipping the
# failures without breaking the successes.
#
# PRE-REGISTERED metric: checkpoint-sweep each (never final.zip alone, per the
# a5 lesson). Success = best checkpoint beats random AND registers key events in
# eval. Target: >= 3/4 seeds succeed (baseline 2/4), and the trajectories of the
# previous failures show SUSTAINED key rate rather than the early collapse.
set -uo pipefail
cd "$(dirname "$0")"
STATE=/vga/train/rl/states/alttp_human-04.state
for SEED in "$@"; do
  RUN=runs/alttp_e$SEED
  echo "=== launching e$SEED -> $RUN  $(date -Is) ==="
  docker run --rm --runtime nvidia --user 1000:1000 \
    -v /home/timothy/projects/VGA:/vga -w /vga/train/rl \
    -e VGA_ALTTP_STATE=$STATE thor-rl:cu130 \
    python3 train_ppo.py --env alttp --total-timesteps 1000000 \
    --target-kl 0 --gamma 0.999 --ent-coef 0.01 --seed "$SEED" \
    --checkpoint-dir "$RUN"
  echo "=== e$SEED finished rc=$? $(date -Is) ==="
done
