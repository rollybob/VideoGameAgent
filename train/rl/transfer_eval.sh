#!/bin/bash
# Zero-shot transfer eval (2026-08-03): drop the -04-trained policies into every
# other banked ALttP save state and measure how they fare -- the first test of
# whether this is a general fast-brain or an -04 overfit.
#
# WHAT IS AND ISN'T VALID ACROSS ROOMS: the -04 reward stack used enemy_dmg at
# sprite SLOT 3 (IWRAM 0x03253), which is -04-specific. But this eval SCORES on
# the GLOBAL channels only -- keys 0x0234F, health 0x0234D (damage/death) -- so
# key/damage/death counts are meaningful in any room. We are NOT retraining here;
# we load the fixed -04 policy and run it elsewhere.
#
# -04 IS INCLUDED AS A CONTROL: the same policy should still score its ~+228/12
# keys there. If it does, any drop in the other rooms is attributable to the ROOM,
# not to a broken eval harness.
#
# POLICIES: lr11 + lr12 -- the two both-mode-perfect seeds (n=2 guards against a
# single-seed transfer fluke). eval_finals.py prints a per-state RANDOM baseline
# for comparison and reports BOTH deterministic(argmax) and sampled.
#
# One docker container per state (clean isolation; solo ALttP has not shown the
# FS SIO hang hazard, but per-state containers keep any single bad room contained).
set -uo pipefail
cd "$(dirname "$0")"
STATES=${*:-"00 01 02 03 04 05"}
for ST in $STATES; do
  echo ""
  echo "############### STATE alttp_human-$ST ###############"
  docker run --rm --runtime nvidia --user 1000:1000 \
    -v /home/timothy/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130 \
    python3 eval_finals.py --runs runs/alttp_lr11 runs/alttp_lr12 \
      --state alttp_human-$ST
done
echo ""
echo "=== transfer_eval done ==="
