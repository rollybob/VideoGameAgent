#!/bin/bash
# A/B the corrected ALttP health address against the mirror it replaces.
#
# WHY THIS EXISTS: a2 and a3 were both measured on EWRAM 0x00C93, which
# 2026-08-01 pixel ground truth showed is a lagging low-resolution mirror, not
# live HP (it reads 28 in alttp_human-02 where max HP is 24). Swapping the
# address redefines the reward, so "the a2/a3 verdicts still hold" is an
# assumption until it is measured on identical seeds. This measures it.
#
# Each policy is evaluated on the state it TRAINED on (a2 -> -02, a3 -> -00),
# otherwise the comparison changes two variables at once.
#
# The env var is read at import time, so every config needs its own process.
set -uo pipefail
cd "$(dirname "$0")"

EPISODES="${EPISODES:-12}"

for ADDR in 0x0234D 0x00C93; do
  for SPEC in "alttp_a2:alttp_human-02" "alttp_a3:alttp_human-00"; do
    RUN="${SPEC%%:*}"
    STATE="${SPEC##*:}"
    MODEL="runs/$RUN/ppo_alttp_final.zip"
    echo ""
    echo "=================================================================="
    if [ "$ADDR" = "0x0234D" ]; then
      echo "ADDR $ADDR (CORRECTED, live HP)   $RUN on $STATE"
    else
      echo "ADDR $ADDR (OLD mirror)           $RUN on $STATE"
    fi
    echo "=================================================================="
    if [ ! -f "$MODEL" ]; then
      echo "MISSING MODEL: $MODEL -- skipped"
      continue
    fi
    VGA_ALTTP_HEALTH_ADDR="$ADDR" python3 eval_policy.py "$MODEL" \
      --state "states/$STATE.state" --episodes "$EPISODES" 2>&1 \
      | grep -v "GBA BIOS: SWI"
  done
done
echo ""
echo "AB COMPLETE"
