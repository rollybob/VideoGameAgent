#!/bin/bash
# Did the agent UNLEARN key-seeking, or never find it?
#
# a4's final policy collects ZERO keys where random collects 7. Those are two
# very different failures with different fixes: if early checkpoints DO collect
# keys and late ones do not, avoidance actively out-competed the key channel and
# the problem is reward BALANCE. If no checkpoint ever collects one, the agent
# never discovered them and the problem is EXPLORATION.
# Uses checkpoints already on disk, so it costs no training time.
set -uo pipefail
cd "$(dirname "$0")"
for STEPS in 19998 99990 199980 399960 599940 799920; do
  M="runs/alttp_a5/ppo_alttp_${STEPS}_steps.zip"
  [ -f "$M" ] || continue
  echo "--- ${STEPS} steps ---"
  python3 eval_policy.py "$M" --state states/alttp_human-04.state --episodes 8 \
    2>&1 | grep -E "^policy" | sed "s/^/  /"
done
echo "--- final ---"
python3 eval_policy.py runs/alttp_a5/ppo_alttp_final.zip \
  --state states/alttp_human-04.state --episodes 8 2>&1 | grep -E "^random|^policy" | sed "s/^/  /"
