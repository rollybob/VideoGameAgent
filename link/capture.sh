#!/bin/bash
# One entry point for the human capture session (Task 09 A1 follow-up).
#
#   ./capture.sh alttp            solo ALttP from the generic seed state
#   ./capture.sh alttp-02         solo ALttP from a BANKED training start-point
#   ./capture.sh taluscave        Four Swords -- mark a hit for the HP address (F11)
#   ./capture.sh deathmountain    ditto
#   ./capture.sh seaoftrees       ditto (HP address already confirmed; for re-checks)
#
# Runs on the real desktop :1. Both tools put the keys in the on-screen HUD bar.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
export DISPLAY="${DISPLAY:-:1}"
VGA_ROOT="$(cd .. && pwd)"

TARGET="${1:-}"
shift || true

case "$TARGET" in
  alttp)
    echo "solo ALttP [seed state] -- F8 banks a savestate, F11 marks a hit, ESC quits"
    exec "$LINK_PY" solo_recorder.py "$@"
    ;;
  alttp-*)
    # The key / room captures must happen in the SAME room the agent trains from,
    # otherwise a confirmed address is anchored to a room PPO never sees. The
    # recorder's default seed state is a different room, so name the bank here.
    BANK="$VGA_ROOT/train/rl/states/alttp_human-${TARGET#alttp-}.state"
    if [ ! -f "$BANK" ]; then
      echo "no such banked state: $BANK"
      echo "available: $(ls "$VGA_ROOT"/train/rl/states/alttp_human-*.state 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
      exit 1
    fi
    echo "solo ALttP [$(basename "$BANK")] -- F8 banks a savestate, F11 marks a hit, ESC quits"
    exec "$LINK_PY" solo_recorder.py --state "$BANK" "$@"
    ;;
  taluscave|deathmountain|seaoftrees|coop)
    echo "Four Swords [$TARGET] -- you are PLAYER 1; F11 marks a hit, ESC quits"
    # --agents idle: the other three stand still, so the only hearts that move
    # are yours. Wander bots would add unrelated RAM churn to every dump.
    exec "$LINK_PY" link_viewer.py --start "$TARGET" --human 1 --agents idle "$@"
    ;;
  *)
    echo "usage: $0 {alttp|alttp-NN|taluscave|deathmountain|seaoftrees}"
    echo "banked ALttP start-points: $(ls "$VGA_ROOT"/train/rl/states/alttp_human-*.state 2>/dev/null | xargs -n1 basename | sed 's/alttp_human-/alttp-/;s/\.state//' | tr '\n' ' ')"
    echo "available Four Swords banks: $("$LINK_PY" link_viewer.py --list-banks)"
    exit 1
    ;;
esac
