#!/usr/bin/env bash
# Rung 3 task 1 -- decisive behavioral test of room-generic enemy addressing.
# Hypothesis (from probe_enemy_slots.py + confirm_enemy_pos.py): the room enemy
# lives at IWRAM 0x03852/54 in EVERY room, adjacent to Link, so the rung-2
# ChainHunter (which homes onto that addr) should execute engage->kill->collect
# in all 7 human rooms, not just -04. Metric: successes>0 in every room.
# Fixed episodes/room for a fair per-room yield comparison; horizon capped short
# (an adjacent enemy is killed in <~100 decisions; no_kill eps then truncate fast).
set -u
OUT=/vga/train/rl/_r3probe
mkdir -p "$OUT"
for i in 0 1 2 3 4 5 6; do
  echo "=========== ROOM alttp_human-0$i ==========="
  python3 -u chain_harvest.py \
      --state states/alttp_human-0$i.state \
      --episodes 12 --target 9999 --horizon 1500 --seed 5000 \
      --out "$OUT/room$i" || echo "ROOM $i ERRORED (continuing)"
done
echo "=========== PER-ROOM SUMMARY ==========="
for i in 0 1 2 3 4 5 6; do
  s="$OUT/room$i/harvest_summary.json"
  if [ -f "$s" ]; then echo -n "room$i: "; cat "$s" | tr -d '\n '; echo; fi
done
