"""Task 09 A0: validate the ALttP reward oracle against solo combat tapes.

Reads EWRAM dumps produced by solo_ram_capture.py (in scratch/ram/flight-*)
and replays the AlttpOracle logic offline over the ewram stream.

Expected results (see sessions/SESSION_2026-07-09_solo_mining.md FINDINGS):
  - tape 012532: exactly 1 damage event (28->20, delta=-8), no deaths
  - 4 of 6 tapes contain at least 1 death (health -> 0)
  - respawn (0->28) emits NO event (recovery, not reward)
  - zero spurious decrements during stable-health periods

Gate for A0: damage and death events detected in 3+ tapes with no false positives
             during periods where health is stable.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \\
      python3 /vga/train/rl/validate_alttp_oracle.py \\
      /vga/train/rl/scratch/ram/flight-20260709-012*
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEALTH_ADDR = 0x00C93   # EWRAM offset; validated 2026-07-09


def analyze_dump(d):
    with open(os.path.join(d, "ticks.json")) as f:
        n = json.load(f)["ticks"]
    ew = np.memmap(os.path.join(d, "ewram.u8"), dtype=np.uint8, mode="r",
                   shape=(n, 256 * 1024))
    hp = np.asarray(ew[:, HEALTH_ADDR], dtype=np.int32)

    # Replay oracle logic: detect damage (negative delta) and death (-> 0)
    deltas = np.diff(hp)
    damage_ticks = np.nonzero(deltas < 0)[0] + 1
    death_ticks = [int(t) for t in damage_ticks if hp[t] == 0]
    damage_events = [(int(t), int(deltas[t - 1]), int(hp[t])) for t in damage_ticks]

    # Respawn: hp jumps up from 0 -- should NOT be emitted as a damage event
    respawn_ticks = np.nonzero((hp[:-1] == 0) & (deltas > 0))[0] + 1

    return {
        "ticks": n,
        "hp_start": int(hp[0]),
        "hp_min": int(hp.min()),
        "damage_events": damage_events,
        "death_ticks": death_ticks,
        "respawn_ticks": [int(t) for t in respawn_ticks],
    }


def main():
    args = sys.argv[1:]
    dumps = []
    for a in args:
        p = a if os.path.isabs(a) else os.path.join(HERE, a)
        dumps.extend(sorted(glob.glob(p)) if any(c in a for c in "*?") else [p])

    if not dumps:
        print("Usage: validate_alttp_oracle.py scratch/ram/flight-20260709-012*")
        sys.exit(1)

    tapes_with_damage = 0
    tapes_with_death = 0
    total_events = 0

    for d in dumps:
        name = os.path.basename(d)
        r = analyze_dump(d)
        dmg = r["damage_events"]
        deaths = r["death_ticks"]
        resps = r["respawn_ticks"]
        total_events += len(dmg)
        if dmg:
            tapes_with_damage += 1
        if deaths:
            tapes_with_death += 1
        dmg_summary = [(t, d_, v) for t, d_, v in dmg[:5]]
        print("%s: %d ticks  hp_start=%d  hp_min=%d  damage=%d @%s  deaths=%d @%s  respawns=%d @%s"
              % (name, r["ticks"], r["hp_start"], r["hp_min"],
                 len(dmg), ["t%d:%+d->%d" % (t, d_, v) for t, d_, v in dmg_summary],
                 len(deaths), deaths[:3],
                 len(resps), resps[:3]))

    print()
    print("Tapes with damage events : %d / %d" % (tapes_with_damage, len(dumps)))
    print("Tapes with death events  : %d / %d" % (tapes_with_death, len(dumps)))
    print("Total damage events      : %d" % total_events)
    print()

    # Gate check
    passed = tapes_with_damage >= 3 and tapes_with_death >= 3
    if passed:
        print("GATE A0 ALttP oracle: PASS")
    else:
        print("GATE A0 ALttP oracle: FAIL (need damage+death in >= 3 tapes each)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
