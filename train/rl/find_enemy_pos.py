"""Find the ENEMY position from an IDLE capture (Link held still).

With the player stationary, its coordinates are constant and the enemy's ramp --
either a wander (both directions) or, since idle Link gets attacked, a straight
APPROACH (monotonic ramp toward Link). So the fingerprint is the position one
(many distinct values, small consecutive steps, real spread) but WITHOUT the
"returns near start" requirement find_position used -- an approaching enemy ends
far from where it started.

Only the pre-death window is usable: once HP hits 0 the room tears down and every
"position" becomes garbage, so the death tick is found first and used as a hard
cutoff.

Enemies in this engine sit in a sprite struct, so the strong confirmation is TWO
ramping u16s a few bytes apart (X and Y), the same shape as player 0x038F0/0x038F4.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load, REGIONS  # noqa: E402

EW = 32 * 1024
# player position + known mirrors, all IWRAM; constant here but excluded anyway
PLAYER = {0x038F4, 0x038F5, 0x038F0, 0x038F1, 0x0391C, 0x0391D,
          0x03920, 0x03921, 0x03838, 0x03839}


def name(off, width=2):
    for reg, base, size in REGIONS:
        if base <= off < base + size:
            return "%s 0x%05X%s" % (reg, off - base, " u16" if width == 2 else "")
    return "?"


def main(dump):
    meta, ring = load(dump)
    n = ring.shape[0]

    # death cutoff: HP (EWRAM 0x0234D) first reaches 0
    hp = ring[:, EW + 0x0234D].astype(np.int32)
    dead = np.flatnonzero(hp == 0)
    cut = int(dead[0]) if dead.size else n
    print("HP trajectory (0x0234D):", " ".join(str(int(v)) for v in hp[::max(1, n // 20)]))
    print("clean pre-death window: snaps 0..%d of %d\n" % (cut, n))
    if cut < 10:
        print("window too short -- re-capture fewer frames"); return 1
    r = ring[:cut]

    # u16 over every adjacent pair
    lo = r[:, :-1].astype(np.int32)
    hi = r[:, 1:].astype(np.int32)
    v = lo | (hi << 8)
    d = np.diff(v, axis=0)
    moves = (d != 0).sum(axis=0)
    maxstep = np.abs(d).max(axis=0)
    spread = v.max(axis=0) - v.min(axis=0)
    up = (d > 0).any(axis=0)
    down = (d < 0).any(axis=0)

    # enemy fingerprint: ramps a lot, in small steps, over a real range. No
    # return-to-start requirement (approach is monotonic). 512 bound keeps it to
    # in-room coordinates and rejects timers/counters that sweep huge ranges.
    smooth = maxstep <= 8            # a sprite moves a pixel or two per 2 frames
    cand = np.flatnonzero((spread >= 8) & (spread <= 512) & (moves >= 8) & smooth)
    cand = [c for c in cand if c not in PLAYER]

    rows = sorted(cand, key=lambda o: -moves[o])
    print("%d ramping u16 candidates (Link excluded)\n" % len(rows))
    print("%-16s %7s %6s %7s %5s %5s  %s"
          % ("ADDR", "spread", "moves", "maxstep", "up", "down", "start->end"))
    seen = set()
    for off in rows[:40]:
        print("%-16s %7d %6d %7d %5s %5s  %d -> %d"
              % (name(off), spread[off], moves[off], maxstep[off],
                 "Y" if up[off] else "-", "Y" if down[off] else "-",
                 v[0, off], v[-1, off]))
        seen.add(off)

    # flag adjacent pairs: X and Y of the same sprite sit a few bytes apart
    print("\nADJACENT PAIRS (candidate X/Y of one sprite struct):")
    S = set(rows)
    for off in rows:
        for delta in (2, 4, -2, -4, 6, 8):
            if off + delta in S and (off + delta) > off:
                print("  %s  +  %s   (%d bytes apart)"
                      % (name(off), name(off + delta), delta))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
