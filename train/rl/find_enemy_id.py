"""Find the enemy alive-flag / type / health from a KILL capture.

Tim's method: Link kills the enemy; at the instant its movement stops, something
that was nonzero goes to 0. The enemy position mirror (0x03852/0x03854) gives the
death anchor -- it moves through ~step 116 then freezes permanently. So:

  SCAN A (alive flag / type / slot / sprite): bytes NONZERO throughout the
      enemy's alive window and ZERO throughout the dead window. Tim's caveat is
      that the sprite sheet unloads too, so expect a CLUSTER; the enemy's game-
      state slot is the one that is a small, stable id-like value (not a big
      graphics buffer), and ideally flips right at the death step.

  SCAN B (enemy health): a byte that DRAINS -- non-increasing across the fight,
      several distinct values, reaching 0 at death. Worth more than the flag: a
      draining health bar is a DENSE engagement reward (pay Link for each hit),
      not just an on/off cutoff.

Windows are read off the movement trace: alive+moving through ~116, frozen after.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load, fmt_addr  # noqa: E402

ALIVE = (30, 100)     # solidly alive and moving
DEATH = 117           # last position change
DEAD = (130, 190)     # solidly dead


def region_of(off):
    return "IWRAM" if off < 32 * 1024 else "EWRAM"


def main(dump):
    meta, ring = load(dump)
    n = ring.shape[0]
    a0, a1 = ALIVE
    d0, d1 = DEAD
    alive = ring[a0:a1].astype(np.int32)
    dead = ring[d0:d1].astype(np.int32)

    # SCAN A: nonzero all through alive, zero all through dead
    present = (alive > 0).all(axis=0)
    absent = (dead == 0).all(axis=0)
    flagA = np.flatnonzero(present & absent)
    # when does each flip to 0 (first dead step)? keep those flipping near DEATH
    print("SCAN A -- nonzero while alive, 0 while dead: %d bytes" % flagA.size)
    print("(clustered = sprite unload; want small stable id-like values flipping ~step %d)\n" % DEATH)
    rows = []
    for off in flagA:
        col = ring[:, off].astype(np.int32)
        flip = int(np.flatnonzero(col[DEATH - 20:] == 0)[0]) + (DEATH - 20) if (col[DEATH - 20:] == 0).any() else n
        rows.append((abs(flip - DEATH), flip, off, int(col[a0])))
    rows.sort()
    print("%-16s %8s %8s   %s" % ("ADDR", "aliveval", "zero@step", "|step-death|"))
    for dd, flip, off, av in rows[:30]:
        print("%-16s %8d %8d   %d" % (fmt_addr(int(off)), av, flip, dd))

    # contiguity: enemy slots are structs, so runs of consecutive flag bytes are
    # the interesting objects (a struct), lone bytes amid churn less so
    print("\ncontiguous runs among SCAN A hits (candidate enemy structs):")
    s = sorted(int(o) for o in flagA)
    run = [s[0]] if s else []
    for x in s[1:] + [None]:
        if x is not None and x - run[-1] <= 2:
            run.append(x)
        else:
            if len(run) >= 3:
                print("  %s .. %s  (%d bytes)" % (fmt_addr(run[0]), fmt_addr(run[-1]), len(run)))
            run = [x] if x is not None else []

    # SCAN B: draining health -- non-increasing over the fight, hits 0, small range
    print("\nSCAN B -- draining-to-zero health candidates:")
    fight = ring[:DEATH + 3].astype(np.int32)
    d = np.diff(fight, axis=0)
    nonincr = (d <= 0).all(axis=0)
    ends0 = ring[DEAD[0]] == 0
    moved = fight.max(axis=0) != fight.min(axis=0)
    small = (fight.max(axis=0) <= 255) & (fight.max(axis=0) >= 2)
    healthc = np.flatnonzero(nonincr & ends0 & moved & small)
    hrows = []
    for off in healthc:
        col = ring[:, off].astype(np.int32)
        vals = [int(v) for v in col[:DEATH + 3]]
        # run-length of distinct values
        seq = [v for k, v in enumerate(vals) if k == 0 or v != vals[k - 1]]
        hrows.append((len(seq), off, seq))
    hrows.sort()
    for nseq, off, seq in hrows[:20]:
        print("  %-16s  %s" % (fmt_addr(int(off)), " ".join(str(x) for x in seq[:10])))
    if not hrows:
        print("  (none -- enemy may die in one hit, so health reads present->0 with no ramp)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
