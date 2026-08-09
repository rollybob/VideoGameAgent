"""Find Link's X/Y position, and the magic meter, from movement/magic captures.

Neither fits the detectors built so far, and that is the point of a separate
fingerprint rather than another pass of the same ranking.

POSITION is a RAMP, not a step. Walking right moves the coordinate smoothly
through dozens of values and walking back returns it. Every existing detector
demands "constant -> change -> constant", so a ramp either fails the dwell test
or gets scored as noise. The distinguishing marks of a coordinate are:
  - many distinct values (a sweep, not a flag)
  - SMALL consecutive deltas -- Link moves a pixel or two per frame, so a byte
    that leaps by 100 is not a position
  - it comes back toward where it started when the player walks back
  - 16-bit, because GBA Zelda coordinates exceed a byte

MAGIC is bounded and ends at a KNOWN value. Tim collected two jars then spammed
magic until the bar was empty, so the meter must RISE at least twice, FALL, and
SETTLE AT EXACTLY ZERO. Ending on a hard zero is the same anchor that cracked
health (the Talus Cave death) -- it turns shape-matching into a known-value test.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load, runs, REGIONS  # noqa: E402


def name(off, width=1, tag=""):
    for reg, base, size in REGIONS:
        if base <= off < base + size:
            return "%s 0x%05X%s" % (reg, off - base, tag)
    return "?"


def u16(ring):
    lo = ring[:, :-1].astype(np.int32)
    hi = ring[:, 1:].astype(np.int32)
    return lo | (hi << 8)


def find_position(dump, axis):
    _m, ring = load(dump)
    v = u16(ring)
    d = np.diff(v, axis=0)
    nz = d != 0
    moves = nz.sum(axis=0)
    # a coordinate sweeps a lot but never teleports
    smooth = (np.abs(d).max(axis=0) <= 16)  # 16 not 6: the ring samples every 2 frames, and Link covers more than 6 units in that time -- 6 returned zero candidates
    spread = v.max(axis=0) - v.min(axis=0)
    both_ways = (d > 0).any(axis=0) & (d < 0).any(axis=0)
    returns = np.abs(v[-1] - v[0]) <= 8
    cand = np.flatnonzero(smooth & both_ways & returns & (moves >= 20) & (spread >= 16))
    rows = sorted(cand, key=lambda o: -spread[o])
    print("\n=== %s from %s: %d candidates ==="
          % (axis, os.path.basename(dump.rstrip("/")), len(rows)))
    print("%-18s %7s %7s %7s  %s" % ("ADDR", "spread", "moves", "maxstep", "start->min->max->end"))
    for off in rows[:15]:
        print("%-18s %7d %7d %7d  %d -> %d -> %d -> %d"
              % (name(off, 2, " u16"), spread[off], moves[off],
                 int(np.abs(d[:, off]).max()),
                 v[0, off], v[:, off].min(), v[:, off].max(), v[-1, off]))
    return rows


def find_magic(dump):
    _m, ring = load(dump)
    print("\n=== magic from %s ===" % os.path.basename(dump.rstrip("/")))
    hits = []
    for addr in range(ring.shape[1]):
        col = ring[:, addr]
        if int(col[-1]) != 0:                 # must END empty
            continue
        r = [x[0] for x in runs(col) if x[2] >= 3]
        if len(r) < 3 or max(r) > 128:
            continue
        ups = sum(1 for i in range(len(r) - 1) if r[i + 1] > r[i])
        downs = sum(1 for i in range(len(r) - 1) if r[i + 1] < r[i])
        if ups < 2 or downs < 1:              # two jars, then drained
            continue
        hits.append((max(r), ups, downs, addr, r))
    hits.sort(reverse=True)
    print("%-18s %6s %5s %6s  %s" % ("ADDR", "peak", "ups", "downs", "value sequence"))
    for peak, ups, downs, addr, r in hits[:15]:
        print("%-18s %6d %5d %6d  %s"
              % (name(addr), peak, ups, downs,
                 ",".join(str(x) for x in r[:12])))
    if not hits:
        print("(nothing rose twice and drained to zero)")
    return hits


def main(argv):
    mode, dump = argv[0], argv[1]
    if mode == "magic":
        find_magic(dump)
    else:
        find_position(dump, mode)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
