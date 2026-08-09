"""Unbiased confirmation that the magic meter is EWRAM 0x0234E.

Yesterday's find_magic used a DWELL-based filter (each value had to hold >= 3
samples). A meter drained by an item ramps smoothly -- dozens of 1-4 unit steps,
no dwell -- so that filter rejected it exactly the way the dwell detectors
rejected Link's position. The fix is the same one that cracked position: use the
SHAPE that actually fits (a bounded ramp), not the counter/dwell shape.

The strong test here is a CROSS-DUMP intersection, the magic analogue of the
health death-anchor:

  DRAIN dump (Tim "used magic until empty"): the byte starts > 0, only ever
      decreases, and ends at EXACTLY 0. Ending on a hard zero is the anchor.
  RISE  dump (Tim "picked up 2 jars"):       the byte starts at 0 and only ever
      increases (two fill animations).

A byte that does BOTH, at the SAME address, across two windows a human anchored
on two opposite events, is the meter. Coordinates ramp smoothly too -- which is
why Tim said to filter them out -- so the known position addresses are excluded
and annotated rather than silently trusted.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load, fmt_addr  # noqa: E402

# already-confirmed addresses, so a hit on one is recognised not rediscovered.
# position is u16 in IWRAM; a draining meter ramps like a coordinate, so these
# are the ones Tim flagged to filter.
KNOWN = {
    32 * 1024 + 0x0234C: "maxHP", 32 * 1024 + 0x0234D: "HP",
    32 * 1024 + 0x0234F: "keys", 32 * 1024 + 0x02340: "rupees.lo",
    32 * 1024 + 0x02341: "rupees.hi", 32 * 1024 + 0x02342: "rupees.disp",
    0x038F4: "X.lo", 0x038F5: "X.hi", 0x038F0: "Y.lo", 0x038F1: "Y.hi",
    0x0391C: "Xmir.lo", 0x0391D: "Xmir.hi", 0x03920: "Ymir.lo", 0x03921: "Ymir.hi",
    0x03838: "posmir",
}


def _unimodal(ring):
    """Per-column: non-decreasing up to a single peak, then non-increasing.

    A drained/refilled meter is unimodal (fill to a peak, spend to zero); a
    plateau at the top is allowed. Oscillating sprite/animation bytes -- the junk
    yesterday's search surfaced -- wiggle up AND down repeatedly and fail this.
    Strict monotonicity was too brittle: 201000 refills THEN drains, so the meter
    is not monotone in either direction, only unimodal.
    """
    r = ring.astype(np.int32)
    d = np.diff(r, axis=0)                       # (n-1, cols)
    neg = (d < 0).astype(np.int32)
    prior_fall = (np.cumsum(neg, axis=0) - neg) > 0   # any fall STRICTLY before k
    rises_after_fall = ((d > 0) & prior_fall).any(axis=0)
    return ~rises_after_fall


def drain_to_zero(ring, cap):
    """start >0, end ==0, bounded, moved, and a single down-slope (unimodal)."""
    first, last = ring[0].astype(np.int32), ring[-1].astype(np.int32)
    moved = ring.max(axis=0) != ring.min(axis=0)
    bounded = ring.max(axis=0) <= cap
    return np.flatnonzero((first > 0) & (last == 0) & moved & bounded
                          & _unimodal(ring))


def excursion_from_zero(ring, cap, floor=8):
    """start ==0, bulge up to a real peak (>= floor), come back -- unimodal.

    end value is NOT constrained: 201000 refills to 32 then Tim drains it back to
    0, so the meter ends where it started. What identifies it is the bounded,
    single-peaked bulge out of zero, not where the window happens to stop.
    """
    first = ring[0].astype(np.int32)
    peak = ring.max(axis=0).astype(np.int32)
    bounded = peak <= cap
    return np.flatnonzero((first == 0) & (peak >= floor) & bounded
                          & _unimodal(ring))


def seq(ring, addr, cap=12):
    col = ring[:, addr]
    edge = np.flatnonzero(col[1:] != col[:-1]) + 1
    b = np.concatenate(([0], edge, [col.size]))
    vals = [int(col[x]) for x in b[:-1]]
    s = " ".join(str(v) for v in vals[:cap])
    if len(vals) > cap:
        s += " ...+%d" % (len(vals) - cap)
    return s


def main(drain_dump, rise_dump, cap=128):
    _m, dr = load(drain_dump)
    _m, ri = load(rise_dump)
    drain = set(int(a) for a in drain_to_zero(dr, cap))
    rise = set(int(a) for a in excursion_from_zero(ri, cap))
    print("DRAIN dump %s: %d bytes drain to exactly 0 (unimodal, <=%d)"
          % (os.path.basename(os.path.normpath(drain_dump)), len(drain), cap))
    print("RISE  dump %s: %d bytes bulge up out of 0 (unimodal, <=%d)"
          % (os.path.basename(os.path.normpath(rise_dump)), len(rise), cap))
    both = sorted(drain & rise)
    print("\nBYTES THAT DO BOTH (drain-to-0 AND excursion-from-0): %d\n" % len(both))
    print("%-16s %-10s  %-28s  %s" % ("ADDR", "known?", "drain (200818)", "rise (201000)"))
    for a in both:
        tag = KNOWN.get(a, "")
        print("%-16s %-10s  %-28s  %s" % (fmt_addr(a), tag,
              seq(dr, a), seq(ri, a)))
    unknown = [a for a in both if a not in KNOWN]
    print("\n%d of the %d are NOT already-known addresses." % (len(unknown), len(both)))
    if len(unknown) == 1:
        print("UNIQUE unknown candidate: %s" % fmt_addr(unknown[0]))
    return 0


if __name__ == "__main__":
    # args: <drain_dump> <rise_dump>
    sys.exit(main(sys.argv[1], sys.argv[2]))
