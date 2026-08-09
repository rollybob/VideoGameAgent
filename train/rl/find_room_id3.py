"""Room ID: intersect ROUND TRIPS across two independent loop captures.

Tim walked a loop of 3 room transitions ending where he started, twice, with a
10 s ring so each loop fits in ONE window. That makes the room ID a within-dump
round trip -- but ram_ring_diff's roundtrip scoring is a COUNTER fingerprint
(+/-1, single digits, touching zero), so 0/1 animation toggles flood the top and
the two dumps surface almost entirely different addresses.

TWO CONSTRAINTS THIS ADDS, neither expressible in a single-dump ranking:

  1. INTERSECTION. A room ID must round-trip in BOTH loops. The 0/1 toggles do
     not -- they are per-dump noise, which is visible in the raw output.
  2. IDENTIFIER SCORING, not counter scoring. A room label is an arbitrary byte
     that HOLDS for seconds; 0/1 is the signature of an animation flag. So dwell
     is rewarded and 0/1 pairs are pushed down rather than up.

Both loops started in the same place, so a genuine ID should also show the SAME
home value in both dumps -- reported, and used to rank.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load, prefilter, within_round_trip, REGIONS  # noqa: E402


def name(off):
    for reg, base, size in REGIONS:
        if base <= off < base + size:
            return "%s 0x%05X" % (reg, off - base)
    return "?"


def trips(dump, min_dwell, min_rooms):
    """Round trips, but counting DISTINCT LONG-DWELLING VALUES.

    within_round_trip reports only v1 and the longest middle value, which throws
    away the structure that makes a 3-transition loop informative. Tim walked
    A -> B -> C -> A, so a real room ID holds THREE distinct values for a second
    or more each. An animation flag toggles between TWO however long you watch.
    Requiring >= min_rooms distinct dwelling values is the constraint that a
    two-value round trip cannot fake.
    """
    from ram_ring_diff import runs
    _meta, ring = load(dump)
    out = {}
    for addr in prefilter(ring, min_dwell, same_ends=True):
        col = ring[:, addr]
        got = within_round_trip(col, min_dwell)
        if got is None:
            continue
        held = [v for v, _s, ln in runs(col) if ln >= min_dwell]
        distinct = len(set(held))
        if distinct < min_rooms:
            continue
        v1, v2, nch, dwell = got
        out[int(addr)] = (v1, v2, nch, dwell, distinct, tuple(held))
    return out


def main(argv):
    min_dwell = 30
    min_rooms = 3          # A -> B -> C -> A holds three distinct values
    dumps = [a for a in argv if not a.startswith("-")]
    per = [trips(d, min_dwell, min_rooms) for d in dumps]
    for d, t in zip(dumps, per):
        print("  %-26s %d round trips" % (os.path.basename(d.rstrip("/")), len(t)))

    common = set(per[0])
    for t in per[1:]:
        common &= set(t)
    print("\nround-trip in ALL %d dumps: %d addresses" % (len(dumps), len(common)))

    rows = []
    for addr in common:
        vals = [t[addr] for t in per]
        same_home = len(set(v[0] for v in vals)) == 1
        binary = all(set([v[0], v[1]]) <= {0, 1} for v in vals)
        dwell = min(v[3] for v in vals)
        rooms = min(v[4] for v in vals)
        # identifier fingerprint: MANY distinct values that each hold, not a
        # two-state flag. Distinct-value count dominates because that is the
        # part a toggling animation byte cannot imitate.
        score = rooms * 10 + dwell // 10 + (0 if binary else 8) + (4 if same_home else 0)
        rows.append((score, addr, vals, binary, same_home, dwell, rooms))
    rows.sort(key=lambda r: (-r[0], r[1]))

    print("\n%-14s %5s %6s %6s  %s"
          % ("ADDR", "score", "rooms", "dwell", "values held per dump"))
    for score, addr, vals, binary, same_home, dwell, rooms in rows[:25]:
        detail = "  |  ".join(",".join(str(x) for x in v[5]) for v in vals)
        print("%-14s %5d %6d %6d  %s" % (name(addr), score, rooms, dwell, detail))
    if not rows:
        print("none -- the two loops may not share a starting room")
    else:
        nonbin = [r for r in rows if not r[3]]
        print("\nnon-binary candidates (the plausible IDs): %d" % len(nonbin))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
