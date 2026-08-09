"""Room ID, using the strongest constraint the capture actually provides.

find_room_id.py returned 3408 candidates because "settled + few labels" is weak.
This uses the session's real structure instead, which Tim's description plus the
churn measurement pins down exactly:

  ANCHOR dumps -- 160918..160944, all in ONE room (key pickup and key use
    happened in the same room, and their RAM churn is baseline ~2300).
    A room ID MUST hold ONE identical value across every one of them.
  TRANSITION dump -- 160955, churn 5550 (2.4x baseline) = a door was crossed.
    A room ID MUST change here, FROM the anchor value, and settle.

That conjunction is what 3 static snapshots could never express. Anything that
merely wobbles is killed by the anchor; anything that ignores the door is killed
by the transition.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load, REGIONS  # noqa: E402

TAIL = 10


def name(off):
    for reg, base, size in REGIONS:
        if base <= off < base + size:
            return "%s 0x%05X" % (reg, off - base)
    return "?"


def main(argv):
    trans = argv[0]
    anchors = argv[1:]

    # constraint 1: identical, and rock-steady, across every anchor dump
    const = None
    val = None
    for d in sorted(anchors):
        _m, ring = load(d)
        steady = (ring == ring[0]).all(axis=0)
        const = steady if const is None else (const & steady)
        # int16, not uint8: 256 is the "disagreed between anchors" sentinel and
        # would silently wrap to 0 in the ring's native dtype.
        first = ring[0].astype(np.int16)
        val = first if val is None else np.where(val == first, val, 256)
    same = const & (val != 256)

    # constraint 2: in the transition dump it LEAVES that value and settles
    _m, tring = load(trans)
    starts_right = tring[0] == val
    tail = tring[-TAIL:, :]
    settled = (tail == tail[0]).all(axis=0)
    moved = tail[0] != tring[0]
    cand = np.flatnonzero(same & starts_right & settled & moved)

    print("anchors: %d dumps   transition: %s"
          % (len(anchors), os.path.basename(trans.rstrip("/"))))
    print("candidates: %d\n" % cand.size)
    print("%-14s %8s %8s" % ("ADDR", "in_room", "after_door"))
    for off in cand[:40]:
        print("%-14s %8d %8d" % (name(off), int(tring[0][off]), int(tail[0][off])))
    if not cand.size:
        print("none -- the ID may be 16-bit, or the door was crossed outside "
              "the 4s window in every anchor dump")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
