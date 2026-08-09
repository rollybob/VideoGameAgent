"""Room/screen ID as a 16-BIT value.

Byte-wise search failed three times (3408, then 572, then 41 candidates, all
either graphics buffers or bytes that failed the cross-dump test). ALttP's room
index is conventionally 16-bit, and the tooling is entirely byte-oriented, so a
two-byte ID is split across two columns and neither half need look like a clean
identifier on its own.

WHY 16-BIT IS NOT JUST A CORRECTNESS FIX -- IT IS THE DISCRIMINATOR:
a byte returning to its exact previous value happens by chance about 1 in 256;
a 16-bit pair returning to its exact previous value happens about 1 in 65536.
The round-trip constraint therefore becomes ~256x more selective, and
selectivity is precisely what the byte-wise attempts lacked.

CONSTRAINTS, in the order they cut the space:
  1. ROUND TRIP: v1 -> v2 -> ... -> v1 inside one window, with real dwell at
     each end and at the middle value. Tim walked a loop of 3 transitions and
     returned to the starting room, so a real ID must come home.
  2. MULTIPLE ROOMS: >= 3 distinct values that each HOLD. A two-state flag
     cannot fake this however long you watch it.
  3. PLAUSIBLE RANGE: ALttP has on the order of a few hundred rooms, so a real
     ID stays under ~0x200. This kills the graphics buffers, which take large
     arbitrary values.
  4. CROSS-DUMP: must be CONSTANT across dumps taken in one room, and DIFFER in
     dumps from elsewhere. This is the test that confirmed the key counter and
     that every byte-wise room candidate failed.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load, runs, REGIONS  # noqa: E402

MAX_ROOM_ID = 0x200
MIN_DWELL = 30          # 1 second at every=2
MIN_ROOMS = 3


def name16(off, little=True):
    for reg, base, size in REGIONS:
        if base <= off < base + size:
            return "%s 0x%05X%s" % (reg, off - base, "" if little else " BE")
    return "?"


def u16_view(ring, little=True):
    """Whole ring as overlapping u16 columns. ~2x memory, still comfortable."""
    lo = ring[:, :-1].astype(np.uint16)
    hi = ring[:, 1:].astype(np.uint16)
    return (lo | (hi << 8)) if little else (hi | (lo << 8))


def loop_candidates(view):
    """Round trip with >= MIN_ROOMS distinct dwelling values, all in range."""
    out = {}
    n = view.shape[0]
    # cheap vectorised prefilter: ends equal, held at both ends, actually moved
    head = np.all(view[:MIN_DWELL] == view[0], axis=0)
    tail = np.all(view[-MIN_DWELL:] == view[-1], axis=0)
    ends = view[0] == view[-1]
    moved = view.max(axis=0) != view.min(axis=0)
    inrange = view.max(axis=0) < MAX_ROOM_ID
    for addr in np.flatnonzero(head & tail & ends & moved & inrange):
        r = runs(view[:, addr])
        held = [v for v, _s, ln in r if ln >= MIN_DWELL]
        distinct = sorted(set(held))
        if len(distinct) < MIN_ROOMS:
            continue
        out[int(addr)] = (tuple(held), tuple(distinct))
    return out


def main(argv):
    loop = argv[0]
    same_room = [a for a in argv[1:] if a]
    for little in (True, False):
        _m, ring = load(loop)
        view = u16_view(ring, little)
        cands = loop_candidates(view)
        print("\n=== %s-endian: %d candidates ===" % ("little" if little else "big",
                                                      len(cands)))
        rows = []
        for addr, (held, distinct) in cands.items():
            # constraint 4: steady within each single-room dump
            steady_vals = []
            ok = True
            for d in same_room:
                _m2, r2 = load(d)
                v2 = u16_view(r2, little)[:, addr]
                u = set(int(x) for x in v2)
                if len(u) != 1:
                    ok = False
                    break
                steady_vals.append(u.pop())
            if not ok:
                continue
            rows.append((len(distinct), addr, held, distinct, steady_vals))
        rows.sort(key=lambda r: (-r[0], r[1]))
        print("%-16s %6s  %-28s %s" % ("ADDR", "rooms", "sequence held", "value in same-room dumps"))
        for nd, addr, held, distinct, steady in rows[:20]:
            print("%-16s %6d  %-28s %s"
                  % (name16(addr, little), nd,
                     ",".join(str(v) for v in held[:8]),
                     ",".join(str(v) for v in steady)))
        if not rows:
            print("(none survived the cross-dump steadiness test)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
