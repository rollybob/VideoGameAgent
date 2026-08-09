"""Find a room/screen ID byte across several capture dumps.

WHY NOT ram_ring_diff --mode roundtrip: its scoring is a COUNTER fingerprint
(+/-1, single digits, touching zero), which is right for keys and actively wrong
here -- a room ID is an arbitrary label like 0x1B -> 0x52, and 0/1 animation
toggles outrank it every time. Ran that first and got four 0/1 flags on top.

THE SHAPE OF A ROOM ID, stated as constraints instead:
  1. SETTLED: constant across the tail of each dump (F11 is pressed after the
     screen settles, so the last samples are all in the new room).
  2. FEW LABELS: across N dumps it takes only a handful of distinct values --
     Tim walked between a small number of rooms, not N different ones.
  3. IT MOVED: it actually changed inside at least one dump, i.e. we caught a
     transition. This is the constraint 3 static snapshots could not apply, and
     it is what cut 11,244 candidates last time.
  4. REVISITED: at least one value appears in 2+ dumps. Walking back through a
     door must reproduce the earlier label; a byte that never repeats is a
     timer or a counter, not an identifier.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load, REGIONS  # noqa: E402

TAIL = 10          # samples at the end of a dump that must agree (settled)
MAX_LABELS = 4     # at most this many distinct values across all dumps


def main(dumps):
    dumps = sorted(dumps)
    tails, moved = [], None
    for d in dumps:
        _meta, ring = load(d)
        tail = ring[-TAIL:, :]
        settled = (tail == tail[0]).all(axis=0)          # constraint 1
        tails.append(np.where(settled, tail[0], 255 - tail[0].astype(np.int16)))
        changed = (ring != ring[0]).any(axis=0)          # constraint 3
        moved = changed if moved is None else (moved | changed)
        if len(tails) == 1:
            settled_all = settled
        else:
            settled_all &= settled

    stack = np.stack(tails)                              # (n_dumps, n_bytes)
    labels = np.array([len(set(stack[:, i].tolist())) for i in range(stack.shape[1])])
    repeats = np.array([len(set(stack[:, i].tolist())) < stack.shape[0]
                        for i in range(stack.shape[1])])  # constraint 4

    cand = np.flatnonzero(settled_all & moved & repeats
                          & (labels > 1) & (labels <= MAX_LABELS))
    print("dumps: %d   candidates: %d\n" % (len(dumps), cand.size))

    def name(off):
        for reg, base, size in REGIONS:
            if base <= off < base + size:
                return "%s 0x%05X" % (reg, off - base)
        return "?"

    rows = sorted(cand, key=lambda o: (labels[o], o))
    print("%-14s %7s  %s" % ("ADDR", "labels", "value per dump (in time order)"))
    for off in rows[:25]:
        print("%-14s %7d  %s" % (name(off), labels[off],
                                 " ".join("%3d" % v for v in stack[:, off])))
    if not cand.size:
        print("none -- loosen MAX_LABELS or check the dumps really span a door")
    else:
        print("\nWant: a byte whose values REPEAT as Tim walked back and forth.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
