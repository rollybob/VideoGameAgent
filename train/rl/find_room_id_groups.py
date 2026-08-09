"""Room ID from GROUPED dumps -- no loop assumption at all.

Every previous attempt leaned on a round trip inside one window, which needs the
human's loop to have closed inside ~10 s. That assumption may simply be false,
and it is not needed: the capture set already contains 16 dumps whose room
membership is KNOWN from Tim's description plus the rupee/key/HP fingerprints.

A room identifier must satisfy, using only settled values:
  1. SETTLED     constant across the tail of each dump (the room ended in).
  2. AGREES      identical across every dump known to be in the SAME room. The
                 key room has EIGHT such dumps, which is a far stronger
                 constraint than any single-window pattern.
  3. DISCRIMINATES  takes >= MIN_DISTINCT different values across the groups.
                 A byte that is 127 everywhere (the 16-bit tilemap candidates)
                 dies here, which is exactly how those were rejected.
  4. IN RANGE    ALttP has on the order of a few hundred rooms.

Checks u8 and u16 (both endiannesses) in one pass.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load, REGIONS  # noqa: E402

TAIL = 20
MAX_ID = 0x200
MIN_DISTINCT = 3

# Room membership. The eight key-room dumps are the anchor: Tim confirmed the
# key pickup and its spend happened in one room, and the surrounding marks are
# the same session in the same place.
GROUPS = {
    "keyroom": ["160918", "160922", "160935", "160940", "160944", "160955",
                "161017", "161030"],
    "areaB":   ["161224"],
    "areaC":   ["162512"],
    "areaD":   ["162718"],
    "areaE":   ["162940"],
    "loop":    ["181124", "181140"],
    "bee":     ["181342", "181359"],
}


def name(off, width, little):
    for reg, base, size in REGIONS:
        if base <= off < base + size:
            tag = "" if width == 1 else (" u16" if little else " u16BE")
            return "%s 0x%05X%s" % (reg, off - base, tag)
    return "?"


def tail_values(dump, width, little):
    """Settled tail value per address, or None-marker where it is not settled."""
    _m, ring = load(dump)
    t = ring[-TAIL:, :]
    if width == 1:
        v = t[0].astype(np.int32)
        steady = np.all(t == t[0], axis=0)
    else:
        lo = t[:, :-1].astype(np.uint16)
        hi = t[:, 1:].astype(np.uint16)
        w = (lo | (hi << 8)) if little else (hi | (lo << 8))
        v = w[0].astype(np.int32)
        steady = np.all(w == w[0], axis=0)
        v = np.append(v, -1)
        steady = np.append(steady, False)
    return np.where(steady, v, -1)


def main(root):
    def path(stamp):
        return os.path.join(root, "20260801-%s-hit" % stamp)

    for width, little in ((1, True), (2, True), (2, False)):
        per_group = {}
        for g, stamps in GROUPS.items():
            vals = [tail_values(path(s), width, little) for s in stamps]
            stack = np.stack(vals)
            agree = np.all(stack == stack[0], axis=0) & (stack[0] >= 0)
            per_group[g] = (stack[0], agree)

        ok = None
        for g, (v, agree) in per_group.items():
            ok = agree if ok is None else (ok & agree)
        vals = np.stack([per_group[g][0] for g in GROUPS])
        distinct = np.array([len(set(vals[:, i].tolist())) for i in range(vals.shape[1])])
        inrange = vals.max(axis=0) < MAX_ID
        cand = np.flatnonzero(ok & inrange & (distinct >= MIN_DISTINCT))

        label = "u8" if width == 1 else ("u16 LE" if little else "u16 BE")
        print("\n=== %s: %d candidates ===" % (label, cand.size))
        rows = sorted(cand, key=lambda o: (-distinct[o], o))
        if rows:
            print("%-18s %8s  %s" % ("ADDR", "distinct", "  ".join("%-8s" % g for g in GROUPS)))
        for off in rows[:15]:
            print("%-18s %8d  %s"
                  % (name(off, width, little), distinct[off],
                     "  ".join("%-8d" % v for v in vals[:, off])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
