"""Characterise each capture dump from its RAM, not from eyeballing screenshots.

Prints a run-length view (NOT a sampled one -- a [::10] view is what made the
real health address look like a sawtooth counter on 2026-07-31) of the known
addresses in every dump, so dumps can be mapped to the events Tim described and
paired correctly for --mode roundtrip.

Also directly tests two things Tim flagged 2026-08-01:
  - max HP is NOT fixed at 24; heart containers raise it. If 0x0234C really is
    max HP it should be constant WITHIN a dump and may differ ACROSS dumps.
  - damage can be a QUARTER heart (bees). In eighths that is a delta of 2, which
    the ram_ring_diff heart-quantised scoring (%8, %4) would not credit.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load  # noqa: E402

EW = 32 * 1024
HP, HP_MAX, RUPEE = 0x0234D, 0x0234C, 0x02340


def rle(col, limit=14):
    """Value sequence with consecutive duplicates collapsed."""
    vals = [int(col[0])]
    for v in col[1:]:
        if int(v) != vals[-1]:
            vals.append(int(v))
    if len(vals) > limit:
        return "%s ... %s (%d changes)" % (
            " ".join(str(v) for v in vals[:6]),
            " ".join(str(v) for v in vals[-4:]), len(vals) - 1)
    return " ".join(str(v) for v in vals)


def main(dumps):
    print("%-26s %6s  %-30s %-22s %s"
          % ("dump", "maxHP", "HP 0x0234D (eighths)", "rupees", "deltas"))
    for d in sorted(dumps):
        meta, ring = load(d)
        hp = ring[:, EW + HP]
        mx = ring[:, EW + HP_MAX]
        rup = (ring[:, EW + RUPEE].astype(np.int32)
               | (ring[:, EW + RUPEE + 1].astype(np.int32) << 8))
        # deltas between consecutive DISTINCT values, to expose quarter-hearts
        seq = [int(hp[0])]
        for v in hp[1:]:
            if int(v) != seq[-1]:
                seq.append(int(v))
        deltas = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
        mxs = sorted(set(int(v) for v in mx))
        print("%-26s %6s  %-30s %-22s %s"
              % (os.path.basename(os.path.normpath(d)),
                 ",".join(str(v) for v in mxs),
                 rle(hp), rle(rup), " ".join("%+d" % x for x in deltas[:10])))

    print("\nquarter-heart check: any HP delta of exactly +/-2 above is a quarter")
    print("max-HP check: a column with >1 distinct maxHP value changed mid-dump")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
