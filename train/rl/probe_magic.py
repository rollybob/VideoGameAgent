"""Probe the save-field cluster (esp. the PREDICTED magic byte) across dumps.

Magic in the SNES ALttP SRAM lives at 0x7EF36E, one byte after current HP
(0x7EF36D) and one before the key/next fields. This GBA port mirrors that SRAM
block into EWRAM at a fixed offset -- proven by four already-confirmed fields:

    field        GBA EWRAM   SNES SRAM    GBA = SNES - 0x7ECD20
    rupees u16   0x02340     0x7EF360     ok
    max HP       0x0234C     0x7EF36C     ok
    cur HP       0x0234D     0x7EF36D     ok
    keys         0x0234F     (cluster)    ok

So magic is PREDICTED at EWRAM 0x0234E (0x7EF36E - 0x7ECD20). This script does
NOT assume that -- it prints the raw run-length-encoded sequence of 0x0234E and
its neighbours in every dump, so the two-jars-then-drain-to-zero signature (if
present) is visible directly and I can see which capture is the magic one.

Run-length encoding, not raw 300-sample dumps, because a smoothly draining meter
is dozens of 1-unit steps and the RLE makes the rise/rise/fall shape legible.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load  # noqa: E402

EW = 32 * 1024  # column offset where EWRAM starts in a snapshot row


def rle(col):
    edge = np.flatnonzero(col[1:] != col[:-1]) + 1
    bounds = np.concatenate(([0], edge, [col.size]))
    return [(int(col[b]), int(e - b)) for b, e in zip(bounds[:-1], bounds[1:])]


def fmt_rle(col, cap=16):
    r = rle(col)
    s = " ".join("%d(x%d)" % (v, n) for v, n in r[:cap])
    if len(r) > cap:
        s += " ...+%d more" % (len(r) - cap)
    return "runs=%d  %s" % (len(r), s)


def u16col(ring, ew_off):
    c = EW + ew_off
    return ring[:, c].astype(np.int32) | (ring[:, c + 1].astype(np.int32) << 8)


def main(dumps):
    for d in dumps:
        _m, ring = load(d)
        name = os.path.basename(os.path.normpath(d))
        print("\n===== %s  (%d snaps) =====" % (name, ring.shape[0]))
        # the single-byte save fields around the predicted magic byte
        for off, tag in ((0x0234C, "maxHP"), (0x0234D, "HP   "),
                         (0x0234E, "MAGIC?"), (0x0234F, "keys "),
                         (0x02350, "+0x50"), (0x02351, "+0x51")):
            col = ring[:, EW + off]
            print("  EWRAM 0x%05X %-6s  %s" % (off, tag, fmt_rle(col)))
        # u16 references (rupees, and both position axes) as a sanity check
        for off, tag in ((0x02340, "rupees"),):
            v = u16col(ring, off)
            print("  EWRAM 0x%05X %-6s  %s" % (off, tag, fmt_rle(v)))
        for ioff, tag in ((0x038F4, "X u16"), (0x038F0, "Y u16")):
            c = ioff
            v = ring[:, c].astype(np.int32) | (ring[:, c + 1].astype(np.int32) << 8)
            print("  IWRAM 0x%05X %-6s  spread=%d" % (ioff, tag,
                  int(v.max() - v.min())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
