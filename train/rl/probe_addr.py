"""Track specific addresses across every dump -- hypothesis testing, not ranking.

ram_ring_diff.py ranks candidates within a dump. Once a dump has NAMED a
suspect, the decisive question is different: does that byte behave like the
counter across ALL the captures, including the ones where it should NOT move?
A key counter must go 0 -> 1 on the pickup, 1 -> 0 on the spend, and sit still
everywhere else. Ranking cannot answer that; this can.

  probe_addr.py EWRAM:0x0234F IWRAM:0x019D9 -- <dump> [<dump> ...]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load  # noqa: E402

EW = 32 * 1024


def parse(spec):
    region, _, off = spec.partition(":")
    base = EW if region.upper() == "EWRAM" else 0
    return spec, base + int(off, 0)


def rle(col):
    vals = [int(col[0])]
    for v in col[1:]:
        if int(v) != vals[-1]:
            vals.append(int(v))
    if len(vals) > 12:
        return "%s ...(%d changes)" % (
            " ".join(str(v) for v in vals[:8]), len(vals) - 1)
    return " ".join(str(v) for v in vals)


def main(argv):
    if "--" in argv:
        i = argv.index("--")
        specs, dumps = argv[:i], argv[i + 1:]
    else:
        specs, dumps = argv[:1], argv[1:]
    cols = [parse(s) for s in specs]

    print("%-26s %s" % ("dump", "  ".join("%-22s" % s for s, _ in cols)))
    for d in sorted(dumps):
        _meta, ring = load(d)
        cells = [rle(ring[:, off]) for _s, off in cols]
        print("%-26s %s" % (os.path.basename(os.path.normpath(d)),
                            "  ".join("%-22s" % c for c in cells)))
    print("\nA real counter moves ONLY on the dumps where the event happened.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
