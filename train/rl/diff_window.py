"""Task 09 A0: windowed RAM diff for probe/capture dumps.

Lists addresses that changed inside an event window but were SILENT during a
quiet reference window -- filters out the free-running animation/timer noise
that drowns naive diffing. Shows each survivor's value at window edges plus
its full change list inside the window.

  python3 diff_window.py scratch/probe/touch --window 60,240 --quiet 0,55
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = {"ewram": 256 * 1024, "iwram": 32 * 1024}
CHUNK = 32 * 1024


def load(dump, region):
    with open(os.path.join(dump, "ticks.json")) as f:
        n = json.load(f)["ticks"]
    return np.memmap(os.path.join(dump, region + ".u8"), dtype=np.uint8,
                     mode="r", shape=(n, SIZES[region]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--window", required=True)
    ap.add_argument("--quiet", required=True)
    ap.add_argument("--region", choices=["ewram", "iwram", "both"], default="both")
    ap.add_argument("--max-changes", type=int, default=12,
                    help="drop addrs changing more than this inside the window")
    args = ap.parse_args()
    dump = args.dump if os.path.isabs(args.dump) else os.path.join(HERE, args.dump)
    w0, w1 = map(int, args.window.split(","))
    q0, q1 = map(int, args.quiet.split(","))

    regions = ["ewram", "iwram"] if args.region == "both" else [args.region]
    for region in regions:
        arr = load(dump, region)
        size = SIZES[region]
        hits = []
        for lo in range(0, size, CHUNK):
            hi = min(lo + CHUNK, size)
            win = np.asarray(arr[w0:w1, lo:hi]).astype(np.int16)
            qui = np.asarray(arr[q0:q1, lo:hi]).astype(np.int16)
            wch = (np.diff(win, axis=0) != 0).sum(axis=0)
            qch = (np.diff(qui, axis=0) != 0).sum(axis=0)
            sel = np.nonzero((wch >= 1) & (wch <= args.max_changes) & (qch == 0))[0]
            for off in sel:
                hits.append(lo + int(off))
        print("== %s: %d addrs changed in [%d,%d) and were quiet in [%d,%d) =="
              % (region, len(hits), w0, w1, q0, q1))
        for addr in hits:
            col = np.asarray(arr[:, addr], dtype=np.int16)
            d = np.diff(col[w0:w1])
            ticks = np.nonzero(d)[0] + w0 + 1
            evs = ["t%d:%d->%d" % (t, col[t - 1], col[t]) for t in ticks]
            print("0x%05X  %s" % (addr, "  ".join(evs[:10])))


if __name__ == "__main__":
    main()
