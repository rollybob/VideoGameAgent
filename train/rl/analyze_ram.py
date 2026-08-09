"""Task 09 A0: mine reward-relevant RAM addresses from ram_capture.py dumps.

Scans every EWRAM/IWRAM byte's per-tick time series for the signatures of
game counters, then intersects candidates across tapes (a real counter lives
at the same address in every session):

  rupee-like: sparse changes; deltas mostly positive and drawn from the
              rupee denomination set, OR long +1 runs (a display counter
              that counts up toward the target, ALttP-style).
  heart-like: sparse changes; small value range; at least one negative delta
              (damage taken).

Prints a ranked report; use --at ADDR to dump one address's full time series
(tick, old -> new) for eyeballing against the periodic PNGs.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/analyze_ram.py scratch/ram/<tape1> <tape2> ...
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EWRAM_SIZE = 256 * 1024
IWRAM_SIZE = 32 * 1024
CHUNK = 32 * 1024
RUPEE_DELTAS = {1, 5, 10, 20, 50, 100, 200, 300}


def load(dump, region):
    with open(os.path.join(dump, "ticks.json")) as f:
        n = json.load(f)["ticks"]
    size = EWRAM_SIZE if region == "ewram" else IWRAM_SIZE
    return np.memmap(os.path.join(dump, region + ".u8"), dtype=np.uint8,
                     mode="r", shape=(n, size))


def changing_addrs(arr, min_ch, max_ch):
    """addresses whose byte value changes within [min_ch, max_ch] times"""
    n, size = arr.shape
    out = {}
    for lo in range(0, size, CHUNK):
        hi = min(lo + CHUNK, size)
        block = np.asarray(arr[:, lo:hi], dtype=np.uint8)
        ch = (np.diff(block.astype(np.int16), axis=0) != 0).sum(axis=0)
        for off in np.nonzero((ch >= min_ch) & (ch <= max_ch))[0]:
            out[lo + int(off)] = int(ch[off])
    return out


def series(arr, addr):
    col = np.asarray(arr[:, addr], dtype=np.int16)
    d = np.diff(col)
    ticks = np.nonzero(d)[0] + 1
    return col, [(int(t), int(col[t - 1]), int(col[t])) for t in ticks]


def classify(col, events):
    deltas = [new - old for _, old, new in events]
    pos = [d for d in deltas if d > 0]
    neg = [d for d in deltas if d < 0]
    vmax = int(col.max())
    # display-counter: mostly +-1 steps in runs
    ones = sum(1 for d in deltas if abs(d) == 1)
    tags = []
    if pos and all(d in RUPEE_DELTAS for d in pos) and len(pos) >= 2:
        tags.append("rupee-target")
    if ones >= 0.8 * len(deltas) and len(deltas) >= 4:
        tags.append("display-run")
    if neg and vmax <= 80 and len(events) <= 40:
        tags.append("heart-like")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--region", choices=["ewram", "iwram", "both"], default="both")
    ap.add_argument("--min-ch", type=int, default=2)
    ap.add_argument("--max-ch", type=int, default=400)
    ap.add_argument("--at", type=lambda s: int(s, 0), default=None,
                    help="dump one address's change list per tape (0x-prefixed ok)")
    ap.add_argument("--at-region", choices=["ewram", "iwram"], default="ewram")
    args = ap.parse_args()
    dumps = [d if os.path.isabs(d) else os.path.join(HERE, d) for d in args.dumps]

    if args.at is not None:
        for d in dumps:
            arr = load(d, args.at_region)
            _, ev = series(arr, args.at)
            print("%s %s[0x%05X]: %d changes" % (os.path.basename(d),
                  args.at_region, args.at, len(ev)))
            for t, old, new in ev[:80]:
                print("  tick %5d: %3d -> %3d (%+d)" % (t, old, new, new - old))
        return

    regions = ["ewram", "iwram"] if args.region == "both" else [args.region]
    for region in regions:
        per_tape = []
        for d in dumps:
            arr = load(d, region)
            per_tape.append(changing_addrs(arr, args.min_ch, args.max_ch))
            print("%s %s: %d changing addrs in [%d,%d]" % (os.path.basename(d),
                  region, len(per_tape[-1]), args.min_ch, args.max_ch))
        common = set(per_tape[0])
        for ch in per_tape[1:]:
            common &= set(ch)
        print("%s: %d addresses change in ALL %d tapes" % (region, len(common), len(dumps)))

        rows = []
        arrs = [load(d, region) for d in dumps]
        for addr in sorted(common):
            tag_sets, ev_counts, samples = [], [], []
            for arr in arrs:
                col, ev = series(arr, addr)
                tag_sets.append(set(classify(col, ev)))
                ev_counts.append(len(ev))
                samples.append(ev[:3])
            tags = set.intersection(*tag_sets)
            if tags:
                rows.append((addr, sorted(tags), ev_counts, samples[0]))
        rows.sort(key=lambda r: (r[1], r[0]))
        print("\n== %s candidates tagged in ALL tapes ==" % region)
        for addr, tags, counts, sample in rows:
            print("0x%05X %-28s changes/tape=%s  first: %s"
                  % (addr, ",".join(tags), counts, sample))


if __name__ == "__main__":
    main()
