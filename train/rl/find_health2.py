"""Task 09 A0: pinpoint solo-ALttP health via the eighths fingerprint.

The HUD shows 3 hearts (full = 24 in eighths; half-heart = 4). So the health
byte's DISTINCT values are all multiples of 4, its max is small (a few hearts),
it sits at its max most of the time, and it dips on damage. That fingerprint is
far more specific than "has a negative delta" and cuts through the position/
animation/OAM noise. Votes across tapes and prints each survivor's value
histogram so the heart structure is visible numerically (no pixel eyeballing).

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/find_health2.py scratch/ram/<t1> scratch/ram/<t2> ...
"""
import argparse
import collections
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


def fingerprint(arr, max_hp, min_modal_frac):
    """addr -> (modal, modal_frac, sorted distinct values) for health-like bytes:
    all distinct values are multiples of 4, max in [8,max_hp], not constant,
    and the modal value is held at least min_modal_frac of the time."""
    n, size = arr.shape
    out = {}
    for lo in range(0, size, CHUNK):
        hi = min(lo + CHUNK, size)
        block = np.asarray(arr[:, lo:hi], dtype=np.uint8)
        vmax = block.max(axis=0)
        vmin = block.min(axis=0)
        cand = np.nonzero((vmax >= 8) & (vmax <= max_hp) & (vmin < vmax))[0]
        for off in cand:
            col = block[:, off]
            vals = np.unique(col)
            if np.any(vals % 4 != 0):
                continue
            counts = np.bincount(col)
            modal = int(counts.argmax())
            frac = counts[modal] / float(n)
            if modal == 0 or frac < min_modal_frac:
                continue
            out[lo + int(off)] = (modal, round(float(frac), 2),
                                  [int(v) for v in vals])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--region", choices=["ewram", "iwram", "both"], default="both")
    ap.add_argument("--max-hp", type=int, default=48,
                    help="reject addrs whose max exceeds this (health is small)")
    ap.add_argument("--min-modal-frac", type=float, default=0.4,
                    help="health sits at full most of the time")
    args = ap.parse_args()
    dumps = [d if os.path.isabs(d) else os.path.join(HERE, d) for d in args.dumps]
    regions = ["ewram", "iwram"] if args.region == "both" else [args.region]

    for region in regions:
        votes = collections.Counter()
        detail = collections.defaultdict(list)
        for d in dumps:
            arr = load(d, region)
            for a, info in fingerprint(arr, args.max_hp, args.min_modal_frac).items():
                votes[a] += 1
                detail[a].append(info)
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        print("\n== %s: eighths-fingerprint health candidates (mult-of-4, max<=%d, "
              "modal>=%.0f%%), voted across %d tapes ==" % (
                  region, args.max_hp, 100 * args.min_modal_frac, len(dumps)))
        for a, v in ranked:
            if v < max(2, len(dumps) // 2):
                continue
            modals = [i[0] for i in detail[a]]
            fracs = [i[1] for i in detail[a]]
            allvals = sorted(set(x for i in detail[a] for x in i[2]))
            print("0x%05X votes=%d/%d  modal=%s frac=%s  values_seen=%s"
                  % (a, v, len(dumps), modals, fracs, allvals))


if __name__ == "__main__":
    main()
