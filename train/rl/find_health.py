"""Task 09 A0: find the solo-ALttP health (hearts) address from capture dumps.

Health signature: a byte that takes NEGATIVE steps (damage), stays in a small
range, and changes sparsely (not an animation/timer). Unlike analyze_ram's
strict "changes in ALL tapes" intersection, this VOTES across tapes -- the
health address only changes in a take where Link actually got hit, so a take
with no damage must not veto it. Ranks by vote count, then cleanliness, and
prints the negative-delta histogram so the heart unit is visible (ALttP stores
health in eighths, so damage deltas cluster on -4 / -8 multiples).

Death falls out for free: it's the same address hitting 0.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/find_health.py scratch/ram/<t1> scratch/ram/<t2> ...
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


def candidates(arr, vmax, max_changes):
    """addr -> (n_changes, vmin, vmax, [negative deltas]) for health-like bytes"""
    n, size = arr.shape
    out = {}
    for lo in range(0, size, CHUNK):
        hi = min(lo + CHUNK, size)
        block = np.asarray(arr[:, lo:hi], dtype=np.int16)
        d = np.diff(block, axis=0)
        n_ch = (d != 0).sum(axis=0)
        has_neg = (d < 0).any(axis=0)
        vmx = block.max(axis=0)
        vmn = block.min(axis=0)
        ok = (n_ch >= 1) & (n_ch <= max_changes) & has_neg & (vmx <= vmax)
        for off in np.nonzero(ok)[0]:
            col = d[:, off]
            out[lo + int(off)] = (int(n_ch[off]), int(vmn[off]), int(vmx[off]),
                                  col[col < 0].tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--region", choices=["ewram", "iwram", "both"], default="both")
    ap.add_argument("--vmax", type=int, default=160, help="max plausible health value")
    ap.add_argument("--max-changes", type=int, default=60,
                    help="reject busier-than-this addresses (timers/animation)")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    dumps = [d if os.path.isabs(d) else os.path.join(HERE, d) for d in args.dumps]
    regions = ["ewram", "iwram"] if args.region == "both" else [args.region]

    for region in regions:
        votes = collections.Counter()
        detail = collections.defaultdict(list)   # addr -> [(name, info), ...]
        for d in dumps:
            arr = load(d, region)
            for a, info in candidates(arr, args.vmax, args.max_changes).items():
                votes[a] += 1
                detail[a].append((os.path.basename(os.path.normpath(d)), info))
        ranked = sorted(votes.items(),
                        key=lambda kv: (-kv[1], sum(i[0] for _, i in detail[kv[0]])))
        print("\n== %s: health-like bytes (neg-delta, vmax<=%d, <=%d changes), "
              "voted across %d tapes ==" % (region, args.vmax, args.max_changes, len(dumps)))
        print("%-8s %-7s %-16s %-12s %s" % ("addr", "votes", "changes/tape",
                                            "value_range", "neg_delta_histogram"))
        for a, v in ranked[:args.top]:
            chs = [i[0] for _, i in detail[a]]
            vmn = min(i[1] for _, i in detail[a])
            vmx = max(i[2] for _, i in detail[a])
            negs = collections.Counter()
            for _, i in detail[a]:
                negs.update(i[3])
            hist = ",".join("%d:%d" % (k, negs[k]) for k in sorted(negs))
            print("0x%05X %2d/%-4d %-16s [%3d,%3d]   %s"
                  % (a, v, len(dumps), str(chs), vmn, vmx, hist))


if __name__ == "__main__":
    main()
