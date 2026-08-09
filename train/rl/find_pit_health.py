"""Task 09 A0: find the FS per-player health address from a fs_pit_probe output.

Player 0 wanders (ticks 0-N); players 1-3 stationary. Scans EWRAM and IWRAM for
addresses that decrease (heart loss) while in a plausible health range, with few
total change events. Also explicitly checks the IWRAM stride-0x80 candidates
noted in FINDINGS (0x01068/0x010E8/0x01168/0x011E8).

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \\
      python3 /vga/train/rl/find_pit_health.py /vga/train/rl/scratch/probe/fs_pit
"""
import json
import os
import sys

import numpy as np

EWRAM_SIZE = 256 * 1024
IWRAM_SIZE = 32 * 1024
CHUNK = 16 * 1024

MAX_HEALTH = 24
MIN_HEALTH = 1
MAX_CHANGES = 20   # health can change many times if player falls repeatedly

# Known candidates from FINDINGS (IWRAM, stride 0x80)
IWRAM_KNOWN = [0x01068, 0x010E8, 0x01168, 0x011E8]


def scan_region(arr, region_name, n):
    size = arr.shape[1]
    results = []
    for lo in range(0, size, CHUNK):
        hi = min(lo + CHUNK, size)
        block = np.asarray(arr[:, lo:hi], dtype=np.int32)
        vmax = block.max(axis=0)
        vmin = block.min(axis=0)
        v0 = block[0]
        deltas = np.diff(block, axis=0)
        min_delta = deltas.min(axis=0)
        changes = np.count_nonzero(deltas, axis=0)
        cand = ((v0 >= MIN_HEALTH) & (v0 <= MAX_HEALTH)
                & (vmin < vmax) & (min_delta < 0)
                & (changes <= MAX_CHANGES))
        for off in np.nonzero(cand)[0]:
            addr = lo + int(off)
            col = np.asarray(arr[:, off + lo - lo], dtype=np.int32)
            drops = list(map(int, (np.nonzero(np.diff(col) < 0)[0] + 1)[:5]))
            results.append((region_name, addr, int(v0[off]), int(min_delta[off]),
                            int(changes[off]), drops))
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: find_pit_health.py <probe_dir>")
        sys.exit(1)
    probe_dir = sys.argv[1]

    with open(os.path.join(probe_dir, "ticks.json")) as f:
        meta = json.load(f)
    n = meta["ticks"]
    print("Loaded %d ticks from %s" % (n, probe_dir))

    ew = np.memmap(os.path.join(probe_dir, "ewram.u8"), dtype=np.uint8,
                   mode="r", shape=(n, EWRAM_SIZE))
    iw_path = os.path.join(probe_dir, "iwram.u8")

    results = scan_region(ew, "ewram", n)

    if os.path.exists(iw_path):
        iw = np.memmap(iw_path, dtype=np.uint8, mode="r", shape=(n, IWRAM_SIZE))
        results += scan_region(iw, "iwram", n)

        # Explicitly print FINDINGS candidates regardless of filter
        print("\nFINDINGS IWRAM candidates (stride-0x80):")
        for addr in IWRAM_KNOWN:
            col = np.asarray(iw[:, addr], dtype=np.int32)
            drops = list(map(int, (np.nonzero(np.diff(col) < 0)[0] + 1)[:8]))
            print("  iwram 0x%05X  start=%d  range=%d..%d  drops@ticks=%s"
                  % (addr, int(col[0]), int(col.min()), int(col.max()), drops))

    results.sort(key=lambda r: r[1])
    print("\nAll candidates (region, addr, start_val, min_delta, n_changes, drop_ticks):")
    for reg, addr, v0, md, nc, drops in results:
        print("  %s 0x%05X  start=%2d  min_delta=%3d  changes=%3d  drops@%s"
              % (reg, addr, v0, md, nc, drops))

    if not results:
        print("  (none -- check PNGs; player 0 may not have fallen)")
    else:
        print("\n%d candidates total." % len(results))
        print("Health candidates have: small start_val, decrements of 1-8, few changes.")


if __name__ == "__main__":
    main()
