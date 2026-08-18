"""slim_captures.py -- shrink text_harvest_* capture folders.

Each `<ts>-hit/ring.bin` is an ~85MB RAM ring, but the data engine only ever uses the LAST
snapshot's EWRAM (256KB) plus mark.png. This extracts `ewram.bin` (that 256KB) per folder and,
with --delete, removes ring.bin -- cutting ~6.8G -> ~20M. make_alttp_data.last_ewram (and the
validation readers) prefer ewram.bin when present, so re-harvest still works after slimming.

Two-phase for safety (this deletes data -- verify first):
  (default)   extract + verify ewram.bin for every folder; touch NO ring.bin.
  --delete    additionally delete ring.bin, but only for folders whose ewram.bin verified
              (round-trips to identical bytes + identical extract_lines) THIS run.

Run (host python; numpy only):
  python3 train/ocr/slim_captures.py [--root <dir>] [--delete]
"""
import os
import sys
import glob
import argparse

import numpy as np

SNAP, IW, EW = 294912, 32768, 262144          # ring snapshot stride; IWRAM offset; EWRAM size
LO, HI = 0x05480, 0x05820                      # dialogue buffer window
BREAKS = {0x0C, 0x0E, 0x0F, 0x17}
SEP = 0x18


def extract_lines(ew):
    lines, cur = [], []
    for off in range(LO, HI):
        b = int(ew[off])
        if b == SEP:
            break
        if b in BREAKS:
            s = "".join(cur).strip()
            if any(c.isalpha() for c in s):
                lines.append(s)
            cur = []
        elif 0x20 <= b <= 0x7E:
            cur.append(chr(b))
    s = "".join(cur).strip()
    if any(c.isalpha() for c in s):
        lines.append(s)
    return lines


def last_snap_ewram(ring):
    n = os.path.getsize(ring) // SNAP
    with open(ring, "rb") as fh:
        fh.seek((n - 1) * SNAP + IW)
        return np.frombuffer(fh.read(EW), np.uint8)


def main():
    default_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                     "link", "sessions", "text_harvest_0810"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root)
    ap.add_argument("--delete", action="store_true", help="delete verified ring.bin files")
    a = ap.parse_args()

    folders = sorted(glob.glob(os.path.join(a.root, "*-hit")))
    if not folders:
        print("no *-hit folders at", a.root)
        sys.exit(1)

    verified, freed, skipped, failed = [], 0, 0, 0
    for f in folders:
        ring = os.path.join(f, "ring.bin")
        ew_path = os.path.join(f, "ewram.bin")
        if not os.path.exists(ring):
            skipped += 1
            continue
        ew = last_snap_ewram(ring)
        lines_ring = extract_lines(ew)
        ew.tofile(ew_path)
        ew2 = np.fromfile(ew_path, np.uint8)      # verify the written file round-trips exactly
        if ew2.shape[0] == EW and np.array_equal(ew2, ew) and extract_lines(ew2) == lines_ring:
            verified.append((f, os.path.getsize(ring)))
            freed += os.path.getsize(ring)
        else:
            failed += 1
            print("VERIFY FAILED (keeping ring):", os.path.basename(f))

    print(f"folders={len(folders)} verified_ewram={len(verified)} "
          f"skipped(no ring)={skipped} failed={failed} "
          f"ring_reclaimable={freed / 1e9:.2f}G")

    if a.delete:
        d = 0
        for f, _ in verified:
            os.remove(os.path.join(f, "ring.bin"))
            d += 1
        print(f"deleted {d} ring.bin (~{freed / 1e9:.2f}G freed)")
    else:
        print("dry pass complete (no deletion). Re-run with --delete to remove verified ring.bin.")


if __name__ == "__main__":
    main()
