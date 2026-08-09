"""Task 09 A0: validate the reward oracle against captured tapes.

For each ram_capture dump: replay the oracle logic over the captured IWRAM
stream and print every event. Gate expectation for the 3 flight tapes:
ZERO rupee events (the tapes' HUD counter is pixel-verified constant), so
any event here is a false positive. Determinism is checked separately via
full-stream sha256 across independent replays.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/validate_oracle.py scratch/ram/flight-*
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUPEE_ADDR = 0x6C88


def main():
    args = sys.argv[1:]
    dumps = []
    for a in args:
        p = a if os.path.isabs(a) else os.path.join(HERE, a)
        dumps.extend(sorted(glob.glob(p)) if any(c in a for c in "*?") else [p])
    total_fp = 0
    for d in dumps:
        n = json.load(open(os.path.join(d, "ticks.json")))["ticks"]
        iw = np.memmap(os.path.join(d, "iwram.u8"), dtype=np.uint8, mode="r",
                       shape=(n, 32768))
        rup = (np.asarray(iw[:, RUPEE_ADDR]).astype(np.int32)
               | (np.asarray(iw[:, RUPEE_ADDR + 1]).astype(np.int32) << 8))
        ch = np.nonzero(np.diff(rup))[0] + 1
        events = [(int(t), int(rup[t] - rup[t - 1]), int(rup[t])) for t in ch]
        print("%s: %d ticks, rupees start=%d, %d rupee events %s"
              % (os.path.basename(d), n, int(rup[0]), len(events), events[:10]))
        total_fp += len(events)
    print("TOTAL events across tapes:", total_fp,
          "(expected 0 -> PASS)" if total_fp == 0 else "(EXPECTED 0 -- FALSE POSITIVES!)")


if __name__ == "__main__":
    main()
