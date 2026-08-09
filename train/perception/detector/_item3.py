"""_item3.py -- crack the item table from 3 Tim-confirmed drops (items store SCREEN coords).
  195309 heart      @ (150,85)
  195639 boss key   @ (165,85)
  201330 green rupee@ (110,85)
The item's screen-X byte is the ONE address holding 150 in ring1, 165 in ring2, 110 in ring3
(same object slot, different item positions). Find it by cross-ring match, then the paired Y
byte (~85) nearby. Scan both byte-encoding and the full IWRAM+EWRAM. thor-rl:cu130, read-only.
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
RINGS = "/work/link/sessions/ramhits_solo"
SNAP = (256 + 32) * 1024; IWRAM = 32 * 1024


def last(tag):
    d = glob.glob(os.path.join(RINGS, f"*{tag}-hit"))[0]
    n = json.load(open(os.path.join(d, "meta.json")))["n"]
    with open(os.path.join(d, "ring.bin"), "rb") as f:
        f.seek((n - 1) * SNAP)
        return np.frombuffer(f.read(SNAP), np.uint8)


def lab(a):
    return f"IW:0x{a:04X}" if a < IWRAM else f"EW:0x{a - IWRAM:05X}"


tags = ["195309", "195639", "201330"]
tx = np.array([150, 165, 110]); ty = np.array([85, 85, 85])
S = np.stack([last(t) for t in tags]).astype(int)   # (3, 288K)

for tol in (5, 8, 12, 16):
    xm = (np.abs(S - tx[:, None]) <= tol).all(0)
    xa = np.where(xm)[0]
    print(f"\n=== screen-X byte candidates at tol={tol}px: {len(xa)} ===", flush=True)
    for a in xa[:40]:
        # y byte within +-4 addresses that matches ~85 in all 3 rings
        ynb = [b for b in range(max(0, a - 4), min(S.shape[1], a + 5))
               if (np.abs(S[:, b] - ty) <= tol).all()]
        yl = ",".join(lab(b) for b in ynb) if ynb else "-"
        print(f"  {lab(a)}: x={list(S[:, a])}  y~85 nearby: [{yl}]", flush=True)
    if 0 < len(xa) <= 10:
        break
print("\nDONE", flush=True)
