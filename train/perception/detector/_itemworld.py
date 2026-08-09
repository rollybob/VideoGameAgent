"""_itemworld.py -- decisive shot at the STABLE item object table (not the flaky render buffer).
Enemies persist as WORLD coords (0x03846); items should have a parallel persistent slot. Using
Tim's refined screen positions + each ring's camera, compute each item's WORLD (x,y) and search
IWRAM+EWRAM for a u16 pair (x@o, y@o+2) that matches all three rings' item-world -- that address
is the stable item slot (it holds different world values per room, so matching all 3 is unique).
thor-rl:cu130, read-only."""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
RINGS = "/work/link/sessions/ramhits_solo"
SNAP = (256 + 32) * 1024; IWRAM = 32 * 1024
CAMX, CAMY = 0x02B82, 0x02B86
# Tim's refined screen positions (px): heart(145,90) bosskey(178,90) rupee(103,86)
TARG = [("195309", 145, 90), ("195639", 178, 90), ("201330", 103, 86)]
KNOWN = {0x038F4, 0x038F0, CAMX, CAMY} | {0x03846 + 4 * i for i in range(16)} | {0x03848 + 4 * i for i in range(16)}


def last(tag):
    d = glob.glob(os.path.join(RINGS, f"*{tag}-hit"))[0]
    n = json.load(open(os.path.join(d, "meta.json")))["n"]
    with open(os.path.join(d, "ring.bin"), "rb") as f:
        f.seek((n - 1) * SNAP)
        return np.frombuffer(f.read(SNAP), np.uint8)


def lab(a):
    return f"IW:0x{a:04X}" if a < IWRAM else f"EW:0x{a - IWRAM:05X}"


snaps = [last(t) for t, _, _ in TARG]
N = SNAP - 3
V = np.stack([b[0:N].astype(np.int32) | (b[1:N + 1].astype(np.int32) << 8) for b in snaps])       # u16 @ o
Vy = np.stack([b[2:N + 2].astype(np.int32) | (b[3:N + 3].astype(np.int32) << 8) for b in snaps])  # u16 @ o+2
wx = np.array([sn[CAMX] | (sn[CAMX + 1] << 8) for sn in snaps]) + np.array([t[1] for t in TARG])
wy = np.array([sn[CAMY] | (sn[CAMY + 1] << 8) for sn in snaps]) + np.array([t[2] for t in TARG])
print("item world targets (x,y) per ring:", list(zip(wx.tolist(), wy.tolist())), flush=True)
for tol in (6, 10, 16, 24):
    xm = (np.abs(V - wx[:, None]) <= tol).all(0)
    ym = (np.abs(Vy - wy[:, None]) <= tol).all(0)
    cand = np.where(xm & ym)[0]
    cand = [o for o in cand if o not in KNOWN]
    print(f"\ntol={tol}: {len(cand)} (x@o,y@o+2) world matches", flush=True)
    for o in cand[:15]:
        print(f"   {lab(o)}: x={list(V[:, o])} y={list(Vy[:, o])}", flush=True)
    if 0 < len(cand) <= 12:
        break
print("\nDONE", flush=True)
