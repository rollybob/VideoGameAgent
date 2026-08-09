"""_find_item_table.py -- locate the overworld item table by cross-frame consistency.

The slot-3 sprite table is a red herring for pot/bush drops (markers miss the items). But an
item IS on-screen at the F11 moment of each drop frame, at a DIFFERENT spot each time. So the
item's position address must, across those frames, project (value - camera) ON-SCREEN in x AND y
but at VARYING positions (a real moving object, not a constant like the HUD or camera-centered
Link). Scan BOTH IWRAM and EWRAM for such (x @ o, y @ o+2) pairs, exclude known addresses,
rank by how many frames they're on-screen in + position spread. thor-rl:cu130, read-only.
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
RINGS = "/work/link/sessions/ramhits_solo"
SNAP = (256 + 32) * 1024; IWRAM = 32 * 1024
TARGETS = ["200420", "200518", "200733", "201005", "201127", "201220", "201350"]
CAMX, CAMY = 0x02B82, 0x02B86
KNOWN = {0x038F4, 0x038F0, 0x02B82, 0x02B86} | {0x03846 + 4 * i for i in range(16)} | {0x03848 + 4 * i for i in range(16)}


def last_full(d):
    n = json.load(open(os.path.join(d, "meta.json")))["n"]
    with open(os.path.join(d, "ring.bin"), "rb") as f:
        f.seek((n - 1) * SNAP)
        return np.frombuffer(f.read(SNAP), np.uint8)


def label(o):
    return f"IW:0x{o:04X}" if o < IWRAM else f"EW:0x{o - IWRAM:05X}"


dirs = [glob.glob(os.path.join(RINGS, f"*{t}-hit"))[0] for t in TARGETS]
blobs = [last_full(d) for d in dirs]
F = len(blobs)
V = np.stack([b[:-1].astype(np.int32) | (b[1:].astype(np.int32) << 8) for b in blobs])  # (F, SNAP-1) u16 LE
camx = V[:, CAMX]; camy = V[:, CAMY]
sx = V - camx[:, None]; sy = V - camy[:, None]
onx = (sx >= 0) & (sx < 240); ony = (sy >= 0) & (sy < 160)
pair_on = onx[:, :-2] & ony[:, 2:]      # x at o, y at o+2
cnt = pair_on.sum(0)                     # frames on-screen per offset o

res = []
for o in np.where(cnt >= 6)[0]:
    m = pair_on[:, o]
    xs = sx[m, o]; ys = sy[m, o + 2]
    if xs.std() < 6 and ys.std() < 6:    # drop near-constant (HUD / camera-centered Link)
        continue
    if o in KNOWN or (o >= IWRAM and 0x02000 <= (o - IWRAM) and False):
        continue
    res.append((int(cnt[o]), float(xs.std() + ys.std()), o))

res.sort(key=lambda r: (-r[0], -r[1]))
print(f"cameras per frame: " + ", ".join(f"{int(camx[i])},{int(camy[i])}" for i in range(F)))
print(f"{len(res)} candidate item-position pairs (on>=6/{F} frames, varying pos)\n")
for c, spread, o in res[:25]:
    proj = []
    for i in range(F):
        if pair_on[i, o]:
            proj.append(f"({int(sx[i,o])},{int(sy[i,o+2])})")
        else:
            proj.append("(--)")
    print(f"{label(o)} on {c}/{F} spread={spread:.0f}: " + " ".join(proj))
print("\nDONE")
