"""_item_locate.py -- crack the item table from ring 205603 (a dropped item resting at
lower-center, below Link, at the F11 frame). Search the LAST snapshot for any u16 pair (x@o,
y@o+2) that projects onto the drop -- trying BOTH world-coord (value - camera) and raw
screen-coord encodings -- then report each candidate's on-screen timeline across the ring
(the real item pops when the pot breaks, so fraction is transient, not 0 or 1). No assumption
about which table it lives in; we search for the coordinate value directly. thor-rl:cu130.
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
RINGS = "/work/link/sessions/ramhits_solo"
SNAP = (256 + 32) * 1024; IWRAM = 32 * 1024
CAMX, CAMY = 0x02B82, 0x02B86
KNOWN = {0x038F4, 0x038F0, 0x02B82, 0x02B86} | {0x03846 + 4 * i for i in range(16)} | {0x03848 + 4 * i for i in range(16)}
# the yellow drop sits below Link, lower-center of the play area
RX0, RX1, RY0, RY1 = 95, 145, 95, 150


def lab(o):
    return f"IW:0x{o:04X}" if o < IWRAM else f"EW:0x{o - IWRAM:05X}"


d = [x for x in glob.glob(os.path.join(RINGS, "*205603-hit"))][0]
n = json.load(open(os.path.join(d, "meta.json")))["n"]
A = np.empty((n, SNAP), np.uint8)
with open(os.path.join(d, "ring.bin"), "rb") as f:
    for t in range(n):
        f.seek(t * SNAP); A[t] = np.frombuffer(f.read(SNAP), np.uint8)
V = A[:, :-1].astype(np.int32) | (A[:, 1:].astype(np.int32) << 8)
camx, camy = V[:, CAMX], V[:, CAMY]
last = n - 1
print(f"ring 205603, camera(last)=({int(camx[last])},{int(camy[last])})", flush=True)

X, Y = V[last, :-2], V[last, 2:]           # x@o, y@o+2 at the F11 frame
wsx, wsy = X - camx[last], Y - camy[last]   # world->screen
in_world = (wsx >= RX0) & (wsx <= RX1) & (wsy >= RY0) & (wsy <= RY1)
in_scr = (X >= RX0) & (X <= RX1) & (Y >= RY0) & (Y <= RY1)   # raw screen-coord encoding

for enc, mask, projx, projy in (("world", in_world, wsx, wsy), ("screen", in_scr, X, Y)):
    hits = [o for o in np.where(mask)[0] if o not in KNOWN]
    print(f"\n--- {enc}-coord candidates in the drop region: {len(hits)} ---", flush=True)
    rows = []
    for o in hits:
        # timeline: fraction of frames this pair projects on-screen (transient => item)
        if enc == "world":
            on = ((V[:, o] - camx >= 0) & (V[:, o] - camx < 240) & (V[:, o + 2] - camy >= 0) & (V[:, o + 2] - camy < 160))
        else:
            on = ((V[:, o] >= 0) & (V[:, o] < 240) & (V[:, o + 2] >= 0) & (V[:, o + 2] < 160))
        frac = on.mean()
        rows.append((frac, o, int(projx[o]), int(projy[o])))
    for frac, o, px, py in sorted(rows)[:20]:
        flag = "  <- TRANSIENT" if 0.03 < frac < 0.9 else ("  (always-on)" if frac >= 0.9 else "")
        print(f"  {lab(o)} @screen({px},{py}) onscreen_frac={frac:.2f}{flag}", flush=True)
print("\nDONE", flush=True)
