"""_pot_spawn.py -- Rosetta stone. Tim's NEWEST ring = a pot that dropped rupees+bombs.
Find the item table by the SPAWN signature in this single known-good ring: address pairs (x@o,
y@o+2) whose world->screen projection is OFF-screen/absent early and becomes ON-screen-stable
late = items that popped when the pot broke. Multiple items (rupees+bombs) => a cluster of
addresses. Overlay the found positions on mark.png so we can confirm they sit on the drops.
thor-rl:cu130, read-only.
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
RINGS = "/work/link/sessions/ramhits_solo"; DET = "/work/train/perception/detector"
SNAP = (256 + 32) * 1024; IWRAM = 32 * 1024
CAMX, CAMY = 0x02B82, 0x02B86
KNOWN = {0x038F4, 0x038F0, 0x02B82, 0x02B86} | {0x03846 + 4 * i for i in range(16)} | {0x03848 + 4 * i for i in range(16)}


def lab(o):
    return f"IW:0x{o:04X}" if o < IWRAM else f"EW:0x{o - IWRAM:05X}"


d = max(glob.glob(os.path.join(RINGS, "20260808-*-hit")), key=os.path.getmtime)
n = json.load(open(os.path.join(d, "meta.json")))["n"]
print("ring:", os.path.basename(d), "snaps:", n, flush=True)
A = np.empty((n, SNAP), np.uint8)
with open(os.path.join(d, "ring.bin"), "rb") as f:
    for t in range(n):
        f.seek(t * SNAP); A[t] = np.frombuffer(f.read(SNAP), np.uint8)
V = A[:, :-1].astype(np.int32) | (A[:, 1:].astype(np.int32) << 8)   # (n, SNAP-1) u16 LE
camx = V[:, CAMX]; camy = V[:, CAMY]
sx = V - camx[:, None]; sy = V - camy[:, None]
onx = (sx >= 0) & (sx < 240); ony = (sy >= 0) & (sy < 160)
E = slice(0, n // 5); L = slice(4 * n // 5, n)

late_onx = onx[L].mean(0); early_onx = onx[E].mean(0)
late_ony = ony[L].mean(0)
xstable = V[L].std(0)
changed = np.abs(V[L].mean(0) - V[E].mean(0))
xcand = np.where((late_onx > 0.85) & (early_onx < 0.15) & (xstable < 12) & (changed > 12))[0]
print(f"{len(xcand)} raw x-candidates before y-pairing", flush=True)

res = []
for o in xcand:
    if o + 2 >= V.shape[1] or o in KNOWN:
        continue
    if late_ony[o + 2] > 0.85 and V[L, o + 2].std() < 12:
        res.append((o, int(sx[-1, o]), int(sy[-1, o + 2])))

res.sort()
merged = []
for o, lx, ly in res:
    if merged and o - merged[-1][0] <= 3:
        continue
    merged.append((o, lx, ly))

for o, lx, ly in merged:
    print(f"{lab(o)} -> screen ({lx},{ly})", flush=True)
print(f"{len(merged)} spawned item-position candidates (deduped)", flush=True)

# overlay candidates (hollow boxes) on the pot frame
fr = np.array(Image.open(os.path.join(d, "mark.png")).convert("RGB"), np.uint8).copy()
lx0, ly0 = int(sx[-1, 0x038F4]), int(sy[-1, 0x038F0])   # Link (green box)
def box(im, x, y, c, h=5):
    for dd in range(-h, h + 1):
        for (yy, xx) in [(y - h, x + dd), (y + h, x + dd), (y + dd, x - h), (y + dd, x + h)]:
            if 0 <= yy < im.shape[0] and 0 <= xx < im.shape[1]: im[yy, xx] = c
if 0 <= lx0 < 240 and 0 <= ly0 < 160: box(fr, lx0, ly0, [0, 255, 0])
for o, lx, ly in merged:
    if 0 <= lx < 240 and 0 <= ly < 160: box(fr, lx, ly, [255, 0, 0])
Image.fromarray(np.kron(fr, np.ones((4, 4, 1), np.uint8))).save(os.path.join(DET, "_pot_spawn.png"))
print("WROTE _pot_spawn.png (4x; green=Link, red=item candidates)", flush=True)
