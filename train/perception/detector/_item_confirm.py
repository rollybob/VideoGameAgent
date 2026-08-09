"""_item_confirm.py -- confirm IW:0x33EF is the dropped item (ring 205603, screen-coord).
Overlay its last-frame screen pos on mark.png, print its x/y timeline (should pop when the pot
broke), and dump the local 0x33E0-0x3410 region as u16s (early vs late) to expose the item-table
structure + any co-dropped items (rupees+bombs). thor-rl:cu130, read-only."""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
RINGS = "/work/link/sessions/ramhits_solo"; DET = "/work/train/perception/detector"
SNAP = (256 + 32) * 1024
d = glob.glob(os.path.join(RINGS, "*205603-hit"))[0]
n = json.load(open(os.path.join(d, "meta.json")))["n"]
A = np.empty((n, SNAP), np.uint8)
with open(os.path.join(d, "ring.bin"), "rb") as f:
    for t in range(n):
        f.seek(t * SNAP); A[t] = np.frombuffer(f.read(SNAP), np.uint8)


def u16(t, a):
    return int(A[t, a]) | (int(A[t, a + 1]) << 8)


print("0x33EF (x) / 0x33F1 (y) timeline, every 20 snaps:", flush=True)
for t in range(0, n, 20):
    print(f"  t={t:3d}  x={u16(t,0x33EF):3d}  y={u16(t,0x33F1):3d}", flush=True)
print(f"  t={n-1:3d}  x={u16(n-1,0x33EF):3d}  y={u16(n-1,0x33F1):3d}  (F11 frame)", flush=True)

print("\nlocal region 0x33E0-0x3410 as u16 (early t=0 vs late t=last):", flush=True)
for a in range(0x33E0, 0x3410, 2):
    e, l = u16(0, a), u16(n - 1, a)
    tag = "  <-- changed" if e != l else ""
    print(f"  0x{a:04X}: early={e:5d}  late={l:5d}{tag}", flush=True)

# overlay a hollow box at (133,128) plus scan the local region for on-screen screen-coord pairs
fr = np.array(Image.open(os.path.join(d, "mark.png")).convert("RGB"), np.uint8).copy()
def box(im, x, y, c, h=5):
    for dd in range(-h, h + 1):
        for (yy, xx) in [(y - h, x + dd), (y + h, x + dd), (y + dd, x - h), (y + dd, x + h)]:
            if 0 <= yy < im.shape[0] and 0 <= xx < im.shape[1]: im[yy, xx] = c
box(fr, u16(n - 1, 0x33EF), u16(n - 1, 0x33F1), [255, 0, 0])          # candidate item (red)
box(fr, u16(n - 1, 0x038F4) - u16(n - 1, 0x02B82), u16(n - 1, 0x038F0) - u16(n - 1, 0x02B86), [0, 255, 0])  # Link (green, world-coord)
Image.fromarray(np.kron(fr, np.ones((4, 4, 1), np.uint8))).save(os.path.join(DET, "_item_confirm.png"))
print("\nWROTE _item_confirm.png (red=0x33EF candidate, green=Link)", flush=True)
