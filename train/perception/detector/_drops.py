"""_drops.py -- lay out TODAY's 21 F11 drop frames (mark.png) at 3x with sprite-table markers
from each ring's LAST snapshot (the frame Tim F11'd right after the drop). green=Link,
red=HP>0 sprite, blue=HP==0 sprite. An item with a blue dot ON it => key/enemy sprite table
(0x03846); an item with NO dot => a different table. Also prints, per ring, the on-screen
sprite slots so we can cross-read image vs RAM. Read-only. thor-rl:cu130."""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageDraw
ROOT = "/work"; DET = os.path.join(ROOT, "train/perception/detector")
RINGS = os.path.join(ROOT, "link/sessions/ramhits_solo")
IWRAM = 32 * 1024; SNAP = (256 + 32) * 1024
CAMX, CAMY = 0x02B82, 0x02B86
Z = 3


def u16(b, a):
    return b[a] | (b[a + 1] << 8)


def last_iw(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    with open(os.path.join(d, "ring.bin"), "rb") as f:
        f.seek((meta["n"] - 1) * SNAP)
        return f.read(IWRAM)


def mark(im, x, y, c, r=3):
    h, w, _ = im.shape
    for d in range(-r, r + 1):
        if 0 <= y < h and 0 <= x + d < w: im[y, x + d] = c
        if 0 <= x < w and 0 <= y + d < h: im[y + d, x] = c


tiles = []
labels = []
for idx, d in enumerate(sorted(glob.glob(os.path.join(RINGS, "20260808-*-hit")))):
    iw = last_iw(d)
    cx, cy = u16(iw, CAMX), u16(iw, CAMY)
    lx, ly = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
    fr = np.array(Image.open(os.path.join(d, "mark.png")).convert("RGB"), np.uint8).copy()
    if 0 <= lx < 240 and 0 <= ly < 160:
        mark(fr, lx, ly, [0, 255, 0])
    desc = []
    for i in range(16):
        ex, ey = u16(iw, 0x03846 + 4 * i), u16(iw, 0x03848 + 4 * i)
        if (ex, ey) == (0, 0):
            continue
        sx, sy = ex - cx, ey - cy
        if 0 <= sx < 240 and 0 <= sy < 160:
            hp = iw[0x03250 + i]
            mark(fr, sx, sy, [255, 0, 0] if hp > 0 else [0, 128, 255])
            desc.append(f"s{i}@{sx},{sy}hp{hp}")
    z = np.kron(fr, np.ones((Z, Z, 1), np.uint8))
    # burn the index number top-left so montage position maps to the printed list
    im = Image.fromarray(z); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 26, 14], fill=(0, 0, 0)); dr.text((2, 2), str(idx), fill=(255, 255, 0))
    tiles.append(np.array(im, np.uint8))
    tag = os.path.basename(d)[9:15]
    labels.append(f"[{idx:2d}] {tag}  link@{lx},{ly}  sprites: " + (", ".join(desc) if desc else "(none)"))

W = 5
rows = []
for r in range(0, len(tiles), W):
    row = tiles[r:r + W]
    while len(row) < W: row.append(np.zeros_like(tiles[0]))
    rows.append(np.hstack(row))
Image.fromarray(np.vstack(rows)).save(os.path.join(DET, "_drops.png"))
print("\n".join(labels))
print(f"\nWROTE _drops.png ({len(tiles)} today rings, {Z}x, grid {W} wide; yellow number = index)")
