"""_item_zoom.py -- decisive test: do the slot-3 HP==0 entries in TODAY's captures land ON the
dropped pot/bush item? Zoom each candidate's mark.png around the slot-3 screen pos (recomputed
from the ring's last snapshot), mark center red, montage. If the red mark sits on a heart/rupee/
bomb/etc -> items use the slot table (story A). If it sits on empty floor -> different table (B).
Read-only. thor-rl:cu130."""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
ROOT = "/work"; DET = os.path.join(ROOT, "train/perception/detector")
RINGS = os.path.join(ROOT, "link/sessions/ramhits_solo")
IWRAM = 32 * 1024; SNAP = (256 + 32) * 1024
CAM_X, CAM_Y = 0x02B82, 0x02B86


def u16(b, a):
    return b[a] | (b[a + 1] << 8)


def last_iw(dumpdir):
    meta = json.load(open(os.path.join(dumpdir, "meta.json")))
    with open(os.path.join(dumpdir, "ring.bin"), "rb") as f:
        f.seek((meta["n"] - 1) * SNAP)
        return f.read(IWRAM)


tiles = []
for d in sorted(glob.glob(os.path.join(RINGS, "20260808-*-hit"))):
    iw = last_iw(d)
    cx, cy = u16(iw, CAM_X), u16(iw, CAM_Y)
    hits = []
    for i in range(16):
        ex, ey = u16(iw, 0x03846 + 4 * i), u16(iw, 0x03848 + 4 * i)
        if (ex, ey) == (0, 0):
            continue
        sx, sy = ex - cx, ey - cy
        if 0 <= sx < 240 and 0 <= sy < 160 and iw[0x03250 + i] == 0:  # on-screen HP==0
            hits.append((i, sx, sy))
    if not hits:
        continue
    fr = np.array(Image.open(os.path.join(d, "mark.png")).convert("RGB"), np.uint8)
    for (i, sx, sy) in hits:
        R, ZM, H = 20, 8, 6
        y0, y1 = max(0, sy - R), min(160, sy + R); x0, x1 = max(0, sx - R), min(240, sx + R)
        crop = fr[y0:y1, x0:x1].copy()
        cyc, cxc = sy - y0, sx - x0            # HOLLOW box (outline only) -> item stays visible inside
        for dd in range(-H, H + 1):
            for (yy, xx) in [(cyc - H, cxc + dd), (cyc + H, cxc + dd), (cyc + dd, cxc - H), (cyc + dd, cxc + H)]:
                if 0 <= yy < crop.shape[0] and 0 <= xx < crop.shape[1]: crop[yy, xx] = [255, 0, 0]
        z = np.kron(crop, np.ones((ZM, ZM, 1), np.uint8))
        pad = np.zeros((ZM * 2 * R, ZM * 2 * R, 3), np.uint8); pad[:z.shape[0], :z.shape[1]] = z
        tiles.append(pad)
        print(f"{os.path.basename(d)[9:15]}  slot{i} @ screen ({sx},{sy})", flush=True)

if tiles:
    W = 4
    rows = []
    for r in range(0, len(tiles), W):
        row = tiles[r:r + W]
        while len(row) < W: row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    Image.fromarray(np.vstack(rows)).save(os.path.join(DET, "_item_zoom.png"))
    print(f"\nWROTE _item_zoom.png ({len(tiles)} slot-3 HP==0 candidates, 6x zoom, red = slot center)")
else:
    print("no slot-3 HP==0 candidates in today's rings")
