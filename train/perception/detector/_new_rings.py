"""_new_rings.py -- montage the 21:xx rings (Tim's captures right after the single-bush ask)
so I can spot a clean lone-bush drop. mark.png at 3x, labeled with timestamp. read-only."""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageDraw
RINGS = "/work/link/sessions/ramhits_solo"; DET = "/work/train/perception/detector"
ds = sorted(glob.glob(os.path.join(RINGS, "20260808-21*-hit")))
tiles = []
for d in ds:
    fr = np.array(Image.open(os.path.join(d, "mark.png")).convert("RGB"), np.uint8)
    z = np.kron(fr, np.ones((3, 3, 1), np.uint8))
    im = Image.fromarray(z); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 60, 14], fill=(0, 0, 0)); dr.text((2, 2), os.path.basename(d)[9:15], fill=(255, 255, 0))
    tiles.append(np.array(im, np.uint8))
    print(os.path.basename(d))
W = 3; rows = []
for r in range(0, len(tiles), W):
    row = tiles[r:r + W]
    while len(row) < W: row.append(np.zeros_like(tiles[0]))
    rows.append(np.hstack(row))
Image.fromarray(np.vstack(rows)).save(os.path.join(DET, "_new_rings.png"))
print(f"WROTE _new_rings.png ({len(tiles)} rings)")
