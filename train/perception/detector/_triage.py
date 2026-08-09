"""_triage.py -- one contact sheet of ALL today's F11 captures, numbered, for Tim to scan on his
phone and flag which tiles show a loose dropped item (or a chest). mark.png at 3x, index burned
top-left. Prints the index->timestamp map. thor-rl:cu130 (PIL). read-only."""
import os, glob
import numpy as np
from PIL import Image, ImageDraw
RINGS = "/work/link/sessions/ramhits_solo"; DET = "/work/train/perception/detector"
ds = sorted(glob.glob(os.path.join(RINGS, "20260808-*-hit")))
Z = 3; tiles = []; idxmap = []
for i, d in enumerate(ds):
    base = os.path.basename(d)[9:15]
    im = Image.open(os.path.join(d, "mark.png")).convert("RGB")
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 22, 9], fill=(0, 0, 0)); dr.text((1, 1), str(i), fill=(255, 255, 0))
    tiles.append(np.array(im.resize((240 * Z, 160 * Z), Image.NEAREST), np.uint8))
    idxmap.append(f"{i}:{base}")
W = 5; rows = []
for r in range(0, len(tiles), W):
    row = tiles[r:r + W]
    while len(row) < W:
        row.append(np.zeros_like(tiles[0]))
    rows.append(np.hstack(row))
Image.fromarray(np.vstack(rows)).save(os.path.join(DET, "_triage.png"))
print("MAP", " ".join(idxmap), flush=True)
print("N", len(tiles), flush=True)
