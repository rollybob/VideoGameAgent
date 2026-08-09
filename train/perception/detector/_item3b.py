"""_item3b.py -- lock the item Y byte next to X@0x08E4 and overlay for Tim's confirmation.
0x08E4 = item screen-X (confirmed across 3 drops). If this is an 8-byte OAM-style sprite entry,
Y sits at entry+0 = 0x08E2. Dump 0x08D8-0x08F6 for each ring to see the structure, then box
(x@0x08E4, y@0x08E2) on each mark.png. thor-rl:cu130, read-only."""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
RINGS = "/work/link/sessions/ramhits_solo"; DET = "/work/train/perception/detector"
SNAP = (256 + 32) * 1024; IWRAM = 32 * 1024
XA, YA = 0x08E4, 0x08E2


def load(tag):
    d = glob.glob(os.path.join(RINGS, f"*{tag}-hit"))[0]
    n = json.load(open(os.path.join(d, "meta.json")))["n"]
    with open(os.path.join(d, "ring.bin"), "rb") as f:
        f.seek((n - 1) * SNAP); iw = f.read(IWRAM)
    fr = np.array(Image.open(os.path.join(d, "mark.png")).convert("RGB"), np.uint8)
    return iw, fr


def box(im, x, y, c, h=5):
    for dd in range(-h, h + 1):
        for (yy, xx) in [(y - h, x + dd), (y + h, x + dd), (y + dd, x - h), (y + dd, x + h)]:
            if 0 <= yy < im.shape[0] and 0 <= xx < im.shape[1]: im[yy, xx] = c


for tag, name in (("195309", "heart"), ("195639", "bosskey"), ("201330", "rupee")):
    iw, fr = load(tag)
    dump = [iw[a] for a in range(0x08D8, 0x08F7)]
    print(f"{tag} {name}: x@{hex(XA)}={iw[XA]} y@{hex(YA)}={iw[YA]}  dump 0x08D8-0x08F6: {dump}", flush=True)
    vis = fr.copy(); box(vis, iw[XA], iw[YA], [255, 0, 0])
    Image.fromarray(np.kron(vis, np.ones((4, 4, 1), np.uint8))).save(os.path.join(DET, f"_itembox_{tag}.png"))
print("wrote _itembox_{195309,195639,201330}.png", flush=True)
