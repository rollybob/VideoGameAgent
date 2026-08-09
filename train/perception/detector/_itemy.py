"""_itemy.py -- triangulate the item Y byte (X is confirmed at 0x08E4).
Real item centers from the overlays: heart(150,~94) bosskey(160,~89) rupee(107,~85). Find the
byte holding ~94/89/85 across the 3 rings (allowing a small sprite-origin offset), preferring
addresses near 0x08E4 (same sprite entry). Overlay (x@0x08E4, y@best) on each frame to confirm.
thor-rl:cu130, read-only."""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
RINGS = "/work/link/sessions/ramhits_solo"; DET = "/work/train/perception/detector"
SNAP = (256 + 32) * 1024; IWRAM = 32 * 1024
XA = 0x08E4
tags = [("195309", "heart"), ("195639", "bosskey"), ("201330", "rupee")]
ty = np.array([94, 89, 85])


def load(tag):
    d = glob.glob(os.path.join(RINGS, f"*{tag}-hit"))[0]
    n = json.load(open(os.path.join(d, "meta.json")))["n"]
    with open(os.path.join(d, "ring.bin"), "rb") as f:
        f.seek((n - 1) * SNAP); iw = f.read(IWRAM)
    fr = np.array(Image.open(os.path.join(d, "mark.png")).convert("RGB"), np.uint8)
    return np.frombuffer(iw, np.uint8).astype(int), fr


def lab(a):
    return f"IW:0x{a:04X}"


snaps, frames = zip(*[load(t) for t, _ in tags])
S = np.stack(snaps)   # (3, 32K)
best = None
for tol in (6, 10, 14, 18):
    ya = np.where((np.abs(S - ty[:, None]) <= tol).all(0))[0]
    ya = sorted(ya, key=lambda a: abs(a - XA))
    print(f"tol={tol}: {len(ya)} candidates; nearest 0x08E4:", flush=True)
    for a in ya[:8]:
        print(f"   {lab(a)} (d={a-XA:+d}) vals={list(S[:,a])}", flush=True)
    if ya:
        best = ya[0]; break

if best is not None:
    print(f"\nbest Y byte: {lab(best)}  vals={list(S[:,best])} (targets {list(ty)})", flush=True)
    def box(im, x, y, c, h=5):
        for dd in range(-h, h + 1):
            for (yy, xx) in [(y - h, x + dd), (y + h, x + dd), (y + dd, x - h), (y + dd, x + h)]:
                if 0 <= yy < im.shape[0] and 0 <= xx < im.shape[1]: im[yy, xx] = c
    for (tag, name), sn, fr in zip(tags, snaps, frames):
        vis = fr.copy(); box(vis, sn[XA], sn[best], [255, 0, 0])
        Image.fromarray(np.kron(vis, np.ones((4, 4, 1), np.uint8))).save(os.path.join(DET, f"_itemy_{tag}.png"))
    print("wrote _itemy_*.png", flush=True)
