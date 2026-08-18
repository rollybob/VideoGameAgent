"""diagnose_drops.py -- HONEST accounting of what make_alttp_data.py dropped, and WHY.
Classifies every capture, and montages the dropped frames so we can SEE if a readable box
was wrongly discarded (recoverable data) vs a genuine no-box/closed/prologue frame."""
import os, sys, glob, difflib
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, "/work/train/ocr")
from model import CRNN
CAPS = "/work/link/sessions/text_harvest_0810"
CKPT = "/work/train/ocr/ckpt_ft_mfreal3/best.pt"
SNAP, IW, EW = 294912, 32768, 262144
LO, HI = 0x05480, 0x05820; BREAKS = {0x0C, 0x0E, 0x0F, 0x17}; SEP = 0x18
IY0, IY1, IX0, IX1 = 104, 154, 12, 228; ALIGN = 0.45

def extract_lines(ew):
    lines, cur = [], []
    for off in range(LO, HI):
        b = int(ew[off])
        if b == SEP: break
        if b in BREAKS:
            s = "".join(cur).strip()
            if any(c.isalpha() for c in s): lines.append(s); cur = []
            else: cur = []
        elif 0x20 <= b <= 0x7E: cur.append(chr(b))
    s = "".join(cur).strip()
    if any(c.isalpha() for c in s): lines.append(s)
    return lines
def last_ewram(rb):
    n = os.path.getsize(rb) // SNAP
    with open(rb, "rb") as fh: fh.seek((n - 1) * SNAP + IW); return np.frombuffer(fh.read(EW), np.uint8)
def gold(a):
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return (R > 175) & (G > 130) & (B < 135) & (R > B + 70) & (G > B + 35)
def detect_box(f): return f.shape[0] >= 157 and bool((gold(f[150:157, 4:236]).sum(1) >= 90).any())
def seg_lines(frame):
    inter = frame[IY0:IY1, IX0:IX1]; B, R = inter[..., 2].astype(int), inter[..., 0].astype(int)
    tm = (B > R + 25) & (B > 110); rowsum = tm.sum(1); bands = []; inrow = False; start = 0
    for r, c in enumerate(rowsum):
        if c >= 3 and not inrow: start, inrow = r, True
        elif c < 3 and inrow: bands.append((start, r)); inrow = False
    if inrow: bands.append((start, len(rowsum)))
    bands = [(a, b) for a, b in bands if b - a >= 4]; crops = []
    for a, b in bands:
        cols = np.where(tm[a:b].sum(0) >= 1)[0]
        if len(cols) < 3: continue
        x0, x1 = cols.min(), cols.max() + 1; crop = frame[IY0 + a - 2:IY0 + b + 2, IX0 + x0 - 2:IX0 + x1 + 2]
        if crop.shape[0] >= 6 and crop.shape[1] >= 8: crops.append(crop)
    return crops
ck = torch.load(CKPT, map_location="cpu", weights_only=False); CHARS = ck["meta"]["chars"]
M = CRNN(ck["meta"]["n_classes"], in_ch=3).eval(); M.load_state_dict(ck["model"])
def greedy(am, blank=0):
    out, prev = [], -1
    for a in am:
        a = int(a)
        if a != prev and a != blank: out.append(a)
        prev = a
    return "".join(CHARS[i - 1] for i in out if 1 <= i <= len(CHARS))
@torch.no_grad()
def ocr(crop):
    im = Image.fromarray(crop); nw = max(8, round(im.width * 32 / im.height))
    a = np.asarray(im.resize((nw, 32), Image.BILINEAR), np.float32) / 255.0
    return greedy(M(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)).argmax(-1)[0].numpy())
def ratio(a, b): return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

cat_counts = {}; dropped = []
for folder in sorted(glob.glob(CAPS + "/*-hit")):
    mp, rb = folder + "/mark.png", folder + "/ring.bin"
    if not (os.path.exists(mp) and os.path.exists(rb)): continue
    frame = np.array(Image.open(mp).convert("RGB")); ram = extract_lines(last_ewram(rb))
    if not ram: cat = "no_ram_text"
    elif len(ram) > 12: cat = "prologue_crawl(>12 lines)"
    else:
        box = detect_box(frame); crops = seg_lines(frame); aligned = 0
        for c in crops:
            r = ocr(c)
            if len(r.strip()) >= 2 and max((ratio(r, L) for L in ram), default=0) >= ALIGN: aligned += 1
        if aligned > 0: cat = "USED"
        elif not box: cat = "DROP_nobox"
        else: cat = "DROP_box_noalign"
    cat_counts[cat] = cat_counts.get(cat, 0) + 1
    if cat.startswith("DROP"):
        dropped.append((frame, cat, ram[0] if ram else "", len(ram)))
print("=== classification of all 82 captures ===")
for k in sorted(cat_counts): print(f"  {k:26s} {cat_counts[k]}")
print(f"\n=== {len(dropped)} DROPPED frames (montage of first 20) ===")
tiles = []
for i, (frame, cat, r0, nl) in enumerate(dropped[:20]):
    print(f"  tile{i:2d} [{cat}] ram_lines={nl}  ram[0]={r0[:50]!r}")
    up = np.kron(frame, np.ones((2, 2, 1), np.uint8))
    ImageDraw.Draw(im := Image.fromarray(up)).text((4, 4), f"{i} {cat.split('_',1)[1]}", fill=(255, 60, 60))
    tiles.append(np.asarray(im))
if tiles:
    cols = 4; rows = (len(tiles) + cols - 1) // cols; h, w = tiles[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles): canvas[(i // cols) * h:(i // cols) * h + h, (i % cols) * w:(i % cols) * w + w] = t
    Image.fromarray(canvas).save("/work/train/ocr/data_alttp/_dropped.png")
    print("wrote _dropped.png")
