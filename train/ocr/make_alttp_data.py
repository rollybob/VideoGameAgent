"""make_alttp_data.py -- build the ALttP CRNN fine-tune set from Tim's F11 text captures.

Each capture folder (link/sessions/text_harvest_0810/*-hit) has mark.png (the frame) +
ring.bin (RAM ring; the LAST snapshot's EWRAM holds the active dialogue message). Per capture:
  1. detect_box(mark.png)               -- skip frames with no on-screen gold box.
  2. extract_lines(last EWRAM)          -- the RAM oracle's exact line texts (whole message).
  3. seg_lines(box interior)            -- blue-dominant text mask + row-projection -> line crops.
  4. ALIGN: ocr() each crop with the CURRENT CRNN (rough read) and fuzzy-match to the best RAM
     line -> exact ground-truth label. This unlocks MULTI-PAGE boxes: the frame shows one page,
     RAM has all lines, content-matching picks the right ones (no page pointer needed).
  5. dedup identical crops, split train/val by unique label (no leakage), write train/val.npz +
     meta.json in train.py's schema. Also writes _align_check.png for eyeball validation.

Run: docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/work -w /work \
        thor-rl:cu130 python3 train/ocr/make_alttp_data.py
"""
import os, sys, glob, json, hashlib, difflib
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, "/work/train/ocr")
from model import CRNN

CAPS = "/work/link/sessions/text_harvest_0810"
OUT  = "/work/train/ocr/data_alttp"; os.makedirs(OUT, exist_ok=True)
CKPT = "/work/train/ocr/ckpt_ft_mfreal3/best.pt"
SNAP, IW, EW = 294912, 32768, 262144
LO, HI = 0x05480, 0x05820; BREAKS = {0x0C, 0x0E, 0x0F, 0x17}; SEP = 0x18
IY0, IY1, IX0, IX1 = 104, 154, 12, 228        # box interior (from _alttp_lineseg)
ALIGN_FLOOR = 0.45                            # min fuzzy ratio to accept a crop<->RAM-line match

# ---- RAM label (extract_lines ported to a raw EWRAM blob) ----
def extract_lines(ew):
    lines, cur = [], []
    for off in range(LO, HI):
        b = int(ew[off])
        if b == SEP: break
        if b in BREAKS:
            s = "".join(cur).strip()
            if any(c.isalpha() for c in s): lines.append(s)
            cur = []
        elif 0x20 <= b <= 0x7E: cur.append(chr(b))
    s = "".join(cur).strip()
    if any(c.isalpha() for c in s): lines.append(s)
    return lines

def last_ewram(folder):
    """Last snapshot's EWRAM. Prefer a slimmed ewram.bin (see slim_captures.py) so harvest still
    works after ring.bin is deleted; else read the ring's last snapshot."""
    ew_path = os.path.join(folder, "ewram.bin")
    if os.path.exists(ew_path):
        return np.fromfile(ew_path, np.uint8, EW)
    ring = os.path.join(folder, "ring.bin")
    nsnap = os.path.getsize(ring) // SNAP
    with open(ring, "rb") as fh:
        fh.seek((nsnap - 1) * SNAP + IW)
        return np.frombuffer(fh.read(EW), np.uint8)

# ---- GENERAL text-line detection (polarity- & box-color-agnostic; RAM alignment is the filter) ----
# ALttP has several box styles (cream gameplay, PURPLE telepathy, WHITE menu, cream+gold intro) at
# many screen positions -- so box COLOR and POSITION are NOT reliable anchors (measured directly).
# The one universal signal is a horizontal strip of character strokes. Detect text lines anywhere by
# horizontal-edge density; the downstream RAM fuzzy-match is what confirms a crop is real text, so
# this stays permissive. No hardcoded box geometry -- fully position/color general.
def _edges(frame):
    g = frame.mean(2)
    return np.abs(np.diff(g, axis=1)) > 40           # strong horizontal-gradient px = character strokes

def seg_lines(frame):
    E = _edges(frame); ec = E.sum(1)
    thr = max(10, ec.max() * 0.22)                   # adaptive: text rows have many edges vs uniform bg
    on = ec >= thr; bands, ins, s = [], False, 0
    for r, v in enumerate(on):
        if v and not ins: s, ins = r, True
        elif not v and ins: bands.append((s, r)); ins = False
    if ins: bands.append((s, len(on)))
    crops = []
    for a, b in bands:
        if not (5 <= b - a <= 22): continue          # plausible single text-line height
        cols = np.where(E[a:b].sum(0) >= 2)[0]
        if len(cols) < 3: continue
        x0, x1 = int(cols.min()), int(cols.max()) + 2
        if x1 - x0 < 20: continue
        crop = frame[max(0, a - 2):b + 2, max(0, x0 - 2):x1 + 2]
        if crop.shape[0] >= 6 and crop.shape[1] >= 12: crops.append(crop)
    return crops

def detect_box(frame): return len(seg_lines(frame)) > 0   # "has text-like lines" -- align does the real filtering

# ---- CRNN inference (for the alignment rough-read) ----
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
CHARS = ck["meta"]["chars"]; NCLS = ck["meta"]["n_classes"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
M = CRNN(NCLS, in_ch=3).eval().to(DEV); M.load_state_dict(ck["model"])
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
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEV)
    return greedy(M(x).argmax(-1)[0].cpu().numpy())
def ratio(a, b): return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ---- harvest ----
samples = []   # (crop_uint8, label_str)
n_box = n_nobox = n_align = 0
for folder in sorted(glob.glob(CAPS + "/*-hit")):
    mp = folder + "/mark.png"
    has_ram = os.path.exists(folder + "/ewram.bin") or os.path.exists(folder + "/ring.bin")
    if not (os.path.exists(mp) and has_ram): continue
    frame = np.array(Image.open(mp).convert("RGB"))
    ram = extract_lines(last_ewram(folder))
    if not ram or len(ram) > 40: continue     # only a truly huge stale-scrollback blob is unusable
    if not detect_box(frame): n_nobox += 1    # SOFT: find_box+align filter non-text; recovers top boxes
    n_box += 1
    for crop in seg_lines(frame):
        rough = ocr(crop)
        if len(rough.strip()) < 2: n_align += 1; continue
        best = max(ram, key=lambda L: ratio(rough, L))
        if ratio(rough, best) < ALIGN_FLOOR: n_align += 1; continue
        samples.append((crop, best))

# ---- dedup identical crops ----
seen, dd = set(), []
for crop, label in samples:
    h = hashlib.md5(np.ascontiguousarray(crop).tobytes()).hexdigest()
    if (label, h) in seen: continue
    seen.add((label, h)); dd.append((crop, label))
print(f"boxes_used={n_box}  raw_line_crops={len(samples)}  deduped={len(dd)}  "
      f"(dropped no_box={n_nobox}, align={n_align})", flush=True)
if not dd:
    print("NO SAMPLES -- check line-seg/align", flush=True); sys.exit(1)

# ---- resize crops to H=32, pad to a common width ----
def to32(crop):
    im = Image.fromarray(crop); nw = max(8, round(im.width * 32 / im.height))
    return np.asarray(im.resize((nw, 32), Image.BILINEAR), np.uint8)
imgs32 = [to32(c) for c, _ in dd]
Wmax = min(512, max(a.shape[1] for a in imgs32))
def pad(a):
    if a.shape[1] > Wmax:
        a = np.asarray(Image.fromarray(a).resize((Wmax, 32), Image.BILINEAR), np.uint8)
    out = np.full((32, Wmax, 3), 255, np.uint8); out[:, :a.shape[1]] = a; return out

# ---- validation montage: first 18 (crop -> assigned label + rough OCR) ----
vis = dd[:18]; rows = []
for crop, label in vis:
    c = Image.fromarray(to32(crop)); cw = min(c.width, 300)
    canvas = Image.new("RGB", (640, 40), (25, 25, 25)); canvas.paste(c.crop((0, 0, cw, 32)), (2, 4))
    ImageDraw.Draw(canvas).text((cw + 8, 3), label[:48], fill=(0, 255, 120))
    ImageDraw.Draw(canvas).text((cw + 8, 20), "ocr:" + ocr(crop)[:44], fill=(150, 150, 150))
    rows.append(np.asarray(canvas))
Image.fromarray(np.vstack(rows)).save(f"{OUT}/_align_check.png")
print(f"wrote {OUT}/_align_check.png  (crop | GREEN=RAM label | grey=current OCR)", flush=True)

# ---- encode + split train/val by unique label (no leakage) ----
c2i = {c: i + 1 for i, c in enumerate(CHARS)}
def encode(s): return [c2i[ch] for ch in s if ch in c2i]
uniq = sorted(set(l for _, l in dd)); rng = np.random.default_rng(0); rng.shuffle(uniq)
val_labels = set(uniq[:max(3, len(uniq) // 6)])

AUG = 12   # real ALttP crops are few (~30) -- augment TRAIN (brightness/contrast/noise/width
def augment(i32):   # jitter) so the light fine-tune sees enough glyph-rendering variety. VAL stays real.
    f = i32.astype(np.float32)
    f = np.clip((f - 128) * rng.uniform(0.85, 1.15) + 128 * rng.uniform(0.9, 1.1), 0, 255)
    f = np.clip(f + rng.normal(0, rng.uniform(0, 6), f.shape), 0, 255)
    p = Image.fromarray(f.astype(np.uint8))
    nw = max(8, int(p.width * rng.uniform(0.9, 1.12)))
    return np.asarray(p.resize((nw, 32), Image.BILINEAR), np.uint8)
def build(pairs, aug):
    IMs, Ws, LB, LL = [], [], [], []
    for (crop, label), i32 in pairs:
        enc = encode(label)
        if not enc: continue
        variants = [i32] + [augment(i32) for _ in range(AUG)] if aug else [i32]
        for v in variants:
            IMs.append(pad(v)); Ws.append(min(v.shape[1], Wmax)); LB.extend(enc); LL.append(len(enc))
    return np.stack(IMs), np.array(Ws, np.int32), np.array(LB, np.int32), np.array(LL, np.int32)
pairs = list(zip(dd, imgs32))
tr = [p for p in pairs if p[0][1] not in val_labels]
va = [p for p in pairs if p[0][1] in val_labels]
for name, items, aug in [("train", tr, True), ("val", va, False)]:
    IMs, Ws, LB, LL = build(items, aug)
    np.savez_compressed(f"{OUT}/{name}.npz", images=IMs, widths=Ws, labels=LB, label_lengths=LL)
    print(f"{name}: {len(LL)} samples ({'aug x%d' % AUG if aug else 'real'})  images={IMs.shape}", flush=True)
meta = {"chars": CHARS, "n_classes": NCLS, "channels": 3, "height": 32, "max_width": int(Wmax),
        "unique_lines": len(uniq), "source": "alttp F11 captures (text_harvest_0810)",
        "note": "RAM-oracle labels (extract_lines) + CRNN-assisted line alignment"}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
print("WROTE", OUT, flush=True)
