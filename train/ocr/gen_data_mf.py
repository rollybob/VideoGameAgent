"""
F18 multi-font RGB synthetic data generator (HOST side -- venv has PIL + cv2 + numpy).

The font-GENERAL recognizer's data engine. Unlike gen_data.py (one game font, grayscale),
this renders text in HUNDREDS of fonts, in COLOR, composited onto real GBA backgrounds,
so the model learns character identity invariant to font/color/background -- the thing a
per-font CRNN (F1-F3) could not do. Self-labeled: the rendered string IS the label.

Key choices (see docs/OCR_F18_PLAN.md):
- MANY fonts (corpus dir, recursively) -> font invariance. Auto-filtered to Latin-capable.
- RGB + COLOR RANDOMIZATION: random fg color, random background (real harvested GBA frame
  patch OR synthetic solid/gradient), optional 1px shadow (GBA text often has one). The
  model keys on shape + LOCAL figure/ground contrast, never absolute color -> generalizes.
- Reuses the F2 augmentation (scale / vertical jitter / resolution loss / blur / contrast)
  which was validated to close the synthetic->real gap, adapted to 3 channels.

Output (--out dir): train.npz, val.npz, meta.json.
  *.npz: images (N,H,MAXW,3) uint8 RGB, widths (N,) int32, labels (sum,) int32 (1-based;
         0=CTC blank), label_lengths (N,) int32.
Fonts that pass the Latin filter are cached to <out>/usable_fonts.json (reuse with default;
--rescan to rebuild). A few fonts are held out for val (unseen-font generalization check).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bitmap_font import BitmapFont, FRLG_FONTS

# General game-text charset. 0 = CTC blank; chars are 1-based.
CHARS = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
         " .,'\"!?-:;()/&é")
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}

WORDS = (
    "the a an of to in is it you your has have will go went came path grass town city "
    "route cave battle wild appeared used move attack potion health level up gym leader "
    "trainer fire water electric run away fled fainted critical hit super effective not "
    "very caught was sent out hello there what name nice meet ready lets begin item bag "
    "heal center nurse joy professor chapter unit enemy turn player save load continue "
    "new game option mode color sound press start select map menu yes no quit save data "
    "magic sword shield armor gold coins score time bonus stage world enter exit").split()


def random_text(rng, max_chars):
    r = rng.random()
    if r < 0.10:                                    # special tokens (digits/punct/accents)
        h, m = int(rng.integers(0, 100)), int(rng.integers(0, 60))
        sp = [f"{h}:{m:02d}", f"No{int(rng.integers(1, 999)):03d}", "POKéMON", "POKéDEX",
              f"Lv{int(rng.integers(1, 100))}", f"{int(rng.integers(0, 9999))}",
              f"HP {int(rng.integers(0, 999))}/{int(rng.integers(0, 999))}", "x99"]
        return sp[int(rng.integers(len(sp)))][:max_chars]
    if r < 0.75:
        parts, used = [], 0
        while True:
            w = WORDS[rng.integers(len(WORDS))]
            if used + len(w) + (1 if parts else 0) > max_chars:
                break
            parts.append(w); used += len(w) + (1 if len(parts) > 1 else 0)
        text = " ".join(parts) if parts else WORDS[rng.integers(len(WORDS))]
    else:                                           # random runs -> full charset coverage
        n = int(rng.integers(2, max_chars + 1))
        text = "".join(CHARS[int(rng.integers(len(CHARS)))] for _ in range(n)).strip()
        if not text:
            text = WORDS[rng.integers(len(WORDS))]
    # Case distribution. GBA UI is heavily ALL-CAPS (menus, item/move names: CONTINUE,
    # NEW GAME, AREA, SIZE) while dialogue is sentence-case. The first decomp run read
    # lowercase well but botched uppercase (CONTINUE->cuntinue, SIZE->SI7e): all-caps
    # strings essentially never appeared in training, so the uppercase classes were
    # undertrained. Give all-caps real mass so those glyph->class mappings get learned.
    c = rng.random()
    if c < 0.30:
        text = text.upper()                         # ALL CAPS (menu / UI text)
    elif c < 0.65:
        text = text[:1].upper() + text[1:]          # sentence case (dialogue)
    return text[:max_chars]


def font_is_latin(path, size=24):
    """Keep only fonts that render basic Latin distinctly (drops icon/symbol/CJK-only/
    blank fonts and ones that render .notdef boxes for everything)."""
    try:
        f = ImageFont.truetype(path, size)
        masks, widths = [], []
        for ch in "AaGg5?.":
            m = f.getmask(ch)
            bb = m.getbbox()
            if bb is None:
                return False                        # missing glyph -> not usable
            masks.append(bb); widths.append(bb[2] - bb[0])
        if len(set(widths)) <= 2:                   # all glyphs same width = .notdef boxes
            return False
        if f.getmask("a").getbbox() == f.getmask("A").getbbox():
            return False                            # no real lowercase
        return True
    except Exception:                               # noqa: BLE001
        return False


def discover_fonts(fonts_dir, cache_path, rescan, max_fonts, logger=print):
    if cache_path and os.path.exists(cache_path) and not rescan:
        fonts = json.load(open(cache_path))
        logger(f"[mf] {len(fonts)} usable fonts (cached {cache_path})")
        return fonts
    allf = sorted(glob.glob(os.path.join(fonts_dir, "**", "*.ttf"), recursive=True))
    logger(f"[mf] scanning {len(allf)} ttf for Latin coverage ...")
    fonts = [p for p in allf if font_is_latin(p)]
    if max_fonts and len(fonts) > max_fonts:
        fonts = fonts[:max_fonts]
    logger(f"[mf] {len(fonts)}/{len(allf)} fonts usable")
    if cache_path:
        json.dump(fonts, open(cache_path, "w"))
    return fonts


def render_mask(text, font, height):
    """Render text -> alpha mask (H,w) uint8 (255=ink), height-normalized, natural width."""
    measure = ImageDraw.Draw(Image.new("L", (8, 8), 0))
    b = measure.textbbox((0, 0), text, font=font)
    tw, th = max(1, b[2] - b[0]), max(1, b[3] - b[1])
    im = Image.new("L", (tw + 4, th + 4), 0)
    ImageDraw.Draw(im).text((2 - b[0], 2 - b[1]), text, fill=255, font=font)
    w = max(1, int(round(im.width * height / im.height)))
    return np.asarray(im.resize((w, height), Image.BILINEAR), np.uint8), w


def rand_color(rng):
    return np.array([int(rng.integers(0, 256)) for _ in range(3)], np.float32)


def make_bg(rng, bgs, H, W):
    """Real harvested GBA patch (preferred) or synthetic solid/gradient."""
    if bgs is not None and rng.random() < 0.5:
        for _ in range(4):
            img = cv2.imread(bgs[int(rng.integers(len(bgs)))])
            if img is None or img.shape[0] < H + 2 or img.shape[1] < 8:
                continue
            ph = int(rng.integers(H, min(img.shape[0], 3 * H) + 1))
            y = int(rng.integers(0, img.shape[1] - 1))  # placeholder, set below
            y0 = int(rng.integers(0, img.shape[0] - ph + 1))
            pw = min(img.shape[1], int(W * ph / H))
            x0 = int(rng.integers(0, img.shape[1] - pw + 1))
            patch = img[y0:y0 + ph, x0:x0 + pw]
            return cv2.resize(patch, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    base = rand_color(rng)
    if rng.random() < 0.5:                          # gradient
        other = rand_color(rng)
        t = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
        return (base[None, None] * (1 - t) + other[None, None] * t) * np.ones((H, W, 1), np.float32)
    return np.ones((H, W, 3), np.float32) * base[None, None]


def composite(mask_full, bg, fg, rng):
    """Paint fg color through alpha mask onto bg, with an optional 1px shadow."""
    out = bg.copy()
    a = (mask_full.astype(np.float32) / 255.0)[:, :, None]
    if rng.random() < 0.5:                          # drop shadow (offset, dark)
        sh = np.zeros_like(mask_full)
        sh[1:, 1:] = mask_full[:-1, :-1]
        sa = (sh.astype(np.float32) / 255.0)[:, :, None]
        dark = np.clip(fg * 0.25, 0, 255)
        out = out * (1 - sa) + dark[None, None] * sa
    return np.clip(out * (1 - a) + fg[None, None] * a, 0, 255).astype(np.uint8)


_INTERPS = (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC)


def augment_rgb(arr, height, rng):
    """F2 augmentation adapted to RGB: vertical scale+jitter, resolution loss, blur,
    brightness/contrast. Width preserved so widths/CTC lengths stay valid."""
    # F18 v2: LIGHTER than v1 -- v1 (res-loss to 0.45x, freq blur, wide contrast) made
    # many samples barely legible, capping accuracy. Keep variety but stay readable.
    H, W = arr.shape[:2]
    ch = int(rng.integers(int(0.70 * height), height + 1))
    res = cv2.resize(arr, (W, ch), interpolation=_INTERPS[rng.integers(len(_INTERPS))])
    pad = np.full((height, W, 3), int(rng.integers(0, 256)), np.uint8)
    y0 = int(rng.integers(0, height - ch + 1))
    pad[y0:y0 + ch] = res
    arr = pad
    f = float(rng.uniform(0.70, 1.0))               # was 0.45 -- less resolution loss
    small = cv2.resize(arr, (max(1, int(W * f)), max(2, int(height * f))), interpolation=cv2.INTER_AREA)
    arr = cv2.resize(small, (W, height), interpolation=_INTERPS[rng.integers(len(_INTERPS))])
    if rng.random() < 0.25:                          # was 0.4
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
    arr = np.clip(arr.astype(np.float32) * float(rng.uniform(0.85, 1.15))
                  + float(rng.uniform(-15, 15)), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        arr = np.clip(arr.astype(np.float32) + rng.normal(0, 4, arr.shape), 0, 255).astype(np.uint8)
    return arr


_FONT_CACHE = {}


def get_font(path, size):
    k = (path, size)
    f = _FONT_CACHE.get(k)
    if f is None:                                   # truetype load is costly; cache it
        f = ImageFont.truetype(path, size)
        _FONT_CACHE[k] = f
    return f


def make_sample(rng, fonts, bgs, height, max_width, downsample):
    """fonts = (ttf_paths, bitmap_fonts, bitmap_weight). Pick a glyph SOURCE -- a real
    decomp BitmapFont (prob bitmap_weight) or a TTF -- then render identically; the rest
    of the pipeline (color/bg/shadow/augment) is source-agnostic (playbook section 5)."""
    ttf_paths, bitmap_fonts, bitmap_weight = fonts
    text = random_text(rng, max_chars=max(2, max_width // 6))
    if bitmap_fonts and rng.random() < bitmap_weight:
        bf = bitmap_fonts[int(rng.integers(len(bitmap_fonts)))]
        # bitmap glyphs cover only the charmap; drop unrenderable chars from text AND
        # label together (render_mask would silently substitute a space -> label drift).
        text = "".join(c for c in text if c in bf.char_to_code) or "the"
        # THICKNESS randomization: FRLG renders text at two weights depending on the
        # screen's shadow contrast -- thin main-stroke (title: CONTINUE) vs BOLD
        # main+shadow (Pokedex/menu: AREA/SIZE/FUMON). Training only on thin main-stroke
        # made the model fail every bold crop (A->H, S->G, O->0). Render ~55% bold so it
        # learns both weights. (This, not horizontal aspect, was the run 2-4 plateau:
        # bold strokes also read as "wider", which width-jitter could not reproduce.)
        ink = (1, 2) if rng.random() < 0.55 else (1,)
        render = lambda t: bf.render_mask(t, height, ink=ink)   # noqa: E731
    else:
        font = get_font(ttf_paths[int(rng.integers(len(ttf_paths)))], int(rng.integers(16, 30)))
        render = lambda t: render_mask(t, font, height)         # noqa: E731
    mask, w = render(text)
    for _ in range(6):                              # trim TEXT (not image) so label matches
        if w <= max_width or len(text) <= 1:
            break
        keep = max(1, int(len(text) * max_width / w) - 1)
        text = text[:keep].rstrip()
        mask, w = render(text)
    if w > max_width:
        mask, w = mask[:, :max_width], max_width
    bg = make_bg(rng, bgs, height, max_width)
    if rng.random() < 0.35:                          # clean text-panel: flatten the bg
        bg = bg * 0.25 + bg.reshape(-1, 3).mean(0)[None, None] * 0.75
    # readability FLOOR (v2): force a strong luma gap so text is always legible.
    fg = rand_color(rng)
    bg_luma = float(bg.mean())
    if abs(fg.mean() - bg_luma) < 90:
        fg = np.clip(fg + (150 if bg_luma < 128 else -150), 0, 255)
    mask_full = np.zeros((height, max_width), np.uint8)
    mask_full[:, :w] = mask
    arr = composite(mask_full, bg, fg, rng)
    arr = augment_rgb(arr, height, rng)
    lab = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
    lab = lab[:max(1, min(len(lab), w // downsample))]
    return arr, w, np.array(lab, np.int32)


def build(n, fonts, bgs, height, max_width, downsample, rng):
    imgs, widths, labels, lengths = [], [], [], []
    for _ in range(n):
        arr, w, lab = make_sample(rng, fonts, bgs, height, max_width, downsample)
        imgs.append(arr); widths.append(w); labels.append(lab); lengths.append(len(lab))
    return (np.stack(imgs), np.array(widths, np.int32),
            np.concatenate(labels).astype(np.int32), np.array(lengths, np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fonts-dir", default="train/ocr/corpus/google-fonts")
    ap.add_argument("--bg-dir", default="captures/harvest")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--max-width", type=int, default=256)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-val", type=int, default=4000)
    ap.add_argument("--val-holdout-fonts", type=int, default=40,
                    help="fonts reserved for val only (unseen-font generalization check)")
    ap.add_argument("--max-fonts", type=int, default=0)
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--bitmap-dir", default="train/ocr/fonts/decomp/pokefirered",
                    help="real decomp bitmap font dir (PNG atlas + text.c + charmap.txt)")
    ap.add_argument("--bitmap-weight", type=float, default=0.0,
                    help="prob a TRAIN sample uses a real bitmap font vs a TTF "
                         "(0=TTF-only legacy, 1.0=real-only; ~0.6 biases toward real)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fonts = discover_fonts(args.fonts_dir, os.path.join(args.out, "usable_fonts.json"),
                           args.rescan, args.max_fonts)
    if len(fonts) < 10:
        print(f"[mf] too few usable fonts ({len(fonts)}); abort"); raise SystemExit(1)

    # Real bitmap fonts (decomp). Loaded into the TRAIN pool only; val stays held-out
    # TTF (the honest REAL-font bar is eval_benchmark.py on bench_pokemon crops).
    bitmap_fonts = []
    if args.bitmap_weight > 0:
        for which in FRLG_FONTS:
            try:
                bitmap_fonts.append(BitmapFont.frlg(args.bitmap_dir, which))
            except Exception as e:                      # noqa: BLE001
                print(f"[mf] bitmap font {which} skipped: {e}", flush=True)
        if not bitmap_fonts:
            print("[mf] bitmap-weight>0 but no bitmap fonts loaded; abort"); raise SystemExit(1)
        print(f"[mf] bitmap fonts: {[b.name for b in bitmap_fonts]} weight={args.bitmap_weight}",
              flush=True)
    bg_paths = sorted(glob.glob(os.path.join(args.bg_dir, "**", "*.png"), recursive=True))
    bgs = bg_paths if bg_paths else None
    print(f"[mf] fonts={len(fonts)} backgrounds={len(bg_paths)} "
          f"chars={len(CHARS)} n_classes={len(CHARS) + 1}", flush=True)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(fonts)
    hold = max(0, min(args.val_holdout_fonts, len(fonts) // 5))
    val_fonts = fonts[:hold] if hold else fonts
    train_fonts = fonts[hold:] if hold else fonts
    print(f"[mf] train_fonts={len(train_fonts)} val_fonts(held out)={len(val_fonts)} "
          f"bitmap_fonts={len(bitmap_fonts)}", flush=True)

    # Each split's font pool is (ttf_paths, bitmap_fonts, bitmap_weight). Real bitmap
    # fonts go to TRAIN only; val is TTF-only (weight 0) so val measures unseen-TTF
    # generalization, while bench_pokemon (real crops) is the true bitmap bar.
    train_pool = (train_fonts, bitmap_fonts, args.bitmap_weight)
    val_pool = (val_fonts, [], 0.0)
    for name, n, fl, seed in (("train", args.n_train, train_pool, args.seed),
                              ("val", args.n_val, val_pool, args.seed + 999)):
        r = np.random.default_rng(seed)
        images, widths, labels, label_lengths = build(
            n, fl, bgs, args.height, args.max_width, args.downsample, r)
        path = os.path.join(args.out, f"{name}.npz")
        np.savez_compressed(path, images=images, widths=widths,
                            labels=labels, label_lengths=label_lengths)
        print(f"[mf] {name}: {n} -> {path} {images.shape}", flush=True)

    meta = {"chars": CHARS, "height": args.height, "max_width": args.max_width,
            "downsample": args.downsample, "channels": 3, "n_fonts": len(fonts),
            "val_holdout_fonts": hold, "n_classes": len(CHARS) + 1,
            "bitmap_fonts": [b.name for b in bitmap_fonts],
            "bitmap_weight": args.bitmap_weight}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"[mf] meta -> {args.out}/meta.json (n_classes={meta['n_classes']})", flush=True)


if __name__ == "__main__":
    main()
