"""
F1 synthetic OCR data generator (HOST side -- runs in the VGA venv, needs PIL).

Renders random text lines with a given font into fixed-height grayscale line
images and writes them as numpy archives the torch container can load with numpy
alone (the container has no PIL/cv2). This is the self-labeled data source for the
GBA-font OCR recognizer: the font is the label oracle, so labels are free and exact.

The font is a CLI argument. v1 validation runs use a bundled monospace; swapping in
the real FRLG dialogue font is a one-flag change (ledger item F1a) and is what makes
the recognizer transfer to real game frames.

Output (in --out dir): train.npz, val.npz, meta.json.
  *.npz: images (N,H,MAXW) uint8 [text dark on light, white-padded on the right],
         widths (N,) int32 [content width, for CTC input lengths],
         labels (sum_len,) int32 [concatenated, 1-based; 0 is CTC blank],
         label_lengths (N,) int32.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 0 is reserved for the CTC blank; characters are 1-based.
# 2026-06-28 (F2): added ':' and accented-e (U+00E9, a necessary target glyph) so the
# recognizer can read TIME 6:37 / POKeDEX / POKeMON lines that the 69-char F1 charset
# had to exclude. Appended at the END so all pre-existing indices are unchanged (only
# matters if warm-starting; we retrain fresh).
CHARS = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
         " .,'!?-" ":" "é")
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}

# Small in-game-flavoured lexicon so lines look like dialogue, mixed with random
# character runs to guarantee full charset (digits/punctuation) coverage.
WORDS = (
    "the a an of to in is it you your has have will go went came path grass town "
    "city route cave battle wild appeared used move attack potion health level up "
    "gym leader trainer pokemon fire water grass electric run away fled fainted "
    "critical hit super effective not very caught was sent out hello there what "
    "name nice meet ready lets begin item bag heal center nurse joy professor"
).split()


def random_text(rng: np.random.Generator, max_chars: int) -> str:
    # Realistic contexts for the newly-added ':' and 'é' so they are not seen only in
    # uniform random runs (which the model can still over-fit as noise). ~12% of lines.
    if rng.random() < 0.12:
        h, m = int(rng.integers(0, 100)), int(rng.integers(0, 60))
        specials = [
            "POKéDEX", "POKéMON", "POKéMON LIST", "SEED POKéMON",
            "TIME", f"{h}:{m:02d}", f"TIME {h}:{m:02d}",
            "POKéDEX " + str(int(rng.integers(0, 200))),
            "No" + f"{int(rng.integers(1, 400)):03d}",
        ]
        return specials[int(rng.integers(len(specials)))][:max_chars]
    mode = rng.random()
    if mode < 0.7:
        parts = []
        used = 0
        while True:
            w = WORDS[rng.integers(len(WORDS))]
            if used + len(w) + (1 if parts else 0) > max_chars:
                break
            parts.append(w)
            used += len(w) + (1 if len(parts) > 1 else 0)
        text = " ".join(parts) if parts else WORDS[rng.integers(len(WORDS))]
    else:
        n = int(rng.integers(2, max_chars + 1))
        text = "".join(CHARS[int(rng.integers(len(CHARS)))] for _ in range(n)).strip()
        if not text:
            text = WORDS[rng.integers(len(WORDS))]
    if rng.random() < 0.3:
        text = text[:1].upper() + text[1:]
    return text[:max_chars]


def render(text: str, font: ImageFont.FreeTypeFont, height: int):
    """Render text to a height-`height` grayscale strip at its NATURAL width (dark
    text on light). Returns (arr[H,W] uint8, W). No cropping/padding here -- fitting
    to max_width is done in build() at the TEXT level so image and label stay
    consistent."""
    measure = ImageDraw.Draw(Image.new("L", (8, 8), 255))
    bbox = measure.textbbox((0, 0), text, font=font)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    img = Image.new("L", (tw + 4, th + 4), 255)
    ImageDraw.Draw(img).text((2 - bbox[0], 2 - bbox[1]), text, fill=0, font=font)
    new_w = max(1, int(round(img.width * height / img.height)))
    img = img.resize((new_w, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8), new_w


_INTERPS = (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC)


def augment_strip(arr, height, rng):
    """Make a clean synthetic strip look like a REAL detector crop fed through the live
    recognizer preprocessing (textrecog/_read_line: variable crop scale, vertical
    margin, low-res capture then integer upscale, mild blur/contrast). F1 trained on
    clean renders and read REAL crops at CER 0.12 purely from this domain gap; spanning
    it here is the whole point of the F2 retrain. Width is preserved so `widths`/CTC
    input lengths stay valid -- only vertical placement, resolution and tone change."""
    h, w = arr.shape
    # 1. vertical scale + jitter: text rarely fills the crop height; place a shrunk
    #    copy on a white canvas at a random vertical offset (top/bottom margin).
    ch = int(rng.integers(int(0.55 * height), height + 1))
    res = cv2.resize(arr, (w, ch), interpolation=_INTERPS[rng.integers(len(_INTERPS))])
    canvas = np.full((height, w), 255, np.uint8)
    y0 = int(rng.integers(0, height - ch + 1))
    canvas[y0:y0 + ch] = res
    arr = canvas
    # 2. resolution loss: downscale to a fraction then integer-upscale back, mimicking
    #    a tiny GBA line later upscaled by the recognizer (the F1->real mismatch).
    f = float(rng.uniform(0.45, 1.0))
    small = cv2.resize(arr, (max(1, int(w * f)), max(2, int(height * f))),
                       interpolation=cv2.INTER_AREA)
    arr = cv2.resize(small, (w, height), interpolation=_INTERPS[rng.integers(len(_INTERPS))])
    # 3. mild blur (anti-alias / capture softness).
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5]))
        arr = cv2.GaussianBlur(arr, (k, k), 0)
    # 4. brightness / contrast jitter (palette + binarization-residue variation).
    contrast = float(rng.uniform(0.7, 1.3))
    bright = float(rng.uniform(-25, 25))
    arr = np.clip(arr.astype(np.float32) * contrast + bright, 0, 255).astype(np.uint8)
    return arr


def build(n: int, font, height, max_width, downsample, rng, augment=False):
    imgs, widths, labels, lengths = [], [], [], []
    for _ in range(n):
        text = random_text(rng, max_chars=max(2, max_width // 6))
        arr, w = render(text, font, height)
        # Trim TEXT until its rendered width fits max_width, re-rendering, so the
        # label always matches exactly what is visible. (The plateau bug: wide text
        # was image-cropped to max_width but the label kept the cropped-off chars,
        # leaving an unlearnable right-side tail the model could only hallucinate.)
        for _ in range(6):
            if w <= max_width or len(text) <= 1:
                break
            keep = max(1, int(len(text) * max_width / w) - 1)
            text = text[:keep].rstrip()
            arr, w = render(text, font, height)
        if w > max_width:                       # final guard (rare)
            arr = arr[:, :max_width]
            w = max_width
        if w < max_width:
            arr = np.concatenate([arr, np.full((height, max_width - w), 255, np.uint8)], axis=1)
        if augment:                             # real-crop domain match (F2)
            arr = augment_strip(arr, height, rng)
        if rng.random() < 0.8:                  # mild noise so it is not pixel-perfect
            arr = np.clip(arr.astype(np.float32) + rng.normal(0, 6, arr.shape), 0, 255).astype(np.uint8)
        lab = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
        lab = lab[:max(1, min(len(lab), w // downsample))]
        imgs.append(arr)
        widths.append(w)
        labels.append(np.array(lab, np.int32))
        lengths.append(len(lab))
    return (np.stack(imgs), np.array(widths, np.int32),
            np.concatenate(labels).astype(np.int32), np.array(lengths, np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--font", required=True)
    ap.add_argument("--font-size", type=int, default=28)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--max-width", type=int, default=256)
    ap.add_argument("--downsample", type=int, default=4)  # CRNN width reduction
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--augment", action="store_true",
                    help="apply real-crop-like augmentation (F2; off = clean F1 renders)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    font = ImageFont.truetype(args.font, args.font_size)
    for name, n, seed in (("train", args.n_train, args.seed), ("val", args.n_val, args.seed + 999)):
        rng = np.random.default_rng(seed)
        images, widths, labels, label_lengths = build(
            n, font, args.height, args.max_width, args.downsample, rng, augment=args.augment)
        path = os.path.join(args.out, f"{name}.npz")
        np.savez_compressed(path, images=images, widths=widths,
                            labels=labels, label_lengths=label_lengths)
        print(f"[gen] {name}: {n} samples -> {path} (images {images.shape})", flush=True)

    meta = {"chars": CHARS, "height": args.height, "max_width": args.max_width,
            "downsample": args.downsample, "font": args.font, "font_size": args.font_size,
            "augment": bool(args.augment), "n_classes": len(CHARS) + 1}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gen] meta -> {os.path.join(args.out, 'meta.json')} (n_classes={meta['n_classes']})", flush=True)


if __name__ == "__main__":
    main()
