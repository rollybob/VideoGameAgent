"""
Step 1 of the CRNN-vs-Tesseract plan: build a small LABELED real-frame benchmark
of Pokemon dialogue/text line-crops.

Pipeline reuse: identical extraction path to eval_pipeline.py -- EAST detect ->
group_lines -> crop each line with a 2px pad. For every line we save:
  - crops/<id>.png   : the RAW BGR line crop. THIS is the benchmark sample; it is
                       exactly what the recognizer (Tesseract today, CRNN next)
                       ingests, so the measured CER is honest.
  - view/<id>.png    : an upscaled (nearest, x6) copy for a human/Claude to READ
                       and hand-label. Labels attach to the raw crop, not this.
And we build montage_NN.png grids (upscaled crops + index id) for fast labeling,
plus manifest.json (id, source frame, box, Tesseract guess) as a label-seed.

No torch (cv2 + pytesseract via the host venv). Run:
  ~/projects/VGA/.venv/bin/python train/ocr/build_benchmark.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser("~/projects/VGA"))
from vga.core.textdetect import TextDetector            # noqa: E402
from vga.plugins.pokemon.ocr import Ocr                 # noqa: E402


def split_columns(crop_bgr, gap_frac=0.7, gap_min=6, ink_eps=1):
    """Split one EAST line-crop into separate column/word sub-crops.

    The dex/stat screens put 2-3 columns on a single detected line ("No001  HAVIMON",
    "PLAYER  Bitly") -- the recognizer should see each column alone. We binarize
    polarity-aware (same recipe as ocr._read_line: Otsu -> normalize to dark-on-
    light), project ink onto the x-axis, and split on runs of background columns
    WIDER than a normal word-space. Threshold = max(gap_min, gap_frac*height) so a
    normal space (~0.3-0.5*h here) keeps words together while the much larger
    column gaps separate. Returns a list of (x_offset, sub_bgr) in left-to-right
    order; a crop with no internal column gap returns itself unchanged.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return [(0, crop_bgr)]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(th.mean()) < 127:                 # normalize to dark text on light bg
        th = 255 - th
    ink = (th < 128)                            # True where glyph ink is
    col_ink = ink.sum(axis=0)                   # ink pixels per column
    h = crop_bgr.shape[0]
    gap_thresh = max(gap_min, int(gap_frac * h))

    fg = col_ink > ink_eps                      # columns that contain ink
    if not fg.any():
        return [(0, crop_bgr)]

    # Find contiguous foreground spans separated by background runs >= gap_thresh.
    spans = []
    x = 0
    n = len(fg)
    while x < n:
        if not fg[x]:
            x += 1
            continue
        start = x
        gap = 0
        while x < n:
            if fg[x]:
                gap = 0
                x += 1
            else:
                gap += 1
                x += 1
                if gap >= gap_thresh:
                    break
        end = x - gap                           # last foreground column + 1
        spans.append((start, end))
    if len(spans) <= 1:
        return [(0, crop_bgr)]
    return [(s, crop_bgr[:, s:e]) for (s, e) in spans]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="pokemon")
    ap.add_argument("--out", default="train/ocr/data/bench_pokemon")
    ap.add_argument("--min-w", type=int, default=24,
                    help="drop line-boxes narrower than this (px) -- noise/fragments")
    ap.add_argument("--min-aspect", type=float, default=1.8,
                    help="drop boxes with w/h below this (single glyphs, not lines)")
    ap.add_argument("--scale", type=int, default=6, help="upscale factor for viewing copies")
    ap.add_argument("--per-montage", type=int, default=22)
    ap.add_argument("--split", action="store_true",
                    help="split each line on large column gaps (dex/stat screens)")
    ap.add_argument("--frames-glob", default=None,
                    help="override frame glob (default captures/<game>/*.png)")
    ap.add_argument("--east-width", type=int, default=640,
                    help="EAST input width; raise for large multi-panel spritesheets")
    args = ap.parse_args()

    crops_dir = os.path.join(args.out, "crops")
    view_dir = os.path.join(args.out, "view")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(view_dir, exist_ok=True)

    det = TextDetector(width=args.east_width)
    ocr = Ocr()
    if not det.enabled:
        print("ERROR: EAST detector disabled (model missing?). Abort.")
        sys.exit(1)

    manifest = []
    views = []  # (id, upscaled_crop_bgr, tess_guess)
    glob_pat = args.frames_glob or f"captures/{args.game}/*.png"
    frames = sorted(glob.glob(glob_pat))
    print(f"{len(frames)} frames in {glob_pat}")
    for p in frames:
        f = cv2.imread(p)
        if f is None:
            continue
        boxes = det.group_lines(det.detect(f))
        fname = os.path.splitext(os.path.basename(p))[0]
        for j, (x, y, w, h) in enumerate(boxes):
            if w < args.min_w or h <= 0 or (w / float(h)) < args.min_aspect:
                continue
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(f.shape[1], x + w + 2), min(f.shape[0], y + h + 2)
            line = f[y0:y1, x0:x1]
            if line.size == 0:
                continue
            # Split into column/word sub-crops (dex/stat screens) or keep the line.
            segs = split_columns(line) if args.split else [(0, line)]
            for k, (xoff, crop) in enumerate(segs):
                # Sub-crops may be short words/columns; only drop tiny fragments.
                min_sub = 8 if len(segs) > 1 else args.min_w
                if crop is None or crop.size == 0 or crop.shape[1] < min_sub:
                    continue
                cid = f"{fname}_l{j}" + (f"c{k}" if len(segs) > 1 else "")
                cv2.imwrite(os.path.join(crops_dir, cid + ".png"), crop)
                up = cv2.resize(crop, (crop.shape[1] * args.scale, crop.shape[0] * args.scale),
                                interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(view_dir, cid + ".png"), up)
                guess = ocr._read_line(crop, 2) or ""
                manifest.append({"id": cid, "src": os.path.basename(p),
                                 "box": [int(x0 + xoff), int(y0), int(crop.shape[1]),
                                         int(crop.shape[0])],
                                 "tesseract": guess, "label": None})
                views.append((cid, up, guess))

    # Montage grids: each row = [id text | white-bg upscaled crop], padded to common width.
    if views:
        row_h = max(v[1].shape[0] for v in views) + 8
        max_cw = max(v[1].shape[1] for v in views)
        label_w = 230
        total_w = label_w + max_cw + 16
        n_m = 0
        for start in range(0, len(views), args.per_montage):
            chunk = views[start:start + args.per_montage]
            canvas = np.full((row_h * len(chunk), total_w, 3), 255, np.uint8)
            for i, (cid, up, guess) in enumerate(chunk):
                y = i * row_h
                cv2.putText(canvas, cid, (4, y + row_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(canvas, f"[{guess[:18]}]", (4, y + row_h // 2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 60, 60), 1)
                ch, cw = up.shape[:2]
                canvas[y + 4:y + 4 + ch, label_w:label_w + cw] = up
                cv2.line(canvas, (0, y), (total_w, y), (210, 210, 210), 1)
            mp = os.path.join(args.out, f"montage_{n_m:02d}.png")
            cv2.imwrite(mp, canvas)
            n_m += 1
        print(f"{len(views)} line-crops kept -> {n_m} montages in {args.out}/")

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest.json: {len(manifest)} entries")


if __name__ == "__main__":
    main()
