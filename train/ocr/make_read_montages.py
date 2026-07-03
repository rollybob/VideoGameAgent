"""
Build READABLE labeling montages from a uniq_crops dir.

The default montages get downscaled ~9x by the image viewer when a few full-width
UI-bar crops blow up the canvas width, making small text unreadable. Here each crop
is normalized to a fixed display HEIGHT (preserving aspect, width-capped so wide
bars shrink instead of stretching the canvas), and only N crops per montage, so the
montage stays small enough to view ~1:1. Order matches uniq_manifest.json exactly
(global index = NN*per_montage + row), so labels apply positionally.

  python train/ocr/make_read_montages.py --dir train/ocr/data/staged_sprites
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--per-montage", type=int, default=14)
    ap.add_argument("--disp-h", type=int, default=72, help="display height per crop (px)")
    ap.add_argument("--max-w", type=int, default=560, help="cap upscaled crop width (px)")
    ap.add_argument("--out-prefix", default="read_montage")
    ap.add_argument("--seed-field", default="",
                    help="manifest field (e.g. 'pred') to render as a seed label under each crop")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.dir, "uniq_manifest.json")))
    crops_dir = os.path.join(args.dir, "uniq_crops")
    seeds = {}

    rows = []
    for e in man:
        if args.seed_field:
            seeds[e["id"]] = e.get(args.seed_field) or ""
        im = cv2.imread(os.path.join(crops_dir, e["id"] + ".png"))
        if im is None:
            rows.append((e["id"], np.full((args.disp_h, 40, 3), 200, np.uint8)))
            continue
        h, w = im.shape[:2]
        scale = args.disp_h / float(h)
        up = cv2.resize(im, (max(1, int(w * scale)), args.disp_h),
                        interpolation=cv2.INTER_NEAREST)
        if up.shape[1] > args.max_w:           # shrink over-wide bars to fit canvas
            up = cv2.resize(up, (args.max_w, args.disp_h), interpolation=cv2.INTER_AREA)
        rows.append((e["id"], up))

    label_w = 70
    row_h = args.disp_h + (28 if args.seed_field else 14)
    n_m = 0
    for start in range(0, len(rows), args.per_montage):
        chunk = rows[start:start + args.per_montage]
        max_cw = max(r[1].shape[1] for r in chunk)
        total_w = label_w + max_cw + 12
        canvas = np.full((row_h * len(chunk), total_w, 3), 255, np.uint8)
        for i, (cid, up) in enumerate(chunk):
            y = i * row_h
            idx = start + i
            cv2.putText(canvas, str(idx), (4, y + row_h // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            ch, cw = up.shape[:2]
            canvas[y + 6:y + 6 + ch, label_w:label_w + cw] = up
            if args.seed_field:
                seed = seeds.get(cid, "")
                cv2.putText(canvas, f"[{seed[:40]}]", (label_w, y + args.disp_h + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 60, 60), 1)
            cv2.line(canvas, (0, y), (total_w, y), (200, 200, 200), 1)
        cv2.imwrite(os.path.join(args.dir, f"{args.out_prefix}_{n_m:02d}.png"), canvas)
        n_m += 1
    print(f"{len(rows)} crops -> {n_m} readable montages "
          f"({args.per_montage}/montage, h={args.disp_h})")


if __name__ == "__main__":
    main()
