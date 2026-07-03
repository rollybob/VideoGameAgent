"""
Dedup harvested line-crops down to unique text instances.

The capture drives a fixed input cycle, so the same screen (hence the same line)
is saved many times. Labeling 1600 near-identical crops is wasted effort -- we want
ONE representative per distinct line. We dedup on the PIXELS (not Tesseract, which
miscounts): normalize each crop polarity-aware to a small binary signature, then
group by Hamming distance. Keep the sharpest (largest) crop per group.

Writes <out>/uniq_crops/, uniq montages, and uniq_manifest.json. No torch.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

SIG_W, SIG_H = 64, 16


def signature(crop_bgr):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(th.mean()) < 127:                 # normalize polarity -> ink=1
        th = 255 - th
    return (th < 128).astype(np.uint8).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="train/ocr/data/real_pokemon")
    ap.add_argument("--ham", type=int, default=18,
                    help="max bit-difference to treat two crops as the same line")
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--per-montage", type=int, default=30)
    args = ap.parse_args()

    crops_dir = os.path.join(args.dir, "crops")
    paths = sorted(glob.glob(os.path.join(crops_dir, "*.png")))
    sigs = []     # (sig, path, area)
    for p in paths:
        im = cv2.imread(p)
        if im is None or im.size == 0:
            continue
        sigs.append((signature(im), p, im.shape[0] * im.shape[1]))
    print(f"{len(sigs)} crops loaded")

    # Greedy grouping: each crop joins the first group within Hamming <= ham.
    reps = []     # list of (sig, best_path, best_area)
    for sig, p, area in sigs:
        placed = False
        for i, (rsig, rpath, rarea) in enumerate(reps):
            if int(np.count_nonzero(sig != rsig)) <= args.ham:
                if area > rarea:           # keep the sharpest (largest) representative
                    reps[i] = (rsig, p, area)
                placed = True
                break
        if not placed:
            reps.append((sig, p, area))
    print(f"{len(reps)} unique lines after dedup (ham<={args.ham})")

    out_crops = os.path.join(args.dir, "uniq_crops")
    out_view = os.path.join(args.dir, "uniq_view")
    os.makedirs(out_crops, exist_ok=True)
    os.makedirs(out_view, exist_ok=True)

    # seed labels from existing manifest (tesseract guess), keyed by crop id
    man = {e["id"]: e for e in json.load(open(os.path.join(args.dir, "manifest.json")))}
    uniq, views = [], []
    for _, p, _ in reps:
        cid = os.path.splitext(os.path.basename(p))[0]
        im = cv2.imread(p)
        cv2.imwrite(os.path.join(out_crops, cid + ".png"), im)
        up = cv2.resize(im, (im.shape[1] * args.scale, im.shape[0] * args.scale),
                        interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_view, cid + ".png"), up)
        guess = man.get(cid, {}).get("tesseract", "")
        uniq.append({"id": cid, "tesseract": guess, "label": None})
        views.append((cid, up, guess))

    if views:
        row_h = max(v[1].shape[0] for v in views) + 10
        max_cw = max(v[1].shape[1] for v in views)
        label_w = 250
        total_w = label_w + max_cw + 16
        n_m = 0
        for start in range(0, len(views), args.per_montage):
            chunk = views[start:start + args.per_montage]
            canvas = np.full((row_h * len(chunk), total_w, 3), 255, np.uint8)
            for i, (cid, up, guess) in enumerate(chunk):
                y = i * row_h
                cv2.putText(canvas, cid, (4, y + row_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(canvas, f"[{guess[:20]}]", (4, y + row_h // 2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 60, 60), 1)
                ch, cw = up.shape[:2]
                canvas[y + 5:y + 5 + ch, label_w:label_w + cw] = up
                cv2.line(canvas, (0, y), (total_w, y), (210, 210, 210), 1)
            cv2.imwrite(os.path.join(args.dir, f"uniq_montage_{n_m:02d}.png"), canvas)
            n_m += 1
        print(f"{len(views)} unique crops -> {n_m} montages")

    json.dump(uniq, open(os.path.join(args.dir, "uniq_manifest.json"), "w"), indent=2)
    print(f"uniq_manifest.json: {len(uniq)} entries")


if __name__ == "__main__":
    main()
