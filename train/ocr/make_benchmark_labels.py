"""
Curate the LABELED Pokemon OCR benchmark: bind hand-verified ground-truth text to
specific crop ids from build_benchmark.py, write bench_labels.json, and render a
review montage (each selected raw crop, upscaled, with its GT) so a human can
confirm/correct the labels before they are trusted as ground truth.

Selection criteria: distinct lines (no near-duplicate frames), all WITHIN the CRNN
charset (a-zA-Z0-9 space . , ' ! ? -) -- excludes colon/accented-e/icon lines that
the CRNN cannot emit -- and a deliberate mix of Tesseract-correct and Tesseract-fail
cases. Internal layout gaps (label/value columns) are normalized to a single space;
the scorer must whitespace-normalize to match.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

BENCH = os.path.join(os.path.dirname(__file__), "data", "bench_pokemon")

# id -> ground-truth text (hand-verified from the upscaled crops). FLAG=True marks a
# label worth a closer human look (dot count / spacing ambiguity).
LABELS = [
    ("pokemon_0024_l0", "CONTINUE", False),
    ("pokemon_0024_l1", "PLAYER Bitly", False),    # big column gap -> single space
    ("pokemon_0024_l4", "BADGES", False),          # tess misread -> "RANGES"
    ("pokemon_0024_l5", "NEW GAME", False),
    ("pokemon_0030_l0", "your", False),
    # NOTE: pokemon_0042_l0 ("Previously on your quest..2") dropped 2026-06-28 --
    # the full frame shows a trailing "..2" animation artifact and the 2px crop
    # truncated it inconsistently -> ambiguous ground truth, not benchmark-worthy.
    ("pokemon_0096_l4", "WT 15.2 lbs", False),     # full frame has "lbs." but the
                                                   # 2px crop CLIPS the period -> GT
                                                   # matches crop content (no period)
    ("pokemon_0096_l5", "Havimon hides in toxic foliage,", False),
    ("pokemon_0096_l6", "using its surroundings for", False),
    ("pokemon_0096_l7", "protection..", False),    # TWO dots: full frame 0096 renders
                                                   # "protection.." and both are in the
                                                   # crop (Tim caught my 1-dot error)
    ("pokemon_0120_l2", "FUMON", False),
    ("pokemon_0120_l3", "AREA", False),
    ("pokemon_0120_l4", "SIZE", False),
    ("pokemon_0120_l5", "AREA UNKNOWN", False),
]


def main():
    man = {e["id"]: e for e in json.load(open(os.path.join(BENCH, "manifest.json")))}
    out = []
    rows = []
    crop_x = 470
    width = 980
    for cid, gt, flag in LABELS:
        e = man[cid]
        out.append({"id": cid, "src": e["src"], "box": e["box"],
                    "label": gt, "tesseract": e["tesseract"], "flag": flag})
        crop = cv2.imread(os.path.join(BENCH, "crops", cid + ".png"))
        scale = min(5, max(1, (width - crop_x - 10) // max(1, crop.shape[1])))
        up = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                        interpolation=cv2.INTER_NEAREST)
        rows.append((cid, gt, e["tesseract"], flag, up))

    row_h = max(46, max(r[4].shape[0] for r in rows) + 14)
    canvas = np.full((row_h * len(rows), width, 3), 255, np.uint8)
    for i, (cid, gt, tess, flag, up) in enumerate(rows):
        y = i * row_h
        cv2.line(canvas, (0, y), (width, y), (210, 210, 210), 1)
        tag = "GT: " + gt + ("   <-- CHECK" if flag else "")
        cv2.putText(canvas, tag, (6, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 110, 0) if not flag else (0, 90, 200), 1)
        cv2.putText(canvas, "tess: " + tess[:42], (6, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 80, 80), 1)
        ch, cw = up.shape[:2]
        cw = min(cw, width - crop_x)
        oy = y + (row_h - ch) // 2
        canvas[oy:oy + ch, crop_x:crop_x + cw] = up[:, :cw]
    cv2.imwrite(os.path.join(BENCH, "review_montage.png"), canvas)
    with open(os.path.join(BENCH, "bench_labels.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote bench_labels.json: {len(out)} labeled crops")
    print(f"review montage -> {os.path.join(BENCH, 'review_montage.png')}")


if __name__ == "__main__":
    main()
