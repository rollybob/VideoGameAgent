"""OCR-label the intro frames for ground-truth text. The crawl is RED text over a
busy starfield -> isolate the red channel, crop to the text band, binarize (black
text on white), upscale, then tesseract. Dedups consecutive identical lines (each
crawl line is held for several frames). Ground truth spot-checked by eye.
"""
import glob
import json
import os
import re
import subprocess

import cv2
import numpy as np


def read_text(path):
    img = cv2.imread(path).astype(np.int16)          # BGR
    b, g, r = cv2.split(img)
    mask = ((r > 105) & (r - g > 25) & (r - b > 25)).astype(np.uint8) * 255  # red text
    h = mask.shape[0]
    mask = mask[int(h * 0.50):, :]                    # bottom half = text band
    inv = cv2.bitwise_not(mask)                       # black text on white for tesseract
    inv = cv2.resize(inv, (inv.shape[1] * 4, inv.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("/tmp/_ocr.png", inv)
    out = subprocess.run(["tesseract", "/tmp/_ocr.png", "stdout", "--psm", "6"],
                         capture_output=True, text=True, timeout=30).stdout
    return re.sub(r"\s+", " ", out).strip()


FRAMES = sorted(glob.glob("train/perception/text_frames/t*.png"))
rows = []
for f in FRAMES:
    txt = read_text(f)
    rows.append({"file": os.path.relpath(f, "train/perception"), "text": txt, "nchars": len(txt)})

# keep text-bearing, dedup near-identical consecutive lines (keep the longest of a run)
import difflib
kept = []
for r in rows:
    if r["nchars"] < 15:
        continue
    if kept and difflib.SequenceMatcher(None, kept[-1]["text"], r["text"]).ratio() > 0.7:
        if r["nchars"] > kept[-1]["nchars"]:
            kept[-1] = r
        continue
    kept.append(r)

with open("train/perception/text_labels.jsonl", "w") as fh:
    for r in kept:
        fh.write(json.dumps(r) + "\n")
print("%d frames -> %d distinct text lines" % (len(FRAMES), len(kept)))
for r in kept:
    print("  %s: %r" % (r["file"], r["text"][:75]))
