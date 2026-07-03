"""
Run the wired EAST -> line-group -> OCR pipeline on captured real frames. For each
game, rank frames by detected text richness, annotate the most text-rich ones (line
boxes drawn) and print the per-line reads. Recognizer = Tesseract for now (the CRNN
swap is pending the augmentation retrain). This is the first real-frame end-to-end
test of the wired pipeline.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser("~/projects/VGA"))
from vga.core.textdetect import TextDetector            # noqa: E402
from vga.plugins.pokemon.ocr import Ocr                 # noqa: E402

GAMES = ["pokemon", "fireemblem", "ffta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default="/tmp/pipe_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    det = TextDetector()
    ocr = Ocr()
    rows = []
    for g in GAMES:
        results = []
        for p in sorted(glob.glob(f"captures/{g}/*.png")):
            f = cv2.imread(p)
            boxes = det.group_lines(det.detect(f))
            reads = []
            for (x, y, w, h) in boxes:
                x0, y0 = max(0, x - 2), max(0, y - 2)
                x1, y1 = min(f.shape[1], x + w + 2), min(f.shape[0], y + h + 2)
                t = ocr._read_line(f[y0:y1, x0:x1], 2)
                if t:
                    reads.append(t)
            results.append((p, f, boxes, reads))
        results.sort(key=lambda r: -len(r[3]))
        tot_lines = sum(len(r[2]) for r in results)
        tot_reads = sum(len(r[3]) for r in results)
        print(f"\n===== {g}: {tot_lines} line-boxes, {tot_reads} non-empty reads "
              f"over {len(results)} frames | top {args.topk}: =====")
        thumbs = []
        for (p, f, boxes, reads) in results[:args.topk]:
            ann = f.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.imwrite(os.path.join(args.out, f"{g}_{os.path.basename(p)}"), ann)
            print(f"  {os.path.basename(p)}: {len(boxes)} lines, {len(reads)} read")
            for t in reads[:8]:
                print(f"      {t!r}")
            thumbs.append(cv2.resize(ann, (240, 160)))
        if thumbs:
            strip = np.hstack(thumbs)
            lab = np.zeros((20, strip.shape[1], 3), np.uint8)
            cv2.putText(lab, g, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            rows.append(np.vstack([lab, strip]))
    if rows:
        cv2.imwrite("/tmp/pipe_montage.png", np.vstack(rows))
        print("\nmontage -> /tmp/pipe_montage.png")


if __name__ == "__main__":
    main()
