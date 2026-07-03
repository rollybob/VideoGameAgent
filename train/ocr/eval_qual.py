"""
Qualitative cross-font probe: run the detector-gated CRNN(F3) AND Tesseract on real
frames of a game we have NO labels for (Fire Emblem, FFTA). No CER (no GT) -- this is
to eyeball whether a CRNN trained on the FRLG/Pokemon font transfers to a DIFFERENT
game's pixel font. Prints, per text-rich frame, the per-line reads from both engines
side by side. Expectation per docs/STATE_OF_VGA.md F18: a per-font CRNN does NOT
transfer; Tesseract (font-general) should hold up better on cleaner fonts (FFTA).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.expanduser("~/projects/VGA"))
from vga.core.textdetect import TextDetector          # noqa: E402
from vga.core.textrecog import CrnnRecognizer          # noqa: E402
from vga.plugins.pokemon.ocr import Ocr                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--onnx", default=os.path.expanduser(
        "~/projects/VGA/train/ocr/ckpt_f3/crnn.onnx"))
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()

    det = TextDetector()
    crnn = CrnnRecognizer(onnx_path=args.onnx)
    ocr = Ocr()
    frames = sorted(glob.glob(os.path.expanduser(f"~/projects/VGA/captures/{args.game}/*.png")))
    results = []
    for p in frames:
        f = cv2.imread(p)
        boxes = det.group_lines(det.detect(f))
        lines = []
        for (x, y, w, h) in boxes:
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(f.shape[1], x + w + 2), min(f.shape[0], y + h + 2)
            crop = f[y0:y1, x0:x1]
            c = crnn.read(crop).strip()
            t = ocr._read_line(crop, 2).strip()
            if c or t:
                lines.append((c, t))
        results.append((os.path.basename(p), lines))
    results.sort(key=lambda r: -len(r[1]))
    print(f"\n===== {args.game}: top {args.topk} text-rich frames "
          f"(CRNN-F3 | Tesseract) =====")
    for name, lines in results[:args.topk]:
        print(f"\n-- {name} ({len(lines)} lines) --")
        for c, t in lines[:10]:
            print(f"   CRNN: {c!r:38}  TESS: {t!r}")


if __name__ == "__main__":
    main()
