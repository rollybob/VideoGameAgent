"""
Real-frame OCR eval -- CONTAINER side (torch). Loads the prepared dialogue-line
images (/tmp/ocr_eval/lines.npz) and an F1 checkpoint, runs the CRNN, and prints the
decoded text per line alongside the Tesseract reading of the same frame. Head-to-head
on REAL game pixels.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CRNN, WIDTH_DOWNSAMPLE  # noqa: E402

CKPT = "train/ocr/ckpt/best.pt"
EVAL = "/tmp/ocr_eval"


def greedy(am, blank=0):
    out, prev = [], -1
    for a in am:
        a = int(a)
        if a != prev and a != blank:
            out.append(a)
        prev = a
    return out


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    chars = meta["chars"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CRNN(meta["n_classes"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()

    d = np.load(os.path.join(EVAL, "lines.npz"))
    imgs, widths = d["images"], d["widths"]
    info = json.load(open(os.path.join(EVAL, "meta.json")))

    def to_str(ids):
        return "".join(chars[i - 1] for i in ids if 1 <= i <= len(chars))

    print(f"[eval] font={meta['font']} lines={len(imgs)}")
    T = imgs.shape[2] // WIDTH_DOWNSAMPLE
    last_frame = None
    with torch.no_grad():
        for k in range(len(imgs)):
            x = torch.from_numpy(imgs[k:k + 1]).float().div_(255.0).unsqueeze(1).to(dev)
            il = min(int(widths[k]) // WIDTH_DOWNSAMPLE, T)
            am = m(x).argmax(-1)[0].cpu().numpy()[:il]
            fr = info[k]["frame"]
            if fr != last_frame:
                print(f"\n--- frame {fr} ---")
                print(f"  TESSERACT: {info[k]['tess']!r}")
                last_frame = fr
            print(f"  CRNN line: {to_str(greedy(am))!r}")


if __name__ == "__main__":
    main()
