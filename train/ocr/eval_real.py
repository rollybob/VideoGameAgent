"""
Real-frame OCR eval -- HOST side (VGA venv: cv2 + the emulator). Drives mGBA through
the intro, finds dialogue frames, splits the dialogue box into text lines, prepares
each line (grayscale, tight-crop, H32) and saves them as a numpy archive the torch
container can read. Also records the current Tesseract reading per frame for a
head-to-head, and saves source frame PNGs for visual ground truth.

Outputs in /tmp/ocr_eval: lines.npz (images NxHxMAXW uint8, widths), meta.json
(per-line frame id + Tesseract text), frameNNN.png source frames.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("DISPLAY", ":99")
sys.path.insert(0, os.path.expanduser("~/projects/VGA"))

from vga.core.contract import Action, Button          # noqa: E402
from vga.core.emulator import GbaEmulator             # noqa: E402
from vga.plugins.pokemon.ocr import Ocr               # noqa: E402

OUT = "/tmp/ocr_eval"
H = 32
MAXW = 512


def split_lines(region_gray):
    _, th = cv2.threshold(region_gray, 128, 255, cv2.THRESH_BINARY_INV)  # text -> 255
    rows = (th > 0).sum(axis=1)
    thr = max(3, region_gray.shape[1] // 40)
    bands, in_band, y0 = [], False, 0
    for y, c in enumerate(rows):
        if c >= thr and not in_band:
            y0, in_band = y, True
        elif c < thr and in_band:
            bands.append((y0, y)); in_band = False
    if in_band:
        bands.append((y0, len(rows)))
    return [(a, b) for a, b in bands if b - a >= 5]


def prep_line(region_bgr, y0, y1):
    g = cv2.cvtColor(region_bgr[y0:y1, :], cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 128, 255, cv2.THRESH_BINARY_INV)
    cols = np.where((th > 0).sum(axis=0) > 0)[0]
    if len(cols):
        g = g[:, cols[0]:cols[-1] + 1]
    h, w = g.shape
    new_w = max(1, int(round(w * H / h)))
    g = cv2.resize(g, (new_w, H), interpolation=cv2.INTER_AREA)
    if g.shape[1] >= MAXW:
        g = g[:, :MAXW]; ww = MAXW
    else:
        ww = g.shape[1]
        g = np.concatenate([g, np.full((H, MAXW - ww), 255, np.uint8)], axis=1)
    return g, ww


def main():
    os.makedirs(OUT, exist_ok=True)
    emu = GbaEmulator()       # crop_chrome=True -> clean GBA framebuffer (F16)
    ocr = Ocr()
    line_imgs, line_w, meta = [], [], []
    WARMUP = 40                                        # drive past title/logos first
    i = saved = 0
    while i < 300 and saved < 10:
        frame = emu.capture()
        h = frame.shape[0]
        region = frame[int(h * 0.66):h, :]            # dialogue-box band
        rg = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        tess = ocr.read(region)
        # A real dialogue line: past warmup, sentence-length, has a space, and is
        # mixed-case (rejects the all-caps title/copyright bar).
        is_dialogue = (i >= WARMUP and len(tess) >= 12 and " " in tess
                       and any(c.islower() for c in tess))
        if is_dialogue:
            bands = split_lines(rg)
            for (y0, y1) in bands:
                img, ww = prep_line(region, y0, y1)
                line_imgs.append(img); line_w.append(ww)
                meta.append({"frame": i, "tess": tess})
            cv2.imwrite(f"{OUT}/frame{i:03d}.png", frame)
            print(f"frame {i:3d}: lines={len(bands)} tesseract={tess!r}", flush=True)
            saved += 1
        emu.send(Action.press(Button.START if i < 12 else Button.A))
        i += 1
        time.sleep(0.06)

    if line_imgs:
        np.savez_compressed(f"{OUT}/lines.npz",
                            images=np.stack(line_imgs), widths=np.array(line_w, np.int32))
        json.dump(meta, open(f"{OUT}/meta.json", "w"))
    print(f"[eval] saved {len(line_imgs)} lines from {saved} dialogue frames -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
