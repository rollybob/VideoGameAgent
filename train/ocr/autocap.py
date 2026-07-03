"""
Background auto-screenshotter for hand-play capture. Tim plays a game on the Thor (:99);
this snaps the mGBA window on an interval, SKIPS near-duplicate/static frames (so sitting on
one screen doesn't flood the folder), and AUTO-BUCKETS by game into <out>/<slug>/NNNN.png
(slug parsed from the mGBA window title). Switch ROMs freely -- it follows the current window
and starts a new bucket automatically. No input is sent (you drive); capture-only.

Run detached so it survives the session, stop it when done:
  thor-job start autocap -- ~/projects/VGA/.venv/bin/python ~/projects/VGA/train/ocr/autocap.py
  thor-job stop autocap
Then I harvest each captures/autocap/<game>/ (build_benchmark + dedup) and label.

Reuses GbaEmulator (xdotool window-find + mss capture). HARD RULE-safe: no blocking xdotool.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser("~/projects/VGA"))
from vga.core.emulator import GbaEmulator, EmulatorNotFound   # noqa: E402


def slug_from_title(title: str) -> str:
    # "mGBA - Fire Emblem (USA, Australia) (60.1 fps) - 0.10.2" -> "fire_emblem_usa_australia"
    t = title
    t = re.sub(r"^mGBA\s*-\s*", "", t)
    t = re.sub(r"\s*\(\d+(\.\d+)?\s*fps\)\s*-\s*[\d.]+\s*$", "", t)   # strip " (60 fps) - 0.10.2"
    t = t.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t or "unknown"


def signature(img, side=16):
    g = cv2.cvtColor(cv2.resize(img, (side, side)), cv2.COLOR_BGR2GRAY)
    return (g > g.mean()).flatten()


def crop_chrome(im):
    """Drop the mGBA window chrome (menu bar + any captured title bar) so only the GBA
    framebuffer is saved -- prevents EAST harvesting "File Emulation Audio/Video Tools" and
    "mGBA - <ROM> (NN fps)" as fake text (the big contaminant, 2026-06-30). Chrome is a
    contiguous LIGHT + DESATURATED band at the very top (GTK/WM bar); game content is
    saturated. Detect that band and crop below it, capped at 12% so a light game screen can
    never be eaten."""
    h = im.shape[0]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].mean(axis=1)      # per-row mean saturation
    cap = int(h * 0.12)                  # never eat more than 12% (safety vs light game tops)
    cut = 0
    for y in range(cap):
        if sat[y] < 55:                  # desaturated = chrome (gray bar) or letterbox; game is saturated
            cut = y + 1
        else:                            # first saturated (game) row -> stop
            break
    return im[cut:] if cut else im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/projects/VGA/captures/autocap"))
    ap.add_argument("--interval", type=float, default=0.8, help="seconds between snaps")
    ap.add_argument("--ham", type=int, default=14,
                    help="min bit-diff vs last SAVED frame to keep (skip static screens)")
    ap.add_argument("--min-fill", type=float, default=0.005,
                    help="skip near-blank/transition frames (ink fraction below this); "
                         "low on purpose -- bright menu/dialogue screens have little dark ink")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    emu = None
    last_sig = {}      # slug -> signature of last saved frame
    counts = {}        # slug -> running index
    saved = skipped = 0
    print(f"[autocap] -> {args.out}  interval={args.interval}s ham>{args.ham}  (Ctrl-C / thor-job stop)")
    while True:
        try:
            if emu is None:
                emu = GbaEmulator(logger=lambda *_: None)
            else:
                emu.refresh_window()
            title = emu._backend.title(emu._handle)
            frame = crop_chrome(emu.capture())
        except (EmulatorNotFound, Exception):
            emu = None
            time.sleep(args.interval)
            continue
        slug = slug_from_title(title)
        sig = signature(frame)
        prev = last_sig.get(slug)
        ink = float((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) < 128).mean())
        novel = prev is None or int(np.count_nonzero(sig != prev)) >= args.ham
        if novel and ink >= args.min_fill:
            d = os.path.join(args.out, slug)
            os.makedirs(d, exist_ok=True)
            idx = counts.get(slug, 0)
            cv2.imwrite(os.path.join(d, f"{idx:05d}.png"), frame)
            counts[slug] = idx + 1
            last_sig[slug] = sig
            saved += 1
            if saved % 25 == 0:
                print(f"[autocap] saved={saved} skipped={skipped}  buckets="
                      + ", ".join(f"{k}:{v}" for k, v in counts.items()))
        else:
            skipped += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
