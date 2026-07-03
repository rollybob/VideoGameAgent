"""
Capture cropped GBA frames from a running mGBA for OCR training/validation data.
Drives with a varied input pattern to traverse titles/menus/dialogue, and saves
every `stride`-th cropped frame (F16 crop -> clean GBA framebuffer, no chrome).
Frames land as PNGs in --out; the sweep is curated/used downstream.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.expanduser("~/projects/VGA"))
from vga.core.contract import Action, Button          # noqa: E402
from vga.core.emulator import GbaEmulator             # noqa: E402

# Varied pattern: START to leave titles, A to advance/confirm, B to back out, and
# movement -- traverses menus and dialogue across different games.
PATTERN = [Button.START, Button.A, Button.A, Button.DOWN, Button.A, Button.B,
           Button.RIGHT, Button.A, Button.UP, Button.A, Button.LEFT, Button.A]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=20)   # skip the intro-logo frames
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    emu = GbaEmulator()
    print(f"[capture] {args.name}: frame {emu.capture().shape}", flush=True)
    saved = 0
    for i in range(args.n):
        frame = emu.capture()
        if i >= args.warmup and i % args.stride == 0:
            cv2.imwrite(os.path.join(args.out, f"{args.name}_{i:04d}.png"), frame)
            saved += 1
        emu.send(Action.press(PATTERN[i % len(PATTERN)], duration=0.08))
        time.sleep(0.06)
    print(f"[capture] {args.name}: saved {saved} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
