"""
Interactive driver: send a sequence of GBA buttons to the running mGBA, then capture
a frame. Used to HAND-DRIVE the game across separate calls (mGBA stays up between
calls; GbaEmulator finds the window by name each time, so it is stateless per call).

  drive.py --keys "A" --out /tmp/drive/01.png            # press A, then shot
  drive.py --keys "START DOWN DOWN A" --out shot.png      # a navigation step
  drive.py --keys "" --out shot.png                       # shot only (no input)
  drive.py --keys "DOWN*8" --hold 0.5 --out shot.png      # repeat/hold a key

Keys are space-separated names (A B START SELECT L R UP DOWN LEFT RIGHT), each
optionally NAME*N to repeat N times. --hold = per-press duration; --settle = wait
after each press; --pre = wait before the final capture.
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

NAMES = {b.name: b for b in Button}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="")
    ap.add_argument("--hold", type=float, default=0.10)
    ap.add_argument("--settle", type=float, default=0.30)
    ap.add_argument("--pre", type=float, default=0.40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    emu = GbaEmulator()
    seq = []
    for tok in args.keys.split():
        name, _, rep = tok.partition("*")
        n = int(rep) if rep else 1
        if name.upper() not in NAMES:
            print(f"[drive] unknown key {name!r}; valid: {list(NAMES)}"); sys.exit(2)
        seq += [NAMES[name.upper()]] * n
    for b in seq:
        emu.send(Action.press(b, duration=args.hold))
        time.sleep(args.settle)
    time.sleep(args.pre)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frame = emu.capture()
    cv2.imwrite(args.out, frame)
    print(f"[drive] sent [{' '.join(b.name for b in seq) or '(none)'}] -> {args.out} {frame.shape}")


if __name__ == "__main__":
    main()
