"""
Probe: where does the current Pokemon save sit, and can we reach the overworld by
spamming B? Drives B (cancel/back) a bunch, capturing a cropped frame after each
press, so we can SEE the screen sequence and pick a navigation anchor for macros.
Saves frames to captures/probe/.
"""
from __future__ import annotations

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.expanduser("~/projects/VGA"))
from vga.core.contract import Action, Button          # noqa: E402
from vga.core.emulator import GbaEmulator             # noqa: E402

OUT = os.path.expanduser("~/projects/VGA/captures/probe")


def main():
    os.makedirs(OUT, exist_ok=True)
    emu = GbaEmulator()
    f0 = emu.capture()
    print(f"[probe] initial frame {f0.shape}", flush=True)
    cv2.imwrite(os.path.join(OUT, "step_00_initial.png"), f0)
    # Spam B to back out of any menu, capturing each step.
    for i in range(1, 25):
        emu.send(Action.press(Button.B, duration=0.08))
        time.sleep(0.25)
        cv2.imwrite(os.path.join(OUT, f"step_{i:02d}_B.png"), emu.capture())
    print(f"[probe] done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
