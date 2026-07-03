"""
Live probe: drive the real mGBA window through the VGA emulator + Pokemon
perception, save frames, and dump what perception/OCR produce on REAL game pixels.

This is an evaluation tool, not part of the agent. It answers: how does the current
(Tesseract-based) OCR and the HSV/contour perception actually behave on live frames?
Saves PNGs to /tmp/vga_probe for visual inspection and prints scene + OCR per frame.
"""
from __future__ import annotations

import os
import time

import cv2

from vga.core.contract import Action, Button
from vga.core.emulator import GbaEmulator
from vga.plugins.pokemon.perception import PokemonPerception

OUT = "/tmp/vga_probe"
N = 150


def main():
    os.makedirs(OUT, exist_ok=True)
    emu = GbaEmulator(title_keyword="mGBA")
    per = PokemonPerception(enable_ocr=True)
    saved = 0
    scene_counts = {}
    ocr_hits = 0
    for i in range(N):
        frame = emu.capture()
        gs = per.perceive(frame, i, time.time())
        scene = gs.regions.get("scene_raw", "?")
        dlg = gs.text.get("dialogue", "")
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
        if dlg:
            ocr_hits += 1
        if i % 10 == 0 or dlg:
            path = f"{OUT}/f{i:03d}_{scene}.png"
            cv2.imwrite(path, frame)
            print(f"frame {i:3d} scene={scene:9s} ocr={dlg!r}", flush=True)
            saved += 1
        # Leave the title screen with START, then pass through menus/dialogue/
        # overworld: mostly A (advance/confirm) with occasional movement.
        if i < 12:
            emu.send(Action.press(Button.START))
        elif i % 4 == 0:
            emu.send(Action.press(Button.DOWN, duration=0.10))
        else:
            emu.send(Action.press(Button.A))
        time.sleep(0.06)
    print(f"\n[probe] frames={N} saved={saved} ocr_nonempty={ocr_hits} scenes={scene_counts}",
          flush=True)


if __name__ == "__main__":
    main()
