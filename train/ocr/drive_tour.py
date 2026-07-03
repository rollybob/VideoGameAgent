"""
Scripted TOUR capture: replays the hand-discovered navigation path (title -> CONTINUE
-> overworld -> field menu -> party -> summary INFO/SKILLS -> bag -> grass battle),
capturing frames densely throughout. This is the bulk real-data engine -- the path was
discovered by hand-driving (drive.py); here it is deterministic and repeatable.

Anchor: the in-game save sits in VIRIDIAN FOREST (tall grass -> wild battles). Boot ->
title. We do NOT save in-game, so CONTINUE always returns to the same spot.

Frames -> captures/tour/<NN>_<tag>.png. Run after mGBA is up (run_tour.sh sets that up).
Designed to over-capture and be curated downstream; blind cursor-memory menus are
sampled densely so mis-navigation still yields useful frames.
"""
from __future__ import annotations

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.expanduser("~/projects/VGA"))
from vga.core.contract import Action, Button          # noqa: E402
from vga.core.emulator import GbaEmulator             # noqa: E402

OUT = os.path.expanduser("~/projects/VGA/captures/tour")
NAMES = {b.name: b for b in Button}
_n = [0]


def shot(emu, tag):
    cv2.imwrite(os.path.join(OUT, f"{_n[0]:03d}_{tag}.png"), emu.capture())
    _n[0] += 1


def press(emu, keys, hold=0.10, settle=0.35):
    for tok in keys.split():
        emu.send(Action.press(NAMES[tok], duration=hold))
        time.sleep(settle)


def step(emu, keys, tag, shots=2, pre=0.6):
    """Send keys, wait, then capture `shots` frames (catches animated transitions)."""
    press(emu, keys)
    time.sleep(pre)
    for _ in range(shots):
        shot(emu, tag)
        time.sleep(0.25)


def main():
    os.makedirs(OUT, exist_ok=True)
    emu = GbaEmulator()
    time.sleep(1)

    # --- title -> CONTINUE -> overworld (recap screens are themselves useful) ---
    step(emu, "START", "title", shots=1, pre=1.2)
    step(emu, "A", "recap", shots=2, pre=1.5)          # begin continue / first recap
    for i in range(6):                                  # advance through recap screens
        step(emu, "A", f"recap{i}", shots=1, pre=1.2)
    step(emu, "", "overworld", shots=1, pre=1.0)

    # --- field menu -> party -> summary INFO/SKILLS/MOVES ---
    step(emu, "START", "fieldmenu", shots=1, pre=0.8)   # POKeDEX/POKeMON/BAG/.../EXIT
    step(emu, "DOWN A", "party", shots=2, pre=1.0)      # POKeMON -> party list
    step(emu, "A", "party_submenu", shots=1, pre=0.8)   # SUMMARY/ITEM/CANCEL
    step(emu, "A", "summary_info", shots=2, pre=1.0)    # INFO page
    step(emu, "RIGHT", "summary_skills", shots=2, pre=1.0)
    step(emu, "RIGHT", "summary_moves", shots=2, pre=1.0)
    step(emu, "RIGHT", "summary_moves2", shots=2, pre=1.0)
    step(emu, "B B B", "back_overworld", shots=1, pre=0.8)

    # --- BAG ---
    step(emu, "START", "fieldmenu2", shots=1, pre=0.8)
    step(emu, "DOWN DOWN A", "bag", shots=2, pre=1.0)   # cursor mem may vary -> dense
    step(emu, "RIGHT", "bag2", shots=1, pre=0.8)
    step(emu, "RIGHT", "bag3", shots=1, pre=0.8)
    step(emu, "B B", "back_overworld2", shots=1, pre=0.8)

    # --- grass walk -> wild battle (alternate directions; encounter is random) ---
    for i in range(40):
        step(emu, "UP" if i % 2 == 0 else "DOWN", f"walk{i}", shots=1, pre=0.15)
    # battle: spam A to push through intro + FIGHT + first move + dialogue
    for i in range(24):
        step(emu, "A", f"battle{i}", shots=1, pre=0.5)

    print(f"[tour] captured {_n[0]} frames -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
