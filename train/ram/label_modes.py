#!/usr/bin/env python3
"""Oracle-ASSISTED mode-label generator for FFTA (F22 perception layer).

Why not a pure RAM enum: FFTA's "mode" is multi-variable (verified 2026-07-02) -
0x02003cb7 flags world-map overlays (map=0, wm-menu=255, help=3, arealist=204) but the
PUB service menu reads 0 like the map, because it is a different UI subsystem. There is
no single clean mode byte. So instead of hunting a state variable per subsystem, we label
frames by the KNOWN CONTEXT we drove them from (scripted tours) and use 0x3cb7 as a
cross-check where it is meaningful. The labels train a VISION classifier, which learns the
functional/visual distinction that generalizes across subsystems:

    field  : free world map / field, no option list and no text box to advance
    menu   : a navigable option LIST is on screen (wm location menu, main menu, area
             list, and crucially the PUB Rumors/Missions menu) - EVEN IF a text box is
             also showing. This is the lesson the VLM missed (it called the pub 'dialog').
    dialog : a text box to advance with A, with NO option list (help boxes, story text)

Output: <out>/frames/NNNNNN.png (native 240x160) + <out>/labels.csv (frame,label,ram_3cb7).

Run (headless, CPU, no GPU):
    .venv/bin/python train/ram/label_modes.py --out train/perception/mode_data --episodes 3
"""

from __future__ import annotations

import argparse
import csv
import os
import random

from emu import Emu

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
STATE = os.path.join(os.path.dirname(__file__), "ffta_worldmap.state")
MODE_ADDR = 0x02003cb7   # world-map overlay flag; cross-check only (0=field/pub, 255=wm-menu, 3=help, 204=arealist)

DIRS = ["UP", "DOWN", "LEFT", "RIGHT"]


class Labeler:
    def __init__(self, out: str):
        self.out = out
        self.frames_dir = os.path.join(out, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.rows: list[tuple] = []
        self.n = 0
        self.raw = open(STATE, "rb").read()
        self.e = Emu(ROM)
        self.e.boot(load_save=True)

    def load(self):
        self.e.core.load_raw_state(self.raw)
        self.e.run(2)

    def hold(self, k, f=6, then=12):
        key = self.e._keymap[k]
        self.e.core.set_keys(key)
        self.e.run(f)
        self.e.core.clear_keys(key)
        self.e.run(then)

    def _is_black(self) -> bool:
        # Skip transition frames (area load): near-black screens carry no mode signal.
        px = self.e.image.to_pil().convert("L").resize((16, 16))
        return sum(px.getdata()) / 256.0 < 8.0

    def cap(self, label: str):
        if self._is_black():
            return
        rel = os.path.join("frames", f"{self.n:06d}.png")
        self.e.save_png(os.path.join(self.out, rel))
        self.rows.append((rel, label, self.e.u8(MODE_ADDR)))
        self.n += 1

    # --- tours (each starts from the clean world-map state) ---
    def tour_field(self, rng: random.Random, steps: int):
        """Wander the world map: varied cursor nodes = varied field frames."""
        self.load()
        for _ in range(steps):
            self.hold(rng.choice(DIRS))
            self.cap("field")

    def tour_wm_menus(self):
        """World-map menus: location menu (A) and main menu (START), cycling the cursor."""
        for opener in (["A"], ["START"]):
            self.load()
            for k in opener:
                self.hold(k)
            self.cap("menu")
            for _ in range(3):
                self.hold("DOWN"); self.cap("menu")
            for _ in range(3):
                self.hold("UP"); self.cap("menu")
        # Area List submenu (START -> Area List -> A) is a scrollable list too.
        self.load()
        self.hold("START"); self.hold("DOWN"); self.hold("A")
        self.cap("menu")
        for _ in range(4):
            self.hold("DOWN"); self.cap("menu")

    def tour_pub(self):
        """Enter the pub (A selects the highlighted Pub) and navigate its service menu.
        The Rumors/Missions/Quit/Leave list is present with a greeting text box on top -
        labeled MENU on purpose (the option list is what the agent must navigate)."""
        self.load()
        self.hold("A")        # location menu, Pub highlighted
        self.hold("A", then=200)   # enter pub, let it load
        if self._is_black():
            self.e.run(200)
        self.cap("menu")
        for _ in range(4):
            self.hold("DOWN"); self.cap("menu")
        for _ in range(4):
            self.hold("UP"); self.cap("menu")

    def tour_dialogs(self, rng: random.Random, count: int):
        """Help/text boxes with NO option list: SELECT at varied map nodes -> help box."""
        for _ in range(count):
            self.load()
            for _ in range(rng.randint(0, 3)):
                self.hold(rng.choice(DIRS))
            self.hold("SELECT")
            self.cap("dialog")

    def save(self):
        with open(os.path.join(self.out, "labels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "label", "ram_3cb7"])
            w.writerows(self.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="train/perception/mode_data")
    ap.add_argument("--episodes", type=int, default=3, help="repeats of the randomized tours")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    lab = Labeler(out)
    rng = random.Random(args.seed)
    for ep in range(args.episodes):
        lab.tour_field(rng, steps=20)
        lab.tour_wm_menus()
        lab.tour_pub()
        lab.tour_dialogs(rng, count=12)
        print(f"[label] episode {ep+1}/{args.episodes} done, total frames={lab.n}", flush=True)
    lab.save()
    # Report class balance + the 0x3cb7 cross-check per label.
    from collections import Counter, defaultdict
    by = Counter(r[1] for r in lab.rows)
    ram = defaultdict(Counter)
    for _, label, v in lab.rows:
        ram[label][v] += 1
    print("[label] class balance:", dict(by))
    for k in ("field", "menu", "dialog"):
        print(f"[label] 0x3cb7 values for {k}:", dict(ram[k]))
    print("[label] wrote", lab.n, "frames ->", out)


if __name__ == "__main__":
    main()
