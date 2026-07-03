#!/usr/bin/env python3
"""Generate CORRECT (frame, goal, button) menu-navigation demonstrations for supervised
fine-tuning. This teaches the exact behaviour the VLM fails at (verified 2026-07-02): in a
menu, MOVE the cursor toward the option named by the goal, and COMMIT with A when the cursor
is on it. Prompt-tuning made this brittle (mash A <-> paralysis); demonstrations teach it
in the weights instead.

Ground truth is by CONSTRUCTION: we drive the headless emulator, so we always know the cursor
position and the target, hence the correct action. No labels to guess.

Demo kinds:
  commit : cursor is ON the target option -> A            (the step the model refuses to take)
  move   : cursor is ABOVE the target      -> down         (navigate toward it)
  dialog : a text box with no option list  -> A            (advance)

Output <out>/frames/NNNNNN.png (native 240x160) + <out>/demos.jsonl
       {frame, goal, button, kind, menu, cursor, target}.
"""
from __future__ import annotations

import argparse
import csv  # noqa: F401 (kept for parity with sibling tools)
import json
import os

from emu import Emu

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
HERE = os.path.dirname(__file__)

# Menus we can reach headlessly, top-to-bottom option labels, and how to open them from a
# base savestate. Cursor always starts at index 0 (top) after opening.
MENUS = [
    {"name": "pub", "state": "ffta_pub.state", "open": [],
     "options": ["Rumors", "Missions", "Quit Mission", "Leave"]},
    {"name": "wm_location", "state": "ffta_worldmap.state", "open": ["A"],
     "options": ["Pub", "Shop", "Monster Bank"]},
    {"name": "wm_main", "state": "ffta_worldmap.state", "open": ["START"],
     "options": ["Party", "Area List", "System"]},
]


class Gen:
    def __init__(self, out: str):
        self.out = out
        os.makedirs(os.path.join(out, "frames"), exist_ok=True)
        self.rows = []
        self.n = 0
        self.e = Emu(ROM)
        self.e.boot(load_save=True)

    def _open(self, menu):
        self.e.core.load_raw_state(open(os.path.join(HERE, menu["state"]), "rb").read())
        self.e.run(2)
        for k in menu["open"]:
            self._press(k)
        self.e.run(20)

    def _press(self, k, f=6, then=12):
        key = self.e._keymap[k]
        self.e.core.set_keys(key); self.e.run(f); self.e.core.clear_keys(key); self.e.run(then)

    def _cursor_to(self, menu, pos):
        """Reopen the menu (cursor at 0) and move down `pos` times."""
        self._open(menu)
        for _ in range(pos):
            self._press("DOWN")

    def _emit(self, goal, button, kind, menu, cursor, target):
        rel = os.path.join("frames", f"{self.n:06d}.png")
        self.e.save_png(os.path.join(self.out, rel))
        self.rows.append({"frame": rel, "goal": goal, "button": button, "kind": kind,
                          "menu": menu, "cursor": cursor, "target": target})
        self.n += 1

    def gen_menu(self, menu):
        opts = menu["options"]
        for target in range(len(opts)):
            goal = f"Select the '{opts[target]}' option from the menu."
            for pos in range(len(opts)):
                self._cursor_to(menu, pos)
                if pos == target:
                    self._emit(goal, "A", "commit", menu["name"], pos, target)   # COMMIT
                elif pos < target:
                    self._emit(goal, "down", "move", menu["name"], pos, target)   # move down
                else:
                    self._emit(goal, "up", "move", menu["name"], pos, target)     # move up

    def gen_dialogs(self):
        # Help boxes (SELECT on the world map) - text, no option list -> advance with A.
        for pre in ([], ["RIGHT"], ["LEFT"], ["RIGHT", "RIGHT"]):
            self.e.core.load_raw_state(open(os.path.join(HERE, "ffta_worldmap.state"), "rb").read())
            self.e.run(2)
            for k in pre:
                self._press(k)
            self._press("SELECT")
            self._emit("Read and advance the dialogue.", "A", "dialog", "help", -1, -1)

    def save(self):
        with open(os.path.join(self.out, "demos.jsonl"), "w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "perception", "demos"))
    args = ap.parse_args()
    g = Gen(os.path.abspath(args.out))
    for menu in MENUS:
        g.gen_menu(menu)
    g.gen_dialogs()
    g.save()
    from collections import Counter
    kinds = Counter(r["kind"] for r in g.rows)
    btns = Counter(r["button"] for r in g.rows)
    print("[gen_demos] wrote", g.n, "demos ->", g.out)
    print("[gen_demos] kinds:", dict(kinds), "| buttons:", dict(btns))


if __name__ == "__main__":
    main()
