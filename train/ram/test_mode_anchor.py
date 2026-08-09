#!/usr/bin/env python3
"""Deterministic unit test of the mode anti-anchor (no VLM/GPU). Drives the REAL VlmPlugin
with a recording stub reasoner over REAL emulator frames, asserting self._mode is:
  - CLEARED when a menu closes (big scene change) so the model re-perceives the map, and
  - PRESERVED within a menu (small change), so we do not lose genuine menu persistence.
This is the offline gate before spending a bench (Tim's brittleness concern)."""
import sys, numpy as np, cv2, time
sys.path.insert(0, "/home/timothy/projects/VGA")
from emu import Emu
from vga.reason.plugin import VlmPlugin
from vga.core.contract import GameState
from vga.reason.base import ReasonerDecision, action_from_choice

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
STATE = "ffta_worldmap.state"

class RecordingReasoner:
    """Always reports mode='menu' (the anchoring scenario) and records the ctx.mode it was
    GIVEN each step -- that is the carried prior the plugin decided to feed (or clear)."""
    name = "rec"
    def __init__(self): self.seen_modes = []
    def decide(self, frame, ctx):
        self.seen_modes.append(ctx.mode)
        return ReasonerDecision(action=action_from_choice("wait"), reason="",
                                meta={"mode": "menu", "subgoal": "", "progress": ""})
    def reset(self): pass

def frame_bgr(emu):
    return cv2.cvtColor(np.array(emu.image.to_pil().convert("RGB")), cv2.COLOR_RGB2BGR)

def gs(emu, i):
    b = frame_bgr(emu); h, w = b.shape[:2]
    return GameState(frame_index=i, timestamp=time.time(), width=w, height=h, frame=b)

def main():
    emu = Emu(ROM); emu.boot(load_save=True)
    with open(STATE, "rb") as f: emu.core.load_raw_state(f.read())
    emu.run(30)

    rec = RecordingReasoner()
    plug = VlmPlugin(reasoner=rec, goal="test")

    i = 0
    def step():
        nonlocal i
        plug.decide(gs(emu, i)); i += 1
        return plug._mode  # mode AFTER this step (model reported 'menu')

    # 1) On a MENU frame: model reports menu -> _mode becomes 'menu'.
    emu.tap("START", hold=6, then=30)          # open Start menu (ovl 255)
    m1 = step()
    # 2) Within the menu (small change): mode must PERSIST.
    emu.tap("DOWN", hold=6, then=30)
    seen_before_nav = rec.seen_modes[-1]
    m2 = step()
    seen_on_nav = rec.seen_modes[-1]
    # 3) CLOSE the menu back to the map (big change): the NEXT step must be fed mode='' (cleared).
    emu.tap("B", hold=6, then=30); emu.tap("B", hold=6, then=30)  # exit to world map
    m3 = step()
    seen_on_map = rec.seen_modes[-1]

    print(f"after menu step:        _mode={m1!r}  (expect 'menu')")
    print(f"within-menu nav:        ctx.mode fed={seen_on_nav!r}  _mode={m2!r}  (expect fed 'menu', persists)")
    print(f"after menu CLOSE->map:  ctx.mode fed to reasoner on the map frame = {seen_on_map!r}  (expect '' = CLEARED)")

    ok = (m1 == "menu") and (seen_on_nav == "menu") and (seen_on_map == "")
    print("\nRESULT:", "PASS - mode persists in-menu, clears on close" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
