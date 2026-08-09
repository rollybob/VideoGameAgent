#!/usr/bin/env python3
"""Isolate the R4 root cause: is the model's 46% false 'menu' belief (mode=menu while the
menu is CLOSED, ovl=0) genuine VISUAL misperception, or CONTEXT-ANCHORING (it carries a
running 'navigate the menu' subgoal forward and parrots 'menu' even on a plain world map)?

Method: build a labeled frame set (true ovl from RAM) spanning world-map (ovl 0) and menu
(ovl 255/204) states, then POST each frame to the VLM /act endpoint under two contexts:
  NEUTRAL  - empty subgoal/mode/history (raw perception)
  ANCHORED - subgoal='navigate to Area List', mode='menu', menu-action history
Compare the returned `mode` to truth. If NEUTRAL is accurate but ANCHORED flips map->menu,
the fix is cheap (stop carrying the stale belief / re-perceive fresh). If NEUTRAL is already
wrong on map frames, it's a genuine vision gap (the Perception->State layer). No training.
"""
import base64, json, sys, urllib.request
import cv2
from emu import Emu
from oracle import for_rom

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
STATE = "ffta_worldmap.state"
URL = "http://127.0.0.1:8077/act"

def enc(emu):
    bgr = cv2.cvtColor(__import__("numpy").array(emu.image.to_pil().convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", bgr)
    return base64.standard_b64encode(buf.tobytes()).decode()

def ask(img_b64, anchored):
    if anchored:
        ctx = dict(subgoal="Move cursor down to 'Area List' and press A", mode="menu",
                   progress="navigating the menu to open the Area List",
                   history=[{"button": "down", "reason": "navigating menu", "changed": False}] * 3)
    else:
        ctx = dict(subgoal="", mode="", progress="", history=[])
    payload = json.dumps({
        "image_b64": img_b64, "goal": "You are playing a game. Decide the next input.",
        "step": 0, "last_action": "", "last_changed": None, "looping": False,
        "dialog_text": "", "already_read": False, "skills": [], "knowledge": [], **ctx,
    }).encode()
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read().decode())
    return resp.get("mode", ""), resp.get("subgoal", ""), resp.get("button", "")

def main():
    emu = Emu(ROM); emu.boot(load_save=True)
    with open(STATE, "rb") as f:
        emu.core.load_raw_state(f.read())
    emu.run(30)
    oracle = for_rom(ROM)

    # Build a labeled set: (label, frame_b64, true_ovl). Map states (ovl 0) and menu states.
    frames = []
    def grab(tag):
        s = oracle.read_state(emu); ov = s.get("mode_overlay")
        frames.append((tag, enc(emu), ov))
    grab("map@load")                                   # ovl 0
    emu.tap("START", hold=6, then=30); grab("startmenu")   # ovl 255
    emu.tap("DOWN", hold=6, then=30);  grab("startmenu2")  # ovl 255
    emu.tap("A", hold=6, then=30);     grab("arealist")    # ovl 204
    emu.tap("B", hold=6, then=30);     grab("afterB1")     # back toward map
    emu.tap("B", hold=6, then=30);     grab("afterB2")     # map again
    emu.tap("START", hold=6, then=30); grab("startmenu3")  # ovl 255
    emu.tap("B", hold=6, then=30);     grab("afterB3")     # map

    print(f"labeled frames: {[(t,o) for t,_,o in frames]}\n")
    truth = lambda ov: "menu" if ov not in (0, 19) else "overworld"
    # Accuracy tracked separately for MAP frames (the false-menu risk) and MENU frames.
    acc = {("map", "neutral"): [0, 0], ("map", "anchored"): [0, 0],
           ("menu", "neutral"): [0, 0], ("menu", "anchored"): [0, 0]}
    for tag, b64, ov in frames:
        tr = truth(ov)
        mn, _, bn = ask(b64, anchored=False)
        ma, _, ba = ask(b64, anchored=True)
        grp = "map" if tr == "overworld" else "menu"
        for cond, m in (("neutral", mn), ("anchored", ma)):
            acc[(grp, cond)][1] += 1
            if m == tr:
                acc[(grp, cond)][0] += 1
        print(f"[{tag:11s} ovl={ov:3d} truth={tr:9s}]  neutral: mode={mn!r:11s} btn={bn:5s}  |  "
              f"anchored: mode={ma!r:11s} btn={ba:5s}")
    print("\n--- classification accuracy (mode == truth) ---")
    for (grp, cond), (ok, tot) in acc.items():
        if tot:
            print(f"  {grp:4s} frames, {cond:8s} context: {ok}/{tot} correct")
    print("\nRead: if MAP/neutral is high but MAP/anchored is low -> CONTEXT-ANCHORING (cheap "
          "fix). If MAP/neutral is already low -> genuine VISION gap (Perception->State layer).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
