#!/usr/bin/env python3
"""Interactive-ish headless explorer for the FFTA pub->mission chain.

Drive ffta_pub.state along a button script given on argv, saving a frame PNG after each
press and printing candidate RAM values, so we can SEE each checkpoint screen (Read the
PNGs) BEFORE trusting any address. Grounds the checkpoint ladder in actual pixels - we
both misread the pub screen last session, so eyes before addresses.

Usage (from repo root, host venv):
    .venv/bin/python train/ram/explore_pub.py DOWN A A
    .venv/bin/python train/ram/explore_pub.py --hold 8 --then 40 --out /tmp/pub DOWN A A

Each press: hold `hold` frames, idle `then` frames (menus need long settle on scene loads).
Frames -> <out>/step_NN_<btn>.png ; also dumps <out>/ewram_final.bin for diffing.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from emu import Emu  # noqa: E402

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
STATE = os.path.join(os.path.dirname(__file__), "ffta_pub.state")

# Candidate address probes we already trust or suspect, printed each step for context.
PROBES = {
    "worldmap_cursor@2c10": (0x02002c10, "u16"),
    "modeflag@3cb7":        (0x02003cb7, "u8"),
}


def read(emu, addr, width):
    return {"u8": emu.u8, "u16": emu.u16, "u32": emu.u32}[width](addr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script", nargs="*", help="button sequence, e.g. DOWN A A")
    ap.add_argument("--hold", type=int, default=6)
    ap.add_argument("--then", type=int, default=30)
    ap.add_argument("--out", default="/tmp/pub_explore")
    ap.add_argument("--state", default=STATE)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    e = Emu(ROM)
    e.boot(load_save=True)
    with open(args.state, "rb") as f:
        e.core.load_raw_state(f.read())
    e.run(2)

    def snap(idx, label):
        path = os.path.join(args.out, f"step_{idx:02d}_{label}.png")
        e.save_png(path)
        probes = " ".join(f"{k}={read(e, a, w)}" for k, (a, w) in PROBES.items())
        print(f"[{idx:02d}] {label:8s} {probes}  -> {path}", flush=True)

    snap(0, "START")
    for i, btn in enumerate(args.script, 1):
        e.tap(btn.upper(), hold=args.hold, then=args.then)
        snap(i, btn.lower())

    # Dump full EWRAM for offline diffing.
    with open(os.path.join(args.out, "ewram_final.bin"), "wb") as f:
        f.write(e.wram_snapshot())
    print(f"[done] wrote frames + ewram_final.bin to {args.out}", flush=True)


if __name__ == "__main__":
    main()
