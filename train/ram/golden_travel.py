#!/usr/bin/env python3
"""Deterministic golden-path probe for the world-map -> Giza Plains travel commit.

Motivation (rescue day 2, 2026-07-04): the R4 wall (at_giza 4/8) is a travel-COMMIT
reliability problem, not navigation ignorance -- the VLM reaches the Area List but
oscillates Start-menu<->Area-List and self-cancels. Before choosing a fix we pin the
EXACT working key macro deterministically (same rigor that cracked naming/timing),
watching clan_pos (20 == Giza) and mode_overlay at every press. No VLM in the loop.

Run in the container from train/ram:
  docker run --rm --runtime nvidia -v ~/projects/VGA:/work -w /work/train/ram \
      thor-torch:cu130 python3 golden_travel.py
"""
import sys
from emu import Emu
from oracle import for_rom

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
STATE = "ffta_worldmap.state"
THEN = 30  # the harness cadence that unblocked naming; boxes ignore input ~30 frames

def snap(emu, oracle, tag):
    s = oracle.read_state(emu)
    print(f"  {tag:16s} scene={s.get('scene'):>3} ovl={s.get('mode_overlay'):>4} "
          f"wmcur={s.get('worldmap_cursor'):>3} pos={s.get('clan_pos'):>3} "
          f"funds={s.get('clan_funds')}")
    return s

def press(emu, oracle, key, tag=""):
    emu.tap(key, hold=6, then=THEN)
    return snap(emu, oracle, tag or key)

def main():
    emu = Emu(ROM)
    emu.boot(load_save=True)
    with open(STATE, "rb") as f:
        emu.core.load_raw_state(f.read())
    emu.run(30)
    oracle = for_rom(ROM)
    print(f"[golden_travel] state={STATE} then={THEN}")
    snap(emu, oracle, "LOAD")
    emu.save_png("/tmp/gt_load.png")

    # Candidate golden macro derived from the winning benches (ep_07/ep_00):
    #   START -> open world-map Start menu (Party / Area List / System)
    #   DOWN  -> move cursor to "Area List"
    #   A     -> open the Area List (Cyril / Sprohm / Giza Plains, cursor on Cyril=top)
    #   DOWN,DOWN -> highlight "Giza Plains" (3rd entry)
    #   A     -> select it; a free map cursor drops onto the Giza Plains node (labeled)
    #   A     -> confirm travel; caravan walks; clan_pos should reach 20 (Giza)
    # Open list, highlight Giza Plains, select -> free cursor drops ONTO Giza node, then a
    # SINGLE A confirms travel; the caravan then ANIMATES node-by-node -- so we idle-wait and
    # watch pos settle rather than mashing A (extra presses re-steer the cursor off Giza).
    seq = ["START", "DOWN", "A", "DOWN", "DOWN", "A", "A"]  # ends with cursor on Giza, 1st confirm
    for i, key in enumerate(seq):
        s = press(emu, oracle, key, tag=f"{i:02d} {key}")
        emu.save_png(f"/tmp/gt_{i:02d}_{key}.png")
        if s.get("clan_pos") == 20:
            print(f"[golden_travel] REACHED GIZA after {i+1} presses (no wait): {seq[:i+1]}")
            return 0
    # Let the travel animation play out.
    for w in range(12):
        emu.run(30)
        s = snap(emu, oracle, f"wait+{w}")
        emu.save_png(f"/tmp/gt_wait_{w:02d}.png")
        if s.get("clan_pos") == 20:
            print(f"[golden_travel] REACHED GIZA (pos=20) after confirm + {w+1} idle waits")
            return 0
    print("[golden_travel] did NOT reach Giza; inspect /tmp/gt_*.png")
    return 1

if __name__ == "__main__":
    sys.exit(main())
