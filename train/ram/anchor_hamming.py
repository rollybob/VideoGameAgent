#!/usr/bin/env python3
"""Measure the aHash (256-bit, plugin _ahash) Hamming distances the mode anti-anchor will
gate on, so the threshold is picked from DATA, not blind (Tim's brittleness concern).

We want: menu-CLOSE (must clear the stale 'menu' mode) to separate cleanly from
within-menu NAVIGATION (must NOT clear -- stay 'menu') and map->map DRIFT. If there is a
wide gap, a threshold in the gap is robust. Uses the exact plugin hash so numbers transfer.
"""
import sys, numpy as np, cv2
from emu import Emu

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
STATE = "ffta_worldmap.state"
_AHASH_SIZE = 16

def ahash(emu):  # identical to vga/reason/plugin.py _ahash (16x16, INTER_AREA, 256-bit)
    bgr = cv2.cvtColor(np.array(emu.image.to_pil().convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (_AHASH_SIZE, _AHASH_SIZE), interpolation=cv2.INTER_AREA)
    bits = (small >= small.mean()).astype(np.uint8).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")

def ham(a, b):
    return int(a ^ b).bit_count()

def main():
    emu = Emu(ROM); emu.boot(load_save=True)
    with open(STATE, "rb") as f:
        emu.core.load_raw_state(f.read())
    emu.run(30)

    def tap(k):
        emu.tap(k, hold=6, then=30); return ahash(emu)

    h_map0 = ahash(emu)
    # map->map drift (walk on the world map, no menu)
    map_drift = []
    h = h_map0
    for mv in ["RIGHT", "DOWN", "LEFT", "UP"]:
        h2 = tap(mv); map_drift.append(ham(h, h2)); h = h2
    # reload clean for the menu sequence
    with open(STATE, "rb") as f:
        emu.core.load_raw_state(f.read())
    emu.run(30)
    h_map = ahash(emu)
    h_open = tap("START")                     # map -> Start menu (OPEN)
    d_open = ham(h_map, h_open)
    h_nav1 = tap("DOWN")                       # within Start menu (NAV)
    d_nav1 = ham(h_open, h_nav1)
    h_deep = tap("A")                          # Start menu -> Area List (deeper)
    d_deep = ham(h_nav1, h_deep)
    h_nav2 = tap("DOWN")                       # within Area List (NAV)
    d_nav2 = ham(h_deep, h_nav2)
    h_nav3 = tap("DOWN")
    d_nav3 = ham(h_nav2, h_nav3)
    # CLOSE the menus back to the map (B may go up a level; press several, record each delta)
    closes = []
    hp = h_nav3
    for _ in range(4):
        h2 = tap("B"); closes.append(ham(hp, h2)); hp = h2

    print(f"map->map DRIFT (no menu):        {map_drift}   max={max(map_drift)}")
    print(f"map->OPEN Start menu:            {d_open}")
    print(f"within-menu NAV (down/A deeper): {[d_nav1, d_deep, d_nav2, d_nav3]}   max={max(d_nav1,d_deep,d_nav2,d_nav3)}")
    print(f"menu CLOSE (B presses):          {closes}   max={max(closes)}")
    print(f"\nSUBGOAL_STALE_HAMMING (current, reused?) = 48")
    print("Robust iff menu-CLOSE >> {map-drift, within-menu-nav}. Pick threshold in the gap.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
