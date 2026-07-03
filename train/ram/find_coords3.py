#!/usr/bin/env python3
"""Nail player X/Y via full directional cross-check, resetting to the SAME
overworld position (savestate) before each direction. Eliminates counters/RNG
(which change regardless of direction) and isolates true coordinates:
  X: increases on RIGHT, decreases on LEFT, UNCHANGED on UP/DOWN
  Y: increases on DOWN,  decreases on UP,   UNCHANGED on LEFT/RIGHT
"""
from emu import Emu, EWRAM_BASE

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Pokemon AI Red.gba"


def reach_overworld(e):
    e.run(300)
    for _ in range(8):
        e.tap('A', hold=3, then=25)
    e.tap('A', hold=4, then=20)
    e.tap('A', hold=4, then=20)
    e.run(360)
    for _ in range(3):
        e.tap('B', hold=3, then=10)
    e.run(30)


def hold(e, key, frames=60):
    k = e._keymap[key.upper()]
    e.core.set_keys(k); e.run(frames); e.core.clear_keys(k); e.run(8)


e = Emu(ROM); e.boot(load_save=True)
reach_overworld(e)
raw = e.core.save_raw_state()            # keep in-memory state object

def walk_from_state(direction):
    e.core.load_raw_state(raw)
    e.run(4)
    base = e.wram_snapshot()
    hold(e, direction, 60)
    after = e.wram_snapshot()
    return base, after

base_R, R = walk_from_state('RIGHT')
base_L, L = walk_from_state('LEFT')
base_D, D = walk_from_state('DOWN')
base_U, U = walk_from_state('UP')
b = base_R  # all bases identical (same savestate)

def d(x, y, i): return (x[i] - y[i]) & 0xFF   # unsigned delta
def inc(x, i):  # increased (small positive)
    dv = x[i] - b[i]; return 1 <= dv <= 12
def dec(x, i):
    dv = x[i] - b[i]; return -12 <= dv <= -1
def same(x, i): return x[i] == b[i]

xcand = [i for i in range(len(b))
         if inc(R, i) and dec(L, i) and same(D, i) and same(U, i)]
ycand = [i for i in range(len(b))
         if inc(D, i) and dec(U, i) and same(R, i) and same(L, i)]

print("=== X (RIGHT+/LEFT-/vert unchanged):", len(xcand))
for i in xcand[:15]:
    print("  %#010x  base=%d R=%d L=%d D=%d U=%d" % (EWRAM_BASE+i, b[i], R[i], L[i], D[i], U[i]))
print("=== Y (DOWN+/UP-/horiz unchanged):", len(ycand))
for i in ycand[:15]:
    print("  %#010x  base=%d R=%d L=%d D=%d U=%d" % (EWRAM_BASE+i, b[i], R[i], L[i], D[i], U[i]))
