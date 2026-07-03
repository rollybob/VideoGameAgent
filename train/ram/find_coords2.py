#!/usr/bin/env python3
"""Locate player X/Y in Pokemon AI Red by scripted-movement diffing.
Reaches the overworld, saves a savestate for reuse, then finds the coordinate
bytes: hold a direction twice; the byte that increases both times is that axis."""
from emu import Emu, EWRAM_BASE

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Pokemon AI Red.gba"
STATE = "/home/timothy/projects/VGA/train/ram/ow.state"


def reach_overworld(e):
    e.run(300)
    for _ in range(8):
        e.tap('A', hold=3, then=25)     # -> main menu (CONTINUE)
    e.tap('A', hold=4, then=20)          # select
    e.tap('A', hold=4, then=20)          # confirm
    e.run(360)                           # load
    for _ in range(3):
        e.tap('B', hold=3, then=10)      # dismiss recap text
    e.run(30)


def hold(e, key, frames):
    k = e._keymap[key.upper()]
    e.core.set_keys(k); e.run(frames); e.core.clear_keys(k); e.run(8)


def increased(a, b, lo=1, hi=8):
    return {i for i in range(len(a)) if lo <= ((b[i] - a[i]) & 0xFF) <= hi}


e = Emu(ROM); e.boot(load_save=True)
reach_overworld(e)
e.save_png("/tmp/fc2_overworld.png")

# save a savestate for fast reuse in later runs
try:
    raw = e.core.save_raw_state()
    with open(STATE, "wb") as f:
        f.write(bytes(raw))
    print("saved savestate ->", STATE, len(bytes(raw)), "bytes")
except Exception as ex:
    print("savestate failed:", ex)

# X: hold RIGHT twice
a = e.wram_snapshot(); hold(e, 'RIGHT', 60); b = e.wram_snapshot()
hold(e, 'RIGHT', 60); c = e.wram_snapshot()
xcand = increased(a, b) & increased(b, c)
print("X candidates:", len(xcand))
for off in sorted(xcand)[:25]:
    print("  X? %#010x  %d -> %d -> %d" % (EWRAM_BASE+off, a[off], b[off], c[off]))

# Y: hold DOWN twice
a2 = e.wram_snapshot(); hold(e, 'DOWN', 60); b2 = e.wram_snapshot()
hold(e, 'DOWN', 60); c2 = e.wram_snapshot()
ycand = increased(a2, b2) & increased(b2, c2)
print("Y candidates:", len(ycand))
for off in sorted(ycand)[:25]:
    print("  Y? %#010x  %d -> %d -> %d" % (EWRAM_BASE+off, a2[off], b2[off], c2[off]))

e.save_png("/tmp/fc2_after.png")
print("done frame=%d" % e.frame_counter())
