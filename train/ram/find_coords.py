#!/usr/bin/env python3
"""Locate player X/Y in EWRAM by scripted-movement diffing -- robust to the
FRLG-hack address drift (we rely on MOVEMENT, not known vanilla offsets).

Method:
  1. reach overworld
  2. snapshot EWRAM
  3. walk one direction k tiles; the byte(s) that changed by exactly k are
     candidate coordinate(s). Repeat to shrink; walk back to confirm sign flip.
Saves frames so we can verify we're actually standing in the overworld.
"""
from emu import Emu, EWRAM_BASE

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Pokemon AI Red.gba"
e = Emu(ROM); e.boot(load_save=True)
e.run(240)
# get into the game (A+START mash like getin.py), then B out of any menu
for i in range(80):
    e.tap('A', hold=3, then=8)
    if i % 10 == 9:
        e.tap('START', hold=3, then=8)
for _ in range(12):
    e.tap('B', hold=3, then=8)
e.run(30)
e.save_png("/tmp/coords_overworld.png")
print("at overworld?, frame=%d" % e.frame_counter())

def walk(direction, tiles):
    # hold direction ~16 frames/tile to actually step (short taps only turn)
    for _ in range(tiles):
        e.tap(direction, hold=16, then=4)

def diff_by(a, b, delta):
    out = []
    for i in range(len(a)):
        d = (b[i] - a[i]) & 0xFF
        if d == delta:
            out.append(i)
    return set(out)

K = 3
s0 = e.wram_snapshot(); walk('RIGHT', K); s1 = e.wram_snapshot()
right1 = diff_by(s0, s1, K)
walk('RIGHT', K); s2 = e.wram_snapshot()
right2 = diff_by(s1, s2, K)
xcand = right1 & right2
print("X candidates (changed by +%d twice on RIGHT): %d" % (K, len(xcand)))
for off in sorted(xcand)[:20]:
    print("  X? addr=%#010x  vals: %d -> %d -> %d" % (EWRAM_BASE+off, s0[off], s1[off], s2[off]))

s3 = e.wram_snapshot(); walk('DOWN', K); s4 = e.wram_snapshot()
down1 = diff_by(s3, s4, K)
walk('DOWN', K); s5 = e.wram_snapshot()
down2 = diff_by(s4, s5, K)
ycand = down1 & down2
print("Y candidates (changed by +%d twice on DOWN): %d" % (K, len(ycand)))
for off in sorted(ycand)[:20]:
    print("  Y? addr=%#010x  vals: %d -> %d -> %d" % (EWRAM_BASE+off, s3[off], s4[off], s5[off]))

e.save_png("/tmp/coords_after.png")
print("done, frame=%d" % e.frame_counter())
