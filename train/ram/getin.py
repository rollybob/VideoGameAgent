#!/usr/bin/env python3
"""Boot Pokemon AI Red headless and mash through the intro to the overworld,
dumping frames along the way so we can see where scripted input lands us.
Once we can reach the overworld we can do the movement-diff address search."""
import sys
from emu import Emu

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Pokemon AI Red.gba"
e = Emu(ROM)
e.boot(load_save=True)
e.run(240)                       # boot through BIOS + first logos
e.save_png("/tmp/getin_000.png")

# Mash A+START alternately to blow through logos, intro cutscene, main menu.
for i in range(1, 81):
    e.tap('A', hold=3, then=10)
    if i % 10 == 0:
        e.tap('START', hold=3, then=10)
        e.save_png("/tmp/getin_%03d.png" % i)
        print("after %d taps, frame=%d" % (i, e.frame_counter()))

e.save_png("/tmp/getin_final.png")
print("done, frame=%d" % e.frame_counter())
