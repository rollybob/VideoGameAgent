#!/usr/bin/env python3
"""Clear the recap text with A (dpad is ignored while it's up), verify the
player actually moves, save a CLEAN interactive savestate, then find X/Y."""
from emu import Emu, EWRAM_BASE

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Pokemon AI Red.gba"
STATE = "/home/timothy/projects/VGA/train/ram/ow_clean.state"

e = Emu(ROM); e.boot(load_save=True)
e.run(300)
for _ in range(8): e.tap('A', hold=3, then=25)   # main menu
e.tap('A', hold=4, then=20); e.tap('A', hold=4, then=20)  # CONTINUE (2 presses)
e.run(360)                                        # load overworld
for _ in range(8): e.tap('A', hold=3, then=16)    # clear recap text with A
e.run(30)
e.save_png("/tmp/fc5_ready.png")

def hold(k, f=32):
    key = e._keymap[k]; e.core.set_keys(key); e.run(f); e.core.clear_keys(key); e.run(12)

# verify interactivity: does holding RIGHT change RAM differently than DOWN?
raw = e.core.save_raw_state()
e.core.load_raw_state(raw); e.run(4); hold('RIGHT'); r = e.wram_snapshot()
e.core.load_raw_state(raw); e.run(4); hold('DOWN'); d = e.wram_snapshot()
print("interactive (RIGHT != DOWN):", r != d)

open(STATE, "wb").write(bytes(raw))

def walk(direction, f=32):
    e.core.load_raw_state(raw); e.run(4); base = e.wram_snapshot()
    hold(direction, f); return base, e.wram_snapshot()

b, R = walk('RIGHT'); _, L = walk('LEFT'); _, D = walk('DOWN'); _, U = walk('UP')
def chg(x,i): return x[i]!=b[i]
def same(x,i): return x[i]==b[i]
xcand = [i for i in range(len(b)) if (chg(R,i) or chg(L,i)) and same(D,i) and same(U,i)]
ycand = [i for i in range(len(b)) if (chg(D,i) or chg(U,i)) and same(R,i) and same(L,i)]
print("changed R=%d L=%d D=%d U=%d" % (sum(chg(R,i) for i in range(len(b))),
      sum(chg(L,i) for i in range(len(b))), sum(chg(D,i) for i in range(len(b))),
      sum(chg(U,i) for i in range(len(b)))))
print("=== X candidates:", len(xcand))
for i in xcand[:20]:
    print("  %#010x base=%3d R=%3d L=%3d D=%3d U=%3d" % (EWRAM_BASE+i,b[i],R[i],L[i],D[i],U[i]))
print("=== Y candidates:", len(ycand))
for i in ycand[:20]:
    print("  %#010x base=%3d R=%3d L=%3d D=%3d U=%3d" % (EWRAM_BASE+i,b[i],R[i],L[i],D[i],U[i]))
