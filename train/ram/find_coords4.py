#!/usr/bin/env python3
"""Player X/Y via short moves from the saved town-path savestate (no grass =>
no wild-encounter contamination). Deterministic reload before each direction.
Relaxed axis test (handles a direction being wall-blocked):
  X: changes on RIGHT and/or LEFT, UNCHANGED on both UP and DOWN
  Y: changes on DOWN and/or UP,    UNCHANGED on both LEFT and RIGHT
"""
from emu import Emu, EWRAM_BASE

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Pokemon AI Red.gba"
STATE = "/home/timothy/projects/VGA/train/ram/ow.state"

e = Emu(ROM); e.boot(load_save=True)
raw = open(STATE, "rb").read()
e.core.load_raw_state(raw); e.run(4)
e.save_png("/tmp/fc4_base.png")

def walk(direction, frames):
    e.core.load_raw_state(raw); e.run(4)
    base = e.wram_snapshot()
    k = e._keymap[direction]; e.core.set_keys(k); e.run(frames); e.core.clear_keys(k); e.run(12)
    return base, e.wram_snapshot()

F = 20   # ~1 tile
b, R = walk('RIGHT', F)
_, L = walk('LEFT', F)
_, D = walk('DOWN', F)
_, U = walk('UP', F)

def chg(x, i): return x[i] != b[i]
def same(x, i): return x[i] == b[i]

xcand = [i for i in range(len(b)) if (chg(R,i) or chg(L,i)) and same(D,i) and same(U,i)]
ycand = [i for i in range(len(b)) if (chg(D,i) or chg(U,i)) and same(R,i) and same(L,i)]

print("frames/dir=%d  changed R=%d L=%d D=%d U=%d" %
      (F, sum(chg(R,i) for i in range(len(b))), sum(chg(L,i) for i in range(len(b))),
       sum(chg(D,i) for i in range(len(b))), sum(chg(U,i) for i in range(len(b)))))
print("=== X candidates (horiz only):", len(xcand))
for i in xcand[:20]:
    print("  %#010x base=%3d R=%3d L=%3d D=%3d U=%3d" % (EWRAM_BASE+i, b[i], R[i], L[i], D[i], U[i]))
print("=== Y candidates (vert only):", len(ycand))
for i in ycand[:20]:
    print("  %#010x base=%3d R=%3d L=%3d D=%3d U=%3d" % (EWRAM_BASE+i, b[i], R[i], L[i], D[i], U[i]))
