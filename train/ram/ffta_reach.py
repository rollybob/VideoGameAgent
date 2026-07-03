#!/usr/bin/env python3
"""Reach the FFTA world map from cold boot, save a clean savestate, and find the
world-map cursor/position coords by the same movement-diff method used for Pokemon.
Cross-game generality check for the F21 RAM oracle."""
from emu import Emu, EWRAM_BASE

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
CLEAN = "/home/timothy/projects/VGA/train/ram/ffta_worldmap.state"


def reach_worldmap(e):
    e.run(300); e.tap('A', hold=4, then=40)              # language -> English
    e.run(950)                                           # logos
    e.tap('START', hold=4, then=40); e.tap('START', hold=4, then=60)  # -> title
    e.tap('START', hold=4, then=40); e.tap('A', hold=4, then=40)      # main menu
    e.tap('A', hold=4, then=40)                          # Saved Game -> submenu
    e.tap('A', hold=4, then=40)                          # Load -> slot picker
    e.tap('A', hold=4, then=40)                          # FILE NO.1 (Marche)
    e.run(400)                                           # load into world map
    for _ in range(6):                                   # dismiss any intro text
        e.tap('A', hold=3, then=16)
    e.run(120)


e = Emu(ROM); e.boot(load_save=True)
reach_worldmap(e)
e.save_png("/tmp/ffta_wm_ready.png")

def hold(k, f):
    key = e._keymap[k]; e.core.set_keys(key); e.run(f); e.core.clear_keys(key); e.run(16)

raw = e.core.save_raw_state()
e.core.load_raw_state(raw); e.run(4); hold('RIGHT', 32); r = e.wram_snapshot()
e.core.load_raw_state(raw); e.run(4); hold('DOWN', 32);  d = e.wram_snapshot()
interactive = (r != d)
print("interactive (R!=D):", interactive)
open(CLEAN, "wb").write(bytes(raw))

if interactive:
    def walk(direction, f=32):
        e.core.load_raw_state(raw); e.run(4); base = e.wram_snapshot()
        hold(direction, f); return base, e.wram_snapshot()
    b, R = walk('RIGHT'); _, L = walk('LEFT'); _, D = walk('DOWN'); _, U = walk('UP')
    def sd(x, i): return x[i] - b[i]
    def axis(pos, neg, o1, o2):
        return [i for i in range(0x1000, len(b))
                if 1 <= sd(pos, i) <= 8 and -8 <= sd(neg, i) <= -1
                and o1[i] == b[i] and o2[i] == b[i]]
    X = axis(R, L, D, U); Y = axis(D, U, R, L)
    # loosen if strict-symmetric finds nothing (cursor may be blocked one way)
    if not X and not Y:
        chg = lambda x, i: x[i] != b[i]; same = lambda x, i: x[i] == b[i]
        X = [i for i in range(0x1000, len(b)) if (chg(R,i) or chg(L,i)) and same(D,i) and same(U,i)]
        Y = [i for i in range(0x1000, len(b)) if (chg(D,i) or chg(U,i)) and same(R,i) and same(L,i)]
        print("(loosened: changed-on-axis / fixed-on-other)")
    print("=== X candidates:", len(X))
    for i in X[:20]:
        print("  X %#010x base=%d R=%d L=%d D=%d U=%d" % (EWRAM_BASE+i, b[i], R[i], L[i], D[i], U[i]))
    print("=== Y candidates:", len(Y))
    for i in Y[:20]:
        print("  Y %#010x base=%d R=%d L=%d D=%d U=%d" % (EWRAM_BASE+i, b[i], R[i], L[i], D[i], U[i]))
print("saved ->", CLEAN)
