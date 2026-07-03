#!/usr/bin/env python3
"""Tight coordinate finder: from a verified-interactive overworld state, require
SYMMETRIC small deltas (X: +k RIGHT / -k LEFT, unchanged vert; Y vice versa) and
exclude the low-EWRAM graphics scratch region. Saves the clean state for reuse."""
from emu import Emu, EWRAM_BASE

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Pokemon AI Red.gba"
CLEAN = "/home/timothy/projects/VGA/train/ram/ow_clean.state"

e = Emu(ROM); e.boot(load_save=True)
e.run(300)
for _ in range(8): e.tap('A', hold=3, then=25)
e.tap('A', hold=4, then=20); e.tap('A', hold=4, then=20)
e.run(360)
for _ in range(12): e.tap('A', hold=3, then=20)   # clear multi-page recap
e.run(600)                                          # settle to interactive overworld
raw = e.core.save_raw_state(); open(CLEAN, "wb").write(bytes(raw))

def hold(k, f):
    key = e._keymap[k]; e.core.set_keys(key); e.run(f); e.core.clear_keys(key); e.run(16)

def walk(direction, f=32):
    e.core.load_raw_state(raw); e.run(4); base = e.wram_snapshot()
    hold(direction, f); return base, e.wram_snapshot()

b, R = walk('RIGHT'); _, L = walk('LEFT'); _, D = walk('DOWN'); _, U = walk('UP')
SCRATCH = 0x1000  # skip low-EWRAM OAM/DMA scratch
def sd(x, i): return x[i] - b[i]   # signed-ish small delta (values small, no wrap)

def axis(pos1, neg1, off1, off2):  # pos increases, neg decreases; off must be unchanged
    out = []
    for i in range(SCRATCH, len(b)):
        dp, dn = sd(pos1, i), sd(neg1, i)
        if 1 <= dp <= 4 and -4 <= dn <= -1 and off1[i] == b[i] and off2[i] == b[i]:
            out.append(i)
    return out

xcand = axis(R, L, D, U)
ycand = axis(D, U, R, L)
print("=== X (RIGHT+ / LEFT- / vert fixed):", len(xcand))
for i in xcand:
    print("  X %#010x  base=%d R=%d L=%d D=%d U=%d" % (EWRAM_BASE+i, b[i], R[i], L[i], D[i], U[i]))
print("=== Y (DOWN+ / UP- / horiz fixed):", len(ycand))
for i in ycand:
    print("  Y %#010x  base=%d R=%d L=%d D=%d U=%d" % (EWRAM_BASE+i, b[i], R[i], L[i], D[i], U[i]))
print("clean state saved ->", CLEAN)
