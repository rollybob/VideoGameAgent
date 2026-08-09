"""_slot_timeline.py -- rigorous: does a bush/pot drop POPULATE the sprite-slot table in-window?

For each today ring, track all 16 slots across the 300 snapshots. Report every slot that is an
on-screen HP==0 sprite at the FINAL snapshot (the frame Tim F11'd right after the drop), and WHEN
it first appeared (contiguous run ending at the last snapshot) + what the slot held just before.
  - appeared mid/late window  => the drop SPAWNED into the sprite table (proof it uses 0x03846)
  - present the whole window   => item already at rest (still in the table, just not a fresh spawn)
Also flags rings with NO sprite-table item at the end (=> item elsewhere, or none on-screen).
Numbers, no eyeballing. Read-only. thor-rl:cu130.
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
RINGS = "/work/link/sessions/ramhits_solo"
IWRAM = 32 * 1024; SNAP = (256 + 32) * 1024
CAMX, CAMY = 0x02B82, 0x02B86


def load_iw(d):
    n = json.load(open(os.path.join(d, "meta.json")))["n"]
    a = np.empty((n, IWRAM), np.uint8)
    with open(os.path.join(d, "ring.bin"), "rb") as f:
        for t in range(n):
            f.seek(t * SNAP); a[t] = np.frombuffer(f.read(IWRAM), np.uint8)
    return a


def u16(a, addr):
    return a[:, addr].astype(np.int32) | (a[:, addr + 1].astype(np.int32) << 8)


spawned = present = none = 0
for d in sorted(glob.glob(os.path.join(RINGS, "20260808-*-hit"))):
    a = load_iw(d); n = a.shape[0]
    cx, cy = u16(a, CAMX), u16(a, CAMY)
    tag = os.path.basename(d)[9:15]
    hits = []
    for i in range(16):
        x, y = u16(a, 0x03846 + 4 * i), u16(a, 0x03848 + 4 * i)
        hp = a[:, 0x03250 + i].astype(np.int32)
        onscr = (x - cx >= 0) & (x - cx < 240) & (y - cy >= 0) & (y - cy < 160) & ~((x == 0) & (y == 0))
        item = onscr & (hp == 0)
        if not item[-1]:
            continue
        # contiguous run of item==True ending at the last snapshot
        t0 = n - 1
        while t0 > 0 and item[t0 - 1]:
            t0 -= 1
        before = "start-of-window" if t0 == 0 else (
            f"hp={hp[t0-1]} onscr={bool(onscr[t0-1])} pos=({int(x[t0-1]-cx[t0-1])},{int(y[t0-1]-cy[t0-1])})")
        secs_before_end = (n - 1 - t0) * 2 / 60.0
        hits.append((i, int(x[-1] - cx[-1]), int(y[-1] - cy[-1]), t0, secs_before_end, before))
    if not hits:
        none += 1
        print(f"{tag}: no sprite-table item at F11 moment")
        continue
    for (i, sx, sy, t0, sec, before) in hits:
        if t0 == 0:
            present += 1; kind = "PRESENT-whole-window"
        else:
            spawned += 1; kind = f"SPAWNED @snap {t0}/{n} (~{sec:.1f}s before F11)"
        print(f"{tag}: slot{i} item @screen({sx},{sy})  {kind}  before=[{before}]")

print(f"\n=== SUMMARY (22 rings) ===")
print(f"  rings with a sprite-table item that SPAWNED in-window : {spawned}")
print(f"  rings with a sprite-table item PRESENT whole window   : {present}")
print(f"  rings with NO sprite-table item at F11                : {none}")
print("DONE")
