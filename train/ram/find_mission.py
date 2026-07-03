#!/usr/bin/env python3
"""Known-change savestate diffing to find FFTA mission-acceptance RAM.

From ffta_pub.state, drive the deterministic golden path to the "Accept these conditions?
Yes/No" prompt (DOWN + A*10), snapshot EWRAM, then branch:
  ACCEPT : press A (Yes)  -> Clan Funds 5000->4700, mission added to accepted list.
  DECLINE: press B (cancel) as a CONTROL -> no acceptance effect.
Intersect: bytes that changed ONLY on accept = the acceptance effect, with menu-navigation
and animation noise subtracted by the control.

Two targets:
  money         : u16/u32 offset that is 5000 pre and 4700 post-accept (clean known-change).
  accept_record : offsets changed on accept but NOT on decline (accepted-mission counter/flag).

Headless, CPU, no GPU. Run from repo root: .venv/bin/python train/ram/find_mission.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from emu import Emu  # noqa: E402

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
STATE = os.path.join(os.path.dirname(__file__), "ffta_pub.state")
EWRAM_BASE = 0x02000000

# DOWN then 10x A reaches the "Accept these conditions? Yes/No" prompt (Yes highlighted).
TO_PROMPT = ["DOWN"] + ["A"] * 10


def drive_to_prompt(e):
    e.core.reset()
    with open(STATE, "rb") as f:
        e.core.load_raw_state(f.read())
    e.run(2)
    for b in TO_PROMPT:
        e.tap(b, hold=6, then=55)


def snap(e):
    return e.wram_snapshot()


def u16(buf, off):
    return buf[off] | (buf[off + 1] << 8)


def u32(buf, off):
    return buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16) | (buf[off + 3] << 24)


def main():
    e = Emu(ROM)
    e.boot(load_save=True)

    # PRE: at the Yes/No prompt.
    drive_to_prompt(e)
    pre = snap(e)

    # ACCEPT branch: A on Yes.
    e.tap("A", hold=6, then=90)
    accept = snap(e)

    # DECLINE control: re-drive, press B (cancel) instead.
    drive_to_prompt(e)
    e.tap("B", hold=6, then=90)
    decline = snap(e)

    n = min(len(pre), len(accept), len(decline))
    print(f"[find] EWRAM bytes compared: {n}")

    # --- money: 5000 -> 4700 (delta 300) as u16 or u32 ---
    print("\n[money] offsets reading 5000 pre and 4700 post-accept:")
    found_money = []
    for width, rd in (("u16", u16), ("u32", u32)):
        step = 1
        for off in range(0, n - 4, step):
            if rd(pre, off) == 5000 and rd(accept, off) == 4700:
                addr = EWRAM_BASE + off
                dec = rd(decline, off)
                print(f"    {width} @ 0x{addr:08x}  pre=5000 accept=4700 decline={dec}")
                found_money.append((width, addr, dec))
    if not found_money:
        print("    (none - money may be stored elsewhere or encoded)")

    # --- acceptance record: changed on accept, NOT on decline (skip graphics scratch) ---
    print("\n[accept_record] bytes changed ONLY on accept (not on decline), off>=0x1000:")
    changed_accept_only = []
    for off in range(0x1000, n):
        if pre[off] != accept[off] and pre[off] == decline[off]:
            changed_accept_only.append(off)
    print(f"    total such bytes: {len(changed_accept_only)}")
    # Group into contiguous runs to spot struct fields (a counter is often 1-4 bytes).
    runs = []
    if changed_accept_only:
        s = p = changed_accept_only[0]
        for off in changed_accept_only[1:]:
            if off == p + 1:
                p = off
            else:
                runs.append((s, p))
                s = p = off
        runs.append((s, p))
    print(f"    contiguous runs: {len(runs)} (showing up to 40, with pre->accept bytes)")
    for s, ep in runs[:40]:
        addr = EWRAM_BASE + s
        pv = " ".join(f"{pre[o]:02x}" for o in range(s, ep + 1))
        av = " ".join(f"{accept[o]:02x}" for o in range(s, ep + 1))
        print(f"    0x{addr:08x}..{EWRAM_BASE+ep:08x} ({ep-s+1}B)  pre[{pv}] accept[{av}]")


if __name__ == "__main__":
    main()
