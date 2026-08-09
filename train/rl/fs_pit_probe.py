"""Task 09 A0: scripted single-player wander in Four Swords (Chambers of Insight).

Resumes p*_coop.state. Player 0 wanders (same rotation as the wander bots that
Tim saw fall into pits). Players 1-3 are stationary (no input). Captures BOTH
EWRAM and IWRAM from core 0 every tick; saves PNGs every --snap ticks.

Prior attempt with just DOWN failed: player 0 was blocked by the stone ledge.
Wander input cycles through all 8 directions and will eventually navigate the
room and fall into a pit.

Also directly checks the IWRAM stride-0x80 candidates from FINDINGS
(0x01068/0x010E8/0x01168/0x011E8) at the end so they aren't missed.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga \\
      thor-torch:cu130 python3 /vga/train/rl/fs_pit_probe.py [--ticks 2000]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(VGA, "link"))

import mgba.log
from mgba._pylib import ffi
from link_engine import LinkSession

mgba.log.silence()

ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                   "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
CKPT = os.path.join(VGA, "link", "sessions", "checkpoints")
EWRAM_SIZE = 256 * 1024
IWRAM_SIZE = 32 * 1024

# Same wander cycle the live bots use (bench_link4.py WANDER).
# GBA key bits: Right=16 Left=32 Up=64 Down=128 A=1
WANDER = [16, 32, 64, 128, 16 | 1, 32 | 1, 64 | 1, 128 | 1, 0]
WANDER_PERIOD = 30  # ticks per direction

# IWRAM candidates from FINDINGS (stride-0x80, plausible per-player health)
IWRAM_CANDIDATES = [0x01068, 0x010E8, 0x01168, 0x011E8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--snap", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(HERE, "scratch", "probe", "fs_pit"))
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM", ROM))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    states = [os.path.join(CKPT, "p%d_coop.state" % i) for i in range(4)]
    sess = LinkSession(args.rom, n=4, state_path=states, trace=False)

    ew_ptr = sess.nodes[0].core._native.memory.wram
    iw_ptr = sess.nodes[0].core._native.memory.iwram
    img = sess.nodes[0].image

    ew = np.memmap(os.path.join(args.out, "ewram.u8"), dtype=np.uint8,
                   mode="w+", shape=(args.ticks, EWRAM_SIZE))
    iw = np.memmap(os.path.join(args.out, "iwram.u8"), dtype=np.uint8,
                   mode="w+", shape=(args.ticks, IWRAM_SIZE))

    print("Running %d ticks (wander on p0, p1-3 stationary) -> %s"
          % (args.ticks, args.out))

    for tick in range(args.ticks):
        # Player 0: wander; players 1-3: stationary
        w_mask = WANDER[(tick // WANDER_PERIOD) % len(WANDER)]
        sess.nodes[0].core.set_keys(raw=w_mask)
        for nd in sess.nodes[1:]:
            nd.core.set_keys(raw=0)
        sess.tick()

        ew[tick] = np.frombuffer(ffi.buffer(ew_ptr, EWRAM_SIZE), dtype=np.uint8)
        iw[tick] = np.frombuffer(ffi.buffer(iw_ptr, IWRAM_SIZE), dtype=np.uint8)

        if tick % args.snap == 0:
            png_path = os.path.join(args.out, "f%05d.png" % tick)
            with open(png_path, "wb") as f:
                img.save_png(f)

    ew.flush()
    iw.flush()
    with open(os.path.join(args.out, "ticks.json"), "w") as f:
        json.dump({"ticks": args.ticks, "input": "wander_p0_only"}, f, indent=2)

    # Print the IWRAM candidates from FINDINGS directly
    iw_arr = np.asarray(iw)
    print("\nIWRAM stride-0x80 candidates (from FINDINGS):")
    for addr in IWRAM_CANDIDATES:
        col = iw_arr[:, addr].astype(np.int32)
        vals = np.unique(col)
        drops = np.nonzero(np.diff(col) < 0)[0] + 1
        print("  iwram 0x%05X  range=%d..%d  unique=%s  drops@ticks=%s"
              % (addr, col.min(), col.max(), list(vals[:8]), list(drops[:5])))

    print("\nDone. Run: python3 find_pit_health.py %s" % args.out)


if __name__ == "__main__":
    main()
