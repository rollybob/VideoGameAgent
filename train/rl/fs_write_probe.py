"""Task 09 A0: targeted write-probe to verify FS per-player health address.

Loads p*_coop.state, writes a test value to a candidate EWRAM address,
runs 60 frames, and saves a screenshot. If writing 0 kills the player (death
animation plays) or a mid-range value shows fewer hearts in the HUD, the
address is confirmed as health.

Run for each candidate in turn:
  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga \\
      thor-torch:cu130 python3 /vga/train/rl/fs_write_probe.py \\
      --ewram 0x30C58 --val 0

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga \\
      thor-torch:cu130 python3 /vga/train/rl/fs_write_probe.py \\
      --ewram 0x30C58 --val 8
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
OUT = os.path.join(HERE, "scratch", "probe", "write")


def read_ewram_byte(core, offset):
    return int(ffi.cast("uint8_t *", core._native.memory.wram)[offset])


def write_ewram_byte(core, offset, val):
    ffi.cast("uint8_t *", core._native.memory.wram)[offset] = val & 0xFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ewram", type=lambda x: int(x, 0), default=0x30C58,
                    help="EWRAM offset to test (hex ok)")
    ap.add_argument("--val", type=int, default=0,
                    help="value to write (0=kill, 8=1 heart, 16=2 hearts, etc.)")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM", ROM))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    states = [os.path.join(CKPT, "p%d_coop.state" % i) for i in range(4)]
    sess = LinkSession(args.rom, n=4, state_path=states, trace=False)
    core0 = sess.nodes[0].core

    before = read_ewram_byte(core0, args.ewram)
    print("ewram 0x%05X before = %d" % (args.ewram, before))

    # Write the test value to all 4 cores (lockstep -- all identical)
    for nd in sess.nodes:
        write_ewram_byte(nd.core, args.ewram, args.val)

    after = read_ewram_byte(core0, args.ewram)
    print("ewram 0x%05X after write = %d (wrote %d)" % (args.ewram, after, args.val))

    # Run frames; save a PNG before and after the write
    snap_name = "wt_e%05X_v%d" % (args.ewram, args.val)
    for f in range(args.frames):
        for nd in sess.nodes:
            nd.core.set_keys(raw=0)
        sess.tick()
        if f in (0, 9, 29, 59, args.frames - 1):
            path = os.path.join(OUT, "%s_f%03d.png" % (snap_name, f))
            with open(path, "wb") as fp:
                sess.nodes[0].image.save_png(fp)

    # Read back to see if the game restored it
    readback = read_ewram_byte(core0, args.ewram)
    print("ewram 0x%05X readback after %d frames = %d" % (args.ewram, args.frames, readback))
    if readback == args.val:
        print("  -> Value HELD: game did not overwrite it (display-only if HUD unchanged)")
    elif readback != before:
        print("  -> Value changed to %d: game PROCESSED the write (health-like behavior)" % readback)
    else:
        print("  -> Value RESTORED to %d: game overwrote with original (display copy?)" % before)

    print("Screenshots in %s/%s_f*.png" % (OUT, snap_name))


if __name__ == "__main__":
    main()
