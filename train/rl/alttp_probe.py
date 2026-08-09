"""Task 09 A0: solo-ALttP ground-truth generator + RAM miner.

Demonstrates the RAM-mapping-ease argument for Track A env choice: with a
single controllable agent we can MANUFACTURE reward events on demand
(unlike the 4P link tapes, whose rooms contain no hostile enemies / organic
pickups). Drives Link from states/alttp_ingame.state through a schedule,
captures per-frame EWRAM+IWRAM, and reports addresses whose value moves.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/alttp_probe.py --schedule explore
"""
import argparse
import json
import os

import numpy as np

import mgba.core
import mgba.gba
import mgba.image
import mgba.log
from mgba._pylib import ffi

mgba.log.silence()
HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                   "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
STATE = os.path.join(HERE, "states", "alttp_ingame.state")
K = mgba.gba.GBA
# set_keys(raw=...) wants a BITMASK (1<<index), not the raw key index -- see
# mgba core._keys_to_int. Passing unshifted indices pressed the wrong buttons
# (e.g. DOWN idx 7 -> A+B+SELECT), so schedules never actually moved Link.
A, B, RIGHT, LEFT, UP, DOWN = (1 << K.KEY_A, 1 << K.KEY_B, 1 << K.KEY_RIGHT,
                               1 << K.KEY_LEFT, 1 << K.KEY_UP, 1 << K.KEY_DOWN)

SCHEDULES = {
    "explore": [
        ("idle", 0, 30),
        ("down", DOWN, 120), ("right", RIGHT, 120), ("up", UP, 120),
        ("left", LEFT, 120), ("down2", DOWN, 120), ("right2", RIGHT, 120),
        ("slash", B, 6), ("idle2", 0, 60),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="explore")
    ap.add_argument("--png-every", type=int, default=15)
    args = ap.parse_args()
    sched = SCHEDULES[args.schedule]
    out = os.path.join(HERE, "scratch", "probe", "alttp_" + args.schedule)
    os.makedirs(out, exist_ok=True)

    core = mgba.core.load_path(ROM)
    w, h = core.desired_video_dimensions()
    img = mgba.image.Image(w, h)
    core.set_video_buffer(img)
    core.reset()
    with open(STATE, "rb") as f:
        assert core.load_raw_state(f.read())

    n = sum(t for _, _, t in sched)
    ew = np.memmap(os.path.join(out, "ewram.u8"), dtype=np.uint8, mode="w+", shape=(n, 262144))
    iw = np.memmap(os.path.join(out, "iwram.u8"), dtype=np.uint8, mode="w+", shape=(n, 32768))
    ew_ptr = core._native.memory.wram
    iw_ptr = core._native.memory.iwram

    timeline, k = [], 0
    for label, mask, ticks in sched:
        timeline.append({"label": label, "start": k, "end": k + ticks})
        for _ in range(ticks):
            core.set_keys(raw=mask)
            core.run_frame()
            ew[k] = np.frombuffer(ffi.buffer(ew_ptr, 262144), dtype=np.uint8)
            iw[k] = np.frombuffer(ffi.buffer(iw_ptr, 32768), dtype=np.uint8)
            if k % args.png_every == 0:
                with open(os.path.join(out, "f%05d.png" % k), "wb") as f:
                    img.save_png(f)
            k += 1
    ew.flush(); iw.flush()
    with open(os.path.join(out, "ticks.json"), "w") as f:
        json.dump({"ticks": n}, f)
    with open(os.path.join(out, "schedule.json"), "w") as f:
        json.dump(timeline, f, indent=1)
    print("alttp probe '%s': %d frames -> %s" % (args.schedule, n, out))


if __name__ == "__main__":
    main()
