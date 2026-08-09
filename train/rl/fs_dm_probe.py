"""Task 09 A0: Death Mountain probe to find FS per-player health.

Starts from p*_stage_select.state, navigates to Death Mountain (press RIGHT
to move cursor, then A to confirm), waits for the room to load, then drives
player 0 into the surrounding lava while capturing EWRAM+IWRAM per tick.

Death Mountain is confirmed hazardous (stage_dm/ screenshots show lava on all
4 sides from the starting position). One step left/right/up/down from center
into the lava wall should trigger damage within ~60 ticks of entering the room.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga \\
      thor-torch:cu130 python3 /vga/train/rl/fs_dm_probe.py
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

# Key bits
KEY_A = 1
KEY_RIGHT = 16
KEY_LEFT = 32

# Input schedule for player 0 (master player selects the stage).
# Goal: select Death Mountain by pressing RIGHT enough times then A.
# Chambers of Insight is default (stage 1). DM is stage 3 in FS.
# Tick 0-20: idle (let stage select settle)
# Tick 21-24: press RIGHT (move to next stage)
# Tick 31-34: press RIGHT again (move to DM = stage 3)
# Tick 41-44: press A (confirm)
# Tick 45-599: idle (wait for stage to load and players to enter room)
# Tick 600+: walk LEFT into lava (lava is on left side from center start)
SCHEDULE_P0 = [
    ("idle_pre",  0,        20),
    ("right1",    KEY_RIGHT, 4),   # ONE right press: CoI -> Death Mountain
    ("gap1",      0,        10),
    ("confirm_a", KEY_A,     4),   # confirm DM selection
    ("wait_load", 0,       570),   # wait for room to load (~500 ticks)
    ("left",      KEY_LEFT, 300),  # walk left into lava (lava on all sides in DM)
    ("idle_post", 0,        200),
]

# IWRAM candidates from FINDINGS
IWRAM_CANDIDATES = [0x01068, 0x010E8, 0x01168, 0x011E8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=1100)
    ap.add_argument("--snap", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(HERE, "scratch", "probe", "fs_dm"))
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM", ROM))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    states = [os.path.join(CKPT, "p%d_stage_select.state" % i) for i in range(4)]
    sess = LinkSession(args.rom, n=4, state_path=states, trace=False)

    ew_ptr = sess.nodes[0].core._native.memory.wram
    iw_ptr = sess.nodes[0].core._native.memory.iwram
    img = sess.nodes[0].image

    ew = np.memmap(os.path.join(args.out, "ewram.u8"), dtype=np.uint8,
                   mode="w+", shape=(args.ticks, EWRAM_SIZE))
    iw = np.memmap(os.path.join(args.out, "iwram.u8"), dtype=np.uint8,
                   mode="w+", shape=(args.ticks, IWRAM_SIZE))

    p0_inputs = []
    for _label, mask, n in SCHEDULE_P0:
        p0_inputs.extend([mask] * n)
    if len(p0_inputs) < args.ticks:
        p0_inputs += [0] * (args.ticks - len(p0_inputs))
    p0_inputs = p0_inputs[:args.ticks]

    # All other players: no input (let p0 master drive stage selection)
    print("Running %d ticks, p0 selects DM then walks left -> %s"
          % (args.ticks, args.out))

    for tick in range(args.ticks):
        sess.nodes[0].core.set_keys(raw=p0_inputs[tick])
        for nd in sess.nodes[1:]:
            nd.core.set_keys(raw=0)
        sess.tick()

        ew[tick] = np.frombuffer(ffi.buffer(ew_ptr, EWRAM_SIZE), dtype=np.uint8)
        iw[tick] = np.frombuffer(ffi.buffer(iw_ptr, IWRAM_SIZE), dtype=np.uint8)

        if tick % args.snap == 0:
            with open(os.path.join(args.out, "f%05d.png" % tick), "wb") as f:
                img.save_png(f)

    ew.flush()
    iw.flush()
    with open(os.path.join(args.out, "ticks.json"), "w") as f:
        json.dump({"ticks": args.ticks, "schedule": SCHEDULE_P0}, f, indent=2)

    # Print IWRAM candidate values
    iw_arr = np.asarray(iw)
    print("\nIWRAM candidates:")
    for addr in IWRAM_CANDIDATES:
        col = iw_arr[:, addr].astype(np.int32)
        drops = list(map(int, (np.nonzero(np.diff(col) < 0)[0] + 1)[:5]))
        print("  iwram 0x%05X  start=%d  range=%d..%d  drops@%s"
              % (addr, int(col[0]), int(col.min()), int(col.max()), drops))

    print("\nDone. Run: python3 find_pit_health.py %s" % args.out)


if __name__ == "__main__":
    main()
