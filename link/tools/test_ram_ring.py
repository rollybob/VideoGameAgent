"""Validate the RamRing capture + ram_ring_diff analysis without a human.

Boots the solo ALttP core headless, runs frames, and part-way through pokes a
fake eighths-style HP drop into a KNOWN address (EWRAM 0x00C93, the confirmed
solo ALttP health byte). If the pipeline is wired correctly, ram_ring_diff.py
must rank exactly that address top. Tests the real read path, the ring layout,
the region split and the ranking -- everything except a human's reflexes.

  source ~/projects/VGA/link/env.sh
  "$LINK_PY" ~/projects/VGA/link/tools/test_ram_ring.py
"""
import os
import sys

import mgba.core
import mgba.image
import mgba.log
from mgba._pylib import ffi

mgba.log.silence()
HERE = os.path.dirname(os.path.abspath(__file__))
LINK = os.path.dirname(HERE)
VGA = os.path.dirname(LINK)
sys.path.insert(0, LINK)
from ram_ring import RamRing   # noqa: E402

ROM = os.environ.get("FOUR_SWORDS_ROM")
STATE = os.path.join(VGA, "train", "rl", "states", "alttp_ingame.state")
HP_ADDR = 0x00C93          # confirmed solo ALttP health, eighths
OUT = os.path.join(LINK, "sessions", "ramhits_test")


def main():
    core = mgba.core.load_path(ROM)
    img = mgba.image.Image(*core.desired_video_dimensions())
    core.set_video_buffer(img)
    core.reset()
    with open(STATE, "rb") as f:
        assert core.load_raw_state(f.read()), "failed to load seed state"

    ring = RamRing(core, every=2, keep=120, label="selftest")
    ew = core._native.memory.wram

    # Pin the byte to a constant, then drop it once -- a synthetic clean hit.
    for tick in range(1, 241):
        core.set_keys(raw=0)
        core.run_frame()
        buf = ffi.buffer(ew, 256 * 1024)
        buf[HP_ADDR] = bytes([24 if tick < 160 else 16])
        ring.record(tick)

    out = ring.dump(OUT, "selftest", note="synthetic 24->16 at EWRAM 0x00C93")
    print("dump: %s" % out)
    return out


if __name__ == "__main__":
    main()
