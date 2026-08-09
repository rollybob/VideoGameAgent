"""Task 09 A0: capture per-tick RAM during a solo ALttP flight-tape replay.

Solo counterpart of ram_capture.py. Replays a solo_recorder.py dump
(link/sessions/flight_solo/<dir>: prefail.state + inputs.json) through one
mGBA core exactly as it was recorded -- set_keys(raw=mask); run_frame() -- and
records the core's full EWRAM (256KB) and IWRAM (32KB) EVERY TICK into raw
uint8 memmaps, plus a PNG every --png-every ticks for HUD ground truth. The
replay is deterministic (same convention proven byte-identical in A0), so the
captured streams reproduce the combat events for hearts/kill/death mining.

Outputs into scratch/ram/<dump-name>/:
  ewram.u8  raw (ticks, 262144) uint8   iwram.u8  raw (ticks, 32768) uint8
  ticks.json {"ticks": N}               fXXXXX.png periodic frames

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/solo_ram_capture.py \
      /vga/link/sessions/flight_solo/<dir>
"""
import argparse
import json
import os

import numpy as np

import mgba.core
import mgba.image
import mgba.log
from mgba._pylib import ffi

mgba.log.silence()
HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                   "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
EWRAM_SIZE = 256 * 1024
IWRAM_SIZE = 32 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="solo_recorder flight dump dir")
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM", ROM))
    ap.add_argument("--png-every", type=int, default=30)
    args = ap.parse_args()

    with open(os.path.join(args.dump, "inputs.json")) as f:
        masks = json.load(f)["masks"]
    state_path = os.path.join(args.dump, "prefail.state")
    name = os.path.basename(os.path.normpath(args.dump))
    out = os.path.join(HERE, "scratch", "ram", name)
    os.makedirs(out, exist_ok=True)

    core = mgba.core.load_path(args.rom)
    img = mgba.image.Image(*core.desired_video_dimensions())
    core.set_video_buffer(img)
    core.reset()
    with open(state_path, "rb") as f:
        assert core.load_raw_state(f.read()), "failed to load prefail.state"
    ew_ptr = core._native.memory.wram
    iw_ptr = core._native.memory.iwram

    n = len(masks)
    ew = np.memmap(os.path.join(out, "ewram.u8"), dtype=np.uint8, mode="w+",
                   shape=(n, EWRAM_SIZE))
    iw = np.memmap(os.path.join(out, "iwram.u8"), dtype=np.uint8, mode="w+",
                   shape=(n, IWRAM_SIZE))

    for k, mask in enumerate(masks):
        core.set_keys(raw=int(mask))
        core.run_frame()
        ew[k] = np.frombuffer(ffi.buffer(ew_ptr, EWRAM_SIZE), dtype=np.uint8)
        iw[k] = np.frombuffer(ffi.buffer(iw_ptr, IWRAM_SIZE), dtype=np.uint8)
        if k % args.png_every == 0:
            with open(os.path.join(out, "f%05d.png" % k), "wb") as f:
                img.save_png(f)
    ew.flush()
    iw.flush()
    with open(os.path.join(out, "ticks.json"), "w") as f:
        json.dump({"ticks": n}, f)
    print("solo ram capture: %d ticks -> %s" % (n, out))


if __name__ == "__main__":
    main()
