"""Task 09 A0: capture per-tick RAM during a deterministic flight-tape replay.

Replays a FlightRecorder dump (link/sessions/flight/<dir>) through the link
engine exactly like link_replay.py, but records core 0's full EWRAM (256KB)
and IWRAM (32KB) EVERY TICK into raw uint8 memmaps, plus a PNG every
--png-every ticks for ground-truth reading of the HUD. In lockstep
multiplayer every core simulates all 4 Links, so core 0's RAM contains the
whole world state (verified downstream by the per-player analysis).

Outputs into scratch/ram/<tape-name>/:
  ewram.u8   raw (ticks, 262144) uint8   iwram.u8   raw (ticks, 32768) uint8
  ticks.json {"ticks": N}                fXXXXX.png periodic frames

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/ram_capture.py /vga/link/sessions/flight/<dir>
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

ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                   "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
EWRAM_SIZE = 256 * 1024
IWRAM_SIZE = 32 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--png-every", type=int, default=30)
    args = ap.parse_args()
    mgba.log.silence()

    with open(os.path.join(args.dump, "inputs.json")) as f:
        masks = json.load(f)["masks"]
    states = [os.path.join(args.dump, "p%d_prefail.state" % i) for i in range(4)]
    out = os.path.join(HERE, "scratch", "ram", os.path.basename(os.path.normpath(args.dump)))
    os.makedirs(out, exist_ok=True)

    n = len(masks)
    ew = np.memmap(os.path.join(out, "ewram.u8"), dtype=np.uint8, mode="w+",
                   shape=(n, EWRAM_SIZE))
    iw = np.memmap(os.path.join(out, "iwram.u8"), dtype=np.uint8, mode="w+",
                   shape=(n, IWRAM_SIZE))

    sess = LinkSession(ROM, n=4, state_path=states, trace=False)
    for nd in sess.nodes:
        if nd.index > 0 and nd.irq_flagged:
            nd.irq_pending = True
            nd.mltsend_seen = False

    c0 = sess.nodes[0]
    ew_ptr = c0.core._native.memory.wram
    iw_ptr = c0.core._native.memory.iwram
    for k, m in enumerate(masks):
        for i in range(4):
            sess.nodes[i].core.set_keys(raw=m[i])
        sess.tick()
        ew[k] = np.frombuffer(ffi.buffer(ew_ptr, EWRAM_SIZE), dtype=np.uint8)
        iw[k] = np.frombuffer(ffi.buffer(iw_ptr, IWRAM_SIZE), dtype=np.uint8)
        if k % args.png_every == 0:
            with open(os.path.join(out, "f%05d.png" % k), "wb") as f:
                c0.image.save_png(f)
    ew.flush(); iw.flush()
    with open(os.path.join(out, "ticks.json"), "w") as f:
        json.dump({"ticks": n, "transfers": sess.transfers}, f)
    print("captured %d ticks -> %s (transfers=%d)" % (n, out, sess.transfers))
    sess.shutdown()


if __name__ == "__main__":
    main()
