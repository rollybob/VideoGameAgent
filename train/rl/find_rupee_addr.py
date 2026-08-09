"""Find the ALttP rupee counter by write-testing every byte that holds the
value currently shown on the HUD.

WHY A WRITE-TEST: a single observed value ("I have 2 rupees") matches hundreds
of bytes, and Tim's three savestates all read 2, so they do not discriminate.
Rather than rank guesses, poke a distinctive value into each candidate and look
at the screen: the byte that redraws the HUD digits IS the counter. This is the
same visual-confirmation step that settled the Four Swords health address --
structural plausibility repeatedly produced false positives there, pixels did not.

Runs multiprocess so the load spreads across cores instead of pegging one.

  source ~/projects/VGA/link/env.sh
  "$LINK_PY" train/rl/find_rupee_addr.py train/rl/states/alttp_human-02.state \\
      --value 2 --others train/rl/states/alttp_human-00.state ...
"""
import argparse
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(VGA, "link"))

import mgba.core
import mgba.image
import mgba.log
from mgba._pylib import ffi

mgba.log.silence()
EWRAM_SIZE, IWRAM_SIZE = 256 * 1024, 32 * 1024
GBA_W, GBA_H = 240, 160
# top-left HUD block holding the rupee/bomb/arrow counters
HUD_X0, HUD_X1, HUD_Y0, HUD_Y1 = 0, 70, 0, 20
SETTLE, POST_WRITE = 12, 8
POKE = 231          # distinctive, and < 999 so it stays renderable as digits


def hud_bytes(img):
    raw = ffi.buffer(img.buffer, GBA_W * GBA_H * 4)
    rows = []
    for y in range(HUD_Y0, HUD_Y1):
        off = (y * GBA_W + HUD_X0) * 4
        rows.append(bytes(raw[off:off + (HUD_X1 - HUD_X0) * 4]))
    return b"".join(rows)


def frame_bytes(img):
    return bytes(ffi.buffer(img.buffer, GBA_W * GBA_H * 4))


class Probe:
    def __init__(self, rom, state):
        self.core = mgba.core.load_path(rom)
        self.img = mgba.image.Image(*self.core.desired_video_dimensions())
        self.core.set_video_buffer(self.img)
        self.core.reset()
        with open(state, "rb") as f:
            self.state = f.read()

    def run(self, region=None, addr=None, value=POKE):
        assert self.core.load_raw_state(self.state)
        for _ in range(SETTLE):
            self.core.set_keys(raw=0)
            self.core.run_frame()
        if addr is not None:
            ptr = (self.core._native.memory.wram if region == "EWRAM"
                   else self.core._native.memory.iwram)
            size = EWRAM_SIZE if region == "EWRAM" else IWRAM_SIZE
            ffi.buffer(ptr, size)[addr] = bytes([value])
        for _ in range(POST_WRITE):
            self.core.set_keys(raw=0)
            self.core.run_frame()
        return hud_bytes(self.img), frame_bytes(self.img)


_probe = None


def _init(rom, state):
    global _probe
    _probe = Probe(rom, state)


def _test(job):
    region, addr, base_hud, base_frame = job
    hud, frame = _probe.run(region, addr)
    return (region, addr, hud != base_hud, frame != base_frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", help="savestate to write-test against")
    ap.add_argument("--others", nargs="*", default=[],
                    help="more states showing the SAME value; a candidate must "
                         "hold the value in all of them")
    ap.add_argument("--value", type=int, required=True,
                    help="counter value currently shown on the HUD")
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # candidates: bytes holding the observed value in every supplied state
    mems = []
    for p in [args.state] + args.others:
        pr = Probe(args.rom, p)
        pr.run()
        mems.append((bytes(ffi.buffer(pr.core._native.memory.wram, EWRAM_SIZE)),
                     bytes(ffi.buffer(pr.core._native.memory.iwram, IWRAM_SIZE))))
    cands = []
    for region, idx, size in (("EWRAM", 0, EWRAM_SIZE), ("IWRAM", 1, IWRAM_SIZE)):
        for a in range(size):
            if all(m[idx][a] == args.value for m in mems):
                cands.append((region, a))
    print("candidates holding %d in all %d states: %d" % (
        args.value, len(mems), len(cands)))

    base = Probe(args.rom, args.state)
    base_hud, base_frame = base.run()
    jobs = [(r, a, base_hud, base_frame) for r, a in cands]

    hits = []
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.rom, args.state)) as pool:
        for region, addr, hud_changed, frame_changed in pool.imap_unordered(
                _test, jobs, chunksize=4):
            if hud_changed:
                hits.append((region, addr, frame_changed))

    print("\naddresses whose write REDREW the HUD: %d" % len(hits))
    clean = [h for h in hits if not h[2]]
    print("  of which changed ONLY the HUD (nothing else on screen): %d" % len(clean))
    for region, addr, frame_changed in sorted(clean or hits):
        print("  %-6s 0x%05X   %s" % (
            region, addr, "HUD only" if not frame_changed else "HUD + rest of frame"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
