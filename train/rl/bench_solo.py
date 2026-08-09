"""Task 09 A0: throughput probe for env Option 1 -- solo ALttP, single core.

Loads states/alttp_ingame.state (built by solo_boot.py), first verifies the
state is actually controllable (holding LEFT must change the framebuffer
within a second -- guards against banking mid-dialog), then measures raw
run_frame() throughput, with and without a per-frame EWRAM+IWRAM snapshot
(the reward-oracle overhead).

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/bench_solo.py
"""
import argparse
import os
import time

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
EWRAM_SIZE = 256 * 1024
IWRAM_SIZE = 32 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5000)
    args = ap.parse_args()

    core = mgba.core.load_path(ROM)
    w, h = core.desired_video_dimensions()
    img = mgba.image.Image(w, h)
    core.set_video_buffer(img)
    core.reset()
    with open(STATE, "rb") as f:
        assert core.load_raw_state(f.read()), "state load failed"

    # controllability probe: framebuffer must react to input
    core.set_keys(raw=0)
    for _ in range(10):
        core.run_frame()
    before = bytes(ffi.buffer(img.buffer))
    core.set_keys(K.KEY_LEFT)
    for _ in range(60):
        core.run_frame()
    core.set_keys(raw=0)
    after = bytes(ffi.buffer(img.buffer))
    controllable = before != after
    print("controllability probe (LEFT 60f changes screen):", controllable)

    ew_ptr = core._native.memory.wram
    iw_ptr = core._native.memory.iwram
    # wander inputs during timing so we measure gameplay, not idling
    wander = [K.KEY_LEFT, K.KEY_RIGHT, K.KEY_UP, K.KEY_DOWN, 0]
    for label, snap in (("plain", False), ("with EWRAM+IWRAM snapshot/frame", True)):
        t0 = time.perf_counter()
        c0 = time.process_time()
        for k in range(args.frames):
            core.set_keys(wander[(k // 30) % len(wander)])
            core.run_frame()
            if snap:
                _ = bytes(ffi.buffer(ew_ptr, EWRAM_SIZE))
                _ = bytes(ffi.buffer(iw_ptr, IWRAM_SIZE))
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        fps = args.frames / wall
        print("[%s] %d frames in %.2fs wall (%.2f cores): %.1f fps"
              % (label, args.frames, wall, cpu / wall, fps))
        print("  per hour per core: %.2fM frames" % (fps * 3600 / 1e6))
    with open(os.path.join(HERE, "scratch", "solo_bench_last.png"), "wb") as f:
        img.save_png(f)


if __name__ == "__main__":
    main()
