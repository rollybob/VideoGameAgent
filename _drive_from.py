"""Temp: LOAD a savestate, then drive with a plan ("n:mask,...") and snapshot each
segment; optionally save a new state at the end. Lets me step the file-select ->
game-selection -> title chain from alttp_start.state. Runs in thor-rl:cu130.

GBA key bits (mgba): A=1 B=2 Select=4 Start=8 Right=16 Left=32 Up=64 Down=128.
"""
import sys
import numpy as np
import mgba.core, mgba.image, mgba.log
from mgba._pylib import ffi

mgba.log.silence()
ROM = "/work/Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"
W, H = 240, 160

load_state = sys.argv[1]
plan = sys.argv[2]
save_path = sys.argv[3] if len(sys.argv) > 3 else None

core = mgba.core.load_path(ROM)
w, h = core.desired_video_dimensions()
img = mgba.image.Image(w, h)
core.set_video_buffer(img)
core.reset()
with open(load_state, "rb") as f:
    assert core.load_raw_state(f.read()), "load failed"
core.set_keys(raw=0)
core.run_frame()


def frame():
    return np.frombuffer(ffi.buffer(img.buffer, W * H * 4), np.uint8).reshape(H, W, 4)[:, :, :3].copy()


for i, seg in enumerate(plan.split(",")):
    n, mask = seg.split(":")
    n, mask = int(n), int(mask)
    core.set_keys(raw=mask)
    for _ in range(n):
        core.run_frame()
    core.set_keys(raw=0)
    core.run_frame()
    np.save("/work/_drv_%d.npy" % i, frame())
    print("seg %d: %d frames mask=%d" % (i, n, mask), flush=True)

if save_path:
    data = bytes(ffi.buffer(core.save_raw_state()))
    with open(save_path, "wb") as f:
        f.write(data)
    print("SAVED %s (%d bytes)" % (save_path, len(data)), flush=True)
