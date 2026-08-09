"""Temp: drive the ALttP + Four Swords compilation ROM headless to the ALttP title
screen. Plan-driven ("n:mask,n:mask,...") so the input sequence can be iterated
without rewriting. Optionally saves a raw state at the end. Runs in thor-rl:cu130.

GBA key bit indices (mgba): A=0 B=1 Select=2 Start=3 Right=4 Left=5 Up=6 Down=7 R=8 L=9
so A=1, B=2, Select=4, Start=8, Right=16, Left=32, Up=64, Down=128.
"""
import sys
import numpy as np
import mgba.core, mgba.image, mgba.log
from mgba._pylib import ffi

mgba.log.silence()
ROM = "/work/Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"
W, H = 240, 160

core = mgba.core.load_path(ROM)
w, h = core.desired_video_dimensions()
img = mgba.image.Image(w, h)
core.set_video_buffer(img)
core.reset()


def frame():
    return np.frombuffer(ffi.buffer(img.buffer, W * H * 4), np.uint8).reshape(H, W, 4)[:, :, :3].copy()


plan = sys.argv[1]                                  # "n:mask,n:mask,..."
save_path = sys.argv[2] if len(sys.argv) > 2 else None
for i, seg in enumerate(plan.split(",")):
    n, mask = seg.split(":")
    n, mask = int(n), int(mask)
    core.set_keys(raw=mask)
    for _ in range(n):
        core.run_frame()
    core.set_keys(raw=0)
    core.run_frame()                                # settle + refresh video buffer
    np.save("/work/_drv_%d.npy" % i, frame())
    print("seg %d: %d frames mask=%d" % (i, n, mask), flush=True)

if save_path:
    data = bytes(ffi.buffer(core.save_raw_state()))
    with open(save_path, "wb") as f:
        f.write(data)
    print("SAVED STATE %s (%d bytes)" % (save_path, len(data)), flush=True)
