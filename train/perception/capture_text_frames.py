"""Capture ALttP intro text frames (story crawl + early dialog) for the reading tier.
Boots alttp_start_normal.state (title), presses Start -> loads the slot-1 save -> the
opening story crawl plays; snapshot frames across it. Runs in thor-rl:cu130 (mgba+PIL).
mgba buffer is BGR -> flip to RGB for the PNG.
"""
import sys
import os

sys.path.insert(0, "/work/train/rl")
import numpy as np
from PIL import Image
import mgba.core, mgba.image, mgba.log
from mgba._pylib import ffi

mgba.log.silence()
ROM = "/work/Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"
W, H = 240, 160
OUT = "/work/train/perception/text_frames"
os.makedirs(OUT, exist_ok=True)

core = mgba.core.load_path(ROM)
w, h = core.desired_video_dimensions()
img = mgba.image.Image(w, h)
core.set_video_buffer(img)
core.reset()
with open("/work/train/rl/states/alttp_start_normal.state", "rb") as f:
    assert core.load_raw_state(f.read()), "load failed"
core.set_keys(raw=0)
core.run_frame()


def run(n, mask=0):
    core.set_keys(raw=mask)
    for _ in range(n):
        core.run_frame()
    core.set_keys(raw=0)
    core.run_frame()


def frame():
    return np.frombuffer(ffi.buffer(img.buffer, W * H * 4), np.uint8).reshape(H, W, 4)[:, :, :3]


# title -> Start -> file-select settle -> A (load slot 1) -> story crawl begins
run(6, 8)
run(150)
run(6, 1)

n = 0
for i in range(40):
    run(70)            # advance ~1.2s of the crawl/intro
    Image.fromarray(frame()[:, :, ::-1]).save(os.path.join(OUT, "t%03d.png" % n))
    n += 1
print("captured %d frames to %s" % (n, OUT), flush=True)
