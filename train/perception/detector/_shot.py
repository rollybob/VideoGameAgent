"""_shot.py <room> -- boot alttp_human-<room>.state, print Link screen pos + ALL hp>0 enemies
(on/off screen), and save a 4x frame _shot_<room>.png for Tim. thor-rl:cu130, read-only."""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import mgba.core, mgba.image, mgba.log
from mgba._pylib import ffi
mgba.log.silence()
ROOT = "/work"; DET = ROOT + "/train/perception/detector"
ROM = ROOT + "/Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"
CAMX, CAMY = 0x02B82, 0x02B86


def u16(b, a):
    return b[a] | (b[a + 1] << 8)


rm = int(sys.argv[1])
core = mgba.core.load_path(ROM); w, h = core.desired_video_dimensions()
img = mgba.image.Image(w, h); core.set_video_buffer(img); core.reset()
with open(f"{ROOT}/train/rl/states/alttp_human-{rm:02d}.state", "rb") as f:
    core.load_raw_state(f.read())
core.run_frame()
arr = np.frombuffer(bytes(ffi.buffer(img.buffer, w * h * 4)), np.uint8).reshape(h, w, 4)[..., :3].copy()
iw = bytes(ffi.buffer(ffi.cast("uint8_t *", core._native.memory.iwram), 32 * 1024))
cx, cy = u16(iw, CAMX), u16(iw, CAMY)
lx, ly = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
en = []
for i in range(16):
    ex, ey = u16(iw, 0x03846 + 4 * i), u16(iw, 0x03848 + 4 * i)
    if (ex, ey) == (0, 0):
        continue
    if iw[0x03250 + i] > 0:
        sx, sy = ex - cx, ey - cy
        en.append((sx, sy, "ON" if 0 <= sx < 240 and 0 <= sy < 160 else "OFF"))
print(f"room{rm}: link_screen=({lx},{ly})  hp>0 enemies={en}", flush=True)
Image.fromarray(arr).resize((w * 4, h * 4), Image.NEAREST).save(f"{DET}/_shot_{rm:02d}.png")
print("wrote _shot", flush=True)
