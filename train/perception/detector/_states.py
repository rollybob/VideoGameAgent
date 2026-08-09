"""_states.py -- verify candidate new detector rooms before training. Boot each F8 savestate
(alttp_human-07..14), render the frame, and report Link screen pos, room cell, and on-screen
enemies (0x03846 slots, HP>0). Flags transition/junk states (Link off-screen). Montage -> _states.png.
thor-rl:cu130, read-only (boots states in a throwaway core; does not touch the live game)."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageDraw
import mgba.core, mgba.image, mgba.log
from mgba._pylib import ffi
mgba.log.silence()
ROOT = "/work"; STATES = ROOT + "/train/rl/states"; DET = ROOT + "/train/perception/detector"
ROM = ROOT + "/Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"
CAMX, CAMY = 0x02B82, 0x02B86


def u16(b, a):
    return b[a] | (b[a + 1] << 8)


def inspect(path):
    core = mgba.core.load_path(ROM); w, h = core.desired_video_dimensions()
    img = mgba.image.Image(w, h); core.set_video_buffer(img); core.reset()
    with open(path, "rb") as f:
        core.load_raw_state(f.read())
    core.run_frame()
    arr = np.frombuffer(bytes(ffi.buffer(img.buffer, w * h * 4)), np.uint8).reshape(h, w, 4)[..., :3].copy()
    iw = bytes(ffi.buffer(ffi.cast("uint8_t *", core._native.memory.iwram), 32 * 1024))
    cx, cy = u16(iw, CAMX), u16(iw, CAMY)
    lx, ly = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
    room = (u16(iw, 0x038F4) >> 9, u16(iw, 0x038F0) >> 9)
    en = []
    for i in range(16):
        ex, ey = u16(iw, 0x03846 + 4 * i), u16(iw, 0x03848 + 4 * i)
        if (ex, ey) == (0, 0):
            continue
        sx, sy = ex - cx, ey - cy
        if 0 <= sx < 240 and 0 <= sy < 160 and iw[0x03250 + i] > 0:
            en.append((sx, sy))
    return arr, (lx, ly), room, en


tiles = []
for rm in range(7, 15):
    p = os.path.join(STATES, f"alttp_human-{rm:02d}.state")
    if not os.path.exists(p):
        print(f"-{rm:02d}: MISSING", flush=True); continue
    arr, (lx, ly), room, en = inspect(p)
    onscreen = 0 <= lx < 240 and 0 <= ly < 160
    verdict = "OK" if onscreen else "JUNK(link off-screen)"
    print(f"-{rm:02d}: link=({lx},{ly}) room={room} enemies_onscreen={len(en)} {en[:5]}  [{verdict}]", flush=True)
    im = Image.fromarray(arr); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 26, 10], fill=(0, 0, 0)); dr.text((1, 1), str(rm), fill=(255, 255, 0))
    tiles.append(np.array(im.resize((240 * 3, 160 * 3), Image.NEAREST), np.uint8))

if tiles:
    W = 4; rows = []
    for r in range(0, len(tiles), W):
        row = tiles[r:r + W]
        while len(row) < W:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    Image.fromarray(np.vstack(rows)).save(os.path.join(DET, "_states.png"))
    print("wrote _states.png", flush=True)
