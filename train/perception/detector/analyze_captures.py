"""analyze_captures.py -- mine Tim's 2026-08-08 pot/bush/etc item-drop captures.

DECISIVE QUESTION: do pot/bush/bomb/fairy/etc drops populate the SAME sprite-slot table as
enemy key-drops (pos 0x03846+4i, HP 0x03250+i, distinguished from enemies by HP==0), just with
a different TYPE byte (0x03160+i)? Or a different table entirely (item visible in mark.png but no
HP==0 slot entry at its screen pos)?

For each F8 state (alttp_human-07+) and each F11 ring dump (last snapshot's IWRAM), scan slots
0..15, list ON-SCREEN non-Link sprites with (slot, screen xy, HP, type). Render a montage
(green=Link, red=HP>0 enemy, blue=HP==0 item) + print a per-capture table + the type-byte palette.
Read-only; boots states/reads rings, never touches the live recorder. Run in thor-rl:cu130.
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import mgba.core, mgba.image, mgba.log
from mgba._pylib import ffi
mgba.log.silence()

ROOT = "/work"; DET = os.path.join(ROOT, "train/perception/detector")
STATES = os.path.join(ROOT, "train/rl/states")
RINGS = os.path.join(ROOT, "link/sessions/ramhits_solo")
ROM = os.path.join(ROOT, "Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
CAM_X, CAM_Y = 0x02B82, 0x02B86
IWRAM = 32 * 1024; SNAP = (256 + 32) * 1024


def u16(b, a):
    return b[a] | (b[a + 1] << 8)


def scan(iw):
    """Return (link_screen, [(slot, sx, sy, hp, type), ...]) for on-screen sprite slots."""
    cx, cy = u16(iw, CAM_X), u16(iw, CAM_Y)
    lx, ly = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
    ents = []
    for i in range(16):
        ex, ey = u16(iw, 0x03846 + 4 * i), u16(iw, 0x03848 + 4 * i)
        if (ex, ey) == (0, 0):
            continue
        sx, sy = ex - cx, ey - cy
        if 0 <= sx < 240 and 0 <= sy < 160:
            ents.append((i, sx, sy, int(iw[0x03250 + i]), int(iw[0x03160 + i])))
    return (lx, ly), ents


def frame_from_state(path):
    core = mgba.core.load_path(ROM)
    w, h = core.desired_video_dimensions()
    img = mgba.image.Image(w, h)
    core.set_video_buffer(img); core.reset()
    with open(path, "rb") as f:
        core.load_raw_state(f.read())
    core.run_frame()
    buf = bytes(ffi.buffer(img.buffer, w * h * 4))          # RGBX, like solo_recorder.draw
    arr = np.frombuffer(buf, np.uint8).reshape(h, w, 4)[..., :3]
    iw = bytes(ffi.buffer(ffi.cast("uint8_t *", core._native.memory.iwram), IWRAM))
    return arr.copy(), iw


def frame_from_ring(dumpdir):
    meta = json.load(open(os.path.join(dumpdir, "meta.json")))
    n = meta["n"]
    with open(os.path.join(dumpdir, "ring.bin"), "rb") as f:
        f.seek((n - 1) * SNAP)          # last snapshot
        iw = f.read(IWRAM)              # iwram is first in each snapshot
    arr = np.array(Image.open(os.path.join(dumpdir, "mark.png")).convert("RGB"), np.uint8)
    return arr, iw


def mark(img, x, y, c, r=3):
    h, w, _ = img.shape
    for d in range(-r, r + 1):
        if 0 <= y < h and 0 <= x + d < w: img[y, x + d] = c
        if 0 <= x < w and 0 <= y + d < h: img[y + d, x] = c


def main():
    caps = []  # (name, frame, iw)
    for p in sorted(glob.glob(os.path.join(STATES, "alttp_human-*.state"))):
        n = int(p.split("-")[-1].split(".")[0])
        if n in (4,) or n >= 7:  # new captures + room04 as a known-key control
            fr, iw = frame_from_state(p); caps.append((f"F8:{os.path.basename(p)}", fr, iw))
    for d in sorted(glob.glob(os.path.join(RINGS, "*-hit"))):
        fr, iw = frame_from_ring(d); caps.append((f"F11:{os.path.basename(d)[9:15]}", fr, iw))

    tiles = []; palette = {}
    print(f"{'capture':28s}  link      on-screen sprites (slot@sx,sy hp=HP ty=TYPE)")
    for name, fr, iw in caps:
        (lx, ly), ents = scan(iw)
        vis = fr.copy(); mark(vis, lx, ly, [0, 255, 0])
        desc = []
        for (i, sx, sy, hp, ty) in ents:
            col = [255, 0, 0] if hp > 0 else [0, 128, 255]
            mark(vis, sx, sy, col)
            desc.append(f"s{i}@{sx},{sy} hp={hp} ty=0x{ty:02X}")
            if hp == 0:
                palette.setdefault(ty, []).append(name)
        print(f"{name:28s}  {lx:3d},{ly:3d}   " + (" | ".join(desc) if desc else "(none on-screen)"))
        tiles.append(np.kron(vis, np.ones((2, 2, 1), np.uint8)))

    # montage grid, 5 wide
    W = 5; cellw = tiles[0].shape[1]; cellh = tiles[0].shape[0]
    rows = []
    for r in range(0, len(tiles), W):
        row = tiles[r:r + W]
        while len(row) < W:
            row.append(np.zeros((cellh, cellw, 3), np.uint8))
        rows.append(np.hstack(row))
    Image.fromarray(np.vstack(rows)).save(os.path.join(DET, "_capture_scan.png"))

    print("\n=== HP==0 item TYPE-BYTE palette (type -> #captures) ===")
    for ty in sorted(palette):
        print(f"  ty=0x{ty:02X}: {len(palette[ty])} captures  e.g. {palette[ty][0]}")
    print(f"\nmontage -> _capture_scan.png ({len(caps)} captures, grid {W} wide, order = print order)")
    print("DONE")


if __name__ == "__main__":
    main()
