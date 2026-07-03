#!/usr/bin/env python3
"""F21 smoke test: drive libmgba headless via the freshly built python bindings.
Proves the whole harness: load ROM (no window/X), run frames, grab the framebuffer,
and read GBA RAM -- all in one synced process. Run with the VGA venv + PYTHONPATH
pointing at the built package."""
import sys, os

ROM = os.path.expanduser("~/projects/VGA/Emulator/mGBA/roms/Pokemon AI Red.gba")
OUT = os.path.expanduser("~/projects/VGA/train/ram/smoke_frame.png")

import mgba.core, mgba.image, mgba.log
mgba.log.silence()  # quiet the core logger

print("mgba version:", mgba.core.version() if hasattr(mgba.core, "version") else "?")
core = mgba.core.load_path(ROM)
if core is None:
    sys.exit("FAIL: load_path returned None")
print("loaded core:", core)

# framebuffer sink
w, h = core.desired_video_dimensions()
print("video dims:", w, h)
image = mgba.image.Image(w, h)
core.set_video_buffer(image)

core.autoload_save()  # battery save (in-game data), not a savestate
core.reset()

N = 700  # ~11.6s: through BIOS + intro to title/first screen
for _ in range(N):
    core.run_frame()
print(f"ran {N} frames; frame_counter={core.frame_counter}")

# --- framebuffer dump ---
try:
    pil = image.to_pil().convert("RGB")
    pil.save(OUT)
    print("saved frame ->", OUT, pil.size)
except Exception as e:
    print("frame dump failed:", e)

# --- memory probe ---
mem = core.memory
# GBA: 'wram' == EWRAM (256KB @0x02000000), 'iwram' == IWRAM (32KB @0x03000000)
ew, iw = mem.wram, mem.iwram
print("EWRAM(wram) len:", len(ew), "IWRAM len:", len(iw))
ew_nz = sum(1 for i in range(len(ew)) if ew[i])
iw_nz = sum(1 for i in range(len(iw)) if iw[i])
print("EWRAM nonzero: %d/%d" % (ew_nz, len(ew)))
print("IWRAM nonzero: %d/%d" % (iw_nz, len(iw)))
print("EWRAM first64:", bytes(ew[i] for i in range(64)).hex())
# flat-bus absolute-address read (what we'll use for known addresses)
print("u8[0x02000000..8]:", bytes(mem.u8[0x02000000 + i] for i in range(8)).hex())
print("u8[0x03000000..8]:", bytes(mem.u8[0x03000000 + i] for i in range(8)).hex())
print("RESULT:", "OK - live RAM readable" if (ew_nz or iw_nz) else "WARN all-zero")
