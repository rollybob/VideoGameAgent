"""
Scout probe: can the mGBA Python binding boot Four Swords HEADLESS and give us
a framebuffer, with NO X server / xdotool / mss involved?

This validates the load-bearing assumption for the future multiplayer co-op
harness (Path B: N cores in-process, lockstep-linked, direct frame + input).
It does NOT test linking yet -- just single-core boot + frame readout.
"""
import sys
import hashlib

import mgba.core
import mgba.image
import mgba.log

mgba.log.silence()  # keep stdout clean

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/" \
      "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"

core = mgba.core.load_path(ROM)
if core is None:
    print("FAIL: load_path returned None (ROM not loadable)")
    sys.exit(1)

w, h = core.desired_video_dimensions()
print("desired_video_dimensions:", (w, h))  # expect (240, 160) for GBA

image = mgba.image.Image(w, h)
core.set_video_buffer(image)
core.reset()

# Run ~5 seconds of emulated time to get past the boot logo into the title.
FRAMES = 320
for _ in range(FRAMES):
    core.run_frame()

buf = bytes(mgba.image.ffi.buffer(image.buffer))
nonzero = sum(1 for b in buf if b != 0)
digest = hashlib.sha1(buf).hexdigest()[:12]
print("frames run:", FRAMES)
print("framebuffer bytes:", len(buf), "nonzero:", nonzero,
      "({:.1f}%)".format(100.0 * nonzero / len(buf)))
print("framebuffer sha1[:12]:", digest)

out = "/home/timothy/mgba-mp-probe/four_swords_frame.png"
with open(out, "wb") as f:
    ok = image.save_png(f)
print("save_png ok:", ok, "->", out)

if nonzero == 0:
    print("RESULT: FAIL -- framebuffer is entirely blank")
    sys.exit(2)
print("RESULT: PASS -- headless boot + non-blank framebuffer via Python binding")
