"""Temp: identify a savestate by its RAM oracle + a single rendered frame.
Runs in thor-rl:cu130 (has mgba + numpy + oracle). Saves frame as .npy (no image
lib needed in-container); host converts to PNG. Cleaned up after use."""
import sys, os
sys.path.insert(0, "/work/train/rl")
import numpy as np
import mgba.core, mgba.image, mgba.log
from mgba._pylib import ffi
from oracle import AlttpOracle

mgba.log.silence()
ROM = "/work/Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"
W, H = 240, 160

states = sys.argv[1:]
o = AlttpOracle()
for sp in states:
    core = mgba.core.load_path(ROM)
    w, h = core.desired_video_dimensions()
    img = mgba.image.Image(w, h)
    core.set_video_buffer(img)
    core.reset()
    with open(sp, "rb") as f:
        ok = core.load_raw_state(f.read())
    assert ok, "load_raw_state failed: %s" % sp
    core.set_keys(raw=0)
    core.run_frame()
    frame = np.frombuffer(ffi.buffer(img.buffer, W * H * 4), np.uint8).reshape(H, W, 4)[:, :, :3].copy()
    x, y = o.read_pos(core)
    name = os.path.basename(sp).replace(".state", "")
    print("STATE %-18s health=%2d/%2d keys=%d magic=%3d rupees=%3d pos=(%5d,%5d) room=%s"
          % (name, o.read_health(core), o.read_health_max(core), o.read_keys(core),
             o.read_magic(core), o.read_rupees(core), x, y, o.room_of(x, y)))
    np.save("/work/_id_%s.npy" % name, frame)
