"""Task 09 perception bench: extract oracle-labeled gameplay frames.

For each in-game ALttP state, roll out with random actions and capture (frame, RAM
ground-truth) pairs. The RAM oracle gives objective labels the VLMs are then tested
against -- can the model READ the screen (hearts, rupees, enemy present)?  These are
knowledge-neutral: memorising ALttP does not help read the current HUD.

Only clean gameplay frames are kept (0 < health <= max), so menu/death/transition
frames with garbage RAM are dropped. Runs in thor-rl:cu130 (mgba + PIL).
mgba's frame buffer is BGR in byte order, so flip to RGB for PIL/VLMs.
"""
import os
import sys
import json
import random
import collections

sys.path.insert(0, "/work/train/rl")
import numpy as np
from PIL import Image
import mgba.core, mgba.image, mgba.log
from mgba._pylib import ffi
from oracle import AlttpOracle

mgba.log.silence()
ROM = "/work/Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"
W, H = 240, 160
STATES = ["alttp_ingame", "alttp_human-00", "alttp_human-01", "alttp_human-02",
          "alttp_human-03", "alttp_human-04", "alttp_human-05", "alttp_human-06"]
OUT = "/work/train/perception/bench_data"
SAMPLES_PER_STATE = 30

# action bitmasks: R=16 L=32 U=64 D=128, A=1, B=2. move / move+attack / plain / none.
DIRS = [16, 32, 64, 128]
ACTIONS = DIRS + [d | 1 for d in DIRS] + [d | 2 for d in DIRS] + [1, 0]

os.makedirs(OUT + "/frames", exist_ok=True)
rng = random.Random(1234)
labels = []
gid = 0
skipped = 0

for st in STATES:
    core = mgba.core.load_path(ROM)
    w, h = core.desired_video_dimensions()
    img = mgba.image.Image(w, h)
    core.set_video_buffer(img)
    core.reset()
    with open("/work/train/rl/states/%s.state" % st, "rb") as f:
        assert core.load_raw_state(f.read()), "load failed: %s" % st
    core.set_keys(raw=0)
    core.run_frame()
    o = AlttpOracle()
    for _ in range(SAMPLES_PER_STATE):
        core.set_keys(raw=rng.choice(ACTIONS))
        for _ in range(rng.randint(8, 18)):
            core.run_frame()
        core.set_keys(raw=0)
        for _ in range(8):          # settle HUD counters/animations before reading
            core.run_frame()
        health = o.read_health(core)
        hmax = o.read_health_max(core)
        if not (0 < health <= hmax <= 80):   # drop menu/death/transition garbage
            skipped += 1
            continue
        ewb = o._ew(core)
        rupees_shown = int(ewb[0x02342]) | (int(ewb[0x02343]) << 8)
        slots = o.read_enemy_hp_slots(core)
        enemy_ct = sum(1 for hp in slots if 0 < hp <= o.ENEMY_HP_SANE_MAX)
        x, y = o.read_pos(core)
        frame = np.frombuffer(ffi.buffer(img.buffer, W * H * 4), np.uint8).reshape(H, W, 4)[:, :, :3]
        fn = "f%04d.png" % gid
        Image.fromarray(frame[:, :, ::-1]).save(OUT + "/frames/" + fn)   # BGR -> RGB
        labels.append(dict(
            id=gid, file="frames/" + fn, state=st,
            health_eighths=health, health_max_eighths=hmax,
            hearts=round(health / 8, 1), max_hearts=round(hmax / 8, 1),
            rupees=rupees_shown, magic=o.read_magic(core), keys=o.read_keys(core),
            enemy_count=enemy_ct, enemy_present=enemy_ct > 0,
            room=[x >> 9, y >> 9], pos=[x, y]))
        gid += 1

with open(OUT + "/labels.jsonl", "w") as f:
    for L in labels:
        f.write(json.dumps(L) + "\n")

print("WROTE %d frames to %s (skipped %d invalid)" % (gid, OUT, skipped))
print("hearts dist:", dict(collections.Counter(L["hearts"] for L in labels)))
print("per-state:", dict(collections.Counter(L["state"] for L in labels)))
print("rupee values:", sorted(set(L["rupees"] for L in labels)))
print("enemy frames:", sum(L["enemy_present"] for L in labels))
