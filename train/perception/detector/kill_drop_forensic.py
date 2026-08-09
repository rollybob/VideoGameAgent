"""kill_drop_forensic.py -- map how an item populates AFTER an enemy is killed (room 04).

Tests the 'death-animation transient' hypothesis for the item-label 0% on room4: when the
enemy dies (slot HP 0x03250+i -> 0), does its sprite slot (0x03846+4i) hold a stable KEY
immediately, or a death-animation first (so the HP==0 label fires before the key renders)?

Kills one enemy with ChainHunter, then STANDS STILL (dir=0) and logs the dying slot's
(hp, world xy, screen xy) every frame for ~60 frames post-kill, plus a 3x-zoom crop strip
around the drop so the death-anim -> key sequence is visible in one image. CPU only, ~30s.
Run in thor-rl:cu130 (mgba bindings + PIL). No training, no GPU.
"""
import os, sys, warnings, random
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.abspath(os.path.join(HERE, "..", "..", "rl"))
sys.path.insert(0, RL)
from alttp_ppo_env import AlttpPpoEnv
from harvest_key_frames import u16
from chain_harvest import ChainHunter
from mgba._pylib import ffi
from PIL import Image
CAM_X, CAM_Y = 0x02B82, 0x02B86


def slots(iw):
    return [(int(iw[0x03250 + i]), u16(iw, 0x03846 + 4 * i), u16(iw, 0x03848 + 4 * i)) for i in range(16)]


def main():
    state = os.path.join(RL, "states", "alttp_human-04.state")
    env = AlttpPpoEnv(state_path=state, horizon=10 ** 7, death_terminates=True); env.reset(seed=3)
    iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
    rng = random.Random(3); _ = rng.random()
    h = ChainHunter(iw, env.oracle, env.core, rng, validate=True)
    prev = slots(iw); kill_slot = kill_step = drop_w = None
    log = []; crops = []; drops = set()
    for step in range(500):
        a = h.act(drops) if kill_slot is None else np.array([0, 0, 0, 0, 0])  # after kill: stand still
        env.step(np.array(a, dtype=np.int64))
        cur = slots(iw)
        if kill_slot is None:
            for i in range(16):
                if prev[i][0] > 0 and cur[i][0] == 0 and prev[i][2] != 0:  # HP>0 -> 0 at a real pos
                    kill_slot, kill_step, drop_w = i, step, (cur[i][1], cur[i][2]); break
            prev = cur
        if kill_slot is not None:
            hp, wx, wy = cur[kill_slot]
            cx, cy = u16(iw, CAM_X), u16(iw, CAM_Y); sx, sy = wx - cx, wy - cy
            rel = step - kill_step
            log.append((rel, hp, wx, wy, sx, sy))
            if rel % 3 == 0 and 0 <= sx < 240 and 0 <= sy < 160:
                f = np.array(env._frames[-1], np.uint8)
                y0, y1 = max(0, sy - 16), min(160, sy + 16); x0, x1 = max(0, sx - 16), min(240, sx + 16)
                c = f[y0:y1, x0:x1]; cc = np.zeros((32, 32, 3), np.uint8); cc[:c.shape[0], :c.shape[1]] = c
                crops.append(cc)
            if rel >= 60: break
    env.close()
    if kill_slot is None:
        print("NO KILL in 500 steps (ChainHunter did not down an enemy) -- rerun or pick another room", flush=True)
        return
    print(f"kill at step {kill_step}, slot {kill_slot}, drop world={drop_w}", flush=True)
    print("rel_frame  hp   worldx worldy   scrx scry", flush=True)
    for r, hp, wx, wy, sx, sy in log[:45]:
        print(f"   {r:3d}    {hp:3d}   {wx:5d} {wy:5d}    {sx:4d} {sy:4d}", flush=True)
    if crops:
        strip = np.hstack([np.kron(c, np.ones((3, 3, 1), np.uint8)) for c in crops])
        Image.fromarray(strip).save(os.path.join(HERE, "_killdrop_strip.png"))
        print(f"WROTE _killdrop_strip.png ({len(crops)} crops @3-frame spacing, 3x zoom)", flush=True)


if __name__ == "__main__":
    main()
