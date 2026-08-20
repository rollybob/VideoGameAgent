"""Room4 statue negatives (detector v5b, 2026-08-20).

The v5 FP receipt: residual item fires sit ON the statues (61.6% of
item-label-EMPTY room4 frames), because statue-heavy PLAIN room4 frames were
never in item training -- suppression came only incidentally from keydrop
frames where spawn statues are rarely in view. This generates a fresh
NO-ATTACK room4 walk (atk=0 -> no kills -> guaranteed item-free), seed
disjoint from the room4.npz eval walk, as EXPLICIT item negatives; keys are
pasted onto half the frames (gen_paste machinery) so the net sees key vs
statue in the same visual context. Trainer mask (0,0,1): item channel only,
so the held-room enemy/link evals stay clean.

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/perception/detector thor-rl:cu130 python3 -u gen_negs.py
"""
import os
import sys
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.abspath(os.path.join(HERE, "..", "..", "rl"))
sys.path.insert(0, RL)
sys.path.insert(0, HERE)
from alttp_ppo_env import AlttpPpoEnv               # noqa: E402
from harvest_key_frames import u16                  # noqa: E402
from gen_paste import key_sprite, paste, far        # noqa: E402

CAM_X, CAM_Y = 0x02B82, 0x02B86
DIRS = [1, 2, 3, 4, 5, 6, 7, 8]
N_FRAMES = 700
SEED = 777          # disjoint from room4.npz (seed 100): same room, unseen frames


def walk_room4():
    """gen_data.gen_room recipe with atk=0 (no kills possible)."""
    from mgba._pylib import ffi
    state = os.path.join(RL, "states", "alttp_human-04.state")

    def new_env():
        e = AlttpPpoEnv(state_path=state, horizon=10 ** 7, death_terminates=True)
        e.reset(seed=SEED)
        return e, ffi.cast("uint8_t *", e.core._native.memory.iwram)

    env, iw = new_env()
    start_cell = (u16(iw, 0x038F4) >> 9, u16(iw, 0x038F0) >> 9)
    rng = random.Random(SEED * 13 + 4)
    F, L, E = [], [], []
    d = rng.choice(DIRS)
    prev_cam = None
    since = 0
    while len(F) < N_FRAMES:
        cx, cy = u16(iw, CAM_X), u16(iw, CAM_Y)
        lsx, lsy = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
        if lsx < 40:
            d = 4
        elif lsx > 200:
            d = 3
        elif lsy < 40:
            d = 2
        elif lsy > 130:
            d = 1
        elif rng.random() < 0.12:
            d = rng.choice(DIRS)
        env.step(np.array([d, 0, 0, 0, 0], dtype=np.int64))
        since += 1
        if env.oracle.read_health(env.core) <= 0 or since >= 300:
            env.close()
            env, iw = new_env()
            prev_cam = None
            since = 0
            continue
        if since % 2:
            continue
        cx, cy = u16(iw, CAM_X), u16(iw, CAM_Y)
        if (u16(iw, 0x038F4) >> 9, u16(iw, 0x038F0) >> 9) != start_cell:
            env.close()
            env, iw = new_env()
            prev_cam = None
            since = 0
            continue
        if prev_cam is not None and abs(cx - prev_cam[0]) + abs(cy - prev_cam[1]) > 6:
            prev_cam = (cx, cy)
            continue
        prev_cam = (cx, cy)
        lsx, lsy = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
        if not (0 <= lsx < 240 and 0 <= lsy < 160):
            continue
        F.append(np.array(env._frames[-1], np.uint8))
        L.append((lsx, lsy))
        en = []
        for i in range(16):
            ex, ey = u16(iw, 0x03846 + 4 * i), u16(iw, 0x03848 + 4 * i)
            if (ex, ey) == (0, 0):
                continue
            sx, sy = ex - cx, ey - cy
            if 0 <= sx < 240 and 0 <= sy < 160 and int(iw[0x03250 + i]) > 0:
                en.append((sx, sy))
        e = np.full((8, 2), -1, np.int16)
        for j, (sx, sy) in enumerate(en[:8]):
            e[j] = (sx, sy)
        E.append(e)
    env.close()
    return F, L, E


def main():
    rng = np.random.RandomState(23)
    spr, msk = key_sprite()
    F0, L0, E0 = walk_room4()
    F, L, E, I = [], [], [], []
    for i in range(len(F0)):
        fr = F0[i]
        it = np.full((8, 2), -1, np.int16)
        if i % 2:                       # half the frames: paste a key for contrast
            lx, ly = L0[i]
            en = [(ex, ey) for ex, ey in E0[i] if ex >= 0]
            for _ in range(20):
                x = int(rng.randint(24, 216))
                y = int(rng.randint(40, 140))
                if far(x, y, [(lx, ly)], 24) and far(x, y, en, 20):
                    out = paste(fr, spr, msk, x, y)
                    if out is not None:
                        fr = out
                        it[0] = (x, y)
                        break
        F.append(fr)
        L.append(L0[i])
        E.append(E0[i])
        I.append(it)
    np.savez_compressed(os.path.join(HERE, "data", "room4_negs.npz"),
                        frames=np.stack(F), link=np.array(L, np.int16),
                        enem=np.stack(E), item=np.stack(I))
    npos = int(sum((ii[:, 0] >= 0).any() for ii in I))
    print(f"room4_negs: {len(F)} frames ({npos} with pasted key, "
          f"{len(F) - npos} pure statue negatives)", flush=True)


if __name__ == "__main__":
    main()
