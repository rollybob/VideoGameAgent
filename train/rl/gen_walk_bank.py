"""Walker position banks: guaranteed-REACHABLE navigation-target positions per savestate.

The goal-conditioned walker (walker_env.py) trains on "walk to target X" with X
sampled at reset. Sampling arbitrary coordinates would set unreachable targets
(inside walls / pits), polluting the reward with unwinnable episodes; instead
each state gets a bank of positions a random walk actually STOOD ON -- reachable
by construction. Reuses gen_data.py's proven patterns: recreate the env every
~300 steps (mgba long-run segfault dodge), recreate on death and on leaving the
start room (walker episodes reset into the start room, so the bank is start-room
only; crossing rooms is System-2's job, not the walker's).

Run (container, one state per process; walkbank job fans out with xargs -P):
  docker run --rm -v ~/projects/VGA:/work -w /work \
    thor-rl:cu130 python3 train/rl/gen_walk_bank.py --state alttp_human-04
"""
import os, sys, argparse, random, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alttp_ppo_env import AlttpPpoEnv
from harvest_key_frames import u16
from mgba._pylib import ffi

GRID = 8          # dedupe: keep one position per 8x8-world-unit cell
DIRS = [1, 2, 3, 4, 5, 6, 7, 8]


def gen(state_name, steps, seed):
    state = os.path.join(HERE, "states", state_name + ".state")
    if not os.path.exists(state):
        raise SystemExit("no such state: %s" % state)

    def new_env():
        e = AlttpPpoEnv(state_path=state, horizon=10 ** 7, death_terminates=True)
        e.reset(seed=seed)
        return e, ffi.cast("uint8_t *", e.core._native.memory.iwram)

    env, iw = new_env()
    home = (u16(iw, 0x038F4) >> 9, u16(iw, 0x038F0) >> 9)
    rng = random.Random(seed)
    seen = {}   # grid cell -> world pos of first visit
    d = rng.choice(DIRS)
    since = 0
    for _ in range(steps):
        if rng.random() < 0.10:
            d = rng.choice(DIRS)
        env.step(np.array([d, 0, 0, 0, 0], dtype=np.int64))
        since += 1
        dead = env.oracle.read_health(env.core) <= 0
        x, y = u16(iw, 0x038F4), u16(iw, 0x038F0)
        if dead or (x >> 9, y >> 9) != home or since >= 300:
            env.close()
            env, iw = new_env()
            since = 0
            continue
        seen.setdefault((x // GRID, y // GRID), (x, y))
    env.close()
    return np.array(sorted(seen.values()), dtype=np.int32), home


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="state basename, e.g. alttp_human-04")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    pos, home = gen(a.state, a.steps, a.seed)
    out_dir = os.path.join(HERE, "walker_banks")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, a.state + ".npz")
    np.savez_compressed(out, pos=pos, home=np.array(home, dtype=np.int32))
    if len(pos):
        span = "x %d-%d y %d-%d" % (pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max())
    else:
        span = "EMPTY"
    print("%s: %d bank positions, home room %s, %s" % (a.state, len(pos), home, span), flush=True)
