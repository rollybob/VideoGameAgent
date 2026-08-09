"""Rung 2 teacher: oracle-scripted kill->collect chain demos (spec:
docs/DISTILL_R2_SPEC_2026-08-07.md).

ChainHunter = harvest_key_frames.ScriptedHunter minus the scatter/loiter harvest
scaffolding: engage -> kill -> (loiter 6-12 decisions while the drop spawns) ->
walk onto the drop -> collect. A small random PRE-ROLL (0-15 random-direction
decisions) diversifies approach paths so the student sees varied engages, not one
groove. RAM guides the teacher OFFLINE ONLY; the recorded npz (same schema as
round 1: frames/actions/source/first_key_step) contains pixels + actions.
SUCCESS episodes only (kill + collect), trimmed to pickup+2. source=0 throughout
(uniform sample weight; round-1 bursts keep their 3x -- pre-registered mix rule).

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/rl thor-rl:cu130 python3 -u chain_harvest.py \\
      --target 350 --episodes 420 --out distill_data/round2
"""
import argparse
import json
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from alttp_ppo_env import AlttpPpoEnv  # noqa: E402
from harvest_key_frames import ScriptedHunter, u16, ENEMY_X, ENEMY_Y  # noqa: E402

DIR_KEYS = [1, 2, 3, 4, 5, 6, 7, 8]   # 8-way MultiDiscrete dir indices


class ChainHunter(ScriptedHunter):
    """Kill then collect directly: no scatter (timer=0 -> straight to a short
    loiter that covers the drop-spawn delay), pickup ALWAYS armed."""

    def on_kill(self, drop):
        super().on_kill(drop)
        self.timer = 0
        self.pickup_target = tuple(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-04.state"))
    ap.add_argument("--target", type=int, default=350, help="stop after this many successes")
    ap.add_argument("--episodes", type=int, default=420, help="hard episode cap")
    ap.add_argument("--horizon", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--far-frac", type=float, default=0.0,
                    help="fraction of episodes where the teacher scatters 8-18 decisions "
                         "away post-kill before collecting (variant C: far-seek IN chain "
                         "context). Scatter/loiter steps are marked source=2 = excluded "
                         "from imitation sampling -- the student clones the RECOVERY, "
                         "never the walk-away.")
    ap.add_argument("--out", default=os.path.join(HERE, "distill_data", "round2"))
    a = ap.parse_args()

    from mgba._pylib import ffi

    os.makedirs(a.out, exist_ok=True)
    successes = 0
    fails = {"no_kill": 0, "no_collect": 0, "death": 0}
    for ep in range(a.episodes):
        if successes >= a.target:
            break
        env = AlttpPpoEnv(state_path=a.state, horizon=a.horizon)
        obs, _ = env.reset(seed=a.seed + ep)
        iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
        rng = random.Random(a.seed * 7 + ep)
        far_ep = rng.random() < a.far_frac
        if far_ep:
            # original wide-scatter hunter: kill -> scatter 8-18 -> loiter -> pickup
            hunter = ScriptedHunter(iw, env.oracle, env.core, rng, validate=True,
                                    short_scatter=False)
        else:
            hunter = ChainHunter(iw, env.oracle, env.core, rng, validate=True)

        rec_frames = [env._frames[-1].copy()]
        rec_actions = []
        rec_source = []
        preroll = rng.randint(0, 15)
        drops = []
        key_step = None
        died = False
        step = 0
        done = False
        tail = 0
        while not done:
            if step < preroll:
                act = np.array([rng.choice(DIR_KEYS), 0, 0, 0, 0], dtype=np.int64)
                src = 2      # random pre-roll: context only, never an imitation target
            else:
                mode_before = hunter.mode
                act = np.array(hunter.act(drops), dtype=np.int64)
                src = 2 if mode_before in ("scatter", "loiter") else 0
            obs, r, term, trunc, info = env.step(act)
            rec_frames.append(env._frames[-1].copy())
            rec_actions.append(act.copy())
            rec_source.append(src)
            done = term or trunc
            step += 1
            alive = env.oracle.prev_health is not None and env.oracle.prev_health > 0
            died_now = any(ch == "death" for _t, ch, _d, _v in info["events"])
            for _t, ch, _d, val in info["events"]:
                if ch == "enemy_dmg" and val == 0:
                    ex, ey = u16(iw, ENEMY_X), u16(iw, ENEMY_Y)
                    if alive and not died_now and (ex, ey) != (0, 0):
                        drops.append([ex, ey, step])
                        hunter.on_kill((ex, ey))
                elif ch == "key":
                    key_step = step
                    hunter.on_pickup()
                elif ch == "death":
                    died = True
            if died:
                break
            if key_step is not None:
                tail += 1
                if tail >= 2:      # keep pickup+2 then stop -- demo complete
                    break
        env.close()

        if key_step is not None and not died:
            end = min(key_step + 2, len(rec_actions))
            path = os.path.join(a.out, "chain_ep%04d_seed%d.npz" % (ep, a.seed + ep))
            np.savez_compressed(
                path,
                frames=np.stack(rec_frames[:end + 1]).astype(np.uint8),
                actions=np.stack(rec_actions[:end]).astype(np.int8),
                source=np.asarray(rec_source[:end], dtype=np.uint8),
                first_key_step=np.int64(key_step))
            successes += 1
        elif died:
            fails["death"] += 1
        elif not drops:
            fails["no_kill"] += 1
        else:
            fails["no_collect"] += 1
        if (ep + 1) % 25 == 0:
            print("ep %d: successes=%d fails=%s" % (ep + 1, successes, fails), flush=True)

    print("DONE: %d successes in %d episodes, fails=%s" % (successes, ep + 1, fails), flush=True)
    with open(os.path.join(a.out, "harvest_summary.json"), "w") as f:
        json.dump({"successes": successes, "episodes": ep + 1, "fails": fails}, f, indent=2)
    return 0 if successes >= min(a.target, 250) else 3


if __name__ == "__main__":
    sys.exit(main())
