"""Rung 2 phase 0: WHERE does the kill->collect chain break, per policy?

Standard episodes (no pre-dropped key) from the -04 start state. Per episode we
log kill events (with Link's room at that step) and key events, then report the
chain stats that decide the rung-2 teacher design:
  kills_04/ep        kills in the START room (the key-dropping enemy lives there;
                     kills elsewhere may not drop keys, so chain stats gate on it)
  collect|kill04     fraction of first in-room kills followed by a key pickup
                     within --window decisions (the chain-completion link)
  delay median       decisions from kill to pickup when it happens
Kill validation copies harvest_key_frames.py's teardown guard (Link alive, no
death this step, live enemy position) -- death teardown fakes 11 kills at once.

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/rl thor-rl:cu130 python3 -u chain_probe.py \\
      --models base=runs/alttp_s1v2_scratch04/ppo_alttp_600000_steps.zip \\
               distilled=runs/distill_r1/s1_distilled.zip
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from alttp_ppo_env import AlttpPpoEnv  # noqa: E402

ENEMY_X, ENEMY_Y = 0x03852, 0x03854


def u16(iw, addr):
    return int(iw[addr]) | (int(iw[addr + 1]) << 8)


def room_of(x, y):
    return (x >> 9, y >> 9)


def run_episode(env, model, seed, ffi, window):
    obs, _ = env.reset(seed=seed)
    # core exists only after reset, and the pointer may change per reset -- cast
    # fresh each episode (harvest_key_frames.py does the same)
    iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
    model.set_random_seed(seed)
    x, y = env.oracle.read_pos(env.core)
    start_room = room_of(x, y)
    kills = []          # (step, in_start_room)
    keys = []           # step
    step = 0
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=False)
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        step += 1
        x, y = env.oracle.read_pos(env.core)
        alive = env.oracle.prev_health is not None and env.oracle.prev_health > 0
        died_now = any(ch == "death" for _t, ch, _d, _v in info["events"])
        for _t, ch, _d, val in info["events"]:
            if ch == "enemy_dmg" and val == 0:
                ex, ey = u16(iw, ENEMY_X), u16(iw, ENEMY_Y)
                if alive and not died_now and (ex, ey) != (0, 0):
                    kills.append((step, room_of(x, y) == start_room))
            elif ch == "key":
                keys.append(step)

    k04 = [s for s, in_room in kills if in_room]
    first_k04 = k04[0] if k04 else None
    collected = None
    delay = None
    if first_k04 is not None:
        after = [t for t in keys if t >= first_k04]
        if after and after[0] - first_k04 <= window:
            collected = True
            delay = after[0] - first_k04
        else:
            collected = False
    return {"kills": len(kills), "kills_04": len(k04), "keys": len(keys),
            "first_kill04_step": first_k04, "collected_after_kill": collected,
            "delay": delay}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path.zip ...")
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-04.state"))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=4000)
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "chain_probe_results.json"))
    a = ap.parse_args()

    from mgba._pylib import ffi
    from stable_baselines3 import PPO

    results = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        model = PPO.load(path, device="cuda")
        eps = []
        env = AlttpPpoEnv(state_path=a.state, horizon=a.horizon)
        for i in range(a.episodes):
            eps.append(run_episode(env, model, a.seed + i, ffi, a.window))
        env.close()
        results[name] = eps

        n = len(eps)
        had = [e for e in eps if e["first_kill04_step"] is not None]
        coll = [e for e in had if e["collected_after_kill"]]
        delays = sorted(e["delay"] for e in coll)
        print("%-10s n=%d | kills/ep=%.2f kills04/ep=%.2f keys/ep=%.2f | "
              "eps-with-04-kill=%d/%d | collect|kill04=%d/%d (%.0f%%) | "
              "delay med=%s | med first kill04 step=%s"
              % (name, n,
                 float(np.mean([e["kills"] for e in eps])),
                 float(np.mean([e["kills_04"] for e in eps])),
                 float(np.mean([e["keys"] for e in eps])),
                 len(had), n, len(coll), max(1, len(had)),
                 100.0 * len(coll) / max(1, len(had)),
                 delays[len(delays) // 2] if delays else None,
                 int(np.median([e["first_kill04_step"] for e in had])) if had else None),
              flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("SAVED", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
