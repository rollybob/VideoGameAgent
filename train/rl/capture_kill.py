"""Capture a trained policy KILLING the -04 enemy, full RAM every step.

Tim's method for the enemy id/alive flag (2026-08-02): now that we can track the
enemy position (IWRAM 0x03852/0x03854), let Link actually kill it and watch what
changes AT THE INSTANT the enemy's movement stops -- a value that was nonzero and
now reads 0 (or freezes) is the alive/type/health/slot info. His caveat: the
sprite sheet may unload at the same moment, so several bytes can drop at once.

Idle Link only ever DIES, so a human-quality kill is needed -- supplied by a
trained policy from runs/BEST that reliably clears the room. Snapshots full RAM
every agent step and logs key/HP/enemy-pos so the death instant is locatable
(the position mirror freezes, and the key drops a moment later).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IW_SIZE = 32 * 1024
EW_SIZE = 256 * 1024
KEY, HP, EX, EY = 0x0234F, 0x0234D, 0x03852, 0x03854


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/BEST/alttp_g14_final.zip")
    ap.add_argument("--state", default="alttp_human-04")
    ap.add_argument("--seeds", type=int, default=8, help="try this many until a kill")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--after", type=int, default=40, help="steps to keep after first key")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS
    from mgba._pylib import ffi

    policy = PPO.load(args.model, device="cpu")
    want = tuple(policy.observation_space.shape)

    def fix_obs(o):
        a = np.asarray(o)
        return a if tuple(a.shape) == want else np.transpose(a, (2, 0, 1))

    state_path = os.path.join(HERE, "states", args.state + ".state")

    for seed in range(args.seeds):
        env = AlttpPpoEnv(horizon=10 ** 9, state_path=state_path, death_terminates=True)
        obs, _ = env.reset()
        torch.manual_seed(seed); np.random.seed(seed)
        core = env.core
        iw = ffi.cast("uint8_t *", core._native.memory.iwram)
        ew = ffi.cast("uint8_t *", core._native.memory.wram)

        def u16iw(a):
            return int(iw[a]) | (int(iw[a + 1]) << 8)

        rows, log = [], []
        key0 = int(ew[KEY])
        key_step = None
        step = 0
        while step < args.max_steps:
            action, _ = policy.predict(fix_obs(obs), deterministic=False)
            obs, r, term, trunc, info = env.step(int(action))
            rows.append(bytes(ffi.buffer(core._native.memory.iwram, IW_SIZE))
                        + bytes(ffi.buffer(core._native.memory.wram, EW_SIZE)))
            log.append({"tick": info["tick"], "key": int(ew[KEY]), "hp": int(ew[HP]),
                        "ex": u16iw(EX), "ey": u16iw(EY)})
            if key_step is None and int(ew[KEY]) > key0:
                key_step = step
            if key_step is not None and step >= key_step + args.after:
                break
            if term or trunc:
                break
            step += 1
        env.close()

        if key_step is not None:
            out = args.out or os.path.join(HERE, "scratch", "kill_" + args.state)
            os.makedirs(out, exist_ok=True)
            np.frombuffer(b"".join(rows), dtype=np.uint8).tofile(os.path.join(out, "ring.bin"))
            meta = {"label": "kill_" + args.state, "n": len(rows), "every": 4,
                    "iwram_size": IW_SIZE, "ewram_size": EW_SIZE,
                    "snap_size": IW_SIZE + EW_SIZE,
                    "layout": "iwram then ewram, per snapshot, snapshot-major",
                    "seed": seed, "key_step": key_step, "log": log,
                    "ticks": [r["tick"] for r in log]}
            with open(os.path.join(out, "meta.json"), "w") as f:
                json.dump(meta, f)
            print("KILL captured on seed %d: %d snaps, key at step %d (tick %d)"
                  % (seed, len(rows), key_step, log[key_step]["tick"]))
            print("enemy-pos + key/hp around the key event:")
            lo = max(0, key_step - 12)
            for i in range(lo, min(len(log), key_step + 6)):
                m = "  <- key" if i == key_step else ""
                print("  step %3d tick %4d  key=%d hp=%2d  enemy=(%d,%d)%s"
                      % (i, log[i]["tick"], log[i]["key"], log[i]["hp"],
                         log[i]["ex"], log[i]["ey"], m))
            print("wrote", out)
            return 0
        print("seed %d: no key in %d steps, trying next" % (seed, args.max_steps))

    print("no kill captured in %d seeds -- try a different --model" % args.seeds)
    return 1


if __name__ == "__main__":
    sys.exit(main())
