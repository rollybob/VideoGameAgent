"""Render a trained policy (and a random baseline) to raw frames for playback.

Why a dump instead of a live viewer: the policy needs SB3 + numpy, which only
exist in the container, while the display lives on the host's :1 desktop and its
venv has neither. So the container writes every emulator frame plus per-frame
state, and play_dump.py (host, pygame) plays it back.

Every frame is captured, not just one per agent step, so playback is true 60Hz
rather than a 15Hz slideshow.

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/rl thor-rl:cu130 python3 dump_policy_run.py \\
      runs/alttp_a2/ppo_alttp_final.zip --out /vga/link/sessions/policy_dumps
"""
import argparse
import json
import os
import random
import sys

import numpy as np
from mgba._pylib import ffi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

W, H = 240, 160


def frame_rgb(image):
    buf = ffi.buffer(image.buffer, W * H * 4)
    return np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4)[:, :, :3].copy()


def dump(tag, policy, state_path, out_dir, max_frames, seed, frame_skip):
    from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS
    from oracle import AlttpOracle

    env = AlttpPpoEnv(horizon=10 ** 9, state_path=state_path, death_terminates=True,
                      frame_skip=1)          # step one frame at a time so every
                                             # frame is captured; the policy is
                                             # still only consulted every
                                             # frame_skip frames, matching training
    obs, _ = env.reset()
    oracle = AlttpOracle()
    oracle.step(0, env.core)
    rng = random.Random(seed)

    path = os.path.join(out_dir, tag + ".rgb")
    recs = []
    action = 0
    n = 0
    with open(path, "wb") as fh:
        while n < max_frames:
            if n % frame_skip == 0:
                if policy is None:
                    action = rng.randrange(N_ACTIONS)
                else:
                    a, _ = policy.predict(obs, deterministic=False)
                    action = int(a)
            obs, r, term, trunc, info = env.step(action)
            n += 1
            fh.write(frame_rgb(env.image).tobytes())
            hp = oracle.read_health(env.core)
            rup = oracle.read_rupees(env.core)
            recs.append([action, int(hp), int(rup),
                         1 if any(c == "damage" for _t, c, _d, _v in info["events"]) else 0,
                         1 if any(c == "death" for _t, c, _d, _v in info["events"]) else 0])
            if term:
                break
    env.close()
    meta = {"tag": tag, "frames": n, "w": W, "h": H, "frame_skip": frame_skip,
            "records": recs, "note": "records = [action, hp, rupees, damage, death]"}
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(meta, f)
    print("%s: %d frames -> %s" % (tag, n, path), flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-02.state"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-skip", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cuda")
    dump("policy", model, args.state, args.out, args.max_frames, args.seed, args.frame_skip)
    dump("random", None, args.state, args.out, args.max_frames, args.seed, args.frame_skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
