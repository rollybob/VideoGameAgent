"""DR eval: per-state rooms-visited + keys + coverage for a domain-randomized
policy. The pre-registered metric for run_dr_seeds.sh:
  PRIMARY  (explore/generalize) rooms-visited/ep beats the random floor,
  RETAIN   keys in -04 (competency kept, not just exploration),
  GENERALIZE zero-shot on the held-out room -01.
Sweeps checkpoints (never final.zip alone -- the a5 lesson). Evals each state
SEPARATELY (fixed reset) so rooms-visited measures how far the policy explores
OUT of that room. --baseline-only measures just the random floor (no checkpoint
needed) -- also the harness self-test.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga/train/rl \\
      thor-rl:cu130 python3 dr_eval.py --run runs/alttp_dr11
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_sweep import ckpt_steps, pick  # noqa: E402


def eval_state(policy, state_path, episodes, horizon):
    import random
    import numpy as np
    import torch
    from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS
    want = tuple(policy.observation_space.shape) if policy is not None else None

    def fix_obs(o):
        a = np.asarray(o)
        if want is None or tuple(a.shape) == want:
            return a
        t = np.transpose(a, (2, 0, 1))
        if tuple(t.shape) == want:
            return t
        raise AssertionError("obs %s cannot match policy %s" % (a.shape, want))

    agg = dict(rew=0.0, rooms=0.0, cells=0.0, keys=0.0, dmg=0.0, dth=0.0)
    for seed in range(episodes):
        env = AlttpPpoEnv(horizon=horizon, state_path=state_path, death_terminates=True)
        obs, _ = env.reset(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = random.Random(seed)
        r = 0.0
        while True:
            if policy is None:
                act = rng.randrange(N_ACTIONS)
            else:
                act = int(policy.predict(fix_obs(obs), deterministic=False)[0])
            obs, rr, term, trunc, info = env.step(act)
            r += rr
            for _t, ch, _d, _v in info["events"]:
                if ch == "key":
                    agg["keys"] += 1
                elif ch == "damage":
                    agg["dmg"] += 1
                elif ch == "death":
                    agg["dth"] += 1
            if term or trunc:
                break
        agg["rew"] += r
        agg["rooms"] += len(env.oracle.rooms_seen)
        agg["cells"] += len(env.oracle.cells_seen)
        env.close()
    return {k: v / episodes for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/alttp_dr11")
    ap.add_argument("--states", nargs="+", default=["02", "03", "04", "01"],
                    help="02/03/04 = train mix; 01 = held-out generalization")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--ckpts", type=int, default=5)
    ap.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args()

    def sp(s):
        return os.path.join(HERE, "states", "alttp_human-%s.state" % s)

    print("=== RANDOM baseline (%d eps, horizon %d) ===" % (a.episodes, a.horizon))
    print("%-4s %8s %6s %6s %6s %5s" % ("st", "rew", "rooms", "cells", "keys", "dth"))
    base = {}
    for s in a.states:
        d = eval_state(None, sp(s), a.episodes, a.horizon)
        base[s] = d
        print("%-4s %8.1f %6.2f %6.1f %6.2f %5.2f"
              % (s, d["rew"], d["rooms"], d["cells"], d["keys"], d["dth"]))
    if a.baseline_only:
        return 0

    from stable_baselines3 import PPO
    steps = ckpt_steps(a.run)
    targets = pick(steps, a.ckpts)
    print("\n=== %s: sweep %s (+final) ===" % (os.path.basename(a.run), targets))
    for st in targets + ["final"]:
        fn = "ppo_alttp_final.zip" if st == "final" else "ppo_alttp_%d_steps.zip" % st
        path = os.path.join(a.run, fn)
        if not os.path.exists(path):
            continue
        pol = PPO.load(path, device="cpu")
        print("-- ckpt %s --" % st)
        for s in a.states:
            d = eval_state(pol, sp(s), a.episodes, a.horizon)
            b = base[s]
            tag = "  <explore>" if d["rooms"] > b["rooms"] + 0.2 else ""
            held = "  [HELD-OUT]" if s == "01" else ""
            print("  %-4s rew %+7.1f | rooms %.2f (rand %.2f) | keys %.2f (rand %.2f) | dth %.2f%s%s"
                  % (s, d["rew"], d["rooms"], b["rooms"], d["keys"], b["keys"], d["dth"], tag, held))
    return 0


if __name__ == "__main__":
    sys.exit(main())
