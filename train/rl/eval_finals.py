"""Eval each run's final.zip both DETERMINISTIC(argmax) and sampled, vs random.

The h13 lesson: training ep_rew_mean lies -- h13 trained to a rolling-mean of 120
but its actual final policy collected 0 keys. So a "settled" final (final==peak in
training reward) must be confirmed by EVALUATING the policy, and deterministic
(argmax) is the deployment-relevant read. Reports keys/damage/death so engagement
is judged on behaviour, not just reward.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def evl(path, det, state_path, episodes, horizon):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS
    if path is None:
        policy, want = None, None
    else:
        policy = PPO.load(path, device="cpu")
        want = tuple(policy.observation_space.shape)
    keys = dmg = dth = 0
    rew = 0.0
    import random as _r
    for seed in range(episodes):
        env = AlttpPpoEnv(horizon=horizon, state_path=state_path, death_terminates=True)
        obs, _ = env.reset()
        torch.manual_seed(seed); np.random.seed(seed)
        rng = _r.Random(seed)
        tot = 0.0
        while True:
            if policy is None:
                act = rng.randrange(N_ACTIONS)
            else:
                a = np.asarray(obs)
                if tuple(a.shape) != want:
                    a = np.transpose(a, (2, 0, 1))
                act, _ = policy.predict(a, deterministic=det)
                act = int(act)
            obs, r, term, trunc, info = env.step(act)
            tot += r
            for _t, ch, _d, _v in info["events"]:
                if ch == "key":
                    keys += 1
                elif ch == "damage":
                    dmg += 1
                elif ch == "death":
                    dth += 1
            if term or trunc:
                break
        rew += tot
        env.close()
    return rew / episodes, keys, dmg, dth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/alttp_lr11", "runs/alttp_lr14"])
    ap.add_argument("--state", default="alttp_human-04")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=1500)
    args = ap.parse_args()
    state_path = os.path.join(HERE, "states", args.state + ".state")

    r, k, dm, dt = evl(None, False, state_path, args.episodes, args.horizon)
    print("RANDOM baseline: rew %+.1f  keys %d  dmg %d  deaths %d\n" % (r, k, dm, dt))
    for run in args.runs:
        path = os.path.join(run, "ppo_alttp_final.zip")
        if not os.path.exists(path):
            print("%s: no final.zip" % run); continue
        for det in (True, False):
            r, k, dm, dt = evl(path, det, state_path, args.episodes, args.horizon)
            print("  %-16s %-13s rew %+7.1f  keys %2d  dmg %2d  deaths %d"
                  % (os.path.basename(run), "DETERMINISTIC" if det else "sampled",
                     r, k, dm, dt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
