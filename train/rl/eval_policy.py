"""Evaluate a trained policy against the random baseline on identical seeds.

A rising training curve is not a result. It says the number went up, not that
the agent learned anything worth having -- and this project has twice mistaken
an artifact for progress (a reward read off a non-health byte, and an agent that
had learned to quit the game). This script answers the blunter question: on the
same start-state and the same seeds, does the policy beat random, and WHAT does
its behaviour look like in terms of events?

  docker run --rm --runtime nvidia -v ~/projects/VGA:/vga -w /vga/train/rl \\
      thor-rl:cu130 python3 eval_policy.py runs/alttp_a2/ppo_alttp_final.zip
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run_episodes(policy, state_path, seeds, horizon, seed_episodes=True):
    import random
    import numpy as np
    from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS

    # OBSERVATION LAYOUT -- the bug that invalidated every eval before
    # 2026-08-01 19:10. SB3 wraps image envs in VecTransposeImage during
    # training, so the policy expects CHW, while the env yields HWC. predict()
    # does NOT raise on the mismatch: it silently returns a CONSTANT action
    # (measured: the same action 200/200 draws from one observation). Every
    # "policy" row was therefore a constant-button agent, which looks exactly
    # like "learned to avoid damage" -- full-horizon episodes, few hits, and
    # never a single collectible. Assert rather than transpose-and-hope, so a
    # future layout change fails loudly instead of quietly reporting nonsense.
    want = tuple(policy.observation_space.shape) if policy is not None else None

    def fix_obs(o):
        import numpy as _np
        a = _np.asarray(o)
        if want is None or tuple(a.shape) == want:
            return a
        if tuple(_np.transpose(a, (2, 0, 1)).shape) == want:
            return _np.transpose(a, (2, 0, 1))
        raise AssertionError("obs %s cannot be matched to policy %s"
                             % (a.shape, want))

    rews, lens = [], []
    ev = collections.Counter()
    for seed in seeds:
        env = AlttpPpoEnv(horizon=horizon, state_path=state_path, death_terminates=True)
        obs, _ = env.reset()
        rng = random.Random(seed)
        # predict(deterministic=False) SAMPLES from torch's global RNG, so
        # without this only the random baseline was reproducible: policy rows
        # drifted between runs (caught 2026-08-01 re-running the a2 config --
        # identical event counts but mean_len 2670 -> 2628). Seed per episode so
        # the policy stays stochastic but the table repeats.
        import torch
        if seed_episodes:
            torch.manual_seed(seed)
            np.random.seed(seed)
        total = 0.0
        while True:
            if policy is None:
                action = rng.randrange(N_ACTIONS)
            else:
                action, _ = policy.predict(fix_obs(obs), deterministic=False)
                action = int(action)
            obs, r, term, trunc, info = env.step(action)
            total += r
            for _t, ch, _d, _v in info["events"]:
                ev[ch] += 1
            if term or trunc:
                break
        rews.append(total)
        lens.append(info["tick"])
        env.close()
    n = len(rews)
    return sum(rews) / n, sum(lens) / n, dict(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="SB3 .zip checkpoint")
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-02.state"))
    ap.add_argument("--episodes", type=int, default=12)
    # Per-episode torch seeding makes the table repeatable, but if the trained
    # policy has low entropy the RNG barely matters and every "episode" replays
    # nearly the SAME trajectory -- 40 samples of one path, not 40 samples.
    # --no-seed-episodes measures the policy's real spread.
    ap.add_argument("--no-seed-episodes", dest="seed_episodes",
                    action="store_false", default=True)
    ap.add_argument("--horizon", type=int, default=3000)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    seeds = list(range(args.episodes))

    r_rew, r_len, r_ev = run_episodes(None, args.state, seeds, args.horizon, args.seed_episodes)
    model = PPO.load(args.model, device="cuda")
    p_rew, p_len, p_ev = run_episodes(model, args.state, seeds, args.horizon, args.seed_episodes)

    print("\nstate: %s   episodes: %d" % (os.path.basename(args.state), args.episodes))
    print("%-10s %10s %10s  %s" % ("", "mean_rew", "mean_len", "events"))
    print("%-10s %10.1f %10.0f  %s" % ("random", r_rew, r_len, r_ev or "NONE"))
    print("%-10s %10.1f %10.0f  %s" % ("policy", p_rew, p_len, p_ev or "NONE"))
    print("\ndelta reward %+.1f, delta length %+.0f frames" % (p_rew - r_rew, p_len - r_len))
    if p_rew <= r_rew:
        print("POLICY DOES NOT BEAT RANDOM -- the training curve is not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
