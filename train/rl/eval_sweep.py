"""Checkpoint-SWEEP eval for the engagement seed-set -- the pre-registered metric.

The a5 lesson: never judge a run on final.zip. An unstable run's final policy can
be its worst, so evaluate a spread of checkpoints and take the best. Success (set
before launch) = a seed's BEST checkpoint beats the random baseline AND registers
KEY events in eval (a2/a3/a4 all beat random while collecting ZERO keys -- that is
the hollow win this bar exists to reject).

Behaviour (key events) is reward-agnostic, so this is directly comparable to the
g11-g14 baseline (2/4 succeeded) even though e11-e14 train on a different reward.
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_policy import run_episodes  # noqa: E402


def ckpt_steps(rundir):
    steps = []
    for f in glob.glob(os.path.join(rundir, "ppo_alttp_*_steps.zip")):
        m = re.search(r"_(\d+)_steps", f)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def pick(steps, k):
    if not steps:
        return []
    if k >= len(steps):
        return steps
    if k <= 1:
        return [steps[-1]]
    idx = sorted(set(int(i * (len(steps) - 1) / (k - 1)) for i in range(k)))
    return [steps[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/alttp_e11", "runs/alttp_e12",
                                                  "runs/alttp_e13", "runs/alttp_e14"])
    ap.add_argument("--state", default="alttp_human-04")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--ckpts", type=int, default=6, help="checkpoints swept per run")
    # kill+key happen by ~frame 600; 1500 captures them with margin and is applied
    # to random AND policy equally, so it is a fair comparison. Full 3000 mostly
    # measures post-kill survival and triples eval time on surviving policies.
    ap.add_argument("--horizon", type=int, default=1500)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    state_path = os.path.join(HERE, "states", args.state + ".state")
    seeds = list(range(args.episodes))

    # run_episodes returns (mean_rew, mean_len, dict(events)) -- already averaged
    rand_rew, _rl, rev = run_episodes(None, state_path, seeds, args.horizon)
    print("RANDOM baseline (%d eps): mean_rew %.1f  keys %d  %s\n"
          % (args.episodes, rand_rew, rev.get("key", 0), dict(rev)))

    summary = []
    for run in args.runs:
        steps = ckpt_steps(run)
        targets = pick(steps, args.ckpts)
        rows = []
        print("=== %s (%d ckpts, sweeping %s) ===" % (
            os.path.basename(run), len(steps), targets))
        for st in targets + ["final"]:
            path = (os.path.join(run, "ppo_alttp_final.zip") if st == "final"
                    else os.path.join(run, "ppo_alttp_%d_steps.zip" % st))
            if not os.path.exists(path):
                continue
            policy = PPO.load(path, device="cpu")
            mr, _pl, pev = run_episodes(policy, state_path, seeds, args.horizon)
            keys = pev.get("key", 0)
            rows.append((mr, keys, st, dict(pev)))
            print("  %-8s mean_rew %+7.1f  keys %2d  %s"
                  % (st, mr, keys, dict(pev)))
        # pre-registered success: BEST checkpoint beats random AND has key events
        best = max(rows, key=lambda r: r[0]) if rows else None
        best_with_keys = max((r for r in rows if r[1] > 0), key=lambda r: r[0], default=None)
        ok = best_with_keys is not None and best_with_keys[0] > rand_rew
        verdict = "SUCCESS" if ok else "fail"
        detail = ("best-with-keys %+.1f @%s (keys %d)" % (
            best_with_keys[0], best_with_keys[2], best_with_keys[1])
            if best_with_keys else "NO checkpoint collected a key")
        print("  -> %s: %s\n" % (verdict, detail))
        summary.append((os.path.basename(run), verdict, detail))

    n_ok = sum(1 for _r, v, _d in summary if v == "SUCCESS")
    print("=" * 60)
    print("RELIABILITY: %d/%d seeds succeed (baseline g11-g14 = 2/4)" % (n_ok, len(summary)))
    for name, v, d in summary:
        print("  %-16s %-8s %s" % (name, v, d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
