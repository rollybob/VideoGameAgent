"""Measure reward-signal density under random play, before spending GPU hours.

The point: PPO cannot learn from a reward that never fires. Two separate causes
of "no signal" were found on 2026-07-31 -- a self-termination exploit (START ->
Quit) that faked all the events, and single-frame actions that left Link unable
to cross a room. Both are fixed; this script is the check that they stayed
fixed, and the gate on whether a given start-point is worth training.

Run it before any training launch. Cheap (CPU only) and parallel across cores.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga/train/rl \\
      thor-rl:cu130 python3 measure_density.py --seeds 12
"""
import argparse
import collections
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _run(job):
    kind, target, seed, horizon = job
    if kind == "alttp":
        from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS
        env = AlttpPpoEnv(horizon=horizon, death_terminates=True,
                          state_path=os.path.join(HERE, "states", target + ".state"))
        n_actions = N_ACTIONS
    else:
        from fs_ppo_env import FsPpoEnv, N_ACTIONS
        env = FsPpoEnv(horizon=horizon, checkpoint_name=target)
        n_actions = N_ACTIONS
    env.reset()
    rng = random.Random(seed)
    total = 0.0
    ev = collections.Counter()
    while True:
        _o, r, term, trunc, info = env.step(rng.randrange(n_actions))
        total += r
        for _t, ch, _d, _v in info["events"]:
            ev[ch] += 1
        if term or trunc:
            break
    length = info["tick"]
    env.close()
    return kind, target, total, length, dict(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs = []
    # 03/04/05 banked by Tim 2026-08-01: ledges with fall damage; a room locked
    # until its enemy dies (plus a chest); and a tough multi-hit enemy.
    for t in ("alttp_human-00", "alttp_human-01", "alttp_human-02",
              "alttp_human-03", "alttp_human-04", "alttp_human-05"):
        jobs += [("alttp", t, s, args.horizon) for s in range(args.seeds)]
    for b in ("taluscave", "deathmountain", "seaoftrees", "coop"):
        jobs += [("fs", b, s, args.horizon) for s in range(args.seeds)]

    agg = collections.defaultdict(lambda: {"rew": [], "len": [], "ev": collections.Counter()})
    with mp.Pool(args.workers) as pool:
        for kind, target, total, length, ev in pool.imap_unordered(_run, jobs):
            a = agg[(kind, target)]
            a["rew"].append(total)
            a["len"].append(length)
            a["ev"].update(ev)

    print("\n%-6s %-16s %9s %8s  %s" % ("ENV", "TARGET", "mean_rew", "mean_len", "events over all seeds"))
    for (kind, target), a in sorted(agg.items()):
        n = len(a["rew"])
        print("%-6s %-16s %9.1f %8.0f  %s" % (
            kind, target, sum(a["rew"]) / n, sum(a["len"]) / n, dict(a["ev"]) or "NONE"))
    print("\n%d seeds x %d-frame horizon each. A target with no events cannot "
          "train -- pick one that fires." % (args.seeds, args.horizon))
    return 0


if __name__ == "__main__":
    sys.exit(main())
