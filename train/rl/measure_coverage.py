"""How dense is a POSITION-COVERAGE signal under random play in -04?

The 2026-08-02 diagnosis of the g11-g14 reliability failures: the collapse is not
an optimiser problem (clip_fraction/approx_kl do not distinguish the 2 successes
from the 2 failures -- if anything the successes take BIGGER update steps). It is
the dense-negative / sparse-positive bistability the 08-01 analysis predicted:
damage fires 3.8/ep, keys 0.45/ep, so avoidance is a deep attractor and whether a
seed escapes it is close to a coin flip.

The log's standing candidate fix is a DENSE positive signal from Link's position
(the only per-step signal). Before designing a reward around it, measure the
thing that makes or breaks the idea: under RANDOM play in -04, how many distinct
position-cells does an episode touch, at a few grid resolutions? A coverage bonus
is only worth building if random play already generates a rich, steadily-growing
set of visited cells -- otherwise there is no dense gradient to shape.

Reports, per grid shift: mean distinct cells/episode and the mean NEW-cell rate
per 100 steps over the first vs second half (does coverage keep arriving, or
saturate immediately?). Compared against the damage event count so the density is
read on the same scale as the negative channel it must counterbalance.
"""
import argparse
import collections
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SHIFTS = (3, 4, 5)  # 8-, 16-, 32-unit cells (room is 512 units = 1<<9)


def _run(job):
    target, seed, horizon = job
    from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS
    env = AlttpPpoEnv(horizon=horizon, death_terminates=True,
                      state_path=os.path.join(HERE, "states", target + ".state"))
    env.reset()
    rng = random.Random(seed)
    oracle = env.oracle
    seen = {s: set() for s in SHIFTS}
    # new-cell arrivals bucketed into first/second half of the episode, at the
    # finest grid, so "does coverage keep coming or saturate" is measurable
    new_first = new_second = 0
    steps = 0
    dmg = 0
    core = env.core
    while True:
        _o, r, term, trunc, info = env.step(rng.randrange(N_ACTIONS))
        x, y = oracle.read_pos(core)
        steps += 1
        for s in SHIFTS:
            cell = (x >> s, y >> s)
            if s == 3:
                if cell not in seen[s]:
                    if steps <= horizon // 2:
                        new_first += 1
                    else:
                        new_second += 1
            seen[s].add(cell)
        for _t, ch, _d, _v in info["events"]:
            if ch == "damage":
                dmg += 1
        if term or trunc:
            break
    env.close()
    return (target, {s: len(seen[s]) for s in SHIFTS},
            new_first, new_second, steps, dmg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="alttp_human-04")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs = [(args.target, s, args.horizon) for s in range(args.seeds)]
    cells = {s: [] for s in SHIFTS}
    nf, ns, st, dm = [], [], [], []
    with mp.Pool(args.workers) as pool:
        for _t, c, a, b, steps, dmg in pool.imap_unordered(_run, jobs):
            for s in SHIFTS:
                cells[s].append(c[s])
            nf.append(a); ns.append(b); st.append(steps); dm.append(dmg)

    n = len(st)
    print("\n%s, %d random episodes, mean length %.0f steps, mean damage %.1f/ep\n"
          % (args.target, n, sum(st) / n, sum(dm) / n))
    print("%-14s %14s" % ("grid (unit)", "cells/episode"))
    for s in SHIFTS:
        print("  >>%d  (%3d u)   %8.1f" % (s, 1 << s, sum(cells[s]) / n))
    print("\nfinest-grid NEW cells: first half %.1f/ep, second half %.1f/ep"
          % (sum(nf) / n, sum(ns) / n))
    half = args.horizon // 2
    print("  => new-cell RATE ~ %.2f/100steps (1st half) vs %.2f/100steps (2nd)"
          % (100 * sum(nf) / n / half, 100 * sum(ns) / n / half))
    print("\nread: a coverage bonus is dense enough to counter avoidance if "
          "cells/episode >> damage/ep AND new cells keep arriving in the 2nd half.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
