"""Derive ENEMY_DMG_SCALE from a random-play break-even, before the seed-set.

Same method that fixed KEY_SCALE (2026-08-01): do NOT pick the coefficient by
feel. Measure, under random play in -04, how many reward-units of Link damage are
spent per point of enemy health removed. Any scale above that break-even makes
engaging net-positive even for a random flailer, so the gradient points at
attacking from step 1 -- which is the whole objective against the avoidance
attractor.

Also reports the DISCOUNTED break-even: PPO optimises discounted return, and the
08-01 lesson was that fixing a magnitude in undiscounted units leaves it below
the real bar. Enemy hits land during the fight, so they discount far less than
keys did, but it is measured here rather than assumed.

Prints the episode reward random play WOULD get at a few candidate scales, the
same sanity check that caught KEY_SCALE=30 being below break-even.
"""
import argparse
import collections
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GAMMA = 0.999
FRAME_SKIP = 4
DEATH_PEN = 20.0


def _run(job):
    target, seed, horizon = job
    import random
    from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS
    env = AlttpPpoEnv(horizon=horizon, death_terminates=True,
                      state_path=os.path.join(HERE, "states", target + ".state"))
    env.reset()
    rng = random.Random(seed)
    dmg_eighths = 0        # Link damage taken (reward-units, 1/eighth)
    deaths = 0
    enemy_hp_removed = 0    # sum of enemy_dmg deltas
    enemy_events = 0
    keys = 0
    first_hit_tick = None
    while True:
        _o, _r, term, trunc, info = env.step(rng.randrange(N_ACTIONS))
        for t, ch, d, _v in info["events"]:
            if ch == "damage":
                dmg_eighths += -d
            elif ch == "death":
                deaths += 1
            elif ch == "enemy_dmg":
                enemy_hp_removed += d
                enemy_events += 1
                if first_hit_tick is None:
                    first_hit_tick = t
            elif ch == "key":
                keys += d
        if term or trunc:
            break
    env.close()
    return (dmg_eighths, deaths, enemy_hp_removed, enemy_events, keys,
            first_hit_tick if first_hit_tick is not None else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="alttp_human-04")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs = [(args.target, s, args.horizon) for s in range(args.seeds)]
    agg = collections.Counter()
    hit_ticks = []
    with mp.Pool(args.workers) as pool:
        for dmg, deaths, ehp, eev, keys, fh in pool.imap_unordered(_run, jobs):
            agg["dmg"] += dmg; agg["deaths"] += deaths
            agg["ehp"] += ehp; agg["eev"] += eev; agg["keys"] += keys
            if fh:
                hit_ticks.append(fh)
    n = args.seeds
    link_cost = agg["dmg"] + DEATH_PEN * agg["deaths"]
    print("\n%s, %d random episodes:" % (args.target, n))
    print("  Link damage eighths %d, deaths %d  -> Link cost %.0f reward-units"
          % (agg["dmg"], agg["deaths"], link_cost))
    print("  enemy_dmg events %d, enemy health removed %d, kills(keys) %d"
          % (agg["eev"], agg["ehp"], agg["keys"]))
    print("  per episode: %.2f enemy hits, %.2f health removed, %.2f keys"
          % (agg["eev"] / n, agg["ehp"] / n, agg["keys"] / n))
    if agg["ehp"] == 0:
        print("\nenemy_dmg NEVER fired under random play -- can't derive a scale, "
              "and the channel would be too sparse. STOP.")
        return 1

    be = link_cost / agg["ehp"]
    mean_hit_step = (sum(hit_ticks) / len(hit_ticks) / FRAME_SKIP) if hit_ticks else 0
    disc = GAMMA ** mean_hit_step
    be_disc = be / disc if disc > 0 else be
    print("\n  UNDISCOUNTED break-even  = %.1f reward-units per enemy-HP" % be)
    print("  mean first-hit at ~%.0f agent steps -> discount %.3f (gamma %.3f)"
          % (mean_hit_step, disc, GAMMA))
    print("  DISCOUNTED break-even    = %.1f" % be_disc)
    print("  -> ~2x margin suggests ENEMY_DMG_SCALE ~ %.0f" % (2 * be_disc))

    print("\n  random-play episode reward at candidate scales (want clearly >0,")
    print("  the KEY_SCALE sanity check that engaging pays from step 1):")
    base = -(link_cost) / n + 120.0 * (agg["keys"] / n)  # existing reward sans enemy_dmg
    for s in (be_disc, 2 * be_disc, 4 * be_disc):
        print("    scale %5.0f -> mean_rew %+.1f" % (s, base + s * agg["ehp"] / n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
