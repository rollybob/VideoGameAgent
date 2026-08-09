"""What must a key be WORTH for key-seeking to beat hiding?

a4 collected one key at 20k steps and none ever again, so keys are reachable and
were ABANDONED -- a balance problem, not an exploration problem. The obvious
response is "raise KEY_SCALE", and the obvious mistake is picking the number by
feel. This derives it.

THE ARGUMENT: early in training the agent plays near-randomly, so the gradient
toward key-seeking exists only if a key pays MORE than the damage a clumsy agent
eats while stumbling into one. Measure that cost directly under random play:

    break-even KEY_SCALE = (reward-units of damage+death incurred) / (keys got)

Below that number every step toward a key is punished more than the key pays,
and the only stable policy is the one a2, a3 and a4 all found: touch nothing.
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-04.state"))
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=3000)
    args = ap.parse_args()

    import random
    from alttp_ppo_env import (AlttpPpoEnv, N_ACTIONS, DEATH_PENALTY)

    keys = 0
    dmg_units = 0.0
    deaths = 0
    ev = collections.Counter()
    for seed in range(args.episodes):
        env = AlttpPpoEnv(horizon=args.horizon, state_path=args.state,
                          death_terminates=True)
        env.reset()
        rng = random.Random(seed)
        while True:
            _o, _r, term, trunc, info = env.step(rng.randrange(N_ACTIONS))
            for _t, ch, delta, _v in info["events"]:
                ev[ch] += 1
                if ch == "damage":
                    dmg_units += float(delta)          # negative
                elif ch == "death":
                    dmg_units += DEATH_PENALTY
                    deaths += 1
                elif ch == "key":
                    keys += int(delta)
            if term or trunc:
                break
        env.close()

    print("\nrandom play on %s, %d episodes" % (os.path.basename(args.state),
                                                args.episodes))
    print("  events: %s" % dict(ev))
    print("  keys collected      : %d" % keys)
    print("  damage+death reward : %.1f" % dmg_units)
    if not keys:
        print("\nNo keys under random play -- break-even is undefined, and the "
              "channel is not reachable from this state.")
        return 1
    breakeven = abs(dmg_units) / keys
    print("  cost per key        : %.1f reward units" % breakeven)
    print("\nBREAK-EVEN KEY_SCALE = %.0f" % breakeven)
    print("Below this, stumbling toward a key is punished more than the key "
          "pays, so 'touch nothing' dominates and there is no gradient to "
          "climb. Set the real scale ABOVE it, with margin, so the gradient is "
          "positive rather than merely non-negative.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
