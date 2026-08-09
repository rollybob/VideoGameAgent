"""Is the 'heal' channel a real heart pickup, or a damage-animation artifact?

The 2026-08-01 address correction made 'heal' fire for the first time (5 events
for a3), and that matters a lot: "no positive channel has ever fired" was the
stated gate on launching another ALttP training run. So it is worth more than a
counter before it is believed.

THE FAILURE MODE THIS LOOKS FOR: if 0x0234D dips transiently during a damage
animation and then settles back up, the oracle sees down-then-up and emits
damage followed by a fake 'heal'. That would look identical in a Counter.

DISCRIMINATOR: a genuine pickup is isolated in time and moves health UP toward
max from below. An artifact rides immediately behind a damage event -- within
the invulnerability/animation window -- and typically restores exactly what was
just lost. So log every health change with its tick and classify by the gap to
the preceding damage event.
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Frames after a damage event within which an "up" is suspicious. ALttP's
# invulnerability flash is roughly 1s; anything settling back inside that is far
# more likely animation than the player finding and walking over a heart.
ARTIFACT_WINDOW = 90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-00.state"))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=3000)
    args = ap.parse_args()

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from alttp_ppo_env import AlttpPpoEnv

    model = PPO.load(args.model, device="cuda")
    verdicts = collections.Counter()
    rows = []

    for seed in range(args.episodes):
        env = AlttpPpoEnv(horizon=args.horizon, state_path=args.state,
                          death_terminates=True)
        obs, _ = env.reset()
        torch.manual_seed(seed)
        np.random.seed(seed)
        last_damage_tick = None
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, _r, term, trunc, info = env.step(int(action))
            for tick, ch, delta, value in info["events"]:
                if ch == "damage":
                    last_damage_tick = tick
                elif ch == "heal":
                    gap = None if last_damage_tick is None else tick - last_damage_tick
                    if gap is not None and gap <= ARTIFACT_WINDOW:
                        verdict = "SUSPECT (rides a damage event)"
                    else:
                        verdict = "GENUINE (isolated)"
                    verdicts[verdict] += 1
                    rows.append((seed, tick, delta, value, gap, verdict))
            if term or trunc:
                break
        env.close()

    print("\nmodel: %s" % os.path.basename(args.model))
    print("state: %s   episodes: %d   artifact window: %d frames\n"
          % (os.path.basename(args.state), args.episodes, ARTIFACT_WINDOW))
    if not rows:
        print("NO heal events at all -- the positive channel did not fire here.")
        return 0
    print("%-5s %8s %7s %7s %8s  %s"
          % ("seed", "tick", "delta", "hp_now", "gap", "verdict"))
    for seed, tick, delta, value, gap, verdict in rows:
        print("%-5d %8d %+7d %7d %8s  %s"
              % (seed, tick, delta, value, "-" if gap is None else gap, verdict))
    print("")
    for v, n in verdicts.most_common():
        print("  %-32s %d" % (v, n))
    genuine = sum(n for v, n in verdicts.items() if v.startswith("GENUINE"))
    per_ep = genuine / float(args.episodes)
    print("\ngenuine heals per episode: %.2f  (damage fires ~4/episode under "
          "random play in this room)" % per_ep)
    # TWO separate questions, and conflating them is how this project has
    # repeatedly mistaken an artifact for progress. "Did it ever fire" is not
    # "is it trainable" -- a channel firing once in 12 episodes is a curiosity,
    # not a learning signal, so the bar here is a RATE and not a boolean.
    if not genuine:
        verdict = "ARTIFACT -- every heal rides a damage event"
    elif per_ep < 0.5:
        verdict = ("REAL BUT TOO THIN TO TRAIN ON (%d genuine in %d episodes). "
                   "The channel exists; it is not yet a usable reward."
                   % (genuine, args.episodes))
    else:
        verdict = "REAL and dense enough to shape behaviour"
    print("VERDICT: %s" % verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
