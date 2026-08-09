"""Are random play's 'key' events real pickups, or a room-load context switch?

measure_density says alttp_human-04 emits 7 key events over 12 random episodes,
which would make keys the first genuinely dense positive channel. Before any GPU
time is spent on that, it has to survive the obvious alternative explanation.

THE FAILURE MODE: 0x0234F is very plausibly a PER-DUNGEON small-key count (it
reads 255 in one non-dungeon capture, i.e. it is a sentinel, not a global). If
so, walking into an area with a different stored count would step the byte 0 ->
1 with no key ever collected, and a Counter cannot tell that from a real pickup.

DISCRIMINATOR: a chest/drop pickup happens INSIDE a room -- the screen barely
changes between the frame before and the frame after. A room load repaints
everything. So measure the mean absolute pixel delta across the key event and
compare it to that episode's typical step. Large delta = screen transition =
context switch, not a pickup.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-04.state"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=3000)
    args = ap.parse_args()

    import random
    from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS

    rows = []
    for seed in range(args.episodes):
        env = AlttpPpoEnv(horizon=args.horizon, state_path=args.state,
                          death_terminates=True)
        obs, _ = env.reset()
        rng = random.Random(seed)
        prev = np.asarray(obs, dtype=np.int16)
        deltas = []
        while True:
            obs, _r, term, trunc, info = env.step(rng.randrange(N_ACTIONS))
            cur = np.asarray(obs, dtype=np.int16)
            d = float(np.abs(cur - prev).mean())
            deltas.append(d)
            for tick, ch, dl, v in info["events"]:
                if ch in ("key", "key_used"):
                    rows.append([seed, tick, ch, dl, v, d, None])
            prev = cur
            if term or trunc:
                break
        typical = float(np.median(deltas)) if deltas else 0.0
        p95 = float(np.percentile(deltas, 95)) if deltas else 0.0
        for r in rows:
            if r[0] == seed and r[6] is None:
                r[6] = (typical, p95)
        env.close()

    if not rows:
        print("no key events -- nothing to verify")
        return 0
    print("%-5s %7s %-9s %6s %10s %10s %10s  %s"
          % ("seed", "tick", "channel", "value", "delta@evt", "median", "p95",
             "verdict"))
    real = susp = 0
    for seed, tick, ch, _dl, v, d, (typ, p95) in rows:
        # A pickup should look like an ordinary step; a room load is an outlier
        # against that episode's own distribution, which avoids hard-coding any
        # absolute pixel threshold.
        if d > max(p95, typ * 3.0):
            verdict, susp = "SUSPECT room load", susp + 1
        else:
            verdict, real = "pickup (in-room)", real + 1
        print("%-5d %7d %-9s %6d %10.2f %10.2f %10.2f  %s"
              % (seed, tick, ch, v, d, typ, p95, verdict))
    print("\nin-room pickups: %d   suspected room loads: %d" % (real, susp))
    print("VERDICT: %s" % ("keys are a REAL positive channel" if real > susp
                           else "key events look like CONTEXT SWITCHES -- do not train on this"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
