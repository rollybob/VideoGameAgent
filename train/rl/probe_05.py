"""Probe a room under RANDOM multi-input play: does Link die fast or just avoid, and can
the agent damage the enemy AT ALL? Settles the -05 mini-boss question -- if random play
never lands an enemy_dmg over thousands of frames, the boss is effectively un-damageable
by the agent (boomerang not reachable via the mapped buttons, or immune without a stun),
which would make -05 unwinnable BY DESIGN rather than a policy failure.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from alttp_ppo_env import AlttpPpoEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=["alttp_human-05"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=6000)
    a = ap.parse_args()
    HERE = os.path.dirname(os.path.abspath(__file__))
    for sname in a.states:
        env = AlttpPpoEnv(state_path=os.path.join(HERE, "states", sname + ".state"),
                          horizon=a.horizon, death_terminates=True)
        env.action_space.seed(0)
        surv, edmg, dmg, died = [], [], [], 0
        for ep in range(a.episodes):
            env.reset(seed=ep)
            e = d = 0; term = trunc = False
            while not (term or trunc):
                _o, _r, term, trunc, info = env.step(env.action_space.sample())
                for ev in info["events"]:
                    if ev[1] == "enemy_dmg": e += 1
                    elif ev[1] == "damage": d += 1
            surv.append(env._tick); edmg.append(e); dmg.append(d); died += int(term)
        print("%-16s died %d/%d | mean survival %5.0f/%d frames | enemy_dmg/ep %4.1f | dmg-taken/ep %4.1f"
              % (sname, died, a.episodes, np.mean(surv), a.horizon, np.mean(edmg), np.mean(dmg)), flush=True)


if __name__ == "__main__":
    main()
