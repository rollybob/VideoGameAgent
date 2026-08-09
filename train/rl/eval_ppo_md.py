"""Eval SB3 PPO models (MultiDiscrete + frame-stacked env) head-to-head on given states:
rooms-visited / keys / reward, sampled AND deterministic, vs a multi-input random baseline.
Used for the Stage-2 warm-start-vs-control comparison.

  docker run --rm --runtime nvidia -v ~/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130 \\
      python3 eval_ppo_md.py --models warm=runs/alttp_ws1/ppo_alttp_final.zip \\
      control=runs/alttp_ctrl1/ppo_alttp_final.zip --states alttp_ingame alttp_human-04
"""
import argparse, os, sys
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from alttp_ppo_env import AlttpPpoEnv


def run_ep(env, act_fn, seed):
    obs, _ = env.reset(seed=seed); keys = 0; total = 0.0; done = False
    while not done:
        obs, r, term, trunc, info = env.step(act_fn(obs)); total += r
        keys += sum(1 for e in info["events"] if e[1] == "key"); done = term or trunc
    return len(env.oracle.rooms_seen), keys, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label=path.zip ...")
    ap.add_argument("--states", nargs="+", default=["alttp_ingame", "alttp_human-04"])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    models = {}
    for spec in a.models:
        lab, p = spec.split("=", 1)
        models[lab] = PPO.load(p, device="cuda")
    print("loaded:", ", ".join(models))
    print("\n%-16s %-14s %7s %7s %9s" % ("state", "policy", "rooms", "keys", "reward"))
    for sname in a.states:
        spath = sname if os.path.isfile(sname) else os.path.join(HERE, "states", sname + ".state")
        env = AlttpPpoEnv(state_path=spath, horizon=a.horizon); env.action_space.seed(a.seed)
        sn = os.path.basename(spath).replace(".state", "")

        def report(label, fn):
            res = [run_ep(env, fn, a.seed + i) for i in range(a.episodes)]
            m = np.mean(res, axis=0)
            print("%-16s %-14s %7.2f %7.2f %+9.1f" % (sn, label, m[0], m[1], m[2]))

        report("random", lambda o: env.action_space.sample())
        for lab, model in models.items():
            report(lab + "-samp", lambda o, m=model: m.predict(o, deterministic=False)[0])
            report(lab + "-det", lambda o, m=model: m.predict(o, deterministic=True)[0])
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
