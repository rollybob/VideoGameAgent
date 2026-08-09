"""Behavioral eval of a BC-cloned policy: the test that actually matters for the
BC-bootstrap plan. Action-match accuracy (train_bc.py) is a proxy; what decides whether
the tape helps is whether a policy cloned from it EXPLORES/ACTS better than random when
dropped into the env -- rooms-visited, keys, reward vs the random floor, zero-shot.

Loads train_bc.py's SmallPolicyNet checkpoint, drives AlttpPpoEnv with it (argmax and
sampled), and compares to a random baseline on the same seeded episodes.

  docker run --rm --runtime nvidia -v ~/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130 \\
      python3 eval_bc_policy.py --ckpt /vga/train/policy/ckpt_bc_alttp/policy_best.pt \\
      --states alttp_ingame alttp_human-04 --episodes 6
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "policy"))

from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS
from train_bc import SmallPolicyNet


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    rooms = keys = 0
    total = 0.0
    done = False
    while not done:
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        total += r
        keys += sum(1 for e in info["events"] if e[1] == "key")
        done = term or trunc
    return {"rooms": len(env.oracle.rooms_seen), "keys": keys, "reward": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--states", nargs="+", default=["alttp_ingame", "alttp_human-04"])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    net = SmallPolicyNet(len(ck["classes"])).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()  # BatchNorm -> running stats; batch-size-1 inference in train mode is garbage
    print("loaded %s (val_bal=%.3f, classes=%d) on %s"
          % (os.path.basename(a.ckpt), ck.get("val_bal", float("nan")), len(ck["classes"]), dev))

    def bc(obs, deterministic):
        with torch.no_grad():
            x = torch.from_numpy(obs).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            logits = net(x)
            if deterministic:
                return int(logits.argmax(1))
            return int(torch.distributions.Categorical(logits=logits).sample())

    rng = np.random.default_rng(a.seed)
    policies = {
        "random":    lambda o: int(rng.integers(N_ACTIONS)),
        "bc-argmax": lambda o: bc(o, True),
        "bc-sample": lambda o: bc(o, False),
    }

    print("\n%-16s %-11s %7s %7s %9s" % ("state", "policy", "rooms", "keys", "reward"))
    for sname in a.states:
        spath = sname if os.path.isfile(sname) else os.path.join(HERE, "states", sname + ".state")
        env = AlttpPpoEnv(state_path=spath, horizon=a.horizon)
        for pname, pol in policies.items():
            res = [run_episode(env, pol, a.seed + 1000 * list(policies).index(pname) + i)
                   for i in range(a.episodes)]
            m = {k: np.mean([r[k] for r in res]) for k in ("rooms", "keys", "reward")}
            print("%-16s %-11s %7.2f %7.2f %+9.1f"
                  % (os.path.basename(spath).replace(".state", ""), pname,
                     m["rooms"], m["keys"], m["reward"]))
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
