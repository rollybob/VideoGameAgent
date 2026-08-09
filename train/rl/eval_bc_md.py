"""Behavioral eval of the multi-head (MultiDiscrete) BC policy -- the go/no-go for whether
the multi-input action space rescues the BC bootstrap. Drives AlttpPpoEnv with the cloned
[dir,A,B,L,R] policy (argmax + sampled) and compares to a MULTI-INPUT random baseline
(action_space.sample(), the matched baseline) plus single-button random for reference.

  docker run --rm --runtime nvidia -v ~/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130 \\
      python3 eval_bc_md.py --ckpt /vga/train/policy/ckpt_bc_md/policy_md_best.pt \\
      --states alttp_ingame alttp_human-04 --episodes 6
"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "policy"))
from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS
from train_bc_md import MultiHeadPolicyNet


def run_ep(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    keys = 0; total = 0.0; done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
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
    net = MultiHeadPolicyNet(ck["nvec"], in_ch=ck.get("in_ch", 3)).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    print("loaded %s (val_score=%.3f) nvec=%s on %s"
          % (os.path.basename(a.ckpt), ck.get("val_score", float("nan")), ck["nvec"], dev))

    def bc(obs, deterministic):
        with torch.no_grad():
            x = torch.from_numpy(obs).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            groups = net.split(net(x))
            if deterministic:
                return np.array([int(g.argmax(1)) for g in groups])
            return np.array([int(torch.distributions.Categorical(logits=g).sample()) for g in groups])

    rng = np.random.default_rng(a.seed)
    policies = {
        "rand-md":   lambda o, env: env.action_space.sample(),
        "rand-1btn": lambda o, env: int(rng.integers(N_ACTIONS)),
        "bc-argmax": lambda o, env: bc(o, True),
        "bc-sample": lambda o, env: bc(o, False),
    }

    print("\n%-16s %-11s %7s %7s %9s" % ("state", "policy", "rooms", "keys", "reward"))
    for sname in a.states:
        spath = sname if os.path.isfile(sname) else os.path.join(HERE, "states", sname + ".state")
        env = AlttpPpoEnv(state_path=spath, horizon=a.horizon)
        env.action_space.seed(a.seed)
        for pi, (pname, pol) in enumerate(policies.items()):
            res = [run_ep(env, (lambda o, p=pol: p(o, env)), a.seed + 1000 * pi + i)
                   for i in range(a.episodes)]
            m = {k: np.mean([r[k] for r in res]) for k in ("rooms", "keys", "reward")}
            print("%-16s %-11s %7.2f %7.2f %+9.1f"
                  % (os.path.basename(spath).replace(".state", ""), pname,
                     m["rooms"], m["keys"], m["reward"]))
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
