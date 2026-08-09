"""Walker eval: PRE-REGISTERED (2026-08-09, set before results were seen).

Arms, per state, EP episodes each of 600 decisions (2400 frames) in WalkerEnv:
  policy  -- the trained goal-conditioned walker checkpoint
  greedy  -- hold the direction of sign(goal - link) each decision. This is the
             honest incumbent: it is exactly what today's drive agent does with
             a VLM direction, so BEATING GREEDY is the capability gain. Greedy
             has ORACLE access to coordinates (the policy only sees the blob),
             so this is a handicap match in greedy's favor.
  random  -- uniform Discrete(9). The floor.

Metrics per (arm, state): reach1 = fraction of episodes with >=1 target reached;
reaches_mean = mean targets reached per episode (in-episode resample makes this
the density number); first_reach_med = median decisions to the first reach.

SUCCESS BAR (docs/-free, recorded here + session log): on HELD-OUT states
{02,04,11,12} pooled, policy reach1 >= 0.70 AND policy reaches_mean clearly
above greedy (no formal test pre-set; report the paired per-state numbers).

Run one (arm, state) per process (parallel via xargs in run_eval_walker.sh):
  docker run --rm -v ~/projects/VGA:/work -w /work thor-rl:cu130 \
    python3 train/rl/eval_walker.py --arm greedy --state alttp_human-02 --episodes 24
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from walker_env import WalkerEnv

EP_DECISIONS = 600


def greedy_action(env):
    lx, ly = env._link()
    gx, gy = env.goal
    dx, dy = gx - lx, gy - ly
    sx = int(np.sign(dx)) if abs(dx) > 4 else 0
    sy = int(np.sign(dy)) if abs(dy) > 4 else 0
    return {(0, 0): 0, (0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4,
            (1, -1): 5, (-1, -1): 6, (1, 1): 7, (-1, 1): 8}[(sx, sy)]


def run(arm, state, episodes, ckpt, seed):
    env = WalkerEnv(state_path=os.path.join(HERE, "states", state + ".state"),
                    horizon=EP_DECISIONS * 4)
    model = None
    if arm == "policy":
        from stable_baselines3 import PPO
        model = PPO.load(ckpt, device="cpu")   # tiny net; GPU stays free for training
    rng = np.random.default_rng(seed)
    eps = []
    for e in range(episodes):
        obs, info = env.reset(seed=seed + e * 977)
        reaches0 = 0
        first = None
        for t in range(EP_DECISIONS):
            if arm == "policy":
                a, _ = model.predict(obs, deterministic=False)
                a = int(a)
            elif arm == "greedy":
                a = greedy_action(env)
            else:
                a = int(rng.integers(9))
            obs, r, term, trunc, info = env.step(a)
            if info["reaches"] > reaches0:
                reaches0 = info["reaches"]
                if first is None:
                    first = t + 1
            if term or trunc:
                break
        eps.append({"reaches": reaches0, "first": first, "steps": t + 1})
    env.close()
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["policy", "greedy", "random"], required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "walker1", "ppo_walker_final.zip"))
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    eps = run(a.arm, a.state, a.episodes, a.ckpt, a.seed)
    reach1 = float(np.mean([1.0 if e["reaches"] >= 1 else 0.0 for e in eps]))
    rmean = float(np.mean([e["reaches"] for e in eps]))
    firsts = [e["first"] for e in eps if e["first"] is not None]
    fmed = float(np.median(firsts)) if firsts else None
    res = {"arm": a.arm, "state": a.state, "episodes": a.episodes, "seed": a.seed,
           "reach1": reach1, "reaches_mean": rmean, "first_reach_med": fmed, "eps": eps}
    out = a.out or os.path.join(HERE, "walker_eval_%s_%s.json" % (a.arm, a.state))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("%s %s: reach1=%.2f reaches_mean=%.2f first_med=%s" %
          (a.arm, a.state, reach1, rmean, fmed), flush=True)


if __name__ == "__main__":
    main()
