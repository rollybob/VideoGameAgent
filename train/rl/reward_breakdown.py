"""Per-channel reward breakdown: WHERE does the 12ch policy's episode reward come from?
The policy scores +755 with 0 keys on -04; this attributes that reward to channels so we
know which one it farms. Uses the env's OWN channel_reward (single source of truth) over
info["events"], so it matches the training reward exactly.

Also empirically checks Tim's gating question: new_room events/ep should ~= rooms visited/ep
(once per room), and explore events/ep should be bounded (once per fine cell), NOT growing
with episode length -- if they were farmable they'd be huge.
"""
import argparse, os, sys, collections
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from alttp_ppo_env import AlttpPpoEnv, channel_reward


def run_ep(env, act, seed):
    obs, _ = env.reset(seed=seed); done = False
    rew = collections.defaultdict(float); cnt = collections.Counter(); total = 0.0; keys = 0
    while not done:
        obs, r, term, trunc, info = env.step(act(obs)); total += r
        for e in info["events"]:
            ch, d = e[1], e[2]
            rew[ch] += channel_reward(ch, d); cnt[ch] += 1
            if ch == "key":
                keys += 1
        done = term or trunc
    return total, rew, cnt, len(env.oracle.rooms_seen), keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--state", default="alttp_human-04")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=15000)
    a = ap.parse_args()
    sp = os.path.join(HERE, "states", a.state + ".state")
    env = AlttpPpoEnv(state_path=sp, horizon=a.horizon)
    model = PPO.load(a.model, device="cuda")
    print("model=%s state=%s ep=%d horizon=%d" % (a.model, a.state, a.episodes, a.horizon))

    for label, act in [("random", lambda o: env.action_space.sample()),
                       ("policy-det", lambda o, m=model: m.predict(o, deterministic=True)[0])]:
        agg = collections.defaultdict(float); aggc = collections.Counter()
        tot = 0.0; rooms = 0.0; keys = 0.0
        for i in range(a.episodes):
            t, rew, cnt, rm, k = run_ep(env, act, i)
            tot += t; rooms += rm; keys += k
            for kk, v in rew.items(): agg[kk] += v
            for kk, v in cnt.items(): aggc[kk] += v
        n = a.episodes
        print("\n=== %-10s mean total %+8.1f/ep  | rooms %.1f  keys %.2f ===" %
              (label, tot / n, rooms / n, keys / n))
        for ch, v in sorted(agg.items(), key=lambda kv: -abs(kv[1])):
            print("  %-16s %+9.1f/ep  (%7.1f events/ep)" % (ch, v / n, aggc[ch] / n))
    env.close()


if __name__ == "__main__":
    sys.exit(main())
