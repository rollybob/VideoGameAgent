"""Distill round 1: fine-tune the reflex S1 (SB3 PPO zip) on arbiter-success
trajectories by pure imitation. Spec: docs/DISTILL_SPEC_2026-08-06.md -- read it
before changing thresholds or knobs; they are pre-registered.

Recipe is bc_pretrain_sb3.py's (per-component inverse-frequency-weighted CE on the
5 MultiCategorical heads via policy.get_distribution) with three deltas:
  1. WARM-START: PPO.load(<scratch04 600k>) and fine-tune the policy IN PLACE,
     then model.save(<out.zip>) -- every existing eval path loads it unchanged.
  2. Data = distill_data/round1/*.npz episodes (raw frames + executed actions +
     source flag); 12ch stacks are rebuilt on the fly (frames[i-3..i], left-pad
     with the reset frame -- exactly the env's deque semantics), so nothing big
     ever sits on disk twice.
  3. Weighted step sampling: source==1 (S2 burst) steps x3 -- the seek signal
     lives there and is only ~10-30%% of steps.

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/rl thor-rl:cu130 python3 -u distill_bc.py \\
      --data distill_data/round1 --out runs/distill_r1/s1_distilled.zip
"""
import argparse
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stable_baselines3 import PPO
from alttp_ppo_env import AlttpPpoEnv

NVEC = [9, 2, 2, 2, 2]
COMPS = ["dir", "A", "B", "L", "R"]
STACK_K = 4


def load_episodes(data_dirs, limit=None):
    files = []
    for d_i, d in enumerate(data_dirs):
        files.extend((d_i, f) for f in sorted(glob.glob(os.path.join(d, "*.npz"))))
    if limit:
        files = files[:limit]
    eps = []
    for d_i, f in files:
        d = np.load(f)
        eps.append({"frames": d["frames"], "actions": d["actions"],
                    "source": d["source"], "name": os.path.basename(f), "dset": d_i})
    return eps


def build_index(eps):
    """Flat (ep_idx, step_idx) index + per-step sampling weights.
    source semantics: 0 = normal (1x), 1 = S2 burst (3x), 2 = context-only
    (weight 0 -- frames feed neighboring stacks but the action is never an
    imitation target; used for teacher walk-away/pre-roll steps)."""
    idx, w = [], []
    for e_i, e in enumerate(eps):
        dw = e.get("dset_weight", 1.0)
        for s_i in range(len(e["actions"])):
            idx.append((e_i, s_i))
            s = int(e["source"][s_i])
            w.append((3.0 if s == 1 else (0.0 if s == 2 else 1.0)) * dw)
    return np.array(idx, dtype=np.int64), np.array(w, dtype=np.float64)


def make_obs(eps, pairs):
    """(b,H,W,12) uint8 stacks rebuilt with the env's exact deque semantics."""
    out = []
    for e_i, s_i in pairs:
        fr = eps[e_i]["frames"]
        chan = [fr[max(0, s_i - k)] for k in range(STACK_K - 1, -1, -1)]
        out.append(np.concatenate(chan, axis=-1))
    return np.stack(out)


def batch_to_dev(x, dev):
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(dev)


def comp_bal(logits_g, yg, n):
    pred = logits_g.argmax(1)
    rec = {c: (pred[yg == c] == c).float().mean().item() for c in range(n) if (yg == c).any()}
    return float(np.mean(list(rec.values()))) if rec else 0.0, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[os.path.join(HERE, "distill_data", "round1")],
                    help="one or more episode dirs (round-2 passes round1 round2 for replay)")
    ap.add_argument("--data-weights", nargs="+", type=float, default=None,
                    help="per-dataset sampling weight multipliers, same order/length as "
                         "--data (variant D mixture-balancing knob)")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "alttp_s1v2_scratch04",
                                                   "ppo_alttp_600000_steps.zip"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "distill_r1",
                                                  "s1_distilled.zip"))
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-04.state"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-eps", type=int, default=None, help="smoke: cap episode count")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    eps = load_episodes(a.data, a.limit_eps)
    if len(eps) < 8:
        print("TOO FEW EPISODES (%d) -- refusing to train" % len(eps))
        return 2
    if a.data_weights:
        if len(a.data_weights) != len(a.data):
            print("--data-weights length mismatch")
            return 2
        for e in eps:
            e["dset_weight"] = a.data_weights[e["dset"]]
        print("dataset weights:", dict(zip(a.data, a.data_weights)), flush=True)
    # episode-level split so val frames never share an episode with train
    perm = rng.permutation(len(eps))
    n_val = max(2, int(len(eps) * a.val_frac))
    val_eps = [eps[i] for i in perm[:n_val]]
    tr_eps = [eps[i] for i in perm[n_val:]]
    tr_idx, tr_w = build_index(tr_eps)
    va_idx, va_w = build_index(val_eps)
    va_idx = va_idx[va_w > 0]        # context-only steps are not val targets either
    tr_mask = tr_w > 0
    tr_w /= tr_w.sum()
    n_steps = len(tr_idx)
    burst_frac = float(np.mean([s for e in tr_eps for s in e["source"]]))
    print("episodes train=%d val=%d | steps train=%d val=%d | burst-step frac=%.2f | dev=%s"
          % (len(tr_eps), len(val_eps), n_steps, len(va_idx), burst_frac, dev), flush=True)

    env = AlttpPpoEnv(state_path=a.state, horizon=200)
    model = PPO.load(a.ckpt, env=env, device=dev)
    policy = model.policy

    # inverse-frequency CE weights from the UNWEIGHTED train action distribution
    # (masked context-only steps excluded -- walk-away dirs would skew frequencies)
    ytr = np.stack([tr_eps[e]["actions"][s] for e, s in tr_idx[tr_mask]]).astype(np.int64)
    weights = []
    for g, nv in enumerate(NVEC):
        cnt = torch.bincount(torch.from_numpy(ytr[:, g]), minlength=nv).float()
        w = torch.where(cnt > 0, cnt.sum() / (cnt * (cnt > 0).sum()), torch.zeros_like(cnt))
        weights.append(w.to(dev))

    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=a.wd)
    best, best_sd = -1.0, None
    yva = np.stack([val_eps[e]["actions"][s] for e, s in va_idx]).astype(np.int64)
    yva_t = torch.from_numpy(yva)

    def val_logits():
        policy.set_training_mode(False)
        outs = [[] for _ in NVEC]
        with torch.no_grad():
            for i in range(0, len(va_idx), 128):
                xb = batch_to_dev(make_obs(val_eps, va_idx[i:i + 128]), dev)
                for g, s in enumerate(policy.get_distribution(xb).distribution):
                    outs[g].append(s.logits.cpu())
        return [torch.cat(o) for o in outs]

    for ep_n in range(1, a.epochs + 1):
        policy.set_training_mode(True)
        order = rng.choice(n_steps, size=n_steps, replace=True, p=tr_w)  # weighted, w/ replacement
        tot = 0.0
        for i in range(0, n_steps, a.bs):
            sel = tr_idx[order[i:i + a.bs]]
            xb = batch_to_dev(make_obs(tr_eps, sel), dev)
            yb = torch.from_numpy(np.stack([tr_eps[e]["actions"][s] for e, s in sel])
                                  .astype(np.int64)).to(dev)
            dist = policy.get_distribution(xb)
            loss = sum(F.cross_entropy(dist.distribution[g].logits, yb[:, g], weight=weights[g])
                       for g in range(len(NVEC)))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(sel)

        vl = val_logits()
        bals, msg = [], []
        for g, nm in enumerate(COMPS):
            bal, rec = comp_bal(vl[g], yva_t[:, g], NVEC[g])
            bals.append(bal)
            msg.append("%s:%s" % (nm, ("on=%.2f" % rec.get(1, float("nan"))) if NVEC[g] == 2
                                  else "bal=%.2f" % bal))
        score = float(np.mean(bals))
        star = ""
        if score > best:
            best = score
            best_sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            star = " *"
        print("ep%2d loss=%.3f val[%s] mean_bal=%.3f%s" % (ep_n, tot / n_steps,
                                                           " ".join(msg), score, star), flush=True)

    policy.load_state_dict(best_sd)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    model.save(a.out)
    print("best mean_bal=%.3f -> %s (full PPO zip, drop-in for arbiter_v1 --ckpt)" % (best, a.out),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
