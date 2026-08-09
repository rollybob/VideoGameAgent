"""Stage 2a: behavior-clone SB3's ActorCriticCnnPolicy on the human-tape dataset, so PPO
can WARM-START from it (the MultiHeadPolicyNet from train_bc_md.py is not an SB3 policy, so
a weight copy is impossible -- this trains the actual SB3 policy net instead).

Reuses the recipe that made BC work: per-component inverse-frequency-weighted cross-entropy
on the 5 MultiCategorical heads [dir,A,B,L,R], pulled from policy.get_distribution(). SB3's
default NatureCNN is ~1-2M params (vs the 0.07M MultiHeadPolicyNet), which also relieves the
B-vs-direction capacity contention seen earlier.

Saves the POLICY state_dict only (not the whole PPO model): the RL run builds a fresh PPO
with the full project config and does policy.load_state_dict(...), so pretrain and RL
hyperparameters stay decoupled. Both sides are 'CnnPolicy' on the same env, so keys match.

  docker run --rm --runtime nvidia -v ~/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130 \\
      python3 bc_pretrain_sb3.py --data /vga/train/policy/bc_data_alttp_md \\
      --out /vga/train/policy/bc_sb3_policy.pt --epochs 30
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warnings
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
from alttp_ppo_env import AlttpPpoEnv


def _comp_bal(logits_g, yg, n):
    pred = logits_g.argmax(1)
    rec = {c: (pred[yg == c] == c).float().mean().item() for c in range(n) if (yg == c).any()}
    return float(np.mean(list(rec.values()))) if rec else 0.0, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="path to save policy state_dict (.pt)")
    ap.add_argument("--state", default="states/alttp_human-04.state", help="any state, just to build the env/spaces")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    env = AlttpPpoEnv(state_path=a.state, horizon=200)
    model = PPO("CnnPolicy", env, n_steps=64, device=dev, seed=a.seed)  # n_steps small: no RL here, just the policy
    policy = model.policy
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    nvec, comps = meta["nvec"], meta["components"]

    dtr = np.load(os.path.join(a.data, "train.npz")); Xtr, ytr = dtr["X"], dtr["y"]
    dva = np.load(os.path.join(a.data, "val.npz")); Xva, yva = dva["X"], dva["y"]
    print(f"train {Xtr.shape} val {Xva.shape} nvec={nvec} "
          f"policy_params={sum(p.numel() for p in policy.parameters())/1e6:.2f}M on {dev}", flush=True)

    ytr_t = torch.from_numpy(ytr).long()
    weights = []
    for g, nv in enumerate(nvec):
        cnt = torch.bincount(ytr_t[:, g], minlength=nv).float()
        w = torch.where(cnt > 0, cnt.sum() / (cnt * (cnt > 0).sum()), torch.zeros_like(cnt))
        weights.append(w.to(dev))

    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=a.wd)
    n = len(Xtr); best = -1.0
    yva_t = torch.from_numpy(yva).long()

    def obs_batch(X, idx):
        return torch.from_numpy(X[idx]).permute(0, 3, 1, 2).contiguous().to(dev)  # (b,C,H,W) uint8

    def val_logits():
        policy.set_training_mode(False)
        outs = [[] for _ in nvec]
        with torch.no_grad():
            for i in range(0, len(Xva), 256):
                idx = np.arange(i, min(i + 256, len(Xva)))
                for g, s in enumerate(policy.get_distribution(obs_batch(Xva, idx)).distribution):
                    outs[g].append(s.logits.cpu())
        return [torch.cat(o) for o in outs]

    for ep in range(1, a.epochs + 1):
        policy.set_training_mode(True)
        perm = np.random.permutation(n); tot = 0.0
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            yb = torch.from_numpy(ytr[idx]).long().to(dev)
            dist = policy.get_distribution(obs_batch(Xtr, idx))
            loss = sum(F.cross_entropy(dist.distribution[g].logits, yb[:, g], weight=weights[g])
                       for g in range(len(nvec)))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)

        vl = val_logits(); comp_bal, msg = [], []
        for g, nm in enumerate(comps):
            bal, rec = _comp_bal(vl[g], yva_t[:, g], nvec[g]); comp_bal.append(bal)
            msg.append(f"{nm}:{'on=%.2f' % rec.get(1, float('nan')) if nvec[g] == 2 else 'bal=%.2f' % bal}")
        score = float(np.mean(comp_bal)); star = ""
        if score > best:
            best = score
            torch.save(policy.state_dict(), a.out); star = " *"
        print(f"ep{ep:3d} loss={tot/n:.3f} val[{' '.join(msg)}] mean_bal={score:.3f}{star}", flush=True)

    print(f"best mean_bal={best:.3f} -> {a.out} (SB3 policy state_dict, warm-start for PPO)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
