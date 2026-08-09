"""Multi-head behavior cloning for the MultiDiscrete([9,2,2,2,2]) = [dir,A,B,L,R] action
space (2026-08-04). Replaces the single-head train_bc.py for ALttP: the net predicts a
9-way direction AND 4 independent buttons, so it can clone diagonal movement and
move-while-attacking -- the human behaviors the single-button clone could not reproduce.

Output = sum(nvec)=17 logits split into groups [9,2,2,2,2] (same structure SB3's
MultiCategorical uses for this action space, to ease the eventual PPO warm-start). Loss =
sum of per-component inverse-frequency-weighted cross-entropy. Metrics are per-component
(dir balanced-acc; per-button ON-recall), because raw accuracy is dominated by 'none'/off.

  docker run --rm --runtime nvidia -v ~/projects/VGA:/work -w /work thor-torch:cu130 \\
      python3 train/policy/train_bc_md.py --data train/policy/bc_data_alttp_md \\
      --out train/policy/ckpt_bc_md --epochs 40
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadPolicyNet(nn.Module):
    def __init__(self, nvec, in_ch=3):
        super().__init__()
        self.nvec = list(nvec)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, 2, 2), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(64, sum(self.nvec)),
        )

    def forward(self, x):
        return self.head(self.features(x))

    def split(self, logits):
        out, off = [], 0
        for n in self.nvec:
            out.append(logits[:, off:off + n]); off += n
        return out


def _load(p):
    d = np.load(p)
    X = torch.from_numpy(d["X"]).permute(0, 3, 1, 2).contiguous()  # uint8 (N,C,H,W); float per-batch
    y = torch.from_numpy(d["y"]).long()
    return X, y


def _forward_batched(model, X, dev, bs=256):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(X[i:i + bs].to(dev).float().div_(255.0)).cpu())
    return torch.cat(outs)


def _comp_metrics(logits_g, yg, n):
    pred = logits_g.argmax(1)
    recalls = {}
    for c in range(n):
        m = yg == c
        if m.any():
            recalls[c] = (pred[m] == c).float().mean().item()
    bal = float(np.mean(list(recalls.values()))) if recalls else 0.0
    top1 = (pred == yg).float().mean().item()
    return top1, bal, recalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    nvec = meta["nvec"]; comps = meta["components"]; dirnames = meta.get("dir_names")
    Xtr, ytr = _load(os.path.join(a.data, "train.npz"))
    Xva, yva = _load(os.path.join(a.data, "val.npz"))
    in_ch = Xtr.shape[1]
    print(f"train {tuple(Xtr.shape)} val {tuple(Xva.shape)} nvec={nvec} in_ch={in_ch}")

    # inverse-frequency class weights per component
    weights = []
    for g, nv in enumerate(nvec):
        cnt = torch.bincount(ytr[:, g], minlength=nv).float()
        w = torch.where(cnt > 0, cnt.sum() / (cnt * (cnt > 0).sum()), torch.zeros_like(cnt))
        weights.append(w.to(dev))

    model = MultiHeadPolicyNet(nvec, in_ch=in_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M on {dev}")
    # X stays uint8 on CPU (12-channel frame stacks are large); normalize per batch on GPU.
    n = len(Xtr); os.makedirs(a.out, exist_ok=True); best = -1.0

    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            xb = Xtr[idx].to(dev).float().div_(255.0)
            yb = ytr[idx].to(dev)
            opt.zero_grad()
            groups = model.split(model(xb))
            loss = sum(F.cross_entropy(groups[g], yb[:, g], weight=weights[g])
                       for g in range(len(nvec)))
            loss.backward(); opt.step(); tot += loss.item() * len(idx)

        model.eval()
        vg = model.split(_forward_batched(model, Xva, dev))
        comp_bal, msg = [], []
        for g, nm in enumerate(comps):
            _t, bal, rec = _comp_metrics(vg[g], yva[:, g], nvec[g])
            comp_bal.append(bal)
            if nvec[g] == 2:                       # button: report ON-recall (class 1)
                msg.append(f"{nm}:on={rec.get(1, float('nan')):.2f}")
            else:
                msg.append(f"{nm}:bal={bal:.2f}")
        score = float(np.mean(comp_bal))
        star = ""
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "nvec": nvec, "components": comps,
                        "dir_names": dirnames, "in_ch": in_ch, "img_h": meta["img_h"],
                        "img_w": meta["img_w"], "epoch": ep, "val_score": score},
                       os.path.join(a.out, "policy_md_best.pt"))
            star = " *"
        print(f"ep{ep:3d} loss={tot/n:.3f}  val[{' '.join(msg)}]  mean_bal={score:.3f}{star}", flush=True)

    # final per-component detail on val
    model.eval()
    vg = model.split(_forward_batched(model, Xva, dev))
    print("\nval detail:")
    for g, nm in enumerate(comps):
        t1, bal, rec = _comp_metrics(vg[g], yva[:, g], nvec[g])
        if nvec[g] == 2:
            print(f"  {nm:4} top1={t1:.3f} on-recall={rec.get(1, float('nan')):.3f} "
                  f"off-recall={rec.get(0, float('nan')):.3f} (support on={int((yva[:,g]==1).sum())})")
        else:
            names = dirnames or list(range(nvec[g]))
            per = " ".join(f"{names[c]}:{rec.get(c, 0):.2f}" for c in range(nvec[g]))
            print(f"  {nm:4} top1={t1:.3f} bal={bal:.3f}  recalls[{per}]")
    print(f"\nbest mean_bal = {best:.3f}  ->  {a.out}/policy_md_best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
