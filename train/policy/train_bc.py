"""
Behavior-clone a small reactive policy from logged VGA trajectories (Phase 2, System-1).

Reads the .npz built by build_dataset.py (frame -> chosen-button), trains a small conv net
(downscaled frame -> 11 button logits), and reports metrics that are honest under the heavy
class imbalance our seed data has (the VLM wandered "right" ~70% of the time):
- top-1 action-match accuracy (raw), AND the majority-class baseline it must beat;
- balanced accuracy (mean per-class recall) -- a policy that always predicts "right" scores
  high on raw acc but ~1/K on balanced acc, so this is the metric that exposes collapse;
- per-class recall so we can see WHICH actions it actually learned.

CrossEntropy is inverse-frequency weighted so the minority actions (A, start, up, ...) are
not drowned out. This is a PIPELINE smoke test first: on the tiny seed set expect the model
to overfit train and barely beat majority on val -- success here = "the loop learns", not a
deployable policy. Retrain once the trajectory set is larger/more diverse.

Container (thor-torch:cu130, has torch + numpy; no image codec needed -- npz is raw arrays):
  docker run --rm --runtime nvidia -v ~/projects/VGA:/work -w /work thor-torch:cu130 \
      python3 train/policy/train_bc.py --data train/policy/data --out train/policy/ckpt \
      --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallPolicyNet(nn.Module):
    """Compact conv classifier: frame -> button logits. ~0.3M params; System-1 must be
    cheap enough to run every frame later, so keep it small deliberately."""

    def __init__(self, n_classes: int, in_ch: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def _load(npz_path: str):
    d = np.load(npz_path)
    X = torch.from_numpy(d["X"]).float().div_(255.0).permute(0, 3, 1, 2).contiguous()  # NHWC->NCHW
    y = torch.from_numpy(d["y"]).long()
    return X, y


def _metrics(logits, y, n_classes):
    pred = logits.argmax(1)
    top1 = (pred == y).float().mean().item()
    # per-class recall (mean over classes that appear in y)
    recalls = []
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            recalls.append((pred[mask] == c).float().mean().item())
    bal = float(np.mean(recalls)) if recalls else 0.0
    return top1, bal, pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train.npz/val.npz/meta.json")
    ap.add_argument("--out", required=True, help="checkpoint output dir")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"
    if dev != args.device:
        print(f"WARN cuda unavailable, falling back to {dev}")

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    classes = meta["classes"]
    n_cls = len(classes)

    Xtr, ytr = _load(os.path.join(args.data, "train.npz"))
    Xva, yva = _load(os.path.join(args.data, "val.npz"))
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  classes={n_cls}")

    # Inverse-frequency class weights (only over classes present in train).
    counts = torch.bincount(ytr, minlength=n_cls).float()
    weights = torch.where(counts > 0, counts.sum() / (counts * (counts > 0).sum()), torch.zeros_like(counts))
    weights = weights.to(dev)

    # Majority baseline on val: always predict train's most frequent class.
    maj_cls = int(counts.argmax())
    maj_acc = (yva == maj_cls).float().mean().item() if len(yva) else float("nan")

    model = SmallPolicyNet(n_cls).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"params={n_params/1e6:.2f}M  majority='{classes[maj_cls]}' val_maj_acc={maj_acc:.3f}")

    Xtr, ytr = Xtr.to(dev), ytr.to(dev)
    Xva, yva = Xva.to(dev), yva.to(dev)
    n = len(Xtr)
    os.makedirs(args.out, exist_ok=True)
    best_bal = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx], weight=weights)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        tr_loss = tot / n

        model.eval()
        with torch.no_grad():
            tr1, tbal, _ = _metrics(model(Xtr), ytr, n_cls)
            if len(Xva):
                v1, vbal, vpred = _metrics(model(Xva), yva, n_cls)
            else:
                v1 = vbal = float("nan")
        msg = (f"ep{ep:3d} loss={tr_loss:.3f}  train[top1={tr1:.3f} bal={tbal:.3f}]  "
               f"val[top1={v1:.3f} bal={vbal:.3f}]  (maj={maj_acc:.3f})")

        # Checkpoint on best val balanced accuracy (raw top1 rewards collapse-to-majority).
        if len(Xva) and vbal > best_bal:
            best_bal = vbal
            torch.save({"model": model.state_dict(), "classes": classes,
                        "img_h": meta["img_h"], "img_w": meta["img_w"],
                        "epoch": ep, "val_bal": vbal, "val_top1": v1},
                       os.path.join(args.out, "policy_best.pt"))
            msg += "  *"
        print(msg, flush=True)

    # Final per-class recall on val for the record.
    if len(Xva):
        model.eval()
        with torch.no_grad():
            vpred = model(Xva).argmax(1)
        print("\nval per-class recall (support):")
        for c in range(n_cls):
            m = yva == c
            if m.any():
                r = (vpred[m] == c).float().mean().item()
                print(f"  {classes[c]:7} {r:.3f}  (n={int(m.sum())})")
    print(f"\nbest val balanced acc = {best_bal:.3f}  ->  {args.out}/policy_best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
