#!/usr/bin/env python3
"""Train the FFTA mode classifier (field/menu/dialog) in the thor-torch container.
Reads the npz built by build_mode_dataset.py; torch+numpy only (no cv2/PIL).

    docker run --rm --runtime nvidia -v ~/projects/VGA:/work -w /work thor-torch:cu130 \
        python3 train/perception/train_mode.py --ds train/perception/ds \
        --out train/perception/mode_cnn.pt --epochs 80

Small data (~180 frames) from a narrow scripted slice, so: small net + dropout + weight
decay + brightness/shift augmentation, and we treat val accuracy as necessary-not-sufficient
- the real test is generalization to LIVE trajectory frames (eval_mode.py), which this data
does not fully cover. NO horizontal flip: FFTA menus are left-anchored; a flip is a lie.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModeCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm2d(16), nn.BatchNorm2d(32), nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.c3(x))), 2)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.drop(x))


def augment(x: torch.Tensor) -> torch.Tensor:
    """Brightness jitter + small random shift (reflect-pad then crop). x: (N,3,H,W) in [0,1]."""
    n, _, h, w = x.shape
    x = x * (0.8 + 0.4 * torch.rand(n, 1, 1, 1, device=x.device))  # brightness 0.8-1.2
    x = torch.clamp(x, 0, 1)
    pad = 6
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    out = torch.empty_like(x)
    for i in range(n):
        dy, dx = int(torch.randint(0, 2 * pad + 1, (1,))), int(torch.randint(0, 2 * pad + 1, (1,)))
        out[i] = xp[i, :, dy:dy + h, dx:dx + w]
    return out


def load_npz(path):
    d = np.load(path)
    X = torch.from_numpy(d["X"]).float().permute(0, 3, 1, 2) / 255.0  # NHWC uint8 -> NCHW [0,1]
    y = torch.from_numpy(d["y"]).long()
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="train/perception/ds")
    ap.add_argument("--out", default="train/perception/mode_cnn.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(args.ds, "meta.json")))
    classes = meta["classes"]
    Xtr, ytr = load_npz(os.path.join(args.ds, "train.npz"))
    Xva, yva = load_npz(os.path.join(args.ds, "val.npz"))
    Xtr, ytr, Xva, yva = Xtr.to(dev), ytr.to(dev), Xva.to(dev), yva.to(dev)

    # Class weights to offset imbalance (menu over-represented).
    counts = torch.bincount(ytr, minlength=len(classes)).float()
    w = (counts.sum() / (len(classes) * counts.clamp(min=1))).to(dev)
    net = ModeCNN(len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = 0.0
    bs = 32
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = augment(Xtr[idx])
            opt.zero_grad()
            loss = F.cross_entropy(net(xb), ytr[idx], weight=w)
            loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pred = net(Xva).argmax(1)
            acc = (pred == yva).float().mean().item()
        if acc >= best:
            best = acc
            torch.save({"state_dict": net.state_dict(), "classes": classes,
                        "img_h": meta["img_h"], "img_w": meta["img_w"]}, args.out)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d} loss {loss.item():.3f} val_acc {acc:.3f} (best {best:.3f})", flush=True)

    # Final per-class report from the best model.
    ckpt = torch.load(args.out, map_location=dev)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    with torch.no_grad():
        pred = net(Xva).argmax(1).cpu().numpy()
    yv = yva.cpu().numpy()
    print("=== best val_acc %.3f ===" % best)
    for c, name in enumerate(classes):
        m = yv == c
        if m.sum():
            print(f"  {name:7} recall {(pred[m] == c).mean():.3f}  (n={int(m.sum())})")
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
