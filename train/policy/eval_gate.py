"""
Offline confidence-gate analysis for the System-1 handoff (Phase 3 of RESCUE_PLAN).

Question this answers BEFORE any live wiring: if the BC policy acted whenever its
softmax confidence >= t (deferring to the VLM otherwise), what fraction of VLM calls
would it absorb (coverage), and how often would its action match what the VLM actually
chose on those steps (agreement)? The live gate ships only if some t gives high
agreement at useful coverage; otherwise the policy needs more data first.

Agreement-with-VLM is the right offline proxy here because the behavioral bar in
RESCUE_PLAN P3 is "R2 rate HOLDS with fewer VLM calls": on the cleared rung the VLM's
actions are, empirically, good enough to clear it, so a System 1 that reproduces them
where confident preserves the rung by construction. (It is NOT a claim the VLM is
optimal.)

Reads the npz written by build_dataset.py -- point --npz at a split the policy did NOT
train on (val.npz with --val-by-traj = whole held-out episodes). No image codec needed,
so it runs in the bare thor-torch container:
  docker run --rm --runtime nvidia -v ~/projects/VGA:/work -w /work thor-torch:cu130 \
      python3 train/policy/eval_gate.py --ckpt train/policy/ckpt_ladder/policy_best.pt \
      --npz train/policy/data_ladder/val.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Reuse the exact training-time model definition; a private copy would drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bc import SmallPolicyNet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True, help="held-out split from build_dataset.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = SmallPolicyNet(n_classes=len(ck["classes"])).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()

    d = np.load(args.npz)
    x = torch.from_numpy(d["X"]).float().permute(0, 3, 1, 2) / 255.0
    y = torch.from_numpy(d["y"]).long()

    with torch.no_grad():
        p = torch.softmax(model(x.to(args.device)), dim=1).cpu()
    conf, pred = p.max(dim=1)
    matches = (pred == y).numpy()
    confs = conf.numpy()

    print(f"steps evaluated: {len(confs)}  overall action-match: {matches.mean():.3f}")
    print(f"{'thresh':>6} {'coverage':>9} {'agreement':>10}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        take = confs >= t
        cov = take.mean()
        agr = matches[take].mean() if take.any() else float("nan")
        print(f"{t:>6.2f} {cov:>9.3f} {agr:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
