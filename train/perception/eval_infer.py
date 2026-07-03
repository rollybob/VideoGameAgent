#!/usr/bin/env python3
"""Container: run the trained mode classifier over a preprocessed npz, write preds.json
({frame_idx: {mode, conf}}). torch+numpy only.

    docker run --rm --runtime nvidia -v ~/projects/VGA:/work -w /work thor-torch:cu130 \
        python3 train/perception/eval_infer.py --model train/perception/mode_cnn.pt \
        --npz /tmp/live.npz --out /tmp/preds.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_mode import ModeCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="train/perception/mode_cnn.pt")
    ap.add_argument("--npz", default="/tmp/live.npz")
    ap.add_argument("--out", default="/tmp/preds.json")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    classes = ckpt["classes"]
    net = ModeCNN(len(classes)); net.load_state_dict(ckpt["state_dict"]); net.eval()
    d = np.load(args.npz)
    X = torch.from_numpy(d["X"]).float().permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        p = F.softmax(net(X), 1)
        conf, pred = p.max(1)
    out = {int(i): {"mode": classes[int(c)], "conf": round(float(cf), 3)}
           for i, c, cf in zip(d["idx"], pred, conf)}
    json.dump(out, open(args.out, "w"), indent=0)
    print("wrote", args.out, "n=", len(out))


if __name__ == "__main__":
    main()
