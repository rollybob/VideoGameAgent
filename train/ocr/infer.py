"""Qualitative inference for an F1 checkpoint (container side): print ground-truth
vs predicted strings for a few val samples, so plateaus can be diagnosed (e.g.
confusable-glyph errors vs systematic failure). Also the eventual inference entry
point the Pokemon plugin will call to read a text region."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CRNN, WIDTH_DOWNSAMPLE  # noqa: E402


def greedy(am, blank=0):
    out, prev = [], -1
    for a in am:
        a = int(a)
        if a != prev and a != blank:
            out.append(a)
        prev = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)  # our own trusted ckpt
    meta = ck["meta"]
    chars = meta["chars"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CRNN(meta["n_classes"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()

    d = np.load(os.path.join(a.data, "val.npz"))
    imgs, widths, flat, lengths = d["images"], d["widths"], d["labels"], d["label_lengths"]
    labels, off = [], 0
    for n in lengths:
        labels.append(flat[off:off + int(n)])
        off += int(n)

    def to_str(ids):
        return "".join(chars[i - 1] for i in ids if 1 <= i <= len(chars))

    print(f"ckpt epoch={ck.get('epoch')} val_CER={ck.get('val_cer'):.4f} exact={ck.get('val_exact'):.3f}")
    T = imgs.shape[2] // WIDTH_DOWNSAMPLE
    with torch.no_grad():
        for k in range(min(a.n, len(labels))):
            x = torch.from_numpy(imgs[k:k + 1]).float().div_(255.0).unsqueeze(1).to(dev)
            il = min(int(widths[k]) // WIDTH_DOWNSAMPLE, T)
            am = m(x).argmax(-1)[0].cpu().numpy()[:il]
            print(f"GT : {to_str(labels[k])!r}")
            print(f"PR : {to_str(greedy(am))!r}")
            print("-")


if __name__ == "__main__":
    main()
