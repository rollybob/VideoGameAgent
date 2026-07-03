"""
Assemble a real-DOMINANT mixed train.npz for fine-tuning (playbook sec 6.2):
real crops oversampled REAL_TILE x, plus a random N_SYNTH synthetic samples as a
light regularizer against forgetting rare chars/fonts. Val stays REAL-only (copied
from the real split) so checkpoints are selected on the real distribution (sec 6.1).

  python train/ocr/build_mix.py --real train/ocr/data_real_combined \
      --synth train/ocr/data_decomp5 --out train/ocr/data_real_mix \
      --real-tile 3 --n-synth 250
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np


def split_labels(labels, lengths):
    out, k = [], 0
    for n in lengths:
        out.append(labels[k:k + n]); k += int(n)
    return out


def load(d):
    z = np.load(os.path.join(d, "train.npz"))
    return (z["images"], z["widths"], split_labels(z["labels"], z["label_lengths"]),
            z["label_lengths"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-tile", type=int, default=3)
    ap.add_argument("--n-synth", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    r_img, r_w, r_lab, r_len = load(args.real)
    s_img, s_w, s_lab, s_len = load(args.synth)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(s_img), size=min(args.n_synth, len(s_img)), replace=False)

    imgs = [r_img] * args.real_tile + [s_img[idx]]
    ws = [r_w] * args.real_tile + [s_w[idx]]
    lens = [r_len] * args.real_tile + [s_len[idx]]
    labs = r_lab * args.real_tile + [s_lab[i] for i in idx]

    images = np.concatenate(imgs)
    widths = np.concatenate(ws)
    label_lengths = np.concatenate(lens)
    labels = np.concatenate(labs).astype(np.int32)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "train.npz"), images=images,
                        widths=widths, labels=labels, label_lengths=label_lengths)
    shutil.copy(os.path.join(args.real, "val.npz"), os.path.join(args.out, "val.npz"))
    shutil.copy(os.path.join(args.real, "meta.json"), os.path.join(args.out, "meta.json"))
    n_real = len(r_img) * args.real_tile
    print(f"mix train: {images.shape[0]} ({n_real} real x{args.real_tile} + {len(idx)} synth), "
          f"val: real (copied)")


if __name__ == "__main__":
    main()
