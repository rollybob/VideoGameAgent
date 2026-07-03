#!/usr/bin/env python3
"""Host-side: turn the oracle-assisted mode-label dump (label_modes.py) into train/val
npz for the mode classifier. Runs in the VGA venv (cv2); the container trainer then needs
only torch+numpy. Frames are resized to a fixed HxW and kept as uint8 NHWC RGB.

    .venv/bin/python train/perception/build_mode_dataset.py \
        --data train/perception/mode_data --out train/perception/ds --img-h 96 --img-w 144
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import cv2
import numpy as np

CLASSES = ["field", "menu", "dialog"]
_IDX = {c: i for i, c in enumerate(CLASSES)}


def load(data_dir: str, h: int, w: int):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
    X, y = [], []
    for r in rows:
        img = cv2.imread(os.path.join(data_dir, r["frame"]))  # BGR
        if img is None:
            continue
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        y.append(_IDX[r["label"]])
    return np.asarray(X, dtype=np.uint8), np.asarray(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/perception/mode_data")
    ap.add_argument("--out", default="train/perception/ds")
    ap.add_argument("--img-h", type=int, default=96)
    ap.add_argument("--img-w", type=int, default=144)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y = load(os.path.abspath(args.data), args.img_h, args.img_w)
    rng = np.random.default_rng(args.seed)
    # Stratified split so each class is represented in val (data is small + imbalanced).
    tr_idx, va_idx = [], []
    for c in range(len(CLASSES)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(len(idx) * args.val_frac))
        va_idx += list(idx[:k]); tr_idx += list(idx[k:])
    rng.shuffle(tr_idx); rng.shuffle(va_idx)
    tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "train.npz"), X=X[tr_idx], y=y[tr_idx])
    np.savez(os.path.join(out, "val.npz"), X=X[va_idx], y=y[va_idx])
    meta = {"classes": CLASSES, "img_h": args.img_h, "img_w": args.img_w,
            "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
            "train_hist": {CLASSES[c]: int((y[tr_idx] == c).sum()) for c in range(len(CLASSES))},
            "val_hist": {CLASSES[c]: int((y[va_idx] == c).sum()) for c in range(len(CLASSES))}}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print("wrote", out, json.dumps(meta))


if __name__ == "__main__":
    main()
