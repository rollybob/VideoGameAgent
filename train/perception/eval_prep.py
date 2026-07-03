#!/usr/bin/env python3
"""Host: preprocess a trajectory's frames into an npz for the container mode classifier.
Optionally crop the green pillarbox bars that the live on-screen capture adds (the native
training frames have none), to test how much that domain gap costs.

    .venv/bin/python train/perception/eval_prep.py --traj sessions/traj-... --out /tmp/live.npz
"""
from __future__ import annotations
import argparse, glob, os
import cv2, numpy as np


def crop_pillarbox(img):
    """Trim near-uniform side/top bars (the mGBA window padding around the GBA screen).
    Keeps the largest central region whose columns/rows have real variance."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col_var = g.var(axis=0); row_var = g.var(axis=1)
    cx = np.where(col_var > col_var.max() * 0.05)[0]
    ry = np.where(row_var > row_var.max() * 0.05)[0]
    if len(cx) and len(ry):
        return img[ry[0]:ry[-1] + 1, cx[0]:cx[-1] + 1]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default="/tmp/live.npz")
    ap.add_argument("--img-h", type=int, default=96)
    ap.add_argument("--img-w", type=int, default=144)
    ap.add_argument("--crop", action="store_true", help="crop the pillarbox before resize")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.traj, "frames", "*.png")))
    X, idx = [], []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        if args.crop:
            img = crop_pillarbox(img)
        img = cv2.resize(img, (args.img_w, args.img_h), interpolation=cv2.INTER_AREA)
        X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        idx.append(int(os.path.splitext(os.path.basename(f))[0]))
    np.savez(args.out, X=np.asarray(X, np.uint8), idx=np.asarray(idx, np.int64))
    print("wrote", args.out, "frames:", len(X), "crop:", args.crop)


if __name__ == "__main__":
    main()
