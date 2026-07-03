"""
Off-the-shelf scene-text detection (EAST via OpenCV DNN) over a directory of frames.
Draws detected text boxes, saves annotated images, prints per-image box counts.
No torch/torchvision -- runs in the host venv (cv2 only). Used to test whether an
off-the-shelf detector generalizes across games (Pokemon vs Fire Emblem) without
any per-game tuning.
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

LAYERS = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]


def decode(scores, geometry, score_thresh):
    nrows, ncols = scores.shape[2:4]
    rects, confs = [], []
    for y in range(nrows):
        sc = scores[0, 0, y]
        d0, d1, d2, d3 = geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y]
        ang = geometry[0, 4, y]
        for x in range(ncols):
            if sc[x] < score_thresh:
                continue
            offx, offy = x * 4.0, y * 4.0
            cos, sin = np.cos(ang[x]), np.sin(ang[x])
            h = d0[x] + d2[x]
            w = d1[x] + d3[x]
            endx = int(offx + cos * d1[x] + sin * d2[x])
            endy = int(offy - sin * d1[x] + cos * d2[x])
            rects.append([endx - int(w), endy - int(h), int(w), int(h)])  # x,y,w,h
            confs.append(float(sc[x]))
    return rects, confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--width", type=int, default=640)   # upscale target (small GBA text)
    ap.add_argument("--score", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    net = cv2.dnn.readNet(a.model)

    files = sorted(glob.glob(os.path.join(a.indir, "*.png")))
    total = 0
    for path in files:
        img = cv2.imread(path)
        H0, W0 = img.shape[:2]
        newW = max(32, int(round(a.width / 32)) * 32)
        newH = max(32, int(round(H0 * newW / W0 / 32)) * 32)
        rW, rH = W0 / float(newW), H0 / float(newH)
        blob = cv2.dnn.blobFromImage(img, 1.0, (newW, newH),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        scores, geom = net.forward(LAYERS)
        rects, confs = decode(scores, geom, a.score)
        n = 0
        if rects:
            keep = cv2.dnn.NMSBoxes(rects, confs, a.score, 0.4)
            for i in np.array(keep).flatten():
                x, y, w, h = rects[i]
                cv2.rectangle(img, (int(x * rW), int(y * rH)),
                              (int((x + w) * rW), int((y + h) * rH)), (0, 0, 255), 1)
                n += 1
        cv2.imwrite(os.path.join(a.outdir, os.path.basename(path)), img)
        total += n
        print(f"  {os.path.basename(path)}: {n} text boxes", flush=True)
    print(f"[east] {a.indir}: {total} boxes over {len(files)} frames", flush=True)


if __name__ == "__main__":
    main()
