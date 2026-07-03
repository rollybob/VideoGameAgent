"""
Build a REAL-crop dataset (.npz) in the EXACT format gen_data_mf/train.py expect, so a
synthetic-trained CRNN can be FINE-TUNED on hand/VLM-labeled real frame crops (the
durable fix for the synth->real glyph residual -- W->u, Z->7 -- that synthetic rendering
couldn't close; see docs/DECOMP_FONT_PLAYBOOK F20 + 2026-06-28 session briefing).

Input: a labels JSON (list of {"id","label"}; bench_labels.json works) + a crops dir whose
files are <id>.png (the RAW detector crops -- same pixels the recognizer ingests at
inference, so train==inference distribution). Output: <out>/{train,val}.npz + meta.json.

Crops are height-normalized to meta.height keeping aspect, left-placed on a meta.max_width
canvas padded with the crop's own mean colour (no hard black edge), RGB -- identical to the
synthetic samples. Labels map through gen_data_mf.CHAR_TO_IDX (1-based; 0=CTC blank);
chars outside the charset are dropped. Val is a deterministic held-out slice.

Host venv (cv2 + numpy). Run e.g.:
  .venv/bin/python train/ocr/build_real_npz.py \
      --labels train/ocr/data/bench_pokemon/bench_labels.json \
      --crops  train/ocr/data/bench_pokemon/crops \
      --meta-from train/ocr/data_decomp5 --out train/ocr/data_real --val-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data_mf import CHARS, CHAR_TO_IDX  # noqa: E402


def crop_to_sample(crop, height, max_width):
    """Real BGR crop -> (height, max_width, 3) uint8 RGB-format canvas + true width.
    Matches the synthetic compositor: height-normalized, left-aligned, mean-pad."""
    h, w = crop.shape[:2]
    nw = max(1, int(round(w * height / h)))
    nw = min(nw, max_width)
    resized = cv2.resize(crop, (nw, height), interpolation=cv2.INTER_AREA)
    pad = int(resized.reshape(-1, 3).mean()) if resized.size else 0
    canvas = np.full((height, max_width, 3), pad, np.uint8)
    canvas[:, :nw] = resized
    return canvas, nw


def build(entries, crops_dir, height, max_width, downsample):
    imgs, widths, labels, lengths = [], [], [], []
    skipped = 0
    for e in entries:
        text = e.get("label")
        if not text:
            skipped += 1
            continue
        p = os.path.join(crops_dir, e["id"] + ".png")
        crop = cv2.imread(p)
        if crop is None:
            skipped += 1
            continue
        arr, w = crop_to_sample(crop, height, max_width)
        lab = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
        lab = lab[:max(1, min(len(lab), w // downsample))]   # CTC: label <= time steps
        if not lab:
            skipped += 1
            continue
        imgs.append(arr); widths.append(w); labels.append(np.array(lab, np.int32))
        lengths.append(len(lab))
    if not imgs:
        return None, skipped
    return (np.stack(imgs), np.array(widths, np.int32),
            np.concatenate(labels).astype(np.int32), np.array(lengths, np.int32)), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="JSON list of {id,label}")
    ap.add_argument("--crops", required=True, help="dir of <id>.png raw crops")
    ap.add_argument("--meta-from", required=True,
                    help="a synthetic data dir whose meta.json fixes height/max_width/downsample/chars")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = json.load(open(os.path.join(args.meta_from, "meta.json")))
    if base.get("chars") != CHARS:
        print("[real] WARN: meta chars != gen_data_mf.CHARS; label indices may mismatch the model")
    H, MW, DS = base["height"], base["max_width"], base["downsample"]
    entries = [e for e in json.load(open(args.labels)) if e.get("label")]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(entries)
    nval = max(1, int(len(entries) * args.val_frac)) if len(entries) > 4 else 0
    splits = {"val": entries[:nval], "train": entries[nval:]}

    os.makedirs(args.out, exist_ok=True)
    for name, ents in splits.items():
        if not ents:
            continue
        packed, skipped = build(ents, args.crops, H, MW, DS)
        if packed is None:
            print(f"[real] {name}: no usable samples (skipped {skipped})"); continue
        images, widths, labels, label_lengths = packed
        np.savez_compressed(os.path.join(args.out, f"{name}.npz"), images=images,
                            widths=widths, labels=labels, label_lengths=label_lengths)
        print(f"[real] {name}: {images.shape[0]} samples (skipped {skipped}) -> {name}.npz {images.shape}")

    meta = dict(base)
    meta["source"] = "real_crops"
    meta["labels_file"] = os.path.abspath(args.labels)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"[real] meta -> {args.out}/meta.json (n_classes={meta['n_classes']})")


if __name__ == "__main__":
    main()
