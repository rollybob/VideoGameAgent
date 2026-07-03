"""
Export the F1 CRNN checkpoint to ONNX (CONTAINER side -- torch). The live VGA loop
runs in the host venv which has no torch, so the recognizer runs via onnxruntime
there. Exports with dynamic batch + width axes (line images vary in width), plus a
sidecar *_meta.json (charset / height / downsample) since ONNX carries no Python
metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CRNN, WIDTH_DOWNSAMPLE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="train/ocr/ckpt/best.pt")
    ap.add_argument("--out", default="train/ocr/ckpt/crnn.onnx")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    in_ch = int(meta.get("channels", 1))
    model = CRNN(meta["n_classes"], in_ch=in_ch)
    model.load_state_dict(ck["model"])
    model.eval()

    h = meta["height"]
    dummy = torch.zeros(1, in_ch, h, 256)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 3: "width"},
                      "logits": {0: "batch", 1: "time"}},
        opset_version=13, dynamo=False)   # legacy exporter: no onnxscript dep

    side = {"chars": meta["chars"], "height": h, "channels": in_ch,
            "downsample": WIDTH_DOWNSAMPLE, "n_classes": meta["n_classes"]}
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(side, f, indent=2)

    with torch.no_grad():
        out_shape = tuple(model(dummy).shape)
    print(f"[export] ONNX -> {args.out}")
    print(f"[export] meta -> {meta_path}")
    print(f"[export] torch sanity out shape {out_shape} (B, T, n_classes={meta['n_classes']})")


if __name__ == "__main__":
    main()
