"""
Step 3 eval: head-to-head CRNN(F2) vs Tesseract on the LABELED real-frame Pokemon
benchmark (data/bench_pokemon/bench_labels.json). Both recognizers get the SAME raw
detector crop each plugin would feed them in the live loop:
  - CRNN: vga.core.textrecog.CrnnRecognizer (onnxruntime) pointed at the F2 ONNX.
  - Tesseract: vga.plugins.pokemon.ocr.Ocr._read_line (the current default path).

Scoring: character error rate (Levenshtein / len(GT)) after whitespace-normalizing
both prediction and GT (collapse internal runs to a single space, strip) -- the
detector merges label/value columns into one line, so multi-space gaps are layout,
not text. Reports per-line reads + aggregate CER and exact-match for each engine.

Host venv (onnxruntime + cv2 + pytesseract). Run:
  ~/projects/VGA/.venv/bin/python train/ocr/eval_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2

sys.path.insert(0, os.path.expanduser("~/projects/VGA"))
from vga.core.textrecog import CrnnRecognizer          # noqa: E402
from vga.plugins.pokemon.ocr import Ocr                # noqa: E402

BENCH = os.path.expanduser("~/projects/VGA/train/ocr/data/bench_pokemon")

# A few menu words that the dex-only training pool doesn't cover, so the lexicon can
# still snap them (kept tiny + game-generic; NOT drawn from any eval label set).
_EXTRA_VOCAB = ["CONTINUE", "NEW", "GAME", "OPTION", "MENU", "SAVE", "EXIT", "lbs"]


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def build_vocab(labels_path):
    """Token vocab from TRAINING labels (not eval) -- the game's known words."""
    vocab = set(_EXTRA_VOCAB)
    if labels_path and os.path.exists(labels_path):
        for e in json.load(open(labels_path)):
            for tok in norm(e.get("label")).split():
                if tok:
                    vocab.add(tok)
    return vocab


def snap(pred, vocab):
    """Snap each OUT-of-vocab token to the nearest vocab token within a tight edit
    distance (1 for short tokens, 2 for long), so proper nouns / correct reads are not
    clobbered. Numeric/short tokens pass through untouched."""
    out = []
    for tok in pred.split():
        if tok in vocab or tok.isdigit() or len(tok) <= 2:
            out.append(tok); continue
        thresh = 1 if len(tok) <= 4 else 2
        best, bestd = tok, thresh + 1
        for w in vocab:
            if abs(len(w) - len(tok)) > thresh:
                continue
            d = lev(tok, w)
            if d < bestd:
                bestd, best = d, w
        out.append(best)
    return " ".join(out)


def lev(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.expanduser(
        "~/projects/VGA/train/ocr/ckpt_f2/crnn.onnx"))
    ap.add_argument("--bench", default=BENCH, help="benchmark dir (bench_labels.json + crops/)")
    ap.add_argument("--lexicon", default="",
                    help="training labels JSON to build a snap-vocab (empty = no lexicon)")
    args = ap.parse_args()

    bench = os.path.expanduser(args.bench)
    labels = json.load(open(os.path.join(bench, "bench_labels.json")))
    vocab = build_vocab(os.path.expanduser(args.lexicon)) if args.lexicon else None
    crnn = CrnnRecognizer(onnx_path=args.onnx)
    ocr = Ocr()
    if not crnn.enabled:
        print("ERROR: CRNN recognizer disabled (onnx/meta missing?). Abort.")
        sys.exit(1)

    agg = {"crnn": [0, 0, 0], "tess": [0, 0, 0]}  # [cer_num, cer_den, exact]
    print(f"{'GT':34} | {'CRNN(F2)':28} | {'Tesseract':28}")
    print("-" * 96)
    for e in labels:
        crop = cv2.imread(os.path.join(bench, "crops", e["id"] + ".png"))
        gt = norm(e["label"])
        crnn_pred = norm(crnn.read(crop))
        if vocab is not None:
            crnn_pred = snap(crnn_pred, vocab)
        preds = {"crnn": crnn_pred, "tess": norm(ocr._read_line(crop, 2))}
        for k, p in preds.items():
            d = lev(p, gt)
            agg[k][0] += d
            agg[k][1] += max(1, len(gt))
            agg[k][2] += int(p == gt)
        mark = lambda p: ("OK " if p == gt else "xx ")  # noqa: E731
        print(f"{gt:34} | {mark(preds['crnn'])}{preds['crnn']:25} | "
              f"{mark(preds['tess'])}{preds['tess']:25}")

    n = len(labels)
    print("-" * 96)
    for k, name in (("crnn", "CRNN(F2)"), ("tess", "Tesseract")):
        cer = agg[k][0] / max(1, agg[k][1])
        print(f"{name:10} CER={cer:.4f}  exact={agg[k][2]}/{n} ({agg[k][2] / n:.1%})")


if __name__ == "__main__":
    main()
