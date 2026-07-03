#!/usr/bin/env bash
# F18 GENERALIST: train one RGB CTC-CRNN from scratch on the multi-font corpus (data_mf:
# 120k lines, 3642 train fonts + FRLG bitmap, RGB+color+real-bg+F2 aug, 50 held-out fonts).
# Then export ONNX and eval head-to-head vs Tesseract on all THREE real game benchmarks
# (no lexicon = honest font-general read). Goal: decently good on Pokemon AND FFTA AND FE
# at once, not SOTA on one. Container train (GPU); host eval. Plan: docs/OCR_F18_PLAN.md.
set -uo pipefail
cd "$HOME/projects/VGA"
VENV="$HOME/projects/VGA/.venv/bin/python"
DK="docker run --rm --runtime nvidia -v $HOME/projects/VGA:/work -w /work thor-torch:cu130 python3"
CKPT="train/ocr/ckpt_f18mf"; mkdir -p "$CKPT"

echo "[f18] $(date -Is) TRAIN from scratch (multi-font RGB)"
$DK train/ocr/train.py --data train/ocr/data_mf --ckpt "$CKPT" \
    --epochs 80 --batch 256 --lr 1e-3 --max-minutes 120

echo "[f18] export ONNX"
$DK train/ocr/export_onnx.py --ckpt "$CKPT/best.pt" --out "$CKPT/crnn.onnx"

echo "[f18] ===== EVAL vs Tesseract on 3 real game benchmarks (no lexicon) ====="
for B in "route2_harvest:Pokemon(206)" "ffta_drive:FFTA(67)" "fe_harvest:FE(29)"; do
  d="${B%%:*}"; name="${B#*:}"
  echo "----- $name -----"
  $VENV train/ocr/eval_benchmark.py --onnx "$CKPT/crnn.onnx" \
      --bench "train/ocr/data/$d" 2>/dev/null | tail -3 | tee "$CKPT/eval_${d}.txt"
done
echo "[f18] $(date -Is) DONE"
