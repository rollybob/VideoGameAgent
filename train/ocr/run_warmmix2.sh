#!/usr/bin/env bash
# Retrain warm-mix on the EXPANDED real set (352 crops incl. tour menus/summary/bag/
# dialogue). warm-start decomp5 + real-dominant mix, select on real val. Eval on
# bench_v2 (deployment-split) +/- lexicon. Bar: Tesseract 0.050/10; prev warm-mix 0.057/11.
set -uo pipefail
cd "$HOME/projects/VGA"
VENV="$HOME/projects/VGA/.venv/bin/python"
DK="docker run --rm --runtime nvidia -v $HOME/projects/VGA:/work -w /work thor-torch:cu130 python3"
CKPT="train/ocr/ckpt_ft_warmmix2"; mkdir -p "$CKPT"
LEX="train/ocr/data/real_combined/labels.json"
echo "[wm2] $(date -Is) train"
$DK train/ocr/train.py --data train/ocr/data_real_mix --ckpt "$CKPT" \
    --epochs 60 --batch 128 --lr 1e-4 --max-minutes 20 --init train/ocr/ckpt_decomp5/best.pt
$DK train/ocr/export_onnx.py --ckpt "$CKPT/best.pt" --out "$CKPT/crnn.onnx"
echo "[wm2] ===== bench_v2 (no lexicon) ====="
$VENV train/ocr/eval_benchmark.py --onnx "$CKPT/crnn.onnx" --bench train/ocr/data/bench_pokemon_v2 | tail -4 | tee "$CKPT/bench_v2.txt"
echo "[wm2] ===== bench_v2 (WITH lexicon) ====="
$VENV train/ocr/eval_benchmark.py --onnx "$CKPT/crnn.onnx" --bench train/ocr/data/bench_pokemon_v2 --lexicon "$LEX" | tail -4 | tee "$CKPT/bench_v2_lex.txt"
echo "[wm2] $(date -Is) DONE"
