#!/usr/bin/env bash
# Real-crop fine-tune ABLATION (docs/OCR_REALDATA_PLAYBOOK.md sec 6 + Tim's fresh-model arm).
# 271 VLM-labeled real crops (live dex/menu + staged PNG-sheet summary/battle/types).
# All arms select the best checkpoint on a REAL val set (sec 6.1) and are scored on
# bench_pokemon vs Tesseract -- the same bar as ckpt_decomp5 (CER 0.105 / Tesseract 0.042).
#
# Arms:
#   A warm-real   : warm-start ckpt_decomp5, fine-tune on REAL only (lr 1e-4)  [primary]
#   B scratch-real: train from SCRATCH on REAL only (lr 1e-3)                  [Tim's fresh model]
#   C warm-mix    : warm-start ckpt_decomp5, REAL x3 + 250 synth (lr 1e-4)     [anti-forgetting]
# Clean split: train+export in container (torch), eval on host (onnxruntime+pytesseract).
set -uo pipefail
cd "$HOME/projects/VGA"
VENV="$HOME/projects/VGA/.venv/bin/python"
DK="docker run --rm --runtime nvidia -v $HOME/projects/VGA:/work -w /work thor-torch:cu130 python3"
MM="${MM:-20}"

run_arm () {
  local name="$1" data="$2" init="$3" lr="$4" ep="$5"
  local ckpt="train/ocr/ckpt_ft_${name}"
  mkdir -p "$ckpt"
  echo "=== [ft] $(date -Is) ARM $name  data=$data init='${init:-none}' lr=$lr ep=$ep ==="
  local initarg=""; [ -n "$init" ] && initarg="--init $init"
  $DK train/ocr/train.py --data "$data" --ckpt "$ckpt" \
      --epochs "$ep" --batch 128 --lr "$lr" --max-minutes "$MM" $initarg || { echo "[ft] $name TRAIN FAILED"; return 1; }
  $DK train/ocr/export_onnx.py --ckpt "$ckpt/best.pt" --out "$ckpt/crnn.onnx" || { echo "[ft] $name EXPORT FAILED"; return 1; }
  echo "--- [ft] $name eval on bench_pokemon (bar: beat Tesseract 0.042; baseline decomp5 0.105) ---"
  "$VENV" train/ocr/eval_benchmark.py --onnx "$ckpt/crnn.onnx" | tee "$ckpt/bench_pokemon.txt"
}

run_arm warm-real    train/ocr/data_real_combined train/ocr/ckpt_decomp5/best.pt 1e-4 60
run_arm scratch-real train/ocr/data_real_combined ""                             1e-3 120
run_arm warm-mix     train/ocr/data_real_mix      train/ocr/ckpt_decomp5/best.pt 1e-4 60

echo "[ft] $(date -Is) ===== ABLATION SUMMARY (bar: Tesseract 0.042; baseline decomp5 0.105) ====="
for a in warm-real scratch-real warm-mix; do
  echo "=== $a ==="; grep -E "CER|exact|OVERALL|mean" "train/ocr/ckpt_ft_${a}/bench_pokemon.txt" 2>/dev/null | tail -4
done
echo "[ft] $(date -Is) DONE"
