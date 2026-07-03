#!/usr/bin/env bash
# Real-crop fine-tune of the MULTI-FONT GENERALIST (docs/OCR_REALDATA_PLAYBOOK.md next step).
# Warm-start ckpt_f18mf (synthetic generalist) and fine-tune on realv2_mix = real_combined (352)
# + tonight's 72 hand-played FE6+Zelda crops, real x3 + 250 synth regularizer (anti-forgetting),
# val real-only. HYPOTHESIS: adding real FE/Zelda crops closes the FE/FFTA domain gap that the
# synthetic-only generalist has, WITHOUT regressing Pokemon -> one model good cross-font.
# Train+export in the thor-torch container (host has no torch); eval on host (onnxruntime).
# f18mf synthetic-only BASELINE to beat:  Pokemon(206) 0.157/Tess .212 | FE 0.227/Tess .171 | FFTA 0.284/Tess .264
set -uo pipefail
cd "$HOME/projects/VGA"
VENV="$HOME/projects/VGA/.venv/bin/python"
DK="docker run --rm --runtime nvidia -v $HOME/projects/VGA:/work -w /work thor-torch:cu130 python3"
CKPT="${CKPT:-train/ocr/ckpt_ft_mfreal}"; mkdir -p "$CKPT"

echo "=== [ftmf] $(date -Is) TRAIN warm-start f18mf -> realv2_mix (1270 train / 84 real-val) ==="
$DK train/ocr/train.py --data train/ocr/data/realv2_mix --ckpt "$CKPT" \
    --init train/ocr/ckpt_f18mf/best.pt --epochs 60 --batch 128 --lr 1e-4 --max-minutes 25 \
    || { echo "[ftmf] TRAIN FAILED"; exit 1; }
$DK train/ocr/export_onnx.py --ckpt "$CKPT/best.pt" --out "$CKPT/crnn.onnx" \
    || { echo "[ftmf] EXPORT FAILED"; exit 1; }

echo "=== [ftmf] $(date -Is) EVAL cross-game (CRNN vs Tesseract; baseline above) ==="
"$VENV" train/ocr/eval_benchmark.py --onnx "$CKPT/crnn.onnx" | tee "$CKPT/bench_pokemon.txt" | tail -3
for B in "route2_harvest:Pokemon" "fe_harvest:FE" "ffta_drive:FFTA"; do
  d="${B%%:*}"; name="${B#*:}"
  echo "--- [ftmf] $name ($d) ---"
  "$VENV" train/ocr/eval_benchmark.py --onnx "$CKPT/crnn.onnx" --bench "train/ocr/data/$d" \
      2>&1 | tee "$CKPT/eval_$d.txt" | grep -E "CER|exact" | tail -2
done
echo "[ftmf] $(date -Is) DONE -> $CKPT"
