#!/usr/bin/env bash
# ITER3 real-crop fine-tune (2026-06-30 session 4). Warm-start the synthetic multi-font
# generalist ckpt_f18mf and fine-tune on data_real_all_mix = ALL accumulated real crops
# (666: 352 original Pokemon + session-3 FE/Zelda 92 + today's FFTA 116 / Baldur 75 / Pokemon 31),
# real x3 + 250 data_mf synth regularizer, val real-only. Same recipe as iter2 (run_ft_mfreal.sh)
# so numbers are comparable.
#   BASELINES on the held-out real benches (no lexicon):
#     f18mf synth-only:  Pokemon(206) 0.157 | FE 0.227 | FFTA 0.284
#     iter2 ckpt_ft_mfreal2 (BEAT Tesseract on all 3): route2 0.115 (Tess .212) | FE 0.161 (.171) | FFTA 0.258 (.264)
#   GOAL: beat iter2, especially WIDEN FFTA (we added 116 real FFTA crops -- the weak font).
#   NOTE: Baldur has no bench, so its gain shows only as no-regression + generalization.
set -uo pipefail
cd "$HOME/projects/VGA"
VENV="$HOME/projects/VGA/.venv/bin/python"
DK="docker run --rm --runtime nvidia -v $HOME/projects/VGA:/work -w /work thor-torch:cu130 python3"
CKPT="${CKPT:-train/ocr/ckpt_ft_mfreal3}"; mkdir -p "$CKPT"
DATA="train/ocr/data_real_all_mix"

echo "=== [ft3] $(date -Is) TRAIN warm-start f18mf -> data_real_all_mix (1849 train / 133 real-val) ==="
$DK train/ocr/train.py --data "$DATA" --ckpt "$CKPT" \
    --init train/ocr/ckpt_f18mf/best.pt --epochs 60 --batch 128 --lr 1e-4 --max-minutes 30 \
    || { echo "[ft3] TRAIN FAILED"; notify "VGA ft3 TRAIN FAILED"; exit 1; }
$DK train/ocr/export_onnx.py --ckpt "$CKPT/best.pt" --out "$CKPT/crnn.onnx" \
    || { echo "[ft3] EXPORT FAILED"; notify "VGA ft3 EXPORT FAILED"; exit 1; }

echo "=== [ft3] $(date -Is) EVAL held-out benches (CRNN vs Tesseract, no lexicon) ==="
"$VENV" train/ocr/eval_benchmark.py --onnx "$CKPT/crnn.onnx" | tee "$CKPT/bench_pokemon13.txt" | tail -3
SUMMARY="[ft3] eval vs iter2(route2 .115/FE .161/FFTA .258) & Tess(.212/.171/.264):"
for B in "route2_harvest:Pokemon" "fe_harvest:FE" "ffta_drive:FFTA"; do
  d="${B%%:*}"; name="${B#*:}"
  echo "--- [ft3] $name ($d) ---"
  line=$("$VENV" train/ocr/eval_benchmark.py --onnx "$CKPT/crnn.onnx" --bench "train/ocr/data/$d" \
        2>&1 | tee "$CKPT/eval_$d.txt" | grep -iE "CER|exact" | tail -2 | tr '\n' ' ')
  echo "$name: $line"
  SUMMARY="$SUMMARY | $name $line"
done
echo "[ft3] $(date -Is) DONE -> $CKPT"
notify "$SUMMARY"
