#!/usr/bin/env bash
# F18-decomp: train the font-general RGB recognizer with REAL FRLG bitmap glyphs
# mixed into the synthetic corpus. THE PIVOT TEST (docs/DECOMP_FONT_PLAYBOOK.md):
# does real pixel-glyph data close the bench_pokemon synthetic->real gap (F18 v2
# stuck at CER 0.44, 0/13) and beat Tesseract (CER 0.042, 9/13)?
#
# Clean host/container split: gen on the HOST venv (PIL+cv2+bitmap_font), train +
# ONNX export in the container (torch), eval back on the host (onnxruntime+pytesseract).
#
# Env knobs (for a fast dry run): NT NV EP MM BW MAXF.  Full run uses the defaults.
#   BW   = bitmap_weight (0.5 = half real FRLG bitmap, half TTF diversity)
#   MAXF = cap TTF font scan (0 = all 3876; set small for a quick dry run)
set -euo pipefail
cd "$HOME/projects/VGA"
VENV="$HOME/projects/VGA/.venv/bin/python"
DATA="${DATA:-train/ocr/data_decomp}"
CKPT="${CKPT:-train/ocr/ckpt_decomp}"
NT="${NT:-60000}"; NV="${NV:-4000}"; EP="${EP:-60}"; MM="${MM:-240}"
BW="${BW:-0.5}"; MAXF="${MAXF:-0}"

mkdir -p "$DATA" "$CKPT"

echo "[decomp] $(date -Is) gen multi-font+bitmap RGB data (n=$NT bw=$BW maxf=$MAXF)"
PYTHONPATH=train/ocr "$VENV" train/ocr/gen_data_mf.py --out "$DATA" \
    --fonts-dir train/ocr/corpus/google-fonts --bg-dir captures/harvest \
    --n-train "$NT" --n-val "$NV" --bitmap-weight "$BW" --max-fonts "$MAXF" --rescan

echo "[decomp] $(date -Is) train in container (epochs=$EP, max ${MM}m)"
docker run --rm --runtime nvidia -v "$HOME/projects/VGA:/work" -w /work thor-torch:cu130 \
    python3 train/ocr/train.py --data "$DATA" --ckpt "$CKPT" \
    --epochs "$EP" --batch 128 --lr 1e-3 --max-minutes "$MM"

echo "[decomp] $(date -Is) export ONNX (RGB, channel-aware)"
docker run --rm --runtime nvidia -v "$HOME/projects/VGA:/work" -w /work thor-torch:cu130 \
    python3 train/ocr/export_onnx.py --ckpt "$CKPT/best.pt" --out "$CKPT/crnn.onnx"

echo "[decomp] $(date -Is) eval vs Tesseract on bench_pokemon (the bar: beat 0.042)"
"$VENV" train/ocr/eval_benchmark.py --onnx "$CKPT/crnn.onnx" | tee "$CKPT/bench_pokemon.txt"

echo "[decomp] $(date -Is) DONE -> $CKPT/best.pt, $CKPT/crnn.onnx, $CKPT/bench_pokemon.txt"
