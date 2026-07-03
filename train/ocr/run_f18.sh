#!/usr/bin/env bash
# F18 - train the FONT-GENERAL RGB recognizer from scratch.
# gen_data_mf.py (host venv, PIL+cv2): hundreds of fonts x RGB x color x real GBA
# backgrounds x F2 augmentation -> data_f18/. val uses HELD-OUT fonts, so val_CER
# measures UNSEEN-font generalization (the whole point). Then train in the container.
# Run as a thor-job (full font scan + 60k-sample gen + train; ~1h). No GPU cap (owner).
set -euo pipefail
cd "$HOME/projects/VGA"
DATA="${DATA:-train/ocr/data_f18}"
CKPT="${CKPT:-train/ocr/ckpt_f18}"
NT="${NT:-60000}"; NV="${NV:-4000}"; EP="${EP:-60}"; MM="${MM:-240}"

mkdir -p "$DATA" "$CKPT"
echo "[f18] $(date -Is) generating multi-font RGB data (n=$NT)"
"$HOME/projects/VGA/.venv/bin/python" train/ocr/gen_data_mf.py --out "$DATA" \
    --fonts-dir train/ocr/corpus/google-fonts --bg-dir captures/harvest \
    --n-train "$NT" --n-val "$NV"

echo "[f18] $(date -Is) training in thor-torch:cu130 (epochs=$EP, max ${MM}m)"
docker run --rm --runtime nvidia -v "$HOME/projects/VGA:/work" -w /work thor-torch:cu130 \
    python3 train/ocr/train.py --data "$DATA" --ckpt "$CKPT" \
    --epochs "$EP" --batch 128 --lr 1e-3 --max-minutes "$MM"

echo "[f18] $(date -Is) done -> $CKPT/best.pt"
echo "[f18] NEXT: export ONNX (RGB), then eval_benchmark on bench_{pokemon,fe,ffta}"
