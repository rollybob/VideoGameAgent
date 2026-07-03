#!/usr/bin/env bash
# F1 driver: render synthetic data (host venv, has PIL) then train the CRNN
# (container, has torch). Run as a thor-job so it survives disconnect/reboot.
# thor-job fires `notify` on exit; train.py honours --max-minutes as a soft cap.
set -euo pipefail
cd "$HOME/projects/VGA"

FONT="${FONT:-/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf}"
DATA=train/ocr/data
CKPT=train/ocr/ckpt
MAX_MIN="${MAX_MIN:-90}"
N_TRAIN="${N_TRAIN:-20000}"
N_VAL="${N_VAL:-2000}"

mkdir -p "$DATA" "$CKPT"
echo "[f1] $(date -Is) generating synthetic data (font=$FONT)"
"$HOME/projects/VGA/.venv/bin/python" train/ocr/gen_data.py \
    --out "$DATA" --font "$FONT" --n-train "$N_TRAIN" --n-val "$N_VAL" --seed 1

echo "[f1] $(date -Is) training in thor-torch:cu130 (max ${MAX_MIN}m)"
docker run --rm --runtime nvidia -v "$HOME/projects/VGA:/work" -w /work thor-torch:cu130 \
    python3 train/ocr/train.py --data "$DATA" --ckpt "$CKPT" \
    --epochs 40 --batch 128 --lr 1e-3 --max-minutes "$MAX_MIN"

echo "[f1] $(date -Is) done; best checkpoint at $CKPT/best.pt"
