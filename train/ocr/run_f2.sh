#!/usr/bin/env bash
# F2 driver: retrain the GBA-font CRNN with (a) real-crop-like AUGMENTATION and
# (b) an EXTENDED charset (':' and accented-e) so it survives real detector crops and
# can read TIME 6:37 / POKeDEX lines. Writes to data_f2/ + ckpt_f2/ so the PROVEN F1
# artifacts (data/, ckpt/best.pt, crnn.onnx) are left untouched until F2 is validated
# on the real-frame benchmark (Step 3).
#
# gen runs HOST-side (PIL+cv2 in the venv); train runs in thor-torch:cu130 (torch).
# Run via thor-job so it survives disconnect/reboot; train.py honours --max-minutes.
set -euo pipefail
cd "$HOME/projects/VGA"

FONT="${FONT:-train/ocr/fonts/pokemon-ds.ttf}"
DATA="${DATA:-train/ocr/data_f2}"
CKPT="${CKPT:-train/ocr/ckpt_f2}"
MAX_MIN="${MAX_MIN:-90}"
N_TRAIN="${N_TRAIN:-40000}"
N_VAL="${N_VAL:-3000}"

mkdir -p "$DATA" "$CKPT"
echo "[f2] $(date -Is) generating AUGMENTED synthetic data (font=$FONT, n=$N_TRAIN)"
"$HOME/projects/VGA/.venv/bin/python" train/ocr/gen_data.py \
    --out "$DATA" --font "$FONT" --n-train "$N_TRAIN" --n-val "$N_VAL" \
    --seed 1 --augment

echo "[f2] $(date -Is) training in thor-torch:cu130 (max ${MAX_MIN}m)"
docker run --rm --runtime nvidia -v "$HOME/projects/VGA:/work" -w /work thor-torch:cu130 \
    python3 train/ocr/train.py --data "$DATA" --ckpt "$CKPT" \
    --epochs 50 --batch 128 --lr 1e-3 --max-minutes "$MAX_MIN"

echo "[f2] $(date -Is) done; best checkpoint at $CKPT/best.pt"
echo "[f2] NEXT (Step 3): export ONNX from $CKPT/best.pt, then eval on bench_pokemon"
