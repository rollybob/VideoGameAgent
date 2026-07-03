#!/usr/bin/env bash
# Download the EAST scene-text model (if needed) and run it on the Pokemon and
# Fire Emblem frame sets. Host venv (cv2 only, CPU). Run as a thor-job: the model
# download + CPU detection can take a minute or two.
set -uo pipefail
cd "$HOME/projects/VGA"
MODEL=train/ocr/models/frozen_east_text_detection.pb
VENV="$HOME/projects/VGA/.venv/bin/python"
mkdir -p train/ocr/models /tmp/det_out/fe /tmp/det_out/pkmn

need_dl=1
[ -s "$MODEL" ] && [ "$(wc -c < "$MODEL")" -gt 90000000 ] && need_dl=0
if [ "$need_dl" = 1 ]; then
  echo "[east] downloading EAST model..."
  for url in \
    "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb" \
    "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb"; do
    echo "  trying $url"
    curl -fsSL -m 180 -o "$MODEL" "$url" || continue
    [ "$(wc -c < "$MODEL")" -gt 90000000 ] && { echo "  got it"; break; }
  done
fi
SZ=$(wc -c < "$MODEL" 2>/dev/null || echo 0)
echo "[east] model size: $SZ bytes"
if [ "$SZ" -lt 90000000 ]; then echo "[east] MODEL DOWNLOAD FAILED"; exit 1; fi

echo "=== POKEMON ==="
"$VENV" train/ocr/detect_east.py --model "$MODEL" --indir /tmp/det_test/pkmn --outdir /tmp/det_out/pkmn
echo "=== FIRE EMBLEM ==="
"$VENV" train/ocr/detect_east.py --model "$MODEL" --indir /tmp/det_test/fe --outdir /tmp/det_out/fe
echo "[east] DONE; annotated frames in /tmp/det_out/{pkmn,fe}"
