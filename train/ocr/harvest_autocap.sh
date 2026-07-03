#!/usr/bin/env bash
# Stage 1 of the real-crop fine-tune (docs/OCR_REALDATA_PLAYBOOK.md): EAST-harvest the
# hand-played autocap frames (2026-06-29) into deduped line crops + montages, ready for
# VLM labeling. Non-GPU (EAST runs on CPU). Per-game so fonts stay separable.
#   build_benchmark.py --split  -> crops/ + manifest.json (tesseract seed) + montages
#   dedup_crops.py              -> uniq_crops/ + uniq_montage_*.png + uniq_manifest.json
# Run via thor-job; checkpointing N/A (idempotent, re-runnable; skips nothing but cheap).
set -uo pipefail
cd "$HOME/projects/VGA"
PY="$HOME/projects/VGA/.venv/bin/python"

# slug | frame source dir (under captures/autocap)
GAMES=(
  "zeldaminish|legend_of_zelda_the_the_minish_cap_usa"
  "fourswords|legend_of_zelda_the_a_link_to_the_past_four_swords_usa"
  "fe6|fireemblem6"
)

for entry in "${GAMES[@]}"; do
  slug="${entry%%|*}"; src="captures/autocap/${entry#*|}"
  out="train/ocr/data/realcap_${slug}"
  n=$(ls "$src"/*.png 2>/dev/null | wc -l)
  echo "[harvest] $(date -Is) $slug  <- $src ($n frames) -> $out"
  "$PY" train/ocr/build_benchmark.py --game "$slug" --out "$out" \
        --frames-glob "$src/*.png" --split || { echo "[harvest] $slug build FAILED"; continue; }
  "$PY" train/ocr/dedup_crops.py --dir "$out" || { echo "[harvest] $slug dedup FAILED"; continue; }
  uc=$(ls "$out"/uniq_crops/*.png 2>/dev/null | wc -l)
  echo "[harvest] $slug DONE: $uc unique crops -> $out/uniq_crops (montages: $out/uniq_montage_*.png)"
done
echo "[harvest] $(date -Is) ALL DONE"
