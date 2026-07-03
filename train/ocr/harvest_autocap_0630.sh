#!/usr/bin/env bash
# Stage 1 of the real-crop fine-tune for the 2026-06-30 hand-play batch (FFTA / Pokemon /
# Baldur's Gate). EAST-harvest the autocap frames -> deduped line crops -> filter texture
# false-fires -> clean_manifest.json + montages, ready for VLM labeling. Non-GPU (EAST on CPU).
# Per-game so fonts stay separable. Frames-glob includes the "_0_10_2" fragment buckets that
# autocap created when the window title momentarily lacked the fps reading.
#   build_benchmark.py --split  -> crops/ + manifest.json + montages
#   dedup_crops.py              -> uniq_crops/ + uniq_manifest.json
#   filter_textlike.py          -> clean_manifest.json (drops colorful/tiny/flat false-fires)
# Idempotent / re-runnable. Run via thor-job.
set -uo pipefail
cd "$HOME/projects/VGA"
PY="$HOME/projects/VGA/.venv/bin/python"

# slug | frame-source glob (under captures/autocap; * sweeps the _0_10_2 fragment buckets)
GAMES=(
  "ffta0630|final_fantasy_tactics_advance_europe*"
  "pkmn0630|pokemon_fire*"
  "baldur0630|baldur_s_gate_dark_alliance_europe*"
)

for entry in "${GAMES[@]}"; do
  slug="${entry%%|*}"; pat="${entry#*|}"
  out="train/ocr/data/realcap_${slug}"
  n=$(ls captures/autocap/$pat/*.png 2>/dev/null | wc -l)
  echo "[harvest] $(date -Is) $slug  <- captures/autocap/$pat ($n frames) -> $out"
  "$PY" train/ocr/build_benchmark.py --game "$slug" --out "$out" \
        --frames-glob "captures/autocap/$pat/*.png" --split || { echo "[harvest] $slug build FAILED"; continue; }
  "$PY" train/ocr/dedup_crops.py --dir "$out" || { echo "[harvest] $slug dedup FAILED"; continue; }
  uc=$(ls "$out"/uniq_crops/*.png 2>/dev/null | wc -l)
  # Per-game saturation ceiling: FFTA/Baldur render COLORED text on COLORED bgs so the default
  # sat>62 gate drops real dialogue/menu text (FFTA lost ~900 crops; audited 2026-06-30). Pokemon
  # is white-on-dark so the default correctly drops grass/map textures.
  case "$slug" in
    ffta0630)   farg="$slug:120" ;;
    baldur0630) farg="$slug:100" ;;
    *)          farg="$slug" ;;
  esac
  "$PY" train/ocr/filter_textlike.py "$farg" || { echo "[harvest] $slug filter FAILED"; continue; }
  cl=$(${PY} -c "import json;print(len(json.load(open('$out/clean_manifest.json'))))" 2>/dev/null || echo "?")
  echo "[harvest] $slug DONE: $uc uniq crops -> $cl clean -> $out/uniq_crops"
done
echo "[harvest] $(date -Is) ALL DONE"
