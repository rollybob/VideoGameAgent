#!/bin/bash
# Task 09 A0: mine solo-ALttP hearts/death (and scan for kills) from the
# solo_recorder flight tapes. Captures per-tick RAM for every take, then runs
# find_health.py (cross-tape vote) and analyze_ram.py (strict intersection +
# counter tags). CPU only, no GPU. Numeric reports -> sessions/solo-mining-0709/.
set -euo pipefail
VGA="$HOME/projects/VGA"
IMG="thor-torch:cu130"
OUT="$VGA/sessions/solo-mining-0709"
mkdir -p "$OUT"
DOCK() { docker run --rm --user 1000:1000 -v "$VGA":/vga -w /vga "$IMG" "$@"; }

echo "=== capture RAM for each solo take ==="
NAMES=()
for d in "$VGA"/link/sessions/flight_solo/*/; do
  name="$(basename "$d")"
  NAMES+=("scratch/ram/$name")
  if [ -f "$VGA/train/rl/scratch/ram/$name/ticks.json" ]; then
    echo "--- $name (already captured, skip) ---"; continue
  fi
  echo "--- $name ---"
  DOCK python3 /vga/train/rl/solo_ram_capture.py "/vga/link/sessions/flight_solo/$name"
done
echo "captured ${#NAMES[@]} tapes: ${NAMES[*]}"

echo ""
echo "=== find_health (cross-tape vote) ==="
DOCK python3 /vga/train/rl/find_health.py "${NAMES[@]}" | tee "$OUT/report_health.txt"

echo ""
echo "=== analyze_ram (strict intersection + counter tags) ==="
DOCK python3 /vga/train/rl/analyze_ram.py "${NAMES[@]}" | tee "$OUT/report_analyze.txt"

echo ""
echo "mining done -> $OUT"
