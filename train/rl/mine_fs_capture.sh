#!/bin/bash
# Task 09 A0: capture per-tick RAM for the Four Swords flight tapes (4P link
# format -> ram_capture.py, core 0 holds the whole lockstep world). Idempotent.
set -euo pipefail
VGA="$HOME/projects/VGA"
IMG="thor-torch:cu130"
DOCK() { docker run --rm --user 1000:1000 -v "$VGA":/vga -w /vga "$IMG" "$@"; }
for d in "$VGA"/link/sessions/flight/*/; do
  name="$(basename "$d")"
  if [ -f "$VGA/train/rl/scratch/ram/$name/ticks.json" ]; then
    echo "skip $name (already captured)"; continue
  fi
  echo "--- capture $name ---"
  DOCK python3 /vga/train/rl/ram_capture.py "/vga/link/sessions/flight/$name"
done
echo "FS capture done"
