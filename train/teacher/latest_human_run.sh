#!/usr/bin/env bash
# The logger (human_logger.lua) now self-creates its run dir on load, so no prep is needed.
# This just reports the most recent human-* run dir and its sample count (to find the data
# after a play session).
set -euo pipefail
VGA=/home/timothy/projects/VGA
D=$(ls -td "$VGA"/sessions/human-* 2>/dev/null | head -1 || true)
[ -z "${D:-}" ] && { echo "no human-* runs yet"; exit 0; }
echo "latest: $D"
echo "samples: $(wc -l < "$D/inputs.csv" 2>/dev/null || echo 0)  frames: $(ls "$D/frames" 2>/dev/null | wc -l)"
