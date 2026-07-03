#!/usr/bin/env bash
# Pokemon-only frame sweep for the real-crop OCR fine-tune (playbook step 1).
# Longer run than run_capture.sh so the deterministic input pattern walks deeper
# into menus/dex/battle/dialogue -> more SCREEN VARIETY. Starts Xvfb + twm
# (twm REQUIRED -- headless mGBA renders blank without a WM; memory vga-headless-needs-wm).
# Run as a thor-job.
set -uo pipefail
cd "$HOME/projects/VGA"
export DISPLAY=:99
VENV="$HOME/projects/VGA/.venv/bin/python"
ROMS="$HOME/projects/VGA/Emulator/mGBA/roms"
ROM="$ROMS/Pokemon AI Red.gba"

pgrep -x Xvfb >/dev/null || ( Xvfb :99 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 & )
sleep 2
pgrep -x twm >/dev/null || ( twm >/tmp/twm.log 2>&1 & )
sleep 1

echo "[cap] $(date -Is) launching pokemon"
pkill -x mgba-qt 2>/dev/null; sleep 1
nohup /usr/games/mgba-qt "$ROM" >"/tmp/mgba_pokemon.log" 2>&1 &
for i in $(seq 1 40); do xdotool search --name mGBA >/dev/null 2>&1 && break; sleep 0.3; done
sleep 5   # let it boot through logos and render

# n=900 stride=6 warmup=20 -> ~146 frames, traversing deeper game states.
"$VENV" train/ocr/capture_frames.py --name pokemon --out "captures/pokemon_real" --n 900 --stride 6

pkill -x mgba-qt 2>/dev/null
echo "[cap] $(date -Is) DONE; frames under captures/pokemon_real"
