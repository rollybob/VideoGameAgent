#!/usr/bin/env bash
# Sweep three games and capture cropped GBA frames for OCR training data.
# Starts Xvfb + twm (twm REQUIRED -- headless mGBA renders blank without a WM).
# Run as a thor-job (boots + drives 3 games; a couple of minutes).
set -uo pipefail
cd "$HOME/projects/VGA"
export DISPLAY=:99
VENV="$HOME/projects/VGA/.venv/bin/python"
ROMS="$HOME/projects/VGA/Emulator/mGBA/roms"

pgrep -x Xvfb >/dev/null || ( Xvfb :99 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 & )
sleep 2
pgrep -x twm >/dev/null || ( twm >/tmp/twm.log 2>&1 & )
sleep 1

capture () {  # name  rom
  local name="$1" rom="$2"
  echo "[cap] $(date -Is) launching $name"
  pkill -x mgba-qt 2>/dev/null; sleep 1
  nohup /usr/games/mgba-qt "$rom" >"/tmp/mgba_$name.log" 2>&1 &
  for i in $(seq 1 40); do xdotool search --name mGBA >/dev/null 2>&1 && break; sleep 0.3; done
  sleep 5   # let it boot through logos and render
  "$VENV" train/ocr/capture_frames.py --name "$name" --out "captures/$name" --n 240 --stride 6
}

capture pokemon     "$ROMS/Pokemon AI Red.gba"
capture fireemblem  "$ROMS/Fire Emblem (USA, Australia).gba"
capture ffta        "$ROMS/Final Fantasy Tactics Advance (E)(Surplus).gba"
pkill -x mgba-qt 2>/dev/null
echo "[cap] $(date -Is) DONE; frames under captures/{pokemon,fireemblem,ffta}"
