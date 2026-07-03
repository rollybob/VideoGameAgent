#!/usr/bin/env bash
# Start Xvfb + twm (twm REQUIRED -- headless mGBA renders blank without a WM), launch
# the Pokemon ROM, then run the overworld probe (spam B, capture each step).
set -uo pipefail
cd "$HOME/projects/VGA"
export DISPLAY=:99
VENV="$HOME/projects/VGA/.venv/bin/python"
ROM="$HOME/projects/VGA/Emulator/mGBA/roms/Pokemon AI Red.gba"

pgrep -x Xvfb >/dev/null || ( Xvfb :99 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 & )
sleep 2
pgrep -x twm >/dev/null || ( twm >/tmp/twm.log 2>&1 & )
sleep 1
pkill -x mgba-qt 2>/dev/null; sleep 1
nohup /usr/games/mgba-qt "$ROM" >/tmp/mgba_probe.log 2>&1 &
for i in $(seq 1 40); do xdotool search --name mGBA >/dev/null 2>&1 && break; sleep 0.3; done
sleep 5
"$VENV" train/ocr/probe_overworld.py
pkill -x mgba-qt 2>/dev/null
echo "[probe] DONE"
