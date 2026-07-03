#!/usr/bin/env bash
set -uo pipefail; cd "$HOME/projects/VGA"; export DISPLAY=:99
VENV="$HOME/projects/VGA/.venv/bin/python"; ROM="$HOME/projects/VGA/Emulator/mGBA/roms/Pokemon AI Red.gba"
pgrep -x Xvfb >/dev/null || ( Xvfb :99 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 & ); sleep 2
pgrep -x twm >/dev/null || ( twm >/tmp/twm.log 2>&1 & ); sleep 1
pkill -x mgba-qt 2>/dev/null; sleep 1; rm -rf captures/battle
nohup /usr/games/mgba-qt "$ROM" >/tmp/mgba_battle.log 2>&1 &
for i in $(seq 1 40); do xdotool search --name mGBA >/dev/null 2>&1 && break; sleep 0.3; done
sleep 6; "$VENV" train/ocr/drive_battle.py; pkill -x mgba-qt 2>/dev/null; echo "[battle] DONE"
