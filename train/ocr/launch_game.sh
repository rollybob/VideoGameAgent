#!/usr/bin/env bash
# Reliable headless mGBA launch for the driving rig (drive.py). Crushes the recurring
# "non-exposed window / input drops on relaunch" bug -- debugged 2026-06-29.
#
# ROOT CAUSE: twm fully exposes/focuses only the FIRST mGBA window it manages. The
# working session-1 recipe was (Xvfb up) -> start twm -> launch mGBA. The bug only ever
# appeared on RELAUNCH (pkill mgba + launch a new one while the SAME long-lived twm kept
# running) -- the new window came up non-exposed and Qt dropped input.
# So the fix is just: restart twm FRESH per game so mGBA is again the first window twm sees.
#
# DO NOT "fix" exposure with windowactivate / getwindowfocus / `wmctrl -a` / windowfocus /
# mousemove: those either block forever under twm or, if signalled mid-X-call, WEDGE the X
# server so every later xdotool hangs. The fresh-twm recipe needs none of them.
#
# Xvfb is left running across games (it's stable; only twm is restarted). Usage:
#   WID=$(launch_game.sh "/abs/path/rom.gba")   # prints WID, exits 0 when booted & ready
set -uo pipefail
ROM="${1:?usage: launch_game.sh /path/to/rom.gba}"
DISP=":99"; export DISPLAY="$DISP"
[ -f "$ROM" ] || { echo "ERROR: ROM not found: $ROM" >&2; exit 2; }

# Xvfb: start only if down (don't disturb a healthy server)
if ! pgrep -x Xvfb >/dev/null; then
  rm -f /tmp/.X99-lock 2>/dev/null
  # 1280x1024 (was 1024x768): mGBA-qt opens ~1024x791 incl. menu + twm title bar,
  # taller than a 768 screen -- twm then shoves it to y=-12 (off the top) and mss
  # refuses the grab ("capture size exceeds allocated buffer"). A taller screen lets
  # the whole window sit on-screen so capture + _crop_to_gba work. (2026-07-01)
  nohup Xvfb "$DISP" -screen 0 1280x1024x24 >/tmp/xvfb.log 2>&1 &
  sleep 3
fi
# fresh twm + fresh mGBA so mGBA is the first window twm manages
pkill -9 -x mgba-qt 2>/dev/null; pkill -9 -x twm 2>/dev/null
sleep 1
nohup twm >/tmp/twm.log 2>&1 &
sleep 1
nohup /usr/games/mgba-qt "$ROM" >/tmp/mgba_launch.log 2>&1 &
WID=""
for i in $(seq 1 50); do
  WID=$(timeout 4 xdotool search --name "mGBA" 2>/dev/null | head -1)
  [ -n "$WID" ] && break
  sleep 0.3
done
[ -z "$WID" ] && { echo "ERROR: mGBA window never appeared" >&2; exit 1; }
sleep 5   # boot through BIOS/logos
echo "[launch] ready WID=$WID" >&2
echo "$WID"
