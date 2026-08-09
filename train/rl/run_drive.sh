#!/usr/bin/env bash
# Wrapper for the 3-tier agent: run drive_agent.py in the thor-rl container (writes annotated
# JPEG frames), then stitch to mp4 with the host static ffmpeg, then notify Tim. Runs on the
# HOST (via thor-job) so notify + ffmpeg are reachable.
#   Env: MINUTES (default 30), FPS (default 15), NOTIFY (default 1; 0 = suppress for validation)
set -uo pipefail
VGA=/home/timothy/projects/VGA
OUTDIR="$VGA/sessions/agent_run"
OUT="$OUTDIR/run.mp4"
FRAMES="$OUTDIR/frames"
FF="$HOME/.local/bin/ffmpeg"
MIN="${MINUTES:-30}"; FPS="${FPS:-15}"
mkdir -p "$OUTDIR"

notify_maybe() { if [ "${NOTIFY:-1}" = "1" ]; then "$HOME/.local/bin/notify" "$1"; else echo "[wrap] (notify off) $1"; fi; }

echo "[wrap] $(date) running 3-tier agent MINUTES=$MIN FPS=$FPS"
docker run --rm --runtime nvidia --network host --user 1000:1000 \
  -v "$VGA":/work -w /work -e MINUTES="$MIN" -e FPS="$FPS" -e NOTIFY=0 \
  -e STATE="${STATE:-/work/train/rl/states/alttp_start_normal.state}" -e SKIP_MENU="${SKIP_MENU:-0}" \
  -e OUT=/work/sessions/agent_run/run.mp4 \
  thor-rl:cu130 python3 -u train/rl/drive_agent.py
RC=$?
NF=$(ls "$FRAMES"/*.jpg 2>/dev/null | wc -l)
echo "[wrap] agent rc=$RC frames=$NF"
if [ "$NF" -lt 30 ]; then
  notify_maybe "VGA 3-tier run FAILED (agent rc=$RC, only $NF frames). Check thor-job log."
  exit 1
fi
echo "[wrap] stitching $NF frames -> mp4"
"$FF" -y -framerate "$FPS" -start_number 0 -i "$FRAMES/f%06d.jpg" -c:v libx264 -pix_fmt yuv420p "$OUT" >/tmp/drive_ff.log 2>&1
if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" -gt 10000 ]; then
  SZ=$(du -h "$OUT" | cut -f1)
  notify_maybe "VGA 3-tier run DONE: $NF frames (~$((NF/FPS/60))min game), video $SZ at $OUT . Watch when you're up."
  rm -f "$FRAMES"/*.jpg
else
  notify_maybe "VGA 3-tier: captured $NF frames but ffmpeg failed (see /tmp/drive_ff.log). Frames kept."
  exit 1
fi
echo "[wrap] done -> $OUT"
