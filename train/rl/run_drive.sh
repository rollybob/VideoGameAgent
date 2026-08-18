#!/usr/bin/env bash
# Wrapper for the 3-tier agent: run drive_agent.py in the thor-rl container -- it now writes the
# annotated mp4 DIRECTLY (imageio/ffmpeg baked into thor-rl), so this wrapper just validates the
# video and notifies Tim. Runs on the HOST (via thor-job) so notify is reachable.
#   Env: MINUTES (default 30), FPS (default 15), NOTIFY (default 1; 0 = suppress for validation)
set -uo pipefail
VGA=/home/timothy/projects/VGA
OUTDIR="$VGA/sessions/agent_run"
OUT="$OUTDIR/run.mp4"
MIN="${MINUTES:-30}"; FPS="${FPS:-15}"
mkdir -p "$OUTDIR"

notify_maybe() { if [ "${NOTIFY:-1}" = "1" ]; then "$HOME/.local/bin/notify" "$1"; else echo "[wrap] (notify off) $1"; fi; }

echo "[wrap] $(date) running 3-tier agent MINUTES=$MIN FPS=$FPS"
docker run --rm --runtime nvidia --network host --user 1000:1000 \
  -v "$VGA":/work -w /work -e MINUTES="$MIN" -e FPS="$FPS" -e NOTIFY=0 \
  -e STATE="${STATE:-/work/train/rl/states/alttp_start_normal.state}" -e SKIP_MENU="${SKIP_MENU:-0}" \
  -e OUT=/work/sessions/agent_run/run.mp4 \
  -e WALKER="${WALKER:-}" -e S1="${S1:-}" \
  thor-rl:cu130 python3 -u train/rl/drive_agent.py
RC=$?
echo "[wrap] agent rc=$RC"
# drive_agent.py writes the mp4 itself now (in-container imageio/ffmpeg); just validate it.
if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" -gt 10000 ]; then
  SZ=$(du -h "$OUT" | cut -f1)
  NF=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$OUT" 2>/dev/null)
  notify_maybe "VGA 3-tier run DONE: video $SZ (${NF:-?} frames) at $OUT . Watch when you're up."
else
  notify_maybe "VGA 3-tier run FAILED (agent rc=$RC, no/empty video at $OUT). Check thor-job log."
  exit 1
fi
echo "[wrap] done -> $OUT"
