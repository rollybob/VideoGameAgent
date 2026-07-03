#!/usr/bin/env bash
# Watch for a new NVIDIA Jetson-Thor vLLM container image.
#
# WHY: FP8 on Thor (sm_110) is blocked upstream - vLLM's FP8 CUTLASS/Triton kernels
# fault on sm_110 (see sessions/2026-07-01 log). It will start working when NVIDIA ships
# an updated image with fixed sm_110 kernels. This checks the ghcr registry for a changed
# digest on the `latest-jetson-thor` tag (a rebuild) or new cu130 tags, and pings Tim so
# we can retry the FP8 path (serve/run_vllm.sh) the moment it's real.
#
# Detection ONLY - does not pull (~35GB) or touch the GPU. Uses curl + python3 + the
# `notify` helper; no docker, no Claude session. Runs from cron so it survives the
# nightly power-off (see the @reboot + periodic entries installed in crontab).
set -uo pipefail
export PATH="$HOME/.local/bin:/usr/bin:/bin:$PATH"   # cron has a minimal PATH; notify lives in ~/.local/bin

REPO="nvidia-ai-iot/vllm"
TAG="latest-jetson-thor"
STATE_DIR="$HOME/.config/vga-vllm-watch"
STATE_FILE="$STATE_DIR/fingerprint"
LOG="$STATE_DIR/watch.log"
mkdir -p "$STATE_DIR"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

TOKEN=$(curl -s --max-time 25 "https://ghcr.io/token?scope=repository:${REPO}:pull" \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('token',''))" 2>/dev/null)
[ -n "$TOKEN" ] || { echo "$(ts) WARN: no registry token (network down / box just booted?)" >>"$LOG"; exit 0; }

ACCEPT="application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json"
DIGEST=$(curl -s --max-time 25 -o /dev/null -D - -H "Authorization: Bearer $TOKEN" -H "Accept: $ACCEPT" \
  "https://ghcr.io/v2/${REPO}/manifests/${TAG}" | tr -d '\r' \
  | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}')
TAGS=$(curl -s --max-time 25 -H "Authorization: Bearer $TOKEN" "https://ghcr.io/v2/${REPO}/tags/list" \
  | python3 -c "import sys,json;print(' '.join(sorted(json.load(sys.stdin).get('tags',[]))))" 2>/dev/null)
[ -n "$DIGEST" ] || { echo "$(ts) WARN: no digest returned (network?)" >>"$LOG"; exit 0; }

FP="$DIGEST | $TAGS"
if [ ! -f "$STATE_FILE" ]; then
  echo "$FP" >"$STATE_FILE"
  echo "$(ts) init baseline: $DIGEST" >>"$LOG"
  exit 0
fi

if [ "$FP" != "$(cat "$STATE_FILE")" ]; then
  echo "$(ts) CHANGED -> new=[$FP]" >>"$LOG"
  echo "$FP" >"$STATE_FILE"
  notify "VGA watcher: NVIDIA pushed a NEW Jetson-Thor vLLM image (${TAG} digest or tag list changed). It may finally have working sm_110 FP8 kernels. To retry FP8: cd ~/projects/VGA && docker pull ghcr.io/${REPO}:${TAG} && thor-job start vga-vllm -- serve/run_vllm.sh, then DISPLAY=:99 .venv/bin/python serve/bench_vllm.py."
else
  echo "$(ts) no change" >>"$LOG"
fi
