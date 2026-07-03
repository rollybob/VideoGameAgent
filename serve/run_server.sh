#!/usr/bin/env bash
# Launch the VGA local VLM server in the thor-vlm container (GPU).
#
# The model weights are mounted read-only from the host (~/models/<name>) so the image
# stays small and the model is swappable. The server binds 127.0.0.1:PORT INSIDE the
# container; --network host publishes it to the host loop so vga/reason/local_reasoner.py
# can reach it at http://127.0.0.1:PORT.
#
# Long-running: launch it under thor-job so it survives disconnect, e.g.
#   thor-job start vlm-server -- serve/run_server.sh
# then wait for the model to load (GET /health -> ok) before running --backend local.
#
# The server code (serve_vlm.py) is bind-mounted from the host over the copy baked into
# the image at /srv/serve_vlm.py, so host edits take effect on the next server restart
# WITHOUT a `docker build`. (Before 2026-07-02 the code was only baked in at build time,
# so host edits were silently ignored by the running container -- a real footgun that
# made a prompt change look like a no-op.) serve_vlm.py is deliberately self-contained
# (no vga import), so mounting the single file is enough.
set -uo pipefail

SRV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-8077}"
IMAGE="${IMAGE:-thor-vlm:cu130}"

[ -d "$MODEL_DIR" ] || { echo "ERROR: model dir not found: $MODEL_DIR" >&2; exit 2; }
[ -f "$SRV_DIR/serve_vlm.py" ] || { echo "ERROR: serve_vlm.py not found in $SRV_DIR" >&2; exit 2; }

exec docker run --rm --runtime nvidia --network host \
  -v "$MODEL_DIR":/models:ro \
  -v "$SRV_DIR/serve_vlm.py":/srv/serve_vlm.py:ro \
  -e MODEL_DIR=/models -e PORT="$PORT" -e SHARP_MODE="${SHARP_MODE:-0}" \
  -e DO_SAMPLE="${DO_SAMPLE:-0}" -e TEMP="${TEMP:-0.7}" \
  --name vga-vlm-server \
  "$IMAGE"
