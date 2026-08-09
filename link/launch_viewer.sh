#!/bin/bash
# Double-click launcher for the Four Swords 4P viewer (also fine from a shell).
# Spectates 4 agents by default; press 1-4 in the window to take a slot.
# Agents: VLM if the local server answers /health quickly, else wander bots.
exec >> "$HOME/projects/VGA/link/sessions/viewer_launch.log" 2>&1
echo "=== launch $(date) DISPLAY=$DISPLAY args: $*"
cd "$HOME/projects/VGA/link" || exit 1
source env.sh
export DISPLAY="${DISPLAY:-:1}"
AGENTS=wander
if curl -s -m 2 http://127.0.0.1:8077/health >/dev/null; then
    AGENTS=vlm
fi
echo "agents: $AGENTS"
# teardown segfault fixed 2026-07-07 (LinkSession.shutdown); rc should be clean now
"$LINK_PY" link_viewer.py --agents "$AGENTS" "$@"
echo "viewer closed rc=$?"
