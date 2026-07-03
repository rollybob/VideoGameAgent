#!/usr/bin/env bash
# Drop the agent into FFTA from the START on the watchable :1 desktop, with tutorial-learning on.
# The mGBA window is already open at the FFTA title screen on :1; run_reasoner attaches to it by
# title. A dedicated KnowledgeStore (fresh per run) lets us see exactly what THIS run learns.
set -uo pipefail
export DISPLAY=:1
export VGA_KNOWLEDGE_PATH="$HOME/projects/VGA/sessions/ffta-newgame-knowledge.json"
cd "$HOME/projects/VGA"

# The KnowledgeStore PERSISTS across runs (learn-once, keep-forever) - do NOT wipe it on
# relaunch. To start fresh, delete $VGA_KNOWLEDGE_PATH by hand before launching.

exec .venv/bin/python run_reasoner.py \
  --backend local --learn-tutorials \
  --game "Final Fantasy Tactics Advance" \
  --goal "You are playing Final Fantasy Tactics Advance from the very beginning. Select New Game and play through the opening. Read every tutorial, message, and instruction the game shows, and follow what it tells you. Advance dialogue and confirm choices with A; in a menu, move the cursor with the D-pad onto the option you want BEFORE pressing A." \
  --max-steps 600 \
  --run-dir "sessions/ffta-newgame-$(date +%s)"
