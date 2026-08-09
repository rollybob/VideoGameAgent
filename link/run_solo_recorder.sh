#!/bin/bash
# Launch the solo ALttP human-play recorder on the :1 desktop (2026-08-03).
# Wraps env-sourcing + LINK_PY + DISPLAY so it can run detached under tmux while
# Tim plays. Args pass through to solo_recorder.py -- e.g. a long-run buffer
# (--bank-secs 20 --banks 120 = ~40 min) so F12 dumps the whole session from the
# start (the deque never fills on a shorter run, so its oldest bank stays tick 0).
cd "$(dirname "$0")"
source env.sh
exec env DISPLAY=:1 "$LINK_PY" solo_recorder.py "$@"
