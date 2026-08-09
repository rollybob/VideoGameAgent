#!/bin/bash
# One-shot: runs at Tim's next GUI login (autostart), marks the desktop icon
# trusted so double-click works with no "Allow Launching" prompt, then removes
# its own autostart entry. Needed because the flag can only be set from inside
# a desktop session (gvfs metadata needs the session bus).
gio set "$HOME/Desktop/Four Swords 4P.desktop" metadata::trusted true \
  && rm -f "$HOME/.config/autostart/trust-four-swords-icon.desktop"
