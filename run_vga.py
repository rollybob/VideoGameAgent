"""
VGA entry point.

Wires the game-agnostic core to a selected plugin and runs the loop:

    capture -> perceive -> decide -> act

Usage:
    python run_vga.py                 # auto-select a plugin for the on-screen game
    python run_vga.py --max-frames 50 # bounded run (useful for smoke testing)

Requires mGBA running with a ROM loaded (window title contains "mGBA").
"""

from __future__ import annotations

import argparse
import sys

from vga.core import plugin as plugin_registry
from vga.core.emulator import EmulatorNotFound, GbaEmulator
from vga.core.loop import run
from vga.plugins import pokemon


def main() -> int:
    ap = argparse.ArgumentParser(description="VGA - vision-based game-playing agent")
    ap.add_argument("--title", default="mGBA", help="emulator window title keyword")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames (default: run until Ctrl+C)")
    ap.add_argument("--frame-delay", type=float, default=0.05,
                    help="seconds to sleep between frames")
    args = ap.parse_args()

    # Register available plugins. (Only Pokemon today. A new game = a new plugin
    # registered here; the core stays untouched.)
    pokemon.register()

    try:
        emu = GbaEmulator(title_keyword=args.title)
    except EmulatorNotFound as e:
        print(f"[vga] {e}")
        return 2

    # Probe one frame and let the registry pick the plugin that fits the game.
    frame = emu.capture()
    selected, score = plugin_registry.select_plugin(frame)
    if selected is None:
        print(f"[vga] no registered plugin matched the current game "
              f"(best score {score:.2f}). A future version would scaffold a new "
              f"plugin here. Nothing to run.")
        return 3

    print(f"[vga] selected plugin '{selected.name}' (score {score:.2f})")
    run(emu, selected, frame_delay=args.frame_delay, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
