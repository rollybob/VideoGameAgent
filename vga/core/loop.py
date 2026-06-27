"""
The game-agnostic main loop: capture -> perceive -> decide -> act.

This is the whole agent at the top level. It contains no game knowledge; the
plugin supplies perception and control. Keeping this tiny is the point -- all the
reasoned-about complexity lives in one plugin, behind one interface.
"""

from __future__ import annotations

import time
from typing import Optional

from .emulator import GbaEmulator
from .plugin import Plugin


def run(emulator: GbaEmulator,
        plugin: Plugin,
        logger=print,
        frame_delay: float = 0.05,
        max_frames: Optional[int] = None) -> None:
    """Drive one plugin against one emulator until interrupted (or max_frames)."""
    plugin.reset()
    logger(f"[loop] running plugin '{plugin.name}' (family '{plugin.family}'). "
           f"Ctrl+C to stop.")
    i = 0
    try:
        while True:
            frame = emulator.capture()
            state = plugin.perceive(frame, i, time.time())
            action = plugin.decide(state)
            emulator.send(action)
            i += 1
            if max_frames is not None and i >= max_frames:
                logger(f"[loop] reached max_frames={max_frames}, stopping.")
                break
            if frame_delay > 0:
                time.sleep(frame_delay)
    except KeyboardInterrupt:
        logger("\n[loop] stopped by user.")
