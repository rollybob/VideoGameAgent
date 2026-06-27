"""
The perception<->control boundary, and the plugin registry that keeps the core
game-agnostic.

A Plugin bundles a game family's perception (frame -> GameState) and control
(GameState -> Action). The core never imports a plugin; plugins register
themselves (see vga/plugins/pokemon). Selection is by `matches()`, which scores
how well a plugin fits the game currently on screen.

Future growth (NOT built now): a plugin scopes to a game *family* -- one Pokemon
plugin spans the Pokemon titles. When no registered plugin matches an unfamiliar
game, that is the seam where an agent could scaffold a brand-new plugin. Today,
select_plugin simply returns (None, 0.0) and the caller reports it. The door is
open; we are not walking through it yet.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .contract import Action, GameState


@runtime_checkable
class Plugin(Protocol):
    name: str       # e.g. "pokemon"
    family: str     # e.g. "gba-pokemon-frlg" -- the game family this plugin covers

    def matches(self, frame: np.ndarray) -> float:
        """Confidence in [0,1] that this plugin handles the game on screen."""
        ...

    def perceive(self, frame: np.ndarray, frame_index: int, timestamp: float) -> GameState:
        """Turn a frame into a GameState. No control decisions here."""
        ...

    def decide(self, state: GameState) -> Action:
        """Turn a GameState into one Action. No pixel access beyond state.frame."""
        ...

    def reset(self) -> None:
        """Clear per-episode state (called at loop start)."""
        ...


_REGISTRY: list[Plugin] = []


def register(plugin: Plugin) -> Plugin:
    """Register a plugin instance. Returns it so it can be used as a decorator-ish
    one-liner at import time."""
    _REGISTRY.append(plugin)
    return plugin


def registered() -> list[Plugin]:
    return list(_REGISTRY)


def select_plugin(frame: np.ndarray, min_score: float = 0.3) -> tuple[Optional[Plugin], float]:
    """Pick the best-matching registered plugin for the current frame.

    Returns (plugin, score). If nothing clears min_score, returns (None, score) --
    the caller decides what to do (today: report and stop; future: scaffold a new
    plugin for this game).
    """
    best: Optional[Plugin] = None
    best_score = 0.0
    for p in _REGISTRY:
        try:
            s = float(p.matches(frame))
        except Exception:
            s = 0.0
        if s > best_score:
            best, best_score = p, s
    if best_score < min_score:
        return None, best_score
    return best, best_score
