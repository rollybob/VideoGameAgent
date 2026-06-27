"""
Pokemon plugin (GBA, FireRed/LeafGreen engine family).

Bundles PokemonPerception + PokemonStrategy behind the core Plugin interface and
registers itself on import. The core never imports this module; the entry point
does (which triggers registration).
"""

from __future__ import annotations

import numpy as np

from vga.core.contract import Action, GameState
from vga.core import plugin as plugin_registry
from .perception import PokemonPerception
from .strategy import PokemonStrategy


class PokemonPlugin:
    name = "pokemon"
    family = "gba-pokemon-frlg"

    def __init__(self, logger=print, enable_ocr: bool = True) -> None:
        self.logger = logger
        self.perception = PokemonPerception(logger=logger, enable_ocr=enable_ocr)
        self.strategy = PokemonStrategy(logger=logger)

    def matches(self, frame: np.ndarray) -> float:
        return self.perception.matches(frame)

    def perceive(self, frame: np.ndarray, frame_index: int, timestamp: float) -> GameState:
        return self.perception.perceive(frame, frame_index, timestamp)

    def decide(self, state: GameState) -> Action:
        return self.strategy.decide(state)

    def reset(self) -> None:
        self.strategy.reset()


def register(logger=print) -> PokemonPlugin:
    """Instantiate and register the Pokemon plugin. Returns the instance."""
    inst = PokemonPlugin(logger=logger)
    plugin_registry.register(inst)
    return inst
