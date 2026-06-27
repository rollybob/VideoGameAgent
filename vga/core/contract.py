"""
The hard data contract between perception and control.

Perception produces a GameState from a frame. Control consumes a GameState and
emits an Action. Neither side knows the other's internals: perception says nothing
about *how* it decided, control says nothing about pixels.

Deliberately, GameState carries NO game-semantic label (no "battle"/"overworld").
That interpretation is the plugin's job. The core only moves frames and buttons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class Button(Enum):
    """Console-level inputs. Game-agnostic: these are the buttons a GBA has, not
    anything about what they mean in a given game."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    A = "A"
    B = "B"
    START = "start"
    SELECT = "select"
    L = "L"
    R = "R"


@dataclass(frozen=True)
class Action:
    """A discrete input the agent issues. button=None means do nothing (wait).

    repeats/gap allow a single decision to express a short mash without the control
    layer having to loop the emulator itself.
    """
    button: Optional[Button] = None
    duration: float = 0.12
    repeats: int = 1
    gap: float = 0.10
    note: str = ""  # human-readable reason, for logs only

    @classmethod
    def wait(cls, note: str = "") -> "Action":
        return cls(button=None, note=note)

    @classmethod
    def press(cls, button: Button, duration: float = 0.12, note: str = "") -> "Action":
        return cls(button=button, duration=duration, note=note)

    @classmethod
    def mash(cls, button: Button, repeats: int = 3, duration: float = 0.05,
             gap: float = 0.10, note: str = "") -> "Action":
        return cls(button=button, duration=duration, repeats=repeats, gap=gap, note=note)

    def __str__(self) -> str:
        name = self.button.value if self.button else "wait"
        tag = f"{name}x{self.repeats}" if self.repeats > 1 else name
        return f"{tag}({self.note})" if self.note else tag


@dataclass(frozen=True)
class Detection:
    """One detected object. Produced by a perception backend (deterministic CV now,
    a trained detector later) behind the same interface."""
    label: str
    bbox: tuple[int, int, int, int]  # x, y, w, h in frame pixels
    confidence: float


@dataclass
class GameState:
    """Everything control is allowed to see about one frame.

    `regions` and `text` are populated by the plugin's perception with whatever
    aggregate features and validated OCR fields that game needs. `objects` is the
    generic detector output. `frame` is included for pragmatic reasons (e.g.
    frame-to-frame similarity); control must treat it as read-only and should
    prefer `regions`/`objects`/`text` for decisions.
    """
    frame_index: int
    timestamp: float
    width: int
    height: int
    objects: list[Detection] = field(default_factory=list)
    text: dict[str, str] = field(default_factory=dict)
    regions: dict[str, Any] = field(default_factory=dict)
    frame: Optional[np.ndarray] = None
