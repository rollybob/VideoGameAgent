"""
Generic temporal-stability primitives.

The single biggest source of flaky behavior in the old code was committing to a
state change on a single frame. Game transitions (a fade to black, an animation)
pass through ambiguous frames; a one-frame threshold either misses the moment or
fires on a transient. These helpers let a plugin require a signal to *persist*
before acting on it, and to make leaving a state harder than entering it
(hysteresis) so it does not flap at boundaries.

These are game-agnostic. A plugin chooses the labels and the confirmation counts.
"""

from __future__ import annotations

from typing import Hashable


class StreakConfirmer:
    """Counts how many consecutive observations have shared the same label.

    A plugin's state machine uses this to implement multi-frame confirmation with
    hysteresis: e.g. require 2 consecutive "looks-like-battle" frames to enter a
    battle, but 4 consecutive "looks-like-overworld" frames to leave one. The
    asymmetry is the hysteresis -- a stray dark frame mid-battle will not bounce
    you out.
    """

    def __init__(self) -> None:
        self._last: Hashable = None
        self._streak: int = 0

    def observe(self, label: Hashable) -> int:
        """Record one observation; return the current consecutive streak length."""
        if label == self._last:
            self._streak += 1
        else:
            self._last = label
            self._streak = 1
        return self._streak

    @property
    def label(self) -> Hashable:
        return self._last

    @property
    def streak(self) -> int:
        return self._streak

    def reset(self) -> None:
        self._last = None
        self._streak = 0
