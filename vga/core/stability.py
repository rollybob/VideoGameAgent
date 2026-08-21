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


class HysteresisLatch:
    """A binary latch with asymmetric confirmation (hysteresis).

    Where StreakConfirmer reports the current run length and leaves the
    thresholding to the caller, this wraps the common enter/leave pattern into a
    single boolean state: require `enter` consecutive truthy observations to flip
    OFF -> ON, and `leave` consecutive falsy observations to flip ON -> OFF. The
    asymmetry is the hysteresis -- a stray frame will not flap the state. Any
    interrupting observation resets the in-progress run.

    `enter` and `leave` are >= 1; a value of 1 flips on the first matching signal.
    """

    def __init__(self, enter: int, leave: int, start_on: bool = False) -> None:
        if enter < 1 or leave < 1:
            raise ValueError("enter and leave must be >= 1")
        self._enter = enter
        self._leave = leave
        self._on = start_on
        self._truthy_count = 0
        self._falsy_count = 0

    def update(self, signal) -> bool:
        """Record one observation; return the latch state (True=ON) afterward.

        `signal` is judged by normal Python truthiness.
        """
        if self._on:
            if not signal:
                self._falsy_count += 1
                if self._falsy_count >= self._leave:
                    self._on = False
                    self._falsy_count = 0
            else:
                self._falsy_count = 0
        else:
            if signal:
                self._truthy_count += 1
                if self._truthy_count >= self._enter:
                    self._on = True
                    self._truthy_count = 0
            else:
                self._truthy_count = 0
        return self._on

    @property
    def on(self) -> bool:
        return self._on
