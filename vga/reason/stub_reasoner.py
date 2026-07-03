"""
A deterministic reasoner for offline plumbing tests - no network, no API key.

It cycles a fixed script of inputs keyed to the step count, so the whole loop
(capture -> perceive -> decide -> act -> log a transition) can be exercised end to
end without the emulator's behavior or a live model. Useful for validating the
trajectory logger and the Action mapping.
"""

from __future__ import annotations

import numpy as np

from .base import Reasoner, ReasonContext, ReasonerDecision, action_from_choice

# A harmless, legible cycle: advance, wait, poke around.
_SCRIPT = ["A", "wait", "down", "A", "right", "wait", "B", "up"]


class StubReasoner:
    name = "stub"

    def __init__(self, script: list[str] | None = None):
        self._script = script or _SCRIPT

    def decide(self, frame_bgr: np.ndarray, context: ReasonContext) -> ReasonerDecision:
        choice = self._script[context.step % len(self._script)]
        reason = f"stub script step {context.step}: {choice}"
        return ReasonerDecision(
            action=action_from_choice(choice, note=reason[:60]),
            reason=reason,
            meta={"model": "stub", "button": choice, "repeats": 1},
        )

    def reset(self) -> None:
        pass
