"""
Pokemon control: GameState -> Action.

The core of the rebuild is here: an explicit two-state machine (OVERWORLD <-> BATTLE)
that commits transitions only after multi-frame confirmation, with hysteresis.

Why this fixes the long-standing bug (battle does not switch back to exploration
when it ends): a fade-to-black happens on BOTH sides of a battle, so a single
black-frame trigger is a directionless edge -- it cannot tell "entering" from
"leaving", and is easily missed between loop sleeps. Instead we ignore the
transient fade and commit to a side only once the STABLE scene on that side is
confirmed: 2 consecutive battle frames to enter, 4 consecutive overworld frames to
leave. The asymmetry (hysteresis) means an animation or dark frame mid-battle will
not knock us out of BATTLE.

Battle and exploration policies are intentionally simple in v1 (Tim's call: close
the loop first). Each step() returns exactly one Action; richer move/PP management
and real navigation are marked TODO for live tuning against mGBA.
"""

from __future__ import annotations

import numpy as np

from vga.core.contract import Action, Button, GameState
from vga.core.stability import StreakConfirmer

ENTER_BATTLE_CONFIRM = 2   # frames of battle evidence to commit to BATTLE
EXIT_BATTLE_CONFIRM = 4    # frames of overworld evidence to commit back (hysteresis)

# Frame-similarity at/above this counts as "no visible change" (stuck).
STUCK_SIM = 0.985


def _similarity(a, b) -> float:
    """1.0 == identical. Operates on the small grayscale fingerprints."""
    if a is None or b is None:
        return 0.0
    da = a.astype(np.float32) / 255.0
    db = b.astype(np.float32) / 255.0
    mse = float(np.mean((da - db) ** 2))
    return 1.0 - mse


class _BattlePolicy:
    """v1: advance the battle by pressing A (in FRLG the default cursor chains
    FIGHT -> move 1 -> text), with a stuck-breaker that backs out with B if the
    screen stops changing (e.g. an out-of-PP move leaving the cursor stuck).

    TODO(live): port the move-cycling / PP-tracking / party-switch logic from the
    old battle_logic_pokemon_v2 once we can watch it against a real battle.
    """
    def __init__(self, logger=print) -> None:
        self.logger = logger
        self.stuck = 0

    def reset(self) -> None:
        self.stuck = 0

    def on_start(self) -> None:
        self.stuck = 0
        self.logger("[battle] new battle")

    def step(self, gs: GameState, sim: float) -> Action:
        if sim >= STUCK_SIM:
            self.stuck += 1
        else:
            self.stuck = 0
        if self.stuck >= 6:
            self.stuck = 0
            return Action.press(Button.B, note="battle stuck -> back out")
        return Action.mash(Button.A, repeats=2, note="battle advance")


class _ExplorePolicy:
    """v1: walk a direction; if the screen stops changing for a while, rotate to a
    new direction and tap A to interact. Weak but functional.

    TODO(live): real navigation (goal-directed pathing) is a known hard part and is
    out of scope for closing the loop.
    """
    DIRECTIONS = (Button.UP, Button.RIGHT, Button.DOWN, Button.LEFT)

    def __init__(self, logger=print) -> None:
        self.logger = logger
        self.dir_idx = 0
        self.stuck = 0
        self.steps = 0

    def reset(self) -> None:
        self.dir_idx = 0
        self.stuck = 0
        self.steps = 0

    def step(self, gs: GameState, sim: float) -> Action:
        self.steps += 1
        if sim >= STUCK_SIM:
            self.stuck += 1
        else:
            self.stuck = 0
        if self.stuck >= 8:
            self.stuck = 0
            self.dir_idx = (self.dir_idx + 1) % len(self.DIRECTIONS)
            self.logger(f"[explore] stuck -> turn {self.DIRECTIONS[self.dir_idx].value}")
        # Occasionally interact (talk / pick up / advance signs).
        if self.steps % 7 == 0:
            return Action.press(Button.A, note="explore interact")
        return Action.press(self.DIRECTIONS[self.dir_idx], duration=0.16, note="explore walk")


class PokemonStrategy:
    def __init__(self, logger=print) -> None:
        self.logger = logger
        self.confirmer = StreakConfirmer()
        self.state = "overworld"          # committed FSM state
        self.prev_fp = None
        self.battle = _BattlePolicy(logger)
        self.explore = _ExplorePolicy(logger)

    def reset(self) -> None:
        self.confirmer.reset()
        self.state = "overworld"
        self.prev_fp = None
        self.battle.reset()
        self.explore.reset()

    def _maybe_transition(self) -> None:
        label = self.confirmer.label
        streak = self.confirmer.streak
        if self.state == "overworld":
            if label == "battle" and streak >= ENTER_BATTLE_CONFIRM:
                self.state = "battle"
                self.battle.on_start()
        elif self.state == "battle":
            if label == "overworld" and streak >= EXIT_BATTLE_CONFIRM:
                self.state = "overworld"
                self.logger("[battle] ended -> overworld")

    def decide(self, state: GameState) -> Action:
        scene = state.regions.get("scene_raw", "other")
        self.confirmer.observe(scene)
        self._maybe_transition()

        fp = state.regions.get("fingerprint")
        sim = _similarity(self.prev_fp, fp)
        self.prev_fp = fp

        if self.state == "battle":
            return self.battle.step(state, sim)

        # Overworld. Ambiguous frames (dialogue/fades/menus) -> advance with A.
        if scene == "other":
            return Action.mash(Button.A, repeats=2, note="advance dialogue/transition")
        return self.explore.step(state, sim)
