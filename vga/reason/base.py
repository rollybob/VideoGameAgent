"""
The Reasoner interface and the button<->string mapping shared by every backend.

A Reasoner sees a frame (BGR, as core/emulator.capture() produces) plus a small
text context, and returns a ReasonerDecision: one abstract Action, the model's
stated reason, and backend metadata for logging. Deliberately no pixels-out and
no game state - the reasoner only ever consumes the frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

from ..core.contract import Action, Button


# The abstract controller vocabulary the reasoner is allowed to emit, as strings.
# "wait" is the no-op (do nothing this decision). Every other value is a Button.
WAIT = "wait"
BUTTON_BY_NAME: dict[str, Button] = {b.value: b for b in Button}
BUTTON_CHOICES: list[str] = [b.value for b in Button] + [WAIT]


def action_from_choice(button: str, repeats: int = 1, note: str = "") -> Action:
    """Map a reasoner's string choice to an Action. Unknown -> wait (safe no-op)."""
    if button == WAIT or button not in BUTTON_BY_NAME:
        return Action.wait(note=note)
    btn = BUTTON_BY_NAME[button]
    if repeats and repeats > 1:
        return Action.mash(btn, repeats=int(repeats), note=note)
    return Action.press(btn, note=note)


@dataclass
class ReasonerDecision:
    """One reasoner step: the Action to take, why, and backend metadata (usage,
    raw model output, latency) kept for trajectory logging."""
    action: Action
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasonContext:
    """What the reasoner is told about the situation beyond the raw frame. Kept
    minimal on purpose - the frame is the signal; this is light framing.

    `history` and `last_changed` are the Phase-A working memory (2026-07-02): a
    stateless per-frame VLM re-derives the same choice every step, so it scrolls in
    tiny increments and re-reads dialogs it already saw. We hand it a short rolling
    log of its own recent EXECUTED actions and whether each one actually changed the
    screen, so it can tell "I pressed down 5x and nothing moved" from a fresh choice.
    """
    goal: str = ""
    step: int = 0
    last_action: str = ""
    # Did the screen change after the most recent executed action? None = unknown
    # (first step, or no prior frame to diff against).
    last_changed: Optional[bool] = None
    # Rolling log of recent executed steps, oldest first. Each entry:
    #   {"button": str, "reason": str, "changed": bool}
    history: list[dict] = field(default_factory=list)
    # True when the recent screens are dominated by a small set that keeps recurring - the
    # agent is spinning in place (cycling a menu, an A/B bounce) rather than progressing.
    # Detected by loop DENSITY over a short window, so a one-off intentional revisit (e.g.
    # re-entering a hub area to take a mission) does NOT trip it - only genuine loops do.
    # This is the "going in circles" signal that frame-change memory (last_changed) misses,
    # because each step legitimately changes the screen. Added 2026-07-02; hardened same day.
    looping: bool = False
    # OCR'd text of the current dialogue/menu region (bottom strip), and whether that text
    # was already read recently. already_read catches the "re-reading a dialog" loop that
    # the pixel signals miss: text can recur while the surrounding scene differs (full-frame
    # revisit won't match) yet each page still changes the screen (last_changed won't fire).
    # Added 2026-07-02 (step 2). Tesseract-based; advisory only.
    dialog_text: str = ""
    already_read: bool = False
    # Goal/task-state scratchpad (2026-07-02, layer b): the agent's OWN current sub-goal and
    # progress note, authored by the model and carried across steps so decisions are goal-
    # directed instead of purely reactive. These are the values from the PREVIOUS step, fed
    # back in; the model returns updated ones each step. Empty on the first step (the model
    # sets the first sub-goal from the objective + screen).
    subgoal: str = ""
    progress: str = ""
    # Perception-first screen MODE the model self-reported on the PREVIOUS step
    # (dialog/menu/battle/overworld/...), and the persistent procedural skills the
    # SkillStore returned for that mode. Skills are terse how-to hints ("in a menu, A
    # acts on the HIGHLIGHTED item - move the cursor first") that persist across
    # episodes, unlike everything else here. Retrieval is one frame behind (keyed on
    # the last reported mode), matching how subgoal/progress already flow. Added
    # 2026-07-02 (skill store): the first reason-layer state that is learn-once,
    # keep-forever rather than wiped by reset().
    mode: str = ""
    skills: list[str] = field(default_factory=list)
    # Declarative facts the agent has LEARNED from this game's tutorials (KnowledgeStore),
    # retrieved for the current context. Fed into the prompt as BACKGROUND reference (not
    # facts about the current screen - the echo/anchor framing rule). The persistent,
    # game-keyed, learn-once-keep-forever world-knowledge side of tutorial-learning
    # (2026-07-02), sibling to the procedural `skills`. See vga/reason/knowledge.py.
    knowledge: list[str] = field(default_factory=list)
    # Durable COMMITMENT (2026-07-04, Task 01 intent-persistence). A bounded, FOREGROUND
    # directive the plugin locks in when the agent reflexively falls back on the per-frame
    # visual walk-prior WHILE it holds relevant learned knowledge that implies a different
    # move. Unlike `subgoal` above (advisory, framed "may be out of date - trust the screen"),
    # this is presented as an ACTIONABLE plan to carry out NOW, weighted ABOVE the reflex,
    # for a few steps or until the screen materially changes (e.g. a menu opens). It is the
    # persistence the validated experiential loop lacked: a mined fact influenced ONE step
    # then got overridden by the walk-prior (day-2 bench-travel-learned). Empty when no
    # commitment is active. Bounded tightly on purpose (anti-anchor). See plugin.py.
    committed: str = ""
    # Durable, RAM-TAUGHT task-state PHASE directive (2026-07-06, Task 07 task-state memory,
    # docs/revitalization/07_task_state_memory.md). Where the agent is in a MULTI-STEP task
    # ("you have ALREADY accepted the mission; now travel"), latched by an OBJECTIVE pixels-only
    # cue and fed as AUTHORITATIVE context. Distinct from everything above: `subgoal` is the
    # model's own free-text note (wiped on scene change, framed "may be stale"); `committed` is
    # a knowledge-derived reflex override bounded to a few steps and CLEARED on a material scene
    # change. This one is DURABLE - it survives the scratchpad wipe AND scene changes and holds
    # for the rest of the task once latched, because the failure it fixes spans ~77 steps (after
    # accepting, the stateless agent forgets and re-tries to accept / re-enters the pub - it lost
    # the accept->travel thread; diagnostic 2026-07-06). It is anti-anchor's deliberate opposite,
    # justified ONLY because the phase is set by an objective RAM-VALIDATED cue (the info-fee
    # dialog == accept committed), not the model's own possibly-wrong belief. RAM is the TRAIN-
    # time teacher for the cue only; at inference the phase comes from pixels (OCR), never RAM.
    # Empty until the phase advances past its initial state / when the mechanism is off. See
    # plugin.py (self._phase).
    task_phase: str = ""


class Reasoner(Protocol):
    name: str

    def decide(self, frame_bgr: np.ndarray, context: ReasonContext) -> ReasonerDecision:
        """Choose one Action from the frame. No side effects."""
        ...

    def reset(self) -> None:
        """Clear any per-episode state."""
        ...
