"""
SkillStore - persistent, mode-keyed procedural memory for the reasoner.

The whole rest of the reason layer is EPISODIC: history, subgoal, progress, the
loop/dedup buffers all live in RAM and are wiped by VlmPlugin.reset() at the end of
a run. So nothing the agent "figures out" in one episode is available in the next -
we (the humans) are the only thing that carries a lesson forward, by hand-editing a
prompt. This is why a menu-navigation bug has to be re-patched by us every time
instead of learned once.

A SkillStore is the first piece that breaks that: a small library of terse
procedural rules ("in a menu, A acts on the HIGHLIGHTED item; move the cursor
first"), keyed by a screen MODE the model self-reports (menu / dialog / battle /
...), persisted to disk as JSON. VlmPlugin retrieves the skills for the current mode
and feeds them into the prompt; a learning loop (the RAM-oracle teacher, or a
reflection pass) can call add() to write NEW skills that then persist across every
future episode. Learn-once, keep-forever - a hand-built stand-in for what a trained
policy would hold in its weights.

Design notes:
- Skills are ADVISORY hints in the prompt, not hard overrides. A mislabeled mode
  just surfaces slightly-off hints; the frame is still the source of truth.
- Retrieval is one frame behind (we key on the PREVIOUS step's reported mode), which
  matches how subgoal/progress already flow - the alternative is a two-pass
  classify-then-act that doubles latency for no real gain.
- add() dedups by fuzzy text match within a mode so a repeatedly-rediscovered lesson
  does not pile up duplicates.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# Where the store lives by default. Committed with its seed skills so a fresh clone
# starts with the hand-authored procedures; learned skills accrete into the same file.
# Override with VGA_SKILLS_PATH (e.g. to keep a scratch store out of git during eval).
DEFAULT_PATH = Path(__file__).with_name("skills.json")

# Fuzzy-dedup threshold: two skill texts in the same mode at/above this ratio are "the
# same lesson" and the new one is dropped (or merged), so add() is idempotent-ish.
_DEDUP_RATIO = 0.85

# The canonical screen modes the model self-reports. Kept small and game-agnostic.
# "unknown" is the fallback when the model does not commit to one.
MODES = ["dialog", "menu", "battle", "overworld", "shop", "title", "cutscene", "unknown"]

# Seed skills: the hand-authored procedural knowledge the agent should never have to
# re-derive. The menu-cursor rule is the fix for the current blocker (the A/B rumor
# oscillation: the agent presses A on the already-read highlighted item instead of
# moving the cursor to a different one first). These are terse on purpose - they go
# into every prompt for their mode, so they must be cheap in tokens.
SEED_SKILLS: list[dict] = [
    {
        "mode": "menu",
        "text": ("The cursor highlights exactly ONE option. Pressing A selects the "
                 "HIGHLIGHTED option only. To choose a DIFFERENT option you must FIRST "
                 "move the cursor with the D-pad (up/down/left/right) onto that option, "
                 "THEN press A. Pressing A again without moving just re-selects the same "
                 "item - if you keep landing on the same choice, MOVE the cursor first."),
    },
    {
        "mode": "menu",
        "text": ("A list you have already opened/read will not advance by re-selecting the "
                 "same entry. Move the cursor to a DIFFERENT, not-yet-tried entry, or press "
                 "B once to back out to the level above - not A on the same row."),
    },
    {
        "mode": "dialog",
        "text": ("A text/dialogue box is up. Press A to advance it. If a blinking arrow or "
                 "prompt shows there is more text, keep pressing A (repeats>1) until the box "
                 "closes; the D-pad does nothing useful while text is showing."),
    },
    {
        "mode": "dialog",
        "text": ("A Yes/No (or multi-choice) prompt starts with the cursor on a DEFAULT "
                 "option that is often not the one you want. Move the cursor to the option "
                 "you actually want BEFORE pressing A - do not assume the default is Yes."),
    },
    {
        "mode": "overworld",
        "text": ("On a map/overworld, hold a direction (repeats>1) to actually travel; a "
                 "single tap barely moves. Walk into doors, NPCs, or highlighted tiles to "
                 "interact; press A when adjacent and facing them."),
    },
    {
        "mode": "battle",
        "text": ("In a battle menu, the same cursor rule applies: A commits the HIGHLIGHTED "
                 "command/target. Move the cursor to the intended unit, command, or target "
                 "tile with the D-pad before confirming with A."),
    },
]


@dataclass
class Skill:
    """One procedural rule. `mode` keys retrieval; `text` is the prompt line. The rest
    is provenance/ranking so a learning loop can add, rank, and prune skills."""
    mode: str
    text: str
    source: str = "seed"          # "seed" | "learned" | "human"
    confidence: float = 1.0       # seed rules trusted; learned start lower and can grow
    uses: int = 0                 # times surfaced (retrieved), for future pruning
    created: float = 0.0          # unix ts; 0 for seeds


class SkillStore:
    def __init__(self, path: Optional[os.PathLike | str] = None):
        env = os.environ.get("VGA_SKILLS_PATH")
        self.path = Path(path or env or DEFAULT_PATH)
        self._skills: list[Skill] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self._skills = [Skill(**s) for s in raw]
                return
            except (json.JSONDecodeError, TypeError, OSError):
                # Corrupt/incompatible store: fall through to reseed rather than crash a run.
                pass
        # First run (or unreadable): seed from the hand-authored SEED_SKILLS and persist.
        self._skills = [Skill(**s) for s in SEED_SKILLS]
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(s) for s in self._skills], indent=2))

    def retrieve(self, mode: str, limit: int = 4) -> list[str]:
        """Return up to `limit` skill texts for `mode`, best (most confident, most used)
        first. Unknown/empty mode -> no skills (stay quiet rather than dump everything)."""
        # A/B kill-switch: VGA_SKILLS_OFF=1 makes retrieval a no-op so a run can be compared
        # skills-on vs skills-off WITHOUT touching code (honors "change one thing per run").
        if not mode or os.environ.get("VGA_SKILLS_OFF") in ("1", "true", "True"):
            return []
        mode = mode.strip().lower()
        hits = [s for s in self._skills if s.mode == mode]
        hits.sort(key=lambda s: (s.confidence, s.uses), reverse=True)
        hits = hits[:limit]
        for s in hits:   # count a retrieval as a use (for future confidence/pruning work)
            s.uses += 1
        return [s.text for s in hits]

    def add(self, mode: str, text: str, source: str = "learned",
            confidence: float = 0.5) -> bool:
        """Add a new skill for `mode`. Dedups by fuzzy text match within the mode so a
        repeatedly-rediscovered lesson does not pile up. Returns True if it was actually
        added (new), False if it duplicated an existing skill. Persists on add."""
        mode = mode.strip().lower()
        text = " ".join(text.split())
        if not mode or not text or mode not in MODES:
            return False
        for s in self._skills:
            if s.mode == mode and SequenceMatcher(None, s.text.lower(), text.lower()).ratio() >= _DEDUP_RATIO:
                return False   # already know this
        self._skills.append(Skill(mode=mode, text=text, source=source,
                                  confidence=confidence, uses=0, created=time.time()))
        self.save()
        return True

    def all(self) -> list[Skill]:
        return list(self._skills)

    def count(self) -> int:
        return len(self._skills)
