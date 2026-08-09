"""
SkillStore - persistent procedural memory for the reasoner, retrieved SEMANTICALLY.

The whole rest of the reason layer is EPISODIC: history, subgoal, progress, the
loop/dedup buffers all live in RAM and are wiped by VlmPlugin.reset() at the end of
a run. So nothing the agent "figures out" in one episode is available in the next -
we (the humans) are the only thing that carries a lesson forward, by hand-editing a
prompt. This is why a menu-navigation bug has to be re-patched by us every time
instead of learned once.

A SkillStore is the first piece that breaks that: a small library of terse
procedural rules ("in a menu, A acts on the HIGHLIGHTED item; move the cursor
first"), each carrying a free-text `context` describing the SITUATION it applies to,
persisted to disk as JSON. VlmPlugin retrieves the skills whose context best overlaps
the current scene and feeds them into the prompt; a learning loop (the RAM-oracle
teacher, or a reflection pass) can call add() to write NEW skills that then persist
across every future episode. Learn-once, keep-forever - a hand-built stand-in for
what a trained policy would hold in its weights.

Retrieval is SEMANTIC, not a discrete key (Task 02, 2026-07-05). It used to be an
EXACT string match on the previous step's self-reported MODE against a fixed
game-specific whitelist (menu / dialog / battle / overworld / shop / ...). That
CLIFFED to [] the moment the mode was wrong (VLM ~10% error), "unknown" (~7% of real
frames), or just a vocab slip ("overworld" store vs "field" report) - a 10%
perception error became a 100% retrieval failure on those frames, and it baked an RPG
ontology into the core (a platformer/racer/puzzle game does not decompose that way).
Now skills are ranked by keyword overlap between the current scene context (the VLM's
free-text descriptor + on-screen text, with the soft mode as ONE token among many)
and each skill's stored context - the same path KnowledgeStore uses
(vga/reason/retrieval.py). Mode is demoted from a hard retrieval KEY to one soft
QUERY token; it still flows UNCHANGED into the prompt (the mode-anchoring finding
proved the prompt prior is load-bearing - this task only changes how retrieval
CONSUMES the mode).

Design notes:
- Skills are ADVISORY hints in the prompt, not hard overrides. A mislabeled mode just
  nudges ranking now instead of zeroing retrieval; the frame is still the source of truth.
- Retrieval is one frame behind (context uses the PREVIOUS step's reported mode), which
  matches how subgoal/progress already flow - the alternative is a two-pass
  classify-then-act that doubles latency for no real gain.
- No fixed genre taxonomy in the retrieval hot path. The query is free text from the
  general VLM; seed skills describe situations in words, not per-game menu paths.
- add() dedups by fuzzy text match so a repeatedly-rediscovered lesson does not pile up.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from .retrieval import rank_by_overlap, tokens

# Where the store lives by default. Committed with its seed skills so a fresh clone
# starts with the hand-authored procedures; learned skills accrete into the same file.
# Override with VGA_SKILLS_PATH (e.g. to keep a scratch store out of git during eval).
DEFAULT_PATH = Path(__file__).with_name("skills.json")

# Fuzzy-dedup threshold: two skill texts at/above this ratio are "the same lesson" and
# the new one is dropped, so add() is idempotent-ish.
_DEDUP_RATIO = 0.85

# Back-compat / migration only: a generic free-text SITUATION phrase per legacy mode
# label, used to synthesize a `context` for skills loaded from an old (mode-only) store
# or add()ed with a bare mode. NOT a retrieval whitelist - any mode string is accepted;
# an unmapped one just seeds context with the raw word. Deliberately general (no genre
# assumption leaks into retrieval beyond these situation words).
# Synonym-rich on purpose: retrieval matches by TOKEN overlap, so each phrase must
# literally contain the words a VLM is likely to emit for that situation (both the mode
# label itself and common near-synonyms - dialog/dialogue, overworld/field/map,
# battle/combat/fight). That is what makes retrieval robust to a vocab slip: a skill is
# reachable whether the model says "overworld" or "field".
_MODE_CONTEXT: dict[str, str] = {
    "menu": "a menu, list, or set of selectable options or choices; moving a cursor to highlight and confirm an item",
    "dialog": "a dialog or dialogue text box, message, conversation, yes/no question, or multiple-choice prompt",
    "battle": "a battle, fight, or combat; choosing a command, action, unit, or target on the field or grid",
    "overworld": "the overworld, world map, field, or town; walking, moving, and travelling to doors, npcs, or tiles",
    "shop": "a shop or store; buying and selling items from a list menu",
    "title": "a title, start, or main menu screen before play begins",
    "cutscene": "a cutscene, scene, or non-interactive scripted sequence",
    "unknown": "",
}


def _mode_to_context(mode: str) -> str:
    """Generic situation phrase for a (legacy) mode label; the raw word if unmapped."""
    mode = (mode or "").strip().lower()
    return _MODE_CONTEXT.get(mode, mode)


# Seed skills: the hand-authored procedural knowledge the agent should never have to
# re-derive. Each carries a free-text `context` (the SITUATION it applies to, in general
# words - NOT a per-game menu path) that retrieval overlaps against the current scene, so
# the right skill surfaces even when the reported mode is wrong or a different word. The
# menu-cursor rule is the fix for the A/B oscillation (pressing A on the already-read
# highlighted item instead of moving the cursor first). Terse on purpose - they go into
# the prompt, so they must be cheap in tokens. `mode` is kept as ONE token folded into
# context for back-compat and as a soft signal; it is no longer a retrieval key.
SEED_SKILLS: list[dict] = [
    {
        "mode": "menu",
        "context": ("a menu, list, or set of selectable options or choices; moving the "
                    "cursor between choices and pressing A to confirm the highlighted one"),
        "text": ("The cursor highlights exactly ONE option. Pressing A selects the "
                 "HIGHLIGHTED option only. To choose a DIFFERENT option you must FIRST "
                 "move the cursor with the D-pad (up/down/left/right) onto that option, "
                 "THEN press A. Pressing A again without moving just re-selects the same "
                 "item - if you keep landing on the same choice, MOVE the cursor first."),
    },
    {
        "mode": "menu",
        "context": ("a menu or list already opened, cursor on an entry that has been tried; "
                    "needing a different option or choice, or to back out of the list"),
        "text": ("A list you have already opened/read will not advance by re-selecting the "
                 "same entry. Move the cursor to a DIFFERENT, not-yet-tried entry, or press "
                 "B once to back out to the level above - not A on the same row."),
    },
    {
        "mode": "dialog",
        "context": ("a dialog or dialogue text box showing on-screen message or conversation "
                    "text, possibly a blinking more arrow; advancing the text"),
        "text": ("A text/dialogue box is up. Press A to advance it. If a blinking arrow or "
                 "prompt shows there is more text, keep pressing A (repeats>1) until the box "
                 "closes; the D-pad does nothing useful while text is showing."),
    },
    {
        "mode": "dialog",
        "context": ("a dialog yes or no question or multiple-choice prompt; accept, decline, "
                    "confirm, or answer a choice with the cursor on a default option"),
        "text": ("A Yes/No (or multi-choice) prompt starts with the cursor on a DEFAULT "
                 "option that is often not the one you want. Move the cursor to the option "
                 "you actually want BEFORE pressing A - do not assume the default is Yes."),
    },
    {
        "mode": "overworld",
        "context": ("the overworld, world map, field, or town; walking, moving, travelling, "
                    "and interacting with doors, npcs, or highlighted tiles"),
        "text": ("On a map/overworld, hold a direction (repeats>1) to actually travel; a "
                 "single tap barely moves. Walk into doors, NPCs, or highlighted tiles to "
                 "interact; press A when adjacent and facing them."),
    },
    {
        "mode": "battle",
        "context": ("a battle, fight, or combat; choosing a command, action, unit, or target "
                    "tile from a cursor before confirming the action"),
        "text": ("In a battle menu, the same cursor rule applies: A commits the HIGHLIGHTED "
                 "command/target. Move the cursor to the intended unit, command, or target "
                 "tile with the D-pad before confirming with A."),
    },
]


@dataclass
class Skill:
    """One procedural rule. `context` is the free-text SITUATION it applies to (what
    retrieval overlaps against); `text` is the prompt line. `mode` is kept as a legacy /
    soft token folded into the retrieval context, NOT a hard key. The rest is
    provenance/ranking so a learning loop can add, rank, and prune skills."""
    mode: str
    text: str
    context: str = ""             # free-text situation; retrieval matches against this + text
    source: str = "seed"          # "seed" | "learned" | "human"
    confidence: float = 1.0       # seed rules trusted; learned start lower and can grow
    uses: int = 0                 # times surfaced (retrieved), for future pruning
    created: float = 0.0          # unix ts; 0 for seeds

    def searchable(self) -> str:
        """What retrieval overlaps the scene context against: the situation `context` ONLY,
        never the rule `text`. The doc specifies matching each skill's stored CONTEXT; matching
        the verbose rule text instead lets cross-cutting words leak (the battle rule says "in a
        battle menu", so it wrongly surfaced on plain menus) and lets garbled OCR coincidentally
        hit a rule, surfacing an off-target skill. The context is a clean per-situation phrase, so
        overlap here means the SITUATION matches, not that a word happened to appear in a rule."""
        return self.context


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
                # Migrate an old (mode-only) store: backfill a free-text context from the
                # legacy mode so semantic retrieval has something to match. Persist once so
                # the migration is durable and does not re-run every load.
                migrated = False
                for s in self._skills:
                    if not (s.context or "").strip():
                        s.context = _mode_to_context(s.mode)
                        migrated = True
                if migrated:
                    self.save()
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

    def retrieve(self, context: str, limit: int = 4) -> list[str]:
        """Return up to `limit` skill texts whose stored situation best overlaps `context`
        (the free-text scene descriptor: mode word + on-screen text + any VLM scene text),
        best (most overlap, then confidence, then uses) first. No content tokens in the
        context -> no skills (stay quiet rather than dump generic advice). This is the
        SEMANTIC path (Task 02): a wrong or "unknown" mode no longer cliffs retrieval to
        [] - the skill still surfaces if its situation words overlap the scene.

        Back-compat: the old signature was retrieve(mode); a bare mode word still works
        because it is folded into the context via _mode_to_context below."""
        # A/B kill-switch: VGA_SKILLS_OFF=1 makes retrieval a no-op so a run can be compared
        # skills-on vs skills-off WITHOUT touching code (honors "change one thing per run").
        if os.environ.get("VGA_SKILLS_OFF") in ("1", "true", "True"):
            return []
        context = (context or "").strip()
        # A/B baseline: VGA_SKILLS_EXACT=1 reproduces the LEGACY exact-mode-match retrieval so
        # the ladder can measure the semantic mechanism against the old one with everything else
        # code-identical (the mode is the first token of the context the plugin builds,
        # f"{mode} {dialog_text}"). Isolates "new mechanism" from "skills at all". Remove once
        # Task 02 is measured; kept only for the clean A/B.
        if os.environ.get("VGA_SKILLS_EXACT") in ("1", "true", "True"):
            mode = (context.split() or [""])[0].lower()
            if not mode:
                return []
            hits = sorted((s for s in self._skills if s.mode == mode),
                          key=lambda s: (s.confidence, s.uses), reverse=True)[:limit]
            for s in hits:
                s.uses += 1
            return [s.text for s in hits]
        # No content tokens in the context -> stay quiet (do not dump generic advice). This is
        # the SkillStore-specific threshold; a bare/garbled scene returns []. Otherwise rank by
        # overlap exactly like KnowledgeStore. The seed contexts are synonym-rich (see
        # _MODE_CONTEXT / SEED_SKILLS) so the mode word the VLM emits ("dialog", "field") matches
        # directly - that, not a query rewrite, is what removes the exact-match cliff.
        if not tokens(context):
            return []
        hits = rank_by_overlap(self._skills, context,
                               searchable=lambda s: s.searchable(),
                               tiebreak=lambda s: (s.confidence, s.uses), limit=limit)
        for s in hits:   # count a retrieval as a use (for future confidence/pruning work)
            s.uses += 1
        return [s.text for s in hits]

    def add(self, mode: str, text: str, source: str = "learned",
            confidence: float = 0.5, context: str = "") -> bool:
        """Add a new skill. `context` is the free-text SITUATION it applies to (what
        retrieval matches); if omitted it is synthesized from `mode` for back-compat. No
        fixed-taxonomy gate any more - any mode string is accepted. Dedups by fuzzy text
        match across the store so a repeatedly-rediscovered lesson does not pile up. Returns
        True if it was actually added (new), False if it duplicated an existing skill.
        Persists on add."""
        mode = (mode or "").strip().lower()
        text = " ".join(text.split())
        context = " ".join((context or "").split()) or _mode_to_context(mode)
        if not text:
            return False
        for s in self._skills:
            if SequenceMatcher(None, s.text.lower(), text.lower()).ratio() >= _DEDUP_RATIO:
                return False   # already know this
        self._skills.append(Skill(mode=mode, text=text, context=context, source=source,
                                  confidence=confidence, uses=0, created=time.time()))
        self.save()
        return True

    def all(self) -> list[Skill]:
        return list(self._skills)

    def count(self) -> int:
        return len(self._skills)
