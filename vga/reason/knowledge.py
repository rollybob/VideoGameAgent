"""
KnowledgeStore - persistent, GAME-keyed DECLARATIVE memory for the reasoner.

Sibling to the SkillStore (vga/reason/skills.py), but a different KIND of memory:
- SkillStore holds PROCEDURAL rules keyed by screen MODE ("in a menu, move the cursor
  before pressing A") - how to operate the UI.
- KnowledgeStore holds DECLARATIVE FACTS keyed by GAME ("win/lose conditions vary by
  engagement; check Mission in the command menu") - what is TRUE about this game's world.

This is the persistent side of tutorial-learning (docs/TUTORIAL_LEARNING_PLAN.md): when the
agent reads an in-game tutorial/help/encyclopedia screen, the VLM distills it into a compact
fact and add()s it here; retrieve() surfaces the game's known facts back into the /act prompt
as BACKGROUND KNOWLEDGE. Learn-once, keep-forever - a hand-built stand-in for the world-model
priors a trained agent would hold in its weights, and the seed of the System-2 world model in
the north star.

Design notes:
- Facts are ADVISORY background, never a hard override of the current screen. The framing at
  retrieval time must present them as reference knowledge ("things you have learned about this
  game"), NOT as facts about the current frame - the echo/anchor bug (burned twice) shows any
  fed-back self-authored text that is presented as authoritative gets parroted. Declarative
  world-facts are safer than fed-back goals, but the framing discipline still applies.
- Keyed by GAME (ROM basename), because tutorial knowledge is game-global, not screen-local.
- add() dedups by fuzzy text match within a game so a re-read tutorial does not pile up.
- retrieve() can keyword-rank against a context string (current mode + on-screen text) so a
  large store still surfaces only the relevant facts within the prompt's token budget.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from .retrieval import rank_by_overlap

# Default store location. Starts EMPTY (no seed facts) - unlike skills, knowledge is meant to
# be LEARNED from the game itself. Override with VGA_KNOWLEDGE_PATH (e.g. a scratch store per
# eval so a run does not pollute the shared knowledge, mirroring VGA_SKILLS_PATH).
DEFAULT_PATH = Path(__file__).with_name("knowledge.json")

# Two fact texts for the same game at/above this ratio are "the same lesson" -> the new one is
# dropped, so re-reading a tutorial is idempotent.
_DEDUP_RATIO = 0.82


def _game_key(rom_or_game: str) -> str:
    """Normalize a ROM path / name to a stable per-game key."""
    base = os.path.basename(rom_or_game or "").lower()
    base = re.sub(r"\.(gba|gbc|gb|nes|sfc|smc|md|zip)$", "", base)
    return base.strip() or "unknown"


@dataclass
class Fact:
    """One declarative game fact. `game` keys retrieval; `topic` is a short label (from the
    tutorial heading, e.g. "Win/Lose Conditions"); `text` is the distilled one-liner. The
    rest is provenance/ranking so the learning loop can add, rank, and prune."""
    game: str
    topic: str
    text: str
    source: str = "learned"       # "learned" (from a tutorial) | "human" | "seed"
    confidence: float = 0.6       # learned facts start moderate; can be reinforced later
    uses: int = 0                 # times surfaced (retrieved), for future pruning
    created: float = 0.0          # unix ts


class KnowledgeStore:
    def __init__(self, path: Optional[os.PathLike | str] = None):
        env = os.environ.get("VGA_KNOWLEDGE_PATH")
        self.path = Path(path or env or DEFAULT_PATH)
        self._facts: list[Fact] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self._facts = [Fact(**f) for f in raw]
                return
            except (json.JSONDecodeError, TypeError, OSError):
                pass  # corrupt/incompatible -> start empty rather than crash a run
        self._facts = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(f) for f in self._facts], indent=2))

    def add(self, game: str, topic: str, text: str, source: str = "learned",
            confidence: float = 0.6) -> bool:
        """Add a distilled fact for `game`. Dedups by fuzzy text match within the game so a
        re-read tutorial does not pile up. Returns True if newly added. Persists on add.
        Callers should NOT pass distiller non-facts (e.g. 'NO ACTIONABLE FACT') - that filter
        lives in the tutorial-learning caller, not here, to keep this store dumb + reusable."""
        game = _game_key(game)
        topic = " ".join((topic or "").split())[:80]
        text = " ".join((text or "").split())
        if not text:
            return False
        for f in self._facts:
            if f.game == game and SequenceMatcher(None, f.text.lower(), text.lower()).ratio() >= _DEDUP_RATIO:
                return False  # already know this
        self._facts.append(Fact(game=game, topic=topic, text=text, source=source,
                                confidence=confidence, uses=0, created=time.time()))
        self.save()
        return True

    def retrieve(self, game: str, context: str = "", limit: int = 4) -> list[str]:
        """Return up to `limit` fact strings ("topic: text") for `game`. If `context` is given
        (e.g. current mode + on-screen text), rank by keyword overlap so only relevant facts
        surface; otherwise by (confidence, uses). A/B kill-switch: VGA_KNOWLEDGE_OFF=1 -> []."""
        if os.environ.get("VGA_KNOWLEDGE_OFF") in ("1", "true", "True"):
            return []
        game = _game_key(game)
        hits = [f for f in self._facts if f.game == game]
        if not hits:
            return []
        # Rank by keyword overlap against the free-text context (mode + on-screen text), the
        # shared semantic path also used by SkillStore. See vga/reason/retrieval.py for the
        # zero-overlap / token-budget behaviour (surface everything when the store is small).
        hits = rank_by_overlap(hits, context,
                               searchable=lambda f: f.topic + " " + f.text,
                               tiebreak=lambda f: (f.confidence, f.uses), limit=limit)
        for f in hits:
            f.uses += 1
        if hits:
            self.save()  # persist use counts
        return [f"{f.topic}: {f.text}" if f.topic else f.text for f in hits]

    def for_game(self, game: str) -> list[Fact]:
        g = _game_key(game)
        return [f for f in self._facts if f.game == g]

    def all(self) -> list[Fact]:
        return list(self._facts)

    def count(self) -> int:
        return len(self._facts)
