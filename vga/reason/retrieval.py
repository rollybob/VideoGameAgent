"""
Shared semantic-retrieval primitive for the reason-layer memory stores.

Both persistent stores - KnowledgeStore (declarative game facts) and SkillStore
(procedural UI rules) - want the SAME retrieval shape: given a free-text `context`
describing the current scene (mode word + on-screen text + any scene descriptor),
surface the few stored items whose text overlaps that context, ranked, within a
token budget. KnowledgeStore has always done this right; SkillStore used to do a
hard exact-match on a fixed game-specific MODE whitelist, which CLIFFED to [] the
moment the mode was wrong / "unknown" / a vocab slip ("overworld" vs "field").

Factoring the ranking here (Task 02, 2026-07-05) unifies the two onto one path so a
future change to how we rank cannot silently drift them apart, and kills the
game-specific taxonomy from the skill hot path. The query is free text produced by
the general VLM; nothing here assumes a genre.

Design:
- Tokenize to lowercased alnum words, drop a tiny stop-list and 1-2 char noise.
- Rank by token-overlap count first, then a caller-supplied tiebreak (confidence,
  uses), all descending.
- Only DROP zero-overlap items under real token-budget pressure (more candidates
  than `limit`). With a small store (<= limit) there is no pressure, so we surface
  everything ranked - otherwise a just-learned item whose wording happens not to
  share a token with the current mode string is silently dropped exactly when it is
  needed. (This was a live break for knowledge on 2026-07-04; keep the behaviour.)
"""

from __future__ import annotations

import re
from typing import Callable

# Words too common to help keyword ranking (kept tiny + game-agnostic). Shared by both
# stores so their notion of "signal" cannot drift.
STOP = frozenset("a an the to of in on is are be it its you your with and or for as at this "
                 "that these those can will may your than then so but if when your not no".split())


def tokens(text: str) -> set:
    """Lowercased alnum content words (drop stop-words and 1-2 char noise)."""
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in STOP and len(w) > 2}


def rank_by_overlap(items: list, context: str, searchable: Callable[[object], str],
                    tiebreak: Callable[[object], tuple], limit: int) -> list:
    """Return up to `limit` items ranked by token overlap with `context`.

    `searchable(item)` -> the item's text to match against; `tiebreak(item)` -> a
    tuple ranked (desc) after overlap (e.g. (confidence, uses)). When `context` has
    no content tokens, fall back to `tiebreak` alone (caller decides whether an empty
    context should even reach here - SkillStore stays quiet, KnowledgeStore surfaces
    its top facts). Mirrors KnowledgeStore's historical logic exactly.
    """
    ctx = tokens(context)
    if ctx:
        ranked = sorted(items,
                        key=lambda it: (len(tokens(searchable(it)) & ctx),) + tuple(tiebreak(it)),
                        reverse=True)
        # Drop zero-overlap items ONLY under token-budget pressure (see module docstring).
        if len(ranked) > limit:
            ranked = [it for it in ranked if tokens(searchable(it)) & ctx]
    else:
        ranked = sorted(items, key=lambda it: tuple(tiebreak(it)), reverse=True)
    return ranked[:limit]
