"""
Task 03B GATE 1 -- offline, deterministic, CPU-only retrieval-routing check (run BEFORE any GPU).

The Task-03 combined-store travel regression was proven to be a RETRIEVAL-CONTEXT bug, not a
self-extraction flaw: on a world-map walk frame the live retrieval context is effectively just
"overworld" (mode) + empty dialog OCR, which overlaps NO stored fact. With a store grown past the
retrieve limit (combined = 6 facts > limit 3), retrieval.py drops all zero-overlap facts under
budget pressure -> retrieve() returns [] -> the Task-01 commit (which locks knowledge[0]) never
fires. The travel-only store (3 facts <= limit) has no budget pressure so the fact survives.

Task 03B folds a short OBJECTIVE VLM scene descriptor into the retrieval context. This script
checks -- with NO GPU -- that the MECHANISM routes correctly given a plausible objective descriptor:

  PRE-FIX  (reproduce the bug): combined-store retrieve("overworld" + dialog) == []   (starved)
  POST-FIX (the descriptor):    combined-store retrieve("overworld" + <map scene> + dialog)[0]
                                == the "Navigate via Area List" fact                  (routed)
           and a <pub scene> descriptor -> the pub "Accept Mission" fact at [0].

HONESTY DISCIPLINE: the descriptors below are OBJECTIVE perception phrases (what the VLM would
literally SEE), and deliberately AVOID echoing a fact's distinctive phrase ("Area List") in the
strict map case -- so a PASS means generic map words route the fact, not a trivial phrase match.
Gate 2 (GPU) is the real test where the actual VLM emits the descriptor; this only gates on the
retrieval primitive so we do not spend GPU on a refuted premise.

Run (CPU, seconds):  PYTHONPATH=. .venv/bin/python train/ram/eval_retrieval_context.py
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from vga.reason.knowledge import KnowledgeStore
from vga.reason.skills import SkillStore

REPO = Path(__file__).resolve().parents[2]
GAME = "Final Fantasy Tactics Advance (E)(Surplus).gba"  # normalized internally by _game_key
LIMIT = 3  # matches plugin.py:decide() knowledge retrieve limit

COMBINED = REPO / "train/ram/knowledge_ffta_local_multi.json"  # 6 facts (travel + pub) -- the bug
TRAVEL   = REPO / "train/ram/knowledge_ffta_local.json"        # 3 facts (travel) -- the ceiling
SKILLS   = REPO / "vga/reason/skills.json"                     # seed procedural store

# A realistic bottom-strip OCR for the FFTA world map (per 03B doc): shares no token with any fact.
WM_DIALOG = "CLAN FUNDS Kingmoon"

# The live retrieval context is f"{mode} {scene} {dialog}". mode on a walk frame ~ "overworld".
MODE_WM = "overworld"
MODE_PUB = "menu"

# Objective world-map scene descriptors, hardest first. STRICT ones name only generic map/terrain
# entities (NO "area"/"list"), so a PASS proves routing on generic perception, not a phrase echo.
WM_SCENES = {
    "strict_terrain":  "world map with regions, roads, and the party unit standing on the terrain",
    "strict_caravan":  "overhead map view, several region areas connected by paths, a caravan marker",
    "with_locations":  "world map, region and location names, party marker, giza plains area shown",
}
# Objective pub / tavern scene descriptors (the off-task sub-task whose facts must NOT hijack the
# map frames, and which SHOULD surface on a pub frame). Uses singular "mission" (what the VLM emits).
PUB_SCENES = {
    "pub_missionlist": "pub tavern room with a mission list menu, options to accept a quest, a bartender",
    "pub_counter":     "inside a tavern, a window listing available mission entries to select and accept",
}

AREA_LIST_TOPIC = "Navigate via Area List"
PUB_ACCEPT_TOPIC = "Accept Mission from Pub"


def _scratch(src: Path) -> str:
    """Copy a store json to a temp file so retrieve()'s use-count save() does not mutate the real
    store during this read-only gate. Returns the temp path."""
    tmp = tempfile.NamedTemporaryFile(prefix="gate1_", suffix=".json", delete=False)
    tmp.close()
    shutil.copy(src, tmp.name)
    return tmp.name


def _k_retrieve(src: Path, ctx: str) -> list[str]:
    return KnowledgeStore(path=_scratch(src)).retrieve(GAME, context=ctx, limit=LIMIT)


def _s_retrieve(ctx: str) -> list[str]:
    return SkillStore(path=_scratch(SKILLS)).retrieve(context=ctx, limit=4)


def _top_topic(hits: list[str]) -> str:
    """The first hit's topic (text before ':') for a compact readout."""
    return hits[0].split(":", 1)[0] if hits else "(none)"


def main() -> None:
    fails = []
    print("=" * 78)
    print("Task 03B Gate 1 -- offline retrieval-routing check (CPU, deterministic)")
    print("=" * 78)

    # --- PRE-FIX: reproduce the bug. bare mode context on the COMBINED store -> starved [] ---
    bare = f"{MODE_WM} {WM_DIALOG}"
    combined_bare = _k_retrieve(COMBINED, bare)
    travel_bare = _k_retrieve(TRAVEL, bare)
    print(f"\n[PRE-FIX / the bug] bare ctx = {bare!r}")
    print(f"  combined store (6>limit): {[_top_topic(combined_bare)] if combined_bare else '[] (STARVED)'}")
    print(f"  travel-only   (3<=limit): [0]={_top_topic(travel_bare)!r}  (no budget pressure)")
    if combined_bare:
        fails.append("PRE-FIX combined bare ctx did NOT return [] -- the reproduced bug is missing")
    if _top_topic(travel_bare) != AREA_LIST_TOPIC:
        fails.append(f"PRE-FIX travel-only [0] != {AREA_LIST_TOPIC!r} (got {_top_topic(travel_bare)!r})")

    # --- POST-FIX: the descriptor routes the Area List fact to [0] on the COMBINED store ---
    print(f"\n[POST-FIX / the descriptor] map scene descriptors -> combined store (6>limit):")
    for name, scene in WM_SCENES.items():
        ctx = f"{MODE_WM} {scene} {WM_DIALOG}"
        hits = _k_retrieve(COMBINED, ctx)
        top = _top_topic(hits)
        ok = bool(hits) and top == AREA_LIST_TOPIC
        print(f"  [{ 'PASS' if ok else 'FAIL' }] {name:16s} -> [0]={top!r}  (n={len(hits)})")
        if not ok:
            fails.append(f"map scene {name!r}: combined [0]={top!r} (want {AREA_LIST_TOPIC!r}, non-empty)")

    # --- GUARD: a pub descriptor must surface the PUB fact at [0], not a travel fact (routing) ---
    print(f"\n[POST-FIX / routing guard] pub scene descriptors -> combined store:")
    for name, scene in PUB_SCENES.items():
        ctx = f"{MODE_PUB} {scene}"
        hits = _k_retrieve(COMBINED, ctx)
        top = _top_topic(hits)
        ok = bool(hits) and top == PUB_ACCEPT_TOPIC
        print(f"  [{ 'PASS' if ok else 'FAIL' }] {name:16s} -> [0]={top!r}  (n={len(hits)})")
        if not ok:
            fails.append(f"pub scene {name!r}: combined [0]={top!r} (want {PUB_ACCEPT_TOPIC!r}, non-empty)")

    # --- Bonus: the same descriptor should also enrich SkillStore routing (both stores share it) ---
    print(f"\n[SkillStore, same descriptor] (advisory -- confirms both stores route on the scene):")
    for name, scene in list(WM_SCENES.items())[:1] + list(PUB_SCENES.items())[:1]:
        ctx = f"{MODE_WM if 'strict' in name or 'with' in name else MODE_PUB} {scene}"
        hits = _s_retrieve(ctx)
        print(f"  {name:16s} -> {len(hits)} skill(s): {hits[0][:60]+'...' if hits else '(none)'}")

    print("\n" + "=" * 78)
    if fails:
        print(f"GATE 1: FAIL ({len(fails)} check(s)) -- premise weakened; do NOT spend GPU:")
        for f in fails:
            print(f"  - {f}")
        raise SystemExit(1)
    print("GATE 1: PASS -- the objective scene descriptor routes the Area List fact to knowledge[0]")
    print("  on the grown combined store (bug flipped), and pub descriptors route the pub fact.")
    print("  Cleared to proceed to Gate 2 (GPU travel acid test).")


if __name__ == "__main__":
    main()
