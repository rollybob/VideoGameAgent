import os
import json
import tempfile

from vga.reason import knowledge


def _serialize(store):
    """Helper to compare stores independent of ordering."""
    return sorted(
        [(f.game, f.topic, f.text) for f in store.all()],
        key=lambda x: (x[0], x[1], x[2]),
    )


# Ensure a clean environment for the test run.
for _var in ("VGA_KNOWLEDGE_PATH", "VGA_KNOWLEDGE_OFF"):
    os.environ.pop(_var, None)

with tempfile.TemporaryDirectory() as td:
    # Point the KnowledgeStore to a temporary file via env var.
    store_path = os.path.join(td, "knowledge.json")
    os.environ["VGA_KNOWLEDGE_PATH"] = store_path

    # ----------------------------------------------------------------------
    # 1. Fresh store starts empty.
    ks = knowledge.KnowledgeStore()
    assert ks.count() == 0
    assert ks.all() == []

    # ----------------------------------------------------------------------
    # 2. Adding a normal fact works and is persisted.
    added = ks.add("mygame.gba", "Win Condition ", " Defeat the boss ")
    assert added is True
    assert ks.count() == 1

    # Game key normalization (strip path, extension, lower‑case)
    facts_my = ks.for_game("MYGAME")
    assert len(facts_my) == 1
    f0 = facts_my[0]
    assert f0.game == "mygame"
    assert f0.topic == "Win Condition"          # whitespace collapsed, trimmed
    assert f0.text == "Defeat the boss"

    # Retrieval returns formatted string and increments uses.
    ret = ks.retrieve("mygame")
    assert isinstance(ret, list) and len(ret) == 1
    assert ret[0] == "Win Condition: Defeat the boss"
    assert ks.for_game("mygame")[0].uses == 1

    # ----------------------------------------------------------------------
    # 3. Deduplication (case‑insensitive fuzzy match).
    added_dup = ks.add("mygame.gba", "Another Topic", "defeat THE BOSS")
    assert added_dup is False
    assert ks.count() == 1

    # ----------------------------------------------------------------------
    # 4. Same text for a different game is allowed.
    added_other = ks.add("othergame.nes", "Topic", "Defeat the boss")
    assert added_other is True
    assert ks.count() == 2
    other_facts = ks.for_game("othergame")
    assert len(other_facts) == 1
    assert other_facts[0].game == "othergame"

    # ----------------------------------------------------------------------
    # 5. Empty or whitespace‑only text is rejected.
    assert ks.add("mygame.gba", "Empty", "") is False
    assert ks.add("mygame.gba", "Spaces", "   ") is False
    assert ks.count() == 2

    # ----------------------------------------------------------------------
    # 6. Long topic strings are truncated to 80 characters after whitespace collapse.
    long_topic = "T" * 100
    ks.add("mygame.gba", long_topic, "some fact")
    trunc_fact = ks.for_game("mygame")[-1]
    assert len(trunc_fact.topic) == 80
    assert trunc_fact.topic == ("T" * 80)

    # ----------------------------------------------------------------------
    # 7. Game key for empty/unknown input becomes 'unknown'.
    added_unknown = ks.add("", "U", "some unknown fact")
    assert added_unknown is True
    unk_facts = ks.for_game("unknown")
    assert len(unk_facts) == 1
    assert unk_facts[0].game == "unknown"

    # ----------------------------------------------------------------------
    # 8. Retrieval ranking with context and limit.
    ks.add("mygame.gba", "SpecialTopic", "unique phrase xyz")
    # Now mygame has three facts (win, long topic, special).
    ranked = ks.retrieve("mygame", context="xyz", limit=2)
    # At least the special fact should appear first; other results may be filtered out.
    assert any(r.startswith("SpecialTopic:") for r in ranked), "SpecialTopic missing"
    assert 1 <= len(ranked) <= 2, f"Unexpected number of results: {len(ranked)}"

    # ----------------------------------------------------------------------
    # 9. Kill‑switch disables retrieval entirely.
    os.environ["VGA_KNOWLEDGE_OFF"] = "1"
    off_ret = ks.retrieve("mygame")
    assert off_ret == []
    del os.environ["VGA_KNOWLEDGE_OFF"]

    # ----------------------------------------------------------------------
    # 10. Persistence: a new store instance loads the same data.
    ks2 = knowledge.KnowledgeStore()
    assert ks2.count() == ks.count()
    assert _serialize(ks2) == _serialize(ks)

print("OK")
