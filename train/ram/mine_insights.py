#!/usr/bin/env python3
"""Experiential insight miner (ExpeL-style) -- the LEARN half of the self-learning loop.

Rationale (rescue day3, 2026-07-04): the agent's KnowledgeStore was never populated because
its only writer (the tutorial reader) never fired on the tutorial-sparse FFTA test path. But
the agent has run 170+ episodes; the RAM oracle labels which REACHED a rung and which did not.
ExpeL (Zhao et al. 2024) shows an agent can improve WITHOUT gradient updates by extracting
declarative INSIGHTS from the contrast between its own successful and failed trajectories and
retrieving them at inference. This script produces that contrast and, with --extract-local,
lets the LOCAL VLM (the same model that plays) do the extraction, which get written to a
KnowledgeStore the agent reads via `bench_ladder.py --knowledge`.

INTEGRITY GUARD: we summarize ONLY objective signals -- the buttons pressed, what the agent
PERCEIVED (the logged `highlighted` menu-option text), and the RAM-verified checkpoints that
fired. We deliberately DROP the goal / reason / subgoal free-text, because on scaffolded runs
those echo a hand-written walkthrough; mining them would launder the walkthrough back in as a
"learned" fact. Contrasting objective what-worked vs what-failed is a genuine inference.

Task 03 (2026-07-05) added two things and kept the guard intact:
  * --extract-local : call the LOCAL VLM's /complete endpoint to extract facts (self-reflection,
    ExpeL's own-model setting), replacing the external Claude teacher. Same fixed prompt.
  * --auto          : auto-select success/fail trajectory pairs per RAM-ladder sub-task from the
    logged episode pool (grouped by furthest rung), so the loop mines across the whole ladder,
    not just the travel contrast, without hand-picking (which would smell of a walkthrough).

Usage:
  # A) local self-extraction on ONE contrast, write straight to the store:
  mine_insights.py --extract-local --write \
      --success sessions/bench-r4-staged-0704/ep_00 --fail sessions/bench-travel-task-0704/ep_00 \
      --topic-hint "traveling on the world map to the mission location" \
      --game "Final Fantasy Tactics Advance (E)(Surplus).gba" \
      --knowledge train/ram/knowledge_ffta_local.json

  # B) auto-discover pairs across the ladder and mine each with the local VLM:
  mine_insights.py --extract-local --auto --write \
      --game "Final Fantasy Tactics Advance (E)(Surplus).gba" \
      --knowledge train/ram/knowledge_ffta_local.json

  # C) legacy: emit the extraction prompt for an external teacher, then write its facts:
  mine_insights.py --success <dir> --fail <dir> --out-prompt /tmp/mine_prompt.txt
  mine_insights.py --write-facts facts.json --game <rom> --knowledge <store.json>

  # D) just show which pairs --auto would pick (no mining):
  mine_insights.py --auto --list-pairs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import urllib.request

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# The Herb Picking checkpoint ladder, in order (train/ram/oracle.py Ffta.checkpoints). The
# per-step `checkpoints` dict in steps.jsonl is booleans over exactly these keys; the furthest
# TRUE index is the episode's furthest rung. A short human label per rung-TRANSITION seeds the
# extraction prompt's {topic} (objective, not a walkthrough: it names the sub-task, no menu path).
LADDER = ["pub_open", "mission_list", "mission_accepted",
          "worldmap_regained", "at_giza", "battle_entered"]
RUNG_TOPIC = {
    1: "opening the Missions list at the pub",
    2: "accepting a mission at the pub",
    3: "leaving the pub back onto the world map",
    4: "traveling on the world map to the mission location",
    5: "entering the mission's battle",
}


def _load_steps(traj_dir: str) -> list[dict]:
    p = os.path.join(traj_dir, "steps.jsonl")
    with open(p) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _rung_idx(cps: dict) -> int:
    """Furthest LADDER index that is True in one checkpoint dict; -1 if none/not-herb-ladder."""
    best = -1
    for i, k in enumerate(LADDER):
        if cps.get(k):
            best = i
    return best


def _is_herb_ladder(steps: list[dict]) -> bool:
    cps0 = (steps[0].get("checkpoints") or {}) if steps else {}
    return set(LADDER).issubset(cps0.keys())


def _objective_trace(traj_dir: str) -> str:
    """Compact, goal-free trace: step | button xN | saw:<highlighted> | +<newly-fired rung>.
    Only objective signals: action, PERCEIVED highlighted option, RAM checkpoints."""
    rows = _load_steps(traj_dir)
    lines = []
    seen: set[str] = set()
    for r in rows:
        act = r.get("action") or {}
        btn = act.get("button", "?")
        rep = act.get("repeats", 1)
        hl = (r.get("highlighted") or "").strip()
        cps = r.get("checkpoints") or {}
        fired = [k for k, v in cps.items() if v and k not in seen]
        for k in fired:
            seen.add(k)
        seg = f"  t{r.get('step'):>3} {btn}x{rep}"
        if hl:
            seg += f"  saw_menu_option='{hl}'"
        if fired:
            seg += f"  >>RAM_REACHED: {', '.join(fired)}"
        lines.append(seg)
    furthest = list(seen)[-1] if seen else "none"
    return f"(furthest RAM checkpoint reached: {furthest})\n" + "\n".join(lines)


EXTRACTION_INSTRUCTIONS = """\
You are a reflection teacher for a game-playing agent that sees ONLY the current screen and
presses one button per step. Below are two OBJECTIVE traces of the agent attempting the SAME
sub-task ({topic}) from the same starting screen: one SUCCEEDED (the RAM oracle confirms it
reached the goal) and one FAILED. Each line shows the button pressed, the menu option the agent
perceived on screen (saw_menu_option), and any RAM checkpoint that fired. No goal text or the
agent's own reasoning is shown -- only what it DID, SAW, and what RESULTED.

Compare the SUCCESS and FAILURE traces. Infer what the successful attempt did that the failed
one did not, at the level of GAME MECHANICS. Then output up to 3 SHORT, GENERAL, TRANSFERABLE
declarative facts an agent could use next time -- facts about how this game works, NOT a
step-by-step script for this one screen. Each fact must be true given the traces and useful to
an agent that only sees the current frame. Prefer facts that name the on-screen menus/options
the successful run used. Avoid coordinates, step numbers, and anything specific to one attempt.

Output ONLY a JSON array, each item {{"topic": "<3-6 word label>", "text": "<one sentence fact>"}}.

=== SUCCESS TRACE ===
{success}

=== FAILURE TRACE ===
{failure}
"""


def _build_prompt(success_dir: str, fail_dir: str, topic_hint: str) -> str:
    return EXTRACTION_INSTRUCTIONS.format(
        topic=topic_hint,
        success=_objective_trace(success_dir),
        failure=_objective_trace(fail_dir))


# ------------------------------- local-VLM extraction ------------------------------------

def _post_complete(url: str, prompt: str, max_tok: int = 320, timeout: int = 180) -> str:
    """POST the extraction prompt to the local VLM's text-only /complete route, return the text.
    Greedy on the server (reproducible). No image, no forced schema (the 8B has failed forced
    JSON before); we parse the free text defensively below."""
    body = {"prompt": prompt, "max_new_tokens": max_tok,
            "system": "You are a careful game-analysis assistant. Compare the two traces and "
                      "output only the requested JSON array of facts, nothing else."}
    req = urllib.request.Request(url.rstrip("/") + "/complete",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["text"].strip()


def _parse_facts(text: str) -> list[dict]:
    """Defensively pull [{topic,text}, ...] out of the model's reply. The 8B may fence it in
    ```json ... ```, prepend prose, or emit loose objects; salvage progressively rather than
    trust one json.loads. Returns [] if nothing parseable (an honest empty result, not a crash)."""
    if not text:
        return []
    # 1) strip a ```json ... ``` fence if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    body = fence.group(1) if fence else text
    # 2) try the first [...] array.
    arr = re.search(r"\[.*\]", body, re.DOTALL)
    for candidate in ([arr.group(0)] if arr else []):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return [_norm_fact(x) for x in obj if _norm_fact(x)]
        except json.JSONDecodeError:
            pass
    # 3) fall back to scraping individual {...} objects with topic+text.
    out = []
    for m in re.finditer(r"\{[^{}]*\}", body, re.DOTALL):
        try:
            x = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        f = _norm_fact(x)
        if f:
            out.append(f)
    return out


def _norm_fact(x) -> dict | None:
    if not isinstance(x, dict):
        return None
    topic = str(x.get("topic", "")).strip()[:60]
    txt = str(x.get("text", "")).strip()[:300]
    if not txt:
        return None
    return {"topic": topic or "learned mechanic", "text": txt}


def _extract_local(url: str, success_dir: str, fail_dir: str, topic_hint: str,
                   verbose: bool = True) -> list[dict]:
    prompt = _build_prompt(success_dir, fail_dir, topic_hint)
    raw = _post_complete(url, prompt)
    facts = _parse_facts(raw)
    if verbose:
        print(f"[mine] local-VLM extraction for '{topic_hint}':", flush=True)
        print(f"       success={success_dir}  fail={fail_dir}")
        print(f"       raw ({len(raw)} chars): {raw[:400].replace(chr(10), ' ')}")
        print(f"       parsed {len(facts)} fact(s):")
        for f in facts:
            print(f"         - {f['topic']}: {f['text']}")
    return facts


# ------------------------------- auto pair selection -------------------------------------

def _survey_pool(roots: list[str]) -> list[dict]:
    """Scan roots for Herb-ladder episodes, return {dir,start,far,scene,n} per episode.
    `scene` is the step-0 RAM scene (6=town/pub, 7=world map, 14=battle) -- the real
    "same starting screen" key (the pub_open rung is definitionally always-true, so the
    start RUNG alone can't tell the pub apart from a new-game intro)."""
    recs = []
    seen_dirs = set()
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "**", "steps.jsonl"), recursive=True)):
            d = os.path.dirname(p)
            if d in seen_dirs:
                continue
            seen_dirs.add(d)
            try:
                st = _load_steps(d)
            except (OSError, json.JSONDecodeError):
                continue
            if not st or not _is_herb_ladder(st):
                continue
            start = _rung_idx(st[0].get("checkpoints") or {})
            far = max((_rung_idx(r.get("checkpoints") or {}) for r in st), default=-1)
            scene = (st[0].get("state") or {}).get("scene")
            recs.append({"dir": d, "start": start, "far": far, "scene": scene, "n": len(st)})
    return recs


def auto_pairs(recs: list[dict]) -> list[dict]:
    """For each rung-TRANSITION T (into LADDER[T]), pick the cleanest success+fail contrast:
      success: an episode that STARTED below T (start < T) and reached >= T -- i.e. it actually
               CROSSED the T boundary within the trace (so the trace demonstrates the mechanic).
      fail:    an episode that started below T and got stuck AT the doorstep (far == T-1).
    Both must share the same starting SCENE (6=pub, 7=world map, ...) so the two traces really do
    begin on comparable screens, as the extraction prompt claims. Tie-break success by furthest-
    then-fewest-steps (a crisp win), fail by most-steps (it genuinely tried and still failed).
    Returns [{target, topic, success, fail, scene}] for transitions that have BOTH sides. Purely
    RAM-label-driven -- no hand-picking, so no walkthrough leakage.

    Requiring start<T is the fix for the subtle trap that an episode which BEGINS past a rung
    (e.g. a post-accept world-map run vs the 'accept the mission' rung) trivially "reached" it
    without ever demonstrating the crossing -- mining that pair would teach nothing or mislead."""
    pairs = []
    for T in range(1, len(LADDER)):
        succ = [r for r in recs if r["start"] < T <= r["far"]]
        fail = [r for r in recs if r["start"] < T and r["far"] == T - 1]
        if not succ or not fail:
            continue
        # Require a shared starting scene; pick the scene with both sides best represented.
        scenes = {r["scene"] for r in succ} & {r["scene"] for r in fail}
        if not scenes:
            continue
        scene = sorted(scenes, key=lambda sc: -min(
            sum(r["scene"] == sc for r in succ), sum(r["scene"] == sc for r in fail)))[0]
        s = sorted([r for r in succ if r["scene"] == scene], key=lambda r: (-r["far"], r["n"]))[0]
        f = sorted([r for r in fail if r["scene"] == scene], key=lambda r: (-r["n"],))[0]
        pairs.append({"target": T, "topic": RUNG_TOPIC.get(T, LADDER[T]),
                      "success": s["dir"], "fail": f["dir"], "scene": scene})
    return pairs


# ------------------------------------- store write ---------------------------------------

def _write_facts(facts: list[dict], game: str, knowledge: str, source: str) -> None:
    from vga.reason.knowledge import KnowledgeStore
    ks = KnowledgeStore(path=knowledge or None)
    added = 0
    for f in facts:
        if ks.add(game, f.get("topic", ""), f.get("text", ""), source=source, confidence=0.7):
            added += 1
    print(f"[mine] wrote {added}/{len(facts)} new facts -> {ks.path} (total {ks.count()})")
    for fa in ks.for_game(game):
        print(f"   - {fa.topic}: {fa.text}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--success", help="trajectory dir that reached the goal")
    ap.add_argument("--fail", help="trajectory dir that did not")
    ap.add_argument("--topic-hint", default="a sub-task")
    ap.add_argument("--out-prompt", default="", help="write the extraction prompt here (teacher mode)")
    ap.add_argument("--write-facts", default="", help="a JSON array of {topic,text} to store")
    ap.add_argument("--extract-local", action="store_true",
                    help="extract facts with the LOCAL VLM (/complete) instead of a teacher")
    ap.add_argument("--auto", action="store_true",
                    help="auto-select success/fail pairs per ladder sub-task from --pool-root")
    ap.add_argument("--list-pairs", action="store_true", help="with --auto: just print the pairs")
    ap.add_argument("--pool-root", action="append", default=[],
                    help="root(s) to scan for episodes (default sessions/); repeatable")
    ap.add_argument("--url", default="http://127.0.0.1:8077", help="local VLM server")
    ap.add_argument("--game", default="", help="ROM name/path (keys the store)")
    ap.add_argument("--knowledge", default="", help="KnowledgeStore json path to write into")
    ap.add_argument("--write", action="store_true", help="write extracted facts into the store")
    ap.add_argument("--source", default="experience-local", help="provenance tag for stored facts")
    args = ap.parse_args()

    # Legacy path: write a pre-made facts JSON (from an external teacher) into the store.
    if args.write_facts:
        facts = json.loads(open(args.write_facts).read()) if os.path.exists(args.write_facts) \
            else json.loads(args.write_facts)
        _write_facts(facts, args.game, args.knowledge, args.source)
        return 0

    # Auto pair discovery (optionally mine each with the local VLM).
    if args.auto:
        roots = args.pool_root or [os.path.join(_REPO, "sessions")]
        recs = _survey_pool(roots)
        pairs = auto_pairs(recs)
        print(f"[mine] surveyed {len(recs)} herb-ladder episodes across {len(roots)} root(s); "
              f"found {len(pairs)} minable sub-task contrast(s):", flush=True)
        for p in pairs:
            print(f"   T{p['target']} {p['topic']}  (start scene={p['scene']})\n"
                  f"      success={p['success']}\n      fail   ={p['fail']}")
        if args.list_pairs:
            return 0
        if not args.extract_local:
            ap.error("--auto without --list-pairs needs --extract-local to mine the pairs")
        all_facts = []
        for p in pairs:
            facts = _extract_local(args.url, p["success"], p["fail"], p["topic"])
            all_facts.extend(facts)
        if args.write and all_facts:
            _write_facts(all_facts, args.game, args.knowledge, args.source)
        elif not args.write:
            print("\n[mine] (dry run; pass --write to store)  all facts:")
            print(json.dumps(all_facts, indent=2))
        return 0

    # Single explicit contrast.
    if not (args.success and args.fail):
        ap.error("need --success and --fail (or --write-facts, or --auto)")

    if args.extract_local:
        facts = _extract_local(args.url, args.success, args.fail, args.topic_hint)
        if args.write and facts:
            _write_facts(facts, args.game, args.knowledge, args.source)
        elif not args.write:
            print("\n[mine] (dry run; pass --write to store)  facts:")
            print(json.dumps(facts, indent=2))
        return 0

    # Teacher mode: emit the extraction prompt for an external reflection teacher.
    prompt = _build_prompt(args.success, args.fail, args.topic_hint)
    if args.out_prompt:
        with open(args.out_prompt, "w") as f:
            f.write(prompt)
        print(f"[mine] extraction prompt -> {args.out_prompt} ({len(prompt)} chars)")
    else:
        print(prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
