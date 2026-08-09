#!/usr/bin/env python3
"""Per-attempt breadcrumbs for the R4 travel acid test (Task 01, intent-persistence).

The furthest-rung number alone hides WHERE a travel episode died and WHETHER the
intent-persistence commit actually fired. This reads the RAM already logged every step in
steps.jsonl (no re-run, no GPU) and prints the fine-grained signal the day-2 lesson asked
for BEFORE trusting small A/B deltas: did the agent reach the visible Start menu (overlay
255) / Area List (204), did the caravan ever launch (clan_pos leaves 18=Cyril), did it
reach Giza (clan_pos==20) -- and did the commit mechanism engage and persist.

Overlay semantics (pixel-ground-truthed, sessions/2026-07-04-rescue-day2.md):
    255 = world-map Start menu (Party / Area List / System)
    204 = Area List submenu (Cyril / Sprohm / Giza Plains)
    0   = field / free-travel cursor on the map     19 = travel animation
clan_pos: 18=Cyril(start) 20=Giza. Leaving 18 == the caravan actually launched.

Verdict (matches docs/revitalization/01_intent_persistence.md):
    PASS    : clan_pos reached 20 (at_giza).
    PARTIAL : reached overlay 255 or 204 (a VISIBLE menu) but not Giza -- persistence worked,
              a downstream menu-nav/commit issue remains (hand to Task 02).
    FAIL    : never reached 255/204 -- the visible menu was never opened.

Usage:
    travel_breadcrumbs.py <path> [<path> ...]
where <path> is a bench out-root (containing ep_* dirs) or a single episode dir
(containing steps.jsonl). Point it at the treatment run and the baseline to compare.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

# Overlay milestones on the world-map travel path.
OVL_START_MENU = 255
OVL_AREA_LIST = 204
CLAN_START = 18   # Cyril, the pre-travel node
CLAN_GIZA = 20    # Giza Plains, the destination


def _episode_dirs(path: str) -> list[str]:
    """Resolve a path to the list of episode dirs it names: a single ep dir (has steps.jsonl),
    or a bench out-root (has ep_* subdirs), sorted."""
    if os.path.isfile(os.path.join(path, "steps.jsonl")):
        return [path]
    eps = sorted(glob.glob(os.path.join(path, "ep_*")))
    return [e for e in eps if os.path.isfile(os.path.join(e, "steps.jsonl"))]


def _analyze_episode(ep_dir: str) -> dict:
    overlays = Counter()
    clan_seen = set()
    reached_start_menu = reached_area_list = False
    launched = reached_giza = False
    furthest_rung = None
    commit_fires = 0            # steps where a NEW commit was locked
    commit_texts: list[str] = []
    max_active_run = 0          # longest stretch a commit stayed live (steps_left seen)
    commit_ever = False

    with open(os.path.join(ep_dir, "steps.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Use next_state (the state AFTER the action) so a milestone reached ON this step
            # is counted; fall back to state for older logs.
            st = rec.get("next_state") or rec.get("state") or {}
            ovl = st.get("mode_overlay")
            if ovl is not None:
                overlays[ovl] += 1
                if ovl == OVL_START_MENU:
                    reached_start_menu = True
                if ovl == OVL_AREA_LIST:
                    reached_area_list = True
            pos = st.get("clan_pos")
            if pos is not None:
                clan_seen.add(pos)
                if pos != CLAN_START:
                    launched = True
                if pos == CLAN_GIZA:
                    reached_giza = True
            # Furthest ladder rung reached this episode (checkpoints are booleans per step).
            cps = rec.get("checkpoints") or {}
            for name, ok in cps.items():
                if ok:
                    furthest_rung = name  # ladder is ordered; the last True seen is furthest
            if rec.get("commit_set"):
                commit_fires += 1
                commit_ever = True
                t = str(rec["commit_set"])
                if t not in commit_texts:
                    commit_texts.append(t)
            act = rec.get("commit_active")
            if act is not None:
                commit_ever = True
                # steps_left counts DOWN from N-1; the run length is (N - steps_left) but we
                # just track the max steps_left observed as a coarse "how long it held" proxy.
                max_active_run = max(max_active_run, int(act) + 1)

    if reached_giza:
        verdict = "PASS"
    elif reached_start_menu or reached_area_list:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "ep": os.path.basename(ep_dir),
        "verdict": verdict,
        "furthest_rung": furthest_rung,
        "reached_start_menu_255": reached_start_menu,
        "reached_area_list_204": reached_area_list,
        "launched_off_cyril": launched,
        "reached_giza": reached_giza,
        "clan_pos_seen": sorted(clan_seen),
        "overlays": dict(sorted(overlays.items(), key=lambda kv: -kv[1])),
        "commit_fired": commit_ever,
        "commit_lock_events": commit_fires,
        "commit_max_hold": max_active_run,
        "commit_texts": commit_texts,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for path in sys.argv[1:]:
        eps = _episode_dirs(path)
        if not eps:
            print(f"\n=== {path} ===\n  (no episode dirs with steps.jsonl found)")
            continue
        print(f"\n=== {path} ===")
        for ep_dir in eps:
            r = _analyze_episode(ep_dir)
            print(f"  {r['ep']}: {r['verdict']}  furthest={r['furthest_rung']}")
            print(f"     menu reached: start_menu(255)={r['reached_start_menu_255']}  "
                  f"area_list(204)={r['reached_area_list_204']}")
            print(f"     travel: launched_off_cyril={r['launched_off_cyril']}  "
                  f"reached_giza={r['reached_giza']}  clan_pos_seen={r['clan_pos_seen']}")
            print(f"     commit: fired={r['commit_fired']}  lock_events={r['commit_lock_events']}  "
                  f"max_hold={r['commit_max_hold']}")
            if r["commit_texts"]:
                for t in r["commit_texts"]:
                    print(f"        locked: {t[:100]}")
            print(f"     overlays: {r['overlays']}")


if __name__ == "__main__":
    main()
