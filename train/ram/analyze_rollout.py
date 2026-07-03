#!/usr/bin/env python3
"""Objective metrics for a vlm rollout (train/ram/rollout.py). Quantifies the behaviour we
are trying to fix: looping/mashing vs actually navigating and progressing.

    .venv/bin/python train/ram/analyze_rollout.py sessions/oracle-XXXX [more dirs...]

Metrics per run:
  unique_states  distinct screens visited (aHash clusters, Hamming>10). Higher = progress.
  progress_rate  fraction of steps that CHANGED the screen. Low = stuck mashing.
  wasted_A       fraction of A-presses that did NOT change the screen (the mash-in-a-menu bug).
  revisit_rate   fraction of steps whose screen re-appears elsewhere (loop density).
  moves          direction presses (cursor navigation) vs A presses.
  modes          the mode the model self-reported.
  cum_reward     oracle progress signal (new world-map nodes etc).
"""
from __future__ import annotations
import json, os, sys
from collections import Counter


def hamming(a, b):
    return bin(a ^ b).count("1")


def analyze(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "steps.jsonl"))]
    n = len(rows)
    hashes = [r.get("phash", 0) for r in rows]
    # greedy unique-state clustering
    reps = []
    for h in hashes:
        if not any(hamming(h, r) <= 10 for r in reps):
            reps.append(h)
    revisit = sum(1 for i, h in enumerate(hashes)
                  if any(hamming(h, o) <= 10 for j, o in enumerate(hashes) if j != i))
    btns = Counter(r["action"]["button"] for r in rows)
    a_steps = [r for r in rows if r["action"]["button"] == "A"]
    wasted_a = sum(1 for r in a_steps if not r.get("changed"))
    changed = sum(1 for r in rows if r.get("changed"))
    dirs = sum(btns.get(d, 0) for d in ("up", "down", "left", "right"))
    modes = Counter(r.get("mode", "") for r in rows)

    # Checkpoint ladder: the diagnostic metric. Rungs are ordered milestones; report the
    # FURTHEST rung the episode ever satisfied and the step it first fired. Order is taken
    # from the first step that carries a checkpoints dict (json preserves insertion order).
    order = []
    for r in rows:
        cps = r.get("checkpoints")
        if cps:
            order = list(cps.keys())
            break
    first_step = {}
    for r in rows:
        for name, ok in (r.get("checkpoints") or {}).items():
            if ok and name not in first_step:
                first_step[name] = r["step"]
    furthest_idx = -1
    for i, name in enumerate(order):
        if name in first_step:
            furthest_idx = i
    ladder = None
    if order:
        ladder = {
            "furthest": order[furthest_idx] if furthest_idx >= 0 else None,
            "furthest_idx": furthest_idx,
            "n_rungs": len(order),
            "first_step": {k: first_step[k] for k in order if k in first_step},
        }

    return {
        "run": os.path.basename(run_dir), "steps": n,
        "unique_states": len(reps),
        "progress_rate": round(changed / n, 3) if n else 0,
        "wasted_A": round(wasted_a / len(a_steps), 3) if a_steps else None,
        "revisit_rate": round(revisit / n, 3) if n else 0,
        "A": btns.get("A", 0), "dirs": dirs, "wait": btns.get("wait", 0),
        "modes": dict(modes),
        "cum_reward": rows[-1].get("cum_reward") if rows else None,
        "ladder": ladder,
    }


def main():
    for d in sys.argv[1:]:
        print(json.dumps(analyze(d)))


if __name__ == "__main__":
    main()
