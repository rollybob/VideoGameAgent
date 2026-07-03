#!/usr/bin/env python3
"""Baseline the current pixels-only agent against the checkpoint ladder.

Runs M INDEPENDENT episodes, each from a fixed savestate (default ffta_pub.state), with a
fresh emulator + fresh VlmPlugin per episode, and reports the FURTHEST-checkpoint
distribution across episodes. This is the honest number every later change (tutorial
reading, memory, an action-head) gets measured against - the eval-first discipline from
docs/CHECKPOINT_LADDER_PLAN.md.

Needs the VLM server up (serve/run_server.sh) for --policy vlm. GPU -> run under thor-job.

    .venv/bin/python train/ram/bench_ladder.py --episodes 10 --steps 40 \
        --goal-file train/ram/goal_herb.txt --out-root sessions/bench-ladder-<ts>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rollout import rollout           # noqa: E402
from analyze_rollout import analyze   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba")
    ap.add_argument("--state", default="train/ram/ffta_pub.state")
    ap.add_argument("--policy", choices=["random", "vlm"], default="vlm")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--goal", default="")
    ap.add_argument("--goal-file", default="")
    ap.add_argument("--url", default="http://127.0.0.1:8077")
    ap.add_argument("--out-root", default="")
    args = ap.parse_args()

    goal = args.goal
    if args.goal_file:
        with open(args.goal_file) as f:
            goal = f.read().strip()

    rom = args.rom if os.path.isabs(args.rom) else os.path.join(_REPO, args.rom)
    state = args.state if os.path.isabs(args.state) else os.path.join(_REPO, args.state)
    out_root = args.out_root or os.path.join(_REPO, "sessions", f"bench-ladder-{int(time.time())}")
    os.makedirs(out_root, exist_ok=True)

    print(f"[bench] {args.episodes} episodes x {args.steps} steps  policy={args.policy}  "
          f"goal={goal!r}\n[bench] -> {out_root}", flush=True)

    furthest = Counter()
    accepted = 0
    per_ep = []
    for ep in range(args.episodes):
        ep_dir = os.path.join(out_root, f"ep_{ep:02d}")
        rollout(rom, args.steps, args.policy, ep_dir, goal=goal, url=args.url,
                load_state=state)
        a = analyze(ep_dir)
        ladder = a.get("ladder") or {}
        rung = ladder.get("furthest")
        furthest[rung] += 1
        got_accept = "mission_accepted" in (ladder.get("first_step") or {})
        accepted += 1 if got_accept else 0
        per_ep.append({"ep": ep, "furthest": rung, "furthest_idx": ladder.get("furthest_idx"),
                       "first_step": ladder.get("first_step"),
                       "unique_states": a.get("unique_states"), "A": a.get("A"),
                       "dirs": a.get("dirs")})
        print(f"[bench] ep {ep+1}/{args.episodes}: furthest={rung} "
              f"first_step={ladder.get('first_step')}", flush=True)

    summary = {
        "episodes": args.episodes, "steps": args.steps, "policy": args.policy, "goal": goal,
        "furthest_distribution": dict(furthest),
        "accept_rate": round(accepted / args.episodes, 3) if args.episodes else 0.0,
        "per_episode": per_ep,
    }
    with open(os.path.join(out_root, "bench_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[bench] DONE")
    print("[bench] furthest-rung distribution:", dict(furthest))
    print(f"[bench] mission-accept rate: {summary['accept_rate']}")
    print(f"[bench] summary -> {os.path.join(out_root, 'bench_summary.json')}", flush=True)


if __name__ == "__main__":
    main()
