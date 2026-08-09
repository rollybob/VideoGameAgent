#!/usr/bin/env python3
"""Drive the LIVE mGBA window (real X display) with the full VGA learning stack.

Unlike run_vga.py this wires the pieces the FFTA agent needs: LocalVlmReasoner + a
pre-populated KnowledgeStore (facts mined/learned from a prior run) + optional live
tutorial-learning + goal, so we can watch on a real display whether the agent PULLS the
knowledge it learned (retrieval is logged per step) and how far it gets.

Input/capture go through vga.core.emulator.GbaEmulator (mss capture + xdotool keydown/keyup
addressed to mGBA's child window -- focus-independent, safe on :1). Requires DISPLAY +
XAUTHORITY pointing at the desktop where mGBA is running, and the VLM server up.

    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
      .venv/bin/python train/ram/live_drive.py --steps 250 \
      --goal-file train/ram/goal_freeplay.txt --knowledge train/ram/knowledge_ffta_fresh.json \
      --game "Final Fantasy Tactics Advance (E)(Surplus).gba" --learn-tutorials \
      --out sessions/live-newgame-0704
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import cv2  # noqa: E402


def _ahash(bgr) -> int:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA)
    bits = (s >= s.mean()).astype("uint8").flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--title", default="mGBA")
    ap.add_argument("--goal", default="")
    ap.add_argument("--goal-file", default="")
    ap.add_argument("--knowledge", default="", help="pre-populated KnowledgeStore json to retrieve")
    ap.add_argument("--game", default="", help="game key for knowledge retrieval (rom basename)")
    ap.add_argument("--learn-tutorials", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8077")
    ap.add_argument("--settle", type=float, default=0.25, help="seconds after each input before "
                    "the change-check capture (the VLM latency is the real settle)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    goal = args.goal
    if args.goal_file:
        with open(args.goal_file) as f:
            goal = f.read().strip()

    from vga.core.contract import GameState
    from vga.core.emulator import GbaEmulator
    from vga.reason.local_reasoner import LocalVlmReasoner
    from vga.reason.plugin import VlmPlugin

    knowledge = tutor = None
    if args.knowledge or args.learn_tutorials:
        from vga.reason.knowledge import KnowledgeStore
        knowledge = KnowledgeStore(path=args.knowledge or None)
    if args.learn_tutorials:
        from vga.reason.tutor import TutorReader
        tutor = TutorReader(url=args.url, knowledge=knowledge, upscale=1)

    emu = GbaEmulator(title_keyword=args.title)
    plugin = VlmPlugin(LocalVlmReasoner(url=args.url), goal=goal, game=args.game,
                       knowledge=knowledge, tutor=tutor)

    out = args.out or os.path.join(_REPO, "sessions", f"live-{int(time.time())}")
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    steps_path = os.path.join(out, "steps.jsonl")
    print(f"[live] title={args.title!r} steps={args.steps} game={args.game!r} "
          f"knowledge={args.knowledge or '-'} tutor={bool(tutor)} -> {out}", flush=True)

    prev_frame = prev_btn = None
    prev_reason = ""
    with open(steps_path, "w") as fh:
        for i in range(args.steps):
            frame = emu.capture()
            if prev_frame is not None:
                changed = _ham(_ahash(prev_frame), _ahash(frame)) > 10
                plugin.note_outcome(prev_btn, prev_reason, changed)
            else:
                changed = None

            h, w = frame.shape[:2]
            gs = GameState(frame_index=i, timestamp=time.time(), width=w, height=h, frame=frame)
            try:
                action = plugin.decide(gs)
            except Exception as e:  # never let one bad decision kill the live run
                print(f"[live] step {i}: decide error {e!r}; waiting", flush=True)
                time.sleep(1.0)
                continue
            meta = (plugin.last_decision.meta if plugin.last_decision else {}) or {}
            btn = action.button.value if action.button is not None else "wait"
            reps = int(getattr(action, "repeats", 1) or 1)

            cv2.imwrite(os.path.join(out, "frames", f"{i:06d}.png"), frame)
            kretr = meta.get("knowledge_retrieved") or []
            rec = {"step": i, "button": btn, "repeats": reps,
                   "mode": meta.get("mode", ""), "subgoal": meta.get("subgoal", ""),
                   "reason": meta.get("reason", ""),
                   "knowledge_retrieved": kretr,
                   "commit_set": meta.get("commit_set", ""),
                   "commit_active": meta.get("commit_active"),
                   "is_tutorial": bool(meta.get("is_tutorial")),
                   "tutorial_learned": meta.get("tutorial_learned"),
                   "changed_since_prev": changed}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            cflag = "COMMIT" if meta.get("commit_set") else (f"c{meta.get('commit_active')}"
                                                             if meta.get("commit_active") is not None else "")
            tl = " LEARNED:" + str(meta.get("tutorial_learned")) if meta.get("tutorial_learned") else ""
            print(f"[live] {i:3} {btn}x{reps} mode={meta.get('mode','')[:8]:<8} "
                  f"K={len(kretr)} {cflag} tut={int(bool(meta.get('is_tutorial')))} "
                  f"| {(meta.get('reason') or '')[:60]}{tl}", flush=True)

            emu.send(action)
            time.sleep(args.settle)
            prev_frame, prev_btn, prev_reason = frame, btn, meta.get("reason", "")

    print(f"[live] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
