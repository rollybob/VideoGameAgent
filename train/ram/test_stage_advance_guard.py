#!/usr/bin/env python3
"""Deterministic unit test of the stage-advance over-advance guard (no VLM/GPU/emulator).

Reproduces the bench-herb-stages3 failure in miniature: a stub reasoner that declares
"STAGE DONE" every step, driven over synthetic frames. The guard must:
  - advance the one-way stage pointer on the FIRST STAGE DONE (fresh screen), and
  - REFUSE a second STAGE DONE on the SAME unchanged screen (the step-49/50 double-jump
    that burned the open-menu stage), and
  - advance again once the screen has materially CHANGED.
This is the offline gate before spending a bench (Tim's brittleness rule)."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/timothy/projects/VGA")
from vga.core.contract import GameState
from vga.reason.base import ReasonerDecision, action_from_choice


class StageDoneReasoner:
    name = "rec"
    def decide(self, frame, ctx):
        return ReasonerDecision(action=action_from_choice("wait"), reason="",
                                meta={"subgoal": "STAGE DONE", "progress": "", "mode": ""})
    def reset(self): pass


def gs(frame, i):
    h, w = frame.shape[:2]
    return GameState(frame_index=i, timestamp=time.time(), width=w, height=h, frame=frame)


def main():
    from vga.reason.plugin import VlmPlugin
    goal = "STAGE: one\nSTAGE: two\nSTAGE: three\nSTAGE: four"
    plug = VlmPlugin(reasoner=StageDoneReasoner(), goal=goal, enable_dedup=False)

    # Two distinct screens (large Hamming apart), each as a stable BGR frame.
    screenA = np.zeros((160, 240, 3), dtype=np.uint8)
    screenB = np.zeros((160, 240, 3), dtype=np.uint8)
    screenB[:80, :] = 255   # top half white -> very different average hash

    seq = []
    plug.decide(gs(screenA, 0)); seq.append(("A", plug._stage_i))   # 0->1 (first advance)
    plug.decide(gs(screenA, 1)); seq.append(("A", plug._stage_i))   # same screen -> BLOCKED, stays 1
    plug.decide(gs(screenA, 2)); seq.append(("A", plug._stage_i))   # still same -> BLOCKED, stays 1
    plug.decide(gs(screenB, 3)); seq.append(("B", plug._stage_i))   # screen changed -> 1->2
    plug.decide(gs(screenB, 4)); seq.append(("B", plug._stage_i))   # same again -> BLOCKED, stays 2

    for scr, si in seq:
        print(f"  frame={scr}  stage_i={si}")
    got = [si for _, si in seq]
    want = [1, 1, 1, 2, 2]
    ok = got == want
    print(f"\nstage_i sequence got={got} want={want}")
    print("RESULT:", "PASS - guard blocks same-screen double-advance, allows on change" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
