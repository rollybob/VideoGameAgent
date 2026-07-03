"""
Teacher harness - generate seed trajectories with a human (or Claude) in the loop.

This is the "teacher under the Max plan" path (docs/NORTH_STAR.md Sec 6): instead
of paying a per-token API, the loop is driven turn-by-turn by whoever is at the
keyboard - including Claude Code reading the frame and choosing the input. Every
(frame -> action + reason) is logged in the same format as the autonomous loop, so
this seed data feeds the exact same distillation/world-model pipeline later.

It is a two-call cycle because each CLI invocation is a fresh process:

    python teacher.py frame                 # capture current frame -> frames/NNNNNN.png
    #   (look at that PNG, decide)
    python teacher.py act A --reason "..."   # send the input to the emulator
    python teacher.py frame                 # capture the result; THIS logs the
    #                                         (prev_frame, action, this_frame) transition
    ...
    python teacher.py end                    # finalize

State (which frame is pending, what action was sent) is persisted in the run dir,
and the active run dir is remembered in sessions/.teacher-active so you don't have
to repeat --run-dir. Use --fake to exercise the harness without X/emulator.

Buttons: up down left right A B start select L R wait  (--repeats for a short mash).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

from vga.reason.base import BUTTON_CHOICES, action_from_choice
from vga.trajectory import TrajectoryLogger

ACTIVE_POINTER = os.path.join("sessions", "state-teacher-active")  # holds the active run dir


class _FakeEmulator:
    def __init__(self, w: int = 240, h: int = 160):
        self._w, self._h = w, h
        self._rng = np.random.default_rng()

    def capture(self) -> np.ndarray:
        return self._rng.integers(0, 256, size=(self._h, self._w, 3), dtype=np.uint8)

    def send(self, action) -> None:
        pass


def _build_emulator(fake: bool, title: str):
    if fake:
        return _FakeEmulator()
    from vga.core.emulator import EmulatorNotFound, GbaEmulator
    try:
        return GbaEmulator(title_keyword=title)
    except EmulatorNotFound as e:
        print(f"[teacher] {e}")
        sys.exit(2)


def _state_path(run_dir: str) -> str:
    return os.path.join(run_dir, "teacher_state.json")


def _load_state(run_dir: str) -> dict:
    p = _state_path(run_dir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"next_index": 0, "pending": None}


def _save_state(run_dir: str, state: dict) -> None:
    with open(_state_path(run_dir), "w") as f:
        json.dump(state, f, indent=2)


def _resolve_run_dir(args, create: bool) -> Optional[str]:
    """--run-dir wins; else the remembered active run; else (only for `frame`)
    create a fresh one. Returns None if none and not creating."""
    if args.run_dir:
        run_dir = args.run_dir
    elif os.path.exists(ACTIVE_POINTER):
        run_dir = open(ACTIVE_POINTER).read().strip()
    elif create:
        run_dir = os.path.join("sessions", f"teacher-{int(time.time())}")
    else:
        return None
    if create:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs("sessions", exist_ok=True)
        with open(ACTIVE_POINTER, "w") as f:
            f.write(run_dir)
    return run_dir


def cmd_frame(args) -> int:
    run_dir = _resolve_run_dir(args, create=True)
    logger = TrajectoryLogger(run_dir, run_meta={"producer": "teacher", "fake": args.fake})
    state = _load_state(run_dir)
    emu = _build_emulator(args.fake, args.title)

    idx = state["next_index"]
    frame = emu.capture()
    rel = logger.save_frame(idx, frame)

    pending = state.get("pending")
    if pending and pending.get("action") is not None:
        # Close the previous transition: prev frame + its action -> this new frame.
        a = pending["action"]
        action = action_from_choice(a["button"], repeats=a.get("repeats", 1), note=a.get("note", ""))
        logger.write_transition(
            frame_index=pending["frame_index"], t=pending["t"],
            frame_before=pending["frame_before"], frame_after=rel,
            action=action, meta=pending.get("meta", {}),
        )
        print(f"[teacher] logged transition {logger.count - 1}: "
              f"frame {pending['frame_index']} + {action} -> frame {idx}")

    state["pending"] = {"frame_index": idx, "frame_before": rel, "t": time.time(), "action": None}
    state["next_index"] = idx + 1
    _save_state(run_dir, state)
    logger.close()

    abspath = os.path.join(run_dir, rel)
    print(f"[teacher] captured frame {idx} -> {abspath}")
    print(f"[teacher] LOOK at that image, then: python teacher.py act <button> --reason \"...\"")
    print(f"[teacher] buttons: {' '.join(BUTTON_CHOICES)}")
    return 0


def cmd_act(args) -> int:
    run_dir = _resolve_run_dir(args, create=False)
    if run_dir is None:
        print("[teacher] no active run. Run `python teacher.py frame` first.")
        return 2
    state = _load_state(run_dir)
    pending = state.get("pending")
    if not pending:
        print("[teacher] nothing pending. Run `python teacher.py frame` first.")
        return 2
    if args.button not in BUTTON_CHOICES:
        print(f"[teacher] unknown button '{args.button}'. Choices: {' '.join(BUTTON_CHOICES)}")
        return 2

    action = action_from_choice(args.button, repeats=args.repeats, note=args.reason[:60])
    emu = _build_emulator(args.fake, args.title)
    emu.send(action)

    pending["action"] = {"button": args.button, "repeats": args.repeats, "note": args.reason[:60]}
    pending["meta"] = {"reason": args.reason, "teacher": "claude-code"}
    state["pending"] = pending
    _save_state(run_dir, state)

    print(f"[teacher] sent {action} for frame {pending['frame_index']}.")
    print(f"[teacher] now: python teacher.py frame   (captures the result and logs the transition)")
    return 0


def cmd_status(args) -> int:
    run_dir = _resolve_run_dir(args, create=False)
    if run_dir is None:
        print("[teacher] no active run.")
        return 0
    state = _load_state(run_dir)
    steps = os.path.join(run_dir, "steps.jsonl")
    n = sum(1 for _ in open(steps)) if os.path.exists(steps) else 0
    pending = state.get("pending")
    print(f"[teacher] run_dir={run_dir}")
    print(f"[teacher] transitions logged: {n}  |  next frame index: {state.get('next_index', 0)}")
    if pending:
        act = pending.get("action")
        print(f"[teacher] pending frame {pending['frame_index']}"
              + (f", action queued: {act['button']}x{act.get('repeats', 1)}" if act else ", awaiting an action"))
    return 0


def cmd_end(args) -> int:
    run_dir = _resolve_run_dir(args, create=False)
    if os.path.exists(ACTIVE_POINTER):
        os.remove(ACTIVE_POINTER)
    if run_dir:
        state = _load_state(run_dir)
        steps = os.path.join(run_dir, "steps.jsonl")
        n = sum(1 for _ in open(steps)) if os.path.exists(steps) else 0
        print(f"[teacher] ended. {n} transitions in {run_dir}.")
    else:
        print("[teacher] no active run to end.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="VGA teacher harness (human/Claude in the loop)")
    ap.add_argument("--run-dir", default=None, help="trajectory dir (default: remembered active run)")
    ap.add_argument("--title", default="mGBA", help="emulator window title keyword")
    ap.add_argument("--fake", action="store_true", help="fake emulator (random frames), no X/emulator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("frame", help="capture the current frame (and log the prior transition)")
    pa = sub.add_parser("act", help="send one controller input")
    pa.add_argument("button", help="one of: " + " ".join(BUTTON_CHOICES))
    pa.add_argument("--repeats", type=int, default=1, help="tap count (short mash)")
    pa.add_argument("--reason", default="", help="one-line rationale (logged as teacher label)")
    sub.add_parser("status", help="show pending state and transition count")
    sub.add_parser("end", help="finalize the active run")

    args = ap.parse_args()
    return {"frame": cmd_frame, "act": cmd_act, "status": cmd_status, "end": cmd_end}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
