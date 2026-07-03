"""
Transition logging - the "log from day one" discipline (docs/NORTH_STAR.md Sec 8).

Every step of a run is recorded as a (frame_before, action, frame_after)
transition: PNG frames on disk + one JSONL line per step. This is the training
data for both the distilled reactive policy (behavior cloning) and the world
model - the scaffold generating its own replacement. Two producers share this
format so their data is interchangeable:
  - the autonomous reasoner loop (run_reasoner.py), in-process; and
  - the human/Claude teacher harness (teacher.py), across separate CLI calls.

Hence the split API: save_frame() + write_transition() are low-level and
path-based (safe across processes); log_transition() is the in-process convenience
that saves both frames from GameStates and writes the row.

Layout under run_dir/:
    frames/000000.png, 000001.png, ...   (each frame saved once, by frame_index)
    steps.jsonl                          (one transition per line)
    meta.json                            (run-level metadata; written once)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import cv2
import numpy as np

from .core.contract import Action, GameState


class TrajectoryLogger:
    def __init__(self, run_dir: str, run_meta: Optional[dict] = None):
        self.run_dir = run_dir
        self.frames_dir = os.path.join(run_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self._steps_path = os.path.join(run_dir, "steps.jsonl")
        self._steps = open(self._steps_path, "a", buffering=1)  # line-buffered, append
        # count = existing lines, so re-opened runs (teacher across calls) continue.
        self.count = self._existing_line_count()
        # Write meta once; don't clobber it on every re-open.
        meta_path = os.path.join(run_dir, "meta.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                json.dump({"started": time.time(), **(run_meta or {})}, f, indent=2)

    def _existing_line_count(self) -> int:
        if not os.path.exists(self._steps_path):
            return 0
        with open(self._steps_path) as f:
            return sum(1 for _ in f)

    def frame_path(self, frame_index: int) -> str:
        return os.path.join("frames", f"{frame_index:06d}.png")

    def save_frame(self, frame_index: int, frame_bgr: np.ndarray) -> str:
        """Save a frame by index (BGR, as capture() produces); return its relative
        path. Overwrites are harmless (same index -> same content)."""
        rel = self.frame_path(frame_index)
        cv2.imwrite(os.path.join(self.run_dir, rel), frame_bgr)
        return rel

    @staticmethod
    def action_dict(action: Action) -> dict:
        return {
            "button": action.button.value if action.button else None,
            "duration": action.duration,
            "repeats": action.repeats,
            "gap": action.gap,
            "note": action.note,
        }

    def write_transition(self, frame_index: int, t: float, frame_before: Optional[str],
                         frame_after: Optional[str], action: Action,
                         meta: Optional[dict] = None) -> None:
        """Append one transition row from already-saved frame paths."""
        row: dict[str, Any] = {
            "step": self.count,
            "t": t,
            "frame_index": frame_index,
            "frame_before": frame_before,
            "frame_after": frame_after,
            "action": self.action_dict(action),
            "meta": meta or {},
        }
        self._steps.write(json.dumps(row) + "\n")
        self.count += 1

    def log_transition(self, prev_state: GameState, action: Action,
                       next_state: GameState, meta: Optional[dict] = None) -> None:
        """In-process convenience: save both frames and write the row."""
        before = self.save_frame(prev_state.frame_index, prev_state.frame) \
            if prev_state.frame is not None else None
        after = self.save_frame(next_state.frame_index, next_state.frame) \
            if next_state.frame is not None else None
        self.write_transition(prev_state.frame_index, prev_state.timestamp,
                              before, after, action, meta)

    def close(self) -> None:
        try:
            self._steps.close()
        except Exception:
            pass
