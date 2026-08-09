"""
LocalVlmReasoner - host-side client for the on-Thor VLM server (serve/serve_vlm.py).

The model runs in the thor-vlm container (GPU); the capture/emulator loop runs on the
host (needs the X display). So the reasoner is a thin HTTP client: encode the frame to
PNG, POST it to localhost, get back the structured action. Same Reasoner interface as
the stub and Claude backends - the loop, plugin, and trajectory logger are unchanged.

No third-party deps (stdlib urllib) so the host venv needs nothing extra.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

import cv2
import numpy as np

from .base import Reasoner, ReasonContext, ReasonerDecision, action_from_choice

DEFAULT_URL = "http://127.0.0.1:8077"


class LocalVlmReasoner:
    name = "local"

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._check_health()

    def _check_health(self) -> None:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
                info = json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Local VLM server not reachable at {self.url} ({e}). "
                "Start it with serve/run_server.sh (and wait for the model to load)."
            ) from e
        if not info.get("ok"):
            raise RuntimeError(
                f"Local VLM server at {self.url} is up but the model is not loaded yet."
            )

    @staticmethod
    def _encode_png(frame_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", frame_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed to encode frame")
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def decide(self, frame_bgr: np.ndarray, context: ReasonContext) -> ReasonerDecision:
        payload = json.dumps({
            "image_b64": self._encode_png(frame_bgr),
            "goal": context.goal,
            "step": context.step,
            "last_action": context.last_action,
            "last_changed": context.last_changed,
            "history": context.history,
            "looping": context.looping,
            "dialog_text": context.dialog_text,
            "already_read": context.already_read,
            "subgoal": context.subgoal,
            "progress": context.progress,
            "mode": context.mode,
            "skills": context.skills,
            "knowledge": context.knowledge,
            "committed": context.committed,
            # Durable RAM-taught task-state phase directive (Task 07). Authoritative "you have
            # already accepted; now travel" context that survives the scratchpad wipe. Empty
            # unless the phase machine is active AND has advanced. See plugin.py / serve_vlm.py.
            "task_phase": context.task_phase,
            # Reproducible-eval seed salt: mixed into the server's per-input RNG seed so the SAME
            # state yields a DIFFERENT-but-reproducible trajectory per salt. Lets a ladder run N
            # seeds/arm to separate a real mechanism effect from single-seed luck. Empty (default)
            # => byte-identical to the un-salted seed (appending "" is a no-op). Set VGA_SEED_SALT.
            "seed_salt": os.environ.get("VGA_SEED_SALT", ""),
        }).encode()
        req = urllib.request.Request(
            self.url + "/act", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read().decode())

        button = resp.get("button", "wait")
        repeats = resp.get("repeats", 1)
        reason = resp.get("reason", "")
        action = action_from_choice(button, repeats=repeats, note=reason[:60])
        return ReasonerDecision(
            action=action,
            reason=reason,
            meta={
                "model": "local-vlm",
                "button": button,
                "repeats": repeats,
                # The model's one-sentence rationale. Also on ReasonerDecision.reason, but the
                # oracle rollout logger reads meta["reason"] - without this key it logged EMPTY
                # reasons for every step (pre-existing gap found Task 01, 2026-07-04), blinding
                # trajectory diagnosis to WHY the agent acted. Mirror it into meta so it is logged.
                "reason": reason,
                "latency_s": resp.get("latency_s"),
                "raw": resp.get("raw"),
                # Goal/task-state scratchpad the model authored this step (layer b).
                "subgoal": resp.get("subgoal", ""),
                "progress": resp.get("progress", ""),
                # Screen mode the model classified this step (keys skill retrieval next step).
                "mode": resp.get("mode", ""),
                # Perception-first menu observation: the option the model reports as highlighted.
                "highlighted": resp.get("highlighted", ""),
                # Objective scene descriptor (Task 03B): the plugin carries this forward one step
                # and folds it into the retrieval context so learned facts/skills route by scene
                # CONTENT, not a bare one-word mode. See vga/reason/plugin.py:_last_scene.
                "scene": resp.get("scene", ""),
                # Tutorial-learning: did the model flag this as an instructional screen?
                "is_tutorial": bool(resp.get("is_tutorial", False)),
            },
        )

    def reset(self) -> None:
        pass
