"""
The Claude-backed reasoner: a frontier VLM as the System-2 teacher.

This is the disposable scaffold from docs/NORTH_STAR.md Sec 6 - a strong vision
model drives the loop during development while we log trajectories to distill a
local model later. It reads ANTHROPIC_API_KEY from the environment; we do NOT
repurpose any other credential. If the key is absent, construction fails loudly
so the caller can fall back to the stub reasoner.

Model default is claude-opus-4-8 (the strongest vision reasoner). It is called at
decision points, not every frame, so per-call latency is acceptable; override the
model for cheaper/faster runs. We force a single strict `act` tool call so the
response is a guaranteed-valid structured action, not free text.
"""

from __future__ import annotations

import base64
import time
from typing import Optional

import cv2
import numpy as np

from .base import Reasoner, ReasonContext, ReasonerDecision, action_from_choice
from .prompt import SYSTEM_PROMPT, act_tool, user_text

# Opus 4.8: 1M context, vision-capable. Thinking is off unless requested; we keep
# it off for latency and rely on the forced tool call for a terse structured reply.
DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeReasoner:
    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024,
                 api_key: Optional[str] = None):
        # Lazy import so the rest of the package loads without the SDK installed.
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "The 'anthropic' package is required for ClaudeReasoner. "
                "Install it in the VGA venv (pip install anthropic), or use the "
                "stub reasoner for offline runs."
            ) from e
        import anthropic
        # Anthropic() resolves ANTHROPIC_API_KEY from the environment. We pass
        # api_key only if explicitly provided.
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        # The SDK defers the auth error to request time; fail fast here instead so
        # a keyless run gives a clear message rather than dying on the first frame.
        if not (self._client.api_key or getattr(self._client, "auth_token", None)):
            raise RuntimeError(
                "No Anthropic credential found. Set ANTHROPIC_API_KEY in the "
                "environment (VGA does not borrow Claude Code's auth), or run with "
                "--stub / --selftest for a no-API loop."
            )
        self.model = model
        self.max_tokens = max_tokens
        self._tool = act_tool()

    @staticmethod
    def _encode_png(frame_bgr: np.ndarray) -> str:
        """BGR frame -> base64 PNG. cv2 takes BGR and writes correct colors."""
        ok, buf = cv2.imencode(".png", frame_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed to encode frame")
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def decide(self, frame_bgr: np.ndarray, context: ReasonContext) -> ReasonerDecision:
        img_b64 = self._encode_png(frame_bgr)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": user_text(
                    context.goal, context.step, context.last_action)},
            ],
        }]
        t0 = time.time()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[self._tool],
            tool_choice={"type": "tool", "name": "act"},
            messages=messages,
        )
        latency = time.time() - t0

        tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
        if tool_use is None:
            # Forced tool_choice should guarantee a tool call; treat absence as a
            # safe no-op rather than crashing the loop.
            return ReasonerDecision(
                action=action_from_choice("wait", note="no tool_use in response"),
                reason="model returned no action",
                meta={"model": self.model, "latency_s": latency,
                      "stop_reason": resp.stop_reason},
            )
        inp = tool_use.input
        button = inp.get("button", "wait")
        repeats = inp.get("repeats", 1)
        reason = inp.get("reason", "")
        action = action_from_choice(button, repeats=repeats, note=reason[:60])
        return ReasonerDecision(
            action=action,
            reason=reason,
            meta={
                "model": self.model,
                "latency_s": round(latency, 3),
                "button": button,
                "repeats": repeats,
                "input_tokens": getattr(resp.usage, "input_tokens", None),
                "output_tokens": getattr(resp.usage, "output_tokens", None),
                "request_id": getattr(resp, "_request_id", None),
            },
        )

    def reset(self) -> None:
        pass
