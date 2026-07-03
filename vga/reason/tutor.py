"""
TutorReader - the read+distill half of tutorial-learning (docs/TUTORIAL_LEARNING_PLAN.md).

Given the frames of an in-game tutorial/help/encyclopedia screen, it:
  1. TRANSCRIBES each page via the VLM server's /read endpoint (word-perfect on the GBA font,
     verified 2026-07-02 - no separate OCR/CRNN needed; this IS the "Tesseract swap"),
  2. accumulates the pages into one transcript,
  3. DISTILLS the transcript into a single compact, actionable FACT,
  4. stores it in the KnowledgeStore (declarative, game-keyed) if it is a real mechanic.

Kept host-side and self-contained (urllib, PIL) like local_reasoner.py; the VLM server does
the heavy lifting. Reading uses /read (free-form, greedy) NOT /act (action-forced).
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
from typing import Optional

from PIL import Image

from .knowledge import KnowledgeStore

_TRANSCRIBE = ("Transcribe ALL on-screen text exactly as written, including any heading in "
               "brackets. Output only the text, nothing else.")

_DISTILL = ("The following is the full text of an in-game tutorial/help screen for a video "
            "game, read across one or more pages:\n\n{text}\n\nIn ONE sentence, state the "
            "single most useful game-mechanics FACT a player should remember and act on. Be "
            "concrete and actionable. If it is ONLY flavor or lore with no actionable mechanic, "
            "output exactly: NO ACTIONABLE FACT")

_NO_FACT = "NO ACTIONABLE FACT"


def _to_pil(frame) -> Image.Image:
    """Accept a PIL image, a path, or a BGR/RGB ndarray (what capture()/plugin pass)."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, str):
        return Image.open(frame).convert("RGB")
    # assume ndarray; the plugin uses BGR (OpenCV), so flip to RGB.
    import numpy as np
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]
    return Image.fromarray(arr).convert("RGB")


def _topic_from(transcript: str) -> str:
    """Pull a short topic label from a tutorial heading: [How Laws Came To Be],
    [! Win/Lose Conditions !], etc. Fall back to the first non-generic line."""
    m = re.search(r"\[\s*!?\s*(.*?)\s*!?\s*\]", transcript)
    if m:
        return m.group(1).strip()
    for line in transcript.splitlines():
        s = line.strip()
        if s and s.upper() not in ("RUMORS", "HELP", "NOTICE", "LIST"):
            return s[:60]
    return ""


def is_no_fact(text: str) -> bool:
    return (not text) or (_NO_FACT.lower() in text.strip().lower()[:40])


class TutorReader:
    def __init__(self, url: str = "http://127.0.0.1:8077",
                 knowledge: Optional[KnowledgeStore] = None, upscale: int = 3):
        self.url = url.rstrip("/")
        self.k = knowledge or KnowledgeStore()
        self.upscale = upscale

    def _post_read(self, pil: Image.Image, prompt: str, max_tok: int) -> str:
        if self.upscale and self.upscale != 1:
            pil = pil.resize((pil.width * self.upscale, pil.height * self.upscale), Image.NEAREST)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        body = {"image_b64": base64.b64encode(buf.getvalue()).decode(),
                "prompt": prompt, "max_new_tokens": max_tok}
        req = urllib.request.Request(self.url + "/read", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["text"].strip()

    def transcribe(self, frame) -> str:
        return self._post_read(_to_pil(frame), _TRANSCRIBE, 256)

    def learn(self, game: str, frames: list) -> Optional[dict]:
        """Read tutorial `frames` (pages, in order), distill, and store if actionable.
        Returns {topic, fact, added, transcript} or None if there was no actionable fact."""
        pages = [_to_pil(f) for f in frames]
        transcript = "\n".join(self._post_read(p, _TRANSCRIBE, 256) for p in pages)
        # Distill from the accumulated TEXT (the last frame is sent only because /read is
        # image-based; the instruction points the model at the provided transcript).
        fact = self._post_read(pages[-1], _DISTILL.format(text=transcript), 90)
        if is_no_fact(fact):
            return None
        topic = _topic_from(transcript)
        added = self.k.add(game, topic, fact, source="learned")
        return {"topic": topic, "fact": fact, "added": added, "transcript": transcript}
