"""
VlmPlugin: the game-agnostic Plugin that fronts a Reasoner.

This is the "universal fallback plugin" the seam was designed for (see
core/plugin.py's docstring). It carries NO game knowledge: perceive() just wraps
the frame into a GameState, and decide() asks the Reasoner. Because it satisfies
the existing Plugin Protocol, it slots into the shared loop and registry unchanged
- but its matches() score is deliberately low so any real per-game plugin wins.

decide() stashes the last ReasonerDecision on `last_decision` so a runner can log
the reasoning/usage alongside the transition without threading it through the
core seam.
"""

from __future__ import annotations

import os
import re
from collections import deque
from difflib import SequenceMatcher
from typing import Optional

import cv2
import numpy as np

from ..core.contract import Action, Button, GameState
from .base import ReasonContext, Reasoner, ReasonerDecision, action_from_choice
from .knowledge import KnowledgeStore
from .skills import SkillStore

DEFAULT_HISTORY_LEN = 6
# Tutorial-learning: cap the pages accumulated for one tutorial before forcing a distill, so a
# model that gets stuck flagging is_tutorial forever cannot accumulate unboundedly.
TUT_MAX_PAGES = 8
# Menu commit-forcer: after this many consecutive cursor-only moves (no A) while looping in
# menu mode, force an A to break "navigate forever, never commit" paralysis. 0 disables.
DEFAULT_MENU_COMMIT_K = 4
_DIRECTIONS = {"up", "down", "left", "right"}
# Cursor directions to CYCLE through when force-moving a stuck cursor (mash-without-progress).
# Ordered so a horizontal Yes/No and a vertical list both get moved within two forcings.
_FORCE_DIRS = ["down", "right", "up", "left"]
# Scene-change threshold (of 256 aHash bits) above which a carried-over subgoal is considered
# STALE and cleared. A carried subgoal surviving a full scene change is the anchor bug (a
# "Select New Game" subgoal persisting into a battle, observed 2026-07-02). Same scene with
# minor motion differs by <~20 bits; a genuine scene change (menu->battle) by >~50.
SUBGOAL_STALE_HAMMING = 48
# State-loop ("going in circles") detector defaults. See note in ReasonContext.
# We flag a loop by DENSITY, not a raw revisit count: what fraction of the recent window is
# made of screens that recur elsewhere in the window. A tight cycle (pub rumor 3-screen loop,
# or an A/B bounce) fills the window with repeats -> high density. Arriving at a hub once
# (surrounded by novel screens en route) -> low density, no warning. This is the fix for the
# hub problem: an intentional revisit is NOT a loop, only spinning-in-place is. (2026-07-02)
DEFAULT_REVISIT_WINDOW = 30   # how many recent screen-hashes to remember (>= loop_window)
DEFAULT_REVISIT_TOL = 10      # Hamming distance (of 256 bits) counted as the "same screen"
DEFAULT_LOOP_WINDOW = 12      # recent screens examined for loop density
DEFAULT_LOOP_DENSITY = 0.6    # fraction of that window that must be recurring to call it a loop
_AHASH_SIZE = 16              # frame is reduced to 16x16 -> a 256-bit average hash
# Dialogue OCR-dedup defaults (step 2). See note in ReasonContext.
DEFAULT_DEDUP_WINDOW = 20     # how many recently-read texts to remember
DEFAULT_DEDUP_RATIO = 0.90    # SequenceMatcher ratio at/above which two reads are "the same"
_DEDUP_MIN_ALNUM = 8          # ignore reads with fewer alnum chars (avoid trivial matches)
_DIALOG_REGION_TOP = 0.65     # OCR the bottom 35% of the frame (dialogue/menu strip)


def _norm_text(s: str) -> str:
    """Normalize OCR text for dedup matching: lowercase, keep alnum+space, collapse runs.
    Robust to Tesseract noise in punctuation/spacing so the same dialog matches itself."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def _ahash(frame_bgr: np.ndarray) -> int:
    """Average-hash a frame into a 256-bit int (robust to scale/minor animation). Reduce
    to 16x16 grayscale, set each bit where the pixel is >= the mean. Near-identical screens
    (same menu, blinking cursor) hash to a tiny Hamming distance; different pages/scenes to
    a large one - so we can tell 'looped back to this screen' from 'made progress'."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (_AHASH_SIZE, _AHASH_SIZE), interpolation=cv2.INTER_AREA)
    bits = (small >= small.mean()).astype(np.uint8).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _hamming(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


class VlmPlugin:
    name = "vlm"
    family = "universal"

    # Low baseline: this plugin "handles" any game, but a real per-game plugin
    # scoring above this should always be preferred by select_plugin().
    BASELINE_MATCH = 0.25

    def __init__(self, reasoner: Reasoner, goal: str = "",
                 history_len: int = DEFAULT_HISTORY_LEN,
                 revisit_window: int = DEFAULT_REVISIT_WINDOW,
                 revisit_tol: int = DEFAULT_REVISIT_TOL,
                 loop_window: int = DEFAULT_LOOP_WINDOW,
                 loop_density: float = DEFAULT_LOOP_DENSITY,
                 enable_dedup: bool = True,
                 dedup_window: int = DEFAULT_DEDUP_WINDOW,
                 dedup_ratio: float = DEFAULT_DEDUP_RATIO,
                 skills: Optional[SkillStore] = None,
                 menu_commit_k: int = DEFAULT_MENU_COMMIT_K,
                 game: str = "", knowledge: Optional[KnowledgeStore] = None,
                 tutor=None):
        self.reasoner = reasoner
        self.goal = goal
        self._step = 0
        # Tutorial-learning (docs/TUTORIAL_LEARNING_PLAN.md). `game` keys the KnowledgeStore
        # (persistent declarative facts learned from this game's tutorials); `tutor` is an
        # optional TutorReader that reads+distills+stores when the model flags an instructional
        # screen (is_tutorial). Retrieval is always safe (empty store -> no facts); learning is
        # only active when a tutor is provided. Both are persistent across episodes, like skills.
        self.game = game
        self.knowledge = knowledge if knowledge is not None else (KnowledgeStore() if tutor is not None else None)
        self.tutor = tutor
        self._tut_pages: list = []       # frames accumulated during the current tutorial
        self._in_tutorial = False
        self._tut_last_hash = 0          # phash of the last accumulated page (dedup)
        # Menu commit-forcer: the VLM reliably PERCEIVES menus now (mode='menu') but is
        # brittle about COMMITTING - it can scroll the cursor forever without ever pressing
        # A (verified 2026-07-02: 40 steps, cursor reached the target option, A pressed 0x).
        # If we are in menu mode and the last K emitted actions were all cursor moves (no A)
        # while spinning in place, force an A to commit. Gated on the now-reliable menu mode
        # so it never fires in dialogue/field. 0 disables. Env VGA_MENU_COMMIT_K overrides
        # (clean A/B). Directly kills the paralysis without a retrain.
        self._menu_commit_k = int(os.environ.get("VGA_MENU_COMMIT_K", menu_commit_k))
        self._recent_emitted: deque[str] = deque(maxlen=max(1, self._menu_commit_k or 1))
        # Persistent, mode-keyed procedural memory. Unlike every other buffer here it
        # is NOT cleared by reset() - skills learned in one episode are meant to carry
        # into the next. Exposed as self.skills so a teacher/reflection loop can add().
        self.skills = skills or SkillStore()
        # Last screen mode the model self-reported; keys skill retrieval for the NEXT
        # step (one-frame lag, like subgoal/progress). Empty until the first report.
        self._mode = ""
        # Rolling log of recent EXECUTED steps (the loop reports the post-unstick
        # button + whether it changed the screen via note_outcome). Oldest first.
        self._history: deque[dict] = deque(maxlen=max(1, history_len))
        # Recent screen perceptual-hashes, for the loop-density "going in circles" signal.
        self._recent_hashes: deque[int] = deque(maxlen=max(1, revisit_window))
        self._revisit_tol = revisit_tol
        self._loop_window = max(2, loop_window)
        self._loop_density = loop_density
        # Dialogue OCR-dedup: recent normalized reads + a Tesseract reader (lazy, optional).
        self._recent_texts: deque[str] = deque(maxlen=max(1, dedup_window))
        self._dedup_ratio = dedup_ratio
        # Goal/task-state scratchpad (layer b): the model's own current sub-goal + progress,
        # carried across steps. The model authors and updates these; we just persist them.
        self._subgoal = ""
        self._progress = ""
        # aHash of the frame the current subgoal was set on; used to clear the subgoal when the
        # scene changes a lot (anti-anchor). None when there is no live subgoal.
        self._subgoal_hash: Optional[int] = None
        # Rotating index into _FORCE_DIRS for the mash-break move-forcer.
        self._force_i = 0
        self._ocr = None
        if enable_dedup:
            try:  # Ocr is a self-contained Tesseract wrapper; keep the import local so a
                  # missing dep just disables dedup instead of breaking the whole plugin.
                from ..plugins.pokemon.ocr import Ocr
                self._ocr = Ocr(logger=lambda *_: None)
                if not getattr(self._ocr, "enabled", False):
                    self._ocr = None
            except Exception:
                self._ocr = None
        self.last_decision: Optional[ReasonerDecision] = None

    def _loop_score(self, cur_hash: int) -> float:
        """Fraction of the recent window (last loop_window screens, incl. the current one)
        that recurs elsewhere in the window. ~1.0 = spinning through a small set of screens;
        ~0.0 = each recent screen is novel (making progress). Distinguishes a pathological
        loop from a one-off hub revisit, which is surrounded by novel screens and scores low."""
        window = list(self._recent_hashes)[-(self._loop_window - 1):] + [cur_hash]
        if len(window) < self._loop_window:
            return 0.0   # not enough history yet to judge a loop; stay quiet
        repeats = 0
        for i, h in enumerate(window):
            if any(_hamming(h, o) <= self._revisit_tol
                   for j, o in enumerate(window) if j != i):
                repeats += 1
        return repeats / len(window)

    def _read_dialog(self, frame: np.ndarray) -> tuple[str, bool]:
        """OCR the bottom dialogue/menu strip and decide if we've read this text before.
        Returns (dialog_text, already_read). Advisory: any failure yields ("", False)."""
        if self._ocr is None or frame is None or frame.size == 0:
            return "", False
        h = frame.shape[0]
        region = frame[int(h * _DIALOG_REGION_TOP):, :]
        text = self._ocr.read(region)
        norm = _norm_text(text)
        if sum(c.isalnum() for c in norm) < _DEDUP_MIN_ALNUM:
            return "", False   # too little text to be a meaningful dialog signature
        already = any(SequenceMatcher(None, norm, prev).ratio() >= self._dedup_ratio
                      for prev in self._recent_texts)
        self._recent_texts.append(norm)
        return text, already

    def note_outcome(self, button: str, reason: str, changed: bool) -> None:
        """Record the outcome of the action just EXECUTED by the loop: which button
        actually went to the game (post-unstick) and whether the screen changed as a
        result. Called by the runner once the next frame is available to diff. This is
        the working memory decide() hands to the reasoner. Recording the executed
        button (not the VLM's pre-unstick choice) keeps the memory truthful: the model
        learns "down did nothing 6x", not "down worked" when an unstick override moved."""
        self._history.append({"button": button, "reason": reason, "changed": bool(changed)})

    def matches(self, frame: np.ndarray) -> float:
        return self.BASELINE_MATCH

    def perceive(self, frame: np.ndarray, frame_index: int, timestamp: float) -> GameState:
        # Game-agnostic: no detections, no OCR, no regions. Just the frame and its
        # geometry. The reasoner works from the pixels directly.
        h, w = frame.shape[:2]
        return GameState(frame_index=frame_index, timestamp=timestamp,
                         width=w, height=h, frame=frame)

    def decide(self, state: GameState) -> Action:
        # last_action/last_changed come from the most recent EXECUTED step (history[-1]),
        # NOT the VLM's own chosen action string. We deliberately do not feed str(action)
        # back: it embeds the model's own reason ("screen is black..."), which the model
        # then parrots and gets stuck on (observed 2026-07-01). The behavioral log +
        # change signal is the framing; keep it purely factual.
        last = self._history[-1] if self._history else None
        # Revisit signal: how many recent screens are near-identical to the current one.
        # Count BEFORE appending the current hash so we don't count the frame against itself.
        cur_hash = _ahash(state.frame)
        # Loop signal by DENSITY (not raw count): only a tight cycle scores high; a lone
        # intentional revisit (e.g. re-entering a hub) stays low and does NOT warn.
        looping = self._loop_score(cur_hash) >= self._loop_density
        self._recent_hashes.append(cur_hash)
        # Anti-anchor: drop a carried-over subgoal once the scene has changed a lot since it was
        # set. Without this, a subgoal like "Select New Game" persisted from the title screen all
        # the way into a battle (echo/anchor bug, 3rd occurrence 2026-07-02), so the model kept
        # trying to start a new game mid-battle. Cleared -> the model re-derives from the CURRENT
        # screen. Only fires on a genuine scene change, so a stable plan across similar screens
        # survives.
        if (self._subgoal and self._subgoal_hash is not None
                and _hamming(cur_hash, self._subgoal_hash) > SUBGOAL_STALE_HAMMING):
            self._subgoal = ""
            self._progress = ""
            self._subgoal_hash = None
        # Dialogue OCR-dedup: have we already read the text currently on screen?
        dialog_text, already_read = self._read_dialog(state.frame)
        # Persistent procedural skills for the mode the model reported LAST step. One
        # frame behind on purpose (the current mode isn't known until the model answers).
        skills = self.skills.retrieve(self._mode)
        # Declarative tutorial-knowledge for this game, ranked by relevance to the current
        # context (mode + on-screen text) so only pertinent facts hit the prompt. Empty when
        # no store / nothing learned yet / VGA_KNOWLEDGE_OFF.
        knowledge = []
        if self.knowledge is not None and self.game:
            knowledge = self.knowledge.retrieve(self.game, context=f"{self._mode} {dialog_text}", limit=3)
        ctx = ReasonContext(
            goal=self.goal, step=self._step,
            last_action=(last["button"] if last else ""),
            last_changed=(last["changed"] if last else None),
            history=list(self._history),
            looping=looping,
            dialog_text=dialog_text,
            already_read=already_read,
            subgoal=self._subgoal,
            progress=self._progress,
            mode=self._mode,
            skills=skills,
            knowledge=knowledge,
        )
        decision = self.reasoner.decide(state.frame, ctx)
        self.last_decision = decision
        # Commit scaffold. The model reliably PERCEIVES a decision screen but is brittle about
        # goal-directed CURSOR COMMIT - two failure modes seen live (2026-07-02): PARALYSIS
        # (moves the cursor forever, never presses A) and MASH (presses A/B in place without
        # ever moving the cursor to the target - the Yes/No-defaults-to-No trap and the battle
        # "open Move then cancel" loop). When we are demonstrably STUCK (looping in place) and
        # the last K emitted actions are all one unproductive pattern, force the COMPLEMENTARY
        # action to break it. Gated on `looping` (not on a mode label, which the model often
        # gets wrong - it called a battle "overworld"): healthy dialogue-advance changes the
        # screen each step so it never looks like a loop, and so is never overridden.
        chosen = decision.action.button.value if decision.action.button is not None else "wait"
        forced = None
        if self._menu_commit_k and looping and len(self._recent_emitted) >= self._menu_commit_k:
            recent = list(self._recent_emitted)
            if all(b in _DIRECTIONS for b in recent):
                forced = ("A", "commit-forced: moved the cursor repeatedly without selecting")
            elif all(b in ("A", "B") for b in recent):
                d = _FORCE_DIRS[self._force_i % len(_FORCE_DIRS)]
                self._force_i += 1
                forced = (d, "move-forced: pressed A/B in place without progress; move the cursor")
        if forced is not None:
            action = action_from_choice(forced[0], note=forced[1])
            if decision.meta is not None:
                decision.meta["commit_scaffold"] = forced[0]
            emitted = forced[0]
        else:
            action = decision.action
            emitted = chosen
        self._recent_emitted.append(emitted)
        # Carry the model's updated scratchpad forward. Keep the prior value if the model
        # returned an empty string, so a momentary omission doesn't wipe the plan.
        new_subgoal = (decision.meta or {}).get("subgoal")
        new_progress = (decision.meta or {}).get("progress")
        new_mode = (decision.meta or {}).get("mode")
        if new_subgoal:
            self._subgoal = str(new_subgoal)
            # Stamp the scene this subgoal was set on, so the anti-anchor check above can tell
            # when the screen has moved on and the subgoal is stale.
            self._subgoal_hash = cur_hash
        if new_progress:
            self._progress = str(new_progress)
        if new_mode:
            self._mode = str(new_mode).strip().lower()
        # Tutorial-learning: if the model flagged this as an instructional screen, accumulate
        # the page (deduped by phash - the agent re-reads a page before advancing); when the
        # tutorial ends (a non-tutorial screen after we had pages) distill+store the whole thing.
        if self.tutor is not None and self.game:
            is_tut = bool((decision.meta or {}).get("is_tutorial"))
            if is_tut:
                if not self._tut_pages or _hamming(cur_hash, self._tut_last_hash) > self._revisit_tol:
                    self._tut_pages.append(state.frame.copy())
                    self._tut_last_hash = cur_hash
                self._in_tutorial = True
                if len(self._tut_pages) >= TUT_MAX_PAGES:
                    self._flush_tutorial()
            elif self._in_tutorial:
                self._flush_tutorial()
        self._step += 1
        return action

    def _flush_tutorial(self) -> None:
        """Distill the accumulated tutorial pages into a fact and store it. Best-effort: any
        failure (server down, parse error) is swallowed so tutorial-learning never breaks play."""
        pages, self._tut_pages = self._tut_pages, []
        self._in_tutorial = False
        self._tut_last_hash = 0
        if not pages or self.tutor is None or not self.game:
            return
        try:
            res = self.tutor.learn(self.game, pages)
            if res and self.last_decision is not None and self.last_decision.meta is not None:
                self.last_decision.meta["tutorial_learned"] = {
                    "topic": res.get("topic"), "added": res.get("added")}
        except Exception:
            pass

    def reset(self) -> None:
        self._step = 0
        self._history.clear()
        self._recent_hashes.clear()
        self._recent_texts.clear()
        self._subgoal = ""
        self._progress = ""
        self._subgoal_hash = None
        self._mode = ""
        self._force_i = 0
        self._recent_emitted.clear()
        # Drop any in-progress tutorial accumulation (a tutorial spanning an episode boundary
        # is rare and the server may be unavailable at reset). NOTE: neither self.skills NOR
        # self.knowledge is reset - both are learn-once-keep-forever across episodes.
        self._tut_pages = []
        self._in_tutorial = False
        self._tut_last_hash = 0
        self.last_decision = None
        self.reasoner.reset()
