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
#  The 5th slot escalates to START: some screens commit ONLY via Start (the naming
# keyboard's OK is the canonical case -- naming baseline 2026-07-03: the model A-mashed
# letters for 40 steps, 0/8 ever pressed Start, because A "confirms" on its GBA prior).
# Directions are tried first (cheap, reversible); Start comes up once a full direction
# cycle has failed to break the mash. Start is safe-reversible on GBA UIs (menus toggle).
_FORCE_DIRS = ["down", "right", "up", "left", "start"]
# Anti-freeze escape (2026-07-06). The DOMINANT real failure, measured across trajectories: the
# agent enters a state where its chosen button is a NO-OP (the screen aHash does not move) and it
# keeps re-justifying the SAME button for dozens of steps -- base_noknow_s0 spent 73% of an episode
# frozen, incl. a 44-step run pressing A at a static "tutorial" it could not dismiss. The menu
# commit-forcer above misses this: it needs a HOMOGENEOUS recent-button window (all d-pad, or all
# A/B) and keys off loop-DENSITY, but a real flail is "mostly A with a few stray moves" -> reads as
# mixed -> never forced. So key off the GROUND TRUTH instead: consecutive executed steps whose
# button did not change the screen (rollout's changed = aHash hamming > 10). After FREEZE_K such
# steps, OVERRIDE the model and cycle through the inputs it has NOT tried since freezing, one novel
# input per step -- so ANY single input that escapes the state is found within ~len(_FREEZE_ESCAPE)
# steps instead of never. Pattern-agnostic and game-agnostic; needs no perception/knowledge. Healthy
# play never triggers it (a real dialog advance / cursor move changes the screen -> streak resets).
# 0 disables; env VGA_FREEZE_K overrides. A first (most confirm/dialog states need it, and it is
# usually already tried -> the sweep skips to B/back-out and the moves), Start last.
DEFAULT_FREEZE_K = 3
_FREEZE_ESCAPE = ["A", "B", "down", "up", "left", "right", "start"]
# Scene-change threshold (of 256 aHash bits) above which a carried-over subgoal is considered
# STALE and cleared. A carried subgoal surviving a full scene change is the anchor bug (a
# "Select New Game" subgoal persisting into a battle, observed 2026-07-02). Same scene with
# minor motion differs by <~20 bits; a genuine scene change (menu->battle) by >~50.
SUBGOAL_STALE_HAMMING = 48
# Intent-persistence commit (Task 01, 2026-07-04, docs/revitalization/01_intent_persistence.md).
# When the agent HAS relevant learned knowledge for the current context but reflexively falls
# back on the per-frame visual walk-prior (chooses a bare d-pad move on the field), lock a
# FOREGROUND directive derived from that knowledge for this many steps, so the plan survives
# the frames that do not immediately confirm progress. This is the persistence the validated
# experiential loop lacked: a mined fact influenced ONE step then got overridden by the
# walk-prior (day-2 bench-travel-learned). Small on purpose - a directive must not zombie past
# its purpose (the echo/anchor bug burned twice). 0 disables; env VGA_COMMIT_N overrides.
DEFAULT_COMMIT_N = 4
# Hamming distance (of 256 aHash bits) since the commit was set, above which the screen counts
# as MATERIALLY changed (the committed action produced real progress) and the commit is cleared.
# A conservative BACKSTOP only: the primary clear signals are the model reporting mode=='menu'
# (its "open the menu" plan visibly succeeded) and the hard steps_left bound. Set high (near a
# full scene cut) so incremental walking/scrolling on the map does NOT prematurely clear it -
# NOT tuned blind against a metric (day-2 lesson); it is a coarse "the whole screen changed" bar.
COMMIT_CLEAR_HAMMING = 55
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


def _fuzzy_contains(text: str, sig: str, thr: float = 0.8) -> bool:
    """True if `sig` appears in `text` as a substring or as a fuzzy (SequenceMatcher>=thr) span
    of the same word-length. Tolerates the heavy OCR garble on GBA pixel-art (e.g. the info-fee
    charge line "That'll be 300 gil" OCRs as 'that ll be 300 gis' - substring catches a clean
    read, fuzzy catches a noisier one). Advisory, like all OCR here; used only to detect
    OBJECTIVE phase-transition cues (Task 07)."""
    if not text or not sig:
        return False
    if sig in text:
        return True
    tw, sw = text.split(), sig.split()
    n = len(sw)
    if not tw or n == 0:
        return False
    return any(SequenceMatcher(None, " ".join(tw[k:k + n]), sig).ratio() >= thr
               for k in range(0, max(1, len(tw) - n + 1)))


# ---- Task 07: durable, RAM-taught task-state phase machine -----------------------------------
# docs/revitalization/07_task_state_memory.md. A GENERAL mechanism: a durable phase, advanced by
# OBJECTIVE pixels-only cues and fed to the prompt as AUTHORITATIVE task-state, that fixes the
# long-horizon "lost the thread" stall. Diagnostic (2026-07-06): from ffta_pub.state the agent
# accepts the mission then FORGETS it - it misreads the world map as a dialog, mashes A re-trying
# to accept, and wanders into the Cyril Monster Bank; furthest rung stays worldmap_regained (R3)
# for ~77 steps. The free-text scratchpad cannot hold "the accept sub-task is DONE" across the
# scene changes, so the durable phase does.
#
# NORTH-STAR DISCIPLINE: the phase cue is PIXELS-ONLY at inference (an OCR match on the frame).
# The RAM oracle (clan_funds < pub baseline == mission accepted) is the TRAIN-time TEACHER that
# VALIDATED the cue offline (train/ram/validate_phase_cue.py replays saved frames through the same
# OCR path): across 4 real accept-episodes the "that ll be" charge-dialog cue fires exactly at the
# fee charge - one decide-step before the RAM funds tick, one step after the confirm box, zero
# false-fire on the pre-accept pub/briefing screens - and the RAM is NEVER read at inference.
#
# A spec is per-task, selected by the VGA_TASK_PHASE env (also the clean A/B toggle: unset -> the
# whole mechanism is off and the plugin behaves exactly as before). Each phase:
#   {"cues": (substr,...), "next": <phase name|None>, "directive": <authoritative prompt text>}
# A frame's full-frame OCR advancing a cue moves the phase to `next`; while in a phase its
# directive (if any) renders as ReasonContext.task_phase. The initial phase's directive is empty
# (the base goal already states the first sub-task); the directive earns its keep AFTER the
# transition, when the stateless agent would otherwise forget the completed sub-task.
TASK_PHASE_SPECS = {
    "herb": {
        "start": "accept",
        "phases": {
            "accept": {
                # Latch on the INFO-FEE CHARGE dialog. On real frames Tesseract reads it as
                # "that ll be 300 gis" (the game text "That'll be 300 gil..."); the cue is that
                # distinctive charge phrase, NOT the assumed "for the info" wording, which never
                # actually OCRs (validated below). The charge dialog appears ONLY after the player
                # picks Yes on "Accept these conditions?" - i.e. the fee is charged == the mission
                # is committed. Deliberately NOT the confirm box: latching there would render the
                # "already accepted" directive BEFORE the accept commits and could stop the agent
                # from ever pressing Yes. Also deliberately NOT bare "300 gil": that substring is
                # in the pre-accept briefing ("fee 300 gil") and false-fires before commit.
                # VALIDATED offline (train/ram/validate_phase_cue.py, 2026-07-06) on 4 real
                # accept-episodes: "that ll be" fires at the charge dialog in ALL 4, exactly one
                # step after the confirm box and one decide-step before the RAM funds tick
                # (5000->4700), and never on the confirm box or earlier pub screens.
                "cues": ("that ll be", "ll be 300"),
                "next": "travel",
                "directive": "",
            },
            "travel": {
                "cues": (),
                "next": None,
                # Task-state framing, NOT a menu-path walkthrough: it asserts the accept sub-task
                # is DONE (the thing the stateless agent forgets) and names the remaining task
                # (travel to Giza, already in the goal) with only general UI guidance (B backs out
                # of a wrong menu). It does not script the travel menu path.
                "directive": (
                    "You have ALREADY accepted the \"Herb Picking\" mission - the info fee was "
                    "paid, so it is registered. Do NOT accept a mission again, do NOT reopen the "
                    "pub's Missions or Rumors list, and do NOT keep pressing A on the pub owner's "
                    "dialogue. Your task now is to TRAVEL to Giza Plains and start its battle: if "
                    "you are still inside the pub, leave it; once on the world map, move to Giza "
                    "Plains and confirm to travel there, then begin the battle. If a shop, service "
                    "menu, or tutorial pop-up opens, it is NOT part of this mission - back out "
                    "with B."
                ),
            },
        },
    },
}


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
                 commit_n: int = DEFAULT_COMMIT_N,
                 game: str = "", knowledge: Optional[KnowledgeStore] = None,
                 tutor=None):
        self.reasoner = reasoner
        self.goal = goal
        # Staged goals (RESCUE_PLAN, 2026-07-03). A single static goal string describing a
        # multi-phase task re-anchors a stateless per-frame decider onto already-completed
        # phases (observed: post-accept, the agent walked BACK into the pub because the goal
        # text still opened with "open the Missions list"). If the goal contains lines
        # starting with "STAGE:", split it into an ordered stage list; the model only ever
        # sees the CURRENT stage, and advances it by declaring the sentinel sub-goal
        # "STAGE DONE" (rides the existing subgoal round-trip -- no server/protocol change).
        # The pointer is ONE-WAY: a completed stage never comes back, which is exactly the
        # task memory the scratchpad lacks. Plain goals (no STAGE: lines) behave as before.
        self._stages = [ln.split(":", 1)[1].strip()
                        for ln in goal.splitlines() if ln.strip().upper().startswith("STAGE:")]
        self._stage_i = 0
        # aHash of the frame at which the stage pointer last advanced. Guards against burning
        # multiple stages on one static screen (see the over-advance guard in decide()).
        self._last_stage_advance_hash: Optional[int] = None
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
        # Last OBJECTIVE scene descriptor the model emitted (Task 03B, 2026-07-05). Carried
        # forward one step (same one-frame lag as _mode) and folded into the retrieval context
        # for BOTH stores, so a grown store routes facts/skills by scene CONTENT instead of a
        # bare one-word mode ("overworld") that overlaps no fact. That thin context was proven
        # (Task 03) to drop the beneficial fact under budget pressure and starve the Task-01
        # commit. Empty until the first report -> step 0 falls back to f"{mode} {dialog_text}".
        self._last_scene = ""
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
        # Intent-persistence commitment (Task 01). None = no active commitment; else a dict
        # {"text": the foreground directive, "steps_left": int, "set_hash": aHash at set time}.
        # Set when the agent reflexively walks despite holding relevant knowledge; rendered as a
        # FOREGROUND directive (ReasonContext.committed) and self-limited by steps_left + a
        # scene-change / menu-appeared clear. General mechanism - no game-specific content here.
        self._commit_n = int(os.environ.get("VGA_COMMIT_N", commit_n))
        self._commit: Optional[dict] = None
        # Durable task-state phase machine (Task 07, 2026-07-06). Selected by the VGA_TASK_PHASE
        # env, which is also the A/B toggle: a known task name installs its phase spec; empty /
        # unknown -> the mechanism is OFF and task_phase is always "" (the plugin behaves exactly
        # as before). Env-driven to match the existing A/B knobs (VGA_COMMIT_N, VGA_GOAL_ANCHOR_
        # OFF) and because the game-agnostic plugin is built without task context. The cue reads
        # PIXELS ONLY (self._ocr); RAM taught it offline and is never consulted here. See
        # TASK_PHASE_SPECS and _advance_phase().
        _task_phase_name = os.environ.get("VGA_TASK_PHASE", "").strip().lower()
        self._phase_spec = TASK_PHASE_SPECS.get(_task_phase_name)
        self._phase = self._phase_spec["start"] if self._phase_spec else ""
        # Rotating index into _FORCE_DIRS for the mash-break move-forcer.
        self._force_i = 0
        # Anti-freeze escape (2026-07-06). _frozen_streak = consecutive EXECUTED steps whose button
        # left the screen unchanged (fed by note_outcome from rollout's aHash diff); _tried_since_
        # frozen = the set of buttons already emitted during the current frozen streak, so the
        # override always injects a NOVEL input. See decide()/_freeze_escape and _FREEZE_ESCAPE.
        _fk = os.environ.get("VGA_FREEZE_K", "")
        self._freeze_k = int(_fk) if _fk.strip().lstrip("-").isdigit() else DEFAULT_FREEZE_K
        self._frozen_streak = 0
        self._tried_since_frozen: set = set()
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
        # Commit-combo follow-through (2026-07-03). A forced 'start' that OPENS a commit box
        # is wasted if the model then A-mashes the default (No) and closes it -- the box is
        # only up for a step or two, so the rotating force cycle rarely lands 'left' inside
        # that window (measured: naming-fix bench, confirm_reached 8/8 but committed 0/8).
        # So when a forced 'start' visibly changes the screen, follow through with the
        # generic select-then-commit gesture: 'left' (non-default option) then 'A'. Dropped
        # immediately if the start changed nothing (no box appeared).
        self._pending_combo: list[str] = []

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

    def _advance_phase(self, frame: np.ndarray) -> Optional[str]:
        """Task 07: if the current task-state phase has OCR cues and one fires on THIS frame,
        advance the durable phase to its `next`. Returns the matched cue string on an advance,
        else None. PIXELS-ONLY (full-frame OCR); the RAM oracle is never consulted here - it was
        only the offline teacher that validated these cues. No-op when the mechanism is off, when
        OCR is unavailable, or when the phase is terminal / cueless (so once latched to the final
        phase it stops OCR'ing)."""
        if self._phase_spec is None or self._ocr is None:
            return None
        ph = self._phase_spec["phases"].get(self._phase)
        if not ph or not ph.get("cues") or ph.get("next") is None:
            return None
        text = _norm_text(self._ocr.read(frame))
        for sig in ph["cues"]:
            if _fuzzy_contains(text, sig):
                self._phase = ph["next"]
                return sig
        return None

    def note_outcome(self, button: str, reason: str, changed: bool) -> None:
        """Record the outcome of the action just EXECUTED by the loop: which button
        actually went to the game (post-unstick) and whether the screen changed as a
        result. Called by the runner once the next frame is available to diff. This is
        the working memory decide() hands to the reasoner. Recording the executed
        button (not the VLM's pre-unstick choice) keeps the memory truthful: the model
        learns "down did nothing 6x", not "down worked" when an unstick override moved."""
        self._history.append({"button": button, "reason": reason, "changed": bool(changed)})
        # Anti-freeze bookkeeping: the EXECUTED button either moved the screen (reset the streak
        # and the tried-set) or was another no-op (extend the streak, remember we tried it so the
        # escape injects something else next). This is the ground-truth "are we stuck" signal.
        if changed:
            self._frozen_streak = 0
            self._tried_since_frozen.clear()
        else:
            self._frozen_streak += 1
            self._tried_since_frozen.add(button)

    def _freeze_escape_pick(self) -> Optional[str]:
        """Anti-freeze escape selection (2026-07-06). If the screen has been static for at least
        FREEZE_K executed steps, return the next input NOT yet tried during this frozen streak,
        sweeping _FREEZE_ESCAPE in order; once every input has been tried without the screen moving,
        clear the tried-set and restart the sweep (a repeat/combo may be what the state needs).
        Returns None when the mechanism is off (FREEZE_K=0) or we are not yet frozen -- so healthy
        play (which resets the streak every time the screen moves) never sees an override. Pure
        function of the frozen-streak state; unit-testable without the emulator."""
        if not self._freeze_k or self._frozen_streak < self._freeze_k:
            return None
        nxt = next((b for b in _FREEZE_ESCAPE if b not in self._tried_since_frozen), None)
        if nxt is None:  # tried everything this streak; reset the sweep and start over
            self._tried_since_frozen.clear()
            nxt = _FREEZE_ESCAPE[0]
        return nxt

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
        # Intent-persistence commit (Task 01): CLEAR an active commitment before it is rendered
        # this step, once (a) its window expired, (b) its purpose is visibly served - the model
        # reported mode=='menu' LAST step, i.e. the committed "open the menu" plan worked, or
        # (c) the scene changed materially (backstop). self._mode here is the PREVIOUS step's
        # report. The hard steps_left bound guarantees the directive cannot zombie past N steps
        # regardless of the perceptual signals - the tight bound the echo/anchor bug demands.
        if self._commit is not None:
            scene_moved = _hamming(cur_hash, self._commit["set_hash"]) > COMMIT_CLEAR_HAMMING
            if self._commit["steps_left"] <= 0 or self._mode == "menu" or scene_moved:
                self._commit = None
        # Durable task-state phase (Task 07): advance the phase if an OBJECTIVE pixels-only cue
        # fires on this frame (e.g. the info-fee dialog == mission accepted). Pure OCR; RAM is
        # never read here. The resulting directive is fed to the prompt as authoritative task-
        # state and PERSISTS across scene changes / the scratchpad wipe (unlike subgoal/commit).
        phase_cue = self._advance_phase(state.frame)
        task_phase = ""
        if self._phase_spec is not None:
            task_phase = (self._phase_spec["phases"].get(self._phase, {}) or {}).get("directive", "") or ""
        # NOTE (2026-07-04): a mode anti-anchor was tried here (clear self._mode on a big scene
        # change, mirroring the subgoal one) to kill a 46% "false menu belief" seen in the
        # ffta_worldmap.state travel proxy. On the REAL ladder (bench-modeanchor-0704) it
        # REGRESSED at_giza 4/8->0/8 and did NOT move the mechanism (real-ladder false-menu was
        # already ~15%, unchanged). The 46% was a proxy artifact; the mode prior is load-bearing.
        # Reverted. See sessions/2026-07-04-rescue-day2.md.
        # Dialogue OCR-dedup: have we already read the text currently on screen?
        dialog_text, already_read = self._read_dialog(state.frame)
        # Retrieval context for BOTH stores (Task 03B, 2026-07-05): the previous step's mode word
        # + the previous step's OBJECTIVE scene descriptor + this step's dialog OCR. The scene
        # descriptor is the fix for the thin-context cliff (Task 03): a bare "overworld" overlaps
        # no fact, so a store grown past the retrieve limit dropped the beneficial fact under
        # budget pressure and the Task-01 commit went dark. A real scene phrase ("world map with
        # regions and the party caravan...") overlaps the relevant fact and routes it to
        # knowledge[0]. One-frame lag (scene/mode are only known after /act), matching how
        # subgoal/progress already flow; step 0 has no scene yet -> falls back to mode+dialog.
        # A/B kill-switch (clean isolation, honors "change one thing per run"): VGA_SCENE_
        # RETRIEVAL_OFF=1 reproduces the PRE-03B thin context (mode + dialog only) while the
        # scene field is STILL emitted by the server, so the descriptor's effect on RETRIEVAL
        # ROUTING can be measured separately from its effect as a prompt perturbation. Default
        # (unset) = the enriched context. Mirrors VGA_SKILLS_EXACT / VGA_KNOWLEDGE_OFF.
        if os.environ.get("VGA_SCENE_RETRIEVAL_OFF") in ("1", "true", "True"):
            retr_ctx = f"{self._mode} {dialog_text}"
        else:
            retr_ctx = f"{self._mode} {self._last_scene} {dialog_text}"
        # Task 03B GATE-2 post-mortem (2026-07-05): the scene descriptor fixed layer-1 routing
        # (menu opens) but a MERGED multi-task store still dropped the layer-2 completion fact.
        # "Confirm Location Selection: Press A" shares ZERO tokens with a "Party/Area List/System"
        # menu-chrome frame, so retrieval.py's zero-overlap trim discards it ONCE the store
        # outgrows the retrieve limit (the travel-only store only survived because 3 facts <=
        # limit=3 never triggers the trim). Gate 1 offline proof: that KEY fact was present on
        # only 18% of menu frames in the 6-fact combined store vs 100% in the 3-fact travel-only
        # store -- the exact cause of combined_on opening the menu but never completing the Giza
        # selection (it drifted into the Party menu and flailed). FIX: anchor the retrieval
        # context on the ACTIVE GOAL, not just the momentary frame, so a task's procedural facts
        # (which mention the task's OWN vocabulary -- "confirm travel", "location") stay
        # retrievable across all of that task's frames even when the on-screen chrome does not
        # name them. Offline this restores the KEY fact to 100% on travel frames and keeps the pub
        # facts on pub frames; the cross-task "press start" decoy it co-surfaces is never the
        # commit-locked top fact (0/82), only a secondary hint. General/taxonomy-free (the goal is
        # free task text from the VLM/harness). Kill-switch VGA_GOAL_ANCHOR_OFF=1 reproduces the
        # pre-fix frame-only context for clean A/B. See sessions/t03b-retrieval-0705/.
        if os.environ.get("VGA_GOAL_ANCHOR_OFF") not in ("1", "true", "True"):
            retr_ctx = f"{self.goal} {retr_ctx}"
        # Persistent procedural skills, retrieved SEMANTICALLY (Task 02, 2026-07-05) against the
        # SAME free-text context the knowledge store uses. A wrong/"unknown" mode now just nudges
        # ranking instead of cliffing retrieval to []. Mode still flows UNCHANGED into the prompt
        # below (load-bearing prior); the scene descriptor only enriches the retrieval KEY.
        skills = self.skills.retrieve(context=retr_ctx)
        # Declarative tutorial-knowledge for this game, ranked by relevance to the current context
        # so only pertinent facts hit the prompt. Empty when no store / nothing learned yet /
        # VGA_KNOWLEDGE_OFF.
        knowledge = []
        if self.knowledge is not None and self.game:
            knowledge = self.knowledge.retrieve(self.game, context=retr_ctx, limit=3)
        ctx = ReasonContext(
            goal=self._current_goal(), step=self._step,
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
            committed=(self._commit["text"] if self._commit else ""),
            task_phase=task_phase,
        )
        decision = self.reasoner.decide(state.frame, ctx)
        self.last_decision = decision
        # Task 07: log the durable phase every step, and the objective cue on the step it latched,
        # so a trajectory shows WHEN the accept->travel transition was detected from pixels (vs the
        # RAM funds-drop) and that the directive was live thereafter. Cheap; general.
        if decision.meta is not None and self._phase_spec is not None:
            decision.meta["phase"] = self._phase
            if phase_cue is not None:
                decision.meta["phase_latched"] = {"to": self._phase, "cue": phase_cue,
                                                   "step": self._step}
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
        # Commit-combo follow-through: play out 'left' then 'A' right after a forced
        # 'start', UNCONDITIONALLY. First version gated this on last.changed and never
        # fired: the 256-bit aHash misses small overlays (a Yes/No box flips <10 bits,
        # rollout's changed threshold), so the gate starved on exactly the transitions
        # it was built for (bench-naming-combo 2026-07-03, committed 0/8 with the gate).
        # Ungated is safe in context: the scaffold only reaches 'start' in a hopeless
        # loop, and if no box opened, 'left' is a cursor nudge and 'A' is the same press
        # the model was already mashing.
        if self._pending_combo:
            forced = (self._pending_combo.pop(0),
                      "commit-combo: following the forced Start with select-then-commit")
        if forced is None and self._menu_commit_k and looping and len(self._recent_emitted) >= self._menu_commit_k:
            recent = list(self._recent_emitted)
            if all(b in _DIRECTIONS for b in recent):
                forced = ("A", "commit-forced: moved the cursor repeatedly without selecting")
            elif all(b in ("A", "B") for b in recent):
                d = _FORCE_DIRS[self._force_i % len(_FORCE_DIRS)]
                self._force_i += 1
                if d == "start":
                    forced = (d, "commit-forced: A/B made no progress even after moving; "
                                 "trying Start (some screens confirm only with Start)")
                    # If this start opens a box, follow through with select-then-commit
                    # on the next steps instead of returning control to the mash.
                    self._pending_combo = ["left", "A"]
                else:
                    forced = (d, "move-forced: pressed A/B in place without progress; move the cursor")
        if forced is not None:
            action = action_from_choice(forced[0], note=forced[1])
            if decision.meta is not None:
                decision.meta["commit_scaffold"] = forced[0]
            emitted = forced[0]
        else:
            action = decision.action
            emitted = chosen
        # Anti-freeze escape (2026-07-06) -- HIGHEST-priority override. When the screen has not
        # moved for FREEZE_K executed steps, neither the model's choice nor the menu-commit forcer
        # is working, so stop trusting them: inject the first input NOT yet tried during this frozen
        # streak (novel input per step), sweeping the whole set so any escape is found fast. Keyed
        # purely on ground truth (screen static), not on button pattern, so the "mostly-A-with-a-
        # few-moves" flail that evades the homogeneous menu-commit forcer is caught here. If the
        # sweep exhausts every input without the screen moving (a state no single press escapes),
        # clear the tried-set and sweep again (a repeat/combo may be needed) -- still >= the old
        # behavior of mashing one dead button forever. Placed after the forcer so it wins.
        esc = self._freeze_escape_pick()
        if esc is not None:
            action = action_from_choice(
                esc, note=f"anti-freeze: screen static {self._frozen_streak} steps, cycling inputs")
            emitted = esc
            if decision.meta is not None:
                decision.meta["anti_freeze"] = {"button": esc, "streak": self._frozen_streak}
        self._recent_emitted.append(emitted)
        # Intent-persistence commit (Task 01): SET or DECREMENT. Fire ONLY when the agent HAS
        # relevant learned knowledge for this context yet the model reflexively CHOSE a bare
        # d-pad move while not in a menu/dialog/battle - i.e. it fell back on the visual
        # walk-prior instead of acting on what it learned (the exact day-2 revert). Lock the
        # top retrieved fact as a foreground directive for N steps; from next step it renders
        # via ReasonContext.committed, weighted above the reflex. `chosen` is the MODEL's own
        # choice (not the scaffold override), and self._mode is still the PREVIOUS step's
        # report here (updated below). No new commit while one is active - it just decrements.
        # GENERAL: nothing game-specific - the only game-specific input is knowledge[0], a fact
        # from the general KnowledgeStore / experiential-mining channel.
        if (self._commit_n and self._commit is None and knowledge
                and chosen in _DIRECTIONS
                and self._mode not in ("menu", "dialog", "battle", "shop", "title", "cutscene")):
            self._commit = {"text": str(knowledge[0]), "steps_left": self._commit_n,
                            "set_hash": cur_hash}
            if decision.meta is not None:
                decision.meta["commit_set"] = self._commit["text"]
        elif self._commit is not None:
            self._commit["steps_left"] -= 1
        if decision.meta is not None and self._commit is not None:
            decision.meta["commit_active"] = self._commit["steps_left"]
        # Log the knowledge facts retrieved this step so a trajectory shows WHICH learned
        # facts were in front of the model when it acted (visibility into "did it pull what it
        # learned"). Cheap; general (helps headless diagnosis too), added 2026-07-04.
        if decision.meta is not None and knowledge:
            decision.meta["knowledge_retrieved"] = list(knowledge)
        # Task 03B diagnosis: record the exact retrieval context that selected this step's
        # facts/skills, so a trajectory shows whether the scene descriptor actually enriched the
        # key (vs. a bare "overworld") and thereby re-armed the commit. Cheap; general.
        if decision.meta is not None:
            decision.meta["retrieval_context"] = retr_ctx
        # Carry the model's updated scratchpad forward. Keep the prior value if the model
        # returned an empty string, so a momentary omission doesn't wipe the plan.
        new_subgoal = (decision.meta or {}).get("subgoal")
        new_progress = (decision.meta or {}).get("progress")
        new_mode = (decision.meta or {}).get("mode")
        # Objective scene descriptor for the NEXT step's retrieval context (Task 03B). Keep the
        # prior value when the model omits it (a momentary parse gap should not blank the routing
        # key), matching how subgoal/progress/mode carry forward. Host-side retrieval key only -
        # never rendered back into the prompt as a fact, so a stale value is low-risk (it just
        # routes retrieval a frame late; the commit already clears on a material scene change).
        new_scene = (decision.meta or {}).get("scene")
        # Staged-goal advance: the model declared the current stage complete. Consume the
        # sentinel (it must not persist as a real sub-goal), advance the one-way pointer, and
        # clear the scratchpad so the next stage starts from what the screen shows.
        if (self._stages and new_subgoal
                and str(new_subgoal).strip().upper().startswith("STAGE DONE")):
            # Over-advance guard (2026-07-04 day3). A stage is a SCREEN-STATE milestone, so
            # two STAGE DONE declarations on the SAME unchanged screen cannot both be real:
            # the model regained the world map and, on that one static frame, declared BOTH
            # "return to world map" AND "open the menu/Area List" complete (steps 49-50,
            # bench-herb-stages3) -- burning the open-menu stage without ever opening the menu,
            # then hallucinating an Area List to match the (unrecoverable, one-way) stage text.
            # Only honor an advance if the screen has materially changed since the last one;
            # a repeat STAGE DONE on the same frame is a no-op (the pointer waits for real
            # progress). First advance (hash None) is always allowed.
            same_screen = (self._last_stage_advance_hash is not None
                           and _hamming(cur_hash, self._last_stage_advance_hash) <= self._revisit_tol)
            if same_screen:
                if decision.meta is not None:
                    decision.meta["stage_advance_blocked"] = self._stage_i
            else:
                if self._stage_i < len(self._stages) - 1:
                    self._stage_i += 1
                self._last_stage_advance_hash = cur_hash
                if decision.meta is not None:
                    decision.meta["stage_advanced_to"] = self._stage_i
                self._subgoal = ""
                self._progress = ""
                self._subgoal_hash = None
            new_subgoal = None
        if new_subgoal:
            self._subgoal = str(new_subgoal)
            # Stamp the scene this subgoal was set on, so the anti-anchor check above can tell
            # when the screen has moved on and the subgoal is stale.
            self._subgoal_hash = cur_hash
        if new_progress:
            self._progress = str(new_progress)
        if new_mode:
            self._mode = str(new_mode).strip().lower()
        if new_scene:
            self._last_scene = str(new_scene).strip()
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

    def _current_goal(self) -> str:
        """The goal string the model sees this step: the whole goal when unstaged, else the
        current stage plus the stage-done protocol. Completed stages are summarized in one
        clause (context without re-anchoring: the model knows they are DONE)."""
        if not self._stages:
            return self.goal
        i = self._stage_i
        parts = []
        if i > 0:
            parts.append(f"Stages already completed (do NOT redo them): "
                         f"{'; '.join(self._stages[:i])}.")
        parts.append(f"CURRENT STAGE ({i + 1} of {len(self._stages)}): {self._stages[i]}")
        if i < len(self._stages) - 1:
            parts.append('If the current stage is ALREADY COMPLETE on this screen, set your '
                         'sub-goal to exactly "STAGE DONE".')
        return " ".join(parts)

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
        self._stage_i = 0
        self._last_stage_advance_hash = None
        self._pending_combo = []
        self._history.clear()
        self._recent_hashes.clear()
        self._recent_texts.clear()
        self._subgoal = ""
        self._progress = ""
        self._subgoal_hash = None
        self._commit = None
        # Task 07: reset the durable phase to its start each episode (the phase spec itself, like
        # skills/knowledge, is not rebuilt - only the live phase pointer is per-episode state).
        self._phase = self._phase_spec["start"] if self._phase_spec else ""
        self._mode = ""
        self._last_scene = ""
        self._force_i = 0
        self._frozen_streak = 0
        self._tried_since_frozen.clear()
        self._recent_emitted.clear()
        # Drop any in-progress tutorial accumulation (a tutorial spanning an episode boundary
        # is rare and the server may be unavailable at reset). NOTE: neither self.skills NOR
        # self.knowledge is reset - both are learn-once-keep-forever across episodes.
        self._tut_pages = []
        self._in_tutorial = False
        self._tut_last_hash = 0
        self.last_decision = None
        self.reasoner.reset()
