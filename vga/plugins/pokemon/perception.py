"""
Pokemon perception: frame -> GameState.

This is Pokemon-specific knowledge (it is in the Pokemon plugin), but it still
only *reads* the frame -- it makes no decisions. It reports instantaneous evidence
(is there battle UI? does this look like the overworld? how dark is the screen?)
and leaves all temporal commitment and action choice to strategy.py.

The battle read is deliberately stricter than the old detector, which treated any
wide green block in the top of the screen as a "health bar" (so overworld grass
read as a battle). Here the battle signal is a strict HP-bar shape ALONE: a very
thin, very wide, saturated bar high in the frame (see _has_hp_bar) -- calibrated so
the overworld never produces one, which is why it can stand on its own. A battle
menu box IS also computed (_has_menu_box) but only as a corroborating/diagnostic
signal, NOT a requirement: in FRLG the command menu is shown only on the player's
input turn, so requiring it would miss enemy-attack/animation/battle-text frames
that have an HP bar but no menu. Single-signal dropouts during animations are
absorbed by strategy.py's multi-frame confirmation + hysteresis, not by loosening
this read. (Reconciles a stale doc/code mismatch; see ledger F6.)
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from vga.core.contract import GameState
from vga.core.textdetect import TextDetector
from .ocr import Ocr


def _pct(mask: np.ndarray, total: int) -> float:
    return float(np.count_nonzero(mask)) / float(total) if total else 0.0


def _has_hp_bar(region_bgr: np.ndarray) -> bool:
    """An HP bar is a very thin, very wide, saturated green/yellow/red rectangle
    high in the frame. Thresholds calibrated against real FRLG battle frames: the
    HP bar runs at aspect ~16 and ~2% of frame height, while the closest overworld
    look-alikes (grass/path strips) top out near aspect 10 and are several times
    thicker. Requiring BOTH a high aspect and extreme thinness cleanly separates a
    real HP bar from any flat color field -- so the HP bar alone is a reliable,
    battle-specific signal (the overworld never shows one)."""
    if region_bgr.size == 0:
        return False
    h, w = region_bgr.shape[:2]
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([90, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    red1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
    for mask in (green, yellow, cv2.bitwise_or(red1, red2)):
        if np.count_nonzero(mask) < 20:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch <= 0:
                continue
            aspect = cw / ch
            if aspect > 13.0 and cw > 0.08 * w and ch < 0.06 * h:
                return True
    return False


def _has_menu_box(region_bgr: np.ndarray) -> bool:
    """The battle command box is a large, bright, rectangular panel with text
    inside -> high local edge density inside a big bright region."""
    if region_bgr.size == 0:
        return False
    h, w = region_bgr.shape[:2]
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    bright = gray > 170
    bright_pct = _pct(bright, h * w)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = _pct(edges > 0, h * w)
    # A menu box fills a good chunk of the bottom strip and carries text edges.
    return bright_pct > 0.20 and edge_density > 0.06


def _fp_diff(a, b) -> float:
    """Mean absolute difference (0..1) between two small grayscale fingerprints --
    a cheap 'how much did the screen change' metric for gating OCR."""
    if a is None or b is None:
        return 1.0
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))) / 255.0


class PokemonPerception:
    def __init__(self, logger=print, enable_ocr: bool = True,
                 use_detector: bool = True, ocr_interval: int = 6,
                 ocr_settle: int = 2, ocr_change: float = 0.005,
                 ocr_stable_eps: float = 0.0015) -> None:
        self.enable_ocr = enable_ocr
        self.ocr = Ocr(logger=logger) if enable_ocr else None
        # Game-agnostic text detector (EAST), region-gating OCR to detected lines.
        self.detector = (TextDetector(logger=logger)
                         if (enable_ocr and use_detector) else None)
        # OCR gating (EAST is CPU-heavy): only run the detector+OCR when the screen
        # has CHANGED since the last read and then held STABLE for `ocr_settle`
        # frames. Skips idle re-reads of an unchanged screen AND half-drawn
        # typewriter frames; `ocr_change` is the per-frame change threshold (mean
        # abs fingerprint diff, 0..1 -- small enough to ignore a blinking cursor);
        # `ocr_interval` is a floor on how often the expensive read can fire.
        self.ocr_interval = max(1, ocr_interval)
        self.ocr_settle = max(1, ocr_settle)
        self.ocr_change = ocr_change
        self.ocr_stable_eps = ocr_stable_eps
        self._ocr_cache: list = []
        self._ocr_last = -10 ** 9
        self._ocr_read_fp = None   # bottom fingerprint at the last OCR (cumulative-change ref)
        self._ocr_prev_fp = None   # previous-frame bottom fingerprint (frame-to-frame stability)
        self._ocr_stable = 0       # consecutive frame-to-frame-quiet frames
        self._ocr_dirty = False    # content changed since the last successful OCR
        self.logger = logger

    def matches(self, frame: np.ndarray) -> float:
        """How confidently this plugin handles the on-screen game.

        v1 placeholder: we only sanity-check that there is a real frame. True
        game-family fingerprinting (title-screen / UI templates that recognize the
        Pokemon FRLG family specifically) is a TODO -- this is the seam where, in
        future, an unrecognized game would route to a different/new plugin. For now
        we return a confident-but-not-certain score so this single plugin is
        selected while leaving headroom for a real matcher.
        """
        if frame is None or frame.size == 0:
            return 0.0
        return 0.75

    def perceive(self, frame: np.ndarray, frame_index: int, timestamp: float) -> GameState:
        h, w = frame.shape[:2]
        total = h * w

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        black_pct = _pct(gray < 40, total)
        white_pct = _pct(gray > 200, total)
        green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_pct = _pct(green > 0, total)

        top = frame[0:int(h * 0.45), :]
        bottom = frame[int(h * 0.65):h, :]
        hp_bar = _has_hp_bar(top)
        menu_box = _has_menu_box(bottom)
        # The HP bar alone is battle-specific (calibrated to not fire on overworld);
        # the FSM's multi-frame confirmation + hysteresis handle animation frames
        # where the bar momentarily drops out. menu_box is kept for diagnostics only.
        battle_evidence = hp_bar

        # Overworld: ordinary gameplay -- not a battle, not a fade/menu blackout, not
        # a near-white flash, with some scene color/texture present.
        overworld_evidence = (
            not battle_evidence
            and black_pct < 0.25
            and white_pct < 0.40
            and green_pct > 0.08
        )

        if battle_evidence:
            scene_raw = "battle"
        elif overworld_evidence:
            scene_raw = "overworld"
        else:
            # Fades, dialogue blackouts, menus, transitions -- ambiguous on purpose.
            scene_raw = "other"

        # Small grayscale fingerprint for cheap frame-to-frame similarity (used by
        # the battle policy to tell whether an action changed anything).
        fingerprint = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

        state = GameState(
            frame_index=frame_index,
            timestamp=timestamp,
            width=w,
            height=h,
            frame=frame,
            regions={
                "black_pct": black_pct,
                "white_pct": white_pct,
                "green_pct": green_pct,
                "hp_bar": hp_bar,
                "menu_box": menu_box,
                "battle_evidence": battle_evidence,
                "overworld_evidence": overworld_evidence,
                "scene_raw": scene_raw,
                "fingerprint": fingerprint,
            },
        )

        # Advisory OCR. Never blocks; validated text or empty. Strategy may use this
        # for flavor/logging but not for transitions. Preferred path: the EAST
        # detector region-gates OCR to detected text lines (throttled + cached, since
        # EAST is CPU-heavy). Falls back to the whole-bottom-strip read if the
        # detector is unavailable.
        if self.enable_ocr and self.ocr is not None:
            if self.detector is not None and self.detector.enabled:
                # Change-gated, settle-based OCR. The change signal is the dialogue
                # (bottom) region at higher resolution -- the whole-frame 32x32
                # fingerprint is too coarse to register a new sentence in the box
                # (text is a small fraction of the frame; measured A->B was 0.003 at
                # 32x32 vs 0.010 on the bottom 128x64). Re-run the expensive
                # detector+OCR only when this region changed and has since held
                # stable, so idle reading of an unchanged screen does NOT re-fire OCR
                # and we read fully-drawn text rather than mid-typewriter frames.
                # (Bottom-region signal fits this Pokemon plugin; a general backend
                # would key the diff on the detected text regions instead.)
                ocr_fp = cv2.resize(cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY),
                                    (128, 64), interpolation=cv2.INTER_AREA)
                # "dirty" = content changed since we last READ -- measured CUMULATIVELY
                # vs the last-read frame, so slow typewriter accumulation is caught
                # even when each frame's increment is sub-threshold. "stable" = quiet
                # frame-to-frame (animation has stopped). Read once dirty AND stable,
                # so we read fully-drawn text once and don't re-read an idle screen.
                if (self._ocr_read_fp is None
                        or _fp_diff(ocr_fp, self._ocr_read_fp) > self.ocr_change):
                    self._ocr_dirty = True
                if (self._ocr_prev_fp is not None
                        and _fp_diff(ocr_fp, self._ocr_prev_fp) <= self.ocr_stable_eps):
                    self._ocr_stable += 1
                else:
                    self._ocr_stable = 0
                self._ocr_prev_fp = ocr_fp
                if (self._ocr_dirty and self._ocr_stable >= self.ocr_settle
                        and frame_index - self._ocr_last >= self.ocr_interval):
                    self._ocr_cache = self.ocr.read_lines_in(frame, self.detector)
                    self._ocr_last = frame_index
                    self._ocr_read_fp = ocr_fp
                    self._ocr_dirty = False
                    self._ocr_stable = 0
                if self._ocr_cache:
                    state.text["lines"] = list(self._ocr_cache)
                    state.text["dialogue"] = " ".join(self._ocr_cache)
            else:
                dialogue = self.ocr.read(bottom)
                if dialogue:
                    state.text["dialogue"] = dialogue

        return state
