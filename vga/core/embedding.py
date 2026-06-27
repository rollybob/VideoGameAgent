"""
Game-agnostic frame embedding for unsupervised state discovery.

A frame becomes a fixed-length vector combining:
  - APPEARANCE: coarse grayscale layout, an HSV color histogram, and brightness/
    edge scalars. Captures "what the screen looks like."
  - MOTION SIGNATURE: a coarse grid of smoothed inter-frame change. Captures
    "where things move over time" -- a cheap proxy for game dynamics that
    distinguishes behaviorally-different modes that look alike (a dialogue box
    overlaid on the overworld looks like the overworld, but only its bottom strip
    changes frame-to-frame; the overworld's whole frame scrolls when you walk).

Each block is L2-normalized and weighted so no single block dominates the distance
purely by having more dimensions. Nothing here knows about any specific game.
"""

from __future__ import annotations

import cv2
import numpy as np

# Per-block weights. Motion and the compact scalars are behaviorally informative,
# so they are not drowned out by the higher-dimensional layout block.
_W_LAYOUT = 1.0
_W_HIST = 1.5
_W_SCALARS = 2.0
_W_MOTION = 2.5

_MOTION_GRID = 4          # 4x4 = 16 motion cells
_MOTION_GRAY = 48         # resolution the motion diff is computed at
_LAYOUT = 16              # 16x16 = 256 layout dims


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


class MotionTracker:
    """Exponential moving average of per-cell inter-frame change."""

    def __init__(self, grid: int = _MOTION_GRID, alpha: float = 0.4) -> None:
        self.grid = grid
        self.alpha = alpha
        self.prev: np.ndarray | None = None
        self.ema = np.zeros(grid * grid, dtype=np.float32)

    def update(self, gray_small: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = gray_small
            return self.ema
        diff = np.abs(gray_small.astype(np.float32) - self.prev.astype(np.float32))
        self.prev = gray_small
        g = self.grid
        h, w = diff.shape
        cells = np.empty(g * g, dtype=np.float32)
        for i in range(g):
            for j in range(g):
                cell = diff[i * h // g:(i + 1) * h // g, j * w // g:(j + 1) * w // g]
                cells[i * g + j] = cell.mean()
        cells /= 255.0
        self.ema = self.alpha * cells + (1.0 - self.alpha) * self.ema
        return self.ema

    def reset(self) -> None:
        self.prev = None
        self.ema[:] = 0.0


def appearance_blocks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    layout = cv2.resize(gray, (_LAYOUT, _LAYOUT),
                        interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 3], [0, 180, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    g = gray.astype(np.float32)
    black = float((g < 40).mean())
    white = float((g > 200).mean())
    edges = cv2.Canny(gray, 50, 150)
    edge = float((edges > 0).mean())
    scalars = np.array([black, white, edge], dtype=np.float32)
    return layout, hist, scalars


class Embedder:
    """Produces a weighted, block-normalized embedding per frame, carrying its own
    MotionTracker so motion is accumulated across the frame stream."""

    def __init__(self) -> None:
        self.motion = MotionTracker()

    def reset(self) -> None:
        self.motion.reset()

    def embed(self, frame: np.ndarray) -> np.ndarray:
        layout, hist, scalars = appearance_blocks(frame)
        gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                (_MOTION_GRAY, _MOTION_GRAY),
                                interpolation=cv2.INTER_AREA)
        motion = self.motion.update(gray_small)
        return np.concatenate([
            _norm(layout) * _W_LAYOUT,
            _norm(hist) * _W_HIST,
            _norm(scalars) * _W_SCALARS,
            _norm(motion) * _W_MOTION,
        ]).astype(np.float32)
