"""
Unsupervised game-state discovery.

Watches a stream of frame embeddings and assigns each to a discovered "mode" with
no game-specific knowledge and no labels. Modes are integer IDs, not words: the
agent learns "this game has a few recurring situations and I can tell them apart,"
not "this is a battle." A human (or a plugin) can attach names/behaviors to
discovered modes afterward, for one game.

Method: online leader clustering. A frame near an existing centroid joins it (and
nudges it); a frame far from all of them starts a new mode. The "near/far"
threshold is auto-calibrated from a warmup window instead of a magic constant, so
it adapts to whatever the embedding scale turns out to be for a given game. Mode
assignments are stabilized with the same multi-frame confirmation used elsewhere,
so animation jitter does not cause mode flicker.

Centroids persist to disk, so the agent remembers a game's modes across runs -- the
seed of a per-game plugin that an agent could grow on its own.

Honest limits: this clusters by appearance + a motion proxy, which approximates but
does not equal the true (dynamics-defined) notion of a game state. It will
sometimes over-split (same mode, different-looking areas) or merge (distinct modes
that look and move alike). It is a foundation to iterate on, not a finished oracle.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .stability import StreakConfirmer


@dataclass
class _Mode:
    centroid: np.ndarray
    count: int = 0


@dataclass
class DiscoveryConfig:
    warmup: int = 80           # frames buffered before clustering starts
    max_modes: int = 12        # cap on discovered modes
    threshold_pct: float = 35  # percentile of warmup pairwise distances -> join threshold
    confirm: int = 3           # frames a raw mode must hold before it is committed
    ema: float = 0.05          # centroid update rate (running mean-ish)


class ModeDiscoverer:
    def __init__(self, config: DiscoveryConfig | None = None, logger=print) -> None:
        self.cfg = config or DiscoveryConfig()
        self.logger = logger
        self.modes: list[_Mode] = []
        self.threshold: float | None = None
        self._warmup_buf: list[np.ndarray] = []
        self._confirmer = StreakConfirmer()
        self.committed: int = -1
        # Diagnostics for calibration.
        self.last_distance: float = 0.0
        self.warmup_dist_stats: dict = {}

    # ---------- core update ----------
    def observe(self, embedding: np.ndarray) -> int:
        """Feed one frame embedding; return the committed mode id (-1 during warmup)."""
        if self.threshold is None:
            self._warmup_buf.append(embedding)
            if len(self._warmup_buf) >= self.cfg.warmup:
                self._finish_warmup()
            return -1

        raw = self._assign(embedding)
        self._confirmer.observe(raw)
        if self._confirmer.streak >= self.cfg.confirm:
            self.committed = raw
        return self.committed

    def _assign(self, x: np.ndarray) -> int:
        """Nearest centroid within threshold, else a new mode. Updates the centroid."""
        best, best_d = -1, float("inf")
        for i, m in enumerate(self.modes):
            d = float(np.linalg.norm(x - m.centroid))
            if d < best_d:
                best, best_d = i, d
        self.last_distance = best_d
        if best == -1 or best_d > self.threshold:
            if len(self.modes) < self.cfg.max_modes:
                self.modes.append(_Mode(centroid=x.copy(), count=1))
                idx = len(self.modes) - 1
                self.logger(f"[modes] discovered mode {idx} "
                            f"(now {len(self.modes)} modes, dist {best_d:.3f})")
                return idx
            # Cap reached: fall back to nearest.
        m = self.modes[best]
        a = self.cfg.ema
        m.centroid = (1 - a) * m.centroid + a * x
        m.count += 1
        return best

    def _finish_warmup(self) -> None:
        buf = np.stack(self._warmup_buf)
        # Pairwise distances on a capped sample (warmup is small anyway).
        n = len(buf)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(float(np.linalg.norm(buf[i] - buf[j])))
        dists = np.array(dists) if dists else np.array([1.0])
        self.threshold = float(np.percentile(dists, self.cfg.threshold_pct))
        self.warmup_dist_stats = {
            "min": float(dists.min()), "p25": float(np.percentile(dists, 25)),
            "median": float(np.median(dists)), "p75": float(np.percentile(dists, 75)),
            "max": float(dists.max()), "threshold": self.threshold,
        }
        self.logger(f"[modes] warmup done ({n} frames). pairwise dist "
                    f"min={dists.min():.3f} med={np.median(dists):.3f} "
                    f"max={dists.max():.3f} -> join threshold={self.threshold:.3f}")
        # Seed modes by leader-clustering the warmup buffer.
        if self.modes:
            return
        for x in buf:
            if not self.modes:
                self.modes.append(_Mode(centroid=x.copy(), count=1))
                continue
            d = min(float(np.linalg.norm(x - m.centroid)) for m in self.modes)
            nearest = int(np.argmin([np.linalg.norm(x - m.centroid) for m in self.modes]))
            if d > self.threshold and len(self.modes) < self.cfg.max_modes:
                self.modes.append(_Mode(centroid=x.copy(), count=1))
            else:
                m = self.modes[nearest]
                m.centroid = 0.9 * m.centroid + 0.1 * x
                m.count += 1
        self.logger(f"[modes] seeded {len(self.modes)} modes from warmup.")

    def reset_stream(self) -> None:
        """Reset per-stream stabilization (keep learned modes)."""
        self._confirmer.reset()
        self.committed = -1

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        data = {
            "threshold": self.threshold,
            "config": self.cfg.__dict__,
            "warmup_dist_stats": self.warmup_dist_stats,
            "modes": [{"centroid": m.centroid.tolist(), "count": m.count}
                      for m in self.modes],
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        self.logger(f"[modes] saved {len(self.modes)} modes -> {path}")

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.threshold = data["threshold"]
        self.modes = [_Mode(centroid=np.array(m["centroid"], dtype=np.float32),
                            count=m["count"]) for m in data["modes"]]
        self.warmup_dist_stats = data.get("warmup_dist_stats", {})
        self.logger(f"[modes] loaded {len(self.modes)} modes <- {path}")
        return True

    def summary(self) -> str:
        parts = [f"threshold={self.threshold:.3f}" if self.threshold else "warming up"]
        for i, m in enumerate(self.modes):
            parts.append(f"mode{i}:{m.count}")
        return " | ".join(parts)
