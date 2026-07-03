"""
Game-agnostic scene-text detection (EAST via OpenCV DNN).

Finds *where* text is on an arbitrary game frame -- no per-game tuning, no torch
(cv2 only, so it runs in the host VGA loop). Verified 2026-06-27 to generalize
across Pokemon and Fire Emblem. Detected word-boxes are grouped into text LINES,
which an OCR recognizer then reads. This is the "where is text" stage; reading it
(font-specific) is a separate recognizer -- detection generalizes, recognition does
not (see docs/STATE_OF_VGA.md F4/F18).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "..", "train", "ocr", "models",
    "frozen_east_text_detection.pb")
_LAYERS = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]


class TextDetector:
    """EAST text detector. detect()->word boxes; group_lines()->line boxes. Degrades
    gracefully to a no-op (enabled=False) if the model is missing or fails to load,
    so perception never crashes when the detector is unavailable."""

    def __init__(self, model_path=None, width=640, score=0.5, nms=0.4, logger=print):
        self.width, self.score, self.nms, self.logger = width, score, nms, logger
        self.net = None
        path = os.path.abspath(model_path or _DEFAULT_MODEL)
        if os.path.exists(path):
            try:
                self.net = cv2.dnn.readNet(path)
                logger(f"[textdetect] EAST loaded: {path}")
            except Exception as e:  # noqa: BLE001
                logger(f"[textdetect] failed to load EAST ({e}); detector disabled")
        else:
            logger(f"[textdetect] EAST model not found at {path}; detector disabled")
        self.enabled = self.net is not None

    def _decode(self, scores, geometry):
        nrows, ncols = scores.shape[2:4]
        rects, confs = [], []
        for y in range(nrows):
            sc = scores[0, 0, y]
            d0, d1, d2, d3 = (geometry[0, 0, y], geometry[0, 1, y],
                              geometry[0, 2, y], geometry[0, 3, y])
            ang = geometry[0, 4, y]
            for x in range(ncols):
                if sc[x] < self.score:
                    continue
                offx, offy = x * 4.0, y * 4.0
                cos, sin = np.cos(ang[x]), np.sin(ang[x])
                h = d0[x] + d2[x]
                w = d1[x] + d3[x]
                endx = int(offx + cos * d1[x] + sin * d2[x])
                endy = int(offy - sin * d1[x] + cos * d2[x])
                rects.append([endx - int(w), endy - int(h), int(w), int(h)])
                confs.append(float(sc[x]))
        return rects, confs

    def detect(self, frame):
        """Return text-region boxes (x, y, w, h) in frame coordinates. [] if disabled."""
        if not self.enabled or frame is None or frame.size == 0:
            return []
        h0, w0 = frame.shape[:2]
        neww = max(32, int(round(self.width / 32)) * 32)
        newh = max(32, int(round(h0 * neww / w0 / 32)) * 32)
        rw, rh = w0 / float(neww), h0 / float(newh)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (neww, newh),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        scores, geom = self.net.forward(_LAYERS)
        rects, confs = self._decode(scores, geom)
        boxes = []
        if rects:
            keep = cv2.dnn.NMSBoxes(rects, confs, self.score, self.nms)
            for i in np.array(keep).flatten():
                x, y, w, h = rects[i]
                boxes.append((int(x * rw), int(y * rh), int(w * rw), int(h * rh)))
        return boxes

    @staticmethod
    def group_lines(boxes, y_tol=0.6):
        """Merge word-boxes into text lines by vertical-center proximity. Returns
        line boxes (x, y, w, h) ordered top-to-bottom."""
        if not boxes:
            return []
        lines = []  # each: [x0, y0, x1, y1]
        for (x, y, w, h) in sorted(boxes, key=lambda b: b[1]):
            cy = y + h / 2.0
            placed = False
            for ln in lines:
                lcy = (ln[1] + ln[3]) / 2.0
                lh = ln[3] - ln[1]
                if abs(cy - lcy) <= y_tol * max(h, lh):
                    ln[0], ln[1] = min(ln[0], x), min(ln[1], y)
                    ln[2], ln[3] = max(ln[2], x + w), max(ln[3], y + h)
                    placed = True
                    break
            if not placed:
                lines.append([x, y, x + w, y + h])
        lines.sort(key=lambda l: l[1])
        return [(l[0], l[1], l[2] - l[0], l[3] - l[1]) for l in lines]
