"""
OCR for the Pokemon plugin: locate tesseract robustly, read a region, and
validate the result before anyone trusts it.

The old code hardcoded a tesseract path that did not exist on this machine, which
made every OCR call raise. Here we auto-locate it and degrade gracefully: if
tesseract is genuinely absent, OCR is disabled and read() returns "" instead of
crashing the loop. OCR is treated as advisory only -- a misread must never drive a
state transition (see strategy.py).
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract
    from PIL import Image
    _HAVE_PYTESSERACT = True
except Exception:
    _HAVE_PYTESSERACT = False


# Common Windows install locations to check after PATH.
_CANDIDATE_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "agent", "tools",
                 "tesseract", "tesseract.exe"),
)


def _locate_tesseract() -> Optional[str]:
    found = shutil.which("tesseract")
    if found:
        return found
    for p in _CANDIDATE_PATHS:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
    return None


class Ocr:
    def __init__(self, logger=print) -> None:
        self.enabled = False
        self.logger = logger
        if not _HAVE_PYTESSERACT:
            logger("[ocr] pytesseract/PIL not importable; OCR disabled.")
            return
        path = _locate_tesseract()
        if path is None:
            logger("[ocr] tesseract executable not found (install Tesseract-OCR "
                   "or add it to PATH); OCR disabled.")
            return
        pytesseract.pytesseract.tesseract_cmd = path
        self.enabled = True
        logger(f"[ocr] using tesseract at: {path}")

    def read(self, region_bgr: np.ndarray, min_chars: int = 2) -> str:
        """Read text from a BGR image region. Returns "" on any failure or if the
        result fails validation. Never raises."""
        if not self.enabled or region_bgr is None or region_bgr.size == 0:
            return ""
        try:
            gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            raw = pytesseract.image_to_string(Image.fromarray(thresh),
                                              config="--oem 3 --psm 6")
        except Exception as e:
            self.logger(f"[ocr] read failed: {e}")
            return ""
        return self._validate(raw, min_chars)

    def read_lines_in(self, frame: np.ndarray, detector, pad: int = 2,
                      min_chars: int = 2) -> list:
        """Region-gated OCR: use `detector` (core.textdetect.TextDetector) to find
        text LINES, then OCR each line crop in single-line mode. Returns the
        validated line strings top-to-bottom. This is the fix for whole-strip OCR:
        the recognizer only ever sees one tight line at a time (the earlier failure
        was feeding it merged multi-line blocks). Empty list if OCR/detector off."""
        if not self.enabled or detector is None or not getattr(detector, "enabled", False):
            return []
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        lines = []
        for (bx, by, bw, bh) in detector.group_lines(detector.detect(frame)):
            x0, y0 = max(0, bx - pad), max(0, by - pad)
            x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            txt = self._read_line(crop, min_chars)
            if txt:
                lines.append(txt)
        return lines

    def _read_line(self, region_bgr: np.ndarray, min_chars: int) -> str:
        """OCR a single tight text line. Upscales small lines, Otsu-binarizes, and
        normalizes to dark-text-on-light (so it works on both polarities, e.g. FE's
        white-on-dark). --psm 7 = treat as one line. Never raises."""
        try:
            gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] < 24:                      # upscale tiny lines for Tesseract
                scale = 24.0 / gray.shape[0]
                gray = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), 24),
                                  interpolation=cv2.INTER_CUBIC)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if float(th.mean()) < 127:                  # ensure dark text on light bg
                th = 255 - th
            raw = pytesseract.image_to_string(Image.fromarray(th), config="--oem 3 --psm 7")
        except Exception as e:  # noqa: BLE001
            self.logger(f"[ocr] line read failed: {e}")
            return ""
        return self._validate(raw, min_chars)

    @staticmethod
    def _validate(raw: str, min_chars: int) -> str:
        """Expected-format check: keep only plausible dialogue text. Reject noise."""
        if not raw:
            return ""
        # Collapse whitespace, drop control chars.
        text = " ".join(raw.split())
        # Require a minimum number of alphabetic characters; OCR noise tends to be
        # punctuation/symbols.
        alpha = sum(c.isalpha() for c in text)
        if alpha < min_chars:
            return ""
        # Reject strings that are mostly non-alphanumeric (garbage reads).
        usable = sum(c.isalnum() or c.isspace() for c in text)
        if usable < 0.6 * len(text):
            return ""
        return text
