"""
Host-side CRNN text recognizer over ONNX (onnxruntime -- no torch, so it runs in
the live VGA loop alongside the EAST detector). Reads a single text-line image crop
to a string via greedy CTC decode. This is the recognizer that replaces Tesseract
on the detector-gated OCR path once wired in.

Degrades gracefully to disabled (read() -> "") if onnxruntime or the model files are
missing, so perception can fall back to Tesseract.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _HAVE_ORT = True
except Exception:  # noqa: BLE001
    _HAVE_ORT = False

_DEFAULT_ONNX = os.path.join(os.path.dirname(__file__), "..", "..",
                             "train", "ocr", "ckpt", "crnn.onnx")


class CrnnRecognizer:
    def __init__(self, onnx_path=None, logger=print):
        self.enabled = False
        self.logger = logger
        path = os.path.abspath(onnx_path or _DEFAULT_ONNX)
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        if not _HAVE_ORT:
            logger("[recog] onnxruntime not importable; CRNN recognizer disabled")
            return
        if not (os.path.exists(path) and os.path.exists(meta_path)):
            logger(f"[recog] onnx/meta not found ({path}); recognizer disabled")
            return
        meta = json.load(open(meta_path))
        self.chars = meta["chars"]
        self.height = meta["height"]
        self.downsample = meta["downsample"]
        self.channels = int(meta.get("channels", 1))
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.enabled = True
        logger(f"[recog] CRNN ONNX loaded: {path}")

    def _preprocess(self, line_bgr):
        """Match training. RGB model (F18, channels=3): keep COLOR -- the model was
        trained with color randomization so it handles arbitrary fg/bg by local
        contrast (no polarity guess). Legacy grayscale model (channels=1): grayscale +
        polarity invert if light-on-dark. Fixed height, natural width, [0,1]."""
        h, w = line_bgr.shape[:2]
        new_w = max(1, int(round(w * self.height / h)))
        if self.channels == 3:
            rgb = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (new_w, self.height), interpolation=cv2.INTER_AREA)
            x = rgb.astype(np.float32) / 255.0
            return x.transpose(2, 0, 1)[None]               # (1, 3, H, W)
        gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < 127:                       # light text on dark -> invert
            gray = 255 - gray
        gray = cv2.resize(gray, (new_w, self.height), interpolation=cv2.INTER_AREA)
        x = gray.astype(np.float32) / 255.0
        return x[None, None, :, :]                          # (1, 1, H, W)

    def _decode(self, logits):
        am = logits[0].argmax(-1)                           # (T,)
        out, prev = [], -1
        for a in am:
            a = int(a)
            if a != prev and a != 0:                        # collapse repeats, drop blank
                out.append(a)
            prev = a
        return "".join(self.chars[i - 1] for i in out if 1 <= i <= len(self.chars))

    def read(self, line_bgr) -> str:
        """Read a single text-line crop -> string. "" if disabled/empty."""
        if not self.enabled or line_bgr is None or line_bgr.size == 0:
            return ""
        x = self._preprocess(line_bgr)
        logits = self.sess.run(None, {self.input_name: x})[0]   # (1, T, n_classes)
        return self._decode(logits)
