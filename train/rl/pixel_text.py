"""pixel_text.py -- read on-screen dialogue text from PIXELS at inference.

North-star ([[project-vga-north-star]]): RAM is a TEACHER, not a sensor. The RAM oracle
(ram_text.extract_text) taught a CRNN to read ALttP text from pixels; this module runs that
CRNN (ckpt_alttp, held-out CER 0.053) live so the agent reads what is on screen instead of
peeking at WRAM. It REPLACES ram_text.extract_text in the /act loop -- RAM stays only as the
scoring ORACLE (see drive_agent's pixel_vs_ram sidecar).

Path: seg_lines(full frame) -> per-line CRNN greedy-CTC -> order top-to-bottom -> join. The
line finder is polarity/position/box-color general (detect the text SIGNAL, not the box
container -- the lesson from make_alttp_data). Input frame = RGB HxWx3 uint8 (mgba buffer is
RGBX, so drive_agent's grab() and the CRNN's training mark.png crops are the same channel
order -- verified 2026-08-18; no BGR swap).

seg_lines()/_edges() below are copied VERBATIM from make_alttp_data.py on purpose: the CRNN
was fine-tuned on the crops that function produces, so the live segmenter MUST match it byte
for byte. If you change one, change both.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_OCR = os.path.normpath(os.path.join(_HERE, "..", "ocr"))
if _OCR not in sys.path:
    sys.path.insert(0, _OCR)                       # for `model.CRNN`
from model import CRNN

DEFAULT_CKPT = os.environ.get("ALTTP_OCR", os.path.join(_OCR, "ckpt_alttp", "best.pt"))


# ---- GENERAL text-line detection (MUST match make_alttp_data.seg_lines) ----
def _edges(frame):
    g = frame.mean(2)
    return np.abs(np.diff(g, axis=1)) > 40           # strong horizontal-gradient px = character strokes


def seg_lines(frame):
    """Return text-line crops (RGB uint8), top-to-bottom, from anywhere on the frame."""
    E = _edges(frame); ec = E.sum(1)
    thr = max(10, ec.max() * 0.22)                   # adaptive: text rows have many edges vs uniform bg
    on = ec >= thr; bands, ins, s = [], False, 0
    for r, v in enumerate(on):
        if v and not ins: s, ins = r, True
        elif not v and ins: bands.append((s, r)); ins = False
    if ins: bands.append((s, len(on)))
    crops = []
    for a, b in bands:
        if not (5 <= b - a <= 22): continue          # plausible single text-line height
        cols = np.where(E[a:b].sum(0) >= 2)[0]
        if len(cols) < 3: continue
        x0, x1 = int(cols.min()), int(cols.max()) + 2
        if x1 - x0 < 20: continue
        crop = frame[max(0, a - 2):b + 2, max(0, x0 - 2):x1 + 2]
        if crop.shape[0] >= 6 and crop.shape[1] >= 12: crops.append(crop)
    return crops


class PixelReader:
    """Loads ckpt_alttp once; read_message(frame) -> str (the on-screen message, or "")."""

    def __init__(self, ckpt=DEFAULT_CKPT, device=None):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.chars = ck["meta"]["chars"]
        n_classes = ck["meta"]["n_classes"]
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CRNN(n_classes, in_ch=3).eval().to(self.dev)
        self.net.load_state_dict(ck["model"])

    def _greedy(self, am, blank=0):                  # CTC greedy collapse (== make_alttp_data.greedy)
        out, prev = [], -1
        for a in am:
            a = int(a)
            if a != prev and a != blank: out.append(a)
            prev = a
        return "".join(self.chars[i - 1] for i in out if 1 <= i <= len(self.chars))

    @torch.no_grad()
    def ocr(self, crop):
        """Read one line crop (RGB uint8) -> string."""
        im = Image.fromarray(crop)
        nw = max(8, round(im.width * 32 / im.height))
        a = np.asarray(im.resize((nw, 32), Image.BILINEAR), np.float32) / 255.0
        x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.dev)
        return self._greedy(self.net(x).argmax(-1)[0].cpu().numpy())

    def read_lines(self, frame):
        """Per-line reads, top-to-bottom. Drops fragments with <2 letters -- box-border/corner
        crops OCR to junk like 'o'/'1'/'.' (validation 2026-08-18: ~35% of raw line-crops on
        dialogue frames were such 1-char fragments); a real text line always has >=2 letters,
        so this trims noise with no risk to genuine words."""
        out = []
        for c in seg_lines(frame):
            t = self.ocr(c).strip()
            if sum(ch.isalpha() for ch in t) >= 2:
                out.append(t)
        return out

    def read_message(self, frame):
        """The whole on-screen message as one string, or "" if there is no prose text.

        Gate mirrors ram_text.extract_text (>= 3 alpha) so pixel and RAM readings are scored
        apples-to-apples. This is a lenient prose filter, NOT a dialogue-box detector -- the
        HUD is mostly icons/digits so it rarely trips the gate, but a learned box gate is the
        principled follow-up if the live sidecar shows HUD/menu false-positives.
        """
        s = " ".join(self.read_lines(frame))
        s = " ".join(s.split())                      # collapse whitespace, like extract_text
        return s if sum(c.isalpha() for c in s) >= 3 else ""


# Module-level singleton so callers can `from pixel_text import read_message` without managing
# the model. Lazily built on first use (keeps import side-effect-free; the CRNN load + cuda
# alloc only happen when a reader is actually needed).
_READER = None


def get_reader():
    global _READER
    if _READER is None:
        _READER = PixelReader()
    return _READER


def read_message(frame):
    return get_reader().read_message(frame)


if __name__ == "__main__":
    # Smoke: read one PNG frame passed on the CLI.
    r = PixelReader()
    for p in sys.argv[1:]:
        fr = np.array(Image.open(p).convert("RGB"))
        print(f"{p}: lines={r.read_lines(fr)!r}")
        print(f"  message={r.read_message(fr)!r}")
