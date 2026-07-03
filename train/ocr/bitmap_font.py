"""
Real GBA bitmap-glyph font provider (decomp route, F18 pivot -- see
docs/DECOMP_FONT_PLAYBOOK.md).

Parses a pret decomp latin font: a PNG glyph atlas (16-wide grid of fixed cells,
2bpp palette) + a per-glyph width table (variable-width) + the charmap (char ->
byte code = glyph index). Exposes render_mask(text, height) -> (alpha mask, width)
with the SAME contract as gen_data_mf.render_mask, so it drops into the existing
color/background/augment pipeline as a new glyph SOURCE (playbook section 5).

Palette semantics (confirmed from latin_normal.png raw indices):
  0 = outside the glyph box (right/bottom fill)   -> transparent
  3 = background paper INSIDE the box             -> transparent
  1 = main stroke (the letter)                    -> ink
  2 = shadow / secondary stroke                   -> ink (configurable)
Background = {0,3}; ink = {1,2}. We extract only the left `width` columns of each
16px cell, so the index-0 right fill is never included.

FRLG (pokefirered) latin fonts are a 16x16-px cell grid, 16 columns. The decomp
ships normal / small / male / female; normal is the main dialogue font that
bench_pokemon was sampled from -- the highest-leverage source for the real gap.
"""
from __future__ import annotations

import os
import re

import numpy as np
from PIL import Image

# Maps a logical FRLG latin font to (png filename, width-table symbol in text.c).
FRLG_FONTS = {
    "normal": ("latin_normal.png", "sFontNormalLatinGlyphWidths"),
    "male":   ("latin_male.png",   "sFontMaleLatinGlyphWidths"),
    "female": ("latin_female.png", "sFontFemaleLatinGlyphWidths"),
    # small uses a different cell height; deferred (see playbook), parse separately.
}


def parse_charmap(path):
    """charmap.txt lines like  'A' = BB  ->  {char: code}. Single-char tokens only
    (control codes and multi-char macros are ignored -- we only render text glyphs)."""
    char_to_code = {}
    line_re = re.compile(r"^'(.)'\s*=\s*([0-9A-Fa-f]{2})\b")
    with open(path, encoding="utf-8") as f:
        for ln in f:
            m = line_re.match(ln)
            if m:
                char_to_code[m.group(1)] = int(m.group(2), 16)
    return char_to_code


def parse_width_table(text_c_path, symbol):
    """Extract `static const u8 <symbol>[] = { ... };` -> list[int]."""
    src = open(text_c_path, encoding="utf-8").read()
    m = re.search(re.escape(symbol) + r"\[\]\s*=\s*\{([^}]*)\}", src, re.S)
    if not m:
        raise ValueError(f"width table {symbol} not found in {text_c_path}")
    return [int(v) for v in re.split(r"[,\s]+", m.group(1)) if v != ""]


class BitmapFont:
    """A parsed decomp bitmap font. render_mask matches gen_data_mf.render_mask."""

    def __init__(self, png_path, width_table, char_to_code,
                 cell=16, cols=16, ink=(1,)):
        # ink defaults to the MAIN stroke (index 1) only: the smoke test showed that
        # baking the shadow (index 2) into a single-color mask over-bolds the glyph,
        # while main-only + gen_data_mf.composite()'s own 1px shadow reproduces the
        # real two-tone FRLG look faithfully (and matches the bench_pokemon crops).
        self.idx = np.array(Image.open(png_path))     # mode 'P' -> palette indices
        self.cell = cell
        self.cols = cols
        self.widths = width_table
        self.char_to_code = char_to_code
        self.ink = tuple(ink)
        self.name = os.path.splitext(os.path.basename(png_path))[0]

    @classmethod
    def frlg(cls, font_dir, which="normal", ink=(1,)):
        png, sym = FRLG_FONTS[which]
        return cls(os.path.join(font_dir, png),
                   parse_width_table(os.path.join(font_dir, "text.c"), sym),
                   parse_charmap(os.path.join(font_dir, "charmap.txt")),
                   ink=ink)

    def _glyph(self, code, ink):
        """Left `width` columns of the code's cell, as an (cell, width) ink mask."""
        w = self.widths[code] if code < len(self.widths) else self.cell
        w = max(1, min(w, self.cell))
        r, c = divmod(code, self.cols)
        sub = self.idx[r * self.cell:(r + 1) * self.cell,
                       c * self.cell:c * self.cell + w]
        return np.isin(sub, ink).astype(np.uint8) * 255, w

    def render_mask(self, text, height, ink=None):
        """text -> (alpha mask (height,w) uint8 255=ink, w). Native-scale compose at
        the font's cell height, advance by each glyph's WIDTH (variable pitch), then
        height-normalize with NEAREST (crisp pixel scaling, like an integer-scaled GBA
        framebuffer -- avoids blurring the bitmap glyph). ink overrides self.ink for this
        call -- e.g. (1,) thin main-stroke vs (1,2) the BOLD main+shadow weight FRLG uses
        on some screens (Pokedex/menus); randomizing it in gen makes the model robust to
        both thicknesses (CONTINUE is thin, AREA/SIZE/FUMON are bold)."""
        ink = self.ink if ink is None else tuple(ink)
        pieces = []
        for ch in text:
            code = self.char_to_code.get(ch)
            if code is None:
                code = self.char_to_code.get(" ", 0)   # unknown -> space-width gap
            g, w = self._glyph(code, ink)
            pieces.append(g)
        if not pieces:
            pieces = [np.zeros((self.cell, 1), np.uint8)]
        line = np.concatenate(pieces, axis=1)
        w_native = line.shape[1]
        w_out = max(1, int(round(w_native * height / self.cell)))
        out = np.array(Image.fromarray(line).resize((w_out, height), Image.NEAREST),
                       np.uint8)
        return out, w_out
