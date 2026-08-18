"""box_reader.py -- ALttP-FS gameplay dialogue-box detection + localization (pixels-only).

The gameplay text box is a gold/tan double-bordered box at a FIXED bottom location. detect_box
gates "is a box on screen NOW" (the inference read-gate AND the harvest advance-and-wait
trigger); it keys on the horizontal gold BORDER runs, validated 2026-08-10 to separate the
uncle box (4 long gold rows) from gameplay/house-floor frames (<=2). Combine with "RAM text
present" during harvest so a stray floor false-fire is filtered.

Line localization (box_lines) is developed on fully-typed captures -- see _alttp_boxharvest.py.
"""
import numpy as np

# bottom region where the box + its gold border live (240x160 frame)
BY0, BY1, BX0, BX1 = 116, 159, 4, 236
# interior text region (inside the border) -- refined once fully-typed boxes exist
IY0, IY1, IX0, IX1 = 120, 158, 10, 230


def gold_mask(a):
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return (R > 175) & (G > 130) & (B < 135) & (R > B + 70) & (G > B + 35)


def detect_box(frame):
    """True if the gameplay dialogue box is on screen. Keys on a near-full-width gold row at
    the box's BOTTOM-BORDER position (y 150-156). The house floor's tan rows sit higher
    (y~137/146, present with OR without a box) or at the screen edge (y~158), so this band is
    box-specific -- verified 2026-08-10 (uncle box y=152/153 vs floor/edge false-fires)."""
    gm = gold_mask(frame[150:157, BX0:BX1])
    return bool((gm.sum(1) >= 90).any())
