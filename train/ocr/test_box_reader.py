import numpy as np

# Import the module under test exactly as specified.
import box_reader

# ----------------------------------------------------------------------
# Basic sanity checks for exported constants
# ----------------------------------------------------------------------
assert hasattr(box_reader, "BY0") and isinstance(box_reader.BY0, int)
assert hasattr(box_reader, "BY1") and isinstance(box_reader.BY1, int)
assert hasattr(box_reader, "BX0") and isinstance(box_reader.BX0, int)
assert hasattr(box_reader, "BX1") and isinstance(box_reader.BX1, int)

# Verify the values match the source (hard‑coded contract).
assert box_reader.BY0 == 116
assert box_reader.BY1 == 159
assert box_reader.BX0 == 4
assert box_reader.BX1 == 236

# Interior region constants – also part of the public API.
assert box_reader.IY0 == 120
assert box_reader.IY1 == 158
assert box_reader.IX0 == 10
assert box_reader.IX1 == 230

# ----------------------------------------------------------------------
# Tests for gold_mask()
# ----------------------------------------------------------------------
def manual_gold_condition(pixel):
    """Return True if the pixel satisfies the gold criteria used in gold_mask."""
    R, G, B = map(int, pixel)
    return (R > 175) and (G > 130) and (B < 135) and (R > B + 70) and (G > B + 35)

# Create a small test array with known pixels.
test_pixels = np.array([
    [[200, 150,   0], [176, 131,  10]],   # both gold
    [[175, 130, 134], [180, 140, 100]],   # first not gold (R==175), second gold
    [[255, 255, 255], [ 50,  60,  70]]    # none gold
], dtype=np.uint8)

mask = box_reader.gold_mask(test_pixels)
# mask should be a boolean array of shape (3,2) – the channel dimension is removed.
assert isinstance(mask, np.ndarray)
assert mask.dtype == bool
assert mask.shape == test_pixels.shape[:2]

# Verify each element matches manual condition.
for i in range(test_pixels.shape[0]):
    for j in range(test_pixels.shape[1]):
        assert mask[i, j] == manual_gold_condition(test_pixels[i, j])

# Edge case: values exactly on the thresholds should be *included* because
# the conditions are strict (>). Use a pixel that satisfies all > checks.
edge_pixel = np.array([[[176, 131,   0]]], dtype=np.uint8)  # passes all > tests
assert box_reader.gold_mask(edge_pixel)[0, 0] == True

# Pixel that fails by one unit on each threshold.
fail_pixel = np.array([[[176, 130,   0]]], dtype=np.uint8)  # G == 130 (fails)
assert box_reader.gold_mask(fail_pixel)[0, 0] == False

# ----------------------------------------------------------------------
# Tests for detect_box()
# ----------------------------------------------------------------------
def make_frame(gold_rows):
    """
    Create a dummy 160x240 RGB frame filled with zeros.
    gold_rows: dict mapping y-coordinate (int) -> number of consecutive gold pixels
               starting at column BX0. Pixels are set to a known gold colour.
    Returns the frame as a uint8 ndarray.
    """
    h, w = 160, 240
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Gold colour that satisfies gold_mask (R=200,G=150,B=0)
    gold_colour = np.array([200, 150, 0], dtype=np.uint8)

    for y, count in gold_rows.items():
        if not (0 <= y < h):
            continue
        start = box_reader.BX0
        end = min(start + count, box_reader.BX1)
        frame[y, start:end] = gold_colour

    return frame

# 1. Positive detection: a row inside the slice with >=90 gold pixels.
frame_true = make_frame({152: 100})   # y=152 is within 150‑156 range.
assert box_reader.detect_box(frame_true) is True
assert isinstance(box_reader.detect_box(frame_true), bool)

# 2. Exact threshold (exactly 90 gold pixels) should still be detected.
frame_exact = make_frame({154: 90})
assert box_reader.detect_box(frame_exact) is True

# 3. Below threshold (89 gold pixels) must NOT trigger detection.
frame_false = make_frame({155: 89})
assert box_reader.detect_box(frame_false) is False

# 4. Gold row outside the vertical slice should be ignored.
frame_outside_y = make_frame({149: 200})   # y=149 is just above the slice.
assert box_reader.detect_box(frame_outside_y) is False

# 5. Gold row inside slice but only partially covering the horizontal region
#    (e.g., start at BX0+10). The sum per row counts only true gold pixels,
#    so we need >=90 true pixels regardless of offset.
frame_offset = make_frame({153: 120})
assert box_reader.detect_box(frame_offset) is True

# 6. Multiple rows, one meeting the threshold and others not – should be True.
frame_multi = make_frame({151: 50, 152: 95, 153: 30})
assert box_reader.detect_box(frame_multi) is True

# 7. Ensure that non‑array input raises an exception (AttributeError from .astype).
try:
    box_reader.detect_box("not a frame")
except Exception as e:
    assert isinstance(e, (AttributeError, TypeError, IndexError))
else:
    raise AssertionError("detect_box did not raise on invalid input")

# 8. Ensure that an array lacking the colour channel raises an exception.
bad_shape = np.zeros((160, 240), dtype=np.uint8)   # missing third dimension
try:
    box_reader.detect_box(bad_shape)
except Exception as e:
    assert isinstance(e, (IndexError, ValueError))
else:
    raise AssertionError("detect_box did not raise on shape without colour channel")

# ----------------------------------------------------------------------
# All tests passed.
print('OK')
