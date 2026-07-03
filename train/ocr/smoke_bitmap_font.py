"""
Smoke test for bitmap_font.BitmapFont (playbook step 2): render every labeled
bench_pokemon string with the FRLG normal font and stack it next to the REAL crop,
in two ink variants (main stroke only vs main+shadow), so we can eyeball fidelity.

Output: data/bench_pokemon/bitmap_smoke.png  (rows: label | real | main | main+shadow)
"""
import json
import os

import cv2
import numpy as np

from bitmap_font import BitmapFont

FONT_DIR = "fonts/decomp/pokefirered"
BENCH = "data/bench_pokemon"
PANEL_W = 360            # fixed canvas width per render panel
ROW_H = 28              # render height per row


def mask_to_img(mask):
    """alpha mask (255=ink) -> dark text on light panel, padded to (ROW_H, PANEL_W)."""
    rgb = 255 - np.repeat(mask[:, :, None], 3, axis=2)     # ink black on white
    h, w = rgb.shape[:2]
    canvas = np.full((ROW_H, PANEL_W, 3), 230, np.uint8)
    canvas[:min(h, ROW_H), :min(w, PANEL_W)] = rgb[:ROW_H, :PANEL_W]
    return canvas


def real_crop(cid):
    p = os.path.join(BENCH, "crops", cid + ".png")
    img = cv2.imread(p)
    if img is None:
        return np.full((ROW_H, PANEL_W, 3), 200, np.uint8)
    s = ROW_H / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * s), ROW_H), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((ROW_H, PANEL_W, 3), 200, np.uint8)
    canvas[:, :min(img.shape[1], PANEL_W)] = img[:, :PANEL_W]
    return canvas


def label_panel(text):
    canvas = np.full((ROW_H, 200, 3), 255, np.uint8)
    cv2.putText(canvas, text[:24], (3, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return canvas


def main():
    f_main = BitmapFont.frlg(FONT_DIR, "normal", ink=(1,))
    f_both = BitmapFont.frlg(FONT_DIR, "normal", ink=(1, 2))
    labels = json.load(open(os.path.join(BENCH, "bench_labels.json")))
    print(f"[smoke] {len(labels)} labeled crops; font glyphs={len(f_main.char_to_code)}")

    rows = []
    for e in labels:
        text = e["label"]
        m1, _ = f_main.render_mask(text, ROW_H)
        m2, _ = f_both.render_mask(text, ROW_H)
        row = np.hstack([label_panel(text), real_crop(e["id"]),
                         mask_to_img(m1), mask_to_img(m2)])
        rows.append(row)
        rows.append(np.full((2, row.shape[1], 3), 120, np.uint8))    # separator

    # header
    hdr = np.full((22, rows[0].shape[1], 3), 255, np.uint8)
    for x, t in [(3, "label"), (203, "REAL crop"), (563, "render main"),
                 (923, "render main+shadow")]:
        cv2.putText(hdr, t, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    montage = np.vstack([hdr] + rows)
    out = os.path.join(BENCH, "bitmap_smoke.png")
    cv2.imwrite(out, montage)
    print(f"[smoke] wrote {out}  {montage.shape}")


if __name__ == "__main__":
    main()
