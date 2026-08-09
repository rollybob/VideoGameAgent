#!/usr/bin/env python3
"""Host: build a HELD-OUT mode-classifier eval set from REAL bench trajectory frames.

These bench frames are native 240x160 emulator output (same domain as the scripted-tour
training data -- no pillarbox), and each carries a logged overlay byte + VLM self-report in
steps.jsonl. The overlay byte is NOT a clean oracle (it takes ~19 values), so this script
only SAMPLES a stratified, diverse set; the ground-truth label comes from a human eyeballing
the contact sheets it emits. Outputs:

  <out>.npz            X (N,96,144,3 uint8 RGB), idx (N,) global ids   -> feed eval_infer.py
  <out>.meta.json      gidx -> {dir, frame, overlay, vlm}              -> join preds + truth
  <out>.sheet_NN.png   contact sheets (native frames, gidx printed)    -> eyeball to label

    .venv/bin/python train/perception/build_real_eval.py --out /tmp/real_eval --per-family 22
"""
from __future__ import annotations
import argparse, glob, json, os, collections
import cv2, numpy as np

# overlay-byte families (labels here are HINTS for stratification only, NOT ground truth)
FAMILIES = {
    "menu_hi":     {255, 204},
    "hi_unknown":  {208, 207, 252, 131, 132, 137},
    "dialog_help": {3},
    "field_lo":    {0, 114},
    "lo_unknown":  {9, 17, 68, 99, 1, 48, 51},
}
SKIP = {19}  # transitions / near-black


def family_of(ov):
    for name, s in FAMILIES.items():
        if ov in s:
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/real_eval")
    ap.add_argument("--per-family", type=int, default=22)
    ap.add_argument("--img-h", type=int, default=96)
    ap.add_argument("--img-w", type=int, default=144)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-sheet", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # gather all (dir, frame, overlay, vlm) rows that have a known family
    pool = collections.defaultdict(list)
    for sp in glob.glob("sessions/**/steps.jsonl", recursive=True):
        d = os.path.dirname(sp)
        for r in (json.loads(l) for l in open(sp)):
            st = r.get("state") or {}
            ov = st.get("mode_overlay")
            if ov is None or ov in SKIP:
                continue
            fam = family_of(ov)
            if fam is None:
                continue
            fp = os.path.join(d, r["frame"])
            if os.path.exists(fp):
                pool[fam].append((d, fp, ov, str(r.get("mode"))))

    # stratified sample: spread across dirs within each family
    chosen = []
    for fam, rows in pool.items():
        by_dir = collections.defaultdict(list)
        for row in rows:
            by_dir[row[0]].append(row)
        for v in by_dir.values():
            rng.shuffle(v)
        dirs = list(by_dir)
        rng.shuffle(dirs)
        picked, i = [], 0
        while len(picked) < args.per_family and any(by_dir.values()):
            d = dirs[i % len(dirs)]; i += 1
            if by_dir[d]:
                picked.append(by_dir[d].pop())
        chosen.extend((fam,) + row for row in picked)
    rng.shuffle(chosen)

    # build npz + meta + sheets
    X, idx, meta = [], [], {}
    thumbs = []
    for g, (fam, d, fp, ov, vlm) in enumerate(chosen):
        img = cv2.imread(fp)
        rs = cv2.resize(img, (args.img_w, args.img_h), interpolation=cv2.INTER_AREA)
        X.append(cv2.cvtColor(rs, cv2.COLOR_BGR2RGB))
        idx.append(g)
        meta[g] = {"dir": d, "frame": fp, "overlay": int(ov), "vlm": vlm, "family": fam}
        thumbs.append((g, img))
    np.savez(args.out + ".npz", X=np.asarray(X, np.uint8), idx=np.asarray(idx, np.int64))
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=0)

    # contact sheets: native frames upscaled 1.5x, gidx printed top-left
    per = args.per_sheet
    cols = 5
    for s in range((len(thumbs) + per - 1) // per):
        batch = thumbs[s * per:(s + 1) * per]
        rows = (len(batch) + cols - 1) // cols
        H, W = 240, 360  # 1.5x of 160x240
        canvas = np.full((rows * (H + 24), cols * W, 3), 30, np.uint8)
        for j, (g, img) in enumerate(batch):
            r, c = divmod(j, cols)
            big = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
            y0 = r * (H + 24) + 24
            canvas[y0:y0 + H, c * W:c * W + W] = big
            cv2.putText(canvas, f"#{g}", (c * W + 4, r * (H + 24) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(f"{args.out}.sheet_{s:02d}.png", canvas)
    print("frames:", len(X), "sheets:", (len(thumbs) + per - 1) // per, "->", args.out)
    print("family counts:", {k: len(v) for k, v in pool.items()})


if __name__ == "__main__":
    main()
