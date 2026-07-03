"""
Apply positional labels to a uniq_manifest.json.

Labeling flow: each uniq_montage_NN.png shows 30 crops top-to-bottom, in the SAME
order as uniq_manifest.json (global index = NN*30 + row). Claude (VLM) reads a
montage and writes train/ocr/data/<pool>/labels/labels_NN.json -- a JSON list of
30 entries, each either the transcribed text (conventions: text-only, accents like
POKeMON->POKéMON, "No001" dex prefix) or null to DROP the crop (junk/non-Latin/
EAST false-positive). This script merges all labels_*.json into uniq_manifest.json
by index, and reports coverage.

  python train/ocr/apply_labels.py --dir train/ocr/data/real_pokemon
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--per-montage", type=int, default=30)
    args = ap.parse_args()

    man_path = os.path.join(args.dir, "uniq_manifest.json")
    man = json.load(open(man_path))
    lab_dir = os.path.join(args.dir, "labels")

    applied = dropped = 0
    for lf in sorted(glob.glob(os.path.join(lab_dir, "labels_*.json"))):
        nn = int(os.path.basename(lf).split("_")[1].split(".")[0])
        labels = json.load(open(lf))
        base = nn * args.per_montage
        for i, lab in enumerate(labels):
            idx = base + i
            if idx >= len(man):
                print(f"WARN {lf}[{i}] -> index {idx} out of range; skipped")
                continue
            man[idx]["label"] = lab
            if lab is None:
                dropped += 1
            else:
                applied += 1

    json.dump(man, open(man_path, "w"), indent=2)
    labeled = sum(1 for e in man if e.get("label"))
    unlabeled = sum(1 for e in man if e.get("label") is None and "label" in e)
    print(f"applied {applied} text labels, {dropped} drops; "
          f"manifest now {labeled} labeled / {len(man)} total")


if __name__ == "__main__":
    main()
