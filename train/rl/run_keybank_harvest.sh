#!/bin/bash
# Regenerate the keydrop state banks with the onscreen safe-box fix (2026-08-06).
# Two DISJOINT-seed banks: keybank_train (distill teacher rollouts + teacher-rate eval)
# and keybank_holdout (distilled-S1 eval ONLY -- never used for training data).
set -u
cd /home/timothy/projects/VGA

DOCKER="docker run --rm --runtime nvidia --user 1000:1000 \
  -v /home/timothy/projects/VGA:/vga -w /vga thor-rl:cu130"

echo "=== harvest keybank_train (seed 500, 36 eps) ==="
$DOCKER python3 -u train/rl/harvest_key_frames.py --save-states --no-frames \
  --episodes 36 --seed 500 --out train/perception/keybank_train \
  || { echo "TRAIN BANK HARVEST FAILED"; exit 4; }

echo "=== harvest keybank_holdout (seed 900, 16 eps) ==="
$DOCKER python3 -u train/rl/harvest_key_frames.py --save-states --no-frames \
  --episodes 16 --seed 900 --out train/perception/keybank_holdout \
  || { echo "HOLDOUT BANK HARVEST FAILED"; exit 5; }

echo "=== verify banked states are inside the safe box ==="
python3 - <<'EOF'
import glob, json, sys
BANK_X, BANK_Y = 104 - 20, 64 - 14
bad = 0
for bank in ["keybank_train", "keybank_holdout"]:
    sides = sorted(glob.glob("train/perception/%s/states/keydrop-*.json" % bank))
    dys = []
    for s in sides:
        d = json.load(open(s))
        dx = d["drop"][0] - d["link"][0]
        dy = d["drop"][1] - d["link"][1]
        dys.append((abs(dx), abs(dy)))
        if abs(dx) > BANK_X or abs(dy) > BANK_Y:
            print("VIOLATION %s dx=%d dy=%d" % (s, dx, dy))
            bad += 1
    n = len(sides)
    print("%s: %d states; max|dx|=%s max|dy|=%s"
          % (bank, n, max((d[0] for d in dys), default=None),
             max((d[1] for d in dys), default=None)))
    if bank == "keybank_train" and n < 24:
        print("TOO FEW TRAIN STATES (<24)"); bad += 1
    if bank == "keybank_holdout" and n < 10:
        print("TOO FEW HOLDOUT STATES (<10)"); bad += 1
sys.exit(1 if bad else 0)
EOF
rc=$?
[ $rc -ne 0 ] && { echo "BANK VERIFY FAILED"; exit 6; }
echo "=== DONE ==="
