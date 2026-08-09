#!/bin/bash
# Phase A: fair re-eval on the FIXED banks (train: teacher rate; holdout: the
#          pre-registered distillation baseline).
# Phase B (gated): arbiter rollout harvest over keybank_train -> distill dataset.
# Gate: train-bank arbiter collect rate must be >= 0.60, else stop -- harvesting
# from a weak teacher wastes 1.5h and poisons the distill set.
set -u
cd /home/timothy/projects/VGA

DOCKER="docker run --rm --runtime nvidia --network host --user 1000:1000 \
  -v /home/timothy/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130"
TRAIN_STATES='/vga/train/perception/keybank_train/states/keydrop-*.state'
HOLD_STATES='/vga/train/perception/keybank_holdout/states/keydrop-*.state'

echo "=== health check ==="
curl -s -m 15 http://127.0.0.1:8077/health || { echo "VLM SERVER DOWN"; exit 3; }
echo

echo "=== Phase A: fair re-eval ==="
for SEED in 0 100 200; do
  for ARM in s1 arbiter random; do
    $DOCKER python3 -u arbiter_v1.py --arm "$ARM" --seed "$SEED" \
      --states "$TRAIN_STATES" --out "arbiter_v2fair_train_${ARM}_s${SEED}.json" \
      || { echo "EVAL FAILED train $ARM s$SEED"; exit 4; }
    $DOCKER python3 -u arbiter_v1.py --arm "$ARM" --seed "$SEED" \
      --states "$HOLD_STATES" --out "arbiter_v2fair_holdout_${ARM}_s${SEED}.json" \
      || { echo "EVAL FAILED holdout $ARM s$SEED"; exit 4; }
  done
done

echo "=== pooled fair results ==="
python3 - <<'EOF'
import glob, json, sys
import statistics as st
rates = {}
for bank in ["train", "holdout"]:
    print("--- %s bank ---" % bank)
    for arm in ["s1", "arbiter", "random"]:
        eps = []
        for f in sorted(glob.glob("train/rl/arbiter_v2fair_%s_%s_s*.json" % (bank, arm))):
            eps.extend(json.load(open(f))["episodes"])
        n = len(eps)
        coll = sum(1 for e in eps if e["keys"] > 0)
        by_s2 = sum(1 for e in eps if e.get("collected_by_interject"))
        tks = [e["first_key_step"] for e in eps if e["first_key_step"] is not None]
        rates[(bank, arm)] = coll / max(1, n)
        print("%-8s n=%d collect=%d/%d (%.0f%%) byS2=%d medianfirst=%s"
              % (arm, n, coll, n, 100.0 * coll / max(1, n), by_s2,
                 int(st.median(tks)) if tks else None))
json.dump({"%s/%s" % k: v for k, v in rates.items()},
          open("train/rl/fair_eval_rates.json", "w"), indent=2)
sys.exit(0 if rates[("train", "arbiter")] >= 0.60 else 7)
EOF
rc=$?
if [ $rc -eq 7 ]; then echo "GATE FAILED: train-bank arbiter collect < 60% -- NOT harvesting"; exit 7; fi
[ $rc -ne 0 ] && { echo "POOLING FAILED"; exit 8; }

echo "=== Phase B: rollout harvest (12 seed-batches x 34 states) ==="
mkdir -p train/rl/distill_data/round1
for BASE in $(seq 1000 100 2100); do
  echo "--- rollout batch seed base $BASE ---"
  $DOCKER python3 -u arbiter_v1.py --arm arbiter --seed "$BASE" \
    --states "$TRAIN_STATES" --record-dir distill_data/round1 \
    --out "distill_data/round1/rollout_s${BASE}.json" \
    || { echo "ROLLOUT BATCH $BASE FAILED"; exit 9; }
done

echo "=== dataset summary ==="
ls train/rl/distill_data/round1/*.npz | wc -l
du -sh train/rl/distill_data/round1
echo "=== DONE ==="
