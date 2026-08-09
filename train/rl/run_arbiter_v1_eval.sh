#!/bin/bash
# Arbiter v1 eval: S1 reflex vs S1+S2-key-interject vs random, on the keydrop state bank.
# 2026-08-06 P0. Needs the vlm8b server healthy on 127.0.0.1:8077 (thor-job vlm8b).
# Smoke first so an interface bug costs seconds, not the full run.
set -u
cd /home/timothy/projects/VGA

DOCKER="docker run --rm --runtime nvidia --network host --user 1000:1000 \
  -v /home/timothy/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130"

echo "=== health check ==="
curl -s -m 15 http://127.0.0.1:8077/health || { echo "VLM SERVER DOWN"; exit 3; }
echo

echo "=== smoke: random arm, 1 state (env+state-bank interface) ==="
$DOCKER python3 -u arbiter_v1.py --arm random \
  --states '/vga/train/perception/keyprobe_v2/states/keydrop-00.state' \
  --out /tmp/smoke_random.json || { echo "SMOKE RANDOM FAILED"; exit 4; }

echo "=== smoke: arbiter arm, 1 state (S2 consult path) ==="
$DOCKER python3 -u arbiter_v1.py --arm arbiter \
  --states '/vga/train/perception/keyprobe_v2/states/keydrop-00.state' \
  --out /tmp/smoke_arbiter.json || { echo "SMOKE ARBITER FAILED"; exit 5; }

echo "=== full eval: 3 seeds x 3 arms x 12 states ==="
for SEED in 0 100 200; do
  for ARM in s1 arbiter random; do
    OUT="arbiter_v1_${ARM}_s${SEED}.json"
    echo "--- arm=$ARM seed=$SEED -> $OUT ---"
    $DOCKER python3 -u arbiter_v1.py --arm "$ARM" --seed "$SEED" --out "$OUT" \
      || { echo "RUN FAILED arm=$ARM seed=$SEED"; exit 6; }
  done
done

echo "=== pooled comparison ==="
python3 - <<'EOF'
import glob, json
import statistics as st
for arm in ["s1", "arbiter", "random"]:
    eps = []
    for f in sorted(glob.glob("train/rl/arbiter_v1_%s_s*.json" % arm)):
        eps.extend(json.load(open(f))["episodes"])
    n = len(eps)
    coll = sum(1 for e in eps if e["keys"] > 0)
    by_s2 = sum(1 for e in eps if e.get("collected_by_interject"))
    tks = [e["first_key_step"] for e in eps if e["first_key_step"] is not None]
    print("%-8s n=%d collect=%d/%d (%.0f%%) keys/ep=%.2f byS2=%d medianfirst=%s dmg=%.0f"
          % (arm, n, coll, n, 100.0 * coll / max(1, n),
             sum(e["keys"] for e in eps) / max(1, n), by_s2,
             int(st.median(tks)) if tks else None,
             sum(e["damage"] for e in eps) / max(1, n)))
EOF
echo "=== DONE ==="
