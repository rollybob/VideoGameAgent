#!/bin/bash
# Distill round 1: train per docs/DISTILL_SPEC_2026-08-06.md, then the pre-registered
# holdout comparison. No VLM needed (all arms are s1-solo). Smoke-train first so a
# loader/shape bug costs 2 minutes, not the dataset's worth of GPU time.
set -u
cd /home/timothy/projects/VGA

DOCKER="docker run --rm --runtime nvidia --user 1000:1000 \
  -v /home/timothy/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130"
HOLD_STATES='/vga/train/perception/keybank_holdout/states/keydrop-*.state'
DATA=train/rl/distill_data/round1

N=$(ls $DATA/*.npz 2>/dev/null | wc -l)
echo "dataset: $N success trajectories"
[ "$N" -lt 150 ] && { echo "DATASET TOO THIN (<150) -- stop and reassess"; exit 3; }

echo "=== smoke train (10 eps, 1 epoch) ==="
$DOCKER python3 -u distill_bc.py --data distill_data/round1 --limit-eps 10 --epochs 1 \
  --out runs/distill_r1/smoke.zip || { echo "SMOKE TRAIN FAILED"; exit 4; }

echo "=== full train ==="
$DOCKER python3 -u distill_bc.py --data distill_data/round1 \
  --out runs/distill_r1/s1_distilled.zip || { echo "TRAIN FAILED"; exit 5; }

echo "=== eval: distilled S1 solo on holdout, 6 seeds ==="
for SEED in 0 100 200 300 400 500; do
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r1/s1_distilled.zip --states "$HOLD_STATES" \
    --out "distill_r1_eval_s${SEED}.json" || { echo "EVAL FAILED s$SEED"; exit 6; }
done

echo "=== baseline extension: base S1 seeds 300-500 (0-200 exist from fair eval) ==="
for SEED in 300 400 500; do
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --states "$HOLD_STATES" --out "arbiter_v2fair_holdout_s1_s${SEED}.json" \
    || { echo "BASELINE EXT FAILED s$SEED"; exit 7; }
done

echo "=== secondary: deterministic runs (1 seed each; det is reproducible) ==="
$DOCKER python3 -u arbiter_v1.py --arm s1 --det --seed 0 \
  --ckpt runs/distill_r1/s1_distilled.zip --states "$HOLD_STATES" \
  --out distill_r1_eval_det.json || echo "WARN: det eval (distilled) failed"
$DOCKER python3 -u arbiter_v1.py --arm s1 --det --seed 0 \
  --states "$HOLD_STATES" --out arbiter_v2fair_holdout_s1_det.json \
  || echo "WARN: det eval (base) failed"

echo "=== VERDICT vs pre-registered thresholds ==="
python3 - <<'EOF'
import glob, json
def pool(pat):
    eps = []
    for f in sorted(glob.glob(pat)):
        eps.extend(json.load(open(f))["episodes"])
    return eps
def rate(eps):
    return sum(1 for e in eps if e["keys"] > 0) / max(1, len(eps))
dist = pool("train/rl/distill_r1_eval_s*.json")
base = pool("train/rl/arbiter_v2fair_holdout_s1_s*.json")
ceil = pool("train/rl/arbiter_v2fair_holdout_arbiter_s*.json")
det_d = pool("train/rl/distill_r1_eval_det.json")
det_b = pool("train/rl/arbiter_v2fair_holdout_s1_det.json")
rd, rb, rc = rate(dist), rate(base), rate(ceil)
delta = (rd - rb) * 100
gap_closed = (rd - rb) / max(1e-9, rc - rb)
verdict = "SUCCESS" if delta >= 15 else ("PARTIAL" if delta >= 5 else "KILL")
dmg = lambda eps: sum(e["damage"] for e in eps) / max(1, len(eps))
out = {"distilled": {"n": len(dist), "collect": rd},
       "baseline": {"n": len(base), "collect": rb},
       "ceiling_arbiter": {"n": len(ceil), "collect": rc},
       "delta_points": round(delta, 1),
       "gap_closed_frac": round(gap_closed, 2),
       "det": {"distilled": rate(det_d), "base": rate(det_b)},
       "damage": {"distilled": dmg(dist), "base": dmg(base)},
       "verdict": verdict}
print(json.dumps(out, indent=2))
json.dump(out, open("train/rl/distill_r1_verdict.json", "w"), indent=2)
print("VERDICT: %s (delta %+.1f pts, gap closed %.0f%%)" % (verdict, delta, 100 * gap_closed))
EOF
echo "=== DONE ==="
