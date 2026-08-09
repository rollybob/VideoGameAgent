#!/bin/bash
# Distill round 1 under AMENDMENT 1 (docs/DISTILL_SPEC_2026-08-06.md): harder
# holdout2 bench, headroom gate BEFORE training, then train -> eval -> verdict.
# Needs vlm8b on :8077 only for the arbiter-ceiling arm.
set -u
cd /home/timothy/projects/VGA

DOCKER="docker run --rm --runtime nvidia --network host --user 1000:1000 \
  -v /home/timothy/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130"
H2_STATES='/vga/train/perception/keybank_holdout2/states/keydrop-*.state'
H1_STATES='/vga/train/perception/keybank_holdout/states/keydrop-*.state'

echo "=== health check ==="
curl -s -m 15 http://127.0.0.1:8077/health || { echo "VLM SERVER DOWN"; exit 3; }
echo

echo "=== harvest keybank_holdout2 (wide scatter, seed 1300, 22 eps) ==="
$DOCKER python3 -u harvest_key_frames.py --save-states --no-frames --wide-scatter \
  --episodes 22 --seed 1300 --out /vga/train/perception/keybank_holdout2 \
  || { echo "HOLDOUT2 HARVEST FAILED"; exit 4; }

echo "=== verify holdout2 ==="
python3 - <<'EOF'
import glob, json, sys
BX, BY = 84, 50
sides = sorted(glob.glob("train/perception/keybank_holdout2/states/keydrop-*.json"))
bad = 0
dists = []
for s in sides:
    d = json.load(open(s))
    dx, dy = d["drop"][0] - d["link"][0], d["drop"][1] - d["link"][1]
    dists.append(round((dx * dx + dy * dy) ** 0.5))
    if abs(dx) > BX or abs(dy) > BY:
        print("VIOLATION", s, dx, dy); bad += 1
print("holdout2: %d states, dists=%s" % (len(sides), sorted(dists)))
sys.exit(1 if (bad or len(sides) < 12) else 0)
EOF
[ $? -ne 0 ] && { echo "HOLDOUT2 VERIFY FAILED"; exit 5; }

echo "=== holdout2 baselines: s1 x6 seeds, arbiter x3, random x3 ==="
for SEED in 0 100 200 300 400 500; do
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" --states "$H2_STATES" \
    --out "h2_base_s1_s${SEED}.json" || { echo "BASELINE FAILED s$SEED"; exit 6; }
done
for SEED in 0 100 200; do
  $DOCKER python3 -u arbiter_v1.py --arm arbiter --seed "$SEED" --states "$H2_STATES" \
    --out "h2_base_arbiter_s${SEED}.json" || { echo "CEILING FAILED s$SEED"; exit 6; }
  $DOCKER python3 -u arbiter_v1.py --arm random --seed "$SEED" --states "$H2_STATES" \
    --out "h2_base_random_s${SEED}.json" || { echo "RANDOM FAILED s$SEED"; exit 6; }
done

echo "=== HEADROOM GATE (s1 <= 70% AND arbiter-s1 >= +10) ==="
python3 - <<'EOF'
import glob, json, sys
def rate(pat):
    eps = []
    for f in sorted(glob.glob(pat)):
        eps.extend(json.load(open(f))["episodes"])
    return sum(1 for e in eps if e["keys"] > 0) / max(1, len(eps)), len(eps)
s1, n1 = rate("train/rl/h2_base_s1_s*.json")
arb, n2 = rate("train/rl/h2_base_arbiter_s*.json")
rnd, n3 = rate("train/rl/h2_base_random_s*.json")
print("holdout2 baselines: s1=%.0f%% (n=%d) arbiter=%.0f%% (n=%d) random=%.0f%% (n=%d)"
      % (s1 * 100, n1, arb * 100, n2, rnd * 100, n3))
json.dump({"s1": s1, "arbiter": arb, "random": rnd},
          open("train/rl/h2_baselines.json", "w"), indent=2)
sys.exit(0 if (s1 <= 0.70 and arb - s1 >= 0.10) else 7)
EOF
rc=$?
[ $rc -eq 7 ] && { echo "HEADROOM GATE FAILED -- no verdict, bench redesign needed"; exit 7; }
[ $rc -ne 0 ] && exit 8

echo "=== smoke train (10 eps, 1 epoch) ==="
$DOCKER python3 -u distill_bc.py --data distill_data/round1 --limit-eps 10 --epochs 1 \
  --out runs/distill_r1/smoke.zip || { echo "SMOKE TRAIN FAILED"; exit 9; }

echo "=== full train ==="
$DOCKER python3 -u distill_bc.py --data distill_data/round1 \
  --out runs/distill_r1/s1_distilled.zip || { echo "TRAIN FAILED"; exit 10; }

echo "=== eval: distilled S1 solo, holdout2 x6 seeds + holdout1 x6 seeds ==="
for SEED in 0 100 200 300 400 500; do
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r1/s1_distilled.zip --states "$H2_STATES" \
    --out "distill_r1_h2_s${SEED}.json" || { echo "EVAL H2 FAILED s$SEED"; exit 11; }
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r1/s1_distilled.zip --states "$H1_STATES" \
    --out "distill_r1_h1_s${SEED}.json" || { echo "EVAL H1 FAILED s$SEED"; exit 11; }
done

echo "=== holdout1 baseline extension to 6 seeds ==="
for SEED in 300 400 500; do
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" --states "$H1_STATES" \
    --out "arbiter_v2fair_holdout_s1_s${SEED}.json" || { echo "H1 BASE EXT FAILED"; exit 12; }
done

echo "=== secondary: det runs ==="
$DOCKER python3 -u arbiter_v1.py --arm s1 --det --seed 0 \
  --ckpt runs/distill_r1/s1_distilled.zip --states "$H2_STATES" \
  --out distill_r1_h2_det.json || echo "WARN det distilled failed"
$DOCKER python3 -u arbiter_v1.py --arm s1 --det --seed 0 --states "$H2_STATES" \
  --out h2_base_s1_det.json || echo "WARN det base failed"

echo "=== VERDICT (amendment 1 thresholds) ==="
python3 - <<'EOF'
import glob, json
import statistics as st
def pool(pat):
    eps = []
    for f in sorted(glob.glob(pat)):
        eps.extend(json.load(open(f))["episodes"])
    return eps
def rate(eps):
    return sum(1 for e in eps if e["keys"] > 0) / max(1, len(eps))
def med_fk(eps):
    t = [e["first_key_step"] for e in eps if e["first_key_step"] is not None]
    return st.median(t) if t else None
d2 = pool("train/rl/distill_r1_h2_s*.json")
b2 = pool("train/rl/h2_base_s1_s*.json")
c2 = pool("train/rl/h2_base_arbiter_s*.json")
d1 = pool("train/rl/distill_r1_h1_s*.json")
b1 = pool("train/rl/arbiter_v2fair_holdout_s1_s*.json")
delta = (rate(d2) - rate(b2)) * 100
gap = (rate(d2) - rate(b2)) / max(1e-9, rate(c2) - rate(b2))
guard_delta = (rate(d1) - rate(b1)) * 100
guard_ok = guard_delta >= -5
verdict = "SUCCESS" if delta >= 15 else ("PARTIAL" if delta >= 5 else "KILL")
if not guard_ok:
    verdict += "+REGRESSION_GUARD_FAILED"
out = {"h2": {"distilled": rate(d2), "base": rate(b2), "arbiter_ceiling": rate(c2),
              "delta_points": round(delta, 1), "gap_closed": round(gap, 2),
              "med_first_key": {"distilled": med_fk(d2), "base": med_fk(b2)}},
       "h1_guard": {"distilled": rate(d1), "base": rate(b1),
                    "delta_points": round(guard_delta, 1), "ok": guard_ok},
       "det": {"distilled": rate(pool("train/rl/distill_r1_h2_det.json")),
               "base": rate(pool("train/rl/h2_base_s1_det.json"))},
       "verdict": verdict}
print(json.dumps(out, indent=2))
json.dump(out, open("train/rl/distill_r1_verdict.json", "w"), indent=2)
print("VERDICT: %s (h2 %+.1f pts, gap %.0f%%, h1 guard %+.1f)"
      % (verdict, delta, 100 * gap, guard_delta))
EOF
echo "=== DONE ==="
