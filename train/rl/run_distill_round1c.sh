#!/bin/bash
# Distill round 1 under AMENDMENT 2 (far-holdout bench). Launch ONLY on Tim's go.
# Banks + baselines already exist; this job: h1 baseline extension -> smoke/full
# train -> distilled evals on both holdouts -> far/near verdict.
set -u
cd /home/timothy/projects/VGA

DOCKER="docker run --rm --runtime nvidia --user 1000:1000 \
  -v /home/timothy/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130"
H2_STATES='/vga/train/perception/keybank_holdout2/states/keydrop-*.state'
H1_STATES='/vga/train/perception/keybank_holdout/states/keydrop-*.state'

echo "=== holdout1 s1 baseline extension (seeds 300-500) ==="
for SEED in 300 400 500; do
  [ -f "train/rl/arbiter_v2fair_holdout_s1_s${SEED}.json" ] && continue
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" --states "$H1_STATES" \
    --out "arbiter_v2fair_holdout_s1_s${SEED}.json" || { echo "H1 BASE EXT FAILED"; exit 4; }
done

echo "=== smoke train (10 eps, 1 epoch) ==="
$DOCKER python3 -u distill_bc.py --data distill_data/round1 --limit-eps 10 --epochs 1 \
  --out runs/distill_r1/smoke.zip || { echo "SMOKE TRAIN FAILED"; exit 5; }

echo "=== full train ==="
$DOCKER python3 -u distill_bc.py --data distill_data/round1 \
  --out runs/distill_r1/s1_distilled.zip || { echo "TRAIN FAILED"; exit 6; }

echo "=== distilled S1 solo: both holdouts x6 seeds ==="
for SEED in 0 100 200 300 400 500; do
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r1/s1_distilled.zip --states "$H2_STATES" \
    --out "distill_r1_h2_s${SEED}.json" || { echo "EVAL H2 FAILED s$SEED"; exit 7; }
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r1/s1_distilled.zip --states "$H1_STATES" \
    --out "distill_r1_h1_s${SEED}.json" || { echo "EVAL H1 FAILED s$SEED"; exit 7; }
done

echo "=== secondary: det runs ==="
$DOCKER python3 -u arbiter_v1.py --arm s1 --det --seed 0 \
  --ckpt runs/distill_r1/s1_distilled.zip --states "$H2_STATES" \
  --out distill_r1_h2_det.json || echo "WARN det distilled failed"

echo "=== VERDICT (amendment 2: far-holdout primary) ==="
python3 - <<'EOF'
import glob, json
import statistics as st

def far_near(bank):
    far, near = set(), set()
    for s in sorted(glob.glob("train/perception/%s/states/keydrop-*.json" % bank)):
        j = json.load(open(s))
        dx, dy = j["drop"][0] - j["link"][0], j["drop"][1] - j["link"][1]
        name = s.split("/")[-1].replace(".json", ".state")
        (far if (dx * dx + dy * dy) ** 0.5 >= 30 else near).add(name)
    return far, near

F2, N2 = far_near("keybank_holdout2")
F1, N1 = far_near("keybank_holdout")

def pool(pats_keeps):
    eps = []
    for pat, keep in pats_keeps:
        for f in sorted(glob.glob(pat)):
            eps.extend(e for e in json.load(open(f))["episodes"] if e["state"] in keep)
    return eps

def stats(eps):
    fks = [e["first_key_step"] for e in eps if e["first_key_step"] is not None]
    return {"n": len(eps),
            "collect": sum(1 for e in eps if e["keys"] > 0) / max(1, len(eps)),
            "med_fk": st.median(fks) if fks else None,
            "at150": sum(1 for t in fks if t <= 150) / max(1, len(eps))}

d_far = stats(pool([("train/rl/distill_r1_h2_s*.json", F2), ("train/rl/distill_r1_h1_s*.json", F1)]))
b_far = stats(pool([("train/rl/h2_base_s1_s*.json", F2), ("train/rl/arbiter_v2fair_holdout_s1_s*.json", F1)]))
c_far = stats(pool([("train/rl/h2_base_arbiter_s*.json", F2), ("train/rl/arbiter_v2fair_holdout_arbiter_s*.json", F1)]))
d_near = stats(pool([("train/rl/distill_r1_h2_s*.json", N2), ("train/rl/distill_r1_h1_s*.json", N1)]))
b_near = stats(pool([("train/rl/h2_base_s1_s*.json", N2), ("train/rl/arbiter_v2fair_holdout_s1_s*.json", N1)]))

delta = (d_far["collect"] - b_far["collect"]) * 100
gap = (d_far["collect"] - b_far["collect"]) / max(1e-9, c_far["collect"] - b_far["collect"])
guard = (d_near["collect"] - b_near["collect"]) * 100
verdict = "SUCCESS" if delta >= 8 else ("PARTIAL" if delta >= 4 else "KILL")
if guard < -5:
    verdict += "+NEAR_REGRESSION"
out = {"far": {"distilled": d_far, "base": b_far, "arbiter_ceiling": c_far,
               "delta_points": round(delta, 1), "gap_closed": round(gap, 2)},
       "near_guard": {"distilled": d_near, "base": b_near, "delta_points": round(guard, 1)},
       "det_far_distilled": stats(pool([("train/rl/distill_r1_h2_det.json", F2)]))["collect"],
       "verdict": verdict}
print(json.dumps(out, indent=2))
json.dump(out, open("train/rl/distill_r1_verdict.json", "w"), indent=2)
print("VERDICT: %s (far %+.1f pts, gap %.0f%% closed, near guard %+.1f)"
      % (verdict, delta, 100 * gap, guard))
EOF
echo "=== DONE ==="
