#!/bin/bash
# Distill round 2 per docs/DISTILL_R2_SPEC_2026-08-07.md (Tim's go, 2026-08-07 am).
# harvest (gate >=250) -> smoke/full train (round1+round2 replay) -> chain eval
# n=60 (primary) -> far-holdout guard -> retention guard -> verdict.
set -u
cd /home/timothy/projects/VGA

DOCKER="docker run --rm --runtime nvidia --user 1000:1000 \
  -v /home/timothy/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130"
H2='/vga/train/perception/keybank_holdout2/states/keydrop-*.state'
H1='/vga/train/perception/keybank_holdout/states/keydrop-*.state'

echo "=== chain harvest (target 350, cap 420) ==="
$DOCKER python3 -u chain_harvest.py --target 350 --episodes 420 --seed 3000 \
  --out distill_data/round2 || { echo "HARVEST GATE FAILED (<250 successes)"; exit 4; }

echo "=== smoke train (10 eps, 1 epoch, mixed dirs) ==="
$DOCKER python3 -u distill_bc.py --data distill_data/round1 distill_data/round2 \
  --limit-eps 10 --epochs 1 --out runs/distill_r2/smoke.zip \
  || { echo "SMOKE TRAIN FAILED"; exit 5; }

echo "=== full train (round1 replay + round2, on top of r1 ckpt) ==="
$DOCKER python3 -u distill_bc.py --data distill_data/round1 distill_data/round2 \
  --ckpt runs/distill_r1/s1_distilled.zip --out runs/distill_r2/s1_distilled_r2.zip \
  || { echo "TRAIN FAILED"; exit 6; }

echo "=== primary: chain eval n=60 (r2 + concurrent r1) ==="
$DOCKER python3 -u chain_probe.py --episodes 60 \
  --models r2=runs/distill_r2/s1_distilled_r2.zip r1=runs/distill_r1/s1_distilled.zip \
  --out chain_probe_r2.json || { echo "CHAIN EVAL FAILED"; exit 7; }

echo "=== guard: far-holdout collect (6 seeds, both banks) ==="
for SEED in 0 100 200 300 400 500; do
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r2/s1_distilled_r2.zip --states "$H2" \
    --out "distill_r2_h2_s${SEED}.json" || { echo "GUARD EVAL FAILED"; exit 8; }
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r2/s1_distilled_r2.zip --states "$H1" \
    --out "distill_r2_h1_s${SEED}.json" || { echo "GUARD EVAL FAILED"; exit 8; }
done

echo "=== guard: standard-ep retention (n=6) ==="
$DOCKER python3 -u eval_ppo_md.py \
  --models r1=runs/distill_r1/s1_distilled.zip r2=runs/distill_r2/s1_distilled_r2.zip \
  --states alttp_human-04 --episodes 6 --horizon 15000 \
  || echo "WARN: retention eval failed (non-blocking, report manually)"

echo "=== VERDICT (spec thresholds: SUCCESS>=45, PARTIAL>=25, KILL<25; far guard >=85) ==="
python3 - <<'EOF'
import glob, json
import statistics as st

probe = json.load(open("train/rl/chain_probe_r2.json"))
def chain_stats(eps):
    n = len(eps)
    comp = sum(1 for e in eps if e["first_kill04_step"] is not None and e["collected_after_kill"])
    kills = sum(1 for e in eps if e["first_kill04_step"] is not None)
    cgk = sum(1 for e in eps if e.get("collected_after_kill")) / max(1, kills)
    return {"n": n, "chain_complete": comp / n, "eps_with_kill": kills / n,
            "collect_given_kill": cgk,
            "keys_per_ep": sum(e["keys"] for e in eps) / n}
r2, r1 = chain_stats(probe["r2"]), chain_stats(probe["r1"])

def far_states(bank):
    out = set()
    for s in glob.glob("train/perception/%s/states/keydrop-*.json" % bank):
        j = json.load(open(s))
        dx, dy = j["drop"][0] - j["link"][0], j["drop"][1] - j["link"][1]
        if (dx * dx + dy * dy) ** 0.5 >= 30:
            out.add(s.split("/")[-1].replace(".json", ".state"))
    return out
F2, F1 = far_states("keybank_holdout2"), far_states("keybank_holdout")
far_eps = []
for pat, keep in [("train/rl/distill_r2_h2_s*.json", F2), ("train/rl/distill_r2_h1_s*.json", F1)]:
    for f in sorted(glob.glob(pat)):
        far_eps.extend(e for e in json.load(open(f))["episodes"] if e["state"] in keep)
far_rate = sum(1 for e in far_eps if e["keys"] > 0) / max(1, len(far_eps))

cc = r2["chain_complete"] * 100
verdict = "SUCCESS" if cc >= 45 else ("PARTIAL" if cc >= 25 else "KILL")
if far_rate < 0.85:
    verdict += "+FAR_GUARD_FAILED"
out = {"r2_chain": r2, "r1_chain_concurrent": r1,
       "far_guard": {"rate": far_rate, "n": len(far_eps), "ok": far_rate >= 0.85},
       "verdict": verdict}
print(json.dumps(out, indent=2))
json.dump(out, open("train/rl/distill_r2_verdict.json", "w"), indent=2)
print("VERDICT: %s (chain %.0f%% vs baseline ~13%%; far guard %.0f%%)" % (verdict, cc, 100 * far_rate))
EOF
echo "=== DONE ==="
