#!/bin/bash
# Variant D per spec: mixture-weighted joint train (no new harvest), then close.
set -u
cd /home/timothy/projects/VGA
DOCKER="docker run --rm --runtime nvidia --user 1000:1000 \
  -v /home/timothy/projects/VGA:/vga -w /vga/train/rl thor-rl:cu130"
H2='/vga/train/perception/keybank_holdout2/states/keydrop-*.state'
H1='/vga/train/perception/keybank_holdout/states/keydrop-*.state'

echo "=== joint train from base, data-weights 1.0/1.5/1.0 ==="
$DOCKER python3 -u distill_bc.py \
  --data distill_data/round1 distill_data/round2 distill_data/round2far \
  --data-weights 1.0 1.5 1.0 \
  --ckpt runs/alttp_s1v2_scratch04/ppo_alttp_600000_steps.zip \
  --out runs/distill_r2/s1_joint_r2d.zip || { echo "TRAIN FAILED"; exit 5; }

echo "=== primary: chain eval n=60 ==="
$DOCKER python3 -u chain_probe.py --episodes 60 \
  --models r2d=runs/distill_r2/s1_joint_r2d.zip \
  --out chain_probe_r2d.json || { echo "CHAIN EVAL FAILED"; exit 6; }

echo "=== far guard (6 seeds, both banks) ==="
for SEED in 0 100 200 300 400 500; do
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r2/s1_joint_r2d.zip --states "$H2" \
    --out "distill_r2d_h2_s${SEED}.json" || { echo "GUARD FAILED"; exit 7; }
  $DOCKER python3 -u arbiter_v1.py --arm s1 --seed "$SEED" \
    --ckpt runs/distill_r2/s1_joint_r2d.zip --states "$H1" \
    --out "distill_r2d_h1_s${SEED}.json" || { echo "GUARD FAILED"; exit 7; }
done

echo "=== retention (n=6) ==="
$DOCKER python3 -u eval_ppo_md.py \
  --models r2d=runs/distill_r2/s1_joint_r2d.zip \
  --states alttp_human-04 --episodes 6 --horizon 15000 || echo "WARN retention failed"

echo "=== VERDICT ==="
python3 - <<'PYEOF'
import glob, json
probe = json.load(open("train/rl/chain_probe_r2d.json"))
eps = probe["r2d"]
n = len(eps)
cc = 100.0 * sum(1 for e in eps if e["first_kill04_step"] is not None and e["collected_after_kill"]) / n
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
for pat, keep in [("train/rl/distill_r2d_h2_s*.json", F2), ("train/rl/distill_r2d_h1_s*.json", F1)]:
    for f in sorted(glob.glob(pat)):
        far_eps.extend(e for e in json.load(open(f))["episodes"] if e["state"] in keep)
far = 100.0 * sum(1 for e in far_eps if e["keys"] > 0) / max(1, len(far_eps))
kills = sum(1 for e in eps if e["first_kill04_step"] is not None)
cgk = 100.0 * sum(1 for e in eps if e.get("collected_after_kill")) / max(1, kills)
verdict = "SUCCESS" if cc >= 45 else ("PARTIAL" if cc >= 25 else "KILL")
if far < 85: verdict += "+FAR_GUARD_FAILED"
out = {"chain_complete_pct": round(cc, 1), "eps_with_kill": kills,
       "collect_given_kill_pct": round(cgk, 1),
       "far_guard_pct": round(far, 1), "n_far": len(far_eps), "verdict": verdict}
print(json.dumps(out, indent=2))
json.dump(out, open("train/rl/distill_r2d_verdict.json", "w"), indent=2)
print("VERDICT: %s (chain %.0f%% far %.0f%%)" % (verdict, cc, far))
PYEOF
echo "=== DONE ==="
