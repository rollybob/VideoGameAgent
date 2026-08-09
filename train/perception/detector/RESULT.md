# Live-Detection Feasibility Probe -- RESULT (2026-08-08)

Spec: `docs/LIVE_DETECTION_SPEC_2026-08-07.md`. Question: can a small RAM-taught CNN localize
game entities (Link, enemies, items) from 240x160 GBA pixels at frame-rate AND generalize to
HELD-OUT rooms? Pre-registered SUCCESS = >=100 FPS forward (>=60 end-to-end) AND held-out
Link>=90% / enemy>=80% within 4px.

## Verdict: FEASIBILITY PROVEN; Link gate missed (data-limited). Qualified success / PARTIAL.

Fast RAM-taught perception works decisively. Speed is a non-issue by ~20x; enemy and item
generalize to unseen rooms and clear their bars. Link localization misses 90 in every config
and is a diagnosed, DATA-limited gap that compute alone cannot close.

## Numbers (held-out rooms {2,4}; honest eval = multi-peak nearest-match, ALL frames)

| @4px            | baseline 0.11M | v2 0.62M+aug | v2b 0.62M no-aug | gate         |
|-----------------|---------------:|-------------:|-----------------:|--------------|
| Link            |           82.7 |         85.4 |             84.8 | >=90  MISS   |
| Enemy           |           91.4 |         79.8 |             86.2 | >=80  (base/v2b pass, v2 fail) |
| Item (keydrop)  |          100.0 |        100.0 |            100.0 | bonus pass   |
| FPS forward     |           2267 |         1059 |             1063 | >=100 PASS   |
| FPS end-to-end  |           1400 |          809 |              815 | >=60  PASS   |

## Findings
- SPEED settled: even the bigger net is ~1000 FPS forward / ~800 e2e (bars 100/60). Non-issue.
- ENEMY + ITEM generalize to unseen rooms and clear their bars (item on real resting keys = 100%).
- LINK misses 90 everywhere. Diagnosed: NOT bias (~1px), NOT edges (edge >= central); it is a
  CAPACITY/ROOM-GENERALIZATION gap (train-room 92 -> held 83). Link's sprite animates (walk/sword),
  so his visual centroid wobbles off the fixed RAM anchor, and 3 train rooms is too few to generalize.
- COMPUTE-ONLY IS A SEE-SAW (Pareto front), cleanly attributed by the v2b ablation:
  bigger net = +2 Link / -5 enemy; color aug (on top) = +0.6 Link / -6 enemy. No compute config
  clears Link 90 without dropping enemy. => the binding lever is DATA (room diversity), not model
  size. Same shape as the rung-3 collect see-saw: too few rooms forces the net to trade classes.
- ITEM label caveat (`kill_drop_forensic.py` + `_killdrop_strip.png`): the HP==0-in-slot rule is
  temporally EAGER. On an enemy kill the slot goes HP->0 immediately at a stable on-screen pos, but
  the KEY sprite does not render until ~frame 12 (frames 0..~10 = death poof). Real resting keys
  detect at 100%; the fresh-drop "0%" is a label artifact, not a detector failure. Fix = a ~15-frame
  HP==0-persistence gate. Pots + bushes/grass still UNMAPPED (may use a different object table).

## Operating model
Keep **baseline `detector.pt`** (widest enemy margin 91.4 + 2x speed). `detector_v2b.pt` is the
max-Link alternative (Link 84.8, enemy still passes at 86.2). Both miss Link 90; the choice is
minor until the data-driven retrain. `detector_v2.pt` (aug) is dominated -- do NOT adopt (color
aug hurts enemy).

## Next (all gated on more DATA = Tim's room captures)
1. Close Link via ROOM DIVERSITY: add rooms {5,6} + fresh dungeon rooms; retrain WITHOUT the
   enemy-hurting color aug. Expectation: lifts Link AND enemy together (the see-saw is a data symptom).
2. Fix the item labeler (persistence gate) + map pots/bushes -> a clean item class from all sources.
3. WIRE detector positions -> S1 as target-relative input = the actual rung-3 collect fix (the point
   of the pillar). Baseline detector is already good enough (item 100 / enemy 91) to start.

## Deliverables / artifacts (all under train/perception/detector/)
Code: gen_data.py, gen_items.py, train_detector.py (baseline), train_detector_v2.py (v2/v2b via
AUG/TAG env), eval_detector.py (honest harness), kill_drop_forensic.py.
Models: detector.pt, detector_v2.pt, detector_v2b.pt. Results: _eval_result.json, _eval_v2_result.json,
_eval_v2b_result.json, _killdrop_strip.png, _label_check.png. (Dir is still git-untracked.)
