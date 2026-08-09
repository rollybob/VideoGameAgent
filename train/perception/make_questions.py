"""Turn oracle labels into scored VQA questions. Host-run (stdlib only).

Scored question types (auto-graded vs the RAM oracle):
  hearts  -- primary. Clean, high-variety (0.5..3.0), knowledge-neutral HUD read.
  rupees  -- secondary. Low variety in this set but pixel-unambiguous.
enemy_present is intentionally NOT a question: the raw 16-slot read false-fires on
~86% of frames (object/garbage slots), so it is not a trustworthy label.
"""
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
BD = os.path.join(D, "bench_data")

qs = []
with open(os.path.join(BD, "labels.jsonl")) as f:
    for line in f:
        L = json.loads(line)
        qs.append(dict(id=L["id"], file=L["file"], type="hearts",
                       question=("How many hearts of health does Link have right now? "
                                 "Count each full heart as 1 and each half heart as 0.5. "
                                 "Reply with only the number."),
                       answer=L["hearts"], tol=0.5))
        qs.append(dict(id=L["id"], file=L["file"], type="rupees",
                       question="What number is shown on the rupee counter? Reply with only the number.",
                       answer=L["rupees"], tol=0))

with open(os.path.join(BD, "questions.jsonl"), "w") as f:
    for q in qs:
        f.write(json.dumps(q) + "\n")

n_h = sum(1 for q in qs if q["type"] == "hearts")
n_r = sum(1 for q in qs if q["type"] == "rupees")
print("wrote %d questions (%d hearts + %d rupees)" % (len(qs), n_h, n_r))
