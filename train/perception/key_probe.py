#!/usr/bin/env python3
"""Decision-gate probe: can the served VLM SEE a dropped dungeon key in a raw GBA frame,
and give a usable direction from Link to it?

This is the load-bearing question for the arbiter's key interject (2026-08-06 reframe:
kill->collect-key is System-2's job). Probe BEFORE building: if the VLM cannot see the
key, the interject design changes (RAM-taught key detector as System-1 perception) and
we want to know that for the cost of ~100 /read calls, not a built-and-failed arbiter.

Frames + labels come from train/rl/harvest_key_frames.py (RAM-oracle-labeled, no hand
labeling). Presence is scored on key_onscreen (positives) vs control (negatives);
direction is scored exact-or-adjacent (within 45 degrees) on positives only. Both
native and 3x-upscale arms: the bake-off says upscale rescues tiny GLYPHS for the 30B;
whether it rescues a tiny key SPRITE for the 8B is exactly what we measure.

Usage (server already healthy on :8077):
    .venv/bin/python train/perception/key_probe.py [--data train/perception/keyprobe_data]
"""
import argparse
import base64
import io
import json
import os
import random
import re
import time
import urllib.request

from PIL import Image

URL = "http://127.0.0.1:8077"
DIRS = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]

PROMPT = (
    "This is a frame from The Legend of Zelda: A Link to the Past on Game Boy Advance. "
    "Link is the small character in the green tunic. QUESTION: is there a small dungeon "
    "KEY item lying on the floor anywhere in this frame? A dropped key is a small "
    "golden/yellow key-shaped sprite lying on the ground (not in the HUD at the top). "
    "Answer with ONLY a JSON object and no other text, exactly like "
    '{"key_visible": true, "direction": "up-left"} . '
    '"direction" is where the key is RELATIVE TO LINK, one of: up, down, left, right, '
    'up-left, up-right, down-left, down-right, "on-link" if Link is standing on it, '
    'or "none" if there is no key.'
)


def b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post(path, payload, timeout=180):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_answer(text):
    """Defensive parse: first {...} blob wins; fall back to keyword scan."""
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return bool(d.get("key_visible")), str(d.get("direction") or "none").lower()
        except Exception:
            pass
    low = text.lower()
    vis = "true" in low or ("yes" in low and "no key" not in low)
    for cand in ["up-left", "up-right", "down-left", "down-right", "on-link",
                 "up", "down", "left", "right"]:
        if cand in low:
            return vis, cand
    return vis, "none"


def dir_ok(pred, truth):
    """Exact or adjacent 45-degree bucket counts as usable steering."""
    if pred == truth:
        return True
    if truth == "on-link" or pred in ("on-link", "none") or truth not in DIRS or pred not in DIRS:
        return False
    return min((DIRS.index(pred) - DIRS.index(truth)) % 8,
               (DIRS.index(truth) - DIRS.index(pred)) % 8) <= 1


def dist_bucket(d):
    return "near<40" if d < 40 else ("mid40-80" if d < 80 else "far>80")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/perception/keyprobe_data")
    ap.add_argument("--n-pos", type=int, default=30)
    ap.add_argument("--n-neg", type=int, default=15)
    ap.add_argument("--upscales", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="train/perception/key_probe_results.json")
    a = ap.parse_args()

    with open(os.path.join(a.data, "labels.jsonl")) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    pos = [r for r in recs if r["phase"] == "key_onscreen"]
    neg = [r for r in recs if r["phase"] == "control"]

    # balanced sample: spread positives across distance buckets and directions
    rng = random.Random(a.seed)
    by_bucket = {}
    for r in pos:
        by_bucket.setdefault((dist_bucket(r["dist"]), r["direction"]), []).append(r)
    sample_pos = []
    cells = list(by_bucket.values())
    for c in cells:
        rng.shuffle(c)
    ci = 0
    while len(sample_pos) < min(a.n_pos, len(pos)) and any(cells):
        cell = cells[ci % len(cells)]
        if cell:
            sample_pos.append(cell.pop())
        ci += 1
    sample_neg = rng.sample(neg, min(a.n_neg, len(neg)))
    print("probe set: %d positives (%d available), %d negatives (%d available)"
          % (len(sample_pos), len(pos), len(sample_neg), len(neg)))

    health = json.load(urllib.request.urlopen(URL + "/health", timeout=30))
    print("HEALTH:", json.dumps(health))

    results = []
    for up in a.upscales:
        t0 = time.time()
        for r in sample_pos + sample_neg:
            img = Image.open(os.path.join(a.data, "frames", r["file"])).convert("RGB")
            if up != 1:
                img = img.resize((img.width * up, img.height * up), Image.LANCZOS)
            try:
                rd = post("/read", {"image_b64": b64(img), "prompt": PROMPT})
                vis, direc = parse_answer(rd.get("text") or "")
                err = None
            except Exception as e:
                vis, direc, err = None, None, repr(e)[:150]
            results.append({"file": r["file"], "upscale": up, "phase": r["phase"],
                            "truth_dir": r.get("direction"), "dist": r.get("dist"),
                            "pred_vis": vis, "pred_dir": direc, "err": err,
                            "raw": (rd.get("text") or "")[:200] if not err else None})
        dt = time.time() - t0
        n = len(sample_pos) + len(sample_neg)
        print("arm up%d done: %d calls in %.0fs (%.1fs/call)" % (up, n, dt, dt / max(1, n)))

        # score this arm
        arm = [x for x in results if x["upscale"] == up and not x["err"]]
        p = [x for x in arm if x["phase"] == "key_onscreen"]
        ng = [x for x in arm if x["phase"] == "control"]
        hit = sum(1 for x in p if x["pred_vis"])
        fa = sum(1 for x in ng if x["pred_vis"])
        du = sum(1 for x in p if x["pred_vis"] and dir_ok(x["pred_dir"], x["truth_dir"]))
        print("  presence: %d/%d hits (%.0f%%) | false alarms %d/%d (%.0f%%)"
              % (hit, len(p), 100.0 * hit / max(1, len(p)),
                 fa, len(ng), 100.0 * fa / max(1, len(ng))))
        print("  direction usable (exact/adjacent, among detected): %d/%d (%.0f%%)"
              % (du, max(1, hit), 100.0 * du / max(1, hit)))
        for b in ["near<40", "mid40-80", "far>80"]:
            pb = [x for x in p if dist_bucket(x["dist"]) == b]
            hb = sum(1 for x in pb if x["pred_vis"])
            print("    %-9s presence %d/%d" % (b, hb, len(pb)))

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("SAVED", a.out)


if __name__ == "__main__":
    main()
