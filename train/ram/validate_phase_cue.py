#!/usr/bin/env python3
"""Task 07 offline cue validation (CPU-only, no GPU).

Replays each episode's SAVED frames through the SAME inference path the live phase
machine uses -- vga.reason.plugin._advance_phase's core: Ocr.read(frame) -> _norm_text
-> _fuzzy_contains(text, cue) -- and compares the resulting latch step against the RAM
ground-truth accept (first clan_funds drop below the pub baseline). The RAM funds are the
TRAIN-time teacher only; the cue itself is pixels-only.

Reports per episode: the cue-fire step, the RAM-accept step, and whether any cue fires on
the pre-accept pub screens (a false latch, which would render the "already accepted"
directive before the accept commits -- the failure mode the design must avoid).

Usage: .venv/bin/python train/ram/validate_phase_cue.py <ep_dir> [<ep_dir> ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vga.reason.plugin import _norm_text, _fuzzy_contains, TASK_PHASE_SPECS  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None

CUE = TASK_PHASE_SPECS["herb"]["phases"]["accept"]["cues"]  # ("for the info",)


def load_ocr():
    from vga.plugins.pokemon.ocr import Ocr
    ocr = Ocr(logger=lambda *_: None)
    if not getattr(ocr, "enabled", False):
        raise SystemExit("OCR (tesseract) not available -- run under the host venv")
    return ocr


def cue_fires(ocr, frame_bgr):
    """Exactly _advance_phase's per-frame test for the 'accept' phase cues."""
    text = _norm_text(ocr.read(frame_bgr))
    hit = next((sig for sig in CUE if _fuzzy_contains(text, sig)), None)
    return hit, text


def validate(ep_dir, ocr):
    recs = [json.loads(l) for l in open(os.path.join(ep_dir, "steps.jsonl"))]
    base = None
    ram_accept = None
    for r in recs:
        f = (r.get("state") or {}).get("clan_funds")
        if f is None:
            continue
        if base is None:
            base = f
        if ram_accept is None and f < base:
            ram_accept = r["step"]

    first_fire = None
    fires = []
    for r in recs:
        fp = os.path.join(ep_dir, r["frame"])
        img = cv2.imread(fp)
        if img is None:
            continue
        hit, text = cue_fires(ocr, img)
        if hit:
            fires.append(r["step"])
            if first_fire is None:
                first_fire = r["step"]

    # The charge dialog ("That'll be 300 gil") shows one decide-step BEFORE the RAM funds tick,
    # so a fire at ram_accept-1 is the expected, post-commit latch (the confirm box / Yes-press is
    # a further step earlier). Only a fire >=2 steps before the RAM accept is genuinely suspicious
    # (it could precede the Yes-press). That is the real safety boundary the design cares about.
    suspicious_early = [s for s in fires if ram_accept is not None and s < ram_accept - 1]
    name = ep_dir.replace("sessions/", "").rstrip("/")
    print(f"{name}")
    print(f"  RAM accept step (first funds drop): {ram_accept}")
    print(f"  cue first-fire step               : {first_fire}   all fires: {fires}")
    if ram_accept is not None and first_fire is not None:
        lag = first_fire - ram_accept
        aligned = -2 <= lag <= 2
        print(f"  latch-vs-accept lag               : {lag:+d}  {'ALIGNED' if aligned else 'OFF'}")
    print(f"  suspicious early fires (>=2 early) : {suspicious_early if suspicious_early else 'NONE (clean)'}")
    print()
    return {
        "ep": name, "ram_accept": ram_accept, "first_fire": first_fire,
        "fires": fires, "suspicious_early": suspicious_early,
    }


def main():
    if cv2 is None:
        raise SystemExit("cv2 not available -- run under the host venv (.venv)")
    eps = sys.argv[1:]
    if not eps:
        raise SystemExit(__doc__)
    ocr = load_ocr()
    results = [validate(e, ocr) for e in eps]
    # Summary verdict
    accepted = [r for r in results if r["ram_accept"] is not None]
    latched = [r for r in accepted if r["first_fire"] is not None]
    clean = [r for r in accepted if not r["suspicious_early"]]
    print("=" * 60)
    print(f"episodes with a RAM accept        : {len(accepted)}/{len(results)}")
    print(f"  of those, cue latched           : {len(latched)}/{len(accepted)}")
    print(f"  of those, no >=2-step-early fire : {len(clean)}/{len(accepted)}")


if __name__ == "__main__":
    main()
