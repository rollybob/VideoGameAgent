"""
Offline retrieval-hit metric for Task 02 (semantic vs exact-match skill retrieval).

Replays logged trajectories WITHOUT the VLM or the GPU: for every logged step it
reconstructs the SAME retrieval context the live loop built - the reported `mode`
(from the log) plus the `dialog_text` re-derived by OCR'ing the saved frame's bottom
strip exactly as VlmPlugin._read_dialog does - then asks the SkillStore for skills two
ways:

  EXACT   - the legacy hard match on `mode` against the fixed MODES whitelist
            (VGA_SKILLS_EXACT=1), i.e. what main did before Task 02.
  SEMANTIC- the new overlap ranking against the free-text context.

It reports how often each returns nothing (K=0), how many frames FLIP from empty
(exact) to a hit (semantic) - the cliff being removed - and, on frames exact already
served, whether semantic keeps the same skills (parity, no regression). This isolates
the mechanism change on real data BEFORE spending any GPU on a ladder run.

Run (CPU only; ~19k tesseract calls -> a few minutes, use thor-job):
  PYTHONPATH=. python3 train/ram/eval_skill_retrieval.py --out sessions/<dir>/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2

from vga.reason.plugin import _DIALOG_REGION_TOP, _norm_text, _DEDUP_MIN_ALNUM
from vga.reason.skills import SkillStore

# The legacy exact-match whitelist (kept only for reporting the "exact-servable" split).
_SEEDED_MODES = {"menu", "dialog", "overworld", "battle"}  # modes that have seed skills


def reconstruct_dialog(ocr, frame_path: Path) -> str:
    """Re-derive dialog_text the way VlmPlugin._read_dialog does: OCR the bottom strip,
    and treat a read with too few alnum chars as empty (no meaningful dialog signature)."""
    img = cv2.imread(str(frame_path))  # BGR, matching the live mss capture Ocr.read expects
    if img is None or img.size == 0:
        return ""
    h = img.shape[0]
    region = img[int(h * _DIALOG_REGION_TOP):, :]
    text = ocr.read(region)
    if sum(c.isalnum() for c in _norm_text(text)) < _DEDUP_MIN_ALNUM:
        return ""
    return text


def retrieve(store: SkillStore, context: str, exact: bool) -> list[str]:
    """Side-effect-free retrieval: reset use counts so ranking does not drift across the
    replay, and toggle the exact/semantic env for this one call."""
    for s in store.all():
        s.uses = 0
    key = "VGA_SKILLS_EXACT"
    prev = os.environ.get(key)
    if exact:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        return store.retrieve(context=context)
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="sessions", help="root to scan for steps.jsonl")
    ap.add_argument("--out", default="", help="write JSON report here")
    ap.add_argument("--limit-frames", type=int, default=0, help="cap frames (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vga.plugins.pokemon.ocr import Ocr
    ocr = Ocr(logger=lambda *_: None)
    if not getattr(ocr, "enabled", False):
        raise SystemExit("OCR (tesseract) not available; cannot reconstruct dialog_text.")

    # Scratch store: fresh seeds, never persisted over the committed skills.json.
    scratch = Path("/tmp/t02_scratch_skills.json")
    if scratch.exists():
        scratch.unlink()
    store = SkillStore(scratch)

    steps_files = sorted(Path(args.sessions).rglob("steps.jsonl"))
    rows = []
    for sf in steps_files:
        base = sf.parent
        with sf.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fp = d.get("frame")
                if not fp:
                    continue
                rows.append((base / fp, (d.get("mode") or "").strip().lower()))
    if args.limit_frames and len(rows) > args.limit_frames:
        random.seed(args.seed)
        rows = random.sample(rows, args.limit_frames)

    # Counters.
    n = 0
    mode_hist: dict[str, int] = {}
    exact_k0 = semantic_k0 = 0
    recovered = regressed = 0            # exact []->semantic hit ; exact hit->semantic []
    parity_ok = parity_tot = 0           # on exact-hit frames, exact skills subset of semantic
    text_frames = 0                      # frames with reconstructed dialog_text
    recovered_with_text = 0
    # Split of the recovery by why exact failed.
    cat = {"unknown_or_empty": [0, 0], "unseeded_known": [0, 0], "seeded": [0, 0]}  # [total, recovered]
    samples = []

    for fp, mode in rows:
        if not fp.exists():
            continue
        n += 1
        mode_hist[mode or "<empty>"] = mode_hist.get(mode or "<empty>", 0) + 1
        dialog = reconstruct_dialog(ocr, fp)
        has_text = bool(dialog)
        text_frames += has_text
        context = f"{mode} {dialog}".strip()
        ex = retrieve(store, context, exact=True)
        se = retrieve(store, context, exact=False)
        exact_k0 += (len(ex) == 0)
        semantic_k0 += (len(se) == 0)
        if not ex and se:
            recovered += 1
            recovered_with_text += has_text
        if ex and not se:
            regressed += 1
        if ex:
            parity_tot += 1
            parity_ok += set(ex).issubset(set(se))
        # Categorize why exact failed (for the recovery breakdown).
        if not ex:
            if mode in ("", "unknown"):
                key = "unknown_or_empty"
            elif mode in _SEEDED_MODES:
                key = "seeded"       # exact should have served but didn't (shouldn't happen)
            else:
                key = "unseeded_known"  # a real mode with no seed skill (shop/title/cutscene/vocab)
            cat[key][0] += 1
            if se:
                cat[key][1] += 1
                if len(samples) < 25:
                    samples.append({"mode": mode, "dialog": dialog[:60],
                                    "skill": se[0][:70] if se else ""})

    def pct(a, b):
        return round(100.0 * a / b, 1) if b else 0.0

    report = {
        "n_frames": n,
        "n_with_dialog_text": text_frames,
        "mode_hist": dict(sorted(mode_hist.items(), key=lambda kv: -kv[1])),
        "exact_empty_rate_pct": pct(exact_k0, n),
        "semantic_empty_rate_pct": pct(semantic_k0, n),
        "recovered_frames": recovered,           # exact [] -> semantic hit (cliff removed)
        "recovered_rate_pct": pct(recovered, n),
        "recovered_with_text": recovered_with_text,
        "regressed_frames": regressed,           # exact hit -> semantic [] (want 0)
        "parity_on_exact_hits_pct": pct(parity_ok, parity_tot),
        "parity_denominator": parity_tot,
        "recovery_by_reason": {k: {"exact_empty": v[0], "recovered": v[1],
                                    "recovered_pct": pct(v[1], v[0])} for k, v in cat.items()},
        "samples": samples,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
