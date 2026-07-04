"""
Build a behavior-cloning dataset (.npz) from logged VGA trajectories.

Phase 2 (System-1 distillation): turn the (frame_before -> chosen button) pairs that
run_reasoner.py / teacher.py logged into a compact tensor dataset a small conv policy can
be trained on in the thor-torch container. Mirrors the OCR pipeline's split of labor:
the HOST venv (cv2 + numpy) decodes/resizes PNGs and writes .npz; the CONTAINER (torch)
trains reading only numpy, so it needs no image codec.

Label = the button the reasoner ACTUALLY chose, read from each step's meta.button (that
field includes "wait", whereas action.button is null for a wait). Labels map through the
canonical vga.reason.base.BUTTON_CHOICES (11 classes) so train==inference action space.

Val split is a contiguous TAIL slice per trajectory (not random): frames within a
trajectory are highly correlated (adjacent frames are near-duplicates), so a random split
would leak almost-identical frames across train/val and inflate the metric. A temporal
tail leaks at most the one boundary pair. Once we have many trajectories, prefer holding
out WHOLE trajectories (--val-by-traj) for a cleaner estimate.

Host venv (cv2 + numpy). Run e.g.:
  .venv/bin/python train/policy/build_dataset.py \
      --traj sessions/traj-1782937436 sessions/traj-1782940464 \
      --out train/policy/data --img-h 96 --img-w 144 --val-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Canonical action space (single source of truth shared with the runtime reasoner).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from vga.reason.base import BUTTON_CHOICES, WAIT  # noqa: E402

_IDX_BY_LOWER = {b.lower(): i for i, b in enumerate(BUTTON_CHOICES)}

# GBA key bit -> button name (authoritative GBAKey enum order, verified vs mGBA source).
_BIT_BUTTON = {0: "A", 1: "B", 2: "select", 3: "start", 4: "right", 5: "left",
               6: "up", 7: "down", 8: "R", 9: "L"}
# For a chord (multiple keys held), pick one label by this priority: deliberate action
# buttons before movement (chords are rare in Pokemon; noted so it's easy to revisit).
_CHORD_PRIORITY = ["A", "B", "start", "select", "L", "R", "up", "down", "left", "right"]


def _keymask_label(keys: int) -> int:
    """Map an mGBA getKeys() bitmask to a single button class index."""
    pressed = [name for bit, name in _BIT_BUTTON.items() if keys & (1 << bit)]
    if not pressed:
        return _IDX_BY_LOWER[WAIT]
    for name in _CHORD_PRIORITY:
        if name in pressed:
            return _IDX_BY_LOWER[name.lower()]
    return _IDX_BY_LOWER[WAIT]


def _label_index(step: dict) -> int:
    """The button the reasoner chose, as a class index. Unknown/missing -> wait."""
    b = (step.get("meta") or {}).get("button")
    if b is None:
        a = step.get("action") or {}
        b = a.get("button")
    if b is None:
        return _IDX_BY_LOWER[WAIT]
    return _IDX_BY_LOWER.get(str(b).strip().lower(), _IDX_BY_LOWER[WAIT])


def _load_frame(path: str, w: int, h: int) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, HxWx3 uint8
    if img is None:
        return None
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_traj(traj_dir: str, w: int, h: int, dedup_thresh: float = 0.0):
    """Yield (frame uint8 HxWx3 RGB, label idx) for each step with a readable frame_before.

    dedup_thresh > 0 drops frames whose mean-abs grayscale diff vs the LAST KEPT frame is
    below the threshold -- i.e. near-duplicate / frozen frames. A self-driving agent that
    picks "wait" on a halted screen (e.g. an unadvanced "press A" text prompt) logs the SAME
    frame->wait pair hundreds of times; keeping those would teach the policy to freeze. This
    filter is general (any game, any stuck span), applied per trajectory."""
    steps_path = os.path.join(traj_dir, "steps.jsonl")
    if not os.path.isfile(steps_path):
        print(f"  WARN no steps.jsonl in {traj_dir}, skipping")
        return
    last_gray = None
    dropped = 0
    kept = 0
    with open(steps_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            step = json.loads(line)
            # run_reasoner/teacher trajectories log "frame_before"; bench_ladder rollouts
            # log the same pre-action observation as "frame" (rollout.py saves the PNG
            # before the policy acts on it) -- accept either.
            rel = step.get("frame_before") or step.get("frame")
            if not rel:
                continue
            frame = _load_frame(os.path.join(traj_dir, rel), w, h)
            if frame is None:
                print(f"  WARN unreadable frame {rel} in {traj_dir}")
                continue
            if dedup_thresh > 0.0:
                gray = frame.mean(axis=2)
                if last_gray is not None and np.abs(gray - last_gray).mean() < dedup_thresh:
                    dropped += 1
                    continue
                last_gray = gray
            kept += 1
            yield frame, _label_index(step)
    if dedup_thresh > 0.0:
        print(f"    dedup(thresh={dedup_thresh}): kept {kept}, dropped {dropped} near-duplicate frames")


def _read_human(run_dir: str, w: int, h: int, dedup_thresh: float = 0.0):
    """Yield (frame RGB, label idx) from a human-demo run (train/teacher/human_logger.lua).

    Reads inputs.csv lines 'sample,frames/NNNNNN.png,keymask' -- one per human input edge,
    so the frames are decision points with no idle-wait spam. Same dedup as _read_traj."""
    csv_path = os.path.join(run_dir, "inputs.csv")
    if not os.path.isfile(csv_path):
        print(f"  WARN no inputs.csv in {run_dir}, skipping")
        return
    last_gray = None
    dropped = kept = 0
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("sample,"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            rel, keys = parts[1], int(parts[2])
            frame = _load_frame(os.path.join(run_dir, rel), w, h)
            if frame is None:
                print(f"  WARN unreadable frame {rel} in {run_dir}")
                continue
            if dedup_thresh > 0.0:
                gray = frame.mean(axis=2)
                if last_gray is not None and np.abs(gray - last_gray).mean() < dedup_thresh:
                    dropped += 1
                    continue
                last_gray = gray
            kept += 1
            yield frame, _keymask_label(keys)
    if dedup_thresh > 0.0:
        print(f"    dedup(thresh={dedup_thresh}): kept {kept}, dropped {dropped} near-duplicate frames")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True,
                    help="one or more trajectory dirs (each has steps.jsonl + frames/)")
    ap.add_argument("--out", required=True, help="output dir for train.npz/val.npz/meta.json")
    ap.add_argument("--img-h", type=int, default=96)
    ap.add_argument("--img-w", type=int, default=144)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="held-out fraction (contiguous tail per trajectory)")
    ap.add_argument("--val-by-traj", action="store_true",
                    help="hold out whole trajectories instead of a per-traj tail "
                         "(cleaner; needs >=2 trajectories)")
    ap.add_argument("--dedup-thresh", type=float, default=0.0,
                    help="drop frames within mean-abs grayscale diff < THRESH of the last "
                         "kept frame (removes frozen/near-duplicate spans; ~2.0 works)")
    ap.add_argument("--human-log", action="store_true",
                    help="--traj dirs are human-demo runs (inputs.csv from human_logger.lua) "
                         "instead of autonomous trajectories (steps.jsonl)")
    args = ap.parse_args()
    reader = _read_human if args.human_log else _read_traj

    trajs = [t for t in args.traj if os.path.isdir(t)]
    missing = [t for t in args.traj if not os.path.isdir(t)]
    for m in missing:
        print(f"WARN not a dir, skipping: {m}")
    if not trajs:
        print("ERROR no valid trajectory dirs")
        return 2

    Xtr, ytr, Xva, yva = [], [], [], []

    if args.val_by_traj:
        if len(trajs) < 2:
            print("ERROR --val-by-traj needs >=2 trajectories")
            return 2
        val_set = {trajs[-1]}  # deterministic: last dir is validation
        for t in trajs:
            frames = list(reader(t, args.img_w, args.img_h, args.dedup_thresh))
            dst_X, dst_y = (Xva, yva) if t in val_set else (Xtr, ytr)
            for fr, lb in frames:
                dst_X.append(fr); dst_y.append(lb)
            print(f"  {t}: {len(frames)} steps -> {'VAL' if t in val_set else 'train'}")
    else:
        for t in trajs:
            frames = list(reader(t, args.img_w, args.img_h, args.dedup_thresh))
            n = len(frames)
            n_val = int(round(n * args.val_frac))
            n_tr = n - n_val
            for fr, lb in frames[:n_tr]:
                Xtr.append(fr); ytr.append(lb)
            for fr, lb in frames[n_tr:]:
                Xva.append(fr); yva.append(lb)
            print(f"  {t}: {n} steps -> {n_tr} train / {n_val} val (tail)")

    if not Xtr:
        print("ERROR no training samples collected")
        return 2

    Xtr = np.asarray(Xtr, dtype=np.uint8)
    ytr = np.asarray(ytr, dtype=np.int64)
    Xva = np.asarray(Xva, dtype=np.uint8) if Xva else np.zeros((0, args.img_h, args.img_w, 3), np.uint8)
    yva = np.asarray(yva, dtype=np.int64) if yva else np.zeros((0,), np.int64)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "train.npz"), X=Xtr, y=ytr)
    np.savez_compressed(os.path.join(args.out, "val.npz"), X=Xva, y=yva)

    def hist(y):
        c = np.bincount(y, minlength=len(BUTTON_CHOICES))
        return {BUTTON_CHOICES[i]: int(c[i]) for i in range(len(BUTTON_CHOICES)) if c[i]}

    meta = {
        "classes": BUTTON_CHOICES,
        "img_h": args.img_h, "img_w": args.img_w,
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "trajs": trajs,
        "val_by_traj": bool(args.val_by_traj),
        "train_hist": hist(ytr), "val_hist": hist(yva),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {args.out}/train.npz ({Xtr.shape}) + val.npz ({Xva.shape})")
    print(f"train class hist: {meta['train_hist']}")
    print(f"val   class hist: {meta['val_hist']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
