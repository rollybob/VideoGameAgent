"""Stage 0 (multi-input rebuild) of the BC-bootstrap plan: turn solo-ALttP human tapes
into a behavior-cloning dataset whose (obs, action) space MATCHES the env's NEW
MultiDiscrete([9,2,2,2,2]) = [dir, A, B, L, R] action, so a BC-pretrained policy can
warm-start PPO on the same action head.

WHY multi-input labels (2026-08-04): the earlier single-button label collapsed the
human's DIAGONALS and simultaneous move+attack onto one button, which was a prime
suspect for BC failing behaviorally (the cloned policy literally could not reproduce
the human's path). The label is now the FULL decomposed mask -- lossless.

Replay is UNCHANGED from the validated version (faithful mask[k]->frame k, no prime
frame, obs at window boundaries, first window skipped) -- it reproduces score_tape.py's
event stream exactly (castle run = 29 rooms / 4 keys), which is the correctness check.

Label per FRAME_SKIP-frame window:
- dir: MODAL 9-way direction over the window (ties -> prefer a real movement dir). D-pad
  contradictions (U+D / L+R) are dropped per frame before matching the env's DIR_MASKS.
- A,B,L,R: each ON if the human held it for >= HALF the window's frames (a charge = B held
  across windows; a stray 1-frame tap in a 4-frame window is dropped as sub-decision noise).

Run (CPU only; --user keeps outputs timothy-owned not root):
  docker run --rm --user $(id -u):$(id -g) -v ~/projects/VGA:/vga -w /vga/train/rl \\
      thor-rl:cu130 python3 build_bc_data.py \\
      --tapes <dir> ... --out /vga/train/policy/bc_data_alttp_md
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mgba.core
import mgba.image
import mgba.log

mgba.log.silence()
from alttp_ppo_env import ROM, _get_frame, FRAME_SKIP, STACK_K, DIR_MASKS, ACTION_NVEC, GBA
from oracle import AlttpOracle

DIR_NAMES = ["none", "U", "D", "L", "R", "NE", "NW", "SE", "SW"]  # index-aligned to DIR_MASKS
COMPONENTS = ["dir", "A", "B", "L", "R"]
_bA, _bB, _bL, _bR = 1 << GBA.KEY_A, 1 << GBA.KEY_B, 1 << GBA.KEY_L, 1 << GBA.KEY_R
_bU, _bD, _bLF, _bRT = 1 << GBA.KEY_UP, 1 << GBA.KEY_DOWN, 1 << GBA.KEY_LEFT, 1 << GBA.KEY_RIGHT


def mask_to_dir(mask):
    """Raw key-bitmask -> dir index, derived FROM the env's DIR_MASKS so they can't drift.
    Opposite D-pad presses cancel (humans don't hold U+D), leaving a valid 9-way dir."""
    up = bool(mask & _bU) and not (mask & _bD)
    dn = bool(mask & _bD) and not (mask & _bU)
    lf = bool(mask & _bLF) and not (mask & _bRT)
    rt = bool(mask & _bRT) and not (mask & _bLF)
    clean = (_bU if up else 0) | (_bD if dn else 0) | (_bLF if lf else 0) | (_bRT if rt else 0)
    return DIR_MASKS.index(clean)


def reduce_window_md(window):
    """FRAME_SKIP masks -> [dir, A, B, L, R]."""
    dirs = [mask_to_dir(m) for m in window]
    c = collections.Counter(dirs)
    top = max(c.values())
    tied = [d for d in c if c[d] == top]
    nz = [d for d in tied if d != 0]
    d = min(nz) if nz else 0                       # modal dir, ties -> a real movement dir
    half = (len(window) + 1) // 2
    a = int(sum(bool(m & _bA) for m in window) >= half)
    b = int(sum(bool(m & _bB) for m in window) >= half)
    l = int(sum(bool(m & _bL) for m in window) >= half)
    r = int(sum(bool(m & _bR) for m in window) >= half)
    return [d, a, b, l, r]


def replay_tape(core, img, tape_dir, frame_skip, stack_k, max_frames):
    with open(os.path.join(tape_dir, "inputs.json")) as f:
        masks = json.load(f)["masks"]
    if max_frames:
        masks = masks[:max_frames]
    state_path = os.path.join(tape_dir, "prefail.state")

    core.reset()
    with open(state_path, "rb") as f:
        assert core.load_raw_state(f.read()), "load_raw_state failed: %s" % state_path
    # NO prime frame -- mask[k] must land on frame k (validated: reproduces 29 rooms/4 keys).
    orc = AlttpOracle()
    orc.step(0, core)

    X, y = [], []
    key_events = 0
    tick = 0
    n = len(masks)
    frames = None                         # deque of the last stack_k RGB decision-frames
    i = 0
    while i + frame_skip <= n:
        window = masks[i:i + frame_skip]
        if frames is not None:            # first window has no rendered pre-obs -> skip it
            X.append(np.concatenate(list(frames), axis=-1))  # (H,W,3*stack_k) == env._stack()
            y.append(reduce_window_md(window))
        for m in window:
            core.set_keys(raw=int(m))
            core.run_frame()
            tick += 1
            for _t, ch, _d, _v in orc.step(tick, core):
                if ch == "key":
                    key_events += 1
        f = _get_frame(img)
        if frames is None:
            frames = collections.deque([f] * stack_k, maxlen=stack_k)  # fill at first boundary
        else:
            frames.append(f)
        i += frame_skip

    return X, y, {"frames": n, "windows": len(X), "rooms": set(orc.rooms_seen), "keys": key_events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", nargs="+", default=None)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM", ROM))
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    ap.add_argument("--stack-k", type=int, default=STACK_K)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    import glob as _glob
    cand = list(a.tapes or [])
    if a.glob:
        cand += sorted(_glob.glob(a.glob))
    tapes = [d for d in cand if os.path.isfile(os.path.join(d, "inputs.json"))
             and os.path.isfile(os.path.join(d, "prefail.state"))]
    if not tapes:
        print("no valid tapes"); return 1
    print("tapes: %d" % len(tapes))

    core = mgba.core.load_path(a.rom)
    img = mgba.image.Image(*core.desired_video_dimensions())
    core.set_video_buffer(img)

    per_tape = []
    t0 = time.time()
    for d in tapes:
        name = os.path.basename(d.rstrip("/"))
        X, y, st = replay_tape(core, img, d, a.frame_skip, a.stack_k, a.max_frames)
        per_tape.append((name, X, y, st))
        print("  %-38s frames=%-6d windows=%-6d rooms=%-3d keys=%-2d"
              % (name, st["frames"], st["windows"], len(st["rooms"]), st["keys"]))
    dt = time.time() - t0
    totf = sum(s["frames"] for _, _, _, s in per_tape)
    print("replayed %d frames in %.1fs (%.0f fps)" % (totf, dt, totf / max(dt, 1e-6)))

    trX, trY, vaX, vaY = [], [], [], []
    for _n, X, y, _s in per_tape:
        cut = int(round(len(X) * (1.0 - a.val_frac)))
        trX += X[:cut]; trY += y[:cut]; vaX += X[cut:]; vaY += y[cut:]
    trX = np.asarray(trX, dtype=np.uint8); trY = np.asarray(trY, dtype=np.int64)
    vaX = np.asarray(vaX, dtype=np.uint8); vaY = np.asarray(vaY, dtype=np.int64)
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "train.npz"), X=trX, y=trY)
    np.savez(os.path.join(a.out, "val.npz"), X=vaX, y=vaY)
    h, w = (int(trX.shape[1]), int(trX.shape[2])) if len(trX) else (0, 0)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"components": COMPONENTS, "nvec": list(ACTION_NVEC), "dir_names": DIR_NAMES,
                   "img_h": h, "img_w": w, "frame_skip": a.frame_skip, "stack_k": a.stack_k,
                   "channels": 3 * a.stack_k, "n_tapes": len(per_tape),
                   "label_agg": "dir=modal/window; A/B/L/R=held>=half-window"}, f, indent=1)

    # ---- report: does the multi-input label recover what single-button lost? ----
    allY = np.concatenate([trY, vaY]) if len(trY) or len(vaY) else np.zeros((0, 5), int)
    rooms_union = set().union(*[s["rooms"] for _, _, _, s in per_tape]) if per_tape else set()
    N = max(len(allY), 1)
    print("\n==== BC-MD DATASET REPORT ====")
    ch = int(trX.shape[3]) if len(trX) else 0
    print("train %d  val %d  (contiguous per-tape)  obs=%dx%dx%d  label=[dir,A,B,L,R]"
          % (len(trX), len(vaX), h, w, ch))
    print("distinct rooms covered: %d" % len(rooms_union))
    dc = collections.Counter(allY[:, 0].tolist())
    print("direction distribution (diagonals were IMPOSSIBLE to label before):")
    for k in range(len(DIR_NAMES)):
        c = dc.get(k, 0)
        print("  %-4s %7d  %5.1f%%  %s" % (DIR_NAMES[k], c, 100.0 * c / N, "#" * int(40 * c / N)))
    diag = sum(dc.get(k, 0) for k in (5, 6, 7, 8))
    print("  -> diagonals (NE/NW/SE/SW) = %.1f%% of windows" % (100.0 * diag / N))
    print("button ON-rates (independent now -- attack no longer collapses onto movement):")
    for j, nm in enumerate(["A", "B", "L", "R"], start=1):
        on = int(allY[:, j].sum())
        print("  %-2s %7d  %5.1f%%" % (nm, on, 100.0 * on / N))
    print("wrote %s/{train.npz,val.npz,meta.json}" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
