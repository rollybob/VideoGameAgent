"""P1a Stage 1 (measure-first, headless, NO Xvfb/xdotool): baseline for the System-1/System-2
arbiter. Runs S1 policies on a room and reports rooms/keys/reward PLUS how often the SHIPPED
freeze gate (aHash Hamming <= 10 for FREEZE_K consecutive steps) would fire.

WHY: the arbiter's planned interject trigger is the shipped screen-static/freeze detector. But
S1's -06 failure may be ineffective WANDERING (screen keeps moving), which that gate would NOT
catch. This decides the next build:
  - freeze events frequent  -> freeze-triggered S2 interject is the right Stage-2 build.
  - freeze events ~0         -> gate mismatch; the interject needs a progress/OOD signal, not
                                screen-static. Report it; do NOT build the freeze-arbiter blind.

Default contrast on alttp_human-06:
  lr12  = the -04-trained SETTLED policy (runs/BEST) -> TRANSFER FAILURE on -06 (S1 under stress)
  ov06ws= the -06-trained policy                     -> native, should progress w/ few freezes
  random= floor
"""
import argparse, os, sys
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from alttp_ppo_env import AlttpPpoEnv

_AH = 16  # 16x16 -> 256-bit average hash (matches vga/reason/plugin.py _ahash)


def _gray16(frame):
    """RGB HxWx3 uint8 -> 16x16 float grayscale, dependency-robust (cv2 -> PIL -> numpy)."""
    g = frame.astype(np.float32).mean(axis=2)
    try:
        import cv2
        return cv2.resize(g, (_AH, _AH), interpolation=cv2.INTER_AREA)
    except Exception:
        pass
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(g.astype(np.uint8)).resize((_AH, _AH))).astype(np.float32)
    except Exception:
        h, w = g.shape
        ys = (np.arange(_AH) * h // _AH); xs = (np.arange(_AH) * w // _AH)
        return g[np.ix_(ys, xs)]


def _ahash(frame):
    s = _gray16(frame)
    bits = (s >= s.mean()).astype(np.uint8).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _hamming(a, b):
    return int(a ^ b).bit_count()


def run_ep(env, act_fn, seed, freeze_k, thresh):
    obs, _ = env.reset(seed=seed)
    keys = 0; total = 0.0; done = False; steps = 0
    prev_h = _ahash(env._frames[-1]); static_run = 0
    freeze_events = 0; longest = 0; static_steps = 0; in_freeze = False
    while not done:
        obs, r, term, trunc, info = env.step(act_fn(obs))
        total += r; steps += 1
        keys += sum(1 for e in info["events"] if e[1] == "key")
        h = _ahash(env._frames[-1])
        if _hamming(h, prev_h) <= thresh:
            static_run += 1; static_steps += 1
            longest = max(longest, static_run)
            if static_run >= freeze_k and not in_freeze:
                freeze_events += 1; in_freeze = True   # count each distinct freeze episode once
        else:
            static_run = 0; in_freeze = False
        prev_h = h
        done = term or trunc
    return dict(rooms=len(env.oracle.rooms_seen), keys=keys, reward=total, steps=steps,
                static_frac=static_steps / max(1, steps), freezes=freeze_events, longest=longest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["ov06ws=runs/alttp_ov06_ws/ppo_alttp_final.zip",
                             "ov06ctrl=runs/alttp_ov06_ctrl/ppo_alttp_final.zip"])
    ap.add_argument("--states", nargs="+", default=["alttp_human-06", "alttp_human-04"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-k", type=int, default=3)
    ap.add_argument("--thresh", type=int, default=10)
    a = ap.parse_args()

    print("states:", a.states, "| episodes:", a.episodes, "| horizon:", a.horizon,
          "| freeze_k:", a.freeze_k, "| static thresh(hamming<=):", a.thresh)

    # Load CURRENT-arch policies once. Pre-2026-08-04 ckpts (single-frame 3ch channels-first)
    # are INCOMPATIBLE with the 12ch stacked env and are skipped with a clear note.
    models = {}
    for spec in a.models:
        lab, p = spec.split("=", 1)
        if not os.path.isfile(p):
            print("SKIP %-10s MISSING %s" % (lab, p)); continue
        try:
            models[lab] = PPO.load(p, device="cuda")
        except Exception as e:
            print("SKIP %-10s load-failed: %r" % (lab, repr(e)[:140])); continue
    print("loaded:", ", ".join(models) or "(none)")

    grid = {}  # (state, label) -> metrics
    for sname in a.states:
        spath = os.path.join(HERE, "states", sname + ".state")
        if not os.path.isfile(spath):
            print("SKIP state MISSING %s" % spath); continue
        env = AlttpPpoEnv(state_path=spath, horizon=a.horizon); env.action_space.seed(a.seed)
        print("\n== %s ==" % sname)

        def report(label, fn):
            try:
                eps = [run_ep(env, fn, a.seed + i, a.freeze_k, a.thresh) for i in range(a.episodes)]
            except Exception as e:
                print("%-16s ERROR %s" % (label, repr(e)[:160])); return None
            avg = {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}
            print("%-16s rooms=%.2f keys=%.2f reward=%+9.1f steps=%.0f static=%4.1f%% "
                  "freezes/ep=%.2f longest_static=%.0f"
                  % (label, avg["rooms"], avg["keys"], avg["reward"], avg["steps"],
                     100 * avg["static_frac"], avg["freezes"], avg["longest"]))
            return avg

        report("random", lambda o: env.action_space.sample())
        for lab, model in models.items():
            av = report(lab + "-det", lambda o, m=model: m.predict(o, deterministic=True)[0])
            if av is not None:
                grid[(sname, lab)] = av
        env.close()

    # Stage-2 gate check on the MOST-under-stress run (lowest keys among policy runs).
    if grid:
        (sname, lab), st = min(grid.items(), key=lambda kv: kv[1]["keys"])
        fz, sf = st["freezes"], 100 * st["static_frac"]
        print("\nSTAGE-2 GATE CHECK  (S1-under-stress = lowest-keys run: %s on %s)" % (lab, sname))
        print("  keys=%.2f rooms=%.2f freezes/ep=%.2f static=%.1f%% longest_static=%.0f"
              % (st["keys"], st["rooms"], fz, sf, st["longest"]))
        if fz >= 1.0:
            print("  => FREEZE GATE FIRES on the failure case: a freeze-triggered S2 interject is a "
                  "valid Stage-2 build.")
        else:
            print("  => FREEZE GATE ~SILENT: S1 fails by ineffective WANDERING, not freezing. The "
                  "arbiter interject needs a PROGRESS/OOD signal, not screen-static (plan P2). Do "
                  "NOT build the freeze-arbiter blind.")
    else:
        print("\nNO usable model runs -- cannot decide Stage 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
