"""Render each banked ALttP save state to a PNG thumbnail for identification
(which room is which -- DR planning, and finding the outdoor pots/bushes room).

Uses AlttpPpoEnv.reset (which ticks a frame, so no stale boot frame) then a few
noop steps to let the display settle before capturing. Saves via PIL/cv2/npy
whichever is available. Colours may be R/B-swapped depending on the mgba buffer
order, but green (middle channel) and grey/stone read the same either way, which
is all we need to tell an outdoor green area from an indoor dungeon.
"""
import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alttp_ppo_env import AlttpPpoEnv

VGA = os.path.dirname(os.path.dirname(HERE))


def save_png(arr, path):
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
        return "PIL"
    except Exception:
        pass
    try:
        import cv2
        cv2.imwrite(path, arr[:, :, ::-1])  # RGB->BGR
        return "cv2"
    except Exception:
        pass
    np.save(path.replace(".png", ".npy"), arr)
    return "npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="*", default=["00", "01", "02", "03", "04", "05"])
    ap.add_argument("--settle", type=int, default=8, help="noop steps before capture")
    ap.add_argument("--outdir", default=os.path.join(VGA, "sessions", "state_thumbs"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    for s in a.states:
        sp = os.path.join(HERE, "states", "alttp_human-%s.state" % s)
        if not os.path.exists(sp):
            print("MISSING %s" % sp)
            continue
        env = AlttpPpoEnv(state_path=sp, horizon=(a.settle + 2) * 8)
        obs, _ = env.reset()
        for _ in range(a.settle):
            obs, _r, term, trunc, _i = env.step(0)
            if term or trunc:
                break
        out = os.path.join(a.outdir, "alttp_human-%s.png" % s)
        how = save_png(obs, out)
        print("state %s -> %s (%s) shape=%s" % (s, out, how, obs.shape))
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
