"""Capture a NEW ALttP savestate by DESCENDING through room transitions
(2026-08-03, Tim's idea). Load a state, hold a direction, and each time Link
walks through a doorway into the next room (room = pos>>9 changes) settle, bank
a snapshot, measure how GREEN the room is, and keep going -- up to --chain
transitions or until Link dies. Purpose: reach the OUTDOOR (bushes) overworld we
lack, which is a couple rooms DOWN from -01 (blue knights -> green knights ->
outside). Outdoors is detected by a spike in green-pixel fraction (grass), so we
find it from numbers instead of eyeballing every room; the greenest room above a
threshold is promoted to the --out state.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga/train/rl \\
      thor-rl:cu130 python3 capture_transition.py --from 01 --direction down --chain 4 --out 06
"""
import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mgba.core
import mgba.image
import mgba.log
from mgba._pylib import ffi

from alttp_ppo_env import ROM, GBA, W, H, _get_frame
from oracle import AlttpOracle

mgba.log.silence()

DIR_KEY = {"up": GBA.KEY_UP, "down": GBA.KEY_DOWN,
           "left": GBA.KEY_LEFT, "right": GBA.KEY_RIGHT}
STATES = os.path.join(HERE, "states")
GREEN_OUTDOOR = 0.12   # green-pixel fraction above which a room is "outdoors"


def build(state_path):
    core = mgba.core.load_path(ROM)
    if core is None:
        raise RuntimeError("could not load ROM: %s" % ROM)
    img = mgba.image.Image(W, H)
    core.set_video_buffer(img)
    core.reset()
    with open(state_path, "rb") as f:
        if not core.load_raw_state(f.read()):
            raise RuntimeError("load_raw_state failed: %s" % state_path)
    core.set_keys(raw=0)
    core.run_frame()
    return core, img


def green_frac(frame):
    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    mask = (g > r + 12) & (g > b + 12) & (g > 60)   # grassy green; R/B-swap invariant
    return float(mask.mean())


def save_state(core, path):
    st = core.save_raw_state()
    if not st:
        raise RuntimeError("save_raw_state returned None")
    with open(path, "wb") as fp:
        fp.write(bytes(ffi.buffer(st)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="01")
    ap.add_argument("--direction", default="down", choices=list(DIR_KEY))
    ap.add_argument("--out", default="06", help="state number to promote the outdoor room to")
    ap.add_argument("--chain", type=int, default=4, help="max transitions to walk through")
    ap.add_argument("--max-frames", type=int, default=1500, help="per-room frame cap")
    ap.add_argument("--settle", type=int, default=45)
    a = ap.parse_args()

    core, img = build(os.path.join(STATES, "alttp_human-%s.state" % a.frm))
    orc = AlttpOracle()
    key = 1 << DIR_KEY[a.direction]
    room = orc.room_of(*orc.read_pos(core))
    print("start: -%s room %s HP %d green %.3f"
          % (a.frm, room, orc.read_health(core), green_frac(_get_frame(img))), flush=True)

    records = []
    for t in range(1, a.chain + 1):
        hit = False
        for _ in range(a.max_frames):
            core.set_keys(raw=key)
            core.run_frame()
            if orc.read_health(core) == 0:
                print("  DIED walking %s before transition %d (reached %d transitions)"
                      % (a.direction, t, t - 1))
                hit = None
                break
            nroom = orc.room_of(*orc.read_pos(core))
            if nroom != room:
                room = nroom
                hit = True
                break
        if hit is None:      # died
            break
        if not hit:
            print("  NO TRANSITION %d in %d frames (walled at room %s)" % (t, a.max_frames, room))
            break
        for _ in range(a.settle):
            core.set_keys(raw=0)
            core.run_frame()
        room = orc.room_of(*orc.read_pos(core))
        gf = green_frac(_get_frame(img))
        hp = orc.read_health(core)
        snap = os.path.join(STATES, "descend_t%d.state" % t)
        save_state(core, snap)
        records.append((t, room, gf, hp, snap))
        print("  transition %d -> room %s  green %.3f  HP %d  (%s)"
              % (t, room, gf, hp, os.path.basename(snap)), flush=True)
        if gf >= GREEN_OUTDOOR:
            print("  ^ OUTDOORS (green >= %.2f) -- stopping descent" % GREEN_OUTDOOR)
            break

    if not records:
        print("no rooms captured.")
        return 1
    best = max(records, key=lambda r: r[2])
    t, room, gf, hp, snap = best
    if gf >= GREEN_OUTDOOR:
        dst = os.path.join(STATES, "alttp_human-%s.state" % a.out)
        shutil.copyfile(snap, dst)
        print("PROMOTED transition %d (room %s, green %.3f, HP %d) -> %s"
              % (t, room, gf, hp, os.path.basename(dst)))
        return 0
    print("NO OUTDOOR room reached (max green %.3f at transition %d, room %s). "
          "Snapshots kept as descend_t*.state for inspection." % (gf, t, room))
    return 2


if __name__ == "__main__":
    sys.exit(main())
