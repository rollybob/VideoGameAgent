"""Test Tim's 'layer' theory for the (0,6)->(3,3) ladder warp (2026-08-03).

Is there a RAM register that tracks area/layer -- one that changes AT THE WARP but
is STABLE while walking (unlike position, which changes every step)? Method: hold
DOWN from -01; accumulate every RAM byte that changes during walking (position,
animation, camera, enemies, AND a normal room-boundary cross at transition 1);
then at the WARP (transition 2) diff the frame just before vs just after.
Candidates for an area/layer register = bytes that change at the warp but were
NEVER touched during ~660 frames of walking incl. a normal room change. Flag-like
(small-valued) candidates are the cleanest -- an indoors flag or area index.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mgba.core
import mgba.image
import mgba.log
from mgba._pylib import ffi

from alttp_ppo_env import ROM, W, H, GBA
from oracle import AlttpOracle

mgba.log.silence()

EW_SZ, IW_SZ = 0x40000, 0x8000


def mem(core):
    ew = np.frombuffer(bytes(ffi.buffer(
        ffi.cast("uint8_t *", core._native.memory.wram), EW_SZ)), dtype=np.uint8).copy()
    iw = np.frombuffer(bytes(ffi.buffer(
        ffi.cast("uint8_t *", core._native.memory.iwram), IW_SZ)), dtype=np.uint8).copy()
    return ew, iw


def main():
    core = mgba.core.load_path(ROM)
    img = mgba.image.Image(W, H)
    core.set_video_buffer(img)
    core.reset()
    with open(os.path.join(HERE, "states", "alttp_human-01.state"), "rb") as f:
        core.load_raw_state(f.read())
    core.set_keys(raw=0)
    core.run_frame()
    orc = AlttpOracle()
    key = 1 << GBA.KEY_DOWN

    proom = orc.room_of(*orc.read_pos(core))
    prev_ew, prev_iw = mem(core)
    walk_ew = np.zeros(EW_SZ, bool)
    walk_iw = np.zeros(IW_SZ, bool)
    trans = 0
    pre_ew = pre_iw = post_ew = post_iw = None
    for f in range(5000):
        core.set_keys(raw=key)
        core.run_frame()
        room = orc.room_of(*orc.read_pos(core))
        ew, iw = mem(core)
        if room != proom:
            trans += 1
            if trans == 2:
                pre_ew, pre_iw = prev_ew, prev_iw
                post_ew, post_iw = ew, iw
                print("WARP at frame %d, room %s -> %s" % (f, proom, room))
                break
            # transition 1 = a normal room cross; fold into walking volatility
        walk_ew |= (ew != prev_ew)
        walk_iw |= (iw != prev_iw)
        prev_ew, prev_iw, proom = ew, iw, room

    if pre_ew is None:
        print("never reached the warp (trans=%d)" % trans)
        return 1

    for name, base, pre, post, walk in [
            ("EWRAM", 0x02000000, pre_ew, post_ew, walk_ew),
            ("IWRAM", 0x03000000, pre_iw, post_iw, walk_iw)]:
        cand = (pre != post) & (~walk)
        idx = np.where(cand)[0]
        flags = [(i, int(pre[i]), int(post[i])) for i in idx
                 if pre[i] < 32 and post[i] < 32]
        print("\n%s: warp-changed & walk-STABLE = %d bytes ; flag-like(<32) = %d"
              % (name, len(idx), len(flags)))
        for i, a, b in flags[:60]:
            print("  0x%05X (%08X)  %2d -> %2d" % (i, base + i, a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
