"""Map how multiple enemies populate the 16-slot sprite arrays (2026-08-03), to
generalize enemy_dmg beyond -04's slot 3. Replays a solo_recorder tape where Tim
clears a multi-enemy room ONE BY ONE, and reports per sprite slot: the type id(s)
seen while alive, max/observed health, how many frames it was active, and the
frame(s) its health fell to 0 (a kill/clear). Correlate the clear frames with the
order Tim killed enemies to learn which slot each enemy used.

IWRAM parallel arrays, stride 1, 16 slots: health base 0x03250, type base 0x03160
(so -04's known enemy = slot 3 at 0x03253 / 0x03163).

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga/train/rl \\
      thor-rl:cu130 python3 find_enemy_slots.py <tape-dir>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mgba.core
import mgba.image
import mgba.log
from mgba._pylib import ffi

mgba.log.silence()
from alttp_ppo_env import ROM

HP_BASE, ID_BASE, N_SLOTS = 0x03250, 0x03160, 16
HP_SANE_MAX = 32  # ignore slots whose "health" never sits in a plausible enemy range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="solo_recorder tape dir")
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM", ROM))
    ap.add_argument("--hp-base", type=lambda x: int(x, 0), default=HP_BASE)
    ap.add_argument("--id-base", type=lambda x: int(x, 0), default=ID_BASE)
    ap.add_argument("--slots", type=int, default=N_SLOTS)
    a = ap.parse_args()

    with open(os.path.join(a.dump, "inputs.json")) as f:
        masks = json.load(f)["masks"]
    core = mgba.core.load_path(a.rom)
    img = mgba.image.Image(*core.desired_video_dimensions())
    core.set_video_buffer(img)
    core.reset()
    with open(os.path.join(a.dump, "prefail.state"), "rb") as f:
        assert core.load_raw_state(f.read()), "load_raw_state failed"
    iw_ptr = core._native.memory.iwram

    # per-slot accumulators
    ids = [set() for _ in range(a.slots)]      # type ids seen while hp in (0, SANE]
    maxhp = [0] * a.slots
    active = [0] * a.slots                      # frames with hp in (0, SANE]
    clears = [[] for _ in range(a.slots)]       # frames hp went (0,SANE] -> 0
    prev_hp = [0] * a.slots
    for k, mask in enumerate(masks):
        core.set_keys(raw=int(mask))
        core.run_frame()
        iw = np.frombuffer(ffi.buffer(iw_ptr, 0x8000), dtype=np.uint8)
        hp = iw[a.hp_base:a.hp_base + a.slots]
        tid = iw[a.id_base:a.id_base + a.slots]
        for s in range(a.slots):
            h = int(hp[s])
            if 0 < h <= HP_SANE_MAX:
                active[s] += 1
                maxhp[s] = max(maxhp[s], h)
                ids[s].add(int(tid[s]))
            if 0 < prev_hp[s] <= HP_SANE_MAX and h == 0:
                clears[s].append(k)
            prev_hp[s] = h

    print("tape %s : %d frames" % (os.path.basename(os.path.normpath(a.dump)), len(masks)))
    print("slot  hp_addr   active  maxhp  type_ids(hex)         clear_frames")
    any_active = False
    for s in range(a.slots):
        if active[s] == 0 and not clears[s]:
            continue
        any_active = True
        idhex = ",".join("0x%02X" % i for i in sorted(ids[s])) or "-"
        cl = ",".join(str(c) for c in clears[s][:12]) + (" ..." if len(clears[s]) > 12 else "")
        print("  %2d  0x%05X  %6d  %5d  %-20s  %s"
              % (s, a.hp_base + s, active[s], maxhp[s], idhex, cl or "-"))
    if not any_active:
        print("NO active enemy slots found -- check hp/id base, or the room had no enemies.")
    print("\n(active = frames with hp in (0,%d]; clear = a kill/despawn. Match clear order to"
          " the order you killed enemies to map each enemy -> slot.)" % HP_SANE_MAX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
