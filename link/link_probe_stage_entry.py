"""Probe (2026-07-07): from the banked 4-core stage_select checkpoint, drive the
parent to pick Chambers of Insight and discover what the game needs to reach
actual co-op gameplay (children may need their own confirms). Iterated by
editing SCHEDULE between runs -- each run is seconds at ~85 ticks/s.

SCHEDULE: list of (tick, core_index_or_None_for_all, key_name_or_None_release).
Keys held from their tick until the release entry.
"""
import os
import sys

import mgba.gba as gba
from mgba._pylib import ffi
from link_engine import LinkSession

ROM = os.environ["FOUR_SWORDS_ROM"]
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "sessions", "checkpoints")
STATES = [os.path.join(CKPT, "p%d_stage_select.state" % i) for i in range(4)]
OUT = os.path.join(HERE, "sessions", "stage_entry")
os.makedirs(OUT, exist_ok=True)
K = gba.GBA

TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
SNAPS = {25, 60, 100, 200, 300, 400, 600, TICKS - 1}

# round 1: parent selects the stage, nothing else
SCHEDULE = [
    (30, 0, "KEY_A"), (34, 0, None),
]

SAVE_AT = 600   # set to a tick to bank p*_coop.state at that tick


def snap(sess, tick):
    for nd in sess.nodes:
        p = os.path.join(OUT, "c%d_t%04d.png" % (nd.index, tick))
        with open(p, "wb") as f:
            nd.image.save_png(f)


sess = LinkSession(ROM, n=4, state_path=STATES)
for nd in sess.nodes:
    if nd.index > 0 and nd.irq_flagged:
        nd.irq_pending = True
        nd.mltsend_seen = False
sched = {(t): (who, key) for (t, who, key) in SCHEDULE}
for tick in range(TICKS):
    if tick in sched:
        who, key = sched[tick]
        keys = () if key is None else (getattr(K, key),)
        targets = sess.nodes if who is None else [sess.nodes[who]]
        for nd in targets:
            if key is None:
                nd.core.set_keys(raw=0)
            else:
                nd.core.set_keys(*keys)
    sess.tick()
    if tick in SNAPS:
        snap(sess, tick)
        s = " ".join("c%d:m%d" % (nd.index, nd.mode) for nd in sess.nodes)
        print("tick %4d transfers=%d [%s]" % (tick, sess.transfers, s), flush=True)
    if SAVE_AT is not None and tick == SAVE_AT:
        for i, nd in enumerate(sess.nodes):
            st = nd.core.save_raw_state()
            with open(os.path.join(CKPT, "p%d_coop.state" % i), "wb") as f:
                f.write(bytes(ffi.buffer(st)))
        print("banked p*_coop.state at tick %d" % tick, flush=True)
print("done. transfers=%d" % sess.transfers, flush=True)
