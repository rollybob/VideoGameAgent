"""Identity-hypothesis test (2026-07-07): same as link_test.py but each core
loads its OWN checkpoint (distinct character files A/B/C/D from
drive_p4_unique.py) instead of 4 byte-identical clones of one save.

PASS if any core advances past 'Linking with other systems' (screen changes /
siocnt keeps evolving after ~tick 60). FAIL if the identical hang reproduces --
which rules the clone-save hypothesis out.
"""
import os
import mgba.image
from link_engine import LinkSession
import mgba.gba as gba

ROM = os.environ["FOUR_SWORDS_ROM"]
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "sessions", "checkpoints")
STATES = [os.path.join(CKPT, "p%d_link_wait.state" % i) for i in range(4)]
OUT = os.path.join(HERE, "sessions", "linktest_unique")
os.makedirs(OUT, exist_ok=True)
K = gba.GBA

sess = LinkSession(ROM, n=4, state_path=STATES)
print("loaded 4 cores from per-player checkpoints")
for nd in sess.nodes:
    print("  core %d: mode=%d siocnt=%04X" % (nd.index, nd.mode, nd.siocnt))

def snap(tick):
    for nd in sess.nodes:
        p = os.path.join(OUT, "c%d_t%04d.png" % (nd.index, tick))
        with open(p, "wb") as f:
            nd.image.save_png(f)

def nz(nd):
    b = bytes(mgba.image.ffi.buffer(nd.image.buffer))
    return sum(1 for x in b if x)

snap(0)
SHOTS = {0, 60, 200, 500, 900, 1399}
for tick in range(1400):
    if 2 <= tick <= 6:
        sess.press_all(K.KEY_START)      # initiate link from PRESS START
    elif tick == 7:
        sess.press_all(raw=0)
    sess.tick()
    if tick in SHOTS:
        snap(tick)
        s = " ".join("c%d:%04X" % (nd.index, nd.siocnt) for nd in sess.nodes)
        print("tick %4d transfers=%d siocnt[%s] nz0=%d" %
              (tick, sess.transfers, s, nz(sess.nodes[0])))
print("done. total transfers:", sess.transfers)
