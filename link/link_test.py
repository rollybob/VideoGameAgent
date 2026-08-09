"""THE TEST: load the Four Swords 'PRESS START' checkpoint into 4 cores, press
START on all, run the deterministic coordinator, and screenshot the cores to see
if they advance PAST 'Linking with other systems' (where mgba's GUI hangs)."""
import os
import mgba.image
from link_engine import LinkSession
import mgba.gba as gba

ROM = os.environ["FOUR_SWORDS_ROM"]
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "sessions", "checkpoints", "link_wait.state")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions", "linktest")
os.makedirs(OUT, exist_ok=True)
K = gba.GBA

sess = LinkSession(ROM, n=4, state_path=STATE)
print("loaded 4 cores from checkpoint")
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
