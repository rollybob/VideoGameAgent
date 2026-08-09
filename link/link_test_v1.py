"""Stage-3 test (2026-07-07): 4 cores, unique saves, v1 callback-driven
coordinator. PASS = any core advances past 'Linking with other systems'
(the screen v0 and the mGBA GUI both hang on)."""
import os
import mgba.image
from link_engine import LinkSession
import mgba.gba as gba

ROM = os.environ["FOUR_SWORDS_ROM"]
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "sessions", "checkpoints")
STATES = [os.path.join(CKPT, "p%d_link_wait.state" % i) for i in range(4)]
OUT = os.path.join(HERE, "sessions", "linktest_v1")
os.makedirs(OUT, exist_ok=True)
K = gba.GBA

sess = LinkSession(ROM, n=4, state_path=STATES, trace=True)
print("loaded 4 cores from per-player checkpoints")
for nd in sess.nodes:
    print("  core %d: mode=%d siocnt=%04X" % (nd.index, nd.mode, nd.siocnt))

def snap(tick):
    for nd in sess.nodes:
        p = os.path.join(OUT, "c%d_t%04d.png" % (nd.index, tick))
        with open(p, "wb") as f:
            nd.image.save_png(f)

SHOTS = {0, 60, 120, 200, 300, 399}
for tick in range(400):
    if 2 <= tick <= 6:
        sess.press_all(K.KEY_START)      # initiate link from PRESS START
    elif tick == 7:
        sess.press_all(raw=0)
    sess.tick()
    if tick in SHOTS:
        snap(tick)
        s = " ".join("c%d:%04X" % (nd.index, nd.siocnt) for nd in sess.nodes)
        print("tick %4d transfers=%d service_steps=%d siocnt[%s]" %
              (tick, sess.transfers, sess.service_steps_total, s))

# dump the last 60 exchanges so we can see live protocol data (not zeros)
print("last exchanges (tick, sent[c0..c3]):")
for tick, send in sess.transfer_log[-60:]:
    print("  t%03d  %04X %04X %04X %04X" % (tick, send[0], send[1], send[2], send[3]))
print("done. total transfers:", sess.transfers)
