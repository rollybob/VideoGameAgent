"""Stage-3 test v2 (2026-07-07): v1 reached the 4-badge negotiation screen
('PLEASE WAIT' + P1-P4 with YOU markers). In Four Swords the parent unit
confirms with START once all players are shown. This run: reach the badge
screen (~tick 400), then press START on the MASTER only, then keep running
and screenshot to see if the game advances into the actual link/game."""
import os
import mgba.image
from link_engine import LinkSession
import mgba.gba as gba

ROM = os.environ["FOUR_SWORDS_ROM"]
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "sessions", "checkpoints")
STATES = [os.path.join(CKPT, "p%d_link_wait.state" % i) for i in range(4)]
OUT = os.path.join(HERE, "sessions", "linktest_v2")
os.makedirs(OUT, exist_ok=True)
K = gba.GBA

sess = LinkSession(ROM, n=4, state_path=STATES, trace=True)
print("loaded 4 cores")

def snap(tag):
    for nd in sess.nodes:
        p = os.path.join(OUT, "c%d_%s.png" % (nd.index, tag))
        with open(p, "wb") as f:
            nd.image.save_png(f)

def status(tick):
    s = " ".join("c%d:m%d/%04X" % (nd.index, nd.mode, nd.siocnt) for nd in sess.nodes)
    print("tick %4d transfers=%d [%s]" % (tick, sess.transfers, s))

MASTER = sess.nodes[0].core
for tick in range(1500):
    if 2 <= tick <= 6:
        sess.press_all(K.KEY_START)          # initiate link from PRESS START
    elif tick == 7:
        sess.press_all(raw=0)
    elif 450 <= tick <= 455:
        MASTER.set_keys(K.KEY_START)         # parent confirms on badge screen
    elif tick == 456:
        MASTER.set_keys(raw=0)
    sess.tick()
    if tick in (400, 449, 470, 500, 600, 800, 1000, 1200, 1499):
        snap("t%04d" % tick)
        status(tick)
print("done. total transfers:", sess.transfers)
