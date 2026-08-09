"""v4 (2026-07-07): resume the banked in-GAMEPLAY 4-core checkpoint in a fresh
process and prove per-player input works over the link: hold RIGHT on CHILD
core 1 (red Link), then DOWN on the parent (green Link). If both characters
move and every core's screen shows the same shared world, the whole stack --
link, savestate banking, per-slot input -- is demo-ready.
"""
import os

import mgba.gba as gba
from link_engine import LinkSession

ROM = os.environ["FOUR_SWORDS_ROM"]
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "sessions", "checkpoints")
STATES = [os.path.join(CKPT, "p%d_coop.state" % i) for i in range(4)]
OUT = os.path.join(HERE, "sessions", "coop_input")
os.makedirs(OUT, exist_ok=True)
K = gba.GBA


def snap(sess, tag):
    for nd in sess.nodes:
        p = os.path.join(OUT, "c%d_%s.png" % (nd.index, tag))
        with open(p, "wb") as f:
            nd.image.save_png(f)


sess = LinkSession(ROM, n=4, state_path=STATES)
for nd in sess.nodes:
    if nd.index > 0 and nd.irq_flagged:
        nd.irq_pending = True
        nd.mltsend_seen = False
print("4 cores resumed from p*_coop.state", flush=True)

for tick in range(200):
    if tick == 20:
        snap(sess, "before")
    if 30 <= tick < 70:
        sess.nodes[1].core.set_keys(K.KEY_RIGHT)   # child: red Link walks right
    elif tick == 70:
        sess.nodes[1].core.set_keys(raw=0)
    if tick == 80:
        snap(sess, "child_moved")
    if 90 <= tick < 130:
        sess.nodes[0].core.set_keys(K.KEY_DOWN)    # parent: green Link walks down
    elif tick == 130:
        sess.nodes[0].core.set_keys(raw=0)
    if tick == 140:
        snap(sess, "parent_moved")
    sess.tick()

snap(sess, "final")
alive = all(nd.in_multi for nd in sess.nodes)
print("done. transfers=%d all_multi=%s" % (sess.transfers, alive), flush=True)
