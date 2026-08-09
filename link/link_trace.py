"""Step 2 (2026-07-07): capture the REAL handshake register traffic during
'Linking with other systems'. Runs the 4-core unique-save session with
trace=True for 100 ticks and dumps, per core, every SIO register write the
game makes (via the driver's writeRegister hook) plus every transfer's
exchanged SIOMLT_SEND vector. Output: sessions/trace_0707.txt
"""
import os
from link_engine import LinkSession, REG_SIOCNT, REG_SIOMLT_SEND, REG_RCNT
import mgba.gba as gba

ROM = os.environ["FOUR_SWORDS_ROM"]
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "sessions", "checkpoints")
STATES = [os.path.join(CKPT, "p%d_link_wait.state" % i) for i in range(4)]
OUT = os.path.join(HERE, "sessions", "trace_0707.txt")
K = gba.GBA

REG_NAMES = {REG_SIOCNT: "SIOCNT", REG_SIOMLT_SEND: "MLTSEND", REG_RCNT: "RCNT"}

sess = LinkSession(ROM, n=4, state_path=STATES, trace=True)
for tick in range(100):
    if 2 <= tick <= 6:
        sess.press_all(K.KEY_START)
    elif tick == 7:
        sess.press_all(raw=0)
    sess.tick()

lines = []
lines.append("=== transfers (tick, sent[c0..c3]) ===")
for tick, send in sess.transfer_log:
    lines.append("t%03d  %04X %04X %04X %04X" % (tick, send[0], send[1], send[2], send[3]))
for nd in sess.nodes:
    lines.append("")
    lines.append("=== core %d SIO writes (tick, reg, value) ===" % nd.index)
    for tick, addr, value in nd.trace:
        name = REG_NAMES.get(addr, "%03X" % addr)
        lines.append("t%03d  %-7s <- %04X" % (tick, name, value))
text = "\n".join(lines)
with open(OUT, "w") as f:
    f.write(text + "\n")
print("wrote %s (%d lines)" % (OUT, len(lines)))
print("write counts per core:", [len(nd.trace) for nd in sess.nodes])
print("transfers:", sess.transfers)
