"""Stage-3 test v3 (2026-07-07): 4-core savestate COHERENCE. v2 proved 4 cores
link and reach CHOOSE A STAGE. Untested question (SESSION_2026-07-07 next-steps
item 2): can we save all 4 linked states to disk and resume the link later --
in a FRESH process -- without the protocol dying? If yes, we can bank per-core
checkpoints anywhere in multiplayer and skip the 700-tick boot every run.

Two phases, two processes (also dodges the known teardown segfault):
  save:   run the v2 choreography to tick 700 (settled at CHOOSE A STAGE),
          save p0..p3_stage_select.state, then run 100 more ticks as the
          CONTROL for the healthy transfer rate.
  resume: fresh LinkSession from those 4 states, reconstruct the slave-IRQ
          bookkeeping the savestate can't hold, run 300 ticks, press DOWN on
          the parent mid-run (cursor should move if the game is truly alive).
          PASS = all cores still in SIO_MULTI and transfer rate >= 50% of
          the control rate.
"""
import os
import sys

import mgba.gba as gba
from mgba._pylib import ffi
from link_engine import LinkSession

ROM = os.environ["FOUR_SWORDS_ROM"]
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "sessions", "checkpoints")
BOOT_STATES = [os.path.join(CKPT, "p%d_link_wait.state" % i) for i in range(4)]
SAVED_STATES = [os.path.join(CKPT, "p%d_stage_select.state" % i) for i in range(4)]
OUT = os.path.join(HERE, "sessions", "linktest_v3")
os.makedirs(OUT, exist_ok=True)
K = gba.GBA

SAVE_TICK = 700       # v2: badge ~400, parent START at 450, in-game by ~600
CONTROL_TICKS = 100
RESUME_TICKS = 300


def snap(sess, tag):
    for nd in sess.nodes:
        p = os.path.join(OUT, "c%d_%s.png" % (nd.index, tag))
        with open(p, "wb") as f:
            nd.image.save_png(f)


def status(sess, tick):
    s = " ".join("c%d:m%d/%04X" % (nd.index, nd.mode, nd.siocnt) for nd in sess.nodes)
    print("tick %4d transfers=%d [%s]" % (tick, sess.transfers, s), flush=True)


def phase_save():
    sess = LinkSession(ROM, n=4, state_path=BOOT_STATES)
    print("phase save: 4 cores loaded from link_wait", flush=True)
    master = sess.nodes[0].core
    for tick in range(SAVE_TICK):
        if 2 <= tick <= 6:
            sess.press_all(K.KEY_START)
        elif tick == 7:
            sess.press_all(raw=0)
        elif 450 <= tick <= 455:
            master.set_keys(K.KEY_START)
        elif tick == 456:
            master.set_keys(raw=0)
        sess.tick()
        if tick % 100 == 0:
            status(sess, tick)
    snap(sess, "presave")
    at_save = sess.transfers
    for i, nd in enumerate(sess.nodes):
        st = nd.core.save_raw_state()
        if st is None:
            print("FAIL: save_raw_state returned None on core %d" % i, flush=True)
            return 1
        with open(SAVED_STATES[i], "wb") as f:
            f.write(bytes(ffi.buffer(st)))
    print("saved 4 states at tick %d (transfers=%d)" % (SAVE_TICK, at_save), flush=True)
    for tick in range(CONTROL_TICKS):
        sess.tick()
    rate = (sess.transfers - at_save) / float(CONTROL_TICKS)
    snap(sess, "control")
    status(sess, SAVE_TICK + CONTROL_TICKS)
    print("CONTROL transfer rate: %.2f/tick" % rate, flush=True)
    with open(os.path.join(OUT, "control_rate.txt"), "w") as f:
        f.write("%f\n" % rate)
    return 0


def phase_resume():
    sess = LinkSession(ROM, n=4, state_path=SAVED_STATES)
    # The savestate holds IF (a raised-but-unserviced SIO IRQ survives), but the
    # session's irq_pending/mltsend_seen bookkeeping does not -- reconstruct it
    # so service_slaves() pre-steps that slave on the first post-resume burst.
    for nd in sess.nodes:
        if nd.index > 0 and nd.irq_flagged:
            nd.irq_pending = True
            nd.mltsend_seen = False
            print("resume: core %d had SIO IRQ pending in-state" % nd.index, flush=True)
    print("phase resume: 4 cores loaded from stage_select", flush=True)
    snap(sess, "resume_t0000")
    master = sess.nodes[0].core
    for tick in range(RESUME_TICKS):
        if tick == 140:
            snap(sess, "resume_t0140")
        if 150 <= tick <= 152:
            master.set_keys(K.KEY_DOWN)   # parent moves the stage cursor
        elif tick == 153:
            master.set_keys(raw=0)
        if tick == 160:
            snap(sess, "resume_t0160")
        sess.tick()
        if tick % 50 == 0:
            status(sess, tick)
    snap(sess, "resume_final")
    status(sess, RESUME_TICKS)
    rate = sess.transfers / float(RESUME_TICKS)
    with open(os.path.join(OUT, "control_rate.txt")) as f:
        control = float(f.read().strip())
    alive = all(nd.in_multi for nd in sess.nodes)
    print("RESUME transfer rate: %.2f/tick (control %.2f)" % (rate, control), flush=True)
    verdict = alive and rate >= 0.5 * control
    print("VERDICT: %s (all_multi=%s rate_ok=%s)"
          % ("PASS" if verdict else "FAIL", alive, rate >= 0.5 * control), flush=True)
    # the teardown segfault at interpreter exit clobbers our exit code, so the
    # wrapper reads this file instead
    with open(os.path.join(OUT, "verdict.txt"), "w") as f:
        f.write("PASS\n" if verdict else "FAIL\n")
    return 0 if verdict else 1


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    if phase == "save":
        sys.exit(phase_save())
    elif phase == "resume":
        sys.exit(phase_resume())
    print("usage: link_test_v3_savestate.py save|resume")
    sys.exit(2)
