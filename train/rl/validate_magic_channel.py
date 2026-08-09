"""Replay AlttpOracle over the captured RAM and check the 'magic' channel.

Same discipline as validate_key_channel.py: run the REAL step() the trainer
calls over the REAL captured bytes, and look at what it actually emits -- do not
assume the wiring is right because the address is right.

Expected from the two magic captures:
  201000 (two jars then a cast): magic RISES sum to +32 (0->16->32), and the
          32->0 drain at the tail emits NOTHING on the magic channel (casting is
          not rewarded). The refill animates, so it arrives as a run of small
          +deltas, not one +16 -- their SUM is what must equal the jar.
  200818 (a pure drain 16->0): NO magic events at all.
A walking dump (200550, magic flat at 16) is the negative control: also nothing.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load  # noqa: E402
from oracle import AlttpOracle  # noqa: E402
from validate_key_channel import FakeCore  # noqa: E402


def replay(dump):
    meta, ring = load(dump)
    core, oracle = FakeCore(), AlttpOracle()
    magic = []
    for i in range(ring.shape[0]):
        core.load(ring[i])
        for t, ch, dl, v in oracle.step(int(meta["ticks"][i]), core):
            if ch == "magic":
                magic.append((t, dl, v))
    return magic


def main(argv):
    RISE = [d for d in argv if "201000" in d]
    DRAIN = [d for d in argv if "200818" in d]
    CTRL = [d for d in argv if "200550" in d]
    ok = True

    for label, dumps, want in (("RISE (two jars)", RISE, "some"),
                               ("DRAIN (pure cast)", DRAIN, "none"),
                               ("CONTROL (walking)", CTRL, "none")):
        for d in dumps:
            ev = replay(d)
            total = sum(dl for _t, dl, _v in ev)
            peak = max((v for _t, _dl, v in ev), default=0)
            print("%-20s %-26s  %d magic events, sum=+%d, peak=%d"
                  % (label, os.path.basename(os.path.normpath(d)), len(ev), total, peak))
            if ev:
                print("     " + "  ".join("+%d->%d" % (dl, v) for _t, dl, v in ev[:20])
                      + (" ..." if len(ev) > 20 else ""))
            if want == "none" and ev:
                print("     FAIL: expected no magic events here")
                ok = False
            if want == "some":
                if total != 32:
                    print("     FAIL: jar refills should sum to +32 (0->16->32), got +%d" % total)
                    ok = False
                if peak != 32:
                    print("     FAIL: meter should peak at 32 after two jars, got %d" % peak)
                    ok = False

    print("\nvalidate_magic_channel: %s" % ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
