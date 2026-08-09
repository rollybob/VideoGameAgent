"""Replay AlttpOracle over the KILL captures and check the 'enemy_dmg' channel.

Same discipline as validate_key/magic: run the real step() over the real captured
RAM, do not assume. Expected, from the kill captures (enemy has 6 HP, 3 hits):
  kill_* : enemy_dmg fires exactly 3 times, +2 each, summing to +6, then NOTHING
           (health stays 0 -- slot not reused, so no fabricated post-death hit).
  idle_* : NO enemy_dmg at all (Link never attacks; the enemy kills him).
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
    ev = []
    ticks = meta.get("ticks") or list(range(ring.shape[0]))
    for i in range(ring.shape[0]):
        core.load(ring[i])
        for t, ch, dl, v in oracle.step(int(ticks[i]), core):
            if ch == "enemy_dmg":
                ev.append((t, dl, v))
    return ev


def main(argv):
    ok = True
    for d in argv:
        ev = replay(d)
        total = sum(dl for _t, dl, _v in ev)
        name = os.path.basename(os.path.normpath(d))
        print("%-30s %d enemy_dmg events, sum=+%d  %s"
              % (name, len(ev), total,
                 " ".join("+%d->%d" % (dl, v) for _t, dl, v in ev)))
        if name.startswith("kill_"):
            if not (len(ev) == 3 and total == 6):
                print("   FAIL: expected 3 hits summing to +6 for a 6-HP enemy")
                ok = False
        if name.startswith("idle_") and ev:
            print("   FAIL: idle capture should have no enemy_dmg")
            ok = False
    print("\nvalidate_enemy_channel: %s" % ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
