"""Task 09 A0: throughput probe for env Option 2 -- the 4P link session.

Resumes the four p*_coop.state checkpoints (live Four Swords co-op in
Chambers of Insight) and runs the deterministic link conductor with
wander-style inputs, measuring ticks/second. One tick = one frame on all 4
cores + SIO servicing, i.e. 4 learner-steps of experience.

Also measures the tick rate with a per-tick full-EWRAM snapshot per core
(bytes() copy via ffi.buffer) -- the worst-case reward-oracle overhead --
so the oracle's cost is known before it is built.

Run (host or container; needs mgba on PYTHONPATH + link/ importable):
  python3 bench_link4.py [--ticks 2000]
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(VGA, "link"))

import mgba.log
from mgba._pylib import ffi
from link_engine import LinkSession

ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                   "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
CKPT = os.path.join(VGA, "link", "sessions", "checkpoints")

# GBA key bits: A=1 B=2 Sel=4 Start=8 Right=16 Left=32 Up=64 Down=128 R=256 L=512
WANDER = [16, 32, 64, 128, 16 | 1, 32 | 1, 64 | 1, 128 | 1, 0]


def bench(sess, n_ticks, snapshot):
    wram_ptrs = None
    if snapshot:
        wram_ptrs = [nd.core._native.memory.wram for nd in sess.nodes]
    t0 = time.perf_counter()
    c0 = time.process_time()
    for k in range(n_ticks):
        for i, nd in enumerate(sess.nodes):
            nd.core.set_keys(raw=WANDER[(k // 30 + i) % len(WANDER)])
        sess.tick()
        if snapshot:
            for p in wram_ptrs:
                _ = bytes(ffi.buffer(p, 256 * 1024))
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    return wall, cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=2000)
    args = ap.parse_args()
    mgba.log.silence()
    states = [os.path.join(CKPT, "p%d_coop.state" % i) for i in range(4)]
    sess = LinkSession(ROM, n=4, state_path=states, trace=False)
    # warmup: let the resumed session settle before timing
    for nd in sess.nodes:
        if nd.index > 0 and nd.irq_flagged:
            nd.irq_pending = True
            nd.mltsend_seen = False
    for _ in range(120):
        sess.tick()

    for label, snap in (("plain", False), ("with 4x256KB EWRAM snapshot/tick", True)):
        wall, cpu = bench(sess, args.ticks, snap)
        tps = args.ticks / wall
        print("[%s] %d ticks in %.2fs wall (%.2fs cpu, %.2f cores)"
              % (label, args.ticks, wall, cpu, cpu / wall))
        print("  %.1f ticks/s = %.1f learner-steps/s (x4 players)"
              % (tps, tps * 4))
        print("  per hour: %.2fM ticks, %.2fM learner-steps, using %.2f cores"
              % (tps * 3600 / 1e6, tps * 4 * 3600 / 1e6, cpu / wall))
    sess.shutdown()


if __name__ == "__main__":
    main()
