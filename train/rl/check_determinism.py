"""Task 09 A0: prove the env is deterministic (needed for replay debugging).

Replays a flight tape twice in two independent processes' worth of work
(two fresh LinkSessions in one process) and sha256-compares the full
per-tick IWRAM+EWRAM streams of core 0. Identical hashes => deterministic.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/check_determinism.py <flight-dir>
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "link"))

import mgba.log
from mgba._pylib import ffi
from link_engine import LinkSession

VGA = "/vga"
ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                   "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")


def run(dump):
    with open(os.path.join(dump, "inputs.json")) as f:
        masks = json.load(f)["masks"]
    states = [os.path.join(dump, "p%d_prefail.state" % i) for i in range(4)]
    sess = LinkSession(ROM, n=4, state_path=states, trace=False)
    for nd in sess.nodes:
        if nd.index > 0 and nd.irq_flagged:
            nd.irq_pending = True
            nd.mltsend_seen = False
    c0 = sess.nodes[0]
    ew = c0.core._native.memory.wram
    iw = c0.core._native.memory.iwram
    h = hashlib.sha256()
    for m in masks:
        for i in range(4):
            sess.nodes[i].core.set_keys(raw=m[i])
        sess.tick()
        h.update(ffi.buffer(ew, 256 * 1024))
        h.update(ffi.buffer(iw, 32 * 1024))
    digest = h.hexdigest()
    sess.shutdown()
    return digest


def main():
    mgba.log.silence()
    dump = sys.argv[1]
    a = run(dump)
    b = run(dump)
    print("run A:", a[:32])
    print("run B:", b[:32])
    print("DETERMINISM OK" if a == b else "NON-DETERMINISTIC")


if __name__ == "__main__":
    main()
