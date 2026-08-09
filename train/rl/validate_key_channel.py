"""Replay AlttpOracle over the captured RAM dumps and report what it emits.

Wiring a channel and asserting it works is how this project got a rupee-only
reward it believed was a health reward. So run the REAL oracle over the REAL
captured bytes and look at the events it actually produces.

Feeds the oracle a fake core whose EWRAM is one ring snapshot, stepped through
the whole window in order -- so this exercises the same step() the trainer calls.
"""
import os
import sys

import numpy as np
from mgba._pylib import ffi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ram_ring_diff import load  # noqa: E402
from oracle import AlttpOracle  # noqa: E402

EW = 32 * 1024
EW_SIZE = 256 * 1024
IW_SIZE = 32 * 1024


class _Mem(object):
    pass


class _Native(object):
    pass


class FakeCore(object):
    """Minimal stand-in exposing just what AlttpOracle._ew() reaches for."""

    def __init__(self):
        self._buf = bytearray(EW_SIZE)
        self._ibuf = bytearray(IW_SIZE)
        self._cdata = ffi.from_buffer(self._buf)
        self._icdata = ffi.from_buffer(self._ibuf)
        self._native = _Native()
        self._native.memory = _Mem()
        self._native.memory.wram = self._cdata
        # IWRAM too: position (0x038F0/0x038F4) lives there, so an EWRAM-only
        # fake core cannot replay the exploration channel.
        self._native.memory.iwram = self._icdata

    def load(self, ring_row):
        self._ibuf[:] = bytes(ring_row[:IW_SIZE])
        self._buf[:] = bytes(ring_row[EW:EW + EW_SIZE])


def main(dumps):
    total = {}
    for d in sorted(dumps):
        meta, ring = load(d)
        core = FakeCore()
        oracle = AlttpOracle()
        events = []
        for i in range(ring.shape[0]):
            core.load(ring[i])
            events.extend(oracle.step(int(meta["ticks"][i]), core))
        name = os.path.basename(os.path.normpath(d))
        summary = {}
        for _t, ch, _dl, _v in events:
            summary[ch] = summary.get(ch, 0) + 1
            total[ch] = total.get(ch, 0) + 1
        print("%-26s %s" % (name, summary or "no events"))
        for t, ch, dl, v in events:
            if ch in ("key", "key_used", "heart_container"):
                print("      tick %-7d %-16s delta %+d -> %d" % (t, ch, dl, v))
    print("\nTOTAL across %d dumps: %s" % (len(dumps), total or "NOTHING"))
    if not total.get("key") and not total.get("key_used"):
        print("KEY CHANNEL DID NOT FIRE -- the wiring is wrong, or the address is")
        return 1
    print("key channel fires on real captured RAM.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
