"""Rolling RAM ring buffer for human-driven address hunting.

WHY THIS EXISTS
---------------
Task 09 A1 confirmed the Sea of Trees health address only after a long chain of
blind write-scans, live-play intersections and screenshot comparisons -- and the
same method then FAILED to converge for Talus Cave and Death Mountain (see the
2026-07-31 notes). The reliable method has always been the human one: take one
real, visible hit and diff RAM across it.

This class makes that a single keypress. It keeps the last few seconds of full
IWRAM+EWRAM snapshots in memory; when the human sees a heart disappear they hit
the mark key, and the whole window lands on disk. Offline, ram_ring_diff.py
looks for bytes that were stable and then stepped DOWN inside that window.

No numpy here on purpose: the recorders run under LINK_PY (the mgba bindings
venv), which has pygame + cffi but NOT numpy. Analysis runs in the container.
"""
import collections
import json
import os
import time

from mgba._pylib import ffi

EWRAM_SIZE = 256 * 1024
IWRAM_SIZE = 32 * 1024
SNAP_SIZE = EWRAM_SIZE + IWRAM_SIZE      # 288 KB per snapshot


class RamRing:
    """Keeps the last `keep` snapshots, sampled every `every` ticks.

    Defaults: every=2, keep=300 -> ~10 s of history at 60 fps, ~88 MB resident.

    Was 120/~4 s until 2026-08-01. Four seconds covers reaction time to a HIT,
    which is what it was designed for, but it proved too short for ROOM
    TRANSITIONS: measured by changed-byte churn, only 1 of 6 dumps in the key
    room actually contained a door crossing, because the crossing had already
    left the window by the time the screen settled and the human pressed F11.
    Ten seconds also lets SEVERAL transitions land in one ring, which beats
    chaining separate dumps. Disk is not the constraint here (597 GB free).
    """

    def __init__(self, core, every=2, keep=300, label="ram"):
        self.core = core
        self.every = max(1, int(every))
        self.snaps = collections.deque(maxlen=int(keep))   # (tick, bytes)
        self.label = label
        self.n_dumps = 0
        # Direct pointers into the core's memory; same access solo_ram_capture.py
        # uses. Cached once -- they do not move for the life of the core.
        self._ew = core._native.memory.wram
        self._iw = core._native.memory.iwram

    def record(self, tick):
        """Call once per emulated frame, after the core has advanced."""
        if tick % self.every:
            return
        blob = (bytes(ffi.buffer(self._iw, IWRAM_SIZE))
                + bytes(ffi.buffer(self._ew, EWRAM_SIZE)))
        self.snaps.append((tick, blob))

    def history_secs(self):
        return len(self.snaps) * self.every / 60.0

    def dump(self, out_root, tag="hit", png_bytes=None, note=""):
        """Write the whole ring out. Returns the created directory."""
        if not self.snaps:
            return None
        out = os.path.join(out_root, time.strftime("%Y%m%d-%H%M%S") + "-" + tag)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "ring.bin"), "wb") as f:
            for _, blob in self.snaps:
                f.write(blob)
        meta = {
            "label": self.label,
            "tag": tag,
            "note": note,
            "n": len(self.snaps),
            "every": self.every,
            "ticks": [t for t, _ in self.snaps],
            "iwram_size": IWRAM_SIZE,
            "ewram_size": EWRAM_SIZE,
            "snap_size": SNAP_SIZE,
            "layout": "iwram then ewram, per snapshot, snapshot-major",
            "secs": self.history_secs(),
        }
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if png_bytes:
            with open(os.path.join(out, "mark.png"), "wb") as f:
                f.write(png_bytes)
        self.n_dumps += 1
        print("ram-ring: dumped %d snaps (%.1fs) -> %s" % (
            len(self.snaps), self.history_secs(), out), flush=True)
        return out
