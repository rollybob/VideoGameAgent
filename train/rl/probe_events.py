"""Task 09 A0: controlled event probes for RAM reward mapping.

Resumes the 4P coop checkpoint and drives core 0's Link (parent, green)
through a scripted input schedule while capturing per-tick EWRAM+IWRAM of
core 0 (same format as ram_capture.py) plus a PNG every --png-every ticks.
Because WE choose the inputs, event times (bush cut, rupee pickup, damage)
are known to within a few ticks -- RAM diffing in those windows is
hypothesis-testing, not correlation mining.

The schedule is a list of (label, key_mask, ticks) pulses for core 0; other
cores idle. Edit SCHEDULES or pass --schedule NAME.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/probe_events.py --schedule scout --out scout
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(VGA, "link"))

import mgba.log
from mgba._pylib import ffi
from link_engine import LinkSession

ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                   "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
CKPT = os.path.join(VGA, "link", "sessions", "checkpoints")
EWRAM_SIZE = 256 * 1024
IWRAM_SIZE = 32 * 1024

A, B, RIGHT, LEFT, UP, DOWN = 1, 2, 16, 32, 64, 128

# (label, mask, ticks) -- labels land in schedule.json for the diff step
SCHEDULES = {
    # look around first: idle, then short walks each way to see the arena
    "scout": [
        ("idle", 0, 60),
        ("down", DOWN, 45), ("idle", 0, 30),
        ("left", LEFT, 45), ("idle", 0, 30),
        ("right", RIGHT, 90), ("idle", 0, 30),
        ("up", UP, 45), ("idle", 0, 60),
        ("slash", B, 6), ("idle", 0, 60),
    ],
    # coop room: walk into the pink enemy above the start row -> contact damage
    "touch": [
        ("idle", 0, 60),
        ("up-into-enemy", UP, 180),
        ("idle", 0, 120),
    ],
    # coop room: walk up to the enemy and slash repeatedly -> kill event
    "kill": [
        ("idle", 0, 60),
        ("up", UP, 90), ("slash1", B, 6), ("idle", 0, 40),
        ("slash2", B, 6), ("idle", 0, 40),
        ("up2", UP, 30), ("slash3", B, 6), ("idle", 0, 40),
        ("slash4", B, 6), ("idle", 0, 120),
    ],
    # bush field (tape3 prefail states): slash around, then sweep for drops
    "bush": [
        ("idle", 0, 30),
        ("slash1", B, 6), ("idle", 0, 30),
        ("left", LEFT, 20), ("slash2", B, 6), ("idle", 0, 30),
        ("right", RIGHT, 40), ("slash3", B, 6), ("idle", 0, 30),
        ("down", DOWN, 20), ("slash4", B, 6), ("idle", 0, 30),
        ("sweep-l", LEFT, 60), ("sweep-r", RIGHT, 90),
        ("sweep-u", UP, 40), ("sweep-d", DOWN, 60), ("idle", 0, 60),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="scout")
    ap.add_argument("--schedule-json", default=None,
                    help="path to a JSON [[label,mask,ticks],...] (overrides --schedule)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png-every", type=int, default=10)
    ap.add_argument("--states", default=None,
                    help="dir with p0..p3 states + suffix, e.g. '<dir>:_prefail.state'"
                         " (default: coop checkpoints)")
    args = ap.parse_args()
    mgba.log.silence()

    if args.schedule_json:
        with open(args.schedule_json) as f:
            sched = [tuple(x) for x in json.load(f)]
    else:
        sched = SCHEDULES[args.schedule]
    out = os.path.join(HERE, "scratch", "probe", args.out)
    os.makedirs(out, exist_ok=True)

    if args.states:
        sdir, suffix = args.states.split(":")
        states = [os.path.join(sdir, "p%d%s" % (i, suffix)) for i in range(4)]
    else:
        states = [os.path.join(CKPT, "p%d_coop.state" % i) for i in range(4)]
    sess = LinkSession(ROM, n=4, state_path=states, trace=False)
    for nd in sess.nodes:
        if nd.index > 0 and nd.irq_flagged:
            nd.irq_pending = True
            nd.mltsend_seen = False

    n = sum(t for _, _, t in sched)
    ew = np.memmap(os.path.join(out, "ewram.u8"), dtype=np.uint8, mode="w+",
                   shape=(n, EWRAM_SIZE))
    iw = np.memmap(os.path.join(out, "iwram.u8"), dtype=np.uint8, mode="w+",
                   shape=(n, IWRAM_SIZE))
    c0 = sess.nodes[0]
    ew_ptr = c0.core._native.memory.wram
    iw_ptr = c0.core._native.memory.iwram

    timeline = []
    k = 0
    for label, mask, ticks in sched:
        timeline.append({"label": label, "mask": mask, "start": k, "end": k + ticks})
        for _ in range(ticks):
            c0.core.set_keys(raw=mask)
            for nd in sess.nodes[1:]:
                nd.core.set_keys(raw=0)
            sess.tick()
            ew[k] = np.frombuffer(ffi.buffer(ew_ptr, EWRAM_SIZE), dtype=np.uint8)
            iw[k] = np.frombuffer(ffi.buffer(iw_ptr, IWRAM_SIZE), dtype=np.uint8)
            if k % args.png_every == 0:
                with open(os.path.join(out, "f%05d.png" % k), "wb") as f:
                    c0.image.save_png(f)
            k += 1
    ew.flush(); iw.flush()
    with open(os.path.join(out, "ticks.json"), "w") as f:
        json.dump({"ticks": n, "transfers": sess.transfers}, f)
    with open(os.path.join(out, "schedule.json"), "w") as f:
        json.dump(timeline, f, indent=1)
    print("probe '%s': %d ticks -> %s" % (args.out, n, out))
    sess.shutdown()


if __name__ == "__main__":
    main()
