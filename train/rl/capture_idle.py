"""Headless IDLE capture: load a state, send ZERO input, snapshot RAM each step.

Tim's method for finding the enemy position (2026-08-02): with Link held
perfectly still, almost the only thing that moves SPATIALLY is the enemy, so its
coordinates ramp smoothly (small steps, both directions) exactly like the player
coordinates did -- and Link's own position stays constant, removing the biggest
confound. This does it without a human: "don't move" is literally set_keys(raw=0).

Writes a ring in the same on-disk layout link/ram_ring.py uses
("iwram then ewram, per snapshot, snapshot-major") so find_position_magic.py and
ram_ring_diff.py read it with no changes.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IW_SIZE = 32 * 1024
EW_SIZE = 256 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="alttp_human-04")
    ap.add_argument("--n", type=int, default=300, help="snapshots to keep")
    ap.add_argument("--every", type=int, default=2, help="frames between snapshots")
    ap.add_argument("--out", default=None, help="output dump dir")
    args = ap.parse_args()

    from alttp_ppo_env import AlttpPpoEnv
    from mgba._pylib import ffi
    import numpy as np

    state_path = os.path.join(HERE, "states", args.state + ".state")
    env = AlttpPpoEnv(state_path=state_path, horizon=10 ** 9, death_terminates=False)
    env.reset()
    core = env.core

    def snap():
        iw = bytes(ffi.buffer(core._native.memory.iwram, IW_SIZE))
        ew = bytes(ffi.buffer(core._native.memory.wram, EW_SIZE))
        return iw + ew

    rows, ticks = [], []
    hp0 = env.oracle.read_health(core)
    frame = 0
    for i in range(args.n):
        for _ in range(args.every):
            core.set_keys(raw=0)          # ZERO input -- Link does not move
            core.run_frame()
            frame += 1
        rows.append(snap())
        ticks.append(frame)
    hp1 = env.oracle.read_health(core)

    ring = np.frombuffer(b"".join(rows), dtype=np.uint8)
    out = args.out or os.path.join(HERE, "scratch", "idle_" + args.state)
    os.makedirs(out, exist_ok=True)
    ring.tofile(os.path.join(out, "ring.bin"))
    meta = {"label": "idle_" + args.state, "tag": "idle", "note": "no input",
            "n": args.n, "every": args.every, "ticks": ticks,
            "iwram_size": IW_SIZE, "ewram_size": EW_SIZE,
            "snap_size": IW_SIZE + EW_SIZE,
            "layout": "iwram then ewram, per snapshot, snapshot-major",
            "secs": args.n * args.every / 60.0}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    env.close()
    print("wrote %s  (%d snaps, every %d frames, %.1fs)" % (out, args.n, args.every, meta["secs"]))
    print("HP start=%d end=%d %s" % (hp0, hp1,
          "(unchanged -- clean idle window)" if hp0 == hp1 else
          "(HP CHANGED -- enemy reached Link; use pre-change snapshots only)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
