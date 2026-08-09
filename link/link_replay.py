"""Deterministically replay a FlightRecorder dump headless (2026-07-07).

Loads the dump's 4 pre-fail savestates and replays the recorded per-tick
input masks. The engine is deterministic, so the replay reproduces the live
session exactly -- verified by comparing the transfer count against the live
meta. Prints diagnostics (mode changes, refused starts) and snaps frames
around the death tick so the moment of link death can be studied offline.

    source env.sh; "$LINK_PY" link_replay.py sessions/flight/<dump-dir>
"""
import argparse
import json
import os
import sys

from link_engine import LinkSession

ROM = os.environ["FOUR_SWORDS_ROM"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--snap-window", type=int, default=90,
                    help="snap all cores every 30 ticks within this window around death")
    ap.add_argument("--trace-window", type=int, default=120,
                    help="write the SIO trace this many ticks around death")
    args = ap.parse_args()
    with open(os.path.join(args.dump, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(args.dump, "inputs.json")) as f:
        tape = json.load(f)
    states = [os.path.join(args.dump, "p%d_prefail.state" % i) for i in range(4)]
    masks = tape["masks"]
    bank_tick = meta["bank_tick"]
    death_rel = (meta["death_tick"] - bank_tick) if meta.get("death_tick") else None
    print("dump: %s" % args.dump)
    print("bank at live tick %d, %d recorded ticks, death at rel tick %s"
          % (bank_tick, len(masks), death_rel))

    sess = LinkSession(ROM, n=4, state_path=states, trace=True)
    for nd in sess.nodes:
        if nd.index > 0 and nd.irq_flagged:
            nd.irq_pending = True
            nd.mltsend_seen = False

    def snap(tag):
        for nd in sess.nodes:
            p = os.path.join(args.dump, "replay_c%d_%s.png" % (nd.index, tag))
            with open(p, "wb") as f:
                nd.image.save_png(f)

    last = 0
    for k, m in enumerate(masks):
        rel = k + 1
        for i in range(4):
            sess.nodes[i].core.set_keys(raw=m[i])
        sess.tick()
        if rel % 60 == 0:
            d = sess.transfers - last
            last = sess.transfers
            marker = ""
            if death_rel and abs(rel - death_rel) <= 60:
                marker = "   <-- death window"
            print("rel %5d  transfers+%3d  modes=%s%s"
                  % (rel, d, [nd.mode for nd in sess.nodes], marker))
        if death_rel and abs(rel - death_rel) <= args.snap_window and rel % 30 == 0:
            snap("t%05d" % rel)
    snap("end")

    live_delta = meta["end_transfers"] - meta["bank_transfers"]
    print("replay transfers=%d  live delta=%d  -> %s"
          % (sess.transfers, live_delta,
             "DETERMINISM OK" if sess.transfers == live_delta else "MISMATCH"))
    print("mode_log:", sess.mode_log)
    print("refused_starts: %d%s" % (len(sess.refused_starts),
          "" if not sess.refused_starts else "; first 10: %s" % sess.refused_starts[:10]))
    # dump the SIO trace around death for close reading
    if death_rel:
        out = os.path.join(args.dump, "replay_trace_death.txt")
        with open(out, "w") as f:
            for nd in sess.nodes:
                f.write("== node %d writes ==\n" % nd.index)
                for tick, addr, val in nd.trace:
                    if abs(tick - death_rel) <= args.trace_window:
                        f.write("%d %03X %04X\n" % (tick, addr, val))
            f.write("== transfers ==\n")
            for tick, words in sess.transfer_log:
                if abs(tick - death_rel) <= args.trace_window:
                    f.write("%d %s\n" % (tick, " ".join("%04X" % w for w in words)))
        print("death-window trace written to", out)
    sess.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
