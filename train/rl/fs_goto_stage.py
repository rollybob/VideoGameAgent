"""Task 09 Phase A1: drive the FS pause -> quit -> stage-select menu and save a
fresh checkpoint at a chosen overworld stage.

Hard-won sequence (see project-vga-task09-a1-ppo-baseline memory): START to
pause, then WAIT for the "P2 PAUSE!" banner to actually clear before the real
Continue/Quit list renders (inputs sent too early are dropped) -- DOWN, A
confirms through to "CHOOSE A STAGE" (which also needs its own settle time),
then LEFT cycles: Chambers of Insight (start) -> x1 Sea of Trees -> x2 Talus
Cave -> x3 Death Mountain. A confirms and loads (another ~150-frame black
transition).

Run:
    docker run --rm --runtime nvidia -v ~/projects/VGA:/work -w /work \\
        thor-rl:cu130 python3 train/rl/fs_goto_stage.py --stage talus_cave \\
        --out-name taluscave
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(VGA, "link"))
sys.path.insert(0, HERE)

import mgba.log
from mgba.gba import GBA
from link_engine import LinkSession

mgba.log.silence()

ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                    "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
COOP_CKPT = os.path.join(VGA, "link", "sessions", "checkpoints")

# LEFT-press count from the Chambers of Insight default stage-select cursor.
STAGE_LEFT_COUNT = {"sea_of_trees": 1, "talus_cave": 2, "death_mountain": 3}

K_START = 1 << GBA.KEY_START
K_DOWN = 1 << GBA.KEY_DOWN
K_LEFT = 1 << GBA.KEY_LEFT
K_A = 1 << GBA.KEY_A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGE_LEFT_COUNT))
    ap.add_argument("--out-dir", default=None, help="default: link/sessions/checkpoints_<out-name>")
    ap.add_argument("--out-name", required=True, help="state filename stem: p{i}_<out-name>.state")
    ap.add_argument("--settle-ticks", type=int, default=60,
                     help="extra ticks to walk after loading, away from the menu-exit point")
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM", ROM))
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(VGA, "link", "sessions", "checkpoints_%s" % args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    states = [os.path.join(COOP_CKPT, "p%d_coop.state" % i) for i in range(4)]
    sess = LinkSession(args.rom, n=4, state_path=states, trace=False)

    def tap(key, hold=5, after=100):
        for _ in range(hold):
            for nd in sess.nodes:
                nd.core.set_keys(raw=key)
            sess.tick()
        for _ in range(after):
            for nd in sess.nodes:
                nd.core.set_keys(raw=0)
            sess.tick()

    tap(K_START)
    tap(K_DOWN)
    tap(K_A)
    tap(K_A)
    for _ in range(STAGE_LEFT_COUNT[args.stage]):
        tap(K_LEFT)
    tap(K_A, after=150)

    for _ in range(args.settle_ticks):
        for nd in sess.nodes:
            nd.core.set_keys(raw=K_DOWN)
        sess.tick()

    for i, nd in enumerate(sess.nodes):
        st = nd.core.save_raw_state()
        with open(os.path.join(out_dir, "p%d_%s.state" % (i, args.out_name)), "wb") as f:
            f.write(bytes(st))
    print("saved 4 checkpoint files to %s (stem=%s)" % (out_dir, args.out_name))
    sess.shutdown()


if __name__ == "__main__":
    main()
